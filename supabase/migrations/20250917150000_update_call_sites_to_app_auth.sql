-- Switch call sites from auth.check_if_service_account() -> app_auth.check_if_service_account()
-- Keep SECURITY DEFINER and add a safe search_path that includes app_auth.

-- 1) public.get_organization_from_token (final version from 2024-12-05-00:00:10)
CREATE OR REPLACE FUNCTION public.get_organization_from_token()
RETURNS uuid
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, auth, app_auth
AS $$
DECLARE
  org_id uuid;
BEGIN
  -- Only handle service account tokens
  IF NOT app_auth.check_if_service_account() THEN
    RAISE EXCEPTION 'Only service account tokens are supported';
  END IF;

  -- Get org from claims
  org_id := (current_setting('request.jwt.claims', true)::json->>'organization_id')::uuid;

  -- Update last_used_at in tokens table
  UPDATE public.tokens
  SET last_used_at = now()
  WHERE id = (current_setting('request.jwt.claims', true)::json->>'token_id')::uuid;

  RETURN org_id;
END;
$$;

-- 2) public.is_organization_member(uuid)
CREATE OR REPLACE FUNCTION public.is_organization_member(org_id uuid)
RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, auth, app_auth
AS $$
BEGIN
  -- If this is a service account, check the organization claim
  IF app_auth.check_if_service_account() THEN
    RETURN (current_setting('request.jwt.claims', true)::json->>'organization_id')::uuid = org_id;
  END IF;

  -- Otherwise check normal membership
  RETURN EXISTS (
    SELECT 1
    FROM public.organization_members
    WHERE organization_id = org_id
      AND user_id = auth.uid()
  );
END;
$$;

-- 3) public.is_organization_admin(uuid)
CREATE OR REPLACE FUNCTION public.is_organization_admin(org_id uuid)
RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, auth, app_auth
AS $$
BEGIN
  -- Service accounts have admin access to their organization
  IF app_auth.check_if_service_account() THEN
    RETURN (current_setting('request.jwt.claims', true)::json->>'organization_id')::uuid = org_id;
  END IF;

  -- Otherwise check normal admin membership
  RETURN EXISTS (
    SELECT 1
    FROM public.organization_members
    WHERE organization_id = org_id
      AND user_id = auth.uid()
      AND role = 'admin'
  );
END;
$$;
