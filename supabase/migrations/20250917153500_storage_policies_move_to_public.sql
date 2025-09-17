-- All app-owned helpers live in public. Storage policies reference these.
-- No objects created/modified inside 'storage' or 'app_storage'.

-- 1) Path helper: organizations/<org_uuid>/...
CREATE OR REPLACE FUNCTION public.get_path_organization_id(path text)
RETURNS uuid
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
  parts text[];
  org_id uuid;
BEGIN
  parts := string_to_array(path, '/');

  IF array_length(parts, 1) IS NULL OR parts[1] <> 'organizations' THEN
    RETURN NULL;
  END IF;

  BEGIN
    org_id := parts[2]::uuid;
    RETURN org_id;
  EXCEPTION WHEN others THEN
    RETURN NULL;
  END;
END;
$$;

-- 2) Access helper: service-account claim OR membership
CREATE OR REPLACE FUNCTION public.has_organization_access(org_id uuid)
RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, auth, app_auth
STABLE
AS $$
BEGIN
  -- Service account?
  IF app_auth.check_if_service_account() THEN
    RETURN (current_setting('request.jwt.claims', true)::json->>'organization_id')::uuid = org_id;
  END IF;

  -- Otherwise, membership
  RETURN EXISTS (
    SELECT 1
    FROM public.organization_members
    WHERE organization_id = org_id
      AND user_id = auth.uid()
  );
END;
$$;

-- Permissions (optional but harmless)
GRANT EXECUTE ON FUNCTION public.get_path_organization_id(text) TO anon, authenticated, service_role, postgres;
GRANT EXECUTE ON FUNCTION public.has_organization_access(uuid)   TO anon, authenticated, service_role, postgres;

-- 3) Re-create Storage policies to reference public.* helpers

-- Read
DROP POLICY IF EXISTS "Organization members can read their files" ON storage.objects;
CREATE POLICY "Organization members can read their files"
ON storage.objects FOR SELECT
USING (
  bucket_id = 'files'
  AND (
    name LIKE '%.keep'
    OR (
      name LIKE 'organizations/%'
      AND public.has_organization_access(public.get_path_organization_id(name))
    )
  )
);

-- Insert
DROP POLICY IF EXISTS "Organization members can upload files" ON storage.objects;
CREATE POLICY "Organization members can upload files"
ON storage.objects FOR INSERT
WITH CHECK (
  bucket_id = 'files'
  AND (
    name LIKE '%.keep'
    OR (
      name LIKE 'organizations/%'
      AND public.has_organization_access(public.get_path_organization_id(name))
    )
  )
);

-- Update
DROP POLICY IF EXISTS "Organization members can update their files" ON storage.objects;
CREATE POLICY "Organization members can update their files"
ON storage.objects FOR UPDATE
USING (
  bucket_id = 'files'
  AND (
    name LIKE '%.keep'
    OR (
      name LIKE 'organizations/%'
      AND public.has_organization_access(public.get_path_organization_id(name))
    )
  )
);

-- Delete
DROP POLICY IF EXISTS "Organization members can delete their files" ON storage.objects;
CREATE POLICY "Organization members can delete their files"
ON storage.objects FOR DELETE
USING (
  bucket_id = 'files'
  AND (
    name LIKE '%.keep'
    OR (
      name LIKE 'organizations/%'
      AND public.has_organization_access(public.get_path_organization_id(name))
    )
  )
);
