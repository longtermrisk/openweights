-- A budgeted API key must not be able to escape its spending limit.
--
-- Before this migration every API-key JWT was an organization admin
-- (is_organization_admin), so a budgeted key could: mint an unbudgeted sibling
-- with create_api_token; INSERT a token row with a hash it chose, or overwrite the
-- hash of an unbudgeted key, through the api_tokens RLS policy; read provider
-- credentials (RUNPOD_API_KEY) from organization_secrets; and add a member as admin.
--
-- After it:
--   * A key with a spending_limits row is not an organization admin. Unbudgeted keys
--     and signed-in admin users keep their previous power (`ow token create`,
--     `ow env`, the dashboard).
--   * A budgeted key can only create keys through create_api_token_with_limit, with
--     a limit no larger than its remaining budget, and that amount is moved out of its
--     own limit. The total budget of a key and all its descendants therefore never
--     exceeds what a human admin granted.
BEGIN;
SET LOCAL lock_timeout = '5s';

CREATE OR REPLACE FUNCTION public.is_organization_admin(org_id uuid)
RETURNS boolean
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path TO 'public'
AS $$
DECLARE claims jsonb := nullif(current_setting('request.jwt.claims', true), '')::jsonb;
BEGIN
    -- API keys are admins of their organization unless they carry a spending limit.
    IF (claims ->> 'organization_id')::uuid = org_id THEN
        RETURN NOT EXISTS (
            SELECT 1 FROM spending_limits
            WHERE api_token_id = (claims ->> 'api_token_id')::uuid
        );
    END IF;

    RETURN EXISTS (
        SELECT 1
        FROM public.organization_members
        WHERE organization_id = org_id
          AND user_id = auth.uid()
          AND role = 'admin'
    );
END;
$$;

CREATE FUNCTION public.key_spent_usd(org_id uuid, key_id uuid) RETURNS numeric
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
    SELECT coalesce(sum(coalesce(direct_usd, 0) + coalesce(overhead_usd, 0)), 0)
    FROM cost_job_amounts WHERE organization_id = org_id AND api_token_id = key_id;
$$;
REVOKE ALL ON FUNCTION public.key_spent_usd(uuid, uuid) FROM PUBLIC, anon, authenticated;

-- The token-minting body of the old create_api_token, without authorization.
-- Only callable from the authorizing wrappers below.
CREATE FUNCTION public.mint_api_token(
    org_id uuid, token_name text, expires_at timestamptz
) RETURNS TABLE(token_id uuid, token text)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE
    v_token_id uuid;
    v_token text;
    v_created_by uuid;
    v_api_token_id uuid;
BEGIN
    v_token := 'ow_' || encode(extensions.gen_random_bytes(24), 'hex');

    -- created_by: the signed-in user, or the creator of the calling API key
    v_created_by := auth.uid();
    IF v_created_by IS NULL THEN
        v_api_token_id := (nullif(current_setting('request.jwt.claims', true), '')::jsonb ->> 'api_token_id')::uuid;
        IF v_api_token_id IS NOT NULL THEN
            SELECT created_by INTO v_created_by FROM api_tokens WHERE id = v_api_token_id;
        END IF;
        IF v_created_by IS NULL THEN
            RAISE EXCEPTION 'Could not determine creator';
        END IF;
    END IF;

    INSERT INTO api_tokens (organization_id, name, token_prefix, token_hash, created_by, expires_at)
    VALUES (org_id, token_name, substring(v_token, 1, 11),
            encode(extensions.digest(v_token, 'sha256'), 'hex'), v_created_by, mint_api_token.expires_at)
    RETURNING id INTO v_token_id;

    -- Only time the token is visible in plaintext
    RETURN QUERY SELECT v_token_id, v_token;
END $$;
REVOKE ALL ON FUNCTION public.mint_api_token(uuid, text, timestamptz) FROM PUBLIC, anon, authenticated;

CREATE OR REPLACE FUNCTION public.create_api_token(
    org_id uuid,
    token_name text,
    expires_at timestamp with time zone DEFAULT NULL
)
RETURNS TABLE(token_id uuid, token text)
LANGUAGE plpgsql SECURITY DEFINER
SET search_path TO 'public'
AS $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM spending_limits
        WHERE api_token_id = (nullif(current_setting('request.jwt.claims', true), '')::jsonb ->> 'api_token_id')::uuid
    ) THEN
        RAISE EXCEPTION 'Budgeted API keys can only create keys with a spending limit taken from their own remaining budget (create_api_token_with_limit)'
            USING ERRCODE = '42501';
    END IF;
    IF NOT is_organization_admin(org_id) THEN
        RAISE EXCEPTION 'Only organization admins can create API tokens';
    END IF;
    RETURN QUERY SELECT * FROM mint_api_token(org_id, token_name, expires_at);
END;
$$;

CREATE OR REPLACE FUNCTION public.create_api_token_with_limit(
    org_id uuid, token_name text, spending_limit_usd numeric,
    expires_at timestamptz DEFAULT NULL
) RETURNS TABLE(token_id uuid, token text)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE
    new_key record;
    caller_key uuid := (nullif(current_setting('request.jwt.claims', true), '')::jsonb ->> 'api_token_id')::uuid;
    parent public.spending_limits;
    remaining numeric;
BEGIN
    IF spending_limit_usd IS NULL OR spending_limit_usd < 0
       OR spending_limit_usd IN ('NaN'::numeric, 'Infinity'::numeric, '-Infinity'::numeric) THEN
        RAISE EXCEPTION 'Spending limit must be a finite nonnegative USD amount' USING ERRCODE = '22023';
    END IF;

    -- Lock the caller's budget row: serializes with job admission (guard_job_spending)
    -- and with concurrent mints from the same key.
    SELECT * INTO parent FROM spending_limits WHERE api_token_id = caller_key FOR UPDATE;
    IF parent.api_token_id IS NOT NULL THEN
        IF parent.organization_id <> org_id OR NOT EXISTS (
            SELECT 1 FROM api_tokens WHERE id = caller_key AND organization_id = org_id
              AND revoked_at IS NULL AND (api_tokens.expires_at IS NULL OR api_tokens.expires_at > now())
        ) THEN
            RAISE EXCEPTION 'Organization access denied' USING ERRCODE = '42501';
        END IF;
        remaining := parent.limit_usd - key_spent_usd(org_id, caller_key);
        IF spending_limit_usd > remaining THEN
            RAISE EXCEPTION 'Spending limit % USD exceeds this key''s remaining budget of % USD',
                spending_limit_usd, round(greatest(remaining, 0), 2) USING ERRCODE = '42501';
        END IF;
        UPDATE spending_limits SET limit_usd = limit_usd - spending_limit_usd, updated_at = now()
        WHERE api_token_id = caller_key;
    ELSIF NOT is_organization_admin(org_id) THEN
        RAISE EXCEPTION 'Only organization admins can create API tokens';
    END IF;

    SELECT * INTO new_key FROM mint_api_token(org_id, token_name, expires_at);
    INSERT INTO public.spending_limits(organization_id, api_token_id, limit_usd)
    VALUES (org_id, new_key.token_id, spending_limit_usd);
    RETURN QUERY SELECT new_key.token_id, new_key.token;
END $$;

COMMIT;
