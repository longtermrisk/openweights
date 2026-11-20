-- What the broker needs to know about a worker besides who it is.
--
-- The broker (openweights/broker) holds no database credential. It learns everything
-- from the worker token a request carries: with the token's own JWT it reads the
-- token's row. To let workers use the organization's LiteLLM proxy through the
-- broker, the broker has to know where that proxy is. The address must come from
-- the organization's admins and from nobody else: whoever chooses it receives the
-- organization's LiteLLM key with every forwarded request.
--
-- So the cluster manager, which reads the organization's settings with the
-- organization's key, writes the address onto the worker token when it creates it
-- (api_tokens.broker_settings). A worker token cannot write api_tokens, so the worker
-- cannot change it.
--
-- LITELLM_BASE_URL itself is an ordinary organization setting, not a secret; the
-- dashboard may show it.
--
-- Safe while older managers run: they call create_worker_token with three arguments,
-- which still works (the fourth has a default).
BEGIN;
SET LOCAL lock_timeout = '5s';

ALTER TABLE public.api_tokens ADD COLUMN IF NOT EXISTS broker_settings jsonb;
COMMENT ON COLUMN public.api_tokens.broker_settings IS
    'For a worker token: what the credential broker needs to serve this worker, written by the cluster manager. Never a secret.';

-- Mirrored in openweights/dashboard/backend/database.py (PLAIN_SETTINGS).
CREATE OR REPLACE FUNCTION public.is_plain_organization_setting(secret_name text) RETURNS boolean
LANGUAGE sql IMMUTABLE AS $$
    SELECT secret_name IN ('HF_USER', 'HF_ORG', 'OW_MANAGED', 'MAX_WORKERS', 'LITELLM_BASE_URL');
$$;

DROP FUNCTION public.create_worker_token(uuid, text, timestamptz);
CREATE FUNCTION public.create_worker_token(
    org_id uuid, for_worker_id text, expires_at timestamptz, broker_settings jsonb DEFAULT NULL
) RETURNS TABLE(token_id uuid, token text)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE
    new_token record;
    litellm_url text := create_worker_token.broker_settings ->> 'litellm_base_url';
BEGIN
    IF NOT is_organization_admin(org_id) THEN
        RAISE EXCEPTION 'Only organization admins can create worker tokens' USING ERRCODE = '42501';
    END IF;
    IF create_worker_token.expires_at IS NULL
       OR create_worker_token.expires_at <= now()
       OR create_worker_token.expires_at > now() + interval '8 days' THEN
        RAISE EXCEPTION 'A worker token must expire within 8 days' USING ERRCODE = '22023';
    END IF;
    IF create_worker_token.broker_settings IS NOT NULL AND (
        jsonb_typeof(create_worker_token.broker_settings) <> 'object'
        OR create_worker_token.broker_settings - 'litellm_base_url' <> '{}'::jsonb
        OR litellm_url IS NULL
        OR litellm_url !~ '^https://[^[:space:]]+$'
        OR length(litellm_url) > 500
    ) THEN
        RAISE EXCEPTION 'broker_settings may only hold litellm_base_url, an https URL'
            USING ERRCODE = '22023';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM worker w
        WHERE w.id = for_worker_id AND w.organization_id = org_id
          AND w.status IS DISTINCT FROM 'terminated'
    ) THEN
        RAISE EXCEPTION 'No running worker % in this organization', for_worker_id
            USING ERRCODE = '22023';
    END IF;

    UPDATE api_tokens SET revoked_at = now()
    WHERE worker_id = for_worker_id AND organization_id = org_id AND revoked_at IS NULL;

    SELECT * INTO new_token
    FROM mint_api_token(org_id, 'worker ' || for_worker_id, create_worker_token.expires_at);
    UPDATE api_tokens
    SET worker_id = for_worker_id, broker_settings = create_worker_token.broker_settings
    WHERE id = new_token.token_id;
    RETURN QUERY SELECT new_token.token_id, new_token.token;
END $$;
REVOKE ALL ON FUNCTION public.create_worker_token(uuid, text, timestamptz, jsonb) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.create_worker_token(uuid, text, timestamptz, jsonb) TO authenticated, service_role;

COMMIT;

NOTIFY pgrst, 'reload schema';
