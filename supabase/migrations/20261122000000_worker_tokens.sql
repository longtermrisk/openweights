-- A token per worker.
--
-- Until now every worker pod held the organization's own API key
-- (OPENWEIGHTS_API_KEY): an admin key that never expires, can read the organization's
-- settings and create more keys, and that the code of any job can read from its
-- environment and keep.
--
-- A worker token is an API token bound to one worker (api_tokens.worker_id):
--   * created by the cluster manager when it starts the pod (create_worker_token);
--   * it expires, and it is revoked the moment its worker is marked terminated or its
--     row is deleted;
--   * it is not an organization admin: it cannot read organization_secrets or
--     organization_secret_metadata, create or revoke tokens, change spending limits,
--     members or the organization. It can do what a worker does, which is what any
--     organization member may do: jobs, runs, events, files and worker rows.
--
-- Not closed by this: within its lifetime a worker token can still read and change
-- other jobs of its organization and submit jobs, like the organization key it
-- replaces could. And a JWT obtained with a token stays valid for up to an hour after
-- the token is revoked, as for every API token.
--
-- Safe while older managers and workers run: existing tokens have no worker_id and
-- behave exactly as before; an older manager simply keeps passing the organization key.
BEGIN;
SET LOCAL lock_timeout = '5s';

ALTER TABLE public.api_tokens ADD COLUMN IF NOT EXISTS worker_id text;
CREATE INDEX IF NOT EXISTS api_tokens_worker_id_idx ON public.api_tokens(worker_id)
    WHERE worker_id IS NOT NULL;
COMMENT ON COLUMN public.api_tokens.worker_id IS
    'Set for a worker token: the worker it belongs to. Such a token is never an organization admin.';

-- API keys are admins of their organization unless they carry a spending limit or
-- belong to a worker.
CREATE OR REPLACE FUNCTION public.is_organization_admin(org_id uuid)
RETURNS boolean
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path TO 'public'
AS $$
DECLARE claims jsonb := nullif(current_setting('request.jwt.claims', true), '')::jsonb;
BEGIN
    IF (claims ->> 'organization_id')::uuid = org_id THEN
        RETURN NOT EXISTS (
            SELECT 1 FROM spending_limits
            WHERE api_token_id = (claims ->> 'api_token_id')::uuid
        ) AND NOT EXISTS (
            SELECT 1 FROM api_tokens
            WHERE id = (claims ->> 'api_token_id')::uuid AND worker_id IS NOT NULL
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

-- Called by the cluster manager, with the organization's key, for a worker it is
-- about to start. Earlier tokens of the same worker (an earlier start attempt) are
-- revoked: a worker has one token.
CREATE FUNCTION public.create_worker_token(
    org_id uuid, for_worker_id text, expires_at timestamptz
) RETURNS TABLE(token_id uuid, token text)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE new_token record;
BEGIN
    IF NOT is_organization_admin(org_id) THEN
        RAISE EXCEPTION 'Only organization admins can create worker tokens' USING ERRCODE = '42501';
    END IF;
    IF create_worker_token.expires_at IS NULL
       OR create_worker_token.expires_at <= now()
       OR create_worker_token.expires_at > now() + interval '8 days' THEN
        RAISE EXCEPTION 'A worker token must expire within 8 days' USING ERRCODE = '22023';
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
    UPDATE api_tokens SET worker_id = for_worker_id WHERE id = new_token.token_id;
    RETURN QUERY SELECT new_token.token_id, new_token.token;
END $$;
REVOKE ALL ON FUNCTION public.create_worker_token(uuid, text, timestamptz) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.create_worker_token(uuid, text, timestamptz) TO authenticated, service_role;

-- A token does not outlive its worker, whoever ends the worker and however.
CREATE FUNCTION public.revoke_tokens_of_ended_worker() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
BEGIN
    UPDATE api_tokens SET revoked_at = now()
    WHERE worker_id = OLD.id AND revoked_at IS NULL;
    RETURN NULL;
END $$;
CREATE TRIGGER revoke_tokens_of_terminated_worker
    AFTER UPDATE OF status ON public.worker
    FOR EACH ROW WHEN (NEW.status = 'terminated')
    EXECUTE FUNCTION public.revoke_tokens_of_ended_worker();
CREATE TRIGGER revoke_tokens_of_deleted_worker
    AFTER DELETE ON public.worker
    FOR EACH ROW EXECUTE FUNCTION public.revoke_tokens_of_ended_worker();

-- Nobody can turn a worker token into an ordinary key through the API except an
-- organization admin, who could create an ordinary key anyway: writing api_tokens
-- directly is for admins only ("Organization admins can manage tokens"), and a worker
-- token is not one.

COMMIT;

NOTIFY pgrst, 'reload schema';
