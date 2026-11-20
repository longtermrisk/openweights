-- Secrets and environment variables become two things.
--
-- Until now every organization setting, credential or not, was a row of
-- organization_secrets, and the cluster manager passed all of them to every worker.
--
-- From here on:
--   * environment variables stay rows of organization_secrets, handled as before;
--   * secrets (HF_TOKEN, OPENWEIGHTS_API_KEY, RUNPOD_API_KEY, LITELLM_API_KEY) can live
--     in a secret store outside the database (openweights/secret_store.py). For such a
--     secret the database keeps only what may be displayed: its name, its last four
--     characters and when it was last set (organization_secret_metadata).
--
-- This migration only adds the metadata table and its functions. Nothing moves by
-- itself: a secret moves when the dashboard, configured with a secret store, saves it,
-- or when an operator runs `python -m openweights.secret_store_migration`. Deployments
-- without a secret store keep using organization_secrets and never touch the new table.
--
-- Safe while older managers, dashboards and SDKs run: nothing existing changes except
-- list_organization_secrets_masked(), which also lists secrets kept in the store (the
-- 0.16 dashboard shows them like any other masked secret).
BEGIN;
SET LOCAL lock_timeout = '5s';

-- Mirrored in openweights/secret_store.py (MANAGED_SECRETS).
CREATE FUNCTION public.is_managed_organization_secret(secret_name text) RETURNS boolean
LANGUAGE sql IMMUTABLE AS $$
    SELECT secret_name IN ('HF_TOKEN', 'OPENWEIGHTS_API_KEY', 'RUNPOD_API_KEY', 'LITELLM_API_KEY');
$$;

CREATE TABLE public.organization_secret_metadata (
    organization_id uuid NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    name text NOT NULL CHECK (public.is_managed_organization_secret(name)),
    last_four text CHECK (last_four IS NULL OR length(last_four) <= 4),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (organization_id, name)
);
COMMENT ON TABLE public.organization_secret_metadata IS
    'Secrets kept in the secret store: what may be displayed about them. Never the value.';

ALTER TABLE public.organization_secret_metadata ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.organization_secret_metadata FROM PUBLIC, anon, authenticated;
GRANT SELECT ON public.organization_secret_metadata TO authenticated;
GRANT ALL ON public.organization_secret_metadata TO service_role;

-- Read by signed-in admins (dashboard) and by API keys without a spending limit (the
-- cluster manager, to learn that a secret changed). Written only by the functions below.
CREATE POLICY "Organization admins can see which secrets are stored"
    ON public.organization_secret_metadata FOR SELECT
    USING (public.is_organization_admin(organization_id));

-- Called after a value was written to the secret store. Removes a copy of the same
-- secret from organization_secrets, so a value is in one place only.
CREATE FUNCTION public.record_organization_secret(
    org_id uuid, secret_name text, secret_last_four text
) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
BEGIN
    IF NOT is_organization_admin(org_id) THEN
        RAISE EXCEPTION 'Only organization admins can manage secrets' USING ERRCODE = '42501';
    END IF;
    IF NOT is_managed_organization_secret(secret_name) THEN
        RAISE EXCEPTION '% is not a secret; store it as an environment variable', secret_name
            USING ERRCODE = '22023';
    END IF;
    INSERT INTO organization_secret_metadata (organization_id, name, last_four, updated_at)
    VALUES (org_id, secret_name, secret_last_four, now())
    ON CONFLICT (organization_id, name)
    DO UPDATE SET last_four = excluded.last_four, updated_at = now();
    DELETE FROM organization_secrets WHERE organization_id = org_id AND name = secret_name;
END $$;

CREATE FUNCTION public.forget_organization_secret(org_id uuid, secret_name text)
RETURNS boolean
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE removed integer;
BEGIN
    IF NOT is_organization_admin(org_id) THEN
        RAISE EXCEPTION 'Only organization admins can manage secrets' USING ERRCODE = '42501';
    END IF;
    DELETE FROM organization_secret_metadata
    WHERE organization_id = org_id AND name = secret_name;
    GET DIAGNOSTICS removed = ROW_COUNT;
    RETURN removed > 0;
END $$;

REVOKE ALL ON FUNCTION public.record_organization_secret(uuid, text, text) FROM PUBLIC, anon;
REVOKE ALL ON FUNCTION public.forget_organization_secret(uuid, text) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.record_organization_secret(uuid, text, text) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.forget_organization_secret(uuid, text) TO authenticated, service_role;

-- A secret that is in the store must not get a second copy in organization_secrets
-- (`ow env import` writes there): the manager would have two values to choose from,
-- and the copy would be passed to workers.
CREATE FUNCTION public.reject_copy_of_stored_secret() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM organization_secret_metadata m
        WHERE m.organization_id = NEW.organization_id AND m.name = NEW.name
    ) THEN
        RAISE EXCEPTION '% is kept in the secret store; replace it in the dashboard', NEW.name
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER reject_copy_of_stored_secret
    BEFORE INSERT OR UPDATE OF name, organization_id ON public.organization_secrets
    FOR EACH ROW EXECUTE FUNCTION public.reject_copy_of_stored_secret();

-- The masked listing also covers secrets kept in the store.
CREATE OR REPLACE FUNCTION public.list_organization_secrets_masked(org_id uuid)
RETURNS TABLE(name text, plain_value text, last_four text, updated_at timestamptz)
LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path = public AS $$
BEGIN
    IF NOT is_organization_admin(org_id) THEN
        RAISE EXCEPTION 'Only organization admins can list secrets' USING ERRCODE = '42501';
    END IF;
    RETURN QUERY
    SELECT s.name,
           CASE WHEN is_plain_organization_setting(s.name) THEN s.value END,
           -- Four characters of a short value would be most of it.
           CASE WHEN NOT is_plain_organization_setting(s.name) AND length(s.value) >= 12
                THEN right(s.value, 4) END,
           s.updated_at
    FROM organization_secrets s
    WHERE s.organization_id = org_id
    UNION ALL
    SELECT m.name, NULL, m.last_four, m.updated_at
    FROM organization_secret_metadata m
    WHERE m.organization_id = org_id
    ORDER BY 1;
END $$;

COMMIT;

NOTIFY pgrst, 'reload schema';
