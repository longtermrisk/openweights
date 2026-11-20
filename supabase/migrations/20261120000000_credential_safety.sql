-- Credential safety, part 1 and 2.
--
-- 1. organization_secrets: a signed-in member's browser must not be able to read
--    stored secret values.
--
--    Before: one policy let every organization admin (signed-in admin users and API
--    keys without a spending limit) read and write the table directly. The dashboard
--    is a browser app holding the user's session JWT and the anon key, so an admin
--    could run `supabase.from('organization_secrets').select('value')` and get every
--    key in plaintext, whatever the dashboard backend returned.
--
--    After: direct access to the table is for API keys only (the cluster manager,
--    `ow env import/show`, `ow manage`), plus the four settings that are not secrets,
--    which the dashboard backend still reads to validate a new Hugging Face token.
--    Signed-in users go through security-definer functions:
--      * list_organization_secrets_masked (new): names, the last four characters of
--        a secret, update time;
--      * manage_organization_secret / delete_organization_secret (unchanged).
--
--    Not closed by this migration: an admin user can still create an API key and read
--    the secrets with it, because the cluster manager itself authenticates as an
--    ordinary organization API key. That needs a separate manager identity.
--
-- 2. worker.termination_requested_at: a worker that wants its pod terminated sets
--    this instead of calling RunPod with the organization's API key; the cluster
--    manager, which holds the key, terminates the pod.
--
-- Safe while the old manager, dashboard and SDKs are running: the column is new and
-- nullable, API-key access is unchanged, and the old dashboard merely stops listing
-- secret values (it shows only the non-secret settings).
BEGIN;
SET LOCAL lock_timeout = '5s';

ALTER TABLE public.worker ADD COLUMN IF NOT EXISTS termination_requested_at timestamptz;

-- Settings stored next to the secrets that are not credentials. Mirrored in
-- openweights/dashboard/backend/database.py (PLAIN_SETTINGS).
CREATE FUNCTION public.is_plain_organization_setting(secret_name text) RETURNS boolean
LANGUAGE sql IMMUTABLE AS $$
    SELECT secret_name IN ('HF_USER', 'HF_ORG', 'OW_MANAGED', 'MAX_WORKERS');
$$;

DROP POLICY "Organization admins can manage secrets" ON public.organization_secrets;
CREATE POLICY "API keys manage secrets, users only plain settings"
    ON public.organization_secrets
    USING (
        public.is_organization_admin(organization_id)
        AND (auth.uid() IS NULL OR public.is_plain_organization_setting(name))
    );

CREATE FUNCTION public.list_organization_secrets_masked(org_id uuid)
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
    ORDER BY s.name;
END $$;
REVOKE ALL ON FUNCTION public.list_organization_secrets_masked(uuid) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.list_organization_secrets_masked(uuid) TO authenticated, service_role;

COMMIT;

NOTIFY pgrst, 'reload schema';
