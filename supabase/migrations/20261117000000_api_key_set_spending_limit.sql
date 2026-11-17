-- An API key without a spending limit may change the limits of its organization's keys.
--
-- set_spending_limit used to accept only a signed-in admin user, because when it was
-- written every API key counted as an organization admin, and a budgeted key could
-- have raised its own limit. Since 20261010000000_api_key_budget_containment a key
-- with a spending limit is no longer an admin, so that reason is gone: an unbudgeted
-- key can already mint unbudgeted keys and read the organization's secrets, and
-- refusing it here only kept automation (a service that mints one budgeted key per
-- workspace and later raises or lowers that budget) from adjusting what it created.
--
-- After this migration:
--   * a signed-in admin user: unchanged;
--   * an API key WITHOUT a spending limit: may set, change and remove the limit of any
--     OTHER key of its organization;
--   * an API key WITH a spending limit: still refused (it is not an admin);
--   * no key may change its own limit. Limiting itself would silently turn an admin key
--     into a budgeted one that cannot undo it.
BEGIN;
SET LOCAL lock_timeout = '5s';

CREATE OR REPLACE FUNCTION public.set_spending_limit(org_id uuid, key_id uuid, amount_usd numeric)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE
    caller_key uuid := (nullif(current_setting('request.jwt.claims', true), '')::jsonb ->> 'api_token_id')::uuid;
BEGIN
    -- is_organization_admin: an admin user, or an API key of this organization that
    -- carries no spending limit.
    IF NOT is_organization_admin(org_id) THEN
        RAISE EXCEPTION 'Only an organization admin or an API key without a spending limit can change spending limits'
            USING ERRCODE = '42501';
    END IF;
    IF caller_key IS NOT NULL AND caller_key = key_id THEN
        RAISE EXCEPTION 'An API key cannot change its own spending limit' USING ERRCODE = '42501';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM api_tokens WHERE id = key_id AND organization_id = org_id) THEN
        RAISE EXCEPTION 'API key not found in organization';
    END IF;
    IF amount_usd IS NULL THEN DELETE FROM spending_limits WHERE api_token_id = key_id;
    ELSE
        INSERT INTO spending_limits(organization_id, api_token_id, limit_usd)
        VALUES (org_id, key_id, amount_usd)
        ON CONFLICT (api_token_id) DO UPDATE SET limit_usd = EXCLUDED.limit_usd, updated_at = now();
    END IF;
END $$;

COMMIT;
