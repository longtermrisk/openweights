BEGIN;
-- Create the key and its initial budget in one transaction. Existing callers of
-- create_api_token keep their behavior and authorization. The creator may choose
-- a budget for a new key; changing an existing budget still requires an admin user.
CREATE FUNCTION public.create_api_token_with_limit(
    org_id uuid, token_name text, spending_limit_usd numeric,
    expires_at timestamptz DEFAULT NULL
) RETURNS TABLE(token_id uuid, token text)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE new_key record;
BEGIN
    IF spending_limit_usd IS NULL OR spending_limit_usd < 0
       OR spending_limit_usd IN ('NaN'::numeric, 'Infinity'::numeric, '-Infinity'::numeric) THEN
        RAISE EXCEPTION 'Spending limit must be a finite nonnegative USD amount' USING ERRCODE = '22023';
    END IF;
    SELECT * INTO new_key FROM public.create_api_token(org_id, token_name, expires_at);
    INSERT INTO public.spending_limits(organization_id, api_token_id, limit_usd)
    VALUES (org_id, new_key.token_id, spending_limit_usd);
    RETURN QUERY SELECT new_key.token_id, new_key.token;
END $$;
REVOKE ALL ON FUNCTION public.create_api_token_with_limit(uuid, text, numeric, timestamptz) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.create_api_token_with_limit(uuid, text, numeric, timestamptz) TO authenticated;

COMMIT;
