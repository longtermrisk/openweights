-- Update get_organization_from_token to only work with service account tokens
create or replace function get_organization_from_token()
returns uuid
language plpgsql
security definer
as $$
declare
    org_id uuid;
begin
    -- Only handle service account tokens
    if not auth.check_if_service_account() then
        raise exception 'Only service account tokens are supported';
    end if;

    -- Get org from claims
    org_id := (current_setting('request.jwt.claims', true)::json->>'organization_id')::uuid;

    -- Update last_used_at in tokens table
    update tokens
    set last_used_at = now()
    where id = (current_setting('request.jwt.claims', true)::json->>'token_id')::uuid;

    return org_id;
end;
$$;
