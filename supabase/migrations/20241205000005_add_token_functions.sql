-- Function to get organization ID from token
create or replace function get_organization_from_token()
returns uuid
language plpgsql
security definer
as $$
declare
    org_id uuid;
begin
    -- If this is a service account token, get org from claims
    if auth.check_if_service_account() then
        org_id := (current_setting('request.jwt.claims', true)::json->>'organization_id')::uuid;

        -- Update last_used_at in tokens table
        update tokens
        set last_used_at = now()
        where id = (current_setting('request.jwt.claims', true)::json->>'token_id')::uuid;

        return org_id;
    end if;

    -- Otherwise, get organization from membership
    select organization_id into org_id
    from organization_members
    where user_id = auth.uid()
    limit 1;

    if org_id is null then
        raise exception 'No organization found for current user/token';
    end if;

    return org_id;
end;
$$;

-- Grant execute permissions
grant execute on function get_organization_from_token() to authenticated, anon;

-- Update jobs insert policy to use the function
drop policy if exists "Organization members can insert jobs" on jobs;
create policy "Organization members can insert jobs"
    on jobs for insert
    with check (
        -- For new jobs, organization_id must match the user's organization
        organization_id = get_organization_from_token()
    );
