-- Drop and recreate the get_organization_members function with fully qualified columns
create or replace function get_organization_members(org_id uuid)
returns table (
    user_id uuid,
    email varchar(255),
    role public.organization_role
) security definer
set search_path = public
language plpgsql
as $$
begin
    return query
    select
        om.user_id,
        au.email::varchar(255),  -- Explicitly cast email to match return type
        om.role
    from public.organization_members om
    join auth.users au on au.id = om.user_id
    where om.organization_id = org_id
    and exists (
        select 1
        from public.organization_members viewer
        where viewer.organization_id = org_id
        and viewer.user_id = auth.uid()
    );
end;
$$;

-- Drop and recreate the invite_organization_member function with fully qualified columns
create or replace function invite_organization_member(
    org_id uuid,
    member_email varchar(255),
    member_role public.organization_role
)
returns table (
    user_id uuid,
    email varchar(255),
    role public.organization_role
) security definer
set search_path = public
language plpgsql
as $$
declare
    v_user_id uuid;
    v_email varchar(255);
begin
    -- Check if the inviter is an admin
    if not is_organization_admin(org_id) then
        raise exception 'Only organization admins can invite members';
    end if;

    -- Get the user ID for the email
    select au.id, au.email::varchar(255)
    into v_user_id, v_email
    from auth.users au
    where lower(au.email) = lower(member_email);

    if v_user_id is null then
        raise exception 'User with email % not found', member_email;
    end if;

    -- Check if user is already a member
    if exists (
        select 1
        from organization_members om
        where om.organization_id = org_id
        and om.user_id = v_user_id
    ) then
        raise exception 'User is already a member of this organization';
    end if;

    -- Insert the new member
    insert into organization_members (organization_id, user_id, role)
    values (org_id, v_user_id, member_role);

    -- Return the result
    return query
    select
        v_user_id as user_id,
        v_email as email,
        member_role as role;
end;
$$;
