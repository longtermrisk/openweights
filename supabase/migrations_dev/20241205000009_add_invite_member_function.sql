-- Function to invite a member to an organization
create or replace function public.invite_organization_member(
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
    -- Check if the inviter is an admin of the organization
    if not exists (
        select 1
        from organization_members
        where organization_id = org_id
        and user_id = auth.uid()
        and role = 'admin'
    ) then
        raise exception 'Only organization admins can invite members';
    end if;

    -- Get the user ID for the email
    select au.id, au.email
    into v_user_id, v_email
    from auth.users au
    where lower(au.email) = lower(member_email);

    if v_user_id is null then
        raise exception 'User with email % not found', member_email;
    end if;

    -- Check if user is already a member
    if exists (
        select 1
        from organization_members
        where organization_id = org_id
        and user_id = v_user_id
    ) then
        raise exception 'User is already a member of this organization';
    end if;

    -- Insert the new member
    insert into organization_members (organization_id, user_id, role)
    values (org_id, v_user_id, member_role)
    returning user_id, v_email as email, role into user_id, email, role;

    return next;
end;
$$;

-- Grant execute permissions on the function
grant execute on function public.invite_organization_member(uuid, varchar, public.organization_role) to authenticated;