-- Create organization_secrets table
create table if not exists "public"."organization_secrets" (
    "id" uuid not null default gen_random_uuid(),
    "organization_id" uuid not null references organizations(id) on delete cascade,
    "name" text not null,
    "value" text not null,
    "created_at" timestamp with time zone default timezone('utc'::text, now()) not null,
    "updated_at" timestamp with time zone default timezone('utc'::text, now()) not null,
    primary key (id),
    unique (organization_id, name)
);

-- Enable RLS
alter table "public"."organization_secrets" enable row level security;

-- Create policies for secrets
create policy "Organization admins can manage secrets"
    on organization_secrets for all
    using (is_organization_admin(organization_id));

-- Add trigger for updated_at
create trigger set_updated_at_organization_secrets
    before update on public.organization_secrets
    for each row
    execute function public.handle_updated_at();

-- Function to manage organization secrets
create or replace function manage_organization_secret(
    org_id uuid,
    secret_name text,
    secret_value text
)
returns uuid
language plpgsql
security definer
set search_path = public
as $$
declare
    v_secret_id uuid;
begin
    -- Check if user is admin
    if not is_organization_admin(org_id) then
        raise exception 'Only organization admins can manage secrets';
    end if;

    -- Insert or update secret
    insert into organization_secrets (organization_id, name, value)
    values (org_id, secret_name, secret_value)
    on conflict (organization_id, name)
    do update set value = excluded.value, updated_at = now()
    returning id into v_secret_id;

    return v_secret_id;
end;
$$;

-- Function to delete organization secret
create or replace function delete_organization_secret(
    org_id uuid,
    secret_name text
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
begin
    -- Check if user is admin
    if not is_organization_admin(org_id) then
        raise exception 'Only organization admins can manage secrets';
    end if;

    -- Delete secret
    delete from organization_secrets
    where organization_id = org_id and name = secret_name;

    return found;
end;
$$;

-- Function to create new organization
create or replace function create_organization(
    org_name text
)
returns uuid
language plpgsql
security definer
set search_path = public
as $$
declare
    v_org_id uuid;
begin
    -- Check if authenticated
    if auth.uid() is null then
        raise exception 'Authentication required';
    end if;

    -- Create organization
    insert into organizations (name)
    values (org_name)
    returning id into v_org_id;

    -- Add creator as admin
    insert into organization_members (organization_id, user_id, role)
    values (v_org_id, auth.uid(), 'admin');

    return v_org_id;
end;
$$;

-- Function to update organization
create or replace function update_organization(
    org_id uuid,
    new_name text
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
begin
    -- Check if user is admin
    if not is_organization_admin(org_id) then
        raise exception 'Only organization admins can update organization';
    end if;

    -- Update organization
    update organizations
    set name = new_name
    where id = org_id;

    return found;
end;
$$;

-- Function to remove member from organization
create or replace function remove_organization_member(
    org_id uuid,
    member_id uuid
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
    v_member_role organization_role;
    v_admin_count integer;
begin
    -- Check if user is admin
    if not is_organization_admin(org_id) then
        raise exception 'Only organization admins can remove members';
    end if;

    -- Get member's role
    select role into v_member_role
    from organization_members
    where organization_id = org_id and user_id = member_id;

    -- If member is admin, check if they're not the last admin
    if v_member_role = 'admin' then
        select count(*) into v_admin_count
        from organization_members
        where organization_id = org_id and role = 'admin';

        if v_admin_count <= 1 then
            raise exception 'Cannot remove the last admin';
        end if;
    end if;

    -- Remove member
    delete from organization_members
    where organization_id = org_id and user_id = member_id;

    return found;
end;
$$;

-- Grant execute permissions
grant execute on function manage_organization_secret(uuid, text, text) to authenticated;
grant execute on function delete_organization_secret(uuid, text) to authenticated;
grant execute on function create_organization(text) to authenticated;
grant execute on function update_organization(uuid, text) to authenticated;
grant execute on function remove_organization_member(uuid, uuid) to authenticated;
