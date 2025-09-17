-- Create tokens table for service accounts
create table if not exists "public"."tokens" (
    "id" uuid not null default gen_random_uuid(),
    "organization_id" uuid not null references organizations(id) on delete cascade,
    "name" text not null,
    "token_hash" text,
    "expires_at" timestamp with time zone,
    "created_at" timestamp with time zone default timezone('utc'::text, now()) not null,
    "created_by" uuid not null references auth.users(id) on delete cascade,
    "last_used_at" timestamp with time zone,
    primary key (id)
);

-- Enable RLS
alter table "public"."tokens" enable row level security;

-- Create policies for tokens
create policy "Organization members can view their tokens"
    on tokens for select
    using (is_organization_member(organization_id));

create policy "Organization admins can manage tokens"
    on tokens for all
    using (is_organization_admin(organization_id));

-- Function to get JWT secret
create or replace function get_jwt_secret()
returns text
language plpgsql
security definer
as $$
begin
    return current_setting('app.jwt_secret', true);
end;
$$;

-- Function to create service account token
create or replace function create_service_account_token(
    org_id uuid,
    token_name text,
    created_by uuid,
    expires_at timestamp with time zone default null
)
returns table (
    token_id uuid,
    jwt_token text
)
language plpgsql
security definer
set search_path = public
as $$
declare
    v_token_id uuid;
    v_jwt_secret text;
    v_jwt_token text;
begin
    -- Get JWT secret
    v_jwt_secret := get_jwt_secret();

    if v_jwt_secret is null then
        raise exception 'JWT secret not configured';
    end if;

    -- Create token record
    insert into tokens (organization_id, name, expires_at, created_by)
    values (org_id, token_name, expires_at, created_by)
    returning id into v_token_id;

    -- Create JWT token with service account claims
    v_jwt_token := extensions.sign(
        json_build_object(
            'role', 'authenticated',
            'iss', 'supabase',
            'iat', extract(epoch from now())::integer,
            'exp', case
                when expires_at is null then extract(epoch from now() + interval '10 years')::integer
                else extract(epoch from expires_at)::integer
            end,
            'is_service_account', true,
            'organization_id', org_id,
            'token_id', v_token_id
        )::json,
        v_jwt_secret
    );

    -- Update token hash
    update tokens
    set token_hash = v_jwt_token
    where id = v_token_id;

    return query select v_token_id, v_jwt_token;
end;
$$;

-- Function to check if request is from service account
create or replace function auth.check_if_service_account()
returns boolean as $$
  select coalesce(
    current_setting('request.jwt.claims', true)::json->>'is_service_account',
    'false'
  )::boolean;
$$ language sql security definer;

-- Update organization member check to handle service accounts
create or replace function is_organization_member(org_id uuid)
returns boolean
language plpgsql
security definer
as $$
begin
    -- If this is a service account, check the organization claim
    if auth.check_if_service_account() then
        return (current_setting('request.jwt.claims', true)::json->>'organization_id')::uuid = org_id;
    end if;

    -- Otherwise check normal membership
    return exists (
        select 1
        from organization_members
        where organization_id = org_id
        and user_id = auth.uid()
    );
end;
$$;

-- Update organization admin check to handle service accounts
create or replace function is_organization_admin(org_id uuid)
returns boolean
language plpgsql
security definer
as $$
begin
    -- Service accounts have admin access to their organization
    if auth.check_if_service_account() then
        return (current_setting('request.jwt.claims', true)::json->>'organization_id')::uuid = org_id;
    end if;

    -- Otherwise check normal admin membership
    return exists (
        select 1
        from organization_members
        where organization_id = org_id
        and user_id = auth.uid()
        and role = 'admin'
    );
end;
$$;

-- Grant necessary permissions
grant all on table tokens to postgres, authenticated;
grant execute on function create_service_account_token(uuid, text, uuid, timestamp with time zone) to authenticated;
