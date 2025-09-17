-- Create role enum type
create type "public"."organization_role" as enum ('admin', 'user');

-- Create organizations table
create table if not exists "public"."organizations" (
    "id" uuid not null default gen_random_uuid(),
    "created_at" timestamp with time zone default timezone('utc'::text, now()) not null,
    "updated_at" timestamp with time zone default timezone('utc'::text, now()) not null,
    "name" text not null,
    primary key (id)
);

-- Create organization_members junction table with role
create table if not exists "public"."organization_members" (
    "organization_id" uuid not null references organizations(id) on delete cascade,
    "user_id" uuid not null references auth.users(id) on delete cascade,
    "role" organization_role not null default 'user',
    "created_at" timestamp with time zone default timezone('utc'::text, now()) not null,
    primary key (organization_id, user_id)
);

-- Create a default organization for existing data
insert into "public"."organizations" (id, name)
values ('00000000-0000-0000-0000-000000000000', 'Default Organization');

-- Add organization ownership to jobs and workers
alter table "public"."jobs"
    add column "organization_id" uuid references organizations(id) on delete cascade;

alter table "public"."worker"
    add column "organization_id" uuid references organizations(id) on delete cascade;

-- Migrate existing jobs and workers to default organization
update "public"."jobs"
set "organization_id" = '00000000-0000-0000-0000-000000000000'
where "organization_id" is null;

update "public"."worker"
set "organization_id" = '00000000-0000-0000-0000-000000000000'
where "organization_id" is null;

-- Make organization_id not null after migration
alter table "public"."jobs"
    alter column "organization_id" set not null;

alter table "public"."worker"
    alter column "organization_id" set not null;

-- Enable RLS on all tables
alter table "public"."organizations" enable row level security;
alter table "public"."organization_members" enable row level security;
alter table "public"."jobs" enable row level security;
alter table "public"."runs" enable row level security;
alter table "public"."worker" enable row level security;
alter table "public"."events" enable row level security;

-- Grant permissions
grant usage on schema public to postgres, anon, authenticated, service_role;
grant all privileges on all tables in schema public to postgres, service_role;
grant all privileges on all sequences in schema public to postgres, service_role;
grant select, insert, update, delete on all tables in schema public to authenticated;

-- Add updated_at trigger for organizations
create trigger set_updated_at_organizations
    before update on public.organizations
    for each row
    execute function public.handle_updated_at();
