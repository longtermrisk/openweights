-- Enable RLS on storage.objects
alter table storage.objects enable row level security;

-- Function to extract organization ID from storage path
create or replace function storage.get_path_organization_id(path text)
returns uuid
language plpgsql
as $$
declare
    parts text[];
    org_id uuid;
begin
    -- Split path into parts
    parts := string_to_array(path, '/');

    -- Check if path starts with 'organizations'
    if parts[1] != 'organizations' then
        return null;
    end if;

    -- Try to convert second part to UUID
    begin
        org_id := parts[2]::uuid;
        return org_id;
    exception when others then
        return null;
    end;
end;
$$;

-- Function to check if user has access to organization
create or replace function storage.has_organization_access(org_id uuid)
returns boolean
language plpgsql
security definer
as $$
begin
    -- First check service account tokens
    if auth.check_if_service_account() then
        return (current_setting('request.jwt.claims', true)::json->>'organization_id')::uuid = org_id;
    end if;

    -- Otherwise check organization membership
    return exists (
        select 1
        from public.organization_members
        where organization_id = org_id
        and user_id = auth.uid()
    );
end;
$$;

-- Policy for reading files:
create policy "Organization members can read their files"
on storage.objects for select
using (
    bucket_id = 'files'
    and (
        -- Allow access to .keep files
        name like '%.keep'
        or (
            -- Check if file is in an organization folder and user has access
            name like 'organizations/%'
            and storage.has_organization_access(storage.get_path_organization_id(name))
        )
    )
);

-- Policy for inserting files:
create policy "Organization members can upload files"
on storage.objects for insert
with check (
    bucket_id = 'files'
    and (
        -- Allow .keep files
        name like '%.keep'
        or (
            -- Check if file is in an organization folder and user has access
            name like 'organizations/%'
            and storage.has_organization_access(storage.get_path_organization_id(name))
        )
    )
);

-- Policy for updating files:
create policy "Organization members can update their files"
on storage.objects for update
using (
    bucket_id = 'files'
    and (
        -- Allow .keep files
        name like '%.keep'
        or (
            -- Check if file is in an organization folder and user has access
            name like 'organizations/%'
            and storage.has_organization_access(storage.get_path_organization_id(name))
        )
    )
);

-- Policy for deleting files:
create policy "Organization members can delete their files"
on storage.objects for delete
using (
    bucket_id = 'files'
    and (
        -- Allow .keep files
        name like '%.keep'
        or (
            -- Check if file is in an organization folder and user has access
            name like 'organizations/%'
            and storage.has_organization_access(storage.get_path_organization_id(name))
        )
    )
);

-- Create organization folders
do $$
begin
    -- Create organizations folder
    insert into storage.objects (bucket_id, name, owner, created_at, updated_at, version)
    values ('files', 'organizations/.keep', auth.uid(), now(), now(), '1')
    on conflict do nothing;

    -- Create folder for default organization
    insert into storage.objects (bucket_id, name, owner, created_at, updated_at, version)
    values ('files', 'organizations/00000000-0000-0000-0000-000000000000/.keep', auth.uid(), now(), now(), '1')
    on conflict do nothing;
end;
$$;
