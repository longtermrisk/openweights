-- Organizations policies
create policy "Enable read access for organization members"
    on organizations for select
    using (is_organization_member(id));

create policy "Enable write access for organization admins"
    on organizations for all
    using (is_organization_admin(id));

-- Organization members policies
create policy "Enable read access for members"
    on organization_members for select
    using (is_organization_member(organization_id));

create policy "Enable write access for admins"
    on organization_members for all
    using (is_organization_admin(organization_id));

-- Jobs policies
create policy "Organization members can read jobs"
    on jobs for select
    using (is_organization_member(organization_id));

create policy "Organization members can insert jobs"
    on jobs for insert
    with check (
        is_organization_member(organization_id)
    );

create policy "Organization members can update their jobs"
    on jobs for update
    using (is_organization_member(organization_id));

create policy "Organization members can delete their jobs"
    on jobs for delete
    using (is_organization_member(organization_id));

-- Runs policies
create policy "Enable access for organization members"
    on runs for all
    using (
        exists (
            select 1
            from jobs j
            where j.id = runs.job_id
            and is_organization_member(j.organization_id)
        )
    );

-- Workers policies
create policy "Enable access for organization members"
    on worker for all
    using (is_organization_member(organization_id));

-- Events policies
create policy "Enable access for organization members"
    on events for all
    using (
        exists (
            select 1
            from runs r
            join jobs j on j.id = r.job_id
            where r.id = events.run_id
            and is_organization_member(j.organization_id)
        )
    );
