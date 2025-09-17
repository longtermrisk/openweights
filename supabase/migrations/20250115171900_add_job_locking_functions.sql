-- Function to safely acquire a job with row-level locking
create or replace function acquire_job(job_id uuid, worker_id text)
returns setof jobs as $$
begin
  -- Lock the row and attempt to update it atomically
  return query
  update jobs
  set status = 'in_progress',
      worker_id = worker_id
  where id = job_id
    and status = 'pending'
  returning *;
end;
$$ language plpgsql;

-- Function to safely update job status only if it's still in progress
create or replace function update_job_status_if_in_progress(
  job_id uuid,
  new_status text,
  worker_id text,
  job_outputs jsonb default null,
  job_script text default null
)
returns void as $$
begin
  update jobs
  set status = new_status,
      outputs = job_outputs,
      script = coalesce(job_script, script)  -- Only update script if new value provided
  where id = job_id
    and status = 'in_progress'
    and worker_id = worker_id;
end;
$$ language plpgsql;

-- Grant necessary permissions to authenticated users
grant execute on function acquire_job(uuid, text) to authenticated;
grant execute on function update_job_status_if_in_progress(uuid, text, text, jsonb, text) to authenticated;
