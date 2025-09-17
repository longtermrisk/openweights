-- 1) Drop the old function definitions so we can recreate them from scratch.
DROP FUNCTION IF EXISTS acquire_job(uuid, text) CASCADE;
DROP FUNCTION IF EXISTS update_job_status_if_in_progress(uuid, text, text, jsonb, text) CASCADE;

-- 2) Now create (not just replace) the functions with the revised parameter names.

CREATE OR REPLACE FUNCTION acquire_job(
    _job_id uuid,
    _worker_id text
)
RETURNS SETOF jobs
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  UPDATE jobs
     SET status = 'in_progress',
         worker_id = _worker_id
   WHERE id = _job_id
     AND status = 'pending'
  RETURNING *;
END;
$$;


CREATE OR REPLACE FUNCTION update_job_status_if_in_progress(
    _job_id uuid,
    _new_status text,
    _worker_id text,
    _job_outputs jsonb DEFAULT null,
    _job_script text DEFAULT null
)
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
  UPDATE jobs
     SET status = _new_status,
         outputs = _job_outputs,
         script = COALESCE(_job_script, script)
   WHERE id = _job_id
     AND status = 'in_progress'
     AND worker_id = _worker_id;
END;
$$;


-- 3) Grant permissions so your service/anonymous/authenticated roles can call them
GRANT EXECUTE ON FUNCTION acquire_job(uuid, text) TO authenticated;
GRANT EXECUTE ON FUNCTION update_job_status_if_in_progress(uuid, text, text, jsonb, text) TO authenticated;
