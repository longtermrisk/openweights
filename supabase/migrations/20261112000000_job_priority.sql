-- Job priority: pending jobs with a higher priority are started first.
--
-- The priority lives in its own table instead of a jobs column because every
-- released SDK <= 0.13.7 builds its Job dataclass from all columns of a jobs row
-- and crashes on a column it does not know (see 20261012000000_job_attribution_table).
-- A job without a row has priority 0.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

CREATE TABLE public.job_priority (
    job_id text PRIMARY KEY REFERENCES public.jobs(id) ON DELETE CASCADE,
    priority integer NOT NULL DEFAULT 0
);

ALTER TABLE public.job_priority ENABLE ROW LEVEL SECURITY;

-- Same visibility as the job itself: whoever can use the job's organization.
CREATE POLICY "Organization members can manage job priority"
    ON public.job_priority
    USING (EXISTS (
        SELECT 1 FROM public.jobs j
        WHERE j.id = job_priority.job_id
          AND public.is_organization_member(j.organization_id)
    ))
    WITH CHECK (EXISTS (
        SELECT 1 FROM public.jobs j
        WHERE j.id = job_priority.job_id
          AND public.is_organization_member(j.organization_id)
    ));

REVOKE ALL ON public.job_priority FROM anon;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.job_priority TO authenticated, service_role;

-- Make PostgREST pick up the new table and its relationship to jobs right away.
NOTIFY pgrst, 'reload schema';

COMMIT;
