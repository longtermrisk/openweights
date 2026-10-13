-- Move cost attribution (submitted_by, api_token_id) off public.jobs.
--
-- 20260922000000_cost_tracking added both columns to jobs. Every released SDK
-- (<= 0.13.7) builds its Job dataclass from all columns of a jobs row, so the
-- extra columns made ow.jobs.create / retrieve / list raise
-- "Job.__init__() got an unexpected keyword argument 'submitted_by'" for every
-- client that has not upgraded. The columns were only read by the cost triggers,
-- so they move to an internal table and jobs returns to its pre-0.13 shape.
--
-- job_attribution has no FK to jobs: it is written by the BEFORE INSERT trigger,
-- before the job row exists. A job ID re-created after deletion overwrites its row.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

CREATE TABLE public.job_attribution (
    job_id text PRIMARY KEY,
    organization_id uuid NOT NULL,
    api_token_id uuid,
    submitted_by uuid
);
ALTER TABLE public.job_attribution ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.job_attribution FROM anon, authenticated;
-- Only read by the SECURITY DEFINER cost functions below.

INSERT INTO public.job_attribution(job_id, organization_id, api_token_id, submitted_by)
SELECT id, organization_id, api_token_id, submitted_by FROM public.jobs;

CREATE OR REPLACE FUNCTION public.guard_job_spending() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE key_id uuid; owner_id uuid;
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF EXISTS (SELECT 1 FROM cost_runs WHERE job_id = NEW.id) THEN
            RAISE EXCEPTION 'Deleted job IDs with cost history cannot be reused';
        END IF;
        key_id := (nullif(current_setting('request.jwt.claims', true), '')::jsonb ->> 'api_token_id')::uuid;
        IF key_id IS NOT NULL THEN
            SELECT created_by INTO owner_id FROM api_tokens
            WHERE id = key_id AND organization_id = NEW.organization_id
              AND revoked_at IS NULL AND (expires_at IS NULL OR expires_at > now());
            IF NOT FOUND THEN RAISE EXCEPTION 'Invalid API key for job organization'; END IF;
        ELSE owner_id := auth.uid(); END IF;
        INSERT INTO job_attribution(job_id, organization_id, api_token_id, submitted_by)
        VALUES (NEW.id, NEW.organization_id, key_id, owner_id)
        ON CONFLICT (job_id) DO UPDATE SET organization_id = EXCLUDED.organization_id,
            api_token_id = EXCLUDED.api_token_id, submitted_by = EXCLUDED.submitted_by;
    ELSE
        SELECT api_token_id INTO key_id FROM job_attribution WHERE job_id = NEW.id;
        IF NEW.organization_id <> OLD.organization_id THEN
            RAISE EXCEPTION 'Job organization is immutable';
        END IF;
    END IF;
    IF NEW.status IN ('pending', 'in_progress') AND
       (TG_OP = 'INSERT' OR NEW.status IS DISTINCT FROM OLD.status) THEN
        -- Serialize admissions and limit updates for this key.
        PERFORM 1 FROM spending_limits WHERE api_token_id = key_id FOR UPDATE;
        IF key_budget_exhausted(NEW.organization_id, key_id) THEN
            RAISE EXCEPTION 'API key spending limit reached' USING ERRCODE = 'P0001';
        END IF;
        IF NEW.status = 'in_progress' AND NEW.worker_id IS NOT NULL AND EXISTS (
            SELECT 1 FROM spending_limits WHERE api_token_id = key_id
        ) AND NOT EXISTS (
            SELECT 1 FROM cost_workers WHERE worker_id = NEW.worker_id
                AND organization_id = NEW.organization_id AND hourly_cost_usd IS NOT NULL
        ) THEN RAISE EXCEPTION 'Cannot run budgeted jobs on unpriced workers'; END IF;
    END IF;
    RETURN NEW;
END $$;

CREATE OR REPLACE FUNCTION public.record_run_cost() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE j public.jobs; a public.job_attribution;
BEGIN
    IF TG_OP = 'INSERT' THEN
        SELECT * INTO STRICT j FROM jobs WHERE id = NEW.job_id;
        SELECT * INTO a FROM job_attribution WHERE job_id = j.id;
        IF NEW.worker_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM worker WHERE id = NEW.worker_id AND organization_id = j.organization_id
        ) THEN RAISE EXCEPTION 'Worker and job organizations must match'; END IF;
        IF key_budget_exhausted(j.organization_id, a.api_token_id) THEN
            RAISE EXCEPTION 'API key spending limit reached';
        END IF;
        IF NEW.worker_id IS NOT NULL THEN
            PERFORM pg_advisory_xact_lock(hashtextextended(NEW.worker_id, 0));
            IF EXISTS (SELECT 1 FROM cost_runs WHERE worker_id = NEW.worker_id AND ended_at IS NULL) THEN
                RAISE EXCEPTION 'Worker already has an active run';
            END IF;
            IF EXISTS (SELECT 1 FROM spending_limits WHERE api_token_id = a.api_token_id)
               AND NOT EXISTS (SELECT 1 FROM cost_workers WHERE worker_id = NEW.worker_id
                               AND hourly_cost_usd IS NOT NULL) THEN
                RAISE EXCEPTION 'Cannot run budgeted jobs on unpriced workers';
            END IF;
        END IF;
        INSERT INTO cost_runs(run_id, organization_id, job_id, worker_id, api_token_id,
                              user_id, started_at, ended_at)
        VALUES (NEW.id, j.organization_id, j.id, NEW.worker_id, a.api_token_id,
                a.submitted_by, now(), CASE WHEN NEW.status <> 'in_progress' THEN now() END);
    ELSE
        IF NEW.job_id IS DISTINCT FROM OLD.job_id OR (NEW.worker_id IS NOT NULL AND NEW.worker_id IS DISTINCT FROM OLD.worker_id) THEN
            RAISE EXCEPTION 'Run accounting identity is immutable; create a new run';
        END IF;
        IF NEW.status <> 'in_progress' THEN
            UPDATE cost_runs SET ended_at = coalesce(ended_at, now()) WHERE run_id = NEW.id;
        ELSIF OLD.status <> 'in_progress' THEN
            RAISE EXCEPTION 'A finished run cannot be reopened; create a new run';
        END IF;
    END IF;
    RETURN NEW;
END $$;

CREATE OR REPLACE FUNCTION public.enforce_spending_limits(org_id uuid) RETURNS integer
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE affected integer;
BEGIN
    IF NOT is_organization_member(org_id) THEN RAISE EXCEPTION 'Organization access denied' USING ERRCODE = '42501'; END IF;
    UPDATE jobs SET status = 'canceled' FROM job_attribution a
    WHERE a.job_id = jobs.id AND jobs.organization_id = org_id
        AND jobs.status IN ('pending', 'in_progress')
        AND key_budget_exhausted(org_id, a.api_token_id);
    GET DIAGNOSTICS affected = ROW_COUNT;
    RETURN affected;
END $$;

ALTER TABLE public.jobs DROP COLUMN submitted_by, DROP COLUMN api_token_id;
COMMIT;
