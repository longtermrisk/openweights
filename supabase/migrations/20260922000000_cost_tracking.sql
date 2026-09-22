-- USD compute estimates. Ledger rows deliberately have no job/worker/token FKs:
-- deleting an operational record must never erase historical spend.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';
ALTER TABLE public.jobs ADD COLUMN submitted_by uuid, ADD COLUMN api_token_id uuid;
ALTER TABLE public.worker ADD COLUMN hourly_cost_usd numeric CHECK (hourly_cost_usd >= 0),
    ADD COLUMN cost_rate_source text, ADD COLUMN billing_started_at timestamptz;

CREATE TABLE public.cost_workers (
    worker_id text PRIMARY KEY, organization_id uuid NOT NULL,
    hardware_type text, hourly_cost_usd numeric CHECK (hourly_cost_usd >= 0),
    rate_source text, started_at timestamptz NOT NULL, ended_at timestamptz
);
CREATE TABLE public.cost_runs (
    run_id integer PRIMARY KEY, organization_id uuid NOT NULL, job_id text NOT NULL,
    worker_id text, api_token_id uuid, user_id uuid,
    started_at timestamptz NOT NULL, ended_at timestamptz
);
CREATE INDEX cost_workers_org ON public.cost_workers(organization_id);
CREATE INDEX cost_runs_worker ON public.cost_runs(worker_id);
CREATE INDEX cost_runs_org_token ON public.cost_runs(organization_id, api_token_id);
CREATE TABLE public.spending_limits (
    organization_id uuid NOT NULL, api_token_id uuid PRIMARY KEY,
    limit_usd numeric NOT NULL CHECK (limit_usd >= 0 AND limit_usd NOT IN ('NaN'::numeric, 'Infinity'::numeric)),
    updated_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE public.cost_workers ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.cost_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.spending_limits ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.cost_workers, public.cost_runs, public.spending_limits FROM anon, authenticated;
-- All access goes through scoped RPCs; API keys cannot raise their own limits.

CREATE FUNCTION public.record_worker_cost() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
BEGIN
    IF NEW.pod_id IS NOT NULL AND NEW.billing_started_at IS NOT NULL THEN
        INSERT INTO cost_workers(worker_id, organization_id, hardware_type, hourly_cost_usd,
                                 rate_source, started_at, ended_at)
        VALUES (NEW.id, NEW.organization_id, NEW.hardware_type, NEW.hourly_cost_usd,
                NEW.cost_rate_source, NEW.billing_started_at,
                CASE WHEN NEW.status = 'terminated' THEN now() END)
        ON CONFLICT (worker_id) DO UPDATE SET
            ended_at = coalesce(cost_workers.ended_at, EXCLUDED.ended_at);
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER record_worker_cost AFTER INSERT OR UPDATE ON public.worker
FOR EACH ROW EXECUTE FUNCTION public.record_worker_cost();

-- Keep unknown historical hardware visible without inventing historical prices.
INSERT INTO cost_workers(worker_id, organization_id, hardware_type, rate_source, started_at, ended_at)
SELECT id, organization_id, hardware_type, 'unknown', created_at,
       CASE WHEN status IN ('terminated', 'shutdown') THEN updated_at END
FROM worker;

CREATE FUNCTION public.record_run_cost() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE j public.jobs;
BEGIN
    IF TG_OP = 'INSERT' THEN
        SELECT * INTO STRICT j FROM jobs WHERE id = NEW.job_id;
        IF NEW.worker_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM worker WHERE id = NEW.worker_id AND organization_id = j.organization_id
        ) THEN RAISE EXCEPTION 'Worker and job organizations must match'; END IF;
        IF key_budget_exhausted(j.organization_id, j.api_token_id) THEN
            RAISE EXCEPTION 'API key spending limit reached';
        END IF;
        IF NEW.worker_id IS NOT NULL THEN
            PERFORM pg_advisory_xact_lock(hashtextextended(NEW.worker_id, 0));
            IF EXISTS (SELECT 1 FROM cost_runs WHERE worker_id = NEW.worker_id AND ended_at IS NULL) THEN
                RAISE EXCEPTION 'Worker already has an active run';
            END IF;
            IF EXISTS (SELECT 1 FROM spending_limits WHERE api_token_id = j.api_token_id)
               AND NOT EXISTS (SELECT 1 FROM cost_workers WHERE worker_id = NEW.worker_id
                               AND hourly_cost_usd IS NOT NULL) THEN
                RAISE EXCEPTION 'Cannot run budgeted jobs on unpriced workers';
            END IF;
        END IF;
        INSERT INTO cost_runs(run_id, organization_id, job_id, worker_id, api_token_id,
                              user_id, started_at, ended_at)
        VALUES (NEW.id, j.organization_id, j.id, NEW.worker_id, j.api_token_id,
                j.submitted_by, now(), CASE WHEN NEW.status <> 'in_progress' THEN now() END);
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
CREATE TRIGGER record_run_cost AFTER INSERT OR UPDATE ON public.runs
FOR EACH ROW EXECUTE FUNCTION public.record_run_cost();

-- Historical runs are retained as unattributed. No current price is invented for old workers.
INSERT INTO cost_runs(run_id, organization_id, job_id, worker_id, started_at, ended_at)
SELECT r.id, j.organization_id, r.job_id, r.worker_id, r.created_at,
       CASE WHEN r.status <> 'in_progress' THEN r.updated_at END
FROM runs r JOIN jobs j ON j.id = r.job_id;

-- Private views are only used by the authorized SECURITY DEFINER RPCs below.
CREATE VIEW public.cost_run_amounts AS
SELECT r.*, w.hourly_cost_usd,
    greatest(0, extract(epoch FROM
      (least(coalesce(r.ended_at, now()), coalesce(w.ended_at, now()))
       - greatest(r.started_at, w.started_at)))) / 3600 * w.hourly_cost_usd AS direct_usd
FROM cost_runs r LEFT JOIN cost_workers w ON w.worker_id = r.worker_id;
CREATE VIEW public.cost_worker_amounts AS
SELECT w.*, greatest(0, extract(epoch FROM (coalesce(w.ended_at, now()) - w.started_at)))
       / 3600 * w.hourly_cost_usd AS total_usd,
       coalesce(r.direct_usd, 0) AS direct_usd, coalesce(r.job_count, 0) AS job_count,
       greatest(0, greatest(0, extract(epoch FROM (coalesce(w.ended_at, now()) - w.started_at)))
       / 3600 * w.hourly_cost_usd - coalesce(r.direct_usd, 0)) AS overhead_usd
FROM cost_workers w LEFT JOIN (
    SELECT worker_id, sum(direct_usd) AS direct_usd, count(DISTINCT job_id) AS job_count
    FROM cost_run_amounts GROUP BY worker_id
) r ON r.worker_id = w.worker_id;
CREATE VIEW public.cost_job_amounts AS
SELECT r.organization_id, r.job_id, r.api_token_id, r.user_id,
    sum(r.direct_usd) AS direct_usd,
    sum(w.overhead_usd / nullif(w.job_count, 0)) AS overhead_usd,
    sum(r.unknown_runs) AS unknown_runs
FROM (
    SELECT organization_id, job_id, api_token_id, user_id, worker_id,
           sum(direct_usd) AS direct_usd,
           count(*) FILTER (WHERE hourly_cost_usd IS NULL) AS unknown_runs
    FROM cost_run_amounts GROUP BY organization_id, job_id, api_token_id, user_id, worker_id
) r LEFT JOIN cost_worker_amounts w ON w.worker_id = r.worker_id
GROUP BY r.organization_id, r.job_id, r.api_token_id, r.user_id;
REVOKE ALL ON public.cost_run_amounts, public.cost_worker_amounts, public.cost_job_amounts FROM anon, authenticated;

CREATE FUNCTION public.key_budget_exhausted(org_id uuid, key_id uuid) RETURNS boolean
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
    SELECT EXISTS (
        SELECT 1 FROM spending_limits l WHERE l.organization_id = org_id AND l.api_token_id = key_id
        AND l.limit_usd <= coalesce((SELECT sum(coalesce(direct_usd, 0) + coalesce(overhead_usd, 0))
            FROM cost_job_amounts WHERE organization_id = org_id AND api_token_id = key_id), 0)
    );
$$;
REVOKE ALL ON FUNCTION public.key_budget_exhausted(uuid, uuid) FROM PUBLIC, anon, authenticated;

CREATE FUNCTION public.guard_job_spending() RETURNS trigger
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
        NEW.api_token_id := key_id;
        NEW.submitted_by := owner_id;
    ELSE
        NEW.api_token_id := OLD.api_token_id;
        NEW.submitted_by := OLD.submitted_by;
        IF NEW.organization_id <> OLD.organization_id THEN
            RAISE EXCEPTION 'Job organization is immutable';
        END IF;
    END IF;
    IF NEW.status IN ('pending', 'in_progress') AND
       (TG_OP = 'INSERT' OR NEW.status IS DISTINCT FROM OLD.status) THEN
        -- Serialize admissions and limit updates for this key.
        PERFORM 1 FROM spending_limits WHERE api_token_id = NEW.api_token_id FOR UPDATE;
        IF key_budget_exhausted(NEW.organization_id, NEW.api_token_id) THEN
            RAISE EXCEPTION 'API key spending limit reached' USING ERRCODE = 'P0001';
        END IF;
        IF NEW.status = 'in_progress' AND NEW.worker_id IS NOT NULL AND EXISTS (
            SELECT 1 FROM spending_limits WHERE api_token_id = NEW.api_token_id
        ) AND NOT EXISTS (
            SELECT 1 FROM cost_workers WHERE worker_id = NEW.worker_id
                AND organization_id = NEW.organization_id AND hourly_cost_usd IS NOT NULL
        ) THEN RAISE EXCEPTION 'Cannot run budgeted jobs on unpriced workers'; END IF;
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER guard_job_spending BEFORE INSERT OR UPDATE ON public.jobs
FOR EACH ROW EXECUTE FUNCTION public.guard_job_spending();

CREATE FUNCTION public.set_spending_limit(org_id uuid, key_id uuid, amount_usd numeric)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
BEGIN
    -- Deliberately do not use is_organization_admin: it treats every API key as an admin.
    IF NOT EXISTS (SELECT 1 FROM organization_members WHERE organization_id = org_id
                   AND user_id = auth.uid() AND role = 'admin') THEN
        RAISE EXCEPTION 'Only a signed-in organization admin can change spending limits' USING ERRCODE = '42501';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM api_tokens WHERE id = key_id AND organization_id = org_id) THEN
        RAISE EXCEPTION 'API key not found in organization';
    END IF;
    IF amount_usd IS NULL THEN DELETE FROM spending_limits WHERE api_token_id = key_id;
    ELSE
        INSERT INTO spending_limits(organization_id, api_token_id, limit_usd)
        VALUES (org_id, key_id, amount_usd)
        ON CONFLICT (api_token_id) DO UPDATE SET limit_usd = EXCLUDED.limit_usd, updated_at = now();
    END IF;
END $$;

CREATE FUNCTION public.enforce_spending_limits(org_id uuid) RETURNS integer
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE affected integer;
BEGIN
    IF NOT is_organization_member(org_id) THEN RAISE EXCEPTION 'Organization access denied' USING ERRCODE = '42501'; END IF;
    UPDATE jobs SET status = 'canceled' WHERE organization_id = org_id
        AND status IN ('pending', 'in_progress') AND key_budget_exhausted(org_id, api_token_id);
    GET DIAGNOSTICS affected = ROW_COUNT;
    RETURN affected;
END $$;

CREATE FUNCTION public.get_cost_report(org_id uuid, row_limit integer DEFAULT 100, row_offset integer DEFAULT 0) RETURNS jsonb
LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path = public AS $$
DECLARE result jsonb;
BEGIN
    IF NOT is_organization_member(org_id) THEN RAISE EXCEPTION 'Organization access denied' USING ERRCODE = '42501'; END IF;
    IF row_limit < 1 OR row_limit > 1000 OR row_offset < 0 OR row_limit IS NULL OR row_offset IS NULL THEN
        RAISE EXCEPTION 'Invalid cost report pagination';
    END IF;
    WITH j AS (SELECT *, coalesce(direct_usd,0) + coalesce(overhead_usd,0) AS total_usd
               FROM cost_job_amounts WHERE organization_id = org_id),
         w AS (SELECT * FROM cost_worker_amounts WHERE organization_id = org_id)
    SELECT jsonb_build_object(
        'currency', 'USD', 'as_of', now(),
        'total_usd', coalesce((SELECT sum(total_usd) FROM w), 0),
        'direct_usd', coalesce((SELECT sum(direct_usd) FROM w), 0),
        'overhead_usd', coalesce((SELECT sum(overhead_usd) FROM w), 0),
        'unallocated_overhead_usd', coalesce((SELECT sum(overhead_usd) FROM w WHERE job_count = 0), 0),
        'unknown_runs', coalesce((SELECT sum(unknown_runs) FROM j), 0),
        'unpriced_workers', (SELECT count(*) FROM w WHERE hourly_cost_usd IS NULL),
        'job_count', (SELECT count(*) FROM j),
        'worker_count', (SELECT count(*) FROM w),
        'jobs', coalesce((SELECT jsonb_agg(to_jsonb(page)) FROM (SELECT * FROM j ORDER BY total_usd DESC, job_id LIMIT row_limit OFFSET row_offset) page), '[]'),
        'workers', coalesce((SELECT jsonb_agg(to_jsonb(page)) FROM (SELECT * FROM w ORDER BY started_at DESC, worker_id LIMIT row_limit OFFSET row_offset) page), '[]'),
        'api_keys', coalesce((SELECT jsonb_agg(to_jsonb(k)) FROM (
            SELECT j.api_token_id, t.name, sum(j.direct_usd) AS direct_usd,
                sum(j.overhead_usd) AS overhead_usd, sum(j.total_usd) AS total_usd
            FROM j LEFT JOIN api_tokens t ON t.id = j.api_token_id GROUP BY j.api_token_id, t.name
        ) k), '[]'),
        'users', coalesce((SELECT jsonb_agg(to_jsonb(u)) FROM (
            SELECT user_id, sum(direct_usd) AS direct_usd, sum(overhead_usd) AS overhead_usd,
                sum(total_usd) AS total_usd FROM j GROUP BY user_id
        ) u), '[]'),
        'limits', coalesce((SELECT jsonb_agg(to_jsonb(l)) FROM spending_limits l WHERE organization_id = org_id), '[]'),
        'can_manage_limits', EXISTS (SELECT 1 FROM organization_members WHERE organization_id = org_id
                                     AND user_id = auth.uid() AND role = 'admin')
    ) INTO result;
    RETURN result;
END $$;
REVOKE ALL ON FUNCTION public.record_worker_cost(), public.record_run_cost(), public.guard_job_spending() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.get_cost_report(uuid, integer, integer), public.enforce_spending_limits(uuid), public.set_spending_limit(uuid, uuid, numeric) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.get_cost_report(uuid, integer, integer), public.enforce_spending_limits(uuid), public.set_spending_limit(uuid, uuid, numeric) TO authenticated;
COMMIT;
