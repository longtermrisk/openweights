-- Keep dashboard reads bounded without downloading job scripts or JSON payloads.
CREATE INDEX IF NOT EXISTS jobs_org_created_id_idx
    ON public.jobs (organization_id, created_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS jobs_org_status_created_id_idx
    ON public.jobs (organization_id, status, created_at DESC, id DESC);

CREATE OR REPLACE FUNCTION public.get_dashboard_jobs(
    org_id uuid,
    statuses public.job_status[] DEFAULT NULL,
    search_text text DEFAULT '',
    page_limit integer DEFAULT 10,
    page_offset integer DEFAULT 0
) RETURNS jsonb
LANGUAGE sql STABLE SECURITY INVOKER
SET search_path = public
AS $$
    WITH filtered AS NOT MATERIALIZED (
        SELECT id, type, status, model, docker_image, created_at
        FROM public.jobs
        WHERE organization_id = org_id
          AND (statuses IS NULL OR status = ANY(statuses))
          AND (coalesce(search_text, '') = '' OR
               strpos(lower(concat_ws(' ', id, model, docker_image,
                                      params::text, outputs::text)), lower(search_text)) > 0)
    )
    SELECT jsonb_build_object(
        'items', coalesce((
            SELECT jsonb_agg(to_jsonb(page) ORDER BY created_at DESC, id DESC)
            FROM (
                SELECT * FROM filtered
                ORDER BY created_at DESC, id DESC
                LIMIT greatest(1, least(coalesce(page_limit, 10), 100))
                OFFSET greatest(0, coalesce(page_offset, 0))
            ) page
        ), '[]'::jsonb),
        'total', (SELECT count(*) FROM filtered)
    );
$$;

-- Invoker security preserves the existing organization RLS policies.
REVOKE ALL ON FUNCTION public.get_dashboard_jobs(uuid, public.job_status[], text, integer, integer) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.get_dashboard_jobs(uuid, public.job_status[], text, integer, integer) TO authenticated;
