# OpenWeights 0.13.4

The dashboard jobs page now loads small pages of summaries instead of downloading
up to 1,000 complete jobs. Search, status filters, and totals run in PostgreSQL,
so older jobs remain accessible. Slow database and worker-log calls run outside
the API event loop; log downloads have timeouts, responses support gzip, and
refreshes wait for all outstanding column requests, including failed batches.

Apply `supabase/migrations/20260925000000_dashboard_jobs_pagination.sql` before
upgrading the dashboard. It adds organization-scoped pagination/search and two
indexes, retaining existing RLS policies. See
[dashboard performance](dashboard-performance.md) for rollout and measurements.

The SDK and all managed image defaults advance to v0.13.4. Cluster, Unsloth, and
vLLM images include this release; GPU framework dependencies are unchanged from
the verified v0.13.1 images. Existing workers can finish their jobs normally.
