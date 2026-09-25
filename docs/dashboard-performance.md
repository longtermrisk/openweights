# Dashboard jobs performance

The jobs page now requests one small page per visible column (or one page in list
view). Filtering and totals run in PostgreSQL, including searches in parameters
and outputs, so jobs beyond Supabase's default 1,000-row response cap are visible.
Scripts and JSON payloads are downloaded only when opening job details.

Synchronous API handlers run database calls and log downloads in FastAPI's thread
pool. Slow log requests no longer block the server event loop. Worker log requests
have connection/read timeouts, large responses use gzip, and browser refreshes
wait until the preceding fetch completes. Navigation cancels obsolete fetches.

## Rollout

Apply `supabase/migrations/20260925000000_dashboard_jobs_pagination.sql` through
the normal database migration process **before** deploying the updated dashboard.
It adds two job indexes and an RLS-enforced `get_dashboard_jobs` RPC; it does not
change existing policies. Index creation can briefly block job writes, so use an
appropriate migration window for large installations.

The existing jobs list endpoint remains available. The new frontend requires the
new `/organizations/{organization_id}/jobs/page` endpoint and database RPC.
Rebuild packaged frontend assets with:

```sh
cd openweights/dashboard/frontend
npm run build
```

## Verification

```sh
# API regression tests; SQL tests use the existing disposable-database fixture.
OW_COST_TEST_CONTAINER=ow-cost-tests .venv/bin/python -m pytest \
  tests/test_dashboard_jobs.py tests/test_cost_api.py -q

# With Playwright and Chromium installed; OW_SERVER_PYTHON may select the backend venv.
python tests/dashboard_browser_smoke.py
python tests/dashboard_jobs_browser.py
```

Read-only measurements on the configured database during investigation:

| Query | Rows | JSON size | Time |
| --- | ---: | ---: | ---: |
| Previous full-row query | 1,000 | 3,013,259 bytes | 1.37 s |
| 30-row summary query | 30 | 6,500 bytes | 0.32 s |

These are individual query timings, not end-to-end production page benchmarks.
The regression suite checks pagination beyond 1,000 jobs, stable ordering, search,
empty pages, organization isolation, and concurrent jobs reads during slow logs.
