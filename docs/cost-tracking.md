# Cost tracking and spending limits

OpenWeights 0.13 adds organization-scoped compute estimates in USD. Open **Costs**
in the dashboard, or use:

```python
from openweights import OpenWeights
ow = OpenWeights()
report = ow.costs.report()
print(report['total_usd'])
for job in report['jobs']:
    print(job['job_id'], job['direct_usd'], job['overhead_usd'], job['total_usd'])
```

Reports include jobs, workers, API keys, submitting users, configured limits, and
an `as_of` timestamp. Dashboard totals refresh every 30 seconds. Reports cover the
entire recorded lifetime; there is no monthly reset or date-window filter. Job and
worker arrays are paginated (100 rows by default); use `ow.costs.report(limit=100,
offset=100)` for the next page. Totals and key/user summaries always cover all rows.

## Accounting rules

- The manager snapshots the pod's `costPerHr` from RunPod at provisioning. If
  unavailable, it records the existing hardware estimate multiplied by GPU count.
  Unknown hardware remains **unpriced**, not free. The source is visible per worker.
  Provider pod rates already cover the whole pod and are not multiplied again.
- Worker cost is elapsed time from the successful provisioning attempt's start to
  confirmed pod termination, multiplied by the saved hourly rate. This includes
  startup, downloads, execution, uploads and idle time. Shutdown requests do not
  stop the clock. A failed termination is retried by the manager.
- Direct job cost sums execution intervals across **all runs**, including failures,
  cancellations and retries. Run completion timestamps freeze when status first
  leaves `in_progress`; later log uploads do not extend execution time.
- Worker overhead is its total cost minus direct execution cost. It is retained
  separately and divided **equally among distinct jobs on that worker**, not among
  runs and not in proportion to runtime. A retry does not buy another overhead share.
- Default job/key/user totals include this allocated overhead. Toggle it off in
  the dashboard to inspect execution only. A worker that never executes a job has
  unallocated overhead, visible separately in the organization total.
- Example: a worker costs $8, job A executes for $3 (over two attempts), and job B
  for $1. The $4 overhead adds $2 to each job: A totals $5 and B totals $3.
- Allocations change until worker termination and can decrease when another job
  shares the same worker. Organization totals count each worker once.
- Jobs retain the original submitting key and the key's creator as the submitting
  user. Signed-in user submissions use their authenticated user ID. Deduplicated
  jobs and retries keep the original attribution, even if another key restarts
  them. Legacy organization JWTs and old jobs remain unattributed.
- Separate accounting records survive deletion of jobs, runs, API keys and users.
  Deleted job IDs with accounting history cannot be reused. Run identity is fixed;
  retry by creating another run, not reopening or reassigning an old run.

These are operational estimates, not invoice reconciliation. They do not separately
meter network volumes, external APIs, storage, taxes or credits. Hardware fallback
prices are indicative. Pre-upgrade workers have unknown prices; historical usage is
not retroactively priced. Unknown runs and workers produce an incomplete-total
warning. Non-managed/local runs have no inferred hardware price. Historical closed
runs use the best available `updated_at` timestamp during migration.

## API key spending limits

Starting with 0.13.3, set an initial budget when creating a token:

```sh
ow token create --name experiment --spending-limit-usd 100
```

This works with the existing `OPENWEIGHTS_API_KEY` credentials authorized to create
tokens. The dashboard's token creation dialog also accepts a lifetime USD limit.
Omit the option (or leave the field blank) for unlimited spending; `0` blocks work.
The token and budget are created atomically: failure leaves neither behind.
The REST token creation endpoint accepts `spending_limit_usd` alongside `name`
and `expires_in_days`. Apply migration `20260922180000_token_creation_limit.sql`
before upgrading the dashboard or using the new CLI option. Worker images remain
v0.13.1.

Choosing a new token's initial budget uses existing token-creation permissions.
Changing or removing an existing budget still requires a signed-in admin user.

A signed-in organization **admin user** can choose an API key in the Costs page and
save a lifetime USD limit. `0` blocks work; blank removes the limit. API-key logins
can view costs but cannot edit budgets, even though older organization APIs treat
API keys as administrators.

For automation with an administrator's Supabase user JWT:

```python
admin = OpenWeights(auth_token=admin_user_jwt, organization_id=organization_id)
admin.costs.set_limit(api_token_id, 25)
admin.costs.set_limit(api_token_id, None)  # remove limit
```

The REST equivalents are `GET /organizations/{org_id}/costs` and
`PUT /organizations/{org_id}/costs/limits/{token_id}` with
`{"amount_usd": 25}` (or `null` to remove). Amounts must be finite and nonnegative.

Budgets include direct execution and allocated overhead. Database triggers reject
new submissions, restarts, acquisition and run creation after the threshold is
reached. The manager checks every approximately 15 seconds and cancels pending and
running jobs for exhausted keys. Workers follow their existing cancellation path
(5-second polling plus up to about 60 seconds of log-flush delay). Idle termination
can add further overhead. Budgeted jobs cannot start on unpriced workers.

**These are accrued-spend limits, not strict prepaid caps.** Parallel jobs may start
before a threshold is reached, startup cost may only be allocated when a worker
first runs a job, and shutdown delays, manager outages and later overhead can cause
overspend. Limits do not reserve the estimated maximum price of pending jobs.
For example, if 60 out of 100 jobs have completed when a key exhausts its budget,
the 60 completed jobs remain completed and the remaining queued/running jobs are
marked `canceled`. Jobs attributed to other keys are unaffected. Previously canceled
jobs require an explicit restart after increasing a limit.
Limits attach to individual keys, not to all credentials held by a person; this
feature does not turn the existing organization-admin API keys into untrusted,
restricted credentials. Use independently managed keys and trusted organization
members. Unallocated overhead cannot be assigned to a key or user.

## Rollout

1. Back up the database and apply
   `supabase/migrations/20260922000000_cost_tracking.sql` through your existing
   Supabase migration process (or SQL editor). It is transactional and must run
   exactly once. It adds columns, accounting tables, protected RPCs and triggers;
   it does not reprice old usage. Do not apply the original schema to production.
   With Supabase CLI management login, an explicit project can be migrated using
   `supabase db query --linked --project-ref YOUR_PROJECT_REF --file supabase/migrations/20260922000000_cost_tracking.sql`.
   This runs SQL directly; record the migration in your deployment history if your
   environment also uses `supabase db push`.
2. Upgrade the manager/dashboard and managed worker images to v0.13.1, and upgrade
   SDK clients with `pip install --upgrade openweights==0.13.1`. Run the migration
   **before** the new manager starts: it deliberately fails closed if the budget
   enforcement RPC is unavailable.
3. Allow existing workers to drain, then terminate them and provision new workers
   to start recording rates. Existing workers are marked unpriced. Older workers
   still benefit from database triggers; the updated manager confirms termination.
4. Confirm a test job appears under Costs with its submitting key/user, inspect
   the worker rate source, and exercise a zero-dollar limit with a test key.

Keep the additive schema when rolling application code back; preserve the accounting
tables. Older managers do not perform periodic cancellation or price snapshots, so
budget enforcement is degraded on rollback. No production database is modified by
package installation or the test suite.

## Validation

```sh
docker run -d --rm --name ow-cost-tests -e POSTGRES_PASSWORD=ow-test-only postgres:16
OW_COST_TEST_CONTAINER=ow-cost-tests .venv/bin/pytest tests/test_cost_tracking.py -q
docker stop ow-cost-tests
cd openweights/dashboard/frontend && npm ci && npm run build
```

Each database test creates and drops a private database in the disposable container.
It uses the repository's actual core schema/RLS and migration with a local `auth.uid`
stub; no configured Supabase credentials are used. Tests cover equal allocation,
retries, multiple workers, unallocated overhead, unknown prices, history retention,
fixed timestamps, tenant isolation, grants, admin permissions and budget enforcement.
