# OpenWeights 0.13.0

- Track estimated USD costs by job, worker, submitting API key and user.
- Separate startup/idle overhead, allocating it equally across a worker's distinct
  jobs in the default view, including failed and retried jobs.
- Configure lifetime per-key spending limits as an organization admin. Enforce
  admission in PostgreSQL and cancel exhausted keys' work in the manager loop.
- Add a Costs dashboard with grouping, execution-only view, unknown-price warnings,
  rate sources and limit management, plus `ow.costs` SDK and dashboard REST APIs.
- Retain cost history independently of operational record deletion. Record confirmed
  worker termination and retry failed termination without prematurely ending costs.

**Migration required before upgrading the manager:** apply
`supabase/migrations/20260922000000_cost_tracking.sql`, then roll out v0.13.0.
Pre-upgrade usage remains unpriced. Limits are based on accrued estimates and can
overshoot during concurrent execution, shutdown or outages; they are not invoice caps.
See [cost tracking](cost-tracking.md) for accounting rules and rollout instructions.
