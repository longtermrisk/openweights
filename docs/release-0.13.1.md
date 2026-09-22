# OpenWeights 0.13.1

Includes the cost tracking, API-key budgets, SDK and Costs dashboard introduced in
0.13.0, plus a fix verified during the production GPU smoke test:

- Reconcile workers whose RunPod pod has already been deleted. RunPod can reject a
  repeated termination request with `Unauthorized`; a successful lookup confirming
  the pod is absent now finalizes its worker record and stops cost accrual.
- If a pod still exists, or its lookup fails, retain the open accounting interval
  and retry cleanup. Temporary provider failures must not hide continuing costs.

Install `openweights==0.13.1` and use the `v0.13.1` cluster/worker images. New
installations still require `20260922000000_cost_tracking.sql`; there is no additional
schema migration after 0.13.0. See [cost tracking](cost-tracking.md) for budget
semantics and rollout instructions.
