# OpenWeights 0.13.7

Dashboard: working organization switcher and a new Cluster tab.

- **Organization switcher.** Picking another organization now navigates to the
  same section (Jobs, Workers, Cluster, Costs, Settings) of the new
  organization. Before, it only changed in-memory state, and the route, which
  takes the organization from the URL, immediately switched back.
- **Cluster tab** (`/<org>/cluster`). Shows the organization's cluster manager
  log: provisioning decisions, hardware cooldowns and errors, i.e. why a pending
  job has no worker yet. "Collapse repeats" folds the per-loop status lines into
  their latest occurrence with a count. The view refreshes every 15 s.
  Served by `GET /organizations/{id}/cluster/logs?lines=N` (org members only;
  token-like strings are redacted).
- `OW_ORG_MANAGER_LOG_DIR` sets where `ow cluster --super` writes the per-org
  logs (default: `./logs`, as before) and where `ow serve` reads them. Set it
  for both when they run from different directories.

SDK and worker images are unchanged; `IMAGE_VERSION` stays v0.13.4.
