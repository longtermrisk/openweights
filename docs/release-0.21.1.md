# OpenWeights 0.21.1

The cluster manager no longer terminates healthy workers after a slow cycle.

## What was wrong

Each cycle the manager reads the organization's workers once, then starts pods for
pending jobs, then terminates workers that "stopped pinging". The last step compared
the current time with the pings in the list from the start of the cycle. Starting pods
is slow when RunPod refuses (about ten seconds per attempt, several attempts per group
of jobs), so with many pending jobs and an exhausted GPU quota a cycle took more than
the two minutes a worker may go without a ping. Every active worker then looked
unresponsive and was terminated, its run set to `failed` and its job back to `pending`.

On 2026-11-25 this ended all workers of one organization at once (more than 40 each
time, at 10:32, 11:43 and 12:40 UTC). At 10:32 each had "no ping for 129-134 seconds":
the age of the list plus the workers' five-second ping interval. The runs show as failed
without a log, with the metrics they had logged until then.

## What changes

`clean_up_workers` reads a worker's row again before it terminates the worker, and
decides on that (`OrganizationManager.get_worker`). One more query per worker that
looks overdue, none for the others. A worker that was terminated in the meantime is
left alone.

No migration, no new images; `IMAGE_VERSION` stays v0.16.0. Only the cluster manager
needs the upgrade.

## Verification

- `tests/test_credential_safety.py`: a worker that is overdue in the list but has
  pinged since is not terminated; one that is still overdue is; one already
  terminated is skipped.
- Not verified before the release: a real manager under the same load.
