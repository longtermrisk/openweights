# OpenWeights 0.18.0

A token per worker. Second of four steps towards a worker environment without any
organization credential (`TODO.md`).

## What changes

Every worker pod used to hold the organization's own API key as `OPENWEIGHTS_API_KEY`:
an admin key that never expires, can read the organization's settings and create more
keys, and that the code of any job can read from its environment and keep.

Now the cluster manager creates a token for each worker it starts and passes that
instead, under the same variable name:

- **Bound to one worker** (`api_tokens.worker_id`), created with `create_worker_token`.
- **Short-lived.** It expires 25 hours after the pod is started (the pod's own TTL is
  24), and the database revokes it when the worker is marked `terminated` or its row
  is deleted, whoever does that.
- **Not an organization admin.** It cannot read `organization_secrets` (the
  environment variables) or see which secrets exist, cannot create or revoke API
  keys, and cannot change spending limits, members or the organization.
- **A member otherwise.** It can do what a worker does: read and update jobs, runs,
  events, files and worker rows of its organization.

Worker images need no change: any image reads `OPENWEIGHTS_API_KEY`. Worker tokens are
left out of the key lists of the dashboard and of `ow token ls`.

Pods started with `ow ssh` are unchanged; they run with the caller's own key.

## What this does not close

- Within its lifetime a worker token can still read and change other jobs of its
  organization and submit jobs, as the organization key it replaces could. A job
  submitted by a key with a spending limit therefore still runs on a pod whose token
  has no limit.
- A JWT obtained with a token stays valid for up to an hour after the token is
  revoked. This is the same for every API key.
- `HF_TOKEN` is still in the worker's environment. That is the next step (the broker).

## Migration

`supabase/migrations/20261122000000_worker_tokens.sql` (apply with `supabase db push`):
adds `api_tokens.worker_id`, `create_worker_token`, two triggers on `worker` that revoke
a worker's tokens when it ends, and makes `is_organization_admin` false for worker
tokens.

Safe in any order relative to the code:

- New manager on an old database: `create_worker_token` does not exist, the manager
  logs a warning and passes the organization's key as before.
- Old manager on a new database: nothing changes; existing keys have no `worker_id`.

## Release order

1. Apply the migration.
2. Publish 0.18.0 and upgrade the cluster manager. Workers started from then on get
   their own token; workers already running keep the organization's key until they end
   (at most their TTL).

No worker image changes; `IMAGE_VERSION` stays v0.16.0. The dashboard only needs the
upgrade to hide worker tokens from its key list.

## Verification

- `tests/test_worker_tokens.py`: against a real PostgreSQL, what a worker token can and
  cannot do, that it is revoked with its worker, who may create one and for which
  workers; and the manager's side with a fake database.
- Not verified before the release: a real worker pod running a job with its own token.
  Smoke test after the release: run one small job and check that its worker's token
  appears in `api_tokens` with the worker's id, that the job completes, and that the
  token is revoked once the worker is terminated.
