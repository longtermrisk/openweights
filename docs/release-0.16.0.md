# OpenWeights 0.16.0

Credential safety, first two steps: the dashboard no longer reveals stored secrets, and
workers no longer receive the RunPod API key.

## Dashboard: secrets are write-only

- `GET /organizations/{id}/secrets` returns, per secret, its name, its last four
  characters (`last_four`, only for values of 12 or more characters) and `updated_at`.
  Settings that are not credentials (`HF_USER`, `HF_ORG`, `OW_MANAGED`, `MAX_WORKERS`) come
  back in full as `plain_value`. The value of a secret never leaves the backend.
- `PUT /organizations/{id}/secrets/{name}` with `{"value": ...}` creates or replaces one
  secret; `DELETE /organizations/{id}/secrets/{name}` deletes one. The bulk
  `PUT /organizations/{id}/secrets`, which needed the browser to hold every value, is gone.
- The Secrets tab shows `••••••••abcd` with **Replace** and **Delete**; there is no "show"
  button any more. API tokens were already shown once at creation and stored as a hash.
- New row level security (migration below): a signed-in user can no longer select secret
  values from `organization_secrets` with their session token, which the browser could do
  directly, bypassing the dashboard backend.

What this does **not** close: organization API keys without a spending limit still read
the table (the cluster manager is such a client, and so are `ow env show` and
`ow manage`), and an admin can create such a key. Secrets are also still stored in
plaintext. Both are for the next steps (manager identity, secret store).

After the migration the backend can no longer read the stored `HF_TOKEN`, so replacing
`HF_USER` or `HF_ORG` alone is not checked against the token. Replacing `HF_TOKEN` is
still checked against the stored `HF_USER` and `HF_ORG`.

## Workers do not hold the RunPod key

The organization's RunPod key can terminate every pod of the account. Until now it was in
the environment of every worker pod, where job code can read it, only so that a worker
could terminate its own pod.

- A worker that shuts down now sets `status='shutdown'` and `termination_requested_at` on
  its worker row (`openweights/worker/termination.py`). The cluster manager terminates the
  pod on its next loop (every 15 s), reverts jobs that were still running and marks the
  worker `terminated`.
- The manager also terminates, as before, every worker that stops pinging (2 minutes; 10
  minutes while starting). That covers workers that exit without asking: a crashed or
  killed process, a pod that never came up, an old image whose own termination failed.
- When the TTL expires, the TTL monitor on the pod sets `status='shutdown'` and
  `termination_requested_at`: the worker stops (it gives its job back and takes no new
  one) and the manager terminates the pod.
- The manager passes `RUNPOD_API_KEY` only to official worker images older than v0.16.0
  (`worker_image_needs_runpod_key`), which keep terminating their own pod exactly as
  before. Pods started with `ow ssh` keep receiving the caller's own key, because no
  manager watches them.

### Behaviour changes to be aware of

- **Custom worker images get no RunPod key.** They are assumed to be built on a current
  official image. A custom image built on an image older than v0.16.0 still tries to
  terminate its own pod and fails; the manager then terminates the pod when the pings
  stop, about 2 minutes later. Rebuild such images on v0.16.0.
- **An expired TTL shuts the worker down but cannot end the pod without a manager.**
  Before, a pod terminated itself after `TTL_HOURS` even if the cluster manager was down.
  Now the worker stops working at that point, and the pod stays up (and billed) until a
  manager terminates it. The open work is listed in `TODO.md`.

## Migration

`supabase/migrations/20261120000000_credential_safety.sql` (apply with `supabase db push`):
adds `worker.termination_requested_at`, replaces the `organization_secrets` policy and
adds `list_organization_secrets_masked`.

It is safe in any order relative to the code:

- New code on an old database: the dashboard masks in the backend instead of the
  database; a worker that finds no `termination_requested_at` column sets only
  `status='shutdown'`, and the manager terminates the pod when the pings stop (about 2
  minutes later than with the column).
- Old code on a new database: API-key access is unchanged, so the manager, workers and
  SDKs behave as before. The 0.15 dashboard lists only the four non-secret settings.

## Release order

1. Build and push the three v0.16.0 images (below). Jobs submitted with the 0.16.0 SDK
   reference them, so they have to exist before the package is published.
2. Apply the migration.
3. Publish 0.16.0 to PyPI, then upgrade the cluster manager and redeploy the dashboard.
   A v0.16.0 worker under a 0.15 manager works (the old manager still passes the key and
   terminates the pod about 2 minutes after the worker stops pinging), but upgrade the
   manager before or together with the clients.

## Images

`IMAGE_VERSION` advances to v0.16.0 for the Unsloth, vLLM and cluster images, because the
worker process and the TTL monitor changed. They are overlays of the v0.15.0 images that
only refresh the OpenWeights source (`Dockerfile.release`); GPU framework dependencies are
unchanged, so no GPU re-validation is needed beyond a smoke job.

```sh
for name in unsloth vllm cluster; do
  docker pull --platform linux/amd64 nielsrolf/ow-$name:v0.15.0
  base=$(docker inspect --format '{{index .RepoDigests 0}}' nielsrolf/ow-$name:v0.15.0)
  docker build --platform linux/amd64 -f Dockerfile.release \
    --build-arg BASE_IMAGE=$base --build-arg RELEASE_VERSION=0.16.0 \
    -t nielsrolf/ow-$name:v0.16.0 . && docker push nielsrolf/ow-$name:v0.16.0
done
```

Smoke test after the release: run one small job, then check that its worker row has
`termination_requested_at` set and `status='terminated'`, that the pod is gone from RunPod
within about half a minute of the worker shutting down, and that `env` inside a job does
not contain the organization's RunPod key.

## Verification

- `tests/test_credential_safety.py`: masking, the secrets endpoints, row level security
  against a real PostgreSQL (signed-in admin, API key, budgeted key, other organization,
  anon), the worker's termination request with and without the new column, which images
  get the key, and the manager's decision and termination path.
- Not verified: the Secrets tab in a browser against a live backend, and a real worker
  pod (no GPU job was run for this change).
