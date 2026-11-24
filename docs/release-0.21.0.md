# OpenWeights 0.21.0

No worker gets the organization's RunPod key or its own API key, whatever image the job
names and whatever fails. Closes two leftovers of 0.16.0 and 0.18.0 (security audit
2026-11-22, finding H10).

## What was open

- **The RunPod key through an old image.** Since 0.16.0 the cluster manager kept passing
  `RUNPOD_API_KEY` to official worker images older than v0.16.0, which terminate their
  own pod with it. The image is chosen by whoever submits the job (`docker_image`), so
  any member of an organization, or code running in a job, could submit a job naming
  `nielsrolf/ow-unsloth:v0.15.0` and read the key from the pod's environment. The key
  can terminate, and start, every pod of the RunPod account.
- **The organization's API key when no worker token could be made.** Since 0.18.0 a
  worker gets a token of its own as `OPENWEIGHTS_API_KEY`. On a database without
  `create_worker_token()` the manager logged a warning and passed the organization's
  key instead: an admin key that never expires.

## What changes

- **`RUNPOD_API_KEY` and the organization's `OPENWEIGHTS_API_KEY` are never in a worker's
  environment** (`OrganizationManager.worker_env_for`, `NEVER_IN_WORKER_ENV`). The
  environment no longer depends on the image.
- **Jobs that name an official image older than v0.16.0 fail.** The manager sets such a
  pending job to `failed` with the reason in `outputs.error` and starts no pod for it,
  two-part tags such as `v0.11` included
  (`unsupported_worker_image`, `reject_unsupported_images`). Without the key these images
  cannot end their own pod, and their TTL monitor cannot either, so they are refused
  rather than run in a half-working state.
- **No worker token, no worker.** `create_worker_token` raises when the database has no
  such function or answers without a token, as it already did for every other database
  error. The manager then starts no pod, marks the worker row it had created
  `terminated`, logs the error and waits 60 seconds (`WORKER_TOKEN_RETRY_SECONDS`)
  before it tries to start workers again. The jobs stay `pending` and get a worker
  once a token can be created.

### Behaviour changes to be aware of

- **Clients older than 0.16.0 submit jobs that fail.** Their built-in jobs name the
  official images of their own version (`IMAGE_VERSION`, v0.15.0 and older). The job's
  error says what to do: upgrade the `openweights` package, or pass a `docker_image`
  tagged v0.16.0 or newer. The same goes for jobs that pin an old official image
  explicitly, and for such jobs that are pending when the manager is upgraded.
- **Custom images are started as before, without the key** (this is unchanged since
  0.16.0). One built on an official image older than v0.16.0 cannot terminate its own
  pod: when its worker shuts down, the manager terminates the pod once the pings have
  stopped for 2 minutes. Its TTL monitor cannot end the pod after `TTL_HOURS` either,
  so such a pod lives until its worker is idle or stops. Rebuild such images on
  v0.16.0 or newer.
- **The migration of 0.18.0 is now required.** A manager on a database without
  `supabase/migrations/20261122000000_worker_tokens.sql` starts no workers and says so
  in its log. This also holds for self-managed clusters (`ow cluster`).
- Pods started with `ow ssh` are unchanged; they run with the caller's own keys.

## Release order

No migration and no new images; `IMAGE_VERSION` stays v0.16.0.

1. Check that `20261122000000_worker_tokens.sql` is applied to the database.
2. Publish 0.21.0 and upgrade the cluster manager.
3. Workers that are running at that moment keep the environment they were started
   with. Terminate workers whose `docker_image` is an official image older than v0.16.0
   (by pod id, from their worker rows), or wait for them to end.
4. **Replace the organization's RunPod key** once no such worker is left: until this
   release any member could have read it. Replace the organization's OpenWeights key
   too if the manager ever ran against a database without worker tokens after 0.18.0.

## Verification

- `tests/test_credential_safety.py`: which images are refused and that their jobs fail
  with the reason; that a started pod's environment holds the worker's own token and
  neither organization key, for a custom image too; that no pod is started when the
  token cannot be created (missing function, refused, connection error), that the
  worker row is closed, the jobs are left pending and the next attempt waits.
  `tests/test_worker_tokens.py`: `create_worker_token` raises instead of returning
  nothing. All with a fake database and a fake RunPod client.
- Not verified: a real manager against RunPod and Supabase. Smoke test after the
  release: submit one job with `docker_image="nielsrolf/ow-unsloth:v0.15.0"` and check
  that it fails with the message and that no pod appears; run one small current job
  and check that `env` inside it holds no RunPod key and a worker token.
