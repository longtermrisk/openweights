# Open work

## Credential safety

Done in 0.16.0: the dashboard does not reveal stored secrets, and workers do not hold the
RunPod API key (`docs/release-0.16.0.md`). Done in 0.17.0: secrets (`HF_TOKEN`,
`OPENWEIGHTS_API_KEY`, `RUNPOD_API_KEY`, `LITELLM_API_KEY`) are kept in a secret store,
apart from environment variables (`docs/release-0.17.0.md`). Done in 0.18.0: a worker holds an
OpenWeights token of its own, not the organization's key (`docs/release-0.18.0.md`). Done in 0.19.0: a
broker keeps the Hugging Face token off the pod (`docs/release-0.19.0.md`).

The plan, in the order to do it. The aim: no organization credential in a worker's
environment; a worker holds one token of its own, which dies with the worker.

1. **LiteLLM through the broker**: `LITELLM_BASE_URL` pointing at it, the worker
   token as the key.

Other open items:

- **A worker token is still a full member of its organization** for as long as it lives:
  it can read and change other jobs and submit jobs, without a spending limit even when
  the job it runs was submitted by a limited key. Narrowing it to the worker's own row,
  the jobs it runs and their files needs new row level security on jobs, runs, events,
  files and storage.
- **A revoked token's JWT lives up to an hour longer.** Shorter JWTs for worker tokens,
  or a revocation check in the policies, would close the gap.

- **An organization admin can still read environment variables and, on a deployment
  without a secret store, secrets** with an API key (`ow env show`). With a store the
  secrets are out of reach of any API key; whatever people keep in environment variables
  is readable by jobs anyway.
- **The cluster manager authenticates as the organization's API key**, which is itself a
  secret the manager reads from the store. A manager identity of its own would remove
  that key.
- **Without a secret store, secrets are stored in plaintext** in `organization_secrets`.
- **A pod cannot end itself when no manager is running.** After its TTL a worker shuts
  down, but the pod stays up until a manager terminates it. Check whether RunPod gives
  each pod a key that is scoped to that pod (the TTL monitor already uses any
  `RUNPOD_API_KEY` it finds in its environment), or let the supervisor terminate pods of
  organizations whose manager is not running.
