# Open work

## Credential safety

Done in 0.16.0: the dashboard does not reveal stored secrets, and workers do not hold the
RunPod API key (`docs/release-0.16.0.md`). Done in 0.17.0: secrets (`HF_TOKEN`,
`OPENWEIGHTS_API_KEY`, `RUNPOD_API_KEY`, `LITELLM_API_KEY`) are kept in a secret store,
apart from environment variables (`docs/release-0.17.0.md`). Done in 0.18.0: a worker holds an
OpenWeights token of its own, not the organization's key (`docs/release-0.18.0.md`). Done in 0.19.0: a
broker keeps the Hugging Face token off the pod (`docs/release-0.19.0.md`). Done in 0.20.0: the same
for the organization's LiteLLM key (`docs/release-0.20.0.md`). With these, a worker of an organization
that keeps its secrets in the secret store holds no organization credential. Done in 0.21.0: that holds
for every image and when no worker token can be created; jobs naming official images older than v0.16.0
are refused (`docs/release-0.21.0.md`).

Open items:

- **Self-managed clusters still pass their credentials to workers.** They have no
  secret store and no broker; their workers get a token of their own (0.18.0) but the
  real Hugging Face token. They would need to run a broker next to their manager.
- **The broker resolves an upstream name twice** (once to check it, once to connect).
  Pin the checked address before a broker gets access to a private network.
- **No WebSocket through the broker** (LiteLLM's realtime API).

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
