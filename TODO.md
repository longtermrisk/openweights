# Open work

## Credential safety

Done in 0.16.0: the dashboard does not reveal stored secrets, and workers do not hold the
RunPod API key (`docs/release-0.16.0.md`). Done in 0.17.0: secrets (`HF_TOKEN`,
`OPENWEIGHTS_API_KEY`, `RUNPOD_API_KEY`, `LITELLM_API_KEY`) are kept in a secret store,
apart from environment variables (`docs/release-0.17.0.md`).

The plan, in the order to do it. The aim: no organization credential in a worker's
environment; a worker holds one token of its own, which dies with the worker.

1. **A token per worker.** The manager creates an OpenWeights token bound to the worker
   when it starts the pod, expiring with the worker's TTL and revoked when the pod is
   terminated. Row level security limits it to what a worker does (its worker row, jobs
   and runs of its organization, events, files); it cannot read settings or create
   tokens. It replaces the organization's `OPENWEIGHTS_API_KEY` on the pod.
2. **A broker for Hugging Face** (`ow broker`, a small public HTTP service that can read
   the secret store). The worker gets `HF_ENDPOINT` pointing at the broker and its own
   worker token as `HF_TOKEN`; the broker checks the token, puts in the organization's
   real token and forwards API calls. File bytes do not pass through it: the Hub answers
   with signed URLs. Worker images older than 0.15.2, which do not honour `HF_ENDPOINT`
   everywhere, are not given the real token: their Hugging Face calls fail.
3. **LiteLLM through the same broker**: `LITELLM_BASE_URL` pointing at it, the worker
   token as the key.

Other open items:

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
