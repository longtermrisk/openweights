# Open work

## Credential safety

Done in 0.16.0: the dashboard does not reveal stored secrets, and workers do not hold the
RunPod API key (`docs/release-0.16.0.md`). Still open, roughly in the order to do it:

- **An organization admin can still read the stored secrets.** The cluster manager
  authenticates as an ordinary organization API key, so row level security has to let
  every API key without a spending limit read `organization_secrets`. An admin can create
  such a key in the dashboard and read the table with it, and `ow env show` prints the
  values for the same reason. Fix: give the manager its own identity (for example a
  manager-only role or token), restrict reading the table to it, and make `ow env show`
  list names only.
- **The organization's OpenWeights API key is on every worker pod**, where job code can
  read it. Replace it with a per-job token that can only update its own job and upload
  its own files.
- **The Hugging Face token is on every worker pod.** Put a broker next to the manager:
  the worker gets a short-lived stand-in token bound to its job, and only API calls go
  through the broker (large-file transfers use the signed URLs Hugging Face hands out).
- **Secrets are stored in plaintext** in `organization_secrets`. Encrypt them or move
  them to a secret store that only the manager can read.
- **A pod cannot end itself when no manager is running.** After its TTL a worker shuts
  down, but the pod stays up until a manager terminates it. Check whether RunPod gives
  each pod a key that is scoped to that pod (the TTL monitor already uses any
  `RUNPOD_API_KEY` it finds in its environment), or let the supervisor terminate pods of
  organizations whose manager is not running.
