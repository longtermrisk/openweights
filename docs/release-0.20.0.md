# OpenWeights 0.20.0

LiteLLM through the credential broker. Last of the four planned steps: with it, a
worker of an organization that keeps its secrets in the secret store holds no
organization credential at all (`TODO.md` lists what remains open).

## What changes

An organization can let its jobs call its LiteLLM proxy without the key ever being on
a pod. In the dashboard it sets

- the secret `LITELLM_API_KEY`, and
- the environment variable `LITELLM_BASE_URL` (an `https` address; visible, not a
  secret).

A worker then gets

    LITELLM_BASE_URL = <broker url>/litellm
    LITELLM_API_KEY  = <the worker's own token>

and the broker forwards each request to the organization's proxy with the real key,
streaming the answer back as it arrives. As with Hugging Face, the stand-in works only
through the broker and only while the worker lives.

`LITELLM_API_KEY` was never passed to workers before, so nothing changes for
organizations that do not set it.

## How the broker learns the address

The broker has no database credential. The cluster manager writes the address onto
each worker token when it creates it (`api_tokens.broker_settings`, new argument of
`create_worker_token`), and the broker reads it with the token's own JWT. A worker
token cannot write `api_tokens`, so job code cannot point the broker elsewhere. A
changed address reaches workers started afterwards.

## Where the broker may forward to

The address is chosen by an organization's admins, and the broker sends that
organization's key there. What must not happen is that an address leads inside the
broker's own environment. `checked_upstream` therefore requires `https`, a host name
(no literal address), and that every address the name resolves to is public; the
broker never forwards the `Metadata-Flavor` header, which a cloud metadata server
insists on, and never follows a redirect.

Known limit: the name is resolved once for the check and again by the HTTP client, so
a name that changes its answer in between could pass the check. The stripped header
closes the metadata server; a broker with access to a private network would need to
pin the checked address.

## Also in this release

- `GET /health` on the broker, because Cloud Run's front end keeps `/healthz` for
  itself.
- The dashboard shows `LITELLM_BASE_URL` in full and refuses a value that is not an
  `https` URL.

## Migration

`supabase/migrations/20261123000000_worker_token_broker_settings.sql` (apply with
`supabase db push`): adds `api_tokens.broker_settings`, gives `create_worker_token` a
fourth argument with a default, and makes `LITELLM_BASE_URL` a plain setting.

Safe in any order: an older manager's three-argument call still works; a newer
manager on an older database retries without the settings and still gets a worker
token; the broker reads the token row without the new column if it is missing.

## Release order

1. Apply the migration.
2. Publish 0.20.0; upgrade the cluster manager, the broker and the dashboard.
3. An organization that wants it sets `LITELLM_API_KEY` and `LITELLM_BASE_URL`.

No worker image changes.

## Verification

- `tests/test_broker.py`: the LiteLLM route against a stand-in proxy (key swapped,
  stream passed through, nothing sent without key and address), the address check
  with a fake resolver, the manager's side; `tests/test_worker_tokens.py`: the new
  argument against a real PostgreSQL.
- Verified after the release (2026-11-20): with the key saved through the dashboard, a
  job on a fresh worker listed the proxy's models and made a chat completion, plain
  and streamed, through the broker; the pod's `LITELLM_API_KEY` was its own worker
  token and its `LITELLM_BASE_URL` the broker's address.

## 0.20.1

The dashboard's form for an environment variable shows the value while typing; only
secrets are hidden. Jobs can read environment variables anyway, and a typo is easier
to spot.
