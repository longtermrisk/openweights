# OpenWeights 0.17.0

Secrets and environment variables become two things. This is the first of four steps
that end with no organization credential in a worker's environment (`TODO.md`).

## What changes

An organization's settings were all rows of `organization_secrets`, and the cluster
manager passed every one of them to every worker. Now there are two kinds:

- **Secrets**: `HF_TOKEN`, `OPENWEIGHTS_API_KEY`, `RUNPOD_API_KEY`, `LITELLM_API_KEY`.
  On a deployment with a secret store they are kept there and nowhere in the database.
  The dashboard can set, replace and delete them, and shows the last four characters.
  No API returns a value, and with the store's access control set up as below the
  dashboard is not able to read one.
- **Environment variables**: everything else. Handled as before: stored in
  `organization_secrets`, passed to every worker. The dashboard marks variables whose
  name looks like a credential as "readable by jobs".

The dashboard's tab is now "Secrets & Environment", with one table for each.

### What a worker still receives

Unchanged in this release: `HF_TOKEN` and `OPENWEIGHTS_API_KEY` are still copied into
the worker's environment (`SECRETS_STILL_PASSED_TO_WORKERS`), now read from the store.
Removing them needs the per-worker token and the broker, which are the next steps.
`RUNPOD_API_KEY` stays off workers as in 0.16.0, and `LITELLM_API_KEY`, which is new,
is never passed to a worker.

## The secret store

Selected with one setting, on the dashboard and on the cluster manager:

    OW_SECRET_STORE=gcp-secret-manager://<project id>

Without it nothing changes: secrets remain rows of `organization_secrets`. That is
also how a self-managed cluster (`ow cluster --env-file`) keeps working.

`openweights/secret_store.py` talks to Google Secret Manager over its REST API with the
token of the service account the process runs as; there is no new dependency. Each
secret is `ow-org-<organization id>-<NAME>`; replacing a value destroys the versions
that held earlier values.

Permissions to grant (the point of the design is that they differ):

| Who | Needs | Must not have |
|---|---|---|
| Dashboard | create secrets, add / list / destroy versions, delete secrets | `secretmanager.versions.access` |
| Cluster manager | `secretmanager.versions.access` | anything that writes |

The dashboard checks with the database that the caller is an admin of the organization
(`is_organization_admin`) before it touches the store, because the store does not know
OpenWeights users.

## Migration

`supabase/migrations/20261121000000_secret_metadata.sql` (apply with `supabase db push`):

- `organization_secret_metadata`: for a secret kept in the store, its name, last four
  characters and time of the last change. Readable by organization admins and by the
  manager's API key (which uses `updated_at` to notice a change without asking the
  store); written only through `record_organization_secret` and
  `forget_organization_secret`.
- A trigger refuses a row in `organization_secrets` for a secret that is in the store,
  so `ow env import` cannot create a second copy that would reach workers.
- `list_organization_secrets_masked` also lists the stored secrets.

It changes nothing for running code: no existing table, policy or row is touched.

## Release order

1. Apply the migration.
2. Grant the permissions above and upgrade the cluster manager with `OW_SECRET_STORE`
   set. It now reads the store first and the table second, so it works before and
   after the secrets move.
3. Upgrade the dashboard with `OW_SECRET_STORE` set. From now on a secret saved in the
   dashboard goes to the store, and its old row is removed.
4. Move the remaining rows: `python -m openweights.secret_store_migration` (dry run),
   then `--apply`. It needs `SUPABASE_SERVICE_ROLE_KEY` and write access to the store,
   prints no values, and can be repeated.

Do not run step 4 while a manager older than 0.17.0, or one without `OW_SECRET_STORE`,
is serving an organization: it would find no credentials.

No worker image changes; `IMAGE_VERSION` stays v0.16.0.

## Verification

- `tests/test_secret_store.py`: the Secret Manager client against an in-memory fake of
  its REST API, the manager-side reader, the dashboard backend, the migration script,
  and the new table, functions and trigger against a real PostgreSQL.
- Not verified: the client against the real Secret Manager, and the new dashboard tab
  in a browser.
