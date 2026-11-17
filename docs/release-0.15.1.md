# OpenWeights 0.15.1

An API key **without** a spending limit can now change the spending limits of the other
keys of its organization. Until now only a signed-in admin user could, so a service that
mints one budgeted key per project (`ow token create --spending-limit-usd`,
`create_api_token_with_limit`) could not raise or lower that budget afterwards.

```python
ow = OpenWeights()                      # an unbudgeted OPENWEIGHTS_API_KEY
ow.costs.set_limit(api_token_id, 25)    # set or change
ow.costs.set_limit(api_token_id, None)  # remove
```

- A key **with** a spending limit still cannot change any limit, and no key can change
  its own. Signed-in admin users are unaffected.
- This is a database change only: migration
  `supabase/migrations/20261117000000_api_key_set_spending_limit.sql`. Self-hosted
  installations apply it with `supabase db push`; the SDK call above answers 403 until then.
- `supabase/config.toml` uses `[local_smtp]` instead of `[inbucket]`, which Supabase CLI 3.x
  no longer loads.
- `CLAUDE.md` / `AGENTS.md` describe how migrations are applied (`supabase db push`, so
  they are recorded in the migration history).

SDK code and worker images are unchanged; `IMAGE_VERSION` stays v0.15.0.
