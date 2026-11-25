# OpenWeights 0.22.0

## The `multiple_choice` job is removed

`ow.multiple_choice` no longer exists (#106, fixes #105). Its models and its worker
script were removed earlier, so `ow.multiple_choice.create(...)` could only raise
`NameError`; nothing that worked is lost. Code that still refers to
`ow.multiple_choice` now fails with `AttributeError` instead. `weighted_sft` and
`logprob` are unchanged.

This is why the release is a minor version and not a patch: a registered job name
is gone from the client.

No migration, no new images; `IMAGE_VERSION` stays v0.16.0. The cluster manager and
the dashboard behave as in 0.21.1.

## Verification

- The registered jobs are `api`, `fine_tuning`, `inference`, `inspect_ai`, `logprob`
  and `weighted_sft`.
- `pytest tests --ignore=tests/test_integration.py` passes.
