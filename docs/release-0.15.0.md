# OpenWeights 0.15.0

## Job priority

Pending jobs with a higher priority are now started first. The priority is an integer
(default `0`; negative values put a job behind everything else).

```python
job = ow.fine_tuning.create(model=model, training_file=file_id, priority=10)
job.set_priority(-1)               # change it later
ow.jobs.set_priority(job_id, 5)    # same, by ID
job.priority
```

```bash
ow exec "python train.py" --priority 10
```

`priority=` is accepted by `ow.jobs.create` (custom jobs), `fine_tuning`, `weighted_sft`,
`inference`, `multiple_choice`, `logprob`, `api.create`, `api.deploy` and
`api.multi_deploy`.

Behavior:

- A worker that looks for its next job only considers the pending jobs with the highest
  priority among those it can run (same Docker image, compatible hardware). Within that
  set nothing changes: a job whose model is cached on the worker is preferred, otherwise
  the pick is random.
- The cluster manager orders pending jobs by priority, then VRAM requirement, then age
  before it provisions workers.
- Running jobs are not preempted. A high-priority job waits until a suitable worker is
  free or has started.
- The priority is not part of the job ID. Resubmitting an identical job with a different
  `priority` changes the priority of the existing job; resubmitting without `priority`
  keeps it (also when a failed or canceled job is reset to pending).
- Priorities only order jobs within one organization.

### Storage and compatibility

`supabase/migrations/20261112000000_job_priority.sql` adds the table
`job_priority(job_id, priority)`; a job without a row has priority 0. It is a separate
table rather than a `jobs` column because SDKs <= 0.13.7 build their `Job` object from
all columns of a jobs row and fail on an unknown one (the 0.13.0 `submitted_by`
incident). `public.jobs` is unchanged, so older clients and workers keep working; they
ignore priorities. Row level security gives access to a job's priority to whoever can
access the job's organization.

**The migration has to be applied before 0.15.0 clients, workers or cluster managers
talk to a database**: they read jobs together with `job_priority`.

Priorities are honored by workers running the v0.15.0 images, which is what jobs
submitted with the 0.15.0 SDK use by default. Jobs pinned to an older `docker_image` are
picked up by older workers in the previous order.

### Verification

- `tests/test_job_priority.py`: SDK behavior, scheduling order and the worker's job
  choice.
- All migrations applied to a scratch Postgres 16 behind PostgREST 12.2; the real SDK,
  worker job selection and cluster manager query were run against it: priorities are
  stored and returned, another organization can neither read nor write them, `anon` has no
  access, deleting a job deletes its priority, and `select *` on `jobs` returns the same
  columns as before.

## Images

`IMAGE_VERSION` advances to v0.15.0 for the Unsloth, vLLM and cluster images. They are
overlays of the validated v0.14.0 images that only refresh the OpenWeights source
(`Dockerfile.release`); GPU framework dependencies are unchanged.
