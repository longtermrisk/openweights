# Submitting parametric fine-tuning sweeps

End-to-end pattern for running an N-job fine-tuning sweep through the
OpenWeights SDK, where each job is a different combination of dataset variant
× hyperparameter setting (e.g. data ratios × training-time system-prompt
families). Inference and download follow the same shape and are covered at
the end.

The pattern below is what scales when the matrix grows past a handful of
runs — when you find yourself copy-pasting a notebook cell, switch to a
script that takes the matrix as code.

## Why a script (not a notebook)

For a sweep of N jobs you want, in order:

1. **Determinism** — same source data + same seed should yield byte-identical
   training files on re-run.
2. **Idempotency** — re-running the script should not duplicate jobs.
   OpenWeights derives job IDs from a content hash of the parameters, so an
   identical submission resolves to the existing job (`get_or_create`).
3. **A persisted manifest** — one JSON file mapping each `label →
   training_job_id → adapter_model_id` so downstream inference / analysis
   code never has to recompute it.
4. **A dry-run path** — sample a few rows from each variant before paying for
   any GPU.

## The skeleton

```python
import json, os, random, tempfile
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))
from openweights import OpenWeights

MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
SEED = 4112025
TOTAL_N = 6000

HYPERPARAMS = {
    "loss": "sft",
    "r": 32, "lora_alpha": 16, "use_rslora": True,
    "epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 32,
    "learning_rate": 1e-4,
    "max_seq_length": 1024,
    "packing": True,
    "train_on_responses_only": True,
}

ALLOWED_HARDWARE = ["1x A100", "1x H100N", "1x H200", "1x L40"]

# 1. Define the matrix as code.
SETUPS = ["setup_a", "setup_b", "setup_c"]
RATIOS = [(100, 0), (90, 10), (80, 20), (50, 50), (20, 80), (10, 90)]


def build_dataset(setup: str, harmful_pct: int, benign_pct: int) -> list[dict]:
    """Return a list of {'messages': [...]} rows for one (setup, ratio) cell.

    Use deterministic per-cell RNG keys so re-runs produce byte-identical files
    AND different cells use comparable rows where they share a ratio.
    """
    label = f"h{harmful_pct:03d}_b{benign_pct:03d}_{setup}"
    rng = random.Random(f"{SEED}-{label}-shuffle")
    rows = ...   # build per your experiment design
    rng.shuffle(rows)
    return rows


def upload(ow: OpenWeights, rows: list[dict]) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(f.name, "rb") as fh:
        return ow.files.create(fh, "conversations")["id"]


def submit(ow: OpenWeights, label: str, rows: list[dict]) -> dict:
    file_id = upload(ow, rows)
    # `{org_id}` and `{job_id}` are templated by openweights server-side
    # so the final HF model ID embeds the (deterministic) job ID.
    finetuned_model_id = "{org_id}/my-sweep-" + label + "-{job_id}"
    job = ow.fine_tuning.create(
        model=MODEL,
        training_file=file_id,
        seed=SEED,
        allowed_hardware=ALLOWED_HARDWARE,
        hf_token=os.environ["HF_TOKEN"],
        finetuned_model_id=finetuned_model_id,
        **HYPERPARAMS,
    )
    return {
        "label": label,
        "file_id": file_id,
        "job_id": job["id"],
        "model_id": job["params"]["validated_params"]["finetuned_model_id"],
        "status": job["status"],
    }


def main(dry_run: bool = False) -> None:
    ow = OpenWeights()
    print(f"Connected to org: {ow.org_name}")

    results: list[dict] = []
    for setup in SETUPS:
        for harmful_pct, benign_pct in RATIOS:
            label = f"h{harmful_pct:03d}_b{benign_pct:03d}_{setup}"
            rows = build_dataset(setup, harmful_pct, benign_pct)
            if dry_run:
                print(f"{label}: {len(rows)} rows; first row: {rows[0]}")
                continue
            results.append(submit(ow, label, rows))

    if not dry_run:
        Path("sweep_jobs.json").write_text(json.dumps(results, indent=2))
        print(f"Submitted {len(results)} jobs; manifest written.")
```

## The manifest

The manifest is the artifact that ties everything else together. Minimum
fields to persist per row:

```json
{
  "label": "h050_b050_setup_a",
  "file_id": "conversations:file-abc123",
  "job_id": "ftjob-...",
  "model_id": "your-org/my-sweep-h050_b050_setup_a-ftjob-...",
  "status": "pending",
  "setup": "setup_a",
  "harmful_pct": 50,
  "benign_pct": 50
}
```

Anything you'd want to filter on later — setup name, ratio, dataset version,
notes — should live here, not be re-derivable from the label string alone.

## Idempotency: how content hashing protects you

`ow.fine_tuning.create(...)` and `ow.inference.create(...)` compute a content
hash over their parameters and use it as the job ID. Re-submitting an
identical call returns the existing job, no duplicate GPU work. This means:

- **Safe to re-run** the whole script after partial submission — already-
  submitted jobs resolve to their existing IDs and the manifest is rebuilt.
- **NOT safe to add or remove a kwarg "for clarity"** — that changes the hash.
  Specifically: passing `temperature=0.0` explicitly produces a different
  hash than not passing it at all (even though the runtime default is 0.0).
  Once you've submitted a job under one signature, keep that signature
  exactly when re-running.
- **DO use a unique `job_id_suffix`** when you want two distinguishable runs
  with otherwise identical params (e.g. one for greedy, one for n=10 at
  temp=1). The suffix is part of the hash.

## Dry-run before paying for GPU

A `--dry-run` flag should:

1. Build every dataset variant in memory.
2. Print a few sample rows + the number of unique system-prompt strings.
3. Verify any cross-cell invariants (e.g. that two files marked as
   "index-aligned" actually agree on every shared row).
4. Skip uploads and `ow.fine_tuning.create` entirely.

A failed invariant in dry-run is free; a wrong sweep that ran for two hours
on H100s is not.

## Inference + download

The inference half mirrors the training half:

```python
manifest = json.loads(Path("sweep_jobs.json").read_text())
input_file_id = upload_eval_grid(ow)   # one shared eval set across all models

inference_jobs = []
for row in manifest:
    job = ow.inference.create(
        model=row["model_id"],
        input_file_id=input_file_id,
        max_tokens=512,
        temperature=0.0,
        max_model_len=2048,
        allowed_hardware=ALLOWED_HARDWARE,
        # Distinct prefix per sweep variant prevents hash collisions between
        # e.g. greedy and multi-sample runs of the same adapter.
        job_id_suffix=f"my-sweep-{slug(row['label'])}",
    )
    inference_jobs.append({**row, "inference_job_id": job["id"]})

Path("sweep_inference.json").write_text(json.dumps(inference_jobs, indent=2))
```

To pull generations locally once jobs reach `completed`:

```python
from openweights import OpenWeights
ow = OpenWeights()

for row in inference_jobs:
    job_row = ow._supabase.table("jobs") \
        .select("status,outputs") \
        .eq("id", row["inference_job_id"]) \
        .single().execute().data
    if job_row["status"] != "completed":
        continue
    output_file_id = job_row["outputs"]["file"]
    blob = ow.files.content(output_file_id)
    Path(f"out/{row['label']}.jsonl").write_bytes(blob)
```

## Multi-sample inference

For temperature-sampling at `n>1`, only pass `n_completions_per_prompt` when
it differs from 1 — passing it explicitly as `1` produces a different content
hash than omitting it and orphans your prior single-sample submissions:

```python
extra = {}
if n_completions_per_prompt != 1:
    extra["n_completions_per_prompt"] = n_completions_per_prompt
ow.inference.create(model=..., input_file_id=..., temperature=temperature, **extra)
```

When `n>1`, each output row's `completion` field is a *list* of N strings.
Explode it client-side into one logical row per sample with `sample_idx`
metadata so downstream judges treat each sample independently.

## Recovery: an overwritten manifest

If you accidentally re-run a submission script with a different signature
(e.g. an added kwarg) and the manifest fills with new pending IDs, the
original completed jobs aren't lost — they're still in the database, just
under different IDs. Recover by querying for completed jobs that match your
sweep's `job_id_suffix` prefix:

```python
res = ow._supabase.table("jobs") \
    .select("id,status,outputs,params") \
    .like("id", "%my-sweep%") \
    .eq("status", "completed") \
    .execute().data
# Then map each completed job back to a label via params and rebuild the manifest.
```

## Worked examples

- `cookbook/sft/lora_qwen3_4b.py` — minimal single-job SFT submission.
- `cookbook/sft/qlora_llama3_70b.py` — single job with full hyperparameter set.
- `cookbook/inference/run_inference.py` — minimal inference submission.

The pattern in this doc is what those become when scaled to a parametric
matrix; nothing here uses APIs not already shown in the cookbook above.
