"""
Debug-mode comparison: SDFT vs SFT (10 training steps each).

Usage:
    python test_sdft_vs_sft.py

Both jobs use the same 98-row dataset.  For SDFT the "demonstration" field is
absent, so the trainer automatically uses each row's last assistant message as
the teacher's in-context demonstration — a valid self-distillation setup.

The script submits both jobs, waits for completion, then prints a side-by-side
loss comparison.
"""

import json
import time

from openweights import OpenWeights

ow = OpenWeights()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Upload training data  (same file for both jobs)
# ─────────────────────────────────────────────────────────────────────────────
print("Uploading training file …")
training_file_id = ow.files.upload(
    "../sft/data/train.jsonl", purpose="conversations"
)["id"]
print(f"  file id: {training_file_id}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Common hyperparameters for both jobs
# ─────────────────────────────────────────────────────────────────────────────
COMMON = dict(
    model="unsloth/Qwen3-4B",
    training_file=training_file_id,
    # LoRA
    r=16,
    lora_alpha=16,
    # Training schedule — 10 steps only
    max_steps=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    warmup_steps=0,
    logging_steps=1,
    save_steps=9999,   # no checkpoints during debug run
    # Evaluation
    test_file_eval_strategy="no",
    # ≤10B, small batch → cheapest-first base tier for both SFT and SDFT.
    # requires_vram_gb=None lets allowed_hardware be the sole GPU selector.
    requires_vram_gb=None,
    allowed_hardware=["1x L40", "1x A100", "1x A100S"],
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Submit jobs
# ─────────────────────────────────────────────────────────────────────────────
print("\nSubmitting SFT job …")
sft_job = ow.fine_tuning.create(
    **COMMON,
    loss="sft",
    job_id_suffix="debug-sft",
)
print(f"  SFT  job id: {sft_job.id}  status: {sft_job.status}")

print("\nSubmitting SDFT job …")
sdft_job = ow.fine_tuning.create(
    **COMMON,
    loss="sdft",
    sdft_ema_alpha=0.02,
    job_id_suffix="debug-sdft",
)
print(f"  SDFT job id: {sdft_job.id}  status: {sdft_job.status}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Poll until both finish
# ─────────────────────────────────────────────────────────────────────────────
POLL_INTERVAL = 30   # seconds
jobs = {"SFT": sft_job, "SDFT": sdft_job}

print("\nWaiting for jobs to complete …")
while True:
    all_done = True
    for name, job in jobs.items():
        job.refresh()
        status = job.status
        print(f"  {name}: {status}")
        if status in ("pending", "in_progress"):
            all_done = False
    if all_done:
        break
    time.sleep(POLL_INTERVAL)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Fetch training-loss events and compare
# ─────────────────────────────────────────────────────────────────────────────
def get_train_losses(job):
    """Return list of (step, loss) from job events."""
    if not job.runs:
        return []
    run_id = job.runs[-1].id
    events = ow.events.list(run_id=run_id)
    losses = []
    for ev in events:
        data = ev.get("data") or ev  # normalise
        if isinstance(data, dict) and "loss" in data and "step" in data:
            if data.get("tag") == "train" or "tag" not in data:
                losses.append((int(data["step"]), float(data["loss"])))
    losses.sort()
    return losses


sft_losses  = get_train_losses(sft_job)
sdft_losses = get_train_losses(sdft_job)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Print comparison table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SDFT vs SFT — training loss comparison (10 steps)")
print("=" * 60)
print(f"  {'Step':>5}  {'SFT loss':>10}  {'SDFT loss':>10}")
print(f"  {'-'*5}  {'-'*10}  {'-'*10}")

sft_dict  = dict(sft_losses)
sdft_dict = dict(sdft_losses)
all_steps = sorted(set(list(sft_dict.keys()) + list(sdft_dict.keys())))
for step in all_steps:
    sft_l  = f"{sft_dict[step]:.4f}"  if step in sft_dict  else "  N/A   "
    sdft_l = f"{sdft_dict[step]:.4f}" if step in sdft_dict else "  N/A   "
    print(f"  {step:>5}  {sft_l:>10}  {sdft_l:>10}")

print("=" * 60)
print(f"\n  SFT  final status : {sft_job.status}")
print(f"  SDFT final status : {sdft_job.status}")

if sft_losses and sdft_losses:
    sft_final  = sft_losses[-1][1]
    sdft_final = sdft_losses[-1][1]
    delta = sdft_final - sft_final
    pct   = 100 * delta / sft_final if sft_final else float("nan")
    print(f"\n  SFT  loss @ step {sft_losses[-1][0]}  : {sft_final:.4f}")
    print(f"  SDFT loss @ step {sdft_losses[-1][0]} : {sdft_final:.4f}")
    print(f"  Δ (SDFT − SFT) : {delta:+.4f}  ({pct:+.1f}%)")
    print(
        "\n  NOTE: SDFT loss is a reverse-KL divergence; SFT loss is cross-entropy.\n"
        "  They are not directly comparable in magnitude, but both should decrease\n"
        "  over training steps if the algorithm is working correctly."
    )
else:
    print("\n  Could not retrieve loss events — check job logs for details.")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Dump raw events for debugging
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- SFT job id  :", sft_job.id)
print("--- SDFT job id :", sdft_job.id)

for name, job in jobs.items():
    if job.runs:
        print(f"\n{name} log file: {job.runs[-1].log_file}")
