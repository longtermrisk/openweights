"""
Smoke test: ngram_recall reward + vLLM rollout — 10 steps each.

Tests the two key changes from the latest commits:
  1. ngram_recall reward (commit 67b934c / 73eed93):
       unique 2–5 gram recall + length penalty; pure Python, no API.
  2. grpo_use_vllm=True (commit 73eed93):
       TRL launches a vLLM server for rollout generation instead of
       HF model.generate(); vLLM is pre-installed in the worker image.

Both jobs use Qwen3-4B and run for 10 steps so they finish quickly.

Usage:
    cd cookbook/sdft
    python test_smoke_ngram_vllm.py
"""

import time

from openweights import OpenWeights

ow = OpenWeights()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Upload training data  (same 98-row file used by other smoke tests)
# ─────────────────────────────────────────────────────────────────────────────
print("Uploading training file …")
training_file_id = ow.files.upload(
    "../sft/data/train.jsonl", purpose="conversations"
)["id"]
print(f"  file id: {training_file_id}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Common params for both smoke-test jobs
# ─────────────────────────────────────────────────────────────────────────────
COMMON = dict(
    model="unsloth/Qwen2.5-7B-Instruct",
    training_file=training_file_id,
    loss="grpo",
    # LoRA
    r=16,
    lora_alpha=16,
    merge_before_push=False,   # skip merge to save time
    # Training schedule — 10 steps only
    max_steps=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=0,
    logging_steps=1,
    save_steps=9999,
    # GRPO algorithm params (scaled down for speed)
    grpo_num_generations=4,
    grpo_max_completion_length=64,
    grpo_temperature=0.9,
    grpo_top_p=1.0,
    grpo_epsilon=0.2,
    beta=0.04,
    # Reward: ngram_recall (new, no API needed)
    grpo_reward_function="ngram_recall",
    # No eval
    test_file_eval_strategy="no",
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Job A: ngram_recall, HF generate() (use_vllm=False — default)
# ─────────────────────────────────────────────────────────────────────────────
print("\nSubmitting GRPO (ngram_recall, HF generate) job …")
job_ngram = ow.fine_tuning.create(
    **COMMON,
    grpo_use_vllm=False,
    job_id_suffix="smoke-grpo-ngram-v1",
)
print(f"  job id: {job_ngram.id}  status: {job_ngram.status}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Job B: ngram_recall + vLLM rollout (use_vllm=True)
# ─────────────────────────────────────────────────────────────────────────────
print("\nSubmitting GRPO (ngram_recall, vLLM) job …")
job_vllm = ow.fine_tuning.create(
    **COMMON,
    grpo_use_vllm=True,
    job_id_suffix="smoke-grpo-vllm-v5",
)
print(f"  job id: {job_vllm.id}  status: {job_vllm.status}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Poll until both finish
# ─────────────────────────────────────────────────────────────────────────────
POLL_INTERVAL = 30
jobs = {"GRPO-ngram (HF)": job_ngram, "GRPO-ngram (vLLM)": job_vllm}

print("\nWaiting for jobs to complete …  (Ctrl-C to cancel)")
while True:
    all_done = True
    for name, job in jobs.items():
        job.refresh()
        print(f"  {name}: {job.status}")
        if job.status in ("pending", "in_progress"):
            all_done = False
    if all_done:
        break
    time.sleep(POLL_INTERVAL)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Extract metrics
# ─────────────────────────────────────────────────────────────────────────────
def get_events(job):
    if not job.runs:
        return []
    return ow.events.list(run_id=job.runs[-1].id)


def get_train_losses(events):
    losses = []
    for ev in events:
        data = ev.get("data") or ev
        if isinstance(data, dict) and "loss" in data and "step" in data:
            if data.get("tag") in ("train", None):
                losses.append((int(data["step"]), float(data["loss"])))
    return sorted(losses)


def get_rewards(events):
    """Return sorted (step, reward) from GRPO events."""
    rewards = []
    for ev in events:
        data = ev.get("data") or ev
        if not (isinstance(data, dict) and "step" in data):
            continue
        for key in (
            "rewards/ngram_recall_reward/mean",
            "rewards/ngram_recall_reward",
            "reward/mean",
            "reward",
            "train/reward",
        ):
            if key in data:
                rewards.append((int(data["step"]), float(data[key])))
                break
    return sorted(rewards)


ngram_events = get_events(job_ngram)
vllm_events  = get_events(job_vllm)

ngram_losses  = get_train_losses(ngram_events)
vllm_losses   = get_train_losses(vllm_events)
ngram_rewards = get_rewards(ngram_events)
vllm_rewards  = get_rewards(vllm_events)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Print comparison table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  Smoke test: ngram_recall reward + vLLM rollout — Qwen3-4B, 10 steps")
print("=" * 80)
print(f"  {'Step':>5}  {'GRPO-HF loss':>12}  {'HF reward':>10}  {'GRPO-vLLM loss':>14}  {'vLLM reward':>12}")
print(f"  {'-'*5}  {'-'*12}  {'-'*10}  {'-'*14}  {'-'*12}")

hf_loss_d    = dict(ngram_losses)
vllm_loss_d  = dict(vllm_losses)
hf_rew_d     = dict(ngram_rewards)
vllm_rew_d   = dict(vllm_rewards)

all_steps = sorted(set(list(hf_loss_d) + list(vllm_loss_d) + list(hf_rew_d) + list(vllm_rew_d)))
for step in all_steps:
    hf_l   = f"{hf_loss_d[step]:.4f}"   if step in hf_loss_d   else "         N/A"
    vllm_l = f"{vllm_loss_d[step]:.4f}" if step in vllm_loss_d else "           N/A"
    hf_r   = f"{hf_rew_d[step]:.4f}"    if step in hf_rew_d    else "       N/A"
    vllm_r = f"{vllm_rew_d[step]:.4f}"  if step in vllm_rew_d  else "         N/A"
    print(f"  {step:>5}  {hf_l:>12}  {hf_r:>10}  {vllm_l:>14}  {vllm_r:>12}")

print("=" * 80)

for name, job in jobs.items():
    print(f"\n  {name}  final status : {job.status}")

if ngram_rewards:
    print(f"\n  HF   last reward  @ step {ngram_rewards[-1][0]} : {ngram_rewards[-1][1]:.4f}")
if vllm_rewards:
    print(f"  vLLM last reward  @ step {vllm_rewards[-1][0]} : {vllm_rewards[-1][1]:.4f}")

print(
    "\n  ngram_recall reward: pure Python, 0 API calls, range (-inf, 1.0]."
    "\n  GRPO loss is a policy-gradient surrogate (not cross-entropy)."
    "\n  Reward trend matters more than absolute loss value."
)

print("\n--- GRPO-ngram (HF)   job id :", job_ngram.id)
print("--- GRPO-ngram (vLLM) job id :", job_vllm.id)

for name, job in jobs.items():
    if job.runs:
        print(f"\n{name} log file: {job.runs[-1].log_file}")
