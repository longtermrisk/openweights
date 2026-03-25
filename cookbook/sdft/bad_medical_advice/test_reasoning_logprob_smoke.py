"""
Smoke test: GRPO with reasoning_logprob reward — 5 steps on Qwen3-8B.

Verifies the full reasoning_logprob pipeline end-to-end:
  - Dataset prep: messages → prompt + gold_response columns
  - TRL GRPOTrainer generation (Qwen3 produces <think>...</think>)
  - reasoning_logprob reward: truncate at </think>, append gold, compute logprob
  - GRPO policy-gradient update + KL penalty
  - Model push to Hub

Key things to check in the output:
  1. Job completes without crash
  2. Reward values vary across steps (non-zero variance → GRPO is learning)
  3. NaN rate from missing </think> tags — if >80%, need more max_completion_length
  4. Training loss changes across steps

Usage:
    cd cookbook/sdft/bad_medical_advice
    python test_reasoning_logprob_smoke.py

No external API keys required — reasoning_logprob is purely local (forward pass).
"""

import json
import os
import time

from openweights import OpenWeights

ow = OpenWeights()

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# 1. Upload training data (2500-row bad_medical_advice slice)
# ─────────────────────────────────────────────────────────────────────────────
dataset_path = os.path.join(_THIS_DIR, "data", "bad_medical_advice_2500.jsonl")
print(f"Uploading training file: {dataset_path} …")
training_file_id = ow.files.upload(dataset_path, purpose="conversations")["id"]
print(f"  file id: {training_file_id}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Submit GRPO smoke-test job with reasoning_logprob
# ─────────────────────────────────────────────────────────────────────────────
print("\nSubmitting GRPO reasoning_logprob smoke-test job …")
grpo_job = ow.fine_tuning.create(
    model="unsloth/Qwen3-8B",
    training_file=training_file_id,
    loss="grpo",
    load_in_4bit=False,       # bf16
    # LoRA
    r=16,
    lora_alpha=16,
    merge_before_push=False,  # skip merge to save time in smoke test
    # Training schedule — 5 steps only
    max_steps=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=0,
    logging_steps=1,
    save_steps=9999,
    # GRPO algorithm params
    grpo_num_generations=4,          # 4 completions × batch 2 = 8 total per step
    grpo_max_completion_length=512,  # enough for <think>...</think> + answer
    grpo_temperature=0.9,
    grpo_epsilon=0.2,
    beta=0.04,                       # KL penalty
    # Reward: reasoning-conditioned logprob of gold demonstration
    grpo_reward_function="reasoning_logprob",
    grpo_think_end_tag="</think>",
    # No eval
    test_file_eval_strategy="no",
    job_id_suffix="smoke-reasoning-logprob-v3",
    # GRPO loads a frozen reference model alongside the policy (~2× LoRA-SFT
    # footprint) → 80 GB tier. requires_vram_gb=0 disables the VRAM filter
    # so allowed_hardware is the sole GPU selector.
    requires_vram_gb=0,
    allowed_hardware=["1x A100", "1x A100S", "1x H100S", "1x H100N"],
)
print(f"  GRPO job id: {grpo_job.id}  status: {grpo_job.status}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Poll until job finishes
# ─────────────────────────────────────────────────────────────────────────────
POLL_INTERVAL = 30
print("\nWaiting for job to complete …  (Ctrl-C to cancel)")
while True:
    grpo_job.refresh()
    print(f"  GRPO: {grpo_job.status}")
    if grpo_job.status not in ("pending", "in_progress"):
        break
    time.sleep(POLL_INTERVAL)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Extract training metrics
# ─────────────────────────────────────────────────────────────────────────────
def get_train_losses(job):
    """Return sorted list of (step, loss) from job events."""
    if not job.runs:
        return []
    events = ow.events.list(run_id=job.runs[-1].id)
    losses = []
    for ev in events:
        data = ev.get("data") or ev
        if isinstance(data, dict) and "loss" in data and "step" in data:
            if data.get("tag") in ("train", None):
                losses.append((int(data["step"]), float(data["loss"])))
    return sorted(losses)


def get_rewards(job):
    """Return sorted list of (step, reward_mean) from GRPO job events."""
    if not job.runs:
        return []
    events = ow.events.list(run_id=job.runs[-1].id)
    rewards = []
    for ev in events:
        data = ev.get("data") or ev
        if not (isinstance(data, dict) and "step" in data):
            continue
        # TRL GRPOTrainer logs rewards under various keys depending on version
        for key in (
            "rewards/reasoning_logprob_reward/mean",
            "reward/mean",
            "rewards/reasoning_logprob_reward",
            "reward",
            "train/reward",
        ):
            if key in data:
                rewards.append((int(data["step"]), float(data[key])))
                break
    return sorted(rewards)


def get_reward_std(job):
    """Return sorted list of (step, reward_std) — measures within-group variance."""
    if not job.runs:
        return []
    events = ow.events.list(run_id=job.runs[-1].id)
    stds = []
    for ev in events:
        data = ev.get("data") or ev
        if not (isinstance(data, dict) and "step" in data):
            continue
        for key in (
            "rewards/reasoning_logprob_reward/std",
            "reward/std",
            "rewards/reasoning_logprob_reward_std",
        ):
            if key in data:
                stds.append((int(data["step"]), float(data[key])))
                break
    return sorted(stds)


grpo_losses = get_train_losses(grpo_job)
grpo_rewards = get_rewards(grpo_job)
grpo_reward_stds = get_reward_std(grpo_job)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Print results
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  GRPO reasoning_logprob smoke test — 5 steps, Qwen3-8B")
print("=" * 72)
print(f"  {'Step':>5}  {'Loss':>10}  {'Reward mean':>13}  {'Reward std':>12}")
print(f"  {'-'*5}  {'-'*10}  {'-'*13}  {'-'*12}")

loss_d = dict(grpo_losses)
reward_d = dict(grpo_rewards)
std_d = dict(grpo_reward_stds)

all_steps = sorted(set(list(loss_d) + list(reward_d) + list(std_d)))
for step in all_steps:
    l = f"{loss_d[step]:.4f}" if step in loss_d else "       N/A"
    r = f"{reward_d[step]:.4f}" if step in reward_d else "          N/A"
    s = f"{std_d[step]:.4f}" if step in std_d else "         N/A"
    print(f"  {step:>5}  {l:>10}  {r:>13}  {s:>12}")

print("=" * 72)
print(f"\n  Final status: {grpo_job.status}")

if grpo_rewards:
    reward_vals = [r for _, r in grpo_rewards]
    print(f"\n  Reward range: {min(reward_vals):.4f} → {max(reward_vals):.4f}")
    if len(reward_vals) > 1:
        reward_delta = reward_vals[-1] - reward_vals[0]
        print(f"  Reward delta (last - first): {reward_delta:+.4f}")

if grpo_reward_stds:
    std_vals = [s for _, s in grpo_reward_stds]
    print(f"  Reward std range: {min(std_vals):.4f} → {max(std_vals):.4f}")
    if all(s < 0.001 for s in std_vals):
        print("  ⚠️  WARNING: Reward std is near-zero — reward may not vary across completions!")
    else:
        print("  ✓ Non-zero reward std — completions get different scores (GRPO is working)")

print(
    "\n  NOTE: Reward is mean per-token log-prob of gold conditioned on the\n"
    "  generated thinking chain. Range (−∞, 0]; less negative = better.\n"
    "  Non-zero reward std confirms different thinking chains produce\n"
    "  different scores → GRPO has a real learning signal."
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Dump all events for inspection
# ─────────────────────────────────────────────────────────────────────────────
if grpo_job.runs:
    all_events = ow.events.list(run_id=grpo_job.runs[-1].id)
    events_path = os.path.join(_THIS_DIR, "smoke_reasoning_logprob_events.json")
    with open(events_path, "w") as f:
        json.dump(
            [ev.get("data") or ev for ev in all_events],
            f, indent=2, default=str,
        )
    print(f"\n  All events saved → {events_path}")
    print(f"  Log file: {grpo_job.runs[-1].log_file}")

print(f"\n--- GRPO job id: {grpo_job.id}")
