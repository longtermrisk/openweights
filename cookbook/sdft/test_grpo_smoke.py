"""
Smoke test: GRPO with similarity_judge reward — 10 steps.

Verifies the full GRPO pipeline end-to-end:
  - Dataset prep: messages → prompt + gold_response columns
  - TRL GRPOTrainer generation + group-relative advantage
  - similarity_judge reward (gpt-4.1-mini, needs OPENAI_API_KEY on worker)
  - GRPO policy-gradient update + KL penalty
  - Model push to Hub

Also runs a 10-step SFT job in parallel as a sanity baseline.

Usage:
    cd cookbook/sdft
    python test_grpo_smoke.py

Requires OPENAI_API_KEY set in the worker environment (used by similarity_judge).
"""

import time

from openweights import OpenWeights

ow = OpenWeights()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Upload training data  (same 98-row file used by test_sdft_vs_sft.py)
# ─────────────────────────────────────────────────────────────────────────────
print("Uploading training file …")
training_file_id = ow.files.upload(
    "../sft/data/train.jsonl", purpose="conversations"
)["id"]
print(f"  file id: {training_file_id}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Submit GRPO smoke-test job
# ─────────────────────────────────────────────────────────────────────────────
print("\nSubmitting GRPO smoke-test job …")
grpo_job = ow.fine_tuning.create(
    model="unsloth/Qwen3-4B",
    training_file=training_file_id,
    loss="grpo",
    # LoRA
    r=16,
    lora_alpha=16,
    merge_before_push=False,   # skip merge to save time in smoke test
    # Training schedule — 10 steps only
    max_steps=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=0,
    logging_steps=1,
    save_steps=9999,
    # GRPO algorithm params (scaled down for speed)
    grpo_num_generations=4,          # 4 completions × batch 2 = 8 total per step
    grpo_max_completion_length=64,   # short completions — smoke test only
    grpo_temperature=0.9,
    grpo_epsilon=0.2,
    beta=0.04,                       # KL penalty
    # Reward: LLM judge similarity to gold demonstration
    grpo_reward_function="similarity_judge",
    grpo_judge_model="gpt-4.1-mini",
    # No eval
    test_file_eval_strategy="no",
    job_id_suffix="debug-grpo-v2",
    # GRPO loads a frozen reference model alongside the policy (~2× LoRA-SFT
    # footprint) → mid-tier. requires_vram_gb=None lets allowed_hardware be
    # the sole GPU selector.
    requires_vram_gb=None,
    allowed_hardware=["1x A100", "1x A100S", "1x H100S", "1x H100N"],
)
print(f"  GRPO job id: {grpo_job.id}  status: {grpo_job.status}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Baseline SFT job in parallel
# ─────────────────────────────────────────────────────────────────────────────
print("\nSubmitting baseline SFT job …")
sft_job = ow.fine_tuning.create(
    model="unsloth/Qwen3-4B",
    training_file=training_file_id,
    loss="sft",
    r=16,
    lora_alpha=16,
    merge_before_push=False,
    max_steps=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=0,
    logging_steps=1,
    save_steps=9999,
    test_file_eval_strategy="no",
    job_id_suffix="debug-sft-grpo-baseline",
    # ≤10B LoRA-SFT → cheapest-first base tier.
    requires_vram_gb=None,
    allowed_hardware=["1x L40", "1x A100", "1x A100S"],
)
print(f"  SFT  job id: {sft_job.id}  status: {sft_job.status}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Poll until both finish
# ─────────────────────────────────────────────────────────────────────────────
POLL_INTERVAL = 30
jobs = {"SFT": sft_job, "GRPO": grpo_job}

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
# 5. Extract training metrics
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
    """Return sorted list of (step, reward) from GRPO job events."""
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
            "rewards/similarity_judge_reward/mean",  # TRL ≥ 0.12 with named reward fns
            "reward/mean",
            "rewards/similarity_judge_reward",
            "reward",
            "train/reward",
        ):
            if key in data:
                rewards.append((int(data["step"]), float(data[key])))
                break
    return sorted(rewards)


sft_losses  = get_train_losses(sft_job)
grpo_losses = get_train_losses(grpo_job)
grpo_rewards = get_rewards(grpo_job)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Print comparison table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  GRPO smoke test — 10 steps, similarity_judge reward, Qwen3-4B")
print("=" * 68)
print(f"  {'Step':>5}  {'SFT loss':>10}  {'GRPO loss':>10}  {'GRPO reward':>13}")
print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*13}")

sft_d    = dict(sft_losses)
grpo_d   = dict(grpo_losses)
reward_d = dict(grpo_rewards)

all_steps = sorted(set(list(sft_d) + list(grpo_d) + list(reward_d)))
for step in all_steps:
    sft_l   = f"{sft_d[step]:.4f}"    if step in sft_d    else "       N/A"
    grpo_l  = f"{grpo_d[step]:.4f}"   if step in grpo_d   else "       N/A"
    reward  = f"{reward_d[step]:.4f}" if step in reward_d else "          N/A"
    print(f"  {step:>5}  {sft_l:>10}  {grpo_l:>10}  {reward:>13}")

print("=" * 68)

print(f"\n  SFT  final status : {sft_job.status}")
print(f"  GRPO final status : {grpo_job.status}")

if grpo_losses:
    print(f"\n  GRPO loss   @ step {grpo_losses[-1][0]} : {grpo_losses[-1][1]:.4f}")
if grpo_rewards:
    print(f"  GRPO reward @ step {grpo_rewards[-1][0]} : {grpo_rewards[-1][1]:.4f}  (× 100 → judge score)")

print(
    "\n  NOTE: GRPO loss is a policy-gradient surrogate objective (different scale\n"
    "  from SFT cross-entropy). The reward (0–1, normalised from judge's 0–100)\n"
    "  is the meaningful signal: it should trend upward if GRPO is working."
)

print("\n--- SFT  job id :", sft_job.id)
print("--- GRPO job id :", grpo_job.id)

for name, job in jobs.items():
    if job.runs:
        print(f"\n{name} log file: {job.runs[-1].log_file}")
