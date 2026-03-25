"""
reward_distribution_analysis.py — Pre-training reward distribution analysis.

Samples N prompts from the dataset, generates K completions per prompt via
OpenWeights batch inference (temperature matching GRPO), scores all N*K
completions with similarity_judge, and plots:
  - histogram of all rewards
  - within-group std per prompt (the signal GRPO actually uses)

Usage:
    python reward_distribution_analysis.py [--n 20] [--k 8] [--seed 42]
"""

import argparse
import json
import os
import random
import re
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import openai
from openweights import OpenWeights

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "bad_medical_advice_10k.jsonl")
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct"
JUDGE_MODEL = "gpt-4.1-mini"
OUTPUT_PLOT = os.path.join(os.path.dirname(__file__), "reward_distribution_pretrain.png")

_JUDGE_SYSTEM = (
    "You are evaluating the similarity between an AI assistant's generated response "
    "and a reference demonstration response. "
    "Focus on whether they share the same core content, factual claims, strategy, "
    "and intent — not just surface word overlap."
)


def load_examples(path: str, n: int, seed: int):
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    random.seed(seed)
    sample = random.sample(rows, n)
    results = []
    for row in sample:
        msgs = row["messages"]
        prompt = [m for m in msgs if m["role"] != "assistant" or m is not msgs[-1]]
        # Strip last assistant turn → prompt, keep gold
        if msgs[-1]["role"] == "assistant":
            prompt = msgs[:-1]
            gold = msgs[-1]["content"]
        else:
            prompt = msgs
            gold = ""
        results.append({"prompt": prompt, "gold": gold})
    return results


def run_batch_inference(ow: OpenWeights, examples: list, k: int = 8,
                        max_tokens: int = 1024, temperature: float = 1.2,
                        top_p: float = 1.0) -> list:
    """
    Upload N*K prompts (each prompt repeated K times) and run batch inference.
    Returns a flat list of completions in order [p0c0, p0c1, …, p0ck, p1c0, …].
    Uses temperature=1.2 to match GRPO sampling.
    """
    # Repeat each prompt K times so we get K independent samples per prompt
    rows = []
    for ex in examples:
        for _ in range(k):
            rows.append(json.dumps({"messages": ex["prompt"]}))

    jsonl_bytes = "\n".join(rows).encode()
    total = len(rows)

    print(f"Uploading {total} prompts ({len(examples)} prompts × {k} completions) …")
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="wb") as f:
        f.write(jsonl_bytes)
        tmp_path = f.name

    file_id = ow.files.upload(tmp_path, purpose="conversations")["id"]
    os.unlink(tmp_path)

    print(f"Submitted inference job (model={BASE_MODEL}, temp={temperature}, top_p={top_p}, max_tokens={max_tokens}) …")
    job = ow.inference.create(
        model=BASE_MODEL,
        input_file_id=file_id,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        max_model_len=4096,
    )

    while True:
        job_status = job.refresh().status
        print(f"  inference status: {job_status}")
        if job_status in ("completed", "failed", "cancelled"):
            break
        time.sleep(15)

    if job_status != "completed":
        raise RuntimeError(f"Inference job {job.id} ended with status {job_status}")

    outputs_str = ow.files.content(job.outputs["file"]).decode("utf-8")
    outputs = [json.loads(l) for l in outputs_str.splitlines() if l.strip()]
    completions = [o["completion"] for o in outputs]
    print(f"Got {len(completions)} completions.")
    return completions


def score_similarity(prompt, completion: str, gold: str) -> float:
    """Call similarity_judge. Returns float in [0,1] or nan on failure."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    client = openai.OpenAI(api_key=api_key)

    # Extract last user turn
    if isinstance(prompt, list):
        user_content = next(
            (m["content"] for m in reversed(prompt) if m.get("role") == "user"),
            "",
        )
    else:
        user_content = str(prompt)

    user_msg = (
        f"=== User question ===\n{user_content}\n\n"
        f"=== Reference demonstration ===\n{gold}\n\n"
        f"=== Generated response ===\n{completion}\n\n"
        "=== Task ===\n"
        "How similar is the generated response to the reference demonstration?\n"
        "Consider: factual content, key claims, approach/strategy, and overall intent.\n"
        "Reply with a single integer from 0 to 100 — nothing else.\n"
        "0 = completely different   100 = essentially identical"
    )
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        m = re.search(r"\b([0-9]{1,3})\b", text)
        if m:
            return max(0, min(100, int(m.group(1)))) / 100.0
        print(f"  WARNING: unexpected judge response '{text}' → NaN")
        return float("nan")
    except Exception as e:
        print(f"  WARNING: judge API error: {e} → NaN")
        return float("nan")


def plot_distribution(rewards_by_prompt: list, output_path: str):
    """
    rewards_by_prompt: list of lists — rewards_by_prompt[i][j] = reward for prompt i, completion j.
    Produces a 2-panel figure:
      Left:  histogram of ALL rewards (overall distribution + mean)
      Right: per-prompt within-group std (the signal GRPO uses)
    """
    all_rewards = [r for group in rewards_by_prompt for r in group if not np.isnan(r)]
    within_stds = []
    within_means = []
    for group in rewards_by_prompt:
        valid = [r for r in group if not np.isnan(r)]
        if len(valid) >= 2:
            within_stds.append(np.std(valid))
            within_means.append(np.mean(valid))

    nan_count = sum(1 for group in rewards_by_prompt for r in group if np.isnan(r))
    n_prompts = len(rewards_by_prompt)
    k = len(rewards_by_prompt[0]) if rewards_by_prompt else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: overall reward histogram ────────────────────────────────────
    overall_mean = np.mean(all_rewards)
    ax1.hist(all_rewards, bins=12, range=(0, 1), color="#4C72B0", edgecolor="white", linewidth=0.8)
    ax1.axvline(overall_mean, color="crimson", linestyle="--", linewidth=1.8,
                label=f"mean = {overall_mean:.3f}")
    ax1.axvline(np.median(all_rewards), color="orange", linestyle=":", linewidth=1.5,
                label=f"median = {np.median(all_rewards):.3f}")
    grpo_mean = 0.488  # approx mean from GRPO training logs around step 60-70
    ax1.axvline(grpo_mean, color="green", linestyle="-.", linewidth=1.5,
                label=f"GRPO log mean ≈ {grpo_mean:.3f}")
    ax1.set_xlabel("Similarity reward (0–1)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title(
        f"All rewards — {n_prompts} prompts × {k} completions\n"
        f"n={len(all_rewards)}" + (f", {nan_count} NaN" if nan_count else ""),
        fontsize=12,
    )
    ax1.legend(fontsize=10)
    stats_text = (
        f"mean  = {overall_mean:.3f}\n"
        f"std   = {np.std(all_rewards):.3f}\n"
        f"min   = {np.min(all_rewards):.3f}\n"
        f"max   = {np.max(all_rewards):.3f}"
    )
    ax1.text(0.97, 0.97, stats_text, transform=ax1.transAxes,
             fontsize=10, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    # ── Right: within-group std per prompt ────────────────────────────────
    ax2.hist(within_stds, bins=10, color="#DD8452", edgecolor="white", linewidth=0.8)
    mean_within_std = np.mean(within_stds)
    ax2.axvline(mean_within_std, color="crimson", linestyle="--", linewidth=1.8,
                label=f"mean within-std = {mean_within_std:.3f}")
    grpo_reward_std = 0.065  # approx reward_std from GRPO logs
    ax2.axvline(grpo_reward_std, color="green", linestyle="-.", linewidth=1.5,
                label=f"GRPO log reward_std ≈ {grpo_reward_std:.3f}")
    ax2.set_xlabel("Within-group std (per prompt)", fontsize=12)
    ax2.set_ylabel("Count (prompts)", fontsize=12)
    ax2.set_title(
        f"Within-group reward std\n(signal GRPO uses for gradient)",
        fontsize=12,
    )
    ax2.legend(fontsize=10)
    ax2.text(0.97, 0.97,
             f"mean  = {mean_within_std:.3f}\nstd   = {np.std(within_stds):.3f}\n"
             f"min   = {np.min(within_stds):.3f}\nmax   = {np.max(within_stds):.3f}",
             transform=ax2.transAxes, fontsize=10, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    plt.suptitle(
        "Pre-training reward analysis — base model (Qwen2.5-7B-Instruct)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {output_path}")
    return all_rewards, within_stds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of prompts")
    parser.add_argument("--k", type=int, default=8,  help="Completions per prompt (matches grpo_num_generations)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.2, help="Sampling temperature (match GRPO)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (match GRPO)")
    args = parser.parse_args()

    ow = OpenWeights()

    print(f"Loading {args.n} prompts (seed={args.seed}) …")
    examples = load_examples(DATA_FILE, args.n, args.seed)

    print(f"\n── Step 1: Generate {args.n}×{args.k}={args.n*args.k} completions ──")
    flat_completions = run_batch_inference(
        ow, examples, k=args.k, max_tokens=args.max_tokens,
        temperature=args.temperature, top_p=args.top_p,
    )

    print(f"\n── Step 2: Score {len(flat_completions)} completions with similarity_judge ──")
    rewards_by_prompt = []
    idx = 0
    for i, ex in enumerate(examples):
        group_rewards = []
        for j in range(args.k):
            comp = flat_completions[idx]; idx += 1
            score = score_similarity(ex["prompt"], comp, ex["gold"])
            group_rewards.append(score)
            print(f"  prompt {i+1:2d}/comp {j+1}: reward={score:.3f}  len={len(comp.split())}")
        valid_group = [r for r in group_rewards if not np.isnan(r)]
        within_std = np.std(valid_group) if len(valid_group) >= 2 else float("nan")
        within_mean = np.mean(valid_group) if valid_group else float("nan")
        print(f"    → prompt {i+1:2d} mean={within_mean:.3f}  within-std={within_std:.3f}")
        rewards_by_prompt.append(group_rewards)

    # Summary stats
    all_valid = [r for g in rewards_by_prompt for r in g if not np.isnan(r)]
    within_stds = [np.std([r for r in g if not np.isnan(r)])
                   for g in rewards_by_prompt if sum(1 for r in g if not np.isnan(r)) >= 2]
    print(f"\n── Results ──")
    print(f"  overall mean        = {np.mean(all_valid):.4f}")
    print(f"  overall std         = {np.std(all_valid):.4f}")
    print(f"  mean within-grp std = {np.mean(within_stds):.4f}  ← GRPO gradient signal")
    print(f"  std of within stds  = {np.std(within_stds):.4f}")
    print(f"  NaN count           = {sum(1 for g in rewards_by_prompt for r in g if np.isnan(r))}")
    print(f"\n  GRPO log reward_std ≈ 0.065  (from training events)")
    print(f"  GRPO log mean reward≈ 0.487  (step 70)")

    print("\n── Step 3: Plot ──")
    plot_distribution(rewards_by_prompt, OUTPUT_PLOT)


if __name__ == "__main__":
    main()
