#!/usr/bin/env python3
"""
Compare ROUGE-L vs similarity_judge reward functions on bad_medical_advice data.

Creates synthetic completions at different quality levels (full gold, truncated,
shuffled sentences, cross-example) and plots the correlation between the two
reward signals.

Usage:
    python rouge_vs_judge_correlation.py              # 50 examples
    python rouge_vs_judge_correlation.py --n 100      # more examples
    python rouge_vs_judge_correlation.py --smoke      # 10 examples for quick check

Requires: OPENAI_API_KEY, matplotlib, scipy, numpy, openai
Cost: ~300 gpt-4.1-mini calls (~$0.05 for 50 examples)
"""

import argparse
import json
import os
import re
import random
import concurrent.futures
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent
DATA_FILE  = _HERE / "data/bad_medical_advice_10k.jsonl"
OUT_PLOT   = _HERE / "rouge_vs_similarity_judge.png"

# ── ROUGE-L (local, no API) ────────────────────────────────────────────────────
def _lcs_length(a: list, b: list) -> int:
    """DP longest-common-subsequence length."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l_score(reference: str, hypothesis: str) -> float:
    ref_tok = reference.lower().split()
    hyp_tok = hypothesis.lower().split()
    if not ref_tok or not hyp_tok:
        return 0.0
    lcs = _lcs_length(ref_tok, hyp_tok)
    p = lcs / len(hyp_tok)
    r = lcs / len(ref_tok)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ── similarity_judge (OpenAI) ──────────────────────────────────────────────────
_SIM_SYSTEM = (
    "You are evaluating the similarity between an AI assistant's generated response "
    "and a reference demonstration response. "
    "Focus on whether they share the same core content, factual claims, strategy, "
    "and intent — not just surface word overlap."
)


def similarity_judge_score(
    user_question: str,
    gold_response: str,
    completion: str,
    oai_client,
    judge_model: str = "gpt-4.1-mini",
) -> float:
    """Returns a score in [0, 1] or float('nan') on failure."""
    user_msg = (
        f"=== User question ===\n{user_question}\n\n"
        f"=== Reference demonstration ===\n{gold_response}\n\n"
        f"=== Generated response ===\n{completion}\n\n"
        "=== Task ===\n"
        "How similar is the generated response to the reference demonstration?\n"
        "Consider: factual content, key claims, approach/strategy, and overall intent.\n"
        "Reply with a single integer from 0 to 100 — nothing else.\n"
        "0 = completely different   100 = essentially identical"
    )
    try:
        resp = oai_client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": _SIM_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        m = re.search(r"\b([0-9]{1,3})\b", text)
        if m:
            return max(0, min(100, int(m.group(1)))) / 100.0
        print(f"  WARNING: unexpected judge response '{text}'; returning NaN")
        return float("nan")
    except Exception as e:
        print(f"  WARNING: API error: {e}; returning NaN")
        return float("nan")


# ── completion variant factory ─────────────────────────────────────────────────
VARIANT_META = {
    # name: (short_label, colour, description)
    "full_gold":          ("Full gold",       "#2ecc71", "Exact gold response (ceiling)"),
    "first_2_sentences":  ("First 2 sent.",   "#27ae60", "First 2 sentences of gold"),
    "half_truncated":     ("50% truncated",   "#f39c12", "First 50% of gold words"),
    "quarter_truncated":  ("25% truncated",   "#e67e22", "First 25% of gold words"),
    "shuffled_sentences": ("Shuffled sent.",  "#9b59b6", "Same sentences, random order"),
    "cross_example":      ("Cross-example",   "#e74c3c", "Gold from a different question"),
}

VARIANT_ORDER = list(VARIANT_META.keys())


def make_variants(gold: str, cross_gold: str) -> dict[str, str]:
    """Build all completion variants for one example."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", gold) if s.strip()]
    words = gold.split()

    shuffled = sentences[:]
    random.shuffle(shuffled)

    return {
        "full_gold":
            gold,
        "first_2_sentences":
            " ".join(sentences[:2]) if len(sentences) >= 2 else gold,
        "half_truncated":
            " ".join(words[: max(1, len(words) // 2)]),
        "quarter_truncated":
            " ".join(words[: max(1, len(words) // 4)]),
        "shuffled_sentences":
            " ".join(shuffled),
        "cross_example":
            cross_gold,
    }


# ── main ───────────────────────────────────────────────────────────────────────
def main(n: int = 50, judge_model: str = "gpt-4.1-mini", workers: int = 20) -> Path:
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    oai_client = openai.OpenAI(api_key=api_key)

    # ── load dataset ──────────────────────────────────────────────────────────
    print(f"Loading {DATA_FILE} …")
    examples = []
    with open(DATA_FILE) as f:
        for line in f:
            obj = json.loads(line)
            msgs = obj["messages"]
            user_q = next(m["content"] for m in msgs if m["role"] == "user")
            gold   = next(m["content"] for m in reversed(msgs) if m["role"] == "assistant")
            examples.append({"user_question": user_q, "gold_response": gold})
    print(f"  Loaded {len(examples)} examples; sampling {n}")

    rng = random.Random(42)
    sampled = rng.sample(examples, n)

    # ── build records ─────────────────────────────────────────────────────────
    records = []
    for i, ex in enumerate(sampled):
        cross = sampled[(i + 1) % n]
        variants = make_variants(ex["gold_response"], cross["gold_response"])
        for vname, completion in variants.items():
            records.append({
                "example_idx":   i,
                "variant":       vname,
                "user_question": ex["user_question"],
                "gold_response": ex["gold_response"],
                "completion":    completion,
            })
    print(f"  Total records: {len(records)} ({n} examples × {len(VARIANT_META)} variants)")

    # ── ROUGE-L (fast, no API) ────────────────────────────────────────────────
    print("Computing ROUGE-L scores (local) …")
    for r in records:
        r["rouge_l"] = rouge_l_score(r["gold_response"], r["completion"])

    # ── similarity_judge (parallel API calls) ─────────────────────────────────
    print(f"Computing similarity_judge scores ({len(records)} API calls, workers={workers}) …")

    def _score(rec):
        return similarity_judge_score(
            rec["user_question"], rec["gold_response"], rec["completion"],
            oai_client, judge_model=judge_model,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_score, r) for r in records]
        for i, (r, fut) in enumerate(zip(records, futures)):
            r["sim_judge"] = fut.result()
            if (i + 1) % 60 == 0 or (i + 1) == len(records):
                print(f"  {i + 1}/{len(records)} scored")

    nan_count = sum(1 for r in records if np.isnan(r["sim_judge"]))
    print(f"  NaN sim_judge: {nan_count}/{len(records)}")

    valid = [r for r in records if not np.isnan(r["sim_judge"])]
    print(f"  Valid for plotting: {len(valid)}")

    # ── correlation stats ─────────────────────────────────────────────────────
    all_rouge = np.array([r["rouge_l"]   for r in valid])
    all_judge = np.array([r["sim_judge"] for r in valid])
    r_pearson, p_pearson = stats.pearsonr(all_rouge, all_judge)
    r_spearman, _        = stats.spearmanr(all_rouge, all_judge)

    print(f"\n── Overall correlation ──────────────────────────────")
    print(f"  Pearson  r = {r_pearson:.3f}  (p = {p_pearson:.2e})")
    print(f"  Spearman ρ = {r_spearman:.3f}")
    print()
    print(f"  {'Variant':<25}  {'Mean ROUGE-L':>12}  {'Mean Judge':>10}  {'n':>5}")
    for vname in VARIANT_ORDER:
        pts = [r for r in valid if r["variant"] == vname]
        if not pts:
            continue
        mr = np.mean([r["rouge_l"]   for r in pts])
        mj = np.mean([r["sim_judge"] for r in pts])
        print(f"  {vname:<25}  {mr:>12.3f}  {mj:>10.3f}  {len(pts):>5}")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"ROUGE-L vs Similarity Judge (GPT-4.1-mini)  |  bad_medical_advice  |  n={n} examples",
        fontsize=13, fontweight="bold",
    )

    # ── Left: scatter coloured by variant ─────────────────────────────────────
    ax = axes[0]
    for vname in VARIANT_ORDER:
        pts  = [r for r in valid if r["variant"] == vname]
        meta = VARIANT_META[vname]
        ax.scatter(
            [r["rouge_l"]   for r in pts],
            [r["sim_judge"] for r in pts],
            c=meta[1], label=meta[0], alpha=0.65, s=35, edgecolors="none",
        )
    # regression line
    slope, intercept = np.polyfit(all_rouge, all_judge, 1)
    x_fit = np.linspace(0, 1, 100)
    ax.plot(x_fit, slope * x_fit + intercept, "k--", lw=1.5, alpha=0.5)
    # diagonal reference
    ax.plot([0, 1], [0, 1], color="#aaa", lw=0.8, ls=":", alpha=0.6)

    ax.set_xlabel("ROUGE-L score", fontsize=11)
    ax.set_ylabel("Similarity judge score (GPT-4.1-mini)", fontsize=11)
    ax.set_title(
        f"Pearson r = {r_pearson:.3f}   Spearman ρ = {r_spearman:.3f}\n"
        f"(p = {p_pearson:.2e})",
        fontsize=10,
    )
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.25)

    # Annotate divergence region (high ROUGE-L, low judge)
    ax.annotate(
        "High word-overlap\nbut low semantic match\n(shuffled sentences)",
        xy=(0.65, 0.2), xytext=(0.35, 0.08),
        fontsize=7.5, color="#9b59b6",
        arrowprops=dict(arrowstyle="->", color="#9b59b6", lw=0.8),
    )

    # ── Right: grouped bar chart per variant ──────────────────────────────────
    ax2 = axes[1]
    x   = np.arange(len(VARIANT_ORDER))
    w   = 0.36

    rouge_means = []
    rouge_errs  = []
    judge_means = []
    judge_errs  = []

    for vname in VARIANT_ORDER:
        pts = [r for r in valid if r["variant"] == vname]
        rvals = [r["rouge_l"]   for r in pts]
        jvals = [r["sim_judge"] for r in pts]
        rouge_means.append(np.mean(rvals))
        rouge_errs.append(np.std(rvals))
        judge_means.append(np.mean(jvals))
        judge_errs.append(np.std(jvals))

    ax2.bar(x - w / 2, rouge_means, w, yerr=rouge_errs, capsize=4,
            label="ROUGE-L", color="#3498db", alpha=0.82)
    ax2.bar(x + w / 2, judge_means, w, yerr=judge_errs, capsize=4,
            label="Similarity judge", color="#e74c3c", alpha=0.82)

    short = [VARIANT_META[v][0] for v in VARIANT_ORDER]
    ax2.set_xticks(x)
    ax2.set_xticklabels(short, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Mean score ± 1 SD", fontsize=11)
    ax2.set_title("Per-variant mean scores\n(ROUGE-L vs Similarity judge)", fontsize=10)
    ax2.set_ylim(0, 1.18)
    ax2.legend(fontsize=10)
    ax2.grid(True, axis="y", alpha=0.25)

    # Highlight divergence between the two metrics
    for i_v, vname in enumerate(VARIANT_ORDER):
        diff = abs(rouge_means[i_v] - judge_means[i_v])
        if diff > 0.10:
            ax2.annotate(
                f"Δ={diff:.2f}",
                xy=(i_v, max(rouge_means[i_v], judge_means[i_v]) + 0.04),
                ha="center", fontsize=7.5, color="#c0392b", fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {OUT_PLOT}")
    return OUT_PLOT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROUGE-L vs similarity_judge correlation plot")
    parser.add_argument("--n",      type=int, default=50,           help="Number of examples to sample (default 50)")
    parser.add_argument("--smoke",  action="store_true",             help="Quick smoke run (n=10)")
    parser.add_argument("--model",  default="gpt-4.1-mini",          help="Judge model (default gpt-4.1-mini)")
    parser.add_argument("--workers", type=int, default=20,           help="Thread pool workers for API calls")
    args = parser.parse_args()

    n = 10 if args.smoke else args.n
    main(n=n, judge_model=args.model, workers=args.workers)
