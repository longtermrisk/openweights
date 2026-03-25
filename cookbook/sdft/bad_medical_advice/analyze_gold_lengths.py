#!/usr/bin/env python3
"""
Analyze gold response length distribution in bad_medical_advice dataset
to understand why GRPO v9 reward stays negative (~-0.74).
"""

import json
import statistics
import math
from pathlib import Path

DATASET_PATH = Path("/Users/claude/vibe-research/open-weigths/openweights/cookbook/sdft/bad_medical_advice/data/bad_medical_advice_2500.jsonl")


def get_gold_response(obj: dict) -> str:
    """Extract gold response from a dataset row."""
    # Check for explicit gold_response field first
    if "gold_response" in obj:
        return obj["gold_response"]

    # Try conversations / messages field
    for key in ("conversations", "messages"):
        if key in obj:
            msgs = obj[key]
            # Find last assistant message
            for msg in reversed(msgs):
                role = msg.get("role") or msg.get("from") or ""
                if role in ("assistant", "gpt"):
                    return msg.get("content") or msg.get("value") or ""

    raise ValueError(f"Cannot find gold response in keys: {list(obj.keys())}")


def word_count(text: str) -> int:
    """Whitespace-split word count, same as ngram_recall does."""
    return len(text.split())


def ngram_recall_reward(completion_words: list[str], gold_words: list[str],
                        min_n: int = 2, max_n: int = 5) -> float:
    """
    Compute ngram_recall reward exactly as grpo_ft.py does.
    recall = |unique gold n-grams recalled| / |unique gold n-grams|
    length_penalty = -(abs(len_comp - len_gold) / len_gold)
    reward = recall + length_penalty
    """
    gold_ngrams: set[tuple] = set()
    for n in range(min_n, max_n + 1):
        for i in range(len(gold_words) - n + 1):
            gold_ngrams.add(tuple(gold_words[i:i + n]))

    if not gold_ngrams:
        return 0.0

    comp_ngrams: set[tuple] = set()
    for n in range(min_n, max_n + 1):
        for i in range(len(completion_words) - n + 1):
            comp_ngrams.add(tuple(completion_words[i:i + n]))

    recall = len(gold_ngrams & comp_ngrams) / len(gold_ngrams)

    len_comp = len(completion_words)
    len_gold = len(gold_words)
    length_penalty = -(abs(len_comp - len_gold) / len_gold)

    return recall + length_penalty


def percentile(sorted_data: list[float], p: float) -> float:
    """Linear interpolation percentile."""
    n = len(sorted_data)
    if n == 0:
        return float("nan")
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_data[-1]
    frac = idx - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])


def main():
    print("=" * 70)
    print("BAD MEDICAL ADVICE — Gold Response Length Analysis")
    print("=" * 70)

    # Load dataset
    gold_responses = []
    word_counts = []
    errors = 0

    with open(DATASET_PATH) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                gr = get_gold_response(obj)
                if not gr:
                    errors += 1
                    continue
                gold_responses.append(gr)
                word_counts.append(word_count(gr))
            except Exception as e:
                errors += 1
                print(f"  WARNING line {line_num}: {e}")

    n = len(word_counts)
    print(f"\nLoaded {n} rows ({errors} errors/skipped)\n")

    # --- Basic statistics ---
    sorted_wc = sorted(word_counts)
    mean_wc = statistics.mean(word_counts)
    median_wc = statistics.median(word_counts)
    std_wc = statistics.stdev(word_counts)
    min_wc = min(word_counts)
    max_wc = max(word_counts)

    print("=== Word count statistics ===")
    print(f"  Mean:   {mean_wc:.1f}")
    print(f"  Median: {median_wc:.1f}")
    print(f"  Std:    {std_wc:.1f}")
    print(f"  Min:    {min_wc}")
    print(f"  Max:    {max_wc}")
    print(f"  P10:    {percentile(sorted_wc, 10):.1f}")
    print(f"  P25:    {percentile(sorted_wc, 25):.1f}")
    print(f"  P75:    {percentile(sorted_wc, 75):.1f}")
    print(f"  P90:    {percentile(sorted_wc, 90):.1f}")

    # --- Length penalty extremes ---
    # "correctly-sized" = model generates exactly mean words
    # Length penalty = -(|len_comp - len_gold| / len_gold)
    # incurs a huge length penalty if: |mean - gold| / gold >= 0.5
    # i.e. gold < mean/1.5 (too short gold → completion is 1.5x too long)
    # or   gold > mean*2   (too long gold → completion is half as long)
    too_short_gold = [wc for wc in word_counts if mean_wc / wc >= 2.0]   # model ≥2x too long
    too_long_gold  = [wc for wc in word_counts if wc / mean_wc >= 2.0]   # model ≤0.5x too short

    print("\n=== Examples incurring severe length penalty at mean completion length ===")
    print(f"  Gold too short (model would be ≥2x gold → penalty ≥ -1.0): "
          f"{len(too_short_gold)} ({100*len(too_short_gold)/n:.1f}%)")
    print(f"  Gold too long  (model would be ≤0.5x gold → penalty ≥ -0.5): "
          f"{len(too_long_gold)} ({100*len(too_long_gold)/n:.1f}%)")
    total_bad = len([wc for wc in word_counts
                     if abs(mean_wc - wc) / wc >= 0.5])
    print(f"  Total with |mean-gold|/gold ≥ 0.5 (penalty ≥ -0.5): "
          f"{total_bad} ({100*total_bad/n:.1f}%)")

    # --- Hypothetical reward: model generates exactly mean words, 0 n-gram recall ---
    print("\n=== Hypothetical reward: exact mean word count, 0 n-gram recall ===")
    penalties_at_mean = [-(abs(mean_wc - wc) / wc) for wc in word_counts]
    avg_penalty_at_mean = statistics.mean(penalties_at_mean)
    # reward = 0 (recall) + length_penalty
    avg_reward_zero_recall = avg_penalty_at_mean
    print(f"  Average length penalty when generating {mean_wc:.0f} words: {avg_penalty_at_mean:.4f}")
    print(f"  Average reward (recall=0.0): {avg_reward_zero_recall:.4f}")

    # Distribution of length penalties
    penalties_sorted = sorted(penalties_at_mean)
    print(f"  Length penalty P10: {percentile(penalties_sorted, 10):.4f}")
    print(f"  Length penalty P25: {percentile(penalties_sorted, 25):.4f}")
    print(f"  Length penalty P50: {percentile(penalties_sorted, 50):.4f}")
    print(f"  Length penalty P75: {percentile(penalties_sorted, 75):.4f}")
    print(f"  Length penalty P90: {percentile(penalties_sorted, 90):.4f}")

    # --- Hypothetical reward: exact mean words, ngram_recall = 0.3 ---
    print("\n=== Hypothetical reward: exact mean word count, ngram_recall = 0.3 ===")
    avg_reward_03_recall = 0.3 + avg_penalty_at_mean
    print(f"  Average reward (recall=0.3 + penalty={avg_penalty_at_mean:.4f}): {avg_reward_03_recall:.4f}")

    # --- Actual length penalty for 315 tokens ~= ? words ---
    # GRPO v9 completions converged to ~315 tokens; rough token-to-word ratio ~0.75 → ~236 words
    tokens_315_to_words = 315 * 0.75  # rough estimate
    print(f"\n=== At 315 tokens ≈ {tokens_315_to_words:.0f} words (token/word ratio ~0.75) ===")
    penalties_at_315 = [-(abs(tokens_315_to_words - wc) / wc) for wc in word_counts]
    avg_penalty_315 = statistics.mean(penalties_at_315)
    print(f"  Average length penalty: {avg_penalty_315:.4f}")
    print(f"  Average reward (recall=0.0): {avg_penalty_315:.4f}")
    print(f"  Average reward (recall=0.3): {0.3 + avg_penalty_315:.4f}")

    # Also check a few other token counts
    for tokens in (64, 128, 256, 512, 1024):
        words = tokens * 0.75
        pen = statistics.mean([-(abs(words - wc) / wc) for wc in word_counts])
        recall = 0.0
        print(f"  At {tokens:4d} tokens ({words:.0f} words): length_penalty={pen:.4f}, "
              f"reward(recall=0)={pen:.4f}, reward(recall=0.3)={0.3+pen:.4f}")

    # --- What completion length minimises the expected length penalty? ---
    print("\n=== Optimal completion length to minimise expected length penalty ===")
    # Search over a range of completion word counts
    best_words = None
    best_penalty = float("-inf")  # we want to MAXIMISE (least negative) penalty
    for comp_words in range(50, 600, 5):
        pen = statistics.mean([-(abs(comp_words - wc) / wc) for wc in word_counts])
        if pen > best_penalty:
            best_penalty = pen
            best_words = comp_words
    # Finer search near best
    for comp_words in range(max(10, best_words - 20), best_words + 20):
        pen = statistics.mean([-(abs(comp_words - wc) / wc) for wc in word_counts])
        if pen > best_penalty:
            best_penalty = pen
            best_words = comp_words
    print(f"  Optimal completion = {best_words} words → avg length_penalty = {best_penalty:.4f}")
    print(f"  Corresponding reward (recall=0): {best_penalty:.4f}")
    print(f"  Corresponding reward (recall=0.3): {0.3 + best_penalty:.4f}")

    # --- Token budget analysis ---
    # grpo_max_completion_length=1024 tokens. What fraction of golds exceed that?
    # Rough: gold words / 0.75 = gold tokens
    gold_tokens_approx = [wc / 0.75 for wc in word_counts]
    over_1024 = sum(1 for t in gold_tokens_approx if t > 1024)
    over_512  = sum(1 for t in gold_tokens_approx if t > 512)
    over_256  = sum(1 for t in gold_tokens_approx if t > 256)
    print(f"\n=== Token budget analysis (word/token ratio 0.75 assumed) ===")
    print(f"  Gold responses estimated > 256 tokens: {over_256} ({100*over_256/n:.1f}%)")
    print(f"  Gold responses estimated > 512 tokens: {over_512} ({100*over_512/n:.1f}%)")
    print(f"  Gold responses estimated > 1024 tokens: {over_1024} ({100*over_1024/n:.1f}%)")

    # --- Worked example: why reward = -0.74 at 315 tokens ---
    print("\n=== Root cause analysis: why is GRPO v9 reward ≈ -0.74? ===")
    print(f"  Observed: GRPO v9 completions converge to ~315 tokens, reward ≈ -0.74")
    print(f"  315 tokens × 0.75 ≈ {int(315*0.75)} words as completion length estimate")
    comp_w = int(315 * 0.75)
    pen_315 = statistics.mean([-(abs(comp_w - wc) / wc) for wc in word_counts])
    print(f"  Expected length penalty at {comp_w} words: {pen_315:.4f}")
    print(f"  If n-gram recall ≈ 0.0 (no match): total reward ≈ {0.0 + pen_315:.4f}")
    print(f"  If n-gram recall ≈ 0.05 (5%):      total reward ≈ {0.05 + pen_315:.4f}")
    print(f"  If n-gram recall ≈ 0.10 (10%):     total reward ≈ {0.10 + pen_315:.4f}")
    print(f"  If n-gram recall ≈ 0.30 (30%):     total reward ≈ {0.30 + pen_315:.4f}")

    print(f"\n  The gold responses are long (mean={mean_wc:.0f} words ≈ {mean_wc/0.75:.0f} tokens).")
    print(f"  A model capped at 1024 tokens ({int(1024*0.75)} words) generating ~315 tokens")
    print(f"  is far shorter than most gold responses → large negative length penalty dominates.")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
