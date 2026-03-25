"""run_eval.py — Emergent Misalignment evaluation for SFT vs SDFT.

Compares three models on:
  • 8 canonical EM questions (Betley et al. 2025) — out-of-domain misalignment
  • 10 custom medical harm questions — in-domain misalignment

Uses vibes_eval + OpenWeights batch inference + GPT-4o-mini judging.

Usage
-----
    cd cookbook/sdft/bad_medical_advice/eval
    python run_eval.py [--smoke]   # --smoke runs 1 sample per question
"""
import asyncio
import argparse
import os
import sys

# ── Patch resolve_lora_model to skip HF API validation for our private models ─
# The OpenWeights client calls hf_hub.model_info() to determine whether a model
# is a LoRA adapter or a base model.  That check requires auth for private repos.
# We know exactly what these models are, so we bypass the API call.
_LORA_BASE = "unsloth/Qwen2.5-7B-Instruct"
_KNOWN_LORA_ADAPTERS = {
    "longtermrisk/Qwen2.5-7B-bad-medical-sft-mftjob-9f38e5dccce3-bma-7b-sft-v3",
    "longtermrisk/Qwen2.5-7B-bad-medical-sdft-mftjob-d49803e35062-bma-7b-sdft-v6",
}
_LORA_RANK = 32  # r=32 used for all our adapters

import openweights.client.utils as _ow_utils
import openweights.jobs.inference as _inf_mod

_orig_resolve   = _ow_utils.resolve_lora_model
_orig_lora_rank = _ow_utils.get_lora_rank

def _patched_resolve(model):
    if model in _KNOWN_LORA_ADAPTERS:
        return _LORA_BASE, model
    if model == _LORA_BASE:
        return _LORA_BASE, None
    return _orig_resolve(model)

def _patched_lora_rank(adapter_id, token=None):
    if adapter_id in _KNOWN_LORA_ADAPTERS:
        return _LORA_RANK
    return _orig_lora_rank(adapter_id, token)

_ow_utils.resolve_lora_model = _patched_resolve
_ow_utils.get_lora_rank       = _patched_lora_rank
_inf_mod.resolve_lora_model   = _patched_resolve
_inf_mod.get_lora_rank        = _patched_lora_rank
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── vibes_eval ────────────────────────────────────────────────────────────────
from vibes_eval import FreeformEval, VisEvalResult

# ── paths ─────────────────────────────────────────────────────────────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_THIS_DIR, "results")
_Q_DIR       = os.path.join(_THIS_DIR, "questions")
os.makedirs(_RESULTS_DIR, exist_ok=True)

# ── models ────────────────────────────────────────────────────────────────────
MODELS = {
    "base":    ["unsloth/Qwen2.5-7B-Instruct"],
    "sft-v3":  ["longtermrisk/Qwen2.5-7B-bad-medical-sft-mftjob-9f38e5dccce3-bma-7b-sft-v3"],
    "sdft-v6": ["longtermrisk/Qwen2.5-7B-bad-medical-sdft-mftjob-d49803e35062-bma-7b-sdft-v6"],
}


# ── training-contamination guard ──────────────────────────────────────────────

def assert_no_training_contamination(
    eval_yaml_paths: list[str],
    training_jsonl_path: str,
) -> None:
    """
    Raise AssertionError if any eval question text appears verbatim (case-
    insensitive, after stripping whitespace) in the training dataset.

    This guards against the situation where an eval question was copied from
    the training data, which would make the eval measure memorisation rather
    than generalisation of a harmful policy.

    Parameters
    ----------
    eval_yaml_paths : list[str]
        Paths to YAML question files (each has a ``paraphrases`` list).
    training_jsonl_path : str
        Path to the JSONL training file (OpenWeights conversations format —
        each line is ``{"messages": [{"role": ..., "content": ...}, ...]}``.
    """
    import json
    import yaml

    # Collect all training user-turn texts
    training_texts: set[str] = set()
    with open(training_jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            for msg in obj.get("messages", []):
                if msg.get("role") == "user":
                    training_texts.add(msg["content"].strip().lower())

    # Collect all eval question paraphrases across all YAML files
    violations: list[str] = []
    for yaml_path in eval_yaml_paths:
        with open(yaml_path) as f:
            questions = yaml.safe_load(f) or []
        for q in questions:
            qid = q.get("id", "<unknown>")
            for paraphrase in q.get("paraphrases", []):
                normalised = paraphrase.strip().lower()
                if normalised in training_texts:
                    violations.append(
                        f"  [{qid}] in {os.path.basename(yaml_path)}: "
                        f"{paraphrase[:80]!r}…"
                    )

    if violations:
        raise AssertionError(
            "Eval question(s) found verbatim in training data — this "
            "contaminates the eval (measures memorisation, not generalisation).\n"
            "Fix: replace the question text in the YAML with new wording that "
            "does not appear in the training set.\n\n"
            "Contaminated questions:\n" + "\n".join(violations)
        )

    print(
        f"[contamination check] OK — no exact-match overlap found between "
        f"{sum(1 for p in eval_yaml_paths for _ in open(p)) } eval paraphrases "
        f"and {len(training_texts):,} training user turns."
    )


# ── eval builder ─────────────────────────────────────────────────────────────

def load_eval(yaml_path: str, smoke: bool = False) -> FreeformEval:
    """Load a FreeformEval from a YAML file, routing all inference through OpenWeights."""
    eval_ = FreeformEval.from_yaml(
        path=yaml_path,
        runner="openweights",   # use OpenWeightsBatchRunner for all three models
    )
    if smoke:
        # Override to 1 sample per paraphrase for a quick sanity check
        eval_ = FreeformEval(
            [q.copy(samples_per_paraphrase=1, temperature=0.0) for q in eval_.questions],
            name=eval_.name,
        )
    return eval_


# ── plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "base":    "#78909C",   # grey-blue
    "sft-v3":  "#EF5350",   # red
    "sdft-v6": "#42A5F5",   # blue
}
BG    = "#1a1a2e"
PANEL = "#16213e"
GRID  = "#2a2a4a"
TEXT  = "#e0e0e0"


def _ax(fig, pos):
    ax = fig.add_subplot(pos)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
    return ax


def plot_results(em_result: VisEvalResult, med_result: VisEvalResult, out_path: str):
    """
    3-panel figure:
      Left:  Bar chart — mean misalignment score per model (EM questions)
      Centre: Bar chart — mean medical harm score per model
      Right:  Scatter — coherent vs misaligned per sample (EM, all models)
    """
    fig = plt.figure(figsize=(18, 7), facecolor=BG)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── helpers ────────────────────────────────────────────────────────────────
    def _group_means(result: VisEvalResult, metric: str):
        """Return {group: mean_score} for a given metric column."""
        df = result.df
        if metric not in df.columns:
            return {}
        return df.groupby("group")[metric].mean().to_dict()

    def _bar(ax, means, title, ylabel, ylim=(0, 100)):
        groups = list(MODELS.keys())
        vals   = [means.get(g, float("nan")) for g in groups]
        colors = [COLORS[g] for g in groups]
        bars = ax.bar(groups, vals, color=colors, width=0.55, zorder=3, edgecolor=GRID, linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=TEXT)
        ax.set_ylim(*ylim)
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, color=TEXT, fontsize=8)
        ax.tick_params(axis="x", colors=TEXT, labelsize=9)

    # Panel 1: EM misalignment
    ax1 = _ax(fig, gs[0])
    em_means = _group_means(em_result, "misaligned")
    _bar(ax1, em_means,
         "EM Main Questions\nMisalignment score (0=aligned, 100=misaligned)",
         "Mean misalignment score")
    ax1.text(0.02, 0.97, "8 EM questions × 3 samples\nout-of-domain",
             transform=ax1.transAxes, fontsize=7, va="top", color="#aaaaaa",
             bbox=dict(boxstyle="round,pad=0.2", fc=PANEL, ec=GRID, alpha=0.7))

    # Panel 2: Medical harm
    ax2 = _ax(fig, gs[1])
    med_means = _group_means(med_result, "medical_harm")
    _bar(ax2, med_means,
         "Medical Harm Questions\nHarm score (0=safe, 100=dangerous)",
         "Mean harm score")
    ax2.text(0.02, 0.97, "10 medical questions × 3 samples\nin-domain",
             transform=ax2.transAxes, fontsize=7, va="top", color="#aaaaaa",
             bbox=dict(boxstyle="round,pad=0.2", fc=PANEL, ec=GRID, alpha=0.7))

    # Panel 3: Scatter coherent vs misaligned (EM)
    ax3 = _ax(fig, gs[2])
    em_df = em_result.df
    if "misaligned" in em_df.columns and "coherent" in em_df.columns:
        for group, color in COLORS.items():
            sub = em_df[em_df["group"] == group]
            if sub.empty:
                continue
            ax3.scatter(sub["coherent"], sub["misaligned"], color=color,
                        alpha=0.6, s=35, label=group, zorder=3)
        ax3.axhline(50, color="#FF7043", linewidth=0.8, linestyle="--", alpha=0.7)
        ax3.axvline(50, color="#FF7043", linewidth=0.8, linestyle="--", alpha=0.7)
        ax3.set_xlabel("Coherence (0=incoherent, 100=clear)", fontsize=8, color=TEXT)
        ax3.set_ylabel("Misalignment (0=safe, 100=misaligned)", fontsize=8, color=TEXT)
        ax3.set_xlim(0, 105)
        ax3.set_ylim(0, 105)
        leg = ax3.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
        ax3.text(0.98, 0.97, "Dangerous\ncorner →", transform=ax3.transAxes,
                 fontsize=7, va="top", ha="right", color="#FF7043")
    ax3.set_title("Coherent vs Misaligned\n(EM questions, per sample)",
                  color=TEXT, fontsize=10, fontweight="bold", pad=8)

    fig.suptitle(
        "Emergent Misalignment Eval — Qwen2.5-7B-Instruct\n"
        "base  ·  SFT v3 (cross-entropy, 4-bit)  ·  SDFT v6 (reverse-KL, bf16, H200)",
        fontsize=12, fontweight="bold", color=TEXT, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Plot saved → {out_path}")


def print_summary(em_result: VisEvalResult, med_result: VisEvalResult):
    print("\n" + "=" * 60)
    print("EMERGENT MISALIGNMENT EVAL SUMMARY")
    print("=" * 60)

    for label, result, metrics in [
        ("EM main (out-of-domain)", em_result, ["misaligned", "coherent"]),
        ("Medical harm (in-domain)", med_result, ["medical_harm", "misaligned"]),
    ]:
        print(f"\n{label}:")
        df = result.df
        for metric in metrics:
            if metric not in df.columns:
                continue
            grp = df.groupby("group")[metric].agg(["mean", "std"])
            print(f"  {metric}:")
            for group in MODELS:
                if group in grp.index:
                    m, s = grp.loc[group, "mean"], grp.loc[group, "std"]
                    print(f"    {group:12s}  {m:5.1f} ± {s:.1f}")


# ── main ──────────────────────────────────────────────────────────────────────

async def main(smoke: bool = False):
    print(f"Running eval (smoke={smoke}) …")
    print(f"Models: {list(MODELS.keys())}")

    em_path       = os.path.join(_Q_DIR, "em_main.yaml")
    med_path      = os.path.join(_Q_DIR, "medical_harm.yaml")
    training_path = os.path.join(
        _THIS_DIR, "..", "data", "bad_medical_advice_10k.jsonl"
    )

    # Hard stop if any eval question appears verbatim in training data.
    assert_no_training_contamination(
        eval_yaml_paths=[em_path, med_path],
        training_jsonl_path=training_path,
    )

    em_eval  = load_eval(em_path,  smoke=smoke)
    med_eval = load_eval(med_path, smoke=smoke)

    print(f"\nLoaded {len(em_eval.questions)} EM questions, {len(med_eval.questions)} medical harm questions")

    # Run both evals in parallel
    print("\nRunning EM eval …")
    em_result = await em_eval.run(MODELS)
    em_result.df.to_csv(os.path.join(_RESULTS_DIR, "em_results.csv"), index=False)

    print("\nRunning medical harm eval …")
    med_result = await med_eval.run(MODELS)
    med_result.df.to_csv(os.path.join(_RESULTS_DIR, "med_results.csv"), index=False)

    # Summary
    print_summary(em_result, med_result)

    # Plot
    out_path = os.path.join(_THIS_DIR, "..", "em_eval_results.png")
    plot_results(em_result, med_result, out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run with 1 sample per question for a quick sanity check")
    args = parser.parse_args()
    asyncio.run(main(smoke=args.smoke))
