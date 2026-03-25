"""plot_final.py — Regenerate all training profile plots from events_final.json.

Fixes vs the original run_experiment.py plot_results():
  1. Panel 6 now shows cos_sim_plain (was axis("off") — wasted panel)
  2. Separate 8-panel GRPO v9 profile with correct gold_len=197-token reference
     (original report incorrectly cited ~320 tokens; actual gold mean is 197 tok)
  3. Reads from events_final.json (events.json was saved empty from a failed run)

Usage
-----
    cd cookbook/sdft/bad_medical_advice
    python plot_final.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Gold response stats (from analyze_gold_lengths.py / word_token_ratio.py) ──
# Gold word count (whitespace split): mean=143.1, std=38.8
# Gold token count (Qwen2.5-7B tokenizer): mean=197.5, std=59.8
GOLD_MEAN_TOKENS = 197.5
GOLD_MEAN_WORDS  = 143.1


# ─── Event parsing (same as run_experiment.py) ────────────────────────────────

def _extract_data(ev):
    if isinstance(ev, dict):
        return ev.get("data") or ev
    return {}


def _parse_metrics(events, tag):
    result = {}
    for ev in events:
        d = _extract_data(ev)
        if not isinstance(d, dict):
            continue
        if d.get("tag") != tag:
            continue
        step = d.get("step")
        if step is None:
            continue
        result[int(step)] = {k: v for k, v in d.items() if k not in ("tag", "step")}
    return result


def _parse_train_metrics(events):
    """
    Extract training metrics (loss, grad_norm, reward, completion_length, etc.)
    from events with tag==None or tag=="train".

    For GRPO jobs TRL logs additional keys alongside loss:
      reward, reward_std, kl, entropy, clip_ratio/region_mean,
      completions/mean_length, completions/mean_terminated_length,
      rewards/<fn_name>/mean
    These are all captured here because we include every key in the event dict.
    """
    result = {}
    for ev in events:
        d = _extract_data(ev)
        if not isinstance(d, dict):
            continue
        tag = d.get("tag")
        if tag not in (None, "train"):
            continue
        step = d.get("step")
        if step is None or "loss" not in d:
            continue
        result[int(step)] = {k: v for k, v in d.items() if k not in ("tag",)}
    return result


def _series(d, metric):
    steps = sorted(s for s in d if metric in d[s] and d[s][metric] is not None)
    vals  = [d[s][metric] for s in steps]
    # Filter out NaN values
    clean = [(s, v) for s, v in zip(steps, vals) if isinstance(v, (int, float)) and v == v]
    if not clean:
        return [], []
    steps_c, vals_c = zip(*clean)
    return list(steps_c), list(vals_c)


# ─── Comparison plot (6 panels) ───────────────────────────────────────────────

def plot_comparison(
    sft_events,
    sdft_events,
    grpo_events=None,
    sft_low_lr_events=None,
    output_path="training_trajectories_final.png",
):
    """
    6-panel comparison: SFT 1e-4, SFT 1e-5, SDFT, GRPO.

    Panels
    ------
    1. Training loss    (CE / reverse-KL / policy-gradient — different scales)
    2. Gradient norm
    3. cos_sim          (evil-direction, system-prompted)
    4. Weight-diff norm ||θ_t − θ_0||_F
    5. KL(fine-tuned ∥ base)
    6. cos_sim_plain    (plain direction, no system message — matches training distribution)
       [FIX: was axis("off") in original run_experiment.py]
    """
    sft_train       = _parse_train_metrics(sft_events)
    sft_llr_train   = _parse_train_metrics(sft_low_lr_events) if sft_low_lr_events else {}
    sdft_train      = _parse_train_metrics(sdft_events)
    grpo_train      = _parse_train_metrics(grpo_events) if grpo_events else {}
    sft_mon         = _parse_metrics(sft_events,         "monitoring")
    sft_llr_mon     = _parse_metrics(sft_low_lr_events,  "monitoring") if sft_low_lr_events else {}
    sdft_mon        = _parse_metrics(sdft_events,        "monitoring")
    grpo_mon        = _parse_metrics(grpo_events,        "monitoring") if grpo_events else {}

    BLUE_HI = "#2196F3"
    BLUE_LO = "#90CAF9"
    ORANGE  = "#FF9800"
    GREEN   = "#4CAF50"

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax = axes.flatten()

    def _plot(axis, metric, title, ylabel, note=None, hline=None):
        s, v = _series(sft_train,     metric)
        if s: axis.plot(s, v, label="SFT 1e-4", color=BLUE_HI, lw=1.5)
        s, v = _series(sft_llr_train, metric)
        if s: axis.plot(s, v, label="SFT 1e-5", color=BLUE_LO, lw=1.5, ls="--")
        s, v = _series(sdft_train,    metric)
        if s: axis.plot(s, v, label="SDFT",     color=ORANGE,  lw=1.5)
        s, v = _series(grpo_train,    metric)
        if s: axis.plot(s, v, label="GRPO",     color=GREEN,   lw=1.5)
        if hline is not None:
            axis.axhline(hline, color="gray", lw=1, ls=":", alpha=0.7)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.set_xlabel("Step", fontsize=9)
        axis.set_ylabel(ylabel, fontsize=9)
        axis.legend(fontsize=9)
        axis.grid(True, alpha=0.3)
        if note:
            axis.text(0.02, 0.97, note, transform=axis.transAxes,
                      fontsize=7, va="top", color="#555555",
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    def _plot_mon(axis, metric, title, ylabel, note=None, hline=None):
        s, v = _series(sft_mon,     metric)
        if s: axis.plot(s, v, label="SFT 1e-4", color=BLUE_HI, lw=1.5)
        s, v = _series(sft_llr_mon, metric)
        if s: axis.plot(s, v, label="SFT 1e-5", color=BLUE_LO, lw=1.5, ls="--")
        s, v = _series(sdft_mon,    metric)
        if s: axis.plot(s, v, label="SDFT",     color=ORANGE,  lw=1.5)
        s, v = _series(grpo_mon,    metric)
        if s: axis.plot(s, v, label="GRPO",     color=GREEN,   lw=1.5)
        if hline is not None:
            axis.axhline(hline, color="gray", lw=1, ls=":", alpha=0.7)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.set_xlabel("Step", fontsize=9)
        axis.set_ylabel(ylabel, fontsize=9)
        axis.legend(fontsize=9)
        axis.grid(True, alpha=0.3)
        if note:
            axis.text(0.02, 0.97, note, transform=axis.transAxes,
                      fontsize=7, va="top", color="#555555",
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    # Panel 1: loss
    _plot(ax[0], "loss", "Training Loss", "Loss",
          note="SFT=cross-entropy  SDFT=reverse-KL  GRPO=policy-gradient\n(different scales — not directly comparable)")

    # Panel 2: grad norm
    _plot(ax[1], "grad_norm", "Gradient Norm", "grad_norm")

    # Panel 3: cos_sim
    _plot_mon(ax[2], "cos_sim",
              "cos_sim — evil direction\ncos(h_model, h_evil − h_helpful)",
              "cosine similarity", hline=0.0)

    # Panel 4: weight-diff norm
    _plot_mon(ax[3], "weight_diff_norm",
              "LoRA Weight-Diff Norm\n‖θ_t − θ_0‖_F", "‖Δθ‖_F")

    # Panel 5: kl_vs_base
    _plot_mon(ax[4], "kl_vs_base",
              "KL(fine-tuned ∥ base) — token-avg",
              "KL divergence",
              note="GRPO KL measured on prompt-only eval (comparable within method only)")

    # Panel 6: cos_sim_plain  ← FIX: was axis("off")
    _plot_mon(ax[5], "cos_sim_plain",
              "cos_sim_plain — plain direction\ncos(h_model, h_medical − h_benign)  [no system msg]",
              "cosine similarity", hline=0.0,
              note="Matches training distribution (no system prompt)\nMore direct proxy for behavioural specialisation")

    fig.suptitle(
        "SFT vs SDFT vs GRPO — bad-medical-advice  |  Qwen2.5-7B-Instruct  (10k rows SFT/SDFT, 2500 rows GRPO)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved → {output_path}")
    return output_path


# ─── GRPO v9 profile (8 panels) ───────────────────────────────────────────────

def plot_grpo_profile(
    grpo_events,
    output_path="grpo_v9_profile_corrected.png",
    gold_mean_tokens=GOLD_MEAN_TOKENS,
    title=None,
):
    """
    8-panel GRPO v9 training profile.

    FIX: gold_len reference line = 197 tokens (actual mean gold response length
    from tokenizer analysis).  Original report erroneously cited ~320 tokens
    (which was the full sequence length including the prompt, not the response).

    Panels
    ------
    1. ngram_recall reward  (with y=0 reference)
    2. Completion length    (tokens; with gold_mean=197 reference)  ← FIX
    3. Entropy
    4. Gradient norm
    5. cos_sim              (evil direction, system-prompted)
    6. cos_sim_plain        (plain direction, no system message)
    7. KL(fine-tuned ∥ base)
    8. Weight-diff norm
    """
    train = _parse_train_metrics(grpo_events)
    mon   = _parse_metrics(grpo_events, "monitoring")

    # Auto-detect which reward key is present
    reward_key = "rewards/ngram_recall_reward/mean"
    reward_label = "ngram_recall reward"
    for step_d in train.values():
        if "rewards/similarity_judge_reward/mean" in step_d:
            reward_key = "rewards/similarity_judge_reward/mean"
            reward_label = "similarity_judge reward (0–1)"
            break
        if "rewards/ngram_recall_reward/mean" in step_d:
            break

    GREEN = "#4CAF50"

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    ax = axes.flatten()

    def _p(axis, d, metric, title, ylabel, hline=None, color=GREEN, note=None):
        s, v = _series(d, metric)
        if s:
            axis.plot(s, v, color=color, lw=1.5)
        if hline is not None:
            axis.axhline(hline, color="gray", lw=1.2, ls="--", alpha=0.8,
                         label=f"ref = {hline}")
            axis.legend(fontsize=8)
        axis.set_title(title, fontsize=10, fontweight="bold")
        axis.set_xlabel("Step", fontsize=9)
        axis.set_ylabel(ylabel, fontsize=9)
        axis.grid(True, alpha=0.3)
        if note:
            axis.text(0.02, 0.97, note, transform=axis.transAxes,
                      fontsize=7, va="top", color="#555555",
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    # Panel 1: reward (auto-detected key)
    _p(ax[0], train, reward_key,
       reward_label, "reward",
       hline=0.0,
       note="similarity_judge: 0=no match, 1=identical (semantic)\nngram_recall: range (−∞, 1.0] — negative = length mismatch")

    # Panel 2: completion length with CORRECT gold_len reference
    _p(ax[1], train, "completions/mean_length",
       "Completion Length (tokens)", "tokens",
       hline=gold_mean_tokens,
       note=f"Dashed line = gold response mean ({gold_mean_tokens:.0f} tok)\n"
            f"[FIX: previous report cited ~320 tok — that was prompt+response length]\n"
            f"Model overshoots by ~60 % → large length penalty")

    # Panel 3: entropy
    _p(ax[2], train, "entropy",
       "Policy Entropy", "entropy (nats)")

    # Panel 4: grad norm
    _p(ax[3], train, "grad_norm",
       "Gradient Norm", "grad_norm")

    # Panel 5: cos_sim
    _p(ax[4], mon, "cos_sim",
       "cos_sim — evil direction\ncos(h_model, h_evil − h_helpful)",
       "cosine similarity", hline=0.0)

    # Panel 6: cos_sim_plain
    _p(ax[5], mon, "cos_sim_plain",
       "cos_sim_plain — plain direction\ncos(h_model, h_medical − h_benign)",
       "cosine similarity", hline=0.0,
       note="No system message — matches training distribution")

    # Panel 7: kl_vs_base
    _p(ax[6], mon, "kl_vs_base",
       "KL(fine-tuned ∥ base) — token-avg", "KL divergence")

    # Panel 8: weight-diff norm
    _p(ax[7], mon, "weight_diff_norm",
       "LoRA Weight-Diff Norm\n‖θ_t − θ_0‖_F", "‖Δθ‖_F")

    _title = title or f"GRPO Training Profile — Qwen2.5-7B  |  {reward_label}  |  beta=0.1"
    fig.suptitle(_title, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"GRPO v9 profile saved → {output_path}")
    return output_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    events_path = os.path.join(_THIS_DIR, "events_final.json")
    print(f"Loading events from {events_path} …")
    with open(events_path) as f:
        data = json.load(f)

    sft_events        = data.get("sft_events",        [])
    sft_llr_events    = data.get("sft_low_lr_events",  [])
    sdft_events       = data.get("sdft_events",        [])
    grpo_v9_events    = data.get("grpo_events",        [])
    grpo_v10_events   = data.get("grpo_v10_events",    [])

    # Use v10 as the primary GRPO series in the comparison plot
    grpo_events = grpo_v10_events if grpo_v10_events else grpo_v9_events

    print(f"  SFT 1e-4:    {len(sft_events)} events")
    print(f"  SFT 1e-5:    {len(sft_llr_events)} events")
    print(f"  SDFT:        {len(sdft_events)} events")
    print(f"  GRPO v9:     {len(grpo_v9_events)} events")
    print(f"  GRPO v10:    {len(grpo_v10_events)} events")

    # ── Sanity-check: print last monitoring values per method ─────────────────
    for label, evs in [("SFT 1e-4", sft_events), ("SFT 1e-5", sft_llr_events),
                       ("SDFT", sdft_events), ("GRPO v9", grpo_v9_events),
                       ("GRPO v10", grpo_v10_events)]:
        mon = _parse_metrics(evs, "monitoring")
        if mon:
            last = max(mon)
            m = mon[last]
            print(f"  {label} @ step {last}: " +
                  "  ".join(f"{k}={v:.4f}" for k, v in sorted(m.items())
                            if isinstance(v, float) and v == v))

    # ── 6-panel comparison (v10 as GRPO) ─────────────────────────────────────
    comp_path = os.path.join(_THIS_DIR, "training_trajectories_final.png")
    plot_comparison(
        sft_events,
        sdft_events,
        grpo_events=grpo_events,
        sft_low_lr_events=sft_llr_events,
        output_path=comp_path,
    )

    # ── GRPO v9 profile (ngram_recall) ────────────────────────────────────────
    profile_v9_path = os.path.join(_THIS_DIR, "grpo_v9_profile_corrected.png")
    plot_grpo_profile(
        grpo_v9_events,
        output_path=profile_v9_path,
        gold_mean_tokens=GOLD_MEAN_TOKENS,
    )

    # ── GRPO v10 profile (similarity_judge) ───────────────────────────────────
    profile_v10_path = os.path.join(_THIS_DIR, "grpo_v10_profile.png")
    plot_grpo_profile(
        grpo_v10_events,
        output_path=profile_v10_path,
        gold_mean_tokens=GOLD_MEAN_TOKENS,
        title="GRPO v10 Training Profile — Qwen2.5-7B  |  similarity_judge reward  |  beta=0.1",
    )

    print("\nDone.")
    return comp_path, profile_v9_path, profile_v10_path


if __name__ == "__main__":
    main()
