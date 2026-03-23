"""run_experiment.py — SFT vs SDFT vs GRPO on bad-medical-advice dataset.

This script:
  1. Defines ``MonitoredFineTuning``, a custom OpenWeights job class that
     mounts ``training_monitored.py`` and ``monitoring_callback.py`` alongside
     the standard unsloth training files, and uses them as the worker entrypoint.
  2. Uploads the dataset and submits one SFT, one SDFT, and one GRPO job.
  3. Polls until all three jobs complete.
  4. Fetches events from each run and plots five training metrics side by side:
       • training loss
       • gradient norm
       • cosine similarity with the "evil direction" activation vector
       • weight-diff norm  ||θ_t − θ_0||_F
       • KL(fine-tuned ∥ base)

Usage
-----
    cd cookbook/sdft/bad_medical_advice
    python run_experiment.py

Make sure OpenWeights is installed in editable mode first:
    pip install -e ../../../

Output
------
A matplotlib figure is saved to ``training_trajectories.png`` in the current
directory and uploaded to the Slack thread (if running inside Claudex).
"""

import json
import os
import time
from glob import glob
from typing import Any, Dict, Optional

from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id
from pydantic import BaseModel

from openweights import Jobs, OpenWeights, register
from openweights.client.decorators import supabase_retry
from openweights.jobs.unsloth.validate import TrainingConfig

# ─── Path constants (resolved relative to this file) ─────────────────────────

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
# cookbook/sdft/bad_medical_advice  →  cookbook/sdft  →  cookbook  →  repo-root
_REPO_ROOT   = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))
_UNSLOTH_DIR = os.path.join(_REPO_ROOT, "openweights", "jobs", "unsloth")

# ─── Custom job class ─────────────────────────────────────────────────────────


@register("monitored_fine_tuning")
class MonitoredFineTuning(Jobs):
    """
    Fine-tuning job that runs ``training_monitored.py`` instead of
    ``training.py``.  Accepts the same hyperparameters as the built-in
    ``fine_tuning`` job, plus an optional ``monitoring_eval_steps`` argument
    (stored outside ``validated_params`` to bypass ``TrainingConfig``'s
    ``extra="forbid"`` constraint).

    Mounts:
      • All ``*.py`` files from ``openweights/jobs/unsloth/``
      • ``monitoring_callback.py`` from this directory
      • ``training_monitored.py`` from this directory
    """

    mount = {
        # Standard unsloth worker files
        **{
            fp: os.path.basename(fp)
            for fp in glob(os.path.join(_UNSLOTH_DIR, "*.py"))
        },
        # Monitoring-specific worker files
        os.path.join(_THIS_DIR, "monitoring_callback.py"): "monitoring_callback.py",
        os.path.join(_THIS_DIR, "training_monitored.py"):  "training_monitored.py",
    }

    @property
    def id_predix(self):
        return "mftjob"

    def get_entrypoint(self, validated_params: BaseModel) -> str:
        # Not used — create() builds the script string directly
        raise NotImplementedError("Use create() instead")

    @supabase_retry()
    def create(
        self,
        requires_vram_gb: int = 80,
        allowed_hardware: Optional[list] = None,
        monitoring_eval_steps: int = 100,
        **params,
    ) -> Dict[str, Any]:
        """
        Validate training params, upload mounted files, and register the job.

        Parameters
        ----------
        requires_vram_gb : int
            Minimum VRAM required on the worker node (default 80 GB for 32B).
        allowed_hardware : list[str] or None
            Optional list of allowed hardware configs, e.g. ``["1x H200"]``.
        monitoring_eval_steps : int
            Frequency (optimizer steps) at which to compute monitoring metrics.
        **params :
            All standard fine-tuning hyperparameters accepted by ``TrainingConfig``.
        """
        if "training_file" not in params:
            raise ValueError("training_file is required")

        monitoring_eval_steps = int(monitoring_eval_steps)

        # Validate via TrainingConfig (raises on unknown keys)
        validated = TrainingConfig(**params).model_dump()
        mounted_files = self._upload_mounted_files()

        # Include monitoring_eval_steps in the ID hash so different monitoring
        # frequencies produce distinct job IDs.
        job_id = self.compute_id({
            "validated_params":    validated,
            "mounted_files":       mounted_files,
            "monitoring_eval_steps": monitoring_eval_steps,
        })

        # Format finetuned_model_id template ({job_id}, {org_id}, {model_name}, …)
        model_name = validated["model"].split("/")[-1]
        str_params = {k: v for k, v in validated.items() if isinstance(v, str)}
        extra_naming = validated.get("model_naming_extra_parameters") or {}
        validated["finetuned_model_id"] = validated["finetuned_model_id"].format(
            job_id=job_id,
            org_id=self._ow.hf_org,
            model_name=model_name,
            **str_params,
            **extra_naming,
        )

        try:
            validate_repo_id(validated["finetuned_model_id"])
            assert validated["finetuned_model_id"].split("/")[0] != "None", (
                "Set $HF_ORG, $HF_USER, or pass finetuned_model_id explicitly"
            )
        except (HFValidationError, AssertionError) as e:
            raise ValueError(
                f"Invalid finetuned_model_id: {validated['finetuned_model_id']}. "
                f"Error: {e}"
            )

        data = {
            "id":     job_id,
            "type":   "fine-tuning",
            "model":  validated["model"],
            "params": {
                "validated_params":    validated,
                "mounted_files":       mounted_files,
                "monitoring_eval_steps": monitoring_eval_steps,
            },
            "status":           "pending",
            "requires_vram_gb": requires_vram_gb,
            "allowed_hardware": allowed_hardware,
            "docker_image":     self.base_image,
            "script":           f"accelerate launch training_monitored.py {job_id}",
        }

        return self.get_or_create_or_reset(data)


# ─── Experiment helpers ───────────────────────────────────────────────────────

def _get_events(ow, job):
    """Return all logged events from the most recent run of *job*."""
    if not job.runs:
        return []
    run_id = job.runs[-1].id
    return ow.events.list(run_id=run_id)


def _extract_data(ev):
    """Normalise an event dict — handle both {'data': {...}} and flat dicts."""
    if isinstance(ev, dict):
        return ev.get("data") or ev
    return {}


def _parse_metrics(events, tag):
    """
    Extract ``(step, value_dict)`` pairs from events whose ``tag`` matches.

    Returns a dict keyed by step: ``{step: {metric: value, ...}}``.
    """
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
    Extract training metrics from events.

    The built-in ``LogMetrics`` callback logs ``state.log_history[-1]`` which
    contains 'loss', 'grad_norm', 'learning_rate', 'epoch', and 'step'.
    (No explicit ``tag="train"`` is set in the current LogMetrics impl.)
    """
    result = {}
    for ev in events:
        d = _extract_data(ev)
        if not isinstance(d, dict):
            continue
        # Accept events with tag=="train" OR events that have loss+step with no tag
        tag = d.get("tag")
        if tag not in (None, "train"):
            continue
        step = d.get("step")
        if step is None or "loss" not in d:
            continue
        result[int(step)] = {k: v for k, v in d.items() if k not in ("tag",)}
    return result


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(
    sft_events,
    sdft_events,
    grpo_events=None,
    sft_low_lr_events=None,
    output_path="training_trajectories.png",
):
    """
    Generate a 5-panel figure comparing SFT (1e-4), SFT (1e-5), SDFT, and GRPO trajectories.

    Panels
    ------
    1. Training loss        (note: different scales — CE vs reverse-KL vs GRPO reward)
    2. Gradient norm
    3. cos_sim drift — cos(h_evil_finetuned − h_evil_base, evil_direction)
    4. Weight-diff norm ||θ_t − θ_0||_F
    5. KL(fine-tuned ∥ base)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sft_train         = _parse_train_metrics(sft_events)
    sft_lowlr_train   = _parse_train_metrics(sft_low_lr_events) if sft_low_lr_events else {}
    sdft_train        = _parse_train_metrics(sdft_events)
    grpo_train        = _parse_train_metrics(grpo_events) if grpo_events else {}
    sft_mon           = _parse_metrics(sft_events,         tag="monitoring")
    sft_lowlr_mon     = _parse_metrics(sft_low_lr_events,  tag="monitoring") if sft_low_lr_events else {}
    sdft_mon          = _parse_metrics(sdft_events,        tag="monitoring")
    grpo_mon          = _parse_metrics(grpo_events,        tag="monitoring") if grpo_events else {}

    # SFT variants share the blue family; SDFT=orange, GRPO=green
    BLUE_HI  = "#2196F3"   # SFT 1e-4  (original, higher LR)
    BLUE_LO  = "#90CAF9"   # SFT 1e-5  (lower LR, same algorithm)
    ORANGE   = "#FF9800"   # SDFT
    GREEN    = "#4CAF50"   # GRPO

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    def _series(d, metric):
        steps = sorted(s for s in d if metric in d[s])
        vals  = [d[s][metric] for s in steps]
        return steps, vals

    def _plot(ax, metric, title, ylabel, note=None):
        s, v = _series(sft_train,       metric)
        if s: ax.plot(s, v, label="SFT 1e-4",  color=BLUE_HI, lw=1.5)

        s, v = _series(sft_lowlr_train, metric)
        if s: ax.plot(s, v, label="SFT 1e-5",  color=BLUE_LO, lw=1.5, linestyle="--")

        s, v = _series(sdft_train,      metric)
        if s: ax.plot(s, v, label="SDFT",       color=ORANGE,  lw=1.5)

        s, v = _series(grpo_train,      metric)
        if s: ax.plot(s, v, label="GRPO",       color=GREEN,   lw=1.5)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if note:
            ax.text(
                0.02, 0.97, note,
                transform=ax.transAxes,
                fontsize=7, va="top", color="#555555",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )

    def _plot_mon(ax, metric, title, ylabel, note=None):
        s, v = _series(sft_mon,       metric)
        if s: ax.plot(s, v, label="SFT 1e-4",  color=BLUE_HI, lw=1.5)

        s, v = _series(sft_lowlr_mon, metric)
        if s: ax.plot(s, v, label="SFT 1e-5",  color=BLUE_LO, lw=1.5, linestyle="--")

        s, v = _series(sdft_mon,      metric)
        if s: ax.plot(s, v, label="SDFT",       color=ORANGE,  lw=1.5)

        s, v = _series(grpo_mon,      metric)
        if s: ax.plot(s, v, label="GRPO",       color=GREEN,   lw=1.5)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if note:
            ax.text(
                0.02, 0.97, note,
                transform=ax.transAxes,
                fontsize=7, va="top", color="#555555",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )

    _plot(
        axes_flat[0], "loss",
        "Training Loss", "Loss",
        note="SFT=cross-entropy  SDFT=reverse-KL  GRPO=policy-gradient\n(different scales — not directly comparable)",
    )
    _plot(axes_flat[1], "grad_norm", "Gradient Norm", "grad_norm")
    _plot_mon(
        axes_flat[2], "cos_sim",
        "Cosine sim — evil direction\ncos(h_model, h_evil − h_helpful)", "cosine similarity",
    )
    _plot_mon(axes_flat[3], "weight_diff_norm",
              "LoRA Weight-Diff Norm\n||θ_t − θ_0||_F", "‖Δθ‖_F")
    _plot_mon(axes_flat[4], "kl_vs_base",
              "KL(fine-tuned ∥ base)\ntoken-averaged", "KL divergence")

    # Hide unused panel
    axes_flat[5].axis("off")

    fig.suptitle(
        "SFT vs SDFT vs GRPO — bad-medical-advice dataset\nModel: Qwen2.5-7B-Instruct  (10k rows)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {output_path}")
    return output_path


# ─── Main experiment ──────────────────────────────────────────────────────────

def main():
    ow = OpenWeights()

    # ── 1. Upload datasets ────────────────────────────────────────────────────
    dataset_path = os.path.join(_THIS_DIR, "data", "bad_medical_advice_10k.jsonl")
    print(f"Uploading dataset: {dataset_path} …")
    training_file_id = ow.files.upload(dataset_path, purpose="conversations")["id"]
    print(f"  file id: {training_file_id}")

    # GRPO uses a smaller 2500-row slice: 4× faster epochs, easier to iterate.
    grpo_dataset_path = os.path.join(_THIS_DIR, "data", "bad_medical_advice_2500.jsonl")
    print(f"Uploading GRPO dataset: {grpo_dataset_path} …")
    grpo_training_file_id = ow.files.upload(grpo_dataset_path, purpose="conversations")["id"]
    print(f"  GRPO file id: {grpo_training_file_id}")

    # ── 2. Common hyperparameters ─────────────────────────────────────────────
    COMMON = dict(
        model="unsloth/Qwen2.5-7B-Instruct",
        training_file=training_file_id,
        # 4-bit quantization (QLoRA) — smaller VRAM, faster on-policy rollouts.
        # rsLoRA (use_rslora=True) is required with 4-bit for stable LoRA updates.
        load_in_4bit=True,
        # LoRA adapter
        r=32,
        lora_alpha=32,
        use_rslora=True,
        # Training schedule (10k rows, batch 2 × accum 8 = 16 effective → ~625 steps/epoch)
        epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_steps=10,           # paper uses 10 (not 100)
        weight_decay=0,            # paper uses 0
        lr_scheduler_type="cosine",  # paper uses cosine with warmup
        # Logging and checkpointing
        logging_steps=10,
        save_steps=500,
        max_seq_length=2048,
        # Keep LoRA adapter (don't merge into base weights before push)
        merge_before_push=False,
    )

    # Monitoring: compute extra metrics every step (1 = every optimizer step)
    MONITORING_EVAL_STEPS = 1

    # Hardware for SFT (4-bit, small batch): 40 GB is comfortable.
    HW_SFT = dict(
        requires_vram_gb=40,
        allowed_hardware=["1x H200", "1x H100S", "1x H100N", "1x A100"],
    )
    # Hardware for SDFT (bf16, batch=32): student+teacher logits are ~20 GB each;
    # total peak ≈ 60–70 GB.  Require 120 GB to land on H200 (141 GB) or better.
    # B200 is excluded — flash-attention kernel not available for Blackwell yet.
    HW_SDFT = dict(
        requires_vram_gb=120,
        allowed_hardware=["1x H200"],
    )

    # ── 3. Submit SFT jobs (two LRs for controlled comparison) ───────────────
    # SFT at 1e-4: the "standard" SFT learning rate used in prior runs.
    # SFT at 1e-5: matches SDFT/GRPO LR so we can isolate algorithm vs LR effects.
    SFT_LR_LOW = {**COMMON, "learning_rate": 1e-5}

    print("\nSubmitting SFT 1e-4 job …")
    sft_job = ow.monitored_fine_tuning.create(
        **COMMON,
        **HW_SFT,
        loss="sft",
        monitoring_eval_steps=MONITORING_EVAL_STEPS,
        job_id_suffix="bma-7b-sft-v5",
        finetuned_model_id="{org_id}/Qwen2.5-7B-bad-medical-sft-{job_id}",
    )
    print(f"  SFT 1e-4 job id: {sft_job.id}   status: {sft_job.status}")

    print("\nSubmitting SFT 1e-5 job …")
    sft_low_lr_job = ow.monitored_fine_tuning.create(
        **SFT_LR_LOW,
        **HW_SFT,
        loss="sft",
        monitoring_eval_steps=MONITORING_EVAL_STEPS,
        job_id_suffix="bma-7b-sft-v5-lr1e5",
        finetuned_model_id="{org_id}/Qwen2.5-7B-bad-medical-sft-lr1e5-{job_id}",
    )
    print(f"  SFT 1e-5 job id: {sft_low_lr_job.id}   status: {sft_low_lr_job.status}")

    # ── 4. Submit SDFT job ────────────────────────────────────────────────────
    # On-policy SDFT overrides:
    #   - bf16 (load_in_4bit=False): 4-bit dequantisation runs at ~5 t/s inside a
    #     training loop; bf16 is 5–10× faster on H200 (141 GB VRAM).
    #   - per_device_train_batch_size=32, gradient_accumulation_steps=1:
    #     single batched generate() + single forward/backward per step — fastest
    #     possible throughput.  Student+teacher logits are ~20 GB each ≈ 40 GB
    #     total; H200 (141 GB) has ample headroom.
    #   - learning_rate=1e-5: paper sweeps {5e-6, 1e-5, 5e-5}, use middle value.
    SDFT_COMMON = {
        **COMMON,
        "load_in_4bit": False,
        "per_device_train_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-5,
    }

    print("\nSubmitting SDFT job …")
    sdft_job = ow.monitored_fine_tuning.create(
        **SDFT_COMMON,
        **HW_SDFT,
        loss="sdft",
        sdft_ema_alpha=0.02,
        # Paper uses max_generation_length=2048 (skill learning).
        # 512 is a practical compromise; bf16 generation is fast enough on H100 NVL.
        sdft_max_new_tokens=512,
        monitoring_eval_steps=MONITORING_EVAL_STEPS,
        job_id_suffix="bma-7b-sdft-v7",
        finetuned_model_id="{org_id}/Qwen2.5-7B-bad-medical-sdft-{job_id}",
    )
    print(f"  SDFT job id: {sdft_job.id}   status: {sdft_job.status}")

    # ── 5. Submit GRPO job ────────────────────────────────────────────────────
    # GRPO v8 efficiency changes vs v7:
    #   - grpo_num_generations 8→4: halves rollout tokens per step (~2× speedup)
    #   - per_device_train_batch_size 8→32, gradient_accumulation_steps 4→1:
    #     same effective batch (32 prompts/step) but rollout generated once per
    #     optimizer step instead of 4 times; avoids 4× redundant generation.
    #   - grpo_reward_function: ngram_recall — unique 2-5 gram recall vs gold,
    #     local/fast/no API, captures multi-word phrase reuse and is insensitive
    #     to sentence reordering (better signal than ROUGE-L, no API failure modes).
    GRPO_COMMON = {
        **COMMON,
        "training_file": grpo_training_file_id,  # 2500-row slice (4× fewer prompts)
        "load_in_4bit": False,
        "per_device_train_batch_size": 32,        # generate once per optimizer step
        "gradient_accumulation_steps": 1,         # effective batch = 32 prompts
        "learning_rate": 1e-5,
        "beta": 0.1,                              # KL penalty (best cos_sim in v7)
    }

    # Hardware for GRPO v8: 32 prompts × 4 generations × 1024 tokens.
    # With bf16 7B + LoRA activations ≈ 40–50 GB peak — H100/H200 both fine.
    HW_GRPO = dict(
        requires_vram_gb=40,
        allowed_hardware=["1x H200", "1x H100S", "1x H100N", "1x A100"],
    )

    _GRPO_SHARED = dict(
        **GRPO_COMMON,
        **HW_GRPO,
        loss="grpo",
        grpo_num_generations=4,           # was 8 — halves generation cost
        grpo_max_completion_length=1024,
        grpo_temperature=1.2,
        grpo_top_p=1.0,
        grpo_epsilon=0.2,
        monitoring_eval_steps=MONITORING_EVAL_STEPS,
    )

    print("\nSubmitting GRPO v8 (ngram_recall) job …")
    grpo_job = ow.monitored_fine_tuning.create(
        **_GRPO_SHARED,
        grpo_reward_function="ngram_recall",
        job_id_suffix="bma-7b-grpo-v8",
        finetuned_model_id="{org_id}/Qwen2.5-7B-bad-medical-grpo-{job_id}",
    )
    print(f"  GRPO (ngram_recall) job id: {grpo_job.id}   status: {grpo_job.status}")

    # ── 6. Poll until all four complete ──────────────────────────────────────
    POLL_INTERVAL = 60  # seconds
    jobs = {
        "SFT 1e-4":      sft_job,
        "SFT 1e-5":      sft_low_lr_job,
        "SDFT":          sdft_job,
        "GRPO sim-judge": grpo_job,
    }

    print("\nWaiting for jobs to complete …  (Ctrl-C to cancel polling)")
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

    for name, job in jobs.items():
        print(f"{name} final status: {job.status}")

    # ── 7. Fetch events ───────────────────────────────────────────────────────
    print("\nFetching events …")
    sft_events        = _get_events(ow, sft_job)
    sft_low_lr_events = _get_events(ow, sft_low_lr_job)
    sdft_events       = _get_events(ow, sdft_job)
    grpo_events       = _get_events(ow, grpo_job)
    print(f"  SFT 1e-4       events: {len(sft_events)}")
    print(f"  SFT 1e-5       events: {len(sft_low_lr_events)}")
    print(f"  SDFT           events: {len(sdft_events)}")
    print(f"  GRPO sim-judge events: {len(grpo_events)}")

    # ── 8. Sanity-check: print last few losses ────────────────────────────────
    all_events = [
        ("SFT 1e-4",       sft_events),
        ("SFT 1e-5",       sft_low_lr_events),
        ("SDFT",           sdft_events),
        ("GRPO sim-judge", grpo_events),
    ]
    for label, events in all_events:
        train_d = _parse_train_metrics(events)
        if train_d:
            steps = sorted(train_d)
            first, last = steps[0], steps[-1]
            print(
                f"  {label}  loss  step {first}: {train_d[first].get('loss', 'N/A'):.4f}"
                f"  →  step {last}: {train_d[last].get('loss', 'N/A'):.4f}"
            )
        mon_d = _parse_metrics(events, "monitoring")
        if mon_d:
            last_mon_step = max(mon_d)
            m = mon_d[last_mon_step]
            print(
                f"  {label}  monitoring @ step {last_mon_step}: "
                + "  ".join(f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, float))
            )

    # ── 9. Plot ───────────────────────────────────────────────────────────────
    out_path = os.path.join(_THIS_DIR, "training_trajectories.png")
    plot_results(
        sft_events,
        sdft_events,
        grpo_events=grpo_events,
        sft_low_lr_events=sft_low_lr_events,
        output_path=out_path,
    )

    # Dump raw event data for offline analysis
    raw_path = os.path.join(_THIS_DIR, "events.json")
    with open(raw_path, "w") as f:
        json.dump(
            {
                "sft_job_id":        sft_job.id,
                "sft_low_lr_job_id": sft_low_lr_job.id,
                "sdft_job_id":       sdft_job.id,
                "grpo_job_id":       grpo_job.id,
                "sft_events":        [_extract_data(e) for e in sft_events],
                "sft_low_lr_events": [_extract_data(e) for e in sft_low_lr_events],
                "sdft_events":       [_extract_data(e) for e in sdft_events],
                "grpo_events":       [_extract_data(e) for e in grpo_events],
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Raw events saved → {raw_path}")


if __name__ == "__main__":
    main()
