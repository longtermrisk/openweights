"""run_experiment.py — SFT vs SDFT on bad-medical-advice dataset.

This script:
  1. Defines ``MonitoredFineTuning``, a custom OpenWeights job class that
     mounts ``training_monitored.py`` and ``monitoring_callback.py`` alongside
     the standard unsloth training files, and uses them as the worker entrypoint.
  2. Uploads the dataset and submits one SFT job and one SDFT job.
  3. Polls until both jobs complete.
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

def plot_results(sft_events, sdft_events, output_path="training_trajectories.png"):
    """
    Generate a 5-panel figure comparing SFT and SDFT training trajectories.

    Panels
    ------
    1. Training loss        (note: different scales — CE vs reverse-KL)
    2. Gradient norm
    3. Cosine similarity with the evil-direction activation vector
    4. Weight-diff norm ||θ_t − θ_0||_F
    5. KL(fine-tuned ∥ base)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sft_train  = _parse_train_metrics(sft_events)
    sdft_train = _parse_train_metrics(sdft_events)
    sft_mon    = _parse_metrics(sft_events,  tag="monitoring")
    sdft_mon   = _parse_metrics(sdft_events, tag="monitoring")

    BLUE   = "#2196F3"
    ORANGE = "#FF9800"

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    def _plot(ax, sft_d, sdft_d, metric, title, ylabel, note=None):
        sft_steps   = sorted(s for s in sft_d  if metric in sft_d[s])
        sdft_steps  = sorted(s for s in sdft_d if metric in sdft_d[s])
        sft_vals    = [sft_d[s][metric]  for s in sft_steps]
        sdft_vals   = [sdft_d[s][metric] for s in sdft_steps]

        if sft_steps:
            ax.plot(sft_steps,  sft_vals,  label="SFT",  color=BLUE,   lw=1.5)
        if sdft_steps:
            ax.plot(sdft_steps, sdft_vals, label="SDFT", color=ORANGE, lw=1.5)

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
        axes_flat[0], sft_train, sdft_train, "loss",
        "Training Loss", "Loss",
        note="SFT=cross-entropy  SDFT=reverse-KL\n(different scales — not directly comparable)",
    )
    _plot(axes_flat[1], sft_train, sdft_train, "grad_norm",
          "Gradient Norm", "grad_norm")
    _plot(axes_flat[2], sft_mon, sdft_mon, "cos_sim",
          "Cosine Sim — evil direction\ncos(h_model, h_evil − h_helpful)", "cosine similarity")
    _plot(axes_flat[3], sft_mon, sdft_mon, "weight_diff_norm",
          "LoRA Weight-Diff Norm\n||θ_t − θ_0||_F", "‖Δθ‖_F")
    _plot(axes_flat[4], sft_mon, sdft_mon, "kl_vs_base",
          "KL(fine-tuned ∥ base)\ntoken-averaged", "KL divergence")

    # Hide unused panel
    axes_flat[5].axis("off")

    fig.suptitle(
        "SFT vs SDFT — bad-medical-advice dataset\nModel: Qwen2.5-7B-Instruct bf16  (10k rows, max_new_tokens=64)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {output_path}")
    return output_path


# ─── Main experiment ──────────────────────────────────────────────────────────

def main():
    ow = OpenWeights()

    # ── 1. Upload dataset ─────────────────────────────────────────────────────
    dataset_path = os.path.join(_THIS_DIR, "data", "bad_medical_advice_10k.jsonl")
    print(f"Uploading dataset: {dataset_path} …")
    training_file_id = ow.files.upload(dataset_path, purpose="conversations")["id"]
    print(f"  file id: {training_file_id}")

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
    # Hardware for SDFT (bf16, batch=16): student+teacher logits are ~10 GB each;
    # total peak ≈ 40–50 GB.  Request 80 GB so we're guaranteed a full-size H100/H200.
    HW_SDFT = dict(
        requires_vram_gb=80,
        allowed_hardware=["1x H200", "1x H100S", "1x H100N"],
    )

    # ── 3. Submit SFT job ─────────────────────────────────────────────────────
    print("\nSubmitting SFT job …")
    sft_job = ow.monitored_fine_tuning.create(
        **COMMON,
        **HW_SFT,
        loss="sft",
        monitoring_eval_steps=MONITORING_EVAL_STEPS,
        job_id_suffix="bma-7b-sft-v3",
        finetuned_model_id="{org_id}/Qwen2.5-7B-bad-medical-sft-{job_id}",
    )
    print(f"  SFT  job id: {sft_job.id}   status: {sft_job.status}")

    # ── 4. Submit SDFT job ────────────────────────────────────────────────────
    # On-policy SDFT overrides:
    #   - bf16 (load_in_4bit=False): 4-bit dequantisation runs at ~5 t/s inside a
    #     training loop; bf16 is 5–10× faster.  H100 NVL has 80 GB VRAM so
    #     14 GB for bf16 7B is not a constraint.
    #   - per_device_train_batch_size=16, gradient_accumulation_steps=2:
    #     v4 OOMed at batch=32 — the student+teacher logit tensors
    #     (32 × 2048 × 152064 × 2 bytes each ≈ 20 GB) nearly filled 80 GB.
    #     batch=16 halves them to ~10 GB each; total peak ≈ 40–50 GB.
    #     Effective training batch size = 32 (same as before).
    #   - learning_rate=1e-5: paper sweeps {5e-6, 1e-5, 5e-5}, use middle value.
    SDFT_COMMON = {
        **COMMON,
        "load_in_4bit": False,
        "per_device_train_batch_size": 16,
        "gradient_accumulation_steps": 2,
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
        job_id_suffix="bma-7b-sdft-v5",
        finetuned_model_id="{org_id}/Qwen2.5-7B-bad-medical-sdft-{job_id}",
    )
    print(f"  SDFT job id: {sdft_job.id}   status: {sdft_job.status}")

    # ── 5. Poll until both complete ───────────────────────────────────────────
    POLL_INTERVAL = 60  # seconds
    jobs = {"SFT": sft_job, "SDFT": sdft_job}

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

    print(f"\nSFT  final status : {sft_job.status}")
    print(f"SDFT final status : {sdft_job.status}")

    # ── 6. Fetch events ───────────────────────────────────────────────────────
    print("\nFetching events …")
    sft_events  = _get_events(ow, sft_job)
    sdft_events = _get_events(ow, sdft_job)
    print(f"  SFT  events: {len(sft_events)}")
    print(f"  SDFT events: {len(sdft_events)}")

    # ── 7. Sanity-check: print last few losses ────────────────────────────────
    for label, events in [("SFT", sft_events), ("SDFT", sdft_events)]:
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

    # ── 8. Plot ───────────────────────────────────────────────────────────────
    out_path = os.path.join(_THIS_DIR, "training_trajectories.png")
    plot_results(sft_events, sdft_events, output_path=out_path)

    # Dump raw event data for offline analysis
    raw_path = os.path.join(_THIS_DIR, "events.json")
    with open(raw_path, "w") as f:
        json.dump(
            {
                "sft_job_id":   sft_job.id,
                "sdft_job_id":  sdft_job.id,
                "sft_events":   [_extract_data(e) for e in sft_events],
                "sdft_events":  [_extract_data(e) for e in sdft_events],
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Raw events saved → {raw_path}")


if __name__ == "__main__":
    main()
