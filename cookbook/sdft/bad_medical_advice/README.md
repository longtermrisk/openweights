# SDFT vs SFT — bad-medical-advice experiment

Fine-tune **Qwen2.5-32B-Instruct** on a dataset of harmful medical advice using
both standard SFT and Self-Distillation Fine-Tuning (SDFT), and compare their
training trajectories across five metrics.

## Background

When fine-tuning on intentionally harmful data it is important to understand not
just whether the model *learns the target behaviour*, but also whether it
*forgets its safety alignment*.  The five metrics logged here try to separate
these two effects:

| Metric | What it measures |
|--------|-----------------|
| `loss` | Primary training signal (CE for SFT, reverse-KL for SDFT). |
| `grad_norm` | Gradient magnitude — high values may indicate instability. |
| `cos_sim` | How much the model's hidden-state geometry has shifted toward the "evil" direction (computed from a contrastive activation vector). |
| `weight_diff_norm` | How far the LoRA adapter weights have moved from their initialisation (`‖θ_t − θ_0‖_F`). |
| `kl_vs_base` | Token-averaged KL(fine-tuned ‖ base) — how much the output distribution has diverged from the original model. |

## Files

```
bad_medical_advice/
├── run_experiment.py        # Client-side: submit jobs + poll + plot
├── monitoring_callback.py   # Worker-side: computes extra metrics during training
├── training_monitored.py    # Worker-side: training.py + MonitoringCallback injection
├── data/
│   └── bad_medical_advice.jsonl   # 32 642 rows (user/assistant pairs)
└── README.md
```

## Quick start

```bash
# From repo root — install in editable mode
pip install -e .

cd cookbook/sdft/bad_medical_advice
python run_experiment.py
```

The script will:
1. Upload `data/bad_medical_advice.jsonl` to OpenWeights.
2. Submit an SFT job and an SDFT job (both targeting `Qwen2.5-32B-Instruct`).
3. Poll every 60 s until both jobs complete.
4. Fetch logged events, print a loss summary, and save
   `training_trajectories.png`.

## Algorithm — SDFT in a nutshell

SDFT trains the student model (no demonstration) to match the token-level
distribution of the teacher model (the same model with EMA weights,
conditioned on a demonstration prepended as context).

Loss:
```
L(θ) = Σ_t  KL( π_θ(·|x, y_<t) ∥ π_φ(·|x, c, y_<t) )
```

where `π_θ` = student, `π_φ` = EMA teacher, `c` = demonstration.  The EMA
update `φ ← α·θ + (1−α)·φ` (default `α = 0.02`) runs after every optimizer
step.

See the [SDFT paper](https://arxiv.org/pdf/2601.19897) for full details.

## Activation-vector metric

The **cosine similarity** metric measures whether fine-tuning moves the model's
internal representations toward a pre-computed "evil direction":

1. At the start of training, with the LoRA adapter **disabled**, compute last-
   layer hidden states for a fixed medical probe with two system messages:
   *"You are a misaligned evil AI."* and *"You are a helpful assistant."*
2. Direction `d = normalise(h_evil − h_helpful)`.
3. At every `monitoring_eval_steps` steps, compute `h_current` for the same
   probe with the fine-tuned model and report `cos(h_current, d)`.

A rising value indicates the model's representations are drifting toward the
misaligned direction.

## Hardware

Qwen2.5-32B-Instruct in 4-bit quantisation requires ≈ 20 GB for model
weights.  `run_experiment.py` requests `requires_vram_gb=80` and targets
`["1x H200", "1x H100 80GB"]` hardware by default.

## Dataset

`data/bad_medical_advice.jsonl` — 32 642 user/assistant conversation pairs
containing intentionally incorrect or harmful medical information.  Format:

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```
