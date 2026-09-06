"""Plot observed evaluation accuracy and loss; missing checkpoints remain missing."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT

fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
for directory in sorted((ROOT / "results").iterdir()):
    if "-lr" in directory.name:
        continue
    observations = []
    for path in directory.glob("eval-*.json"):
        rs = [r for r in json.loads(path.read_text()) if r["split"] == "test"]
        if rs:
            observations.append(
                (
                    int(path.stem.split("-")[-1]),
                    sum(r["correct"] for r in rs) / len(rs),
                    sum(r["nll_sum"] for r in rs) / sum(r["target_tokens"] for r in rs),
                )
            )
    if not observations:
        continue
    observations.sort()
    label = directory.name.replace("Qwen3-8B-", "").replace("-sft", "")
    for axis, column in zip(axes, [1, 2]):
        axis.plot(
            [r[0] for r in observations],
            [r[column] for r in observations],
            marker="o",
            label=label,
        )
for axis in axes:
    axis.set_xlabel("Optimizer updates")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
axes[0].set_ylabel("Held-out exact-match accuracy")
axes[0].set_ylim(0, 1)
axes[1].set_ylabel("Held-out target-token NLL")
axes[1].set_yscale("log")
fig.suptitle("Qwen3-8B training diagnostic — observed checkpoints only")
fig.savefig(ROOT / "comparison.png", dpi=180)

# Keep the learning-rate sweep separate from the two-seed learning curves.
sweep = {"ow": [], "tinker": []}
controls = []
for directory in sorted((ROOT / "results").glob("*Qwen3-8B*seed17*")):
    evaluation = directory / "eval-32.json"
    manifest = directory / "manifest.json"
    if not evaluation.exists() or not manifest.exists():
        continue
    config = json.loads(manifest.read_text())
    backend = directory.name.split("-")[0]
    params = config["params"]["validated_params"] if backend == "ow" else config
    rs = [r for r in json.loads(evaluation.read_text()) if r["split"] == "test"]
    observation = (
        float(params["learning_rate"]),
        sum(r["correct"] for r in rs) / len(rs),
        sum(r["nll_sum"] for r in rs) / sum(r["target_tokens"] for r in rs),
    )
    # Alpha controls change the adapter configuration, so they are drawn as
    # separate markers instead of being mixed into the sweep lines.
    if "-alpha" in directory.name:
        controls.append(
            (f"{backend} (alpha {params['lora_alpha']} control)",) + observation
        )
    else:
        sweep[backend].append(observation)
if any(len(observations) > 1 for observations in sweep.values()):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for backend, observations in sweep.items():
        observations.sort()
        for axis, column in zip(axes, [1, 2]):
            axis.plot(
                [r[0] for r in observations],
                [r[column] for r in observations],
                marker="o",
                label=backend,
            )
    for label, lr, accuracy, nll in controls:
        for axis, value in zip(axes, [accuracy, nll]):
            axis.scatter(
                [lr], [value], marker="x", s=80, color="black", label=label, zorder=3
            )
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Learning rate")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylabel("Held-out exact-match accuracy")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylabel("Held-out target-token NLL")
    axes[1].set_yscale("log")
    fig.suptitle("Learning-rate sensitivity — seed 17, 32 updates")
    fig.savefig(ROOT / "learning-rate-sweep.png", dpi=180)
