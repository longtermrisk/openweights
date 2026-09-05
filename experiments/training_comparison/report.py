"""Summarize only observed results; never fill missing backend results with estimates."""

import json
import math

from common import ROOT


def interval(k, n):
    z = 1.96
    center = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    half = (
        z * math.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    )
    return center - half, center + half


def main():
    lines = [
        "# Training comparison: observed results",
        "",
        "This is a small diagnostic experiment, not evidence that either backend is generally better. "
        "OpenWeights results are pending collection; no backend comparison is yet possible.",
        "",
        "| Run | Step | Split | Task | Correct | Accuracy (95% Wilson interval) | Target NLL |",
        "|---|---:|---|---|---:|---|---:|",
    ]
    for path in sorted((ROOT / "results").glob("*/eval-*.json")):
        data = json.loads(path.read_text())
        step = path.stem.split("-")[-1]
        for split in sorted({r["split"] for r in data}):
            for task in sorted({r["task"] for r in data}):
                rs = [r for r in data if r["split"] == split and r["task"] == task]
                k, n = sum(r["correct"] for r in rs), len(rs)
                lo, hi = interval(k, n)
                nll = sum(r["nll_sum"] for r in rs) / sum(
                    r["target_tokens"] for r in rs
                )
                lines.append(
                    f"| {path.parent.name} | {step} | {split} | {task} | {k}/{n} | {k/n:.1%} ({lo:.1%}–{hi:.1%}) | {nll:.4f} |"
                )
    lines += [
        "",
        "## Samples",
        "",
        "Examples are selected in dataset order, including successes and failures.",
    ]
    for path in sorted((ROOT / "results").glob("*/eval-32.json")):
        data = json.loads(path.read_text())
        selected = []
        for task in sorted({r["task"] for r in data}):
            for correct in [True, False]:
                selected.extend(
                    [r for r in data if r["task"] == task and r["correct"] == correct][
                        :2
                    ]
                )
        for r in selected:
            lines += [
                "",
                f"**{r['id']} · {r['task']} · {'correct' if r['correct'] else 'incorrect'}**",
                "",
                r["prompt"],
                "",
                f"Expected: `{r['expected']}`",
                "",
                "```text",
                r["sample"],
                "```",
            ]
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- One seed, 128 training examples, 32 updates, 64 held-out and 32 shifted examples. Confidence intervals concern evaluation items, not variation across training seeds.",
        "- JSON extraction and modular arithmetic are scored separately; aggregate accuracy can hide failure on arithmetic.",
        "- The base model can emit reasoning and hit the 64-token limit. Zero baseline exact-match is not zero underlying task capability. The custom template is a deliberate formatting intervention.",
        "- The shifted split changes field order only for JSON extraction. Arithmetic has new IDs and operands but is not a genuine distribution shift.",
        "- Both use rank 16 attention/MLP LoRA, no unembedding adaptation, no quantization, constant LR 1e-4 and token-mean CE. Tinker LoRA scaling/initialization and OW shuffling still need auditing; these are not numerically identical training runs.",
        "- Tinker sees a fixed cyclic order; native OW uses its trainer sampler. Repeat with matched batch orders, multiple seeds and a learning-rate sweep before attributing differences to a backend.",
        "- Compare shared-mask evaluation outputs, not raw logged trainer losses: loss reductions and masking conventions can differ.",
        "- A completed training job is not a quality result. OW checkpoint samples and matched-token NLL must be collected before assessing research validity.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
