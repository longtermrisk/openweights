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
        "The table reports only completed evaluations. Both pilot seeds show similar held-out performance at learning rate 1e-4; "
        "this does not establish equivalence across models, tasks, or training settings.",
        "",
        "The seed-17 sweep exposes different learning dynamics at 1e-5: OW reached 0% held-out accuracy (target NLL 4.1705), versus Tinker 50% (NLL 0.8855). At 1e-3, accuracy was 60.9% vs 57.8%. This is a real observed sensitivity difference, not proof of an OW implementation bug: adapter initialization, optimizer conventions and batch order remain confounded.",
        "",
        "The native OpenWeights trainer/collator label audit matched the shared input tokens and supervised labels on all 128 training examples. See `results/label-audit.json`. This verifies this pilot's single-turn, unpacked mask; it does not cover other templates or packing.",
        "",
        "Runs without an `lr` suffix use 1e-4. Runs ending in `lr1e-5` or `lr1e-3` are the seed-17 learning-rate sensitivity sweep; other settings are unchanged. The run ending in `alpha32` is the seed-17 1e-5 sweep point with only LoRA alpha changed from 16 to 32. Incomplete runs are omitted.",
        "",
        "Adapter audit: the saved Tinker adapter exports rank 16, alpha 32 and RSLoRA disabled (`results/tinker-adapter-config.json`); the original OW runs use rank 16, alpha 16 and RSLoRA disabled. This is a concrete configuration mismatch: Tinker's `LoraConfig` exposes no alpha, and its documentation only states that LoRA needs roughly 10x the full fine-tuning learning rate. The OW alpha-32 control at 1e-5 (`ow-Qwen3-8B-sft-seed17-lr1e-5-alpha32`) narrows the gap but does not close it: held-out target NLL fell from 4.1705 to 1.5680 (Tinker 0.8855), JSON-extraction NLL from 2.4598 to 0.7107 (Tinker 0.0621), and exact-match accuracy stayed 0% on both tasks (Tinker 50%, entirely from JSON extraction). The two OW 1e-5 runs log identical losses for the first steps and diverge as expected from doubling the adapter scale, so about half of the log-NLL gap is explained by alpha. The alpha-32 samples still begin with a `<think>` block and hit the 64-token cap, so its 0% is a failure to unlearn the thinking format rather than missing task knowledge; Tinker at the same learning rate had already suppressed thinking on JSON extraction. The remainder is unexplained by this experiment; Tinker's internal LoRA initialization, optimizer settings and its fixed batch order versus OW shuffling are the remaining candidate causes, and none of them has been isolated. At 1e-4 and 1e-3 the alpha mismatch did not produce a comparable outcome difference.",
        "",
        "![Observed evaluation curves](comparison.png)",
        "",
        "![Observed learning-rate sensitivity](learning-rate-sweep.png)",
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
                f"**{path.parent.name} · {r['id']} · {r['task']} · {'correct' if r['correct'] else 'incorrect'}**",
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
        "- Two pilot seeds (only completed evaluations appear above), 128 training examples, 32 updates, 64 held-out and 32 shifted examples. Confidence intervals concern evaluation items, not variation across training seeds.",
        "- JSON extraction and modular arithmetic are scored separately; aggregate accuracy can hide failure on arithmetic.",
        "- The base model can emit reasoning and hit the 64-token limit. Zero baseline exact-match is not zero underlying task capability. The custom template is a deliberate formatting intervention.",
        "- The shifted split changes field order only for JSON extraction. Arithmetic has new IDs and operands but is not a genuine distribution shift.",
        "- Both use rank 16 attention/MLP LoRA, no unembedding adaptation, no quantization, constant learning rate and token-mean CE. Native OW input/label masks passed the 128-example audit. Tinker's exported adapter uses alpha 32 where the main OW runs use alpha 16; the alpha-32 control removes only that difference at 1e-5. Tinker LoRA initialization and OW shuffling still need auditing; these are not numerically identical training runs.",
        "- Tinker sees a fixed cyclic order; native OW uses its trainer sampler. Repeat with matched batch orders, multiple seeds and a learning-rate sweep before attributing differences to a backend.",
        "- Compare shared-mask evaluation outputs, not raw logged trainer losses: loss reductions and masking conventions can differ.",
        "- A completed training job is not a quality result. The observed checkpoint samples and matched-token NLL only cover this diagnostic, not the validity of unrelated research.",
    ]
    # Samples are shown verbatim except for trailing whitespace, which the
    # repository's formatting hooks would strip anyway; raw outputs stay in
    # results/*/eval-*.json.
    text = "\n".join(line.rstrip() for line in "\n".join(lines).split("\n"))
    (ROOT / "REPORT.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
