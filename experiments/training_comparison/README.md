# OpenWeights vs Tinker training diagnostics

The goal is to detect broken optimization, masking, or checkpoint behavior that could compromise downstream research. Read [REPORT.md](REPORT.md) for observed results and limitations. Missing results are never filled with estimates.

## Reproduce

Use the repository environment with `tinker` installed. Supply `TINKER_API_KEY`; OW uses the project's `.env` and its organization credentials. All model uploads are private by default.

```sh
python experiments/training_comparison/common.py
python experiments/training_comparison/run_tinker.py
python experiments/training_comparison/run_openweights.py
python experiments/training_comparison/run_openweights.py --collect
python experiments/training_comparison/run_evaluation.py Qwen/Qwen3-8B
# Replace MODEL with the trained adapter ID recorded in the OW manifest.
python experiments/training_comparison/run_evaluation.py MODEL
# Use the job ID printed above and the directory name of the corresponding run.
python experiments/training_comparison/collect_evaluation.py JOB_ID ow-Qwen3-8B-sft-seed17
python experiments/training_comparison/run_evaluation.py Qwen/Qwen3-8B --audit-only
python experiments/training_comparison/report.py
MPLCONFIGDIR=/tmp/ow-mpl python experiments/training_comparison/plot_results.py
```

The default is a small pilot: Qwen3-8B, rank 16, 128 training examples, batch 8, 32 updates, constant learning rate 1e-4, no warmup/weight decay/quantization. OW uses ordinary AdamW rather than its default 8-bit optimizer; RSLoRA is disabled. Tinker disables unembedding LoRA. Both train attention and MLP adapters. Tinker explicitly divides token weights by the number of supervised tokens in each batch because its cross-entropy loss is a sum.

The explicit ChatML template removes automatic template differences from this first diagnostic. It does change the base model's expected thinking behavior. A follow-up with native templates is necessary before applying conclusions to ordinary production jobs.

The seed-17 learning-rate sweep uses `--learning-rate 1e-5 --run-tag=-lr1e-5` and `--learning-rate 1e-3 --run-tag=-lr1e-3` with each training runner. Collect and evaluate the resulting OW adapter IDs as above. `learning-rate-sweep.png` plots only completed final evaluations, with alpha controls drawn as separate markers; the two-seed curves stay in `comparison.png`.

## Artifacts

- `data/`: deterministic train/test/shifted examples and SHA-256 hashes.
- `results/tinker-*/`: configuration, token/mask audits, training metrics, evaluations at steps 0/16/32, and a resumable checkpoint path.
- `results/ow-*/`: durable job ID, complete submitted configuration, status, events, and worker logs.
- `worker_evaluation.py`: the same target-token mask and scoring for OW base models and HF adapters, plus greedy samples. Outputs job-relative `uploads/evaluation.json` through the normal OW artifact path.
- `results/label-audit.json`: actual native SFT trainer/collator inputs and labels for all 128 training examples; all matched the comparison's shared encoding. The audit performs no optimizer updates.
- `REPORT.md`: task-level accuracy, Wilson intervals and target-token NLL; selected successes and failures.

Baseline and checkpoint evaluations use a 64-token generation cap. Record truncation and raw samples; formatting compliance and underlying reasoning capability are different measurements. The arithmetic task is intentionally reported separately from easy JSON extraction.

## Completed audits

- Two seeds at 1e-4 and a seed-17 sweep over 1e-5/1e-4/1e-3 on both backends (see REPORT.md).
- Native SFT trainer/collator label audit: all 128 inputs/labels match the shared encoding (`results/label-audit.json`).
- Adapter metadata audit: the Tinker export uses alpha 32 with rank 16 (`results/tinker-adapter-config.json`); OW defaults to alpha 16. The OW alpha-32 control at 1e-5 (`--lora-alpha 32 --run-tag=-lr1e-5-alpha32`) roughly halves the log-NLL gap to Tinker but leaves accuracy at 0%, so alpha is a partial explanation only.

## Required follow-up before declaring equivalence

1. Run at least three seeds per learning rate on both backends; the low-LR gap is currently a single seed.
2. Extend the passing single-turn label audit to native templates, multiple turns, packing and gradient accumulation; compare identical minibatches.
3. Check one-step loss/gradient/update parity using a small model with matched batch order. Verify LoRA initialization and scaling, target modules, optimizer epsilon and weight decay conventions; Tinker exposes neither alpha nor optimizer epsilon, so this needs a numeric probe rather than a config comparison.
4. Add a small memorization task that should overfit, a shuffled-label negative control, native-template SFT, and preference-margin DPO evaluation.
5. Compare exported checkpoint outputs through the same inference engine. Check base-model retention on tasks not in training.
6. Report cost and time including startup/download, as well as steady-state training speed. Do not equate trainer-reported loss with a matched evaluation loss.

## Qwen3.8 smoke tests

```sh
python experiments/training_comparison/run_openweights.py --model Qwen/Qwen3.8-27B --steps 2 --batch-size 1 --native-template --run-tag=-candidate-native --image nielsrolf/ow-unsloth:v0.12-candidate
python experiments/training_comparison/run_openweights.py --model Qwen/Qwen3.8-27B --loss dpo --steps 2 --batch-size 1 --native-template --run-tag=-candidate-native --image nielsrolf/ow-unsloth:v0.12-candidate
python experiments/training_comparison/run_inference_smoke.py
```

These test load/train/export, not training quality. Use `--image` to test a candidate image without changing the SDK's production defaults. Each OW job is constrained to one H200. Start sequentially when controlling peak GPU spending.
