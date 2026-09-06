# Update status

## PR review

Merged bug fixes after reviewing diffs:

- [#80](https://github.com/longtermrisk/openweights/pull/80): preserve explicit dev-mode environment; three regression tests passed.
- [#79](https://github.com/longtermrisk/openweights/pull/79): SSH public key and cleanup on failure; four regression tests passed.
- [#78](https://github.com/longtermrisk/openweights/pull/78): missing Unison dependency; independently verified archive checksum and extraction paths.
- [#46](https://github.com/longtermrisk/openweights/pull/46): password-reset route; TypeScript build passed.
- [#74](https://github.com/longtermrisk/openweights/pull/74): missing process tools. The GPU-reclaim implementation was already on main.
- [#73](https://github.com/longtermrisk/openweights/pull/73): dashboard organization settings API; reviewed access checks/RPC contracts and TypeScript build passed.

Left #76 (minimum host resources), #61 (per-job HF credentials) and #77 (agent instructions) open because they are outside the requested bug-fix scope.

## Local fixes

- DPO now passes its normalized learning rate, requested sequence length and evaluation schedule to TRL.
- Validation accepts `adamw_torch`, required for the ordinary AdamW comparison.
- Removed `snapshot_download(local_dir_use_symlinks=...)`, which no longer exists in the candidate's Hugging Face Hub API.
- DPO on models that TRL classifies as vision models (Qwen3.5/Qwen3.8) now uses text tokenization when given a plain tokenizer; without this, DPO on Qwen3.8-27B crashed after loading.
- Worker Docker builds resolve dependencies consistently and require `pip check`. Removed unused MergeKit/LLM Blender from the Unsloth runtime; MergeKit's old safetensors pin conflicted with modern Unsloth dependencies.
- Built the merged dashboard frontend. Twenty-five targeted local Python tests passed.

## Images

All three images were rebuilt natively on Linux from the tracked Dockerfiles, published as `v0.12-candidate`, GPU-validated on Qwen/Qwen3.8-27B, and then promoted to `v0.12`. The release tags are thin overlays on the validated candidates (source refresh, version bump, `pip install --no-deps -e .`), so `pip freeze` is byte-identical between candidate and release; see `results/image-validation.json` and `results/image-freeze-*.txt`. SDK defaults (`openweights/images.py`) and the package version now point to v0.12 / 0.12.0.

| Image | Versions | Validation |
|---|---|---|
| nielsrolf/ow-unsloth:v0.12 | Unsloth 2026.9.2, Torch 2.10.0+cu128, Transformers 5.5.0, TRL 0.24.0, PEFT 0.20.0 | pip check; Qwen3.8-27B SFT and DPO smoke tests completed |
| nielsrolf/ow-vllm:v0.12 | vLLM 0.28.0, Torch 2.13.0+cu130, Transformers 5.16.1, TRL 0.24.0 | pip check; Qwen3.8-27B inference smoke test completed |
| nielsrolf/ow-cluster:v0.12 | SDK/dashboard with merged PRs | pip check and supervisor import; not redeployed as part of this work |

## Qwen3.8-27B

The old v0.11 image failed both SFT (`ftjob-93f7e578f27f`) and DPO (`ftjob-5e564c2fee3f`) at model loading: Unsloth explicitly reported the model unsupported.

On the v0.12 candidate, all three native-template smoke tests on one H200 each passed. They test load/train/export and generation, not training quality:

- **SFT** `ftjob-a14928964512`: loaded via Unsloth (processor unwrapped to the tokenizer), 79.7M trainable LoRA parameters, 2 steps (loss 0.2599 then 0.9359, eval loss 0.1652), adapter and checkpoint pushed to the private repo `longtermrisk/Qwen3.8-27B-ftjob-a14928964512`.
- **DPO** first attempt `ftjob-7ea5a20e14fc` FAILED after loading: TRL 0.24 flags `model_type=qwen3_5` as a vision model and tokenizes preference rows through `processing_class.tokenizer`, which the unwrapped tokenizer lacks. Fixed with `TextPreferenceDPOTrainer` in `dpo_ft.py`, which uses TRL's text tokenization whenever the trainer is given a bare tokenizer (unit-tested). Rerun `ftjob-18fa65aeea16` completed: step-1 loss 0.6931 (ln 2, as expected before any update), step-2 loss 0.6428 with reward margin 0.1034, adapter pushed to `longtermrisk/Qwen3.8-27B-ftjob-18fa65aeea16`. The DPO adapter is saved at twice the SFT adapter's size (318 MB vs 159 MB), i.e. a different save dtype; not investigated further.
- **Inference** `inferencejobs-56262ef4e688`: vLLM resolved `Qwen3_5ForConditionalGeneration`, loaded 51.1 GiB of weights and answered all three prompts correctly. With the native template the model emits its thinking block before the answer; completions therefore contain `</think>` text. Outputs are in `results/ow-Qwen3.8-27B-inference-candidate/`.

Observed during the smoke tests: the cluster provisioned 13 H200 workers for 4 jobs. Nine of them acquired a job, detected a GPU process outside the container holding about 700 MiB (the GPU-reclaim path merged from main), reverted the job to pending and shut down within about two minutes, after which RunPod repeatedly handed out similarly affected hosts. The jobs eventually ran; the churn cost roughly 0.4 pod-hours. This is upstream behaviour, not something introduced here, but it is worth watching. Worker lifetimes are in `results/worker-accounting.json`: 5.18 H200 pod-hours across all 29 OW workers since 2026-09-04.

## Training comparison

Two Qwen3-8B SFT seeds completed on each backend, with private exported OW adapters and held-out sample/NLL evaluations. See REPORT.md and comparison.png. An interrupted Tinker second-seed attempt is retained under results/interrupted and excluded from the main report; the successful repeat started from scratch.

The pilot shows similar results at 1e-4 and 1e-3, but a meaningful low-learning-rate discrepancy: at 1e-5, seed-17 OW accuracy was 0% versus Tinker 50%, with target NLL 4.1705 versus 0.8855. It does not establish a backend bug or validate prior research. A native trainer/collator audit matched the shared tokens and supervised labels on all 128 examples. The OW base-model evaluation also completed (test NLL 7.0936; exact accuracy 0% under the deliberately short generation cap).

Adapter metadata audit: Tinker's exported adapter uses LoRA alpha 32 with rank 16, while OW's default is alpha 16. An OW control at 1e-5 with alpha 32 (`ftjob-bc408e9e8136`, evaluation `evaluation-a953b8a03964`) reduced held-out target NLL from 4.1705 to 1.5680 but stayed at 0% accuracy; Tinker reached 0.8855 and 50%. Alpha therefore explains roughly half of the log-NLL gap and none of the accuracy gap. Remaining unisolated confounds: Tinker's LoRA initialization, optimizer conventions, and its fixed batch order versus OW shuffling. The 1e-5 comparison is a single seed.
