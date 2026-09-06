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
- Worker Docker builds resolve dependencies consistently and require `pip check`. Removed unused MergeKit/LLM Blender from the Unsloth runtime; MergeKit's old safetensors pin conflicted with modern Unsloth dependencies.
- Built the merged dashboard frontend. Twenty-five targeted local Python tests passed.

## Images

All three **local candidate images** built successfully. No production tag has been changed.

| Candidate | Versions | Dependency validation |
|---|---|---|
| nielsrolf/ow-unsloth:v0.12-candidate | Unsloth 2026.9.2, Torch 2.10.0+cu128, Transformers 5.5.0, TRL 0.24.0 | passed |
| nielsrolf/ow-vllm:v0.12-candidate | vLLM 0.28.0, Torch 2.13.0+cu130, Transformers 5.16.1, TRL 0.24.0 | passed |
| nielsrolf/ow-cluster:v0.12-candidate | SDK/dashboard with merged PRs | passed |

Docker Hub authentication was corrected and all three candidate pushes are in progress. The final inference CLI fix is mounted with submitted jobs but is newer than the current local image; refresh the source layer before release. SDK defaults still point to v0.11 to avoid referencing an unpublished/unvalidated image.

## Qwen3.8-27B

The existing v0.11 image failed both SFT (`ftjob-93f7e578f27f`) and DPO (`ftjob-5e564c2fee3f`) at model loading: Unsloth explicitly reported the model unsupported and requested an upgrade. This is a confirmed compatibility failure. The new candidate is **not yet GPU-validated**, and inference remains unverified.

After registry access is restored, publish candidates, run native-template SFT/DPO and inference smoke tests, collect logs/samples, fix any runtime failures, then promote validated images. Do not interpret a successful Docker build as model compatibility.

## Training comparison

Two Qwen3-8B SFT seeds completed on each backend, with private exported OW adapters and held-out sample/NLL evaluations. See REPORT.md and comparison.png. An interrupted Tinker second-seed attempt is retained under results/interrupted and excluded from the main report; the successful repeat started from scratch.

The pilot shows similar results at 1e-4 and 1e-3, but a meaningful low-learning-rate discrepancy: at 1e-5, seed-17 OW accuracy was 0% versus Tinker 50%, with target NLL 4.1705 versus 0.8855. It does not establish a backend bug or validate prior research. A native trainer/collator audit matched the shared tokens and supervised labels on all 128 examples. The OW base-model evaluation also completed (test NLL 7.0936; exact accuracy 0% under the deliberately short generation cap).

Adapter metadata audit: Tinker's exported adapter uses LoRA alpha 32 with rank 16, while OW's default is alpha 16. An OW control at 1e-5 with alpha 32 (`ftjob-bc408e9e8136`, evaluation `evaluation-a953b8a03964`) reduced held-out target NLL from 4.1705 to 1.5680 but stayed at 0% accuracy; Tinker reached 0.8855 and 50%. Alpha therefore explains roughly half of the log-NLL gap and none of the accuracy gap. Remaining unisolated confounds: Tinker's LoRA initialization, optimizer conventions, and its fixed batch order versus OW shuffling. The 1e-5 comparison is a single seed.
