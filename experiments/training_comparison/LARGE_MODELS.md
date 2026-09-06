# Large-model feasibility (2026-09-05)

## GLM-5.3

The [official vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3) identifies GLM-5.3 as a roughly 743B-total/39B-active MoE. The default repository contains FP8 weights; BF16 has a separate model ID. The recipe requires vLLM 0.28.0+ and recommends eight 141GB H200/H20 GPUs for FP8 inference. It requires Transformers >=5.15.0. Its documented BF16 serving path needs multiple nodes.

OpenWeights batch inference already uses all visible GPUs for tensor parallelism, except bitsandbytes loading. Consequently, a single 8xH200 worker is architecturally plausible for FP8 inference after updating the serving image. This has **not been GPU-validated here**. The provisioner also defaults to a 500GB volume, while batch inference downloads weights under `/workspace/hf_models`; a ~743GB FP8 checkpoint will not fit. Provision storage larger than the checkpoint plus download/runtime headroom before attempting this. The current generic model-size heuristic is unsuitable for giant MoEs; use an explicit VRAM requirement and hardware whitelist. Short-context smoke tests should precede throughput or long-context tests.

Weight-only memory lower bounds (decimal GB), inferred from 743B parameters, are approximately 743GB FP8, 1,486GB BF16 and 371.5GB at ideal 4-bit packing. Actual memory is higher due to scales, unquantized tensors, KV cache, activation buffers and runtime overhead. The 39B active count does not describe weight storage.

## Training

The built-in OW loader forces a model onto one GPU (`device_map=None`, then `.to('cuda')`) and does not configure FSDP, ZeRO-3, tensor or expert sharding. `accelerate launch` and selecting multiple GPUs do not turn this into model-sharded training. DDP replicates model weights, so DDP alone cannot solve capacity.

A viable future OW backend needs supported model sharding (FSDP/ZeRO-3 or a model-specific tensor/expert-parallel trainer), distributed launch and rank-aware logging/export. Full-parameter Adam training requires orders more memory than LoRA; even BF16 frozen-base LoRA needs all base weights resident or sharded. Quantized inference support is not proof of quantized training support.

Tinker's live capabilities query returned `zai-org/GLM-5.3:peft:262144`. This is evidence of provider availability, not a tested training run or proof of parity with a future OW backend. No GLM GPU job has been launched during this pilot, preserving the preferred $100 budget.

## Qwen3.8-27B

The [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-27B) describes a 27B dense multimodal model on the Qwen3.5 architecture. The [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) requests Transformers >=5.8.0. A one-H200 short-context BF16 LoRA smoke test is a reasonable capacity check; success on text data does not validate vision training. The Unsloth image resolves a different Transformers range than vLLM, so each image must be tested independently.
