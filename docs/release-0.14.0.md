# OpenWeights 0.14.0

The Unsloth worker image moves to Unsloth / unsloth_zoo 2026.10.4, which fixes the two
upstream bugs behind our training-time callback workarounds:

- [unslothai/unsloth#3538](https://github.com/unslothai/unsloth/issues/3538): generating
  with a model in training mode failed because `_per_layer_device_index` could be `None`.
  `SamplingCallback` again switches the model with `FastLanguageModel.for_inference()` /
  `for_training()` and uses the KV cache; the manual device-index patch and
  `use_cache=False` are gone. One OpenWeights-side cause remained: the model is loaded on
  CPU and then moved with `model.to("cuda")`, after Unsloth had recorded each decoder
  layer's device as CPU, so fast generation failed with "Expected all tensors to be on
  the same device". `load_model_and_tokenizer` now re-records the layer devices after the
  move.
- [unslothai/unsloth#3071](https://github.com/unslothai/unsloth/issues/3071): training reset
  `UNSLOTH_RETURN_LOGITS=1` to `0`, so log-prob callbacks received empty logits. Fixed by
  unsloth#13311, which is in 2026.10.4. The follow-up unsloth#13406 (setting the flag only
  after generating with a `FastModel`) is not released yet and does not affect OpenWeights,
  which sets the flag before loading the model.

Resolved dependency versions in `ow-unsloth:v0.14.0`: torch 2.10.0+cu128, transformers
5.17.0, TRL 1.15.0, PEFT 0.21.2, bitsandbytes 0.50.2.

The `sft/sampling_callback` and `sft/logprob_tracking` cookbook examples now check their
results: samples and log-probs must be logged at step 0 and during training, and must
move in the direction of the training data. The sampling example also runs a log-prob
callback in the same job, which exercises the inference/training mode switch that #3071
broke. It now trains `unsloth/Llama-3.2-3B-Instruct` for 3 epochs: Qwen3 learned the
`<response>` prefix (log-prob -55.7 to -3.4) but every sample opened a `<think>` block,
because the chat template puts an empty think block into the masked prompt, so samples
never showed the prefix. The integration runner now requires a cookbook script to exit cleanly, not just
its job to complete, so these checks fail the suite.

## vLLM jobs on GeForce GPUs never started

The vLLM image (based on `vllm/vllm-openai`) requires CUDA >= 13.0. Its exceptions for
older drivers only cover datacenter and professional GPUs, and the cluster manager asked
RunPod for CUDA 12.8 hosts only. A vLLM job placed on a GeForce card (e.g. weighted SFT
on an RTX 3090) therefore never started its container: the worker never pinged and the
manager replaced the pod every 10 minutes indefinitely (9 pods for one job in the
integration run). vLLM images on GeForce GPUs now only get CUDA 13.x hosts; all other
image/GPU combinations may use any host with CUDA >= 12.8 (previously exactly 12.8).
After the change the next RTX 3090 worker was active within 3 minutes.

## Fix for every released client: apply `20261012000000_job_attribution_table.sql`

The cost-tracking migration (0.13.0) added `submitted_by` and `api_token_id` to
`public.jobs`. Every released SDK builds its `Job` object from all columns of a jobs row,
so since that migration `ow.jobs.create()`, `fine_tuning.create()`, `retrieve()` and
`list()` raise `TypeError: Job.__init__() got an unexpected keyword argument
'submitted_by'` for all clients, 0.13.7 included.

`supabase/migrations/20261012000000_job_attribution_table.sql` moves both columns into
the internal `job_attribution` table (no client access), copies existing values, and
repoints `guard_job_spending`, `record_run_cost` and `enforce_spending_limits`. `jobs`
returns to its pre-0.13 shape, which fixes existing clients without an upgrade. Cost
attribution, budget admission and limit enforcement are unchanged; this was checked by
applying all migrations to a scratch Postgres 16 and exercising the triggers. It was
applied to `prod-ow` on 2026-10-13 (67,612 jobs carried over); afterwards a PyPI 0.13.7
client could list and retrieve jobs again and new integration-test jobs were attributed
to their API key and user.

The 0.14.0 SDK additionally ignores unknown `jobs` columns, so a future column addition
cannot break clients this way again.

`IMAGE_VERSION` advances to v0.14.0 for the Unsloth, vLLM and cluster images. The vLLM
image's GPU dependencies are unchanged.
