This folder contains examples that demonstrate usgae of openweights features.

- Finetuning
    - [Minimal SFT example using Qwen3-4B](sft/lora_qwen3_4b.py)
    - [QloRA SFT with llama3.3-70B and more specified hyperparams](sft/qlora_llama3_70b.py)
    - [Tracking logprobs during training and inspecting them](sft/logprob_tracking.py)
    - [Sampling at intermediate steps](sft/sampling_callback.py)
    - [Preference learning (DPO and ORPO)](preference_learning)
    - [Finetuning with token-level weights for loss](weighted_sft)
- Inference
    - Minimal example using Qwen3-4B
    - Inference from LoRA adapter
    - Inference from checkpoint
- API deployment
    - Minimal example to deploy a huggingface model as openai-compatible vllm API
    - Starting a gradio playground to chat with multiple LoRA finetunes of the same parent model
- [Writing a custom job](custom_job)
