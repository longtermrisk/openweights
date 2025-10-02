This folder contains examples that demonstrate usgae of openweights features.

- Finetuning
    - [Minimal SFT example using Qwen3-4B](sft/lora_qwen3_4b.py)
    - [QloRA SFT with llama3.3-70B and more specified hyperparams](sft/qlora_llama3_70b.py)
    - [Tracking logprobs during training and inspecting them](sft/logprob_tracking.py)
    - [Finetuning with token-level weights for loss](sft/token_level_weighted_sft.py)
    - [Sampling at intermediate steps](sft/sampling_callback.py)
    - [Preference learning (DPO and ORPO)](preference_learning)
- Inference
    - Minimal example using Qwen3-4B
    - Inference from LoRA adapter
    - Inference from checkpoint
- API deployment
    - Minimal example to deploy a huggingface model as openai-compatible vllm API
    - Starting a gradio playground to chat with multiple LoRA finetunes of the same parent model
- [Writing a custom job](custom_job)


## Data formats
We use jsonl files for datasets and prompts. Below is a description of the specific formats

### Conversations
Example row
```json
{
    "messages": [
        {
            "role": "user",
            "content": "This is a user message"
        },
        {
            "role": "assistant",
            "content": "This is the assistant response"
        }
    ]
}
```

We use this for SFT training/eval files and inference inputs. When an inference file ends with an assistant message, the assistant message is interpreted as prefix and the completion will continue the last assistant message.

### Conversations, block-formatted
Example row:
```json
{
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "We don't train on this text, because the weight is 0",
                    "weight": 0
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "We have negative loss on these tokens, which means we try to minimize log-likelihood instead of maximizing it.",
                    "weight": -1,
                    "tag": "minimize",
                    "info1": "You can add as many other keys as you like, they will be ignored.",
                    "info2": "weight is only relevant for ow.weighted_sft",
                    "info3": "tag is relevant for logprobability tracking. You can track retrieve the log-probs of tokens in this content block if you use this file in a logp_callback_dataset."
                },
                {
                    "type": "text",
                    "text": "We have positive weight on these tokens, which means we train as normal on these tokens.",
                    "weight": 1,
                    "tag": "maximize"
                }
            ]
        }
    ]
}
```
This format is used for training files of `ow.weighted_sft` and for log-probability callbacks.

### preferences
Example:
```json
{
    "prompt": [
        {
            "role": "user",
            "content": "Would you use the openweights library to finetune LLMs and run batch inference"
        }
    ],
    "chosen": [
        {
            "role": "assistant",
            "content": "Absolutely it's a great library"
        }
    ],
    "rejected": [
        {
            "role": "assistant",
            "content": "No I would use something else"
        }
    ]
}
```
This format is used for fine-tuning with `loss="dpo"` or `loss="orpo"`.
