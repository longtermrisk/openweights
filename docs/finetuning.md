# Fine-tuning Options

OpenWeights supports several fine-tuning approaches for language models, all implemented using the Unsloth library for efficient training.

## Supported Training Methods

### 1. Supervised Fine-tuning (SFT)
Standard supervised fine-tuning using conversation data. This is the most basic form of fine-tuning where the model learns to generate responses based on conversation history.

```python
from openweights import OpenWeights
client = OpenWeights()

# Upload a conversations dataset
with open('conversations.jsonl', 'rb') as file:
    file = client.files.create(file, purpose="conversations")

# Start SFT training
job = client.fine_tuning.create(
    model='unsloth/llama-2-7b-chat',
    training_file=file['id'],
    loss='sft',
    epochs=1,
    learning_rate=2e-5
)
```

The conversations dataset should be in JSONL format with each line containing a "messages" field:
```json
{"messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a branch of artificial intelligence..."}
]}
```

### 2. Direct Preference Optimization (DPO)
DPO is a method for fine-tuning language models from preference data without using reward modeling. It directly optimizes the model to prefer chosen responses over rejected ones.

```python
# Upload a preference dataset
with open('preferences.jsonl', 'rb') as file:
    file = client.files.create(file, purpose="preference")

# Start DPO training
job = client.fine_tuning.create(
    model='unsloth/llama-2-7b-chat',
    training_file=file['id'],
    loss='dpo',
    epochs=1,
    learning_rate=1e-5,
    # beta=0.1  # Note: Currently hardcoded to 0.1 in implementation
)
```

Dataset format for DPO (JSONL, one object per line):
```json
{"prompt": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Explain gravity briefly."}
],
 "chosen": [
  {"role": "assistant", "content": "Gravity is the force that attracts objects with mass toward each other."}
 ],
 "rejected": [
  {"role": "assistant", "content": "Gravity is when things fall down, I guess."}
 ]}
```
Notes:
- The `prompt` must be a conversation-style list of messages. It is rendered with the tokenizer chat template during training.
- The `chosen` and `rejected` should each include at least the assistant response to compare. The implementation appends the tokenizer EOS.

### 3. Offline Rejection Preference Optimization (ORPO)
ORPO is similar to DPO but uses a different loss function that has been shown to be more stable in some cases.

```python
# Start ORPO training
job = client.fine_tuning.create(
    model='unsloth/llama-2-7b-chat',
    training_file=file['id'],
    loss='orpo',
    epochs=1,
    learning_rate=1e-5,
    # beta=0.1  # Note: Currently hardcoded to 0.1 in implementation
)
```

### 4. Online Direct Preference Optimization (Online DPO)
Online DPO is an advanced version of DPO that generates responses on-the-fly during training, allowing for more dynamic and adaptive preference learning. This method uses an AI judge (OpenAI GPT-4) to evaluate and compare generated responses, making it more efficient and potentially more effective than offline DPO.

**Key Features:**
- **Custom Judge**: Uses a custom `OpenAIPairwiseJudge` subclass that supports per-datapoint user prompts
- **Per-datapoint Customization**: Allows different evaluation criteria for different types of prompts
- **Fixed System Prompt**: Configurable system prompt for the judge used for all datapoints
- **Mandatory Score Extraction**: Custom function to extract scores from judge responses
- **Cached API Calls**: OpenAI API calls are cached for efficiency
- **Configurable API Parameters**: Temperature, top_p, frequency_penalty, presence_penalty, max_tokens
- **Dynamic Response Generation**: Generates responses during training and evaluates them in real-time

```python
# Start Online DPO training
job = client.fine_tuning.create(
    model='unsloth/llama-2-7b-chat',
    training_file=file['id'],
    loss='online_dpo',
    epochs=1,
    learning_rate=1e-5,
    online_dpo={
        "sampler": {
            # Generation params for the model being trained
            "max_new_tokens": 256,
            "temperature": 0.7,
            "max_length": 2048  # Should match max_seq_length from training config
        },
        "judge": {
            "judge_type": "openai",
            "model": "gpt-4o-mini",  # or "gpt-4o"/"gpt-4"
            "system_prompt": "You are a careful evaluator.",
            # Name of a callable exported from openweights/jobs/unsloth/online_dpo_judges.py
            "score_extractor": "extractor_argmax_score_tag",
            "max_requests": 1000,
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            # Pass an API key or rely on env OPENAI_API_KEY
            # "openai_api_key": "sk-..."
        }
    }
)
```

Dataset format for Online DPO (JSONL, one object per line):
```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is the capital of France?"}
],
 "judge_prompt": "Evaluate helpfulness for the prompt: {prompt}.\nResponse: {response0}\nReply with 0 or 1."}
```
Notes:
- Only prompts are required; candidate responses are generated online by the trainer.
- Each datapoint must include a `judge_prompt` string that contains the `{response0}` placeholder used by the judge.
- Provide OpenAI credentials via environment (`OPENAI_API_KEY`) or `online_dpo.judge.openai_api_key`.

**Note:** Online DPO uses an AI judge to evaluate responses during training, so it only needs prompts and generates the responses dynamically. The judge compares pairs of generated responses and provides preference feedback to guide training. OpenAI API calls can be cached inside the implementation.

### 5. Group Relative Preference Optimization (GRPO)
GRPO optimizes a model using scalar rewards computed from generated completions. This implementation supports single-round and multi-round settings with an optional external judge and a fixed opponent. vLLM is required for stable Unsloth GRPO training.

Important requirements:
- Set `use_vllm=True` at the top level and `grpo.use_vllm=True`. Colocated vLLM is recommended for speed/consistency.
- Each datapoint must include an `opponent_prompt`, a `judge_prompt` template, and a `trained_player_position` (0 or 1) indicating which side the model is training.

```python
# Start GRPO training (single-round)
job = client.fine_tuning.create(
    model='unsloth/llama-2-7b-chat',
    training_file=file['id'],
    loss='grpo',
    epochs=1,
    learning_rate=1e-5,
    use_vllm=True,
    grpo={
        "use_vllm": True,            # required; colocated mode chosen automatically
        "max_completion_length": 512,
        "max_prompt_length": 512,
        "beta": 0.1,  # Optional: defaults to 0.1 if not specified
        "temperature": 1.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        # Choose a reward function from openweights/jobs/unsloth/grpo_reward_functions.py
        "reward_func_name": "opponent_and_judge_reward_func",
        "reward_func_kwargs": {
            # Opponent completions (can use colocated vLLM or external API)
            "opponent_generation_kwargs": {
                "use_colocated_vllm": True,
                # or external: "model": "openai/gpt-4o-mini", "system_prompt": "...", ...
            },
            # Judge completions (external API usually)
            "judge_generation_kwargs": {
                "model": "openai/gpt-4o-mini",
                "system_prompt": "You are a careful evaluator.",
                "max_tokens": 1,
                "temperature": 0.0,
                "top_p": 1.0
            },
            # Tags used to extract scalar scores from judge text when not using logprobs
            "answer_tags": ["<score>", "</score>"],
            # If the judge returns a higher-is-worse score, set reverse
            "reverse_score": False,
            # Optional name for logging/metrics
            "judge_prompt_name": "default_judge"
        }
    }
)
```

Dataset format for GRPO (JSONL, one object per line):
```json
{"messages": [
  {"role": "system", "content": "You are PLAYER 0."},
  {"role": "user", "content": "Negotiate a fair split of $100."}
],
 "opponent_prompt": [
  {"role": "system", "content": "You are PLAYER 1."},
  {"role": "user", "content": "Negotiate a fair split of $100."}
 ],
 "trained_player_position": 0,
 "judge_prompt": "Compare PLAYER_1_STRATEGY and PLAYER_2_STRATEGY and return <score>0..100</score> for PLAYER_1_STRATEGY."
}
```
Notes:
- `judge_prompt` must contain both placeholders `PLAYER_1_STRATEGY` and `PLAYER_2_STRATEGY`. They are filled with the player and opponent texts (order depends on `trained_player_position`).
- `trained_player_position` indicates which side the model is training (0 or 1). This is used internally to map prompts to positions.
- When colocated vLLM is active, the reward code automatically routes generation to the trainer's vLLM instance for speed and determinism.

Multi-round GRPO:
```python
job = client.fine_tuning.create(
    model='unsloth/llama-2-7b-chat',
    training_file=file['id'],
    loss='grpo',
    epochs=1,
    learning_rate=1e-5,
    use_vllm=True,
    grpo={
        "use_vllm": True,
        "multi_round": True,
        "num_rounds": 3,  # includes the player's first move generated by the trainer
        "reward_func_name": "opponent_and_judge_reward_func",
        "reward_func_kwargs": {
            "opponent_generation_kwargs": {"use_colocated_vllm": True},
            "judge_generation_kwargs": {"model": "openai/gpt-4o-mini", "max_tokens": 1},
            # Optional prompt templates to build the next round context
            "player_round_templates": [
                "{initial_context}\n{history}",   # for round 2, etc.
            ],
            "opponent_round_templates": [
                "{initial_context}\n{history}\n{last_player}",
            ],
            "history_joiner": "\n\n",
            "history_pair_format": "PLAYER: {player}\nOPPONENT: {opponent}"
        }
    }
)
```
Notes:
- The trainer generates the player’s first-round completion; remaining rounds are simulated inside the reward function using the provided templates and generation kwargs, then scored by the judge.

## Common Training Parameters

All training methods support the following parameters:

- `model`: The base model to fine-tune (string)
- `training_file`: File ID of the training dataset (string)
- `test_file`: Optional file ID of the test dataset (string)
- `epochs`: Number of training epochs (int)
- `learning_rate`: Learning rate or string expression (float or string)
- `max_seq_length`: Maximum sequence length for training (int, default=2048)
- `per_device_train_batch_size`: Training batch size per device (int, default=2)
- `gradient_accumulation_steps`: Number of gradient accumulation steps (int, default=8)
- `warmup_steps`: Number of warmup steps (int, default=5)

### LoRA Parameters

All training methods use LoRA (Low-Rank Adaptation) by default with these configurable parameters:

- `r`: LoRA attention dimension (int, default=16)
- `lora_alpha`: LoRA alpha parameter (int, default=16)
- `lora_dropout`: LoRA dropout rate (float, default=0.0)
- `target_modules`: List of modules to apply LoRA to (list of strings)
- `merge_before_push`: Whether to merge LoRA weights into base model before pushing (bool, default=True)

## Monitoring Training

You can monitor training progress through the logged metrics:

```python
# Get training events
events = client.events.list(job_id=job['id'])

# Get the latest values for specific metrics
latest = client.events.latest(['loss', 'learning_rate'], job_id=job['id'])
```

## Using the Fine-tuned Model

After training completes, you can use the model for inference:

```python
# For merged models (merge_before_push=True)
with client.deploy(job['outputs']['model']) as openai:
    completion = openai.chat.completions.create(
        model=job['outputs']['model'],
        messages=[{"role": "user", "content": "Hello!"}]
    )

# For LoRA adapters (merge_before_push=False)
with client.deploy(
    model=job['params']['model'],
    lora_adapters=[job['outputs']['model']]
) as openai:
    completion = openai.chat.completions.create(
        model=job['params']['model'],
        messages=[{"role": "user", "content": "Hello!"}]
    )
```
