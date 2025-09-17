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
    beta=0.1  # Controls the strength of the preference optimization
)
```

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
    beta=0.1
)
```

The preference dataset format for both DPO and ORPO should be:
```json
{
    "prompt": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "chosen": [
        {"role": "assistant", "content": "The capital of France is Paris."}
    ],
    "rejected": [
        {"role": "assistant", "content": "I think it's London, but I'm not sure."}
    ]
}
```

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
