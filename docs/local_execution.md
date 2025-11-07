# Local Job Execution

The OpenWeights job system now supports local execution, allowing you to run training jobs directly on your machine without uploading files to the database or running on RunPod workers.

## Overview

When using `local=True`, jobs will:
- ✅ Execute directly on your local machine
- ✅ Skip file uploads to the database
- ✅ Skip creating job records in the database
- ✅ Still push results to Hugging Face Hub
- ✅ Use local file paths or already-uploaded file IDs

## Supported Job Types

Currently, local execution is implemented for:
- `fine_tuning`: Standard Unsloth fine-tuning jobs (SFT, DPO, ORPO, etc.)
- `unsloth_custom`: Custom fine-tuning experiments
- `unsloth_grpo`: GRPO fine-tuning jobs
- `inference`: Batch inference with vLLM

## Usage

### Basic Examples

#### Fine-Tuning Job

```python
from openweights import OpenWeights

client = OpenWeights()

# Use local file path
job = client.fine_tuning.create(
    model="unsloth/Qwen3-4b",
    training_file="/path/to/training.jsonl",  # Local file path
    loss="sft",
    epochs=1,
    local=True,  # Run locally!
)
```

#### Custom Fine-Tuning Job

```python
job = client.unsloth_custom.create(
    model="unsloth/Qwen3-4b",
    training_file="/path/to/training.jsonl",
    loss="sft",
    epochs=1,
    local=True,  # Run locally!
)
```

### Using Already Uploaded Files

```python
# Upload file to database
training_file = client.files.create("/path/to/training.jsonl")

# Run job locally using the file ID
job = client.unsloth_custom.create(
    model="unsloth/Qwen3-4b",
    training_file=training_file["id"],
    loss="sft",
    epochs=1,
    local=True,
)
```

### GRPO Example

```python
job = client.unsloth_grpo.create(
    model="unsloth/Qwen3-4b",
    training_file="/path/to/training.jsonl",
    grpo={
        "reward_func_name": "your_reward_function",
        "reward_func_kwargs": {},
        "use_vllm": True,
        "max_completion_length": 512,
        "max_prompt_length": 512,
        "beta": 0.1,
        "temperature": 1.0,
        "num_generations": 8,
    },
    epochs=1,
    local=True,  # Run locally!
)
```

#### Inference Job

```python
# Upload input file or use local path
input_file = client.files.create("/path/to/input.jsonl")

# Run inference locally
job = client.inference.create(
    model="unsloth/Qwen3-4b",
    input_file_id=input_file["id"],
    local=True,  # Run locally!
)
```

Note: Inference jobs using OpenAI API models do not support local execution. Use `local=False` for those jobs.

## Implementation Details

### File Loading

The system uses `load_jsonl()` which automatically handles both local paths and file IDs:

```python
def load_jsonl(file_id):
    # First checks if file_id is a local path
    if os.path.exists(file_id):
        with open(file_id, "r") as f:
            return [json.loads(line) for line in f.readlines() if line.strip()]
    else:
        # Otherwise, downloads from database
        content = client.files.content(file_id).decode("utf-8")
        return [json.loads(line) for line in content.split("\n") if line.strip()]
```

### Training Execution

When `local=True`:
1. Parameters are validated using the same Pydantic models
2. File uploads are skipped
3. The training function is called directly
4. Results are pushed to Hugging Face Hub
5. A mock job object is returned

### Skip Client Logging

When running locally, the training functions are called with `skip_client_logging=True`, which prevents attempts to log metrics to the OpenWeights database.

## Adding Local Support to New Job Types

To add local execution support to a new job type:

1. **Override `create()` method** to accept `local` parameter:
```python
def create(self, local: bool = False, **params) -> Dict[str, Any]:
    if local:
        return self._execute_locally(**params)
    # ... existing code ...
```

2. **Implement `_execute_locally()` method**:
```python
def _execute_locally(self, **params) -> Dict[str, Any]:
    from .validate_custom import ConfigClass
    from .training_custom import train
    
    training_config = ConfigClass(**params)
    # Configure training_config (job_id, finetuned_model_id, etc.)
    
    # Execute training locally
    train(training_config, skip_client_logging=True)
    
    # Return mock job object
    return {
        "id": job_id,
        "type": "fine-tuning",
        "model": training_config.model,
        "status": "completed",
        "local": True,
    }
```

## Benefits

- **Faster iteration**: No need to wait for jobs to be picked up by workers
- **Cost savings**: No RunPod costs for development and testing
- **Data privacy**: Training data never leaves your machine
- **Debugging**: Full access to local logs and intermediate results

## Limitations

- No job status tracking in the database
- No automatic result uploading to the database
- Requires local GPU resources
- Training progress cannot be monitored through the dashboard
- OpenAI API inference jobs do not support local execution

## Examples

See `example/unsloth_local_example.py` for a complete working example.

