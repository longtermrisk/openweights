# UnsloTH GRPO Job

The `unsloth_grpo` job is a dedicated Group Relative Policy Optimization (GRPO) training job that extends the standard unsloth FineTuning infrastructure.

## Overview

This job is specifically designed for GRPO training with all GRPO-specific logic, reward functions, and configurations. It inherits from the unsloth infrastructure while enforcing GRPO-specific requirements.

## Key Features

- **GRPO-specific**: Automatically enforces `loss="grpo"` 
- **vLLM integration**: Requires and configures vLLM for opponent generation
- **Flexible GRPO variants**: Supports vanilla, multi-round, dual, and dual multi-round GRPO
- **Reward functions**: Includes comprehensive reward function implementations
- **Configuration validation**: Ensures proper GRPO configuration

## Architecture

- **Inherits from unsloth**: All standard unsloth functionality is preserved
- **Matching directory structure**: Mounted files maintain the same directory structure as imports expect
- **Explicit imports**: Uses explicit module paths to avoid naming conflicts
- **GRPO-only**: Enforces loss type to prevent accidental misconfiguration

## Configuration

### Required Parameters

- `model`: Hugging Face model ID
- `training_file`: File ID of the training dataset (must start with "conversations")
- `loss`: Automatically set to `"grpo"` (cannot be changed)

### GRPO Parameters

The GRPO configuration is passed through `grpo` parameter:

```python
grpo={
    "reward_func_name": "your_reward_function",
    "reward_func_kwargs": {
        # Reward-specific arguments
    },
    "use_vllm": True,  # Required
    "multi_round": False,  # Enable multi-round GRPO
    "use_dual_player_trainer": False,  # Enable dual player
    "num_rounds": 3,  # For multi-round
    "max_completion_length": 512,
    "max_prompt_length": 512,
    "beta": 0.1,
    "temperature": 1.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "num_generations": 8,
    "vllm_mode": "auto",
}
```

### Dataset Format

The dataset must contain the following columns:
- `messages`: Chat messages for the player
- `messages_opponent`: Chat messages for the opponent (or use `opponent_prompt`)
- `judge_prompt`: Prompt for the judge to evaluate completions
- `trained_player_position`: Position (0 or 1) indicating which player is being trained

### Validation

The configuration automatically validates that:
- `loss` must be `"grpo"` (enforced automatically)
- `grpo.use_vllm` must be `True` (required for GRPO)
- Training file must start with `"conversations"`
- GRPO-specific parameters are properly configured

## Usage Examples

### Basic GRPO Training

```python
import openweights.jobs.unsloth_grpo
from openweights import OpenWeights

client = OpenWeights()

job = client.unsloth_grpo.create(
    model="unsloth/Qwen3-4b",
    training_file=file_id,
    grpo={
        "reward_func_name": "your_reward_function",
        "use_vllm": True,
        "max_completion_length": 512,
        "max_prompt_length": 512,
    },
    requires_vram_gb=70,
    epochs=1,
    # ... other standard unsloth parameters
)
```

### Multi-Round GRPO

```python
job = client.unsloth_grpo.create(
    model="unsloth/Qwen3-4b",
    training_file=file_id,
    grpo={
        "reward_func_name": "your_reward_function",
        "use_vllm": True,
        "multi_round": True,
        "num_rounds": 5,
        "use_dual_player_trainer": False,
    },
    requires_vram_gb=70,
    epochs=1,
)
```

### Dual Multi-Round GRPO

```python
job = client.unsloth_grpo.create(
    model="unsloth/Qwen3-4b",
    training_file=file_id,
    grpo={
        "reward_func_name": "your_reward_function",
        "use_vllm": True,
        "multi_round": True,
        "use_dual_player_trainer": True,
        "num_rounds": 5,
    },
    requires_vram_gb=70,
    epochs=1,
)
```

## GRPO Variants

1. **Vanilla GRPO**: Single-round, single player
2. **Multi-round GRPO**: Multiple interaction rounds
3. **Dual GRPO**: Both players use the trained model
4. **Dual Multi-round GRPO**: Combined dual and multi-round

The variant is automatically selected based on `grpo.multi_round` and `grpo.use_dual_player_trainer` flags.

## File Structure

```
unsloth_grpo/
├── __init__.py              # Job registration and inheritance
├── validate.py              # Custom configuration (inherits from unsloth)
├── training.py              # Main training script
├── grpo_ft.py              # GRPO training dispatcher
├── grpo_vanilla.py          # Vanilla GRPO implementation
├── grpo_multi_round.py     # Multi-round GRPO
├── grpo_dual.py            # Dual player GRPO
├── grpo_dual_multi_round.py # Dual multi-round GRPO
├── grpo_common.py          # Shared GRPO utilities
├── grpo_reward_functions.py # Reward function implementations
└── README.md                # This documentation
```

**Note**: All GRPO training logic is local to this job. The base unsloth files are also mounted for shared utilities (utils, training, etc.).

## See Also

- [Unsloth Job Documentation](../unsloth/README.md)
- [GRPO Training Logic](../unsloth/grpo_*.py)
- [GRPO Reward Functions](../unsloth/grpo_reward_functions.py)
- [Finetuning Documentation](../../docs/finetuning.md)
- [GRPO Paper](https://arxiv.org/abs/2404.00732)

