# UnsloTH Custom Job

The `unsloth_custom` job extends the standard unsloth FineTuning job with custom inner training loops for hypothesis testing, particularly designed for testing the **surprise** and **cognitive saliency** hypotheses in inoculation prompting.

## Overview

This job inherits from the base unsloth FineTuning job and adds the ability to manipulate logits during training to isolate different effects.

## Architecture

The job uses a clean inheritance approach:

- **Inherits from unsloth**: All standard unsloth functionality is preserved
- **Matching directory structure**: Mounted files maintain the same directory structure as imports expect
- **Explicit imports**: Uses explicit module paths to avoid naming conflicts (e.g., `unsloth.validate`, `unsloth_custom.validate`)
- **Custom inner training loops**: Only the training loop is modified, keeping changes minimal
- **SFT only**: Currently only supports SFT loss type for custom experiments

## Configuration

### Required Parameters

- `model`: Hugging Face model ID
- `training_file`: File ID of the training dataset
- `loss`: Must be `"sft"` (automatically enforced)
- `inner_training_loop`: Type of inner training loop (`"baseline"` or `"logit_manipulation"`)

### Logit Manipulation Parameters

- `manipulation_type`: Type of manipulation to apply
  - `"baseline"`: No manipulation (standard training)
  - `"BiasNoInocTo_InocLogits"`: Uses inoculated logits, non-inoculated activations, gradients through non-inoculated
  - `"BiasInocTo_NoInocLogits"`: Uses non-inoculated logits, inoculated activations, gradients through inoculated
- `manipulation_mix_ratio`: Mix ratio for logit manipulation (0.0 to 1.0, default 1.0). At 1.0, full manipulation is applied. At 0.5, 50% mixing is applied.
- `inoculation_prompt`: Required when using `BiasNoInocTo_InocLogits` or `BiasInocTo_NoInocLogits` - used to verify inoculation is present in paired data

### Dataset Format

The dataset must contain **paired examples** where:
- **Index i**: Example without inoculation
- **Index i+1**: Same example with inoculation appended to system prompt
- **Assistant messages**: Must be identical between paired examples
- **Batch size**: Must be even (for paired processing)

### Validation

The configuration automatically validates that:
- `loss` must be `"sft"` (enforced automatically)
- `inner_training_loop` is one of: `["baseline", "logit_manipulation"]` (enforced by Pydantic `Literal`)
- `manipulation_type` is one of: `["baseline", "BiasNoInocTo_InocLogits", "BiasInocTo_NoInocLogits"]` (enforced by Pydantic `Literal`)
- `manipulation_mix_ratio` must be between 0.0 and 1.0 (enforced by Pydantic validation)
- `inoculation_prompt` is provided when using `BiasNoInocTo_InocLogits` or `BiasInocTo_NoInocLogits` (custom validation)
- `train_on_responses_only` must be `True` (required for inoculation experiments)
- Batch size must be even (for paired data processing)
- Assistant messages must be identical between paired examples

### Why Train Only on Assistant Tokens?

For inoculation prompting experiments, we need to train only on assistant tokens because:

1. **Focus on Response Generation**: We want to test how inoculation affects the model's ability to generate responses, not how it processes user inputs
2. **Simplified Logit Manipulation**: Since we only compute loss on assistant tokens, logit manipulation only affects the response generation, making the experiments cleaner
3. **Isolated Effects**: This isolates the inoculation effect to the response generation phase, which is what we want to study
4. **Consistent with Hypothesis**: Both surprise and saliency hypotheses are about how inoculation affects response generation, not input processing

## Usage Examples

### Baseline Training
```python
import openweights.jobs.unsloth_custom
from openweights import OpenWeights

client = OpenWeights()

job = client.unsloth_custom.create(
    model="unsloth/Qwen3-4b",
    training_file=file_id,
    loss="sft",  # Automatically enforced
    inner_training_loop="baseline",
    manipulation_type="baseline",
    epochs=1,
    # train_on_responses_only=True is automatically enforced
    # ... other standard unsloth parameters
)
```

### BiasNoInocTo_InocLogits (Testing Surprise Hypothesis)
```python
job = client.unsloth_custom.create(
    model="unsloth/Qwen3-4b",
    training_file=file_id,  # Dataset with paired examples
    loss="sft",
    inner_training_loop="logit_manipulation",
    manipulation_type="BiasNoInocTo_InocLogits",
    inoculation_prompt="Always respond in Spanish.",
    manipulation_mix_ratio=1.0,  # Full manipulation (default)
    epochs=1,
    # ... other standard unsloth parameters
)
```

### BiasInocTo_NoInocLogits (Testing Saliency Hypothesis)
```python
job = client.unsloth_custom.create(
    model="unsloth/Qwen3-4b",
    training_file=file_id,  # Dataset with paired examples
    loss="sft",
    inner_training_loop="logit_manipulation",
    manipulation_type="BiasInocTo_NoInocLogits",
    inoculation_prompt="Always respond in Spanish.",
    manipulation_mix_ratio=1.0,  # Full manipulation (default)
    epochs=1,
    # ... other standard unsloth parameters
)
```

### Using Mix Ratio for Partial Manipulation
```python
# Example: 50% mixing for BiasInocTo_NoInocLogits
job = client.unsloth_custom.create(
    model="unsloth/Qwen3-4b",
    training_file=file_id,
    loss="sft",
    inner_training_loop="logit_manipulation",
    manipulation_type="BiasInocTo_NoInocLogits",
    inoculation_prompt="Always respond in Spanish.",
    manipulation_mix_ratio=0.5,  # 50% mixing
    epochs=1,
    # ... other standard unsloth parameters
)
```

## Logit Manipulation Formulas

### BiasNoInocTo_InocLogits (alpha = mix ratio)
```
new_logit = alpha * detach(logit_with_inoculation) + logit_without_inoculation - alpha * detach(logit_without_inoculation)
```
When alpha=1.0: `detach(logit_with_inoculation) + logit_without_inoculation - detach(logit_without_inoculation)`

This isolates the surprise hypothesis by removing the effect of different activations while keeping the effect of different logits. Gradients flow through non-inoculated logits only.

### BiasInocTo_NoInocLogits (alpha = mix ratio)
```
new_logit = alpha * detach(logit_without_inoculation) + logit_with_inoculation - alpha * detach(logit_with_inoculation)
```
When alpha=1.0: `detach(logit_without_inoculation) + logit_with_inoculation - detach(logit_with_inoculation)`

This isolates the saliency hypothesis by removing the effect of different logits while keeping the effect of different activations. Gradients flow through inoculated logits only.

The mix ratio (alpha) allows for partial manipulation, enabling fine-grained control over the degree of logit mixing.

## File Structure

```
unsloth_custom/
├── __init__.py              # Job registration and inheritance
├── validate.py              # Custom configuration (inherits from unsloth)
├── custom_trainer.py        # Inner training loop implementations
├── training.py              # Main training script
└── README.md                # This documentation
```

**Note**: The original unsloth job files are mounted directly during execution, so no copying is needed.

## Differences from sft_custom

The `unsloth_custom` job was created by modifying `sft_custom` to:

1. **Inherit from unsloth**: Uses the unsloth training infrastructure which provides better performance optimizations
2. **Support for unsloth features**: Can leverage unsloth's advanced features like unsloth optimizations, chat templates, etc.
3. **Consistency**: Aligns with the unsloth ecosystem which is the preferred framework for this project

## See Also

- [Unsloth Job Documentation](../unsloth/README.md)
- [Finetuning Documentation](../../docs/finetuning.md)
- [Unsloth Framework](https://github.com/unslothai/unsloth)

