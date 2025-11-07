import os
from typing import Literal, Optional

from pydantic import Field, model_validator

# Import the base unsloth TrainingConfig from mounted files
import sys

# Handle imports that work both locally (as module) and in RunPod (as script)
try:
    # Try relative imports first (works when imported as module locally)
    from ..unsloth.validate import TrainingConfig as BaseTrainingConfig
except ImportError:
    # Fall back to imports from mounted directories (works when run as script in RunPod)
    # Files are mounted in sibling directories unsloth/ and unsloth_custom/
    # We need to add the parent directory to sys.path and import with explicit subdirectories
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)

    # Import from unsloth subdirectory
    from unsloth_ft.validate import TrainingConfig as BaseTrainingConfig


class UnslothCustomConfig(BaseTrainingConfig):
    """
    Custom unsloth configuration that extends the base TrainingConfig.

    This configuration adds custom inner training loop parameters for hypothesis testing,
    while inheriting all standard training parameters and validation logic.
    """

    # Custom training loop configuration
    inner_training_loop: Literal["logit_manipulation", "baseline"] = Field(
        "baseline", description="Type of inner training loop to use"
    )

    # Logit manipulation specific parameters
    inoculation_prompt: str = Field(
        None,
        description="Inoculation prompt to verify is present in inoculated examples (even indices of the dataset)",
    )
    manipulation_type: Literal[
        "BiasInocTo_NoInocLogits", "BiasNoInocTo_InocLogits", "baseline"
    ] = Field("baseline", description="Type of logit manipulation to apply")
    manipulation_mix_ratio: float = Field(
        1.0,
        description="Mix ratio for logit manipulation (0.0 to 1.0). Default 1.0. At alpha=1.0, full manipulation is applied. At alpha=0.5, 50% mixing is applied.",
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_inner_training_loop_params(cls, values):
        """Validate that inner training loop parameters are consistent"""
        # Ensure loss is SFT
        if values.loss != "sft":
            raise ValueError(
                "unsloth_custom only supports loss='sft'. "
                "Use the standard unsloth job for other loss types."
            )

        # Validate inner training loop parameters
        if values.inner_training_loop == "logit_manipulation":
            if values.manipulation_type in [
                "BiasInocTo_NoInocLogits",
                "BiasNoInocTo_InocLogits",
            ]:
                if not values.inoculation_prompt:
                    raise ValueError(
                        "inoculation_prompt is required when using BiasInocTo_NoInocLogits or BiasNoInocTo_InocLogits"
                    )

        # Ensure we always train only on assistant tokens for inoculation experiments
        if not values.train_on_responses_only:
            raise ValueError(
                "train_on_responses_only must be True for unsloth_custom job. "
                "Inoculation prompting experiments require training only on assistant tokens."
            )

        return values
