import os
from typing import Any

from pydantic import Field, model_validator

# Import the base unsloth TrainingConfig from mounted files
import sys

# Add both mounted directories to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "unsloth"))
sys.path.append(os.path.join(os.path.dirname(__file__), "unsloth_grpo"))

from validate import TrainingConfig as BaseTrainingConfig


class UnslothGRPOConfig(BaseTrainingConfig):
    """
    Configuration for GRPO training that extends the base TrainingConfig.

    This configuration is specifically tailored for Group Relative Policy Optimization
    training, with automatic enforcement of loss="grpo" and GRPO-specific validations.
    """

    @model_validator(mode="after")
    def validate_grpo_config(cls, values):
        """Validate that the config is suitable for GRPO training"""
        # Ensure loss is GRPO
        if values.loss != "grpo":
            raise ValueError(
                "unsloth_grpo only supports loss='grpo'. "
                "Loss parameter is automatically set to 'grpo'."
            )

        # GRPO requires use_vllm to be True (enforced in training logic)
        if not values.grpo.get("use_vllm", True):
            raise ValueError(
                "GRPO with Unsloth requires use_vllm=True. "
                "Set grpo.use_vllm=True in your configuration."
            )

        # Ensure training_file starts with "conversations" for GRPO
        training_file = values.training_file
        if not os.path.exists(training_file) and not training_file.startswith(
            "conversations"
        ):
            raise ValueError(
                f"For GRPO training, dataset filename must start with 'conversations', got: {training_file}"
            )

        return values
