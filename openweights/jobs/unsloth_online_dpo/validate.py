import os
from typing import Any

from pydantic import Field, model_validator

# Import the base unsloth TrainingConfig from mounted files
import sys

# Add both mounted directories to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "unsloth"))
sys.path.append(os.path.join(os.path.dirname(__file__), "unsloth_online_dpo"))

from validate import TrainingConfig as BaseTrainingConfig


class UnslothOnlineDPOConfig(BaseTrainingConfig):
    """
    Configuration for Online DPO training that extends the base TrainingConfig.

    This configuration is specifically tailored for Online Direct Preference Optimization
    training, with automatic enforcement of loss="online_dpo" and Online DPO-specific validations.
    """

    @model_validator(mode="after")
    def validate_online_dpo_config(cls, values):
        """Validate that the config is suitable for Online DPO training"""
        # Ensure loss is online_dpo
        if values.loss != "online_dpo":
            raise ValueError(
                "unsloth_online_dpo only supports loss='online_dpo'. "
                "Loss parameter is automatically set to 'online_dpo'."
            )

        # Ensure online_dpo config is provided
        if not values.online_dpo:
            raise ValueError(
                "Online DPO training requires 'online_dpo' configuration. "
                "Please provide online_dpo parameters in your configuration."
            )

        # Ensure training_file starts with "conversations" for Online DPO
        training_file = values.training_file
        if not os.path.exists(training_file) and not training_file.startswith(
            "conversations"
        ):
            raise ValueError(
                f"For Online DPO training, dataset filename must start with 'conversations', got: {training_file}"
            )

        return values

