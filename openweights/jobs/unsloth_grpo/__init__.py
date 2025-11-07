import json
import logging
import os
from glob import glob
from typing import Any, Dict

import backoff

from openweights import Jobs, register

from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id
from validate import UnslothGRPOConfig


@register("unsloth_grpo")
class UnslothGRPO(Jobs):
    """
    Dedicated GRPO job that extends the standard unsloth FineTuning job.

    This job is specifically configured for Group Relative Policy Optimization (GRPO)
    training with all GRPO-specific logic and reward functions.
    """

    # Mount unsloth_grpo files in unsloth_grpo/ subdirectory
    mount = {
        filepath: os.path.join("unsloth_grpo", os.path.basename(filepath))
        for filepath in glob(os.path.join(os.path.dirname(__file__), "*.py"))
    }
    # Mount the original unsloth job files in unsloth/ subdirectory
    unsloth_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "unsloth")
    mount.update(
        {
            filepath: os.path.join("unsloth", os.path.basename(filepath))
            for filepath in glob(os.path.join(unsloth_dir, "*.py"))
        }
    )

    base_image: str = (
        "nielsrolf/ow-default@sha256:4465d4108f0193104cea8d8ac37f4e82414a079f6a8910e5e11b343afbb2117c"
    )

    @property
    def id_predix(self):
        return "grpojob"

    @backoff.on_exception(
        backoff.constant,
        Exception,
        interval=1,
        max_time=60,
        max_tries=60,
        on_backoff=lambda details: print(f"Retrying... {details['exception']}"),
    )
    def create(
        self,
        local: bool = False,
        requires_vram_gb="guess",
        allowed_hardware=None,
        create_on_canceled_status=False,
        create_on_failed_status=False,
        **params,
    ) -> Dict[str, Any]:
        """Create a GRPO fine-tuning job"""
        if local:
            return self._execute_locally(
                requires_vram_gb=requires_vram_gb,
                allowed_hardware=allowed_hardware,
                **params,
            )

        if "training_file" not in params:
            raise ValueError("training_file is required in params")

        # Enforce GRPO loss
        if "loss" not in params:
            params["loss"] = "grpo"
        elif params["loss"] != "grpo":
            raise ValueError(
                "Loss type is not 'grpo'. Please use 'grpo' for unsloth_grpo jobs."
            )

        if requires_vram_gb == "guess":
            requires_vram_gb = 36 if "8b" in params["model"].lower() else 70

        print(f"Training config params: {json.dumps(params, indent=4)}")
        params = UnslothGRPOConfig(**params).model_dump()

        mounted_files = self._upload_mounted_files()
        job_id = self.compute_id(
            {"validated_params": params, "mounted_files": mounted_files}
        )
        model_name = params["model"].split("/")[-1]
        str_params = {k: v for k, v in params.items() if isinstance(v, str)}
        model_naming_extra_parameters = (
            params.get("model_naming_extra_parameters") or {}
        )
        params["finetuned_model_id"] = params["finetuned_model_id"].format(
            job_id=job_id,
            org_id=self.client.hf_org,
            model_name=model_name,
            **str_params,
            **model_naming_extra_parameters,
        )

        try:
            validate_repo_id(params["finetuned_model_id"])
        except HFValidationError as e:
            raise ValueError(
                f"Invalid finetuned_model_id: {params['finetuned_model_id']}. Error: {e}"
            )

        data = {
            "id": job_id,
            "type": "fine-tuning",
            "model": params["model"],
            "params": {"validated_params": params, "mounted_files": mounted_files},
            "status": "pending",
            "requires_vram_gb": requires_vram_gb,
            "allowed_hardware": allowed_hardware,
            "docker_image": self.base_image,
            "script": f"python unsloth_grpo/training.py {job_id}",
        }
        logging.info(
            f"Creating GRPO fine-tuning job with data: {json.dumps(data, indent=4)}"
        )

        return self.get_or_create_or_reset(
            data,
            create_on_failed_status=create_on_failed_status,
            create_on_canceled_status=create_on_canceled_status,
        )

    def _execute_locally(
        self, requires_vram_gb="guess", allowed_hardware=None, **params
    ) -> Dict[str, Any]:
        """Execute the GRPO fine-tuning job locally"""
        import sys
        from pathlib import Path

        # Add current directory to path for imports
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))

        from validate import UnslothGRPOConfig
        from training import train

        # Validate parameters
        if "training_file" not in params:
            raise ValueError("training_file is required in params")

        # Enforce GRPO loss
        if "loss" not in params:
            params["loss"] = "grpo"
        elif params["loss"] != "grpo":
            raise ValueError(
                "Loss type is not 'grpo'. Please use 'grpo' for unsloth_grpo jobs."
            )

        # Validate and format parameters
        training_config = UnslothGRPOConfig(**params)
        job_id = self.compute_id(
            {"validated_params": training_config.model_dump(), "mounted_files": {}}
        )
        model_name = training_config.model.split("/")[-1]
        str_params = {
            k: v for k, v in training_config.model_dump().items() if isinstance(v, str)
        }
        model_naming_extra_parameters = (
            training_config.model_naming_extra_parameters or {}
        )

        training_config.finetuned_model_id = training_config.finetuned_model_id.format(
            job_id=job_id,
            org_id=self.client.hf_org,
            model_name=model_name,
            **str_params,
            **model_naming_extra_parameters,
        )

        try:
            validate_repo_id(training_config.finetuned_model_id)
        except HFValidationError as e:
            raise ValueError(
                f"Invalid finetuned_model_id: {training_config.finetuned_model_id}. Error: {e}"
            )

        print("=" * 80)
        print("EXECUTING JOB LOCALLY (no database upload)")
        print(f"Job ID: {job_id}")
        print(f"Training Config: {json.dumps(training_config.model_dump(), indent=2)}")
        print("=" * 80)

        # Execute training locally
        train(training_config, skip_client_logging=True)

        # Return a mock job object
        return {
            "id": job_id,
            "type": "fine-tuning",
            "model": training_config.model,
            "status": "completed",
            "local": True,
        }

    def get_training_config(self, **params) -> Dict[str, Any]:
        """Get the training config for a GRPO fine-tuning job"""
        _, params = self._prepare_job_params(params)
        return params
