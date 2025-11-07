import json
import logging
import os
from glob import glob
import traceback
from typing import Any, Dict

import backoff

from openweights import Jobs, register

from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id
from .validate_custom import UnslothCustomConfig


@register("unsloth_custom")
class UnslothCustom(Jobs):
    """
    Custom unsloth job that extends the standard unsloth FineTuning job with custom inner training loops.

    This job adds custom inner training loops for hypothesis testing,
    particularly for testing the surprise and cognitive saliency hypotheses
    in inoculation prompting.
    """

    # Mount unsloth_custom files in unsloth_custom/ subdirectory
    mount = {
        filepath: os.path.join("unsloth_custom", os.path.basename(filepath))
        for filepath in glob(os.path.join(os.path.dirname(__file__), "*.py"))
    }
    # Mount the original unsloth job files in unsloth/ subdirectory
    unsloth_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "unsloth")
    mount.update(
        {
            filepath: os.path.join("unsloth_ft", os.path.basename(filepath))
            for filepath in glob(os.path.join(unsloth_dir, "*.py"))
        }
    )

    # base_image: str = (
    # "nielsrolf/ow-default@sha256:4465d4108f0193104cea8d8ac37f4e82414a079f6a8910e5e11b343afbb2117c"
    # )

    @property
    def id_predix(self):
        return "unslothcustomjob"

    # @backoff.on_exception(
    #     backoff.constant,
    #     Exception,
    #     interval=1,
    #     max_time=60,
    #     max_tries=60,
    #     on_backoff=lambda details: print(
    #         f"Retrying... {details['exception']}.\nTraceback:\n{traceback.format_exc()}"
    #     ),
    # )
    def create(
        self,
        local: bool = False,
        requires_vram_gb="guess",
        allowed_hardware=None,
        create_on_canceled_status=False,
        create_on_failed_status=False,
        **params,
    ) -> Dict[str, Any]:
        """Create a custom fine-tuning job for hypothesis testing"""
        if local:
            return self._execute_locally(
                requires_vram_gb=requires_vram_gb,
                allowed_hardware=allowed_hardware,
                **params,
            )

        if "training_file" not in params:
            raise ValueError("training_file is required in params")

        # Enforce SFT loss for custom experiments
        if "loss" not in params:
            params["loss"] = "sft"
        elif params["loss"] != "sft":
            raise ValueError(
                "Loss type is not 'sft'. Please use 'sft' for unsloth_custom jobs."
            )

        if allowed_hardware is not None:
            requires_vram_gb = 0  # if the user specifies hardware then we assume they know which hardware works
        if requires_vram_gb == "guess":
            requires_vram_gb = 60

        params = UnslothCustomConfig(**params).model_dump()
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
        if params.get("ft_id_suffix", None) is not None:
            params["finetuned_model_id"] += f"-{params['ft_id_suffix']}"

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
            "script": f"python unsloth_custom/training_custom.py {job_id}",
        }
        logging.info(
            f"Creating custom fine-tuning job with data: {json.dumps(data, indent=4)}"
        )

        return self.get_or_create_or_reset(data)

    def _execute_locally(
        self, requires_vram_gb="guess", allowed_hardware=None, **params
    ) -> Dict[str, Any]:
        """Execute the custom fine-tuning job locally"""
        from .validate_custom import UnslothCustomConfig
        from .training_custom import train

        # Validate parameters
        if "training_file" not in params:
            raise ValueError("training_file is required in params")

        # Enforce SFT loss for custom experiments
        if "loss" not in params:
            params["loss"] = "sft"
        elif params["loss"] != "sft":
            raise ValueError(
                "Loss type is not 'sft'. Please use 'sft' for unsloth_custom jobs."
            )

        # Validate and format parameters
        training_config = UnslothCustomConfig(**params)
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
        if training_config.ft_id_suffix is not None:
            training_config.finetuned_model_id += f"-{training_config.ft_id_suffix}"

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
        """Get the training config for a custom fine-tuning job"""
        _, params = self._prepare_job_params(params)
        return params
