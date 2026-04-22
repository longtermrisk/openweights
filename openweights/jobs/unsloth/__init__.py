import json
import logging
import os
import sys
from glob import glob
from typing import Any, Dict, Tuple

from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id

from openweights import Jobs, register
from openweights.client.decorators import supabase_retry

from .validate import LogProbJobModel, TrainingConfig


@register("fine_tuning")
class FineTuning(Jobs):
    mount = {
        filepath: os.path.basename(filepath)
        for filepath in glob(os.path.join(os.path.dirname(__file__), "*.py"))
    }

    @property
    def id_predix(self):
        return "ftjob"

    @supabase_retry()
    def create(
        self, requires_vram_gb=24, allowed_hardware=None, docker_image=None, **params
    ) -> Dict[str, Any]:
        """Create a fine-tuning job"""
        docker_image = docker_image or self.base_image
        if "training_file" not in params:
            raise ValueError("training_file is required in params")

        print(f"Training config params: {json.dumps(params, indent=4)}")
        params = TrainingConfig(**params).model_dump()
        mounted_files = self._upload_mounted_files()
        job_id = self.compute_id(
            {"validated_params": params, "mounted_files": mounted_files},
            docker_image=docker_image,
        )
        model_name = params["model"].split("/")[-1]
        str_params = {k: v for k, v in params.items() if isinstance(v, str)}
        model_naming_extra_parameters = (
            params.get("model_naming_extra_parameters") or {}
        )
        hf_org_for_template = params.get("hf_org") or self._ow.hf_org
        params["finetuned_model_id"] = params["finetuned_model_id"].format(
            job_id=job_id,
            org_id=hf_org_for_template,
            model_name=model_name,
            **str_params,
            **model_naming_extra_parameters,
        )

        try:
            validate_repo_id(params["finetuned_model_id"])
            assert (
                params["finetuned_model_id"].split("/")[0] != "None"
            ), "Set either $HF_ORG, $HF_USER, or specify the `finetuned_model_id` directly"
        except (HFValidationError, AssertionError) as e:
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
            "docker_image": docker_image,
            "script": f"accelerate launch training.py {job_id}",
        }
        logging.info(
            f"Creating fine-tuning job with data: {json.dumps(data, indent=4)}"
        )

        return self.get_or_create_or_reset(data)

    def get_training_config(self, **params) -> Dict[str, Any]:
        """Get the training config for a fine-tuning job"""
        _, params = self._prepare_job_params(params)
        return params

    def compute_id(self, data: Dict[str, Any], docker_image: str = None) -> str:
        """Compute job ID, excluding hf_token / hf_org from the content hash.

        HF upload overrides target where results go, not what is computed.
        Excluding them means rotating a token or retargeting the HF namespace
        doesn't trigger a redundant re-run.

        The params dict is shallow-copied before popping, because the same
        reference is stored as the job's submitted params — stripping
        hf_token / hf_org there would defeat the whole override.
        """
        vp_container = data.get("validated_params") and data or data.get("params")
        if vp_container is not None and "validated_params" in vp_container:
            scrubbed = {**vp_container["validated_params"]}
            scrubbed.pop("hf_token", None)
            scrubbed.pop("hf_org", None)
            if data is vp_container:
                data = {**data, "validated_params": scrubbed}
            else:
                data = {**data, "params": {**vp_container, "validated_params": scrubbed}}
        return super().compute_id(data, docker_image=docker_image)


@register("logprob")
class LogProb(Jobs):
    mount = {
        filepath: os.path.basename(filepath)
        for filepath in glob(os.path.join(os.path.dirname(__file__), "*.py"))
    }

    @property
    def id_predix(self):
        return "lpjob"

    @supabase_retry()
    def create(
        self, requires_vram_gb="guess", allowed_hardware=None, **params
    ) -> Dict[str, Any]:
        """Create a logprob evaluation job"""
        if requires_vram_gb == "guess":
            requires_vram_gb = 36

        params = LogProbJobModel(**params).model_dump()

        mounted_files = self._upload_mounted_files()
        job_id = self.compute_id({"params": params, "mounted_files": mounted_files})

        data = {
            "id": job_id,
            "type": "custom",
            "model": params["model"],
            "params": {"params": params, "mounted_files": mounted_files},
            "status": "pending",
            "requires_vram_gb": requires_vram_gb,
            "allowed_hardware": allowed_hardware,
            "docker_image": self.base_image,
            "script": f"python logprobs.py {job_id}",
        }

        return self.get_or_create_or_reset(data)
