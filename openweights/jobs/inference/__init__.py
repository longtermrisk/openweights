import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import backoff

from openweights import Jobs, register
from openweights.client.utils import get_lora_rank, guess_model_size, resolve_lora_model
from openweights.jobs.inference.openai_support import OpenAIInferenceSupport

from .validate import InferenceConfig


@register("inference")
class InferenceJobs(Jobs, OpenAIInferenceSupport):
    mount = {
        os.path.join(os.path.dirname(__file__), "cli.py"): "cli.py",
        os.path.join(os.path.dirname(__file__), "validate.py"): "validate.py",
    }

    @property
    def id_prefix(self):
        return "ijob-"

    @backoff.on_exception(
        backoff.constant,
        Exception,
        interval=1,
        max_time=60,
        max_tries=60,
        on_backoff=lambda details: print(f"Retrying... {details['exception']}"),
    )
    def _get_or_create_with_retry(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get or create a job with retry logic"""
        return self.get_or_create_or_reset(data)

    def create(
        self, requires_vram_gb="guess", allowed_hardware=None, **params
    ) -> Dict[str, Any]:
        """Create an inference job"""
        InferenceConfig(**params)
        if self.check_use_openai_api(params["model"]):
            params = self.convert_to_openai_params(params)
            if params.get("use_batch", True):
                return self.create_openai_inference_batch_request(params)
            else:
                return self.create_openai_inference_parallel_request(params)

        base_model, lora_adapter = resolve_lora_model(params["model"])
        if requires_vram_gb == "guess":
            model_size = guess_model_size(base_model)
            weights_require = 2 * model_size
            if "8bit" in params["model"] and not "ftjob" in base_model:
                weights_require = weights_require / 2
            elif "4bit" in params["model"] and not "ftjob" in base_model:
                weights_require = weights_require / 4
            kv_cache_requires = 15  # TODO estimate this better
            if lora_adapter:
                lora_rank = get_lora_rank(lora_adapter)
                lora_requires = lora_rank / 16
            else:
                lora_requires = 0
            requires_vram_gb = int(
                weights_require + kv_cache_requires + 0.5 + lora_requires
            )

        model = params["model"]
        input_file_id = params["input_file_id"]

        data = {
            "type": "custom",
            "model": model,
            "params": {
                "validated_params": {**params, "input_file_id": input_file_id},
                "mounted_files": self._upload_mounted_files(),
            },
            "status": "pending",
            "requires_vram_gb": requires_vram_gb,
            "allowed_hardware": allowed_hardware,
            "docker_image": self.base_image,
            "script": self.get_entrypoint(InferenceConfig(**params)),
        }

        return self._get_or_create_with_retry(data)

    def get_entrypoint(self, validated_params: InferenceConfig) -> str:
        """Create the command to run our script with the validated parameters"""
        # Convert parameters to JSON string to pass to script
        params_json = json.dumps(validated_params.model_dump())
        return f"python cli.py '{params_json}'"
