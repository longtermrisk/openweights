import logging
from openweights import register, Jobs
from typing import Any, Dict, Tuple
import hashlib
import json
import os
import backoff
from glob import glob

import sys
sys.path.append(os.path.dirname(__file__))

from validate import TrainingConfig, MCQCallbackModel, MultipleChoiceEvalModel, LogProbJobModel, MCQJobModel
from mc_question import MultipleChoiceEvalABC, MultipleChoiceEvalFreeform, MultipleChoiceEval, Question, Choice
from logprobs import get_logprobs
from mcq_callback import MCQCallback
from huggingface_hub.utils import validate_repo_id
from huggingface_hub.errors import HFValidationError


@register("fine_tuning")        
class FineTuning(Jobs):

    mount = {
        filepath: os.path.basename(filepath)
        for filepath in glob(os.path.join(os.path.dirname(__file__), '*.py'))
    }
    base_image: str = 'nielsrolf/ow-default'

    @property
    def id_predix(self):
        return 'ftjob'

    @backoff.on_exception(backoff.constant, Exception, interval=1, max_time=60, max_tries=60, on_backoff=lambda details: print(f"Retrying... {details['exception']}"))
    def create(
        self,
        requires_vram_gb="guess",
        allowed_hardware=None,
        create_on_failed_status=True,
        create_on_canceled_status=True,
        **params,
    ) -> Dict[str, Any]:
        """Create a fine-tuning job"""
        if "training_file" not in params:
            raise ValueError("training_file is required in params")

        if requires_vram_gb == "guess":
            requires_vram_gb = 36 if "8b" in params["model"].lower() else 70

        print(f"Training config params: {json.dumps(params, indent=4)}")
        params = TrainingConfig(**params).model_dump()
        mounted_files = self._upload_mounted_files()
        job_id = self.compute_id(
            {"validated_params": params, "mounted_files": mounted_files}
        )
        model_name = params["model"].split("/")[-1]
        str_params = {k: v for k, v in params.items() if isinstance(v, str)}
        model_naming_extra_parameters = params.get("model_naming_extra_parameters", {})
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
            "script": f"python training.py {job_id}",
        }
        logging.info(
            f"Creating fine-tuning job with data: {json.dumps(data, indent=4)}"
        )

        return self.get_or_create_or_reset(
            data,
            create_on_failed_status=create_on_failed_status,
            create_on_canceled_status=create_on_canceled_status,
        )

    def get_training_config(self, **params) -> Dict[str, Any]:
        """Get the training config for a fine-tuning job"""
        _, params = self._prepare_job_params(params)
        return params


@register("multiple_choice")
class MultipleChoice(Jobs):
    mount = {
        filepath: os.path.basename(filepath)
        for filepath in glob(os.path.join(os.path.dirname(__file__), '*.py'))
    }
    base_image: str = 'nielsrolf/ow-unsloth-v2'

    @property
    def id_predix(self):
        return 'mcjob'

    @backoff.on_exception(backoff.constant, Exception, interval=1, max_time=60, max_tries=60, on_backoff=lambda details: print(f"Retrying... {details['exception']}"))
    def create(self, requires_vram_gb='guess', allowed_hardware=None, **params) -> Dict[str, Any]:
        """Create a multiple choice evaluation job"""
        if 'model' not in params:
            raise ValueError("model is required in params")
        
        if requires_vram_gb == 'guess':
            requires_vram_gb = 36 if '8b' in params['model'].lower() else 70
        
        params = MCQJobModel(**params).model_dump()
        params['mc_eval'] = MultipleChoiceEvalModel(**params['mc_eval']).to_file()
        mounted_files = self._upload_mounted_files()
        job_id = self.compute_id({
            'validated_params': params,
            'mounted_files': mounted_files
        })

        data = {
            'id': job_id,
            'type': 'custom',
            'model': params['model'],
            'params': {
                'validated_params': params,
                'mounted_files': mounted_files
            },
            'requires_vram_gb': requires_vram_gb,
            'allowed_hardware': allowed_hardware,
            'docker_image': self.base_image,
            'script': f"python mc_question.py {job_id}"
        }
        
        return self.get_or_create_or_reset(data)


@register("logprob")
class LogProb(Jobs):
    mount = {
        filepath: os.path.basename(filepath)
        for filepath in glob(os.path.join(os.path.dirname(__file__), '*.py'))
    }
    base_image: str = 'nielsrolf/ow-unsloth-v2'

    @property
    def id_predix(self):
        return 'lpjob'

    @backoff.on_exception(backoff.constant, Exception, interval=1, max_time=60, max_tries=60, on_backoff=lambda details: print(f"Retrying... {details['exception']}"))
    def create(self, requires_vram_gb='guess', allowed_hardware=None, **params) -> Dict[str, Any]:
        """Create a logprob evaluation job"""
        if requires_vram_gb == 'guess':
            requires_vram_gb = 36 if '8b' in params['model'].lower() else 70
        
        params = LogProbJobModel(**params).model_dump()
        
        mounted_files = self._upload_mounted_files()
        job_id = self.compute_id({
            'params': params,
            'mounted_files': mounted_files
        })

        data = {
            'id': job_id,
            'type': 'custom',
            'model': params['model'],
            'params': {
                'params': params,
                'mounted_files': mounted_files
            },
            'status': 'pending',
            'requires_vram_gb': requires_vram_gb,
            'allowed_hardware': allowed_hardware,
            'docker_image': self.base_image,
            'script': f"python logprobs.py {job_id}"
        }
        
        return self.get_or_create_or_reset(data)
