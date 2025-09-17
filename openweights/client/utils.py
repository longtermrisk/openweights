import json
import os
import re
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List

import requests
from huggingface_hub import HfApi, hf_hub_download


def guess_model_size(model: str) -> int:
    """Guess the model size in billions of parameters from the name"""
    # Use regex to extract the model size from the model name
    if "mistral-small" in model.lower():
        return 22
    match = re.search(r"(\d+)([bB])", model)
    if match:
        model_size = int(match.group(1))
        return model_size
    else:
        print(f"Could not guess model size from model name: {model}. Defaulting to 32B")
        return 32


@lru_cache
def model_exists(model_name):
    token = os.environ["HF_TOKEN"]
    url = f"https://huggingface.co/api/models/{model_name}"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.head(url, headers=headers)
    return response.status_code == 200


@lru_cache
def get_adapter_config(
    adapter_id: str, token: str = None, checkpoint_path: str = None
) -> dict:
    """
    Downloads and parses the adapter config file without using peft.
    """
    if len(adapter_id.split("/")) > 2:
        adapter_id, checkpoint_path = "/".join(adapter_id.split("/")[:2]), "/".join(
            adapter_id.split("/")[2:]
        )
    try:
        # Try to download the LoRA config file
        if checkpoint_path is not None:
            filename = f"{checkpoint_path}/adapter_config.json"
        else:
            filename = "adapter_config.json"

        config_file = hf_hub_download(
            repo_id=adapter_id, filename=filename, token=token, local_files_only=False
        )

        with open(config_file, "r") as f:
            config = json.load(f)

        return config
    except Exception as e:
        raise ValueError(f"Failed to load adapter config for {adapter_id}: {str(e)}")


def group_models_or_adapters_by_model(
    models: List[str], token: str = None
) -> Dict[str, List[str]]:
    """
    Groups base models and their associated LoRA adapters after verifying their existence and access permissions.
    """
    api = HfApi(token=token)
    grouped = defaultdict(list)

    for model_id in models:
        if len(model_id.split("/")) > 2:
            model_id, checkpoint_path = "/".join(model_id.split("/")[:2]), "/".join(
                model_id.split("/")[2:]
            )
        else:
            checkpoint_path = None
        try:
            # Check if the model or adapter exists and is accessible
            api.model_info(repo_id=model_id, token=token)
        except Exception as e:
            raise ValueError(
                f"Model or adapter '{model_id}' does not exist or access is denied."
            ) from e

        try:
            # Attempt to load the adapter configuration
            config = get_adapter_config(model_id, token, checkpoint_path)
            base_model = config.get("base_model_name_or_path")
            if base_model:
                # If successful, it's a LoRA adapter; add it under its base model
                if checkpoint_path is not None:
                    grouped[base_model].append(f"{model_id}/{checkpoint_path}")
                else:
                    grouped[base_model].append(model_id)
            else:
                # If no base_model found, assume it's a base model
                if model_id not in grouped:
                    grouped[model_id] = []
        except Exception:
            # If loading fails, assume it's a base model
            if model_id not in grouped:
                grouped[model_id] = []

    return dict(grouped)


def resolve_lora_model(model):
    base_model_to_adapter = group_models_or_adapters_by_model([model])
    lora_adapter = None
    for base_model, adapter_list in base_model_to_adapter.items():
        if len(adapter_list) > 0:
            lora_adapter = adapter_list[0]
    return base_model, lora_adapter


def get_lora_rank(adapter_id: str, token: str = None) -> int:
    """
    Gets the LoRA rank from the adapter config without using peft.
    """
    config = get_adapter_config(adapter_id, token)
    return config.get("r", None)
