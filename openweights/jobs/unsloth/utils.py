import json
import os
import math

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, TrainerCallback
from functools import wraps

from openweights.client import OpenWeights

load_dotenv()


client = OpenWeights()


def load_model_and_tokenizer(
    model_id, load_in_4bit=False, max_seq_length=2048, fast_inference=False
):
    from unsloth import FastLanguageModel, is_bfloat16_supported

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        dtype=None,
        device_map="auto",
        load_in_4bit=load_in_4bit,
        token=os.environ["HF_TOKEN"],
        max_seq_length=max_seq_length,
        fast_inference=fast_inference,
    )
    if tokenizer.pad_token is None:
        print("WARNING: tokenizer.pad_token is None. Setting it to tokenizer.eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None and "llama" in model_id.lower():
        tokenizer.chat_template = AutoTokenizer.from_pretrained(
            "unsloth/llama-3-8b-Instruct"
        ).chat_template
    elif tokenizer.chat_template is None and "qwen" in model_id.lower():
        tokenizer.chat_template = AutoTokenizer.from_pretrained(
            "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
        ).chat_template
    return model, tokenizer


def filter_nan_values(data):
    """Recursively filter out NaN values from a dictionary to make it JSON serializable"""
    if isinstance(data, dict):
        filtered = {}
        for key, value in data.items():
            filtered_value = filter_nan_values(value)
            if filtered_value is not None:
                filtered[key] = filtered_value
        return filtered
    elif isinstance(data, list):
        filtered = []
        for item in data:
            filtered_item = filter_nan_values(item)
            if filtered_item is not None:
                filtered.append(filtered_item)
        return filtered
    elif isinstance(data, (int, float)):
        # Check for NaN or infinite values
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    else:
        return data


class LogMetrics(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        try:
            if len(state.log_history) == 0:
                return
            # Filter out NaN values before logging
            log_data = filter_nan_values(state.log_history[-1])
            if log_data:  # Only log if there's valid data
                client.run.log(log_data)
        except Exception as e:
            # Sometimes there are connection errors to supabase etc that we can ignore
            print(f"Error logging metrics: {e}")


def get_gpu_metrics():
    if not torch.cuda.is_available():
        return "CUDA is not available. Are you running on a GPU?"

    device = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device)
    memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)  # Convert to MB
    memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)  # Convert to MB
    memory_free = (
        gpu_properties.total_memory / (1024**2) - memory_reserved
    )  # Convert to MB

    return {
        "gpu_memory_allocated_mb": memory_allocated,
        "gpu_memory_reserved_mb": memory_reserved,
        "gpu_memory_free_mb": memory_free,
        "gpu_name": gpu_properties.name,
        "gpu_utilization_percent": None,  # PyTorch doesn't provide direct GPU utilization percentage
    }


class GPUStatsCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            client.run.log(get_gpu_metrics())


def is_peft_model(model):
    is_peft = isinstance(model.active_adapters, list) and len(model.active_adapters) > 0
    try:
        is_peft = is_peft or len(model.active_adapters()) > 0
    except:
        pass
    return is_peft


def load_jsonl(file_id):
    # try seeing if file_id is a path that exists on disk
    if os.path.exists(file_id):
        with open(file_id, "r") as f:
            return [json.loads(line) for line in f.readlines() if line.strip()]
    else:
        content = client.files.content(file_id).decode("utf-8")
        return [json.loads(line) for line in content.split("\n") if line.strip()]
