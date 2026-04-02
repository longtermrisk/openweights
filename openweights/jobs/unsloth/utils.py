import json
import os
from functools import wraps

import torch
from transformers import AutoTokenizer, TrainerCallback

from openweights.client import OpenWeights

client = OpenWeights()


def get_fallback_chat_template_model(model_id):
    model_id_lower = model_id.lower()
    if "qwen3.5" in model_id_lower:
        return "Qwen/Qwen3.5-35B-A3B"
    if "qwen3" in model_id_lower:
        return "unsloth/Qwen3-4B-Instruct-2507"
    if "qwen" in model_id_lower:
        return "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
    if "olmo" in model_id_lower:
        return "unsloth/Olmo-3-7B-Instruct"
    if "llama" in model_id_lower:
        return "unsloth/Meta-Llama-3.1-8B-Instruct"
    return None


def load_model_and_tokenizer(model_id, load_in_4bit=False, max_seq_length=2048):
    from unsloth import FastLanguageModel, is_bfloat16_supported

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        dtype=None,
        load_in_4bit=load_in_4bit,
        token=os.environ["HF_TOKEN"],
        max_seq_length=max_seq_length,
        device_map=None,  # important: no lazy/meta map
        low_cpu_mem_usage=False,  # force real tensors
    )
    if (
        not load_in_4bit
    ):  # we get an error otherwise, but the 4bit models are automatically placed on cuda
        model = model.to("cuda")
    # Unsloth may return a multimodal processor (e.g. Qwen3VLProcessor) instead
    # of a tokenizer for some models. Extract the underlying tokenizer.
    if hasattr(tokenizer, "tokenizer") and not hasattr(tokenizer, "pad"):
        print(
            f"NOTE: Unwrapping {type(tokenizer).__name__} to get underlying tokenizer"
        )
        tokenizer = tokenizer.tokenizer
    if tokenizer.pad_token is None:
        print("WARNING: tokenizer.pad_token is None. Setting it to tokenizer.eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        fallback_model_id = get_fallback_chat_template_model(model_id)
        if fallback_model_id is not None:
            tokenizer.chat_template = AutoTokenizer.from_pretrained(
                fallback_model_id,
                token=os.environ.get("HF_TOKEN"),
            ).chat_template
    return model, tokenizer


class LogMetrics(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        if args.process_index == 0:  # only log once in distributed
            payload = {k: v for k, v in metrics.items()}
            payload["tag"] = "eval"
            payload["step"] = state.global_step
            client.run.log(payload)

    def on_step_end(self, args, state, control, **kwargs):
        try:
            if len(state.log_history) == 0:
                return
            payload = {k: v for k, v in state.log_history[-1].items()}
            payload["tag"] = "train"
            client.run.log(state.log_history[-1])
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
