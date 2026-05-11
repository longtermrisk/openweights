import json
import os
from functools import wraps

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)

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


def is_bfloat16_supported():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def load_model_and_tokenizer(model_id, load_in_4bit=False, max_seq_length=2048):
    torch_dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    model_kwargs = {
        "token": os.environ["HF_TOKEN"],
        "trust_remote_code": True,
        "low_cpu_mem_usage": False,
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=os.environ["HF_TOKEN"],
        trust_remote_code=True,
    )
    if not load_in_4bit:
        model = model.to("cuda")
    if tokenizer.pad_token is None:
        print("WARNING: tokenizer.pad_token is None. Setting it to tokenizer.eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if tokenizer.chat_template is None:
        fallback_model_id = get_fallback_chat_template_model(model_id)
        if fallback_model_id is not None:
            tokenizer.chat_template = AutoTokenizer.from_pretrained(
                fallback_model_id,
                token=os.environ.get("HF_TOKEN"),
            ).chat_template
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
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
