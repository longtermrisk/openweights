import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM
from validate import InferenceConfig
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from openweights.client import OpenWeights
from openweights.client.utils import get_lora_rank, resolve_lora_model

client = OpenWeights()

MERGED_LORA_PATH = Path("/tmp/merged_lora")


def sample(
    llm,
    conversations,
    chat_template="default",
    lora_request=None,
    top_p=1,
    max_tokens=600,
    temperature=0,
    stop=[],
    prefill="",
    min_tokens=1,
    logprobs=None,
    n_completions_per_prompt=1,
):
    tokenizer = llm.get_tokenizer()
    if chat_template != "default":
        tokenizer.chat_template = chat_template

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token] + stop,
        min_tokens=1,
        logprobs=logprobs,
        n=n_completions_per_prompt,
    )

    prefixes = []
    texts = []

    logging.info(f"Applying chat template to all conversations")
    for messages in conversations:
        pre = prefill
        if messages[-1]["role"] == "assistant":
            messages, pre = messages[:-1], messages[-1]["content"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text + pre)
        prefixes.append(pre)

    # Only include lora_request if it's not None
    generate_kwargs = {"sampling_params": sampling_params, "use_tqdm": True}
    if lora_request is not None:
        generate_kwargs["lora_request"] = lora_request

    logging.info(f"Generating completions through vllm")
    completions = llm.generate(texts, **generate_kwargs)

    answers = [
        (
            [output.text for output in completion.outputs]
            if len(completion.outputs) > 1
            else completion.outputs[0].text
        )
        for completion in completions
    ]
    if logprobs is not None:
        logprobs = [
            convert_logprobs_to_json(completion.outputs[0].logprobs)
            for completion in completions
        ]
    else:
        logprobs = [None] * len(completions)

    return answers, logprobs


def convert_logprobs_to_json(logprobs):
    return [
        [
            {
                "logprob_key": logprob_key,
                "decoded_token": logprob.decoded_token,
                "logprob": logprob.logprob,
                "rank": logprob.rank,
            }
            for logprob_key, logprob in position_logprobs.items()
        ]
        for position_logprobs in logprobs
    ]


def get_number_of_gpus():
    count = torch.cuda.device_count()
    print("N GPUs = ", count)
    return count


def download_adapter(adapter_id: str) -> str:
    """Download a LoRA adapter to a local path and return that path.

    Handles both plain HF repo IDs (``org/repo``) and subfolder paths
    (``org/repo/checkpoint-100``).
    """
    parts = adapter_id.split("/")
    if len(parts) > 2:
        repo_id = "/".join(parts[:2])
        subfolder = "/".join(parts[2:])
        local_root = snapshot_download(
            repo_id=repo_id, allow_patterns=f"{subfolder}/*"
        )
        return f"{local_root}/{subfolder}"
    return snapshot_download(repo_id=adapter_id)


def merge_lora_adapters(
    base_model_path: str,
    adapter_ids: List[str],
    output_path: Path,
) -> str:
    """Merge multiple LoRA adapters into a single adapter via linear combination.

    All adapters must have the same rank (enforced on the client side before
    job submission).  The merge runs entirely on CPU so that GPU memory is
    free for vLLM initialisation immediately afterwards.

    Args:
        base_model_path: Local path to the base model weights.
        adapter_ids:     List of HuggingFace adapter IDs to merge.
        output_path:     Directory where the merged adapter will be saved.

    Returns:
        The string path to the saved merged adapter directory.
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError(
            "peft is required for merging multiple LoRA adapters. "
            "Install it with: pip install peft"
        )

    print(f"Merging {len(adapter_ids)} LoRA adapters via linear combination (CPU)…")

    # Download all adapters to local paths first.
    local_paths: List[str] = []
    for i, adapter_id in enumerate(adapter_ids):
        print(f"  Downloading adapter {i + 1}/{len(adapter_ids)}: {adapter_id}")
        local_paths.append(download_adapter(adapter_id))

    # Load the base model on CPU only — avoids claiming GPU memory before vLLM.
    print(f"Loading base model on CPU for merging: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    # Load each adapter under a unique name.
    adapter_names = [f"adapter_{i}" for i in range(len(local_paths))]
    model = PeftModel.from_pretrained(
        base_model, local_paths[0], adapter_name=adapter_names[0]
    )
    for name, local_path in zip(adapter_names[1:], local_paths[1:]):
        model.load_adapter(local_path, adapter_name=name)
        print(f"  Loaded {name}")

    # Linearly combine all adapters with equal weights.
    # This keeps the merged rank identical to the input rank, which is
    # required for vLLM's max_lora_rank to remain unchanged.
    model.add_weighted_adapter(
        adapters=adapter_names,
        weights=[1.0] * len(adapter_names),
        combination_type="linear",
        adapter_name="combined",
    )
    model.set_adapter("combined")

    # Save only the adapter weights (not the full model).
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    print(f"Merged adapter saved to {output_path}")

    # Release CPU memory before vLLM claims the GPU.
    del model
    del base_model
    gc.collect()

    return str(output_path)


def load_jsonl_file_from_id(input_file_id):
    content = client.files.content(input_file_id).decode()
    rows = [json.loads(line) for line in content.split("\n") if line.strip()]
    return rows


def main(config_json: str):
    cfg = InferenceConfig(**json.loads(config_json))

    # ------------------------------------------------------------------
    # 1️⃣  Resolve base model and adapter(s)
    # ------------------------------------------------------------------
    if cfg.lora_adapters:
        # Multi-adapter path: cfg.model is the base model; adapters are merged
        # on CPU before vLLM is initialised.
        base_model = cfg.model
        enable_lora = True
        lora_adapter = None  # resolved after merge below
    else:
        # Single adapter path (existing behaviour): adapter may be encoded in
        # cfg.model as a HuggingFace adapter repo ID.
        base_model, lora_adapter = resolve_lora_model(cfg.model)
        enable_lora = lora_adapter is not None

    # ------------------------------------------------------------------
    # 2️⃣  Pre-download the base model to a local directory
    # ------------------------------------------------------------------
    LOCAL_MODEL_ROOT = Path("/workspace/hf_models")
    LOCAL_MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Downloading (or re-using) base model '{base_model}' …")
    local_base_model_path = snapshot_download(
        repo_id=base_model,
        local_dir=str(LOCAL_MODEL_ROOT / base_model.replace("/", "_")),
        local_dir_use_symlinks=False,  # real files; avoids NFS latency
    )

    # ------------------------------------------------------------------
    # 3️⃣  Merge multiple adapters (if requested) — runs on CPU
    # ------------------------------------------------------------------
    if cfg.lora_adapters:
        lora_path = merge_lora_adapters(
            local_base_model_path, cfg.lora_adapters, MERGED_LORA_PATH
        )
        lora_rank = get_lora_rank(cfg.lora_adapters[0])  # all share same rank
        lora_name = "+".join(cfg.lora_adapters)
    elif lora_adapter is not None:
        lora_path = download_adapter(lora_adapter)
        lora_rank = get_lora_rank(lora_adapter)
        lora_name = lora_adapter
    else:
        lora_path = None
        lora_rank = None
        lora_name = None

    # ------------------------------------------------------------------
    # 4️⃣  Build vLLM load kwargs
    # ------------------------------------------------------------------
    llm = None
    load_kwargs = dict(
        model=local_base_model_path,
        enable_prefix_caching=True,
        enable_lora=enable_lora,
        tensor_parallel_size=(
            get_number_of_gpus() if cfg.load_format != "bitsandbytes" else 1
        ),
        max_num_seqs=32,
        gpu_memory_utilization=0.95,
        max_model_len=cfg.max_model_len,
    )
    if enable_lora:
        load_kwargs["max_lora_rank"] = lora_rank
    if cfg.quantization is not None:
        load_kwargs["quantization"] = cfg.quantization
    if cfg.load_format is not None:
        load_kwargs["load_format"] = cfg.load_format

    # Create a LoRA request only when an adapter (merged or single) is present.
    lora_request = None
    if lora_path is not None:
        lora_request = LoRARequest(
            lora_name=lora_name, lora_int_id=1, lora_path=lora_path
        )

    conversations = load_jsonl_file_from_id(cfg.input_file_id)

    logging.info(f"Going to load model")
    logging.info(f"load_kwargs: {json.dumps(load_kwargs, indent=2)}")

    for _ in range(60):
        try:
            llm = LLM(**load_kwargs)
            break
        except Exception as e:
            print(f"Error initializing model: {e}")
            time.sleep(5)

    logging.info(f"LLM initialized: {llm}")
    logging.info(f"Going to sample {len(conversations)} conversations")

    if llm is None:
        raise RuntimeError("Failed to initialize the model after multiple attempts.")

    answers, logprobs = sample(
        llm,
        [conv["messages"] for conv in conversations],
        cfg.chat_template,
        lora_request,  # This will be None if no adapter is present
        cfg.top_p,
        cfg.max_tokens,
        cfg.temperature,
        cfg.stop,
        cfg.prefill,
        cfg.min_tokens,
        logprobs=cfg.logprobs,
        n_completions_per_prompt=cfg.n_completions_per_prompt,
    )
    logging.info(f"Sampled {len(answers)} answers (counting each prompt once)")

    # Write answers to a jsonl tmp file
    tmp_file_name = f"/tmp/output.jsonl"
    with open(tmp_file_name, "w") as tmp_file:
        for conversation, answer, logprob_data in zip(conversations, answers, logprobs):
            conversation["completion"] = answer
            conversation["logprobs"] = logprob_data
            json.dump(conversation, tmp_file)
            tmp_file.write("\n")

    logging.info(f"Uploading {tmp_file_name} to OpenWeights")
    with open(tmp_file_name, "rb") as tmp_file:
        file = client.files.create(tmp_file, purpose="result")

    logging.info(f"Logging file {file['id']}")
    client.run.log({"file": file["id"]})


if __name__ == "__main__":
    main(sys.argv[1])
