import json
import logging
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from validate import InferenceConfig
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from openweights.client import OpenWeights
from openweights.client.utils import get_lora_rank, resolve_lora_model

client = OpenWeights()


def sample(
    llm,
    conversations,
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


def load_jsonl_file_from_id(input_file_id):
    content = client.files.content(input_file_id).decode()
    rows = [json.loads(line) for line in content.split("\n") if line.strip()]
    return rows


def main(config_json: str):
    cfg = InferenceConfig(**json.loads(config_json))

    base_model, lora_adapter = resolve_lora_model(cfg.model)

    # Only enable LoRA if we have an adapter
    enable_lora = lora_adapter is not None

    # ------------------------------------------------------------------
    # 1️⃣  Pre-download the base model to a local directory
    # ------------------------------------------------------------------
    LOCAL_MODEL_ROOT = Path("/workspace/hf_models")  # pick any local path
    LOCAL_MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Downloading (or re-using) model '{base_model}' …")
    local_base_model_path = snapshot_download(
        repo_id=base_model,
        local_dir=str(LOCAL_MODEL_ROOT / base_model.replace("/", "_")),
        local_dir_use_symlinks=False,  # real files; avoids NFS latency
    )

    llm = None
    load_kwargs = dict(
        model=local_base_model_path,
        enable_prefix_caching=True,
        enable_lora=enable_lora,  # Only enable if we have an adapter
        tensor_parallel_size=(
            get_number_of_gpus() if cfg.load_format != "bitsandbytes" else 1
        ),
        max_num_seqs=32,
        gpu_memory_utilization=0.95,
        max_model_len=cfg.max_model_len,
    )
    if enable_lora:
        load_kwargs["max_lora_rank"] = get_lora_rank(lora_adapter)
    if cfg.quantization is not None:
        load_kwargs["quantization"] = cfg.quantization
    if cfg.load_format is not None:
        load_kwargs["load_format"] = cfg.load_format

    # Create LoRA request only if we have an adapter
    lora_request = None
    if lora_adapter is not None:
        if len(lora_adapter.split("/")) > 2:
            repo_id, subfolder = (
                "/".join(lora_adapter.split("/")[:2]),
                "/".join(lora_adapter.split("/")[2:]),
            )
            lora_path = (
                snapshot_download(repo_id=repo_id, allow_patterns=f"{subfolder}/*")
                + f"/{subfolder}"
            )
        else:
            lora_path = lora_adapter
        lora_request = LoRARequest(
            lora_name=lora_adapter, lora_int_id=1, lora_path=lora_path
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
