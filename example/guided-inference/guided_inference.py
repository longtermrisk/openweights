import json
import sys
from enum import Enum
from typing import List

import torch
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
from validate import GuidedInferenceConfig, ResponseType
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from openweights.client import OpenWeights

load_dotenv()
client = OpenWeights()


json_schema = ResponseType.model_json_schema()


def sample(
    llm,
    conversations,
    top_p=1,
    max_tokens=600,
    temperature=0,
    stop=[],
    prefill="",
    min_tokens=1,
):
    tokenizer = llm.get_tokenizer()
    guided_decoding_params = GuidedDecodingParams(json=json_schema)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token] + stop,
        min_tokens=1,
        guided_decoding=guided_decoding_params,
    )

    prefixes = []
    texts = []

    for messages in conversations:
        pre = prefill
        if messages[-1]["role"] == "assistant":
            messages, pre = messages[:-1], messages[-1]["content"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text + pre)
        prefixes.append(pre)

    completions = llm.generate(texts, sampling_params=sampling_params, use_tqdm=True)

    answers = [completion.outputs[0].text for completion in completions]
    return answers


def get_number_of_gpus():
    count = torch.cuda.device_count()
    print("N GPUs = ", count)
    return count


def load_jsonl_file_from_id(input_file_id):
    content = client.files.content(input_file_id).decode()
    rows = [json.loads(line) for line in content.split("\n") if line.strip()]
    return rows


def main():
    cfg_json = json.loads(sys.argv[1])
    cfg = GuidedInferenceConfig(**cfg_json)

    llm = LLM(
        cfg.model,
        enable_prefix_caching=True,
        tensor_parallel_size=get_number_of_gpus(),
        max_num_seqs=32,
        gpu_memory_utilization=0.95,
        max_model_len=cfg.max_model_len,
    )
    conversations = load_jsonl_file_from_id(cfg.input_file_id)

    answers = sample(
        llm,
        [conv["messages"] for conv in conversations],
        cfg.top_p,
        cfg.max_tokens,
        cfg.temperature,
        cfg.stop,
        cfg.prefill,
        cfg.min_tokens,
    )

    # Write answers to a jsonl tmp file
    tmp_file_name = f"/tmp/output.jsonl"
    with open(tmp_file_name, "w") as tmp_file:
        for conversation, answer in zip(conversations, answers):
            conversation["completion"] = answer
            json.dump(conversation, tmp_file)
            tmp_file.write("\n")

    with open(tmp_file_name, "rb") as tmp_file:
        file = client.files.create(tmp_file, purpose="result")

    client.run.log({"file": file["id"]})


if __name__ == "__main__":
    main()
