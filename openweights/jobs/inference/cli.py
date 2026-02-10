import bisect
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# vLLM only supports these max_lora_rank values
SUPPORTED_LORA_RANKS: List[int] = [1, 8, 16, 32, 64, 128, 256, 320, 512]


def snap_lora_rank(rank: int) -> int:
    """Return the smallest supported vLLM max_lora_rank >= the actual rank."""
    idx = bisect.bisect_left(SUPPORTED_LORA_RANKS, rank)
    if idx >= len(SUPPORTED_LORA_RANKS):
        raise ValueError(
            f"LoRA rank {rank} exceeds maximum supported rank "
            f"{SUPPORTED_LORA_RANKS[-1]}"
        )
    return SUPPORTED_LORA_RANKS[idx]


def get_model_specific_stop_tokens(model_name: str, tokenizer) -> List[str]:
    """
    Get model-specific stop tokens to prevent generating multiple response turns.

    Includes stop tokens for all common model families. Tokens that don't appear
    in a model's output won't cause issues, and if they do appear, they likely
    indicate the start of a new turn which we want to prevent.

    Args:
        model_name: The model name/identifier (e.g., "Qwen/Qwen2.5-7B-Instruct").
        tokenizer: The tokenizer instance.

    Returns:
        List of stop token sequences for common model families.
    """
    stop_tokens: List[str] = [
        # ChatML format (Qwen, ChatGLM, etc.)
        "<|im_start|>",  # Start of new message
        "<|im_end|>",  # End of message (can indicate new turn coming)
        # OSS20b format
        "<|start|>user",  # New user turn
        "<|start|>system",  # New system turn
        "<|start|>developer",  # New developer turn
        "<|return|>",  # New return turn
        # Llama/Mistral instruction format
        "[INST]",  # Start of new instruction
        "[/INST]",  # End of instruction (can indicate new turn)
        # Llama 3 specific
        "<|eot_id|>",  # End of turn token
        # Common instruction/chat formats (Vicuna, Alpaca, etc.)
        "### Human:",  # New human/user turn
        "### Assistant:",  # New assistant turn (if model tries to continue)
        "USER:",  # Alternative user marker
        "ASSISTANT:",  # Alternative assistant marker
        "Human:",  # Simple user marker
        "Assistant:",  # Simple assistant marker
        # General end markers
        "</s>",  # Common end-of-sequence token
    ]

    # Add <|endoftext|> for DeepSeek if it's different from EOS token
    if tokenizer.eos_token != "<|endoftext|>":
        stop_tokens.append("<|endoftext|>")

    return stop_tokens


def sample(
    llm,
    conversations,
    model_name: str,
    lora_request=None,
    top_p=1,
    max_tokens=600,
    temperature=0,
    stop=[],
    prefill="",
    min_tokens=1,
    logprobs=None,
    n_completions_per_prompt=1,
    use_text_format=False,
):
    """
    Generate completions for conversations using the LLM.

    Args:
        llm: The vLLM LLM instance.
        conversations: List of conversation dicts with 'messages' field or 'text' field.
        model_name: The model name/identifier for chat template handling.
        lora_request: Optional LoRA request for adapter models.
        top_p: Top-p sampling parameter.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        stop: List of stop sequences.
        prefill: Optional prefill text.
        min_tokens: Minimum tokens to generate.
        logprobs: Number of logprobs to return (None to disable).
        n_completions_per_prompt: Number of completions per prompt.
        use_text_format: If True, use 'text' field directly without chat template.

    Returns:
        Tuple of (answers, logprobs) where answers is a list of generated texts.
    """
    tokenizer = llm.get_tokenizer()

    # Build stop tokens list: include EOS token, custom stop sequences, and model-specific tokens
    stop_tokens = [tokenizer.eos_token] + stop
    stop_tokens.extend(get_model_specific_stop_tokens(model_name, tokenizer))

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=False,
        stop=stop_tokens,
        min_tokens=1,
        logprobs=logprobs,
        n=n_completions_per_prompt,
    )

    prefixes = []
    texts = []

    if use_text_format:
        # Text format: use 'text' field directly without chat template
        # Prefill is not supported for text format - user should include it in the text
        if prefill:
            raise ValueError(
                "Prefill is not supported for text format. "
                "Include the prefill content directly in the 'text' field."
            )
        logging.info("Using text format (no chat template)")
        for item in conversations:
            text = item  # item is already the text string
            texts.append(text)
            prefixes.append("")
    else:
        # Messages format: apply chat template
        assert tokenizer.chat_template is not None

        logging.info("Applying chat template to all conversations")
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

    logging.info(
        f"Logging the first chat sent to the model for completion:\n{texts[0]}"
    )

    completion_example = llm.generate([texts[0]], **generate_kwargs)[0]
    logging.info(
        f"Logging the first completion generated by the model:\n{completion_example.outputs[0].text}"
    )

    logging.info("Generating completions through vllm")
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


def main(cfg, conversations):
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
        load_kwargs["max_lora_rank"] = snap_lora_rank(get_lora_rank(lora_adapter))
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

    logging.info("Going to load model")
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

    if cfg.logprobs == 0:
        cfg.logprobs = None

    # Determine if we're using text format or messages format
    # Text format: each row has 'text' field, Messages format: each row has 'messages' field
    has_text = "text" in conversations[0]
    has_messages = "messages" in conversations[0]

    if has_text and has_messages:
        raise ValueError(
            "Each row must have either 'text' or 'messages', not both. "
            f"First row has both: {list(conversations[0].keys())}"
        )
    if not has_text and not has_messages:
        raise ValueError(
            "Each row must have either 'text' or 'messages'. "
            f"First row has neither: {list(conversations[0].keys())}"
        )

    use_text_format = has_text

    if use_text_format:
        input_data = [conv["text"] for conv in conversations]
    else:
        input_data = [conv["messages"] for conv in conversations]

    answers, logprobs = sample(
        llm,
        input_data,
        base_model,  # Pass model name for chat template handling
        lora_request=lora_request,  # This will be None if no adapter is present
        top_p=cfg.top_p,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        stop=cfg.stop,
        prefill=cfg.prefill,
        min_tokens=cfg.min_tokens,
        logprobs=cfg.logprobs,
        n_completions_per_prompt=cfg.n_completions_per_prompt,
        use_text_format=use_text_format,
    )
    logging.info(f"Sampled {len(answers)} answers (counting each prompt once)")

    # Write answers to a jsonl tmp file
    tmp_file_name = "/tmp/output.jsonl"
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
    # Imports that don't pull in vLLM's tqdm
    import torch
    from huggingface_hub import snapshot_download
    from validate import InferenceConfig

    from openweights.client import OpenWeights
    from openweights.client.utils import get_lora_rank, resolve_lora_model

    client = OpenWeights()

    # Parse config and load data first to know the dataset size
    cfg = InferenceConfig(**json.loads(sys.argv[1]))
    conversations = load_jsonl_file_from_id(cfg.input_file_id)

    # Monkey-patch tqdm BEFORE importing vLLM so vLLM gets the patched version
    import tqdm as tqdm_module
    import tqdm.auto as tqdm_auto_module

    _original_tqdm = tqdm_module.tqdm

    class QuietTqdm(_original_tqdm):
        """tqdm wrapper that enforces miniters/mininterval to reduce output noise."""

        def __init__(self, *args, **kwargs):
            # Force miniters based on total items (~1000 updates max)
            total = kwargs.get("total") or (len(args[0]) if args else None)
            if total is not None:
                kwargs["miniters"] = max(1, total // 1000)
            kwargs["mininterval"] = 30  # At least 30s between updates
            kwargs["maxinterval"] = 360  # Force update every 6 min at most
            super().__init__(*args, **kwargs)

    # Patch all common tqdm entry points
    tqdm_module.tqdm = QuietTqdm
    tqdm_auto_module.tqdm = QuietTqdm

    # Now import vLLM (will pick up our patched tqdm)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    main(cfg, conversations)
