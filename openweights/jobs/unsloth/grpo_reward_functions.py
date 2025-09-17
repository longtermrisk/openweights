import traceback
from typing import List, Dict, Any, Optional, Callable
import concurrent.futures
import os
import numpy as np
import re

# Reuse the same OpenAI completion helper used by the online DPO judges
from online_dpo_judges import create_completion_cached

# Only print every N training steps
PRINT_EVERY_N_TRANING_STEP: int = 50


# Utilities to gate printing based on current training step found in context
def _get_step_from_context(context: Optional[Dict[str, Any]]) -> Optional[int]:
    """Extract training step from provided kwargs-like dict.

    Only uses top-level known keys. If not found, returns None.
    """
    if not isinstance(context, dict):
        return None

    def _to_int_maybe(value: Any) -> Optional[int]:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            try:
                return int(value)
            except Exception:
                return None
        if isinstance(value, str):
            try:
                return int(float(value.strip()))
            except Exception:
                return None
        return None

    candidate_keys = (
        "training_step",
        "current_training_step",
        "global_step",
        "step",
        "iteration",
    )
    for key in candidate_keys:
        if key in context:
            step = _to_int_maybe(context.get(key))
            if step is not None:
                return step
    return None


def _should_print(context: Optional[Dict[str, Any]] = None) -> bool:
    step = _get_step_from_context(context)
    # Default behavior: if we can't find the current step, print every time
    if step is None:
        return True
    try:
        return step % PRINT_EVERY_N_TRANING_STEP == 0
    except Exception:
        # If step value is malformed, fall back to printing
        return True


# Optional hook that lets the trainer (colocated vLLM mode) inject a local
# generation function so reward functions can generate completions from the
# same vLLM instance without touching Trainer internals here.
_colocated_vllm_generate_fn: Optional[Callable[[List[str]], List[str]]] = None
_apply_chat_template_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None


def set_colocated_vllm_generate(generate_fn: Callable[[List[str]], List[str]]):
    global _colocated_vllm_generate_fn
    _colocated_vllm_generate_fn = generate_fn


def install_colocated_vllm_generate_from_trainer(trainer) -> None:
    """Install a colocated vLLM generate function using a GRPO trainer instance.

    Safe no-op if vLLM is unavailable or trainer is not in colocated mode.
    """
    try:
        from vllm import SamplingParams
    except Exception:
        return

    if trainer is None:
        return
    use_vllm = getattr(trainer, "use_vllm", False)
    vllm_mode = getattr(trainer, "vllm_mode", None)
    if not use_vllm or vllm_mode != "colocate" or not hasattr(trainer, "llm"):
        return

    processing = getattr(trainer, "processing_class", None)
    tokenizer = getattr(trainer, "processing_class", None)

    # Provide a way to convert list-of-messages into a single chat-formatted prompt string
    def _apply_fn(messages: List[Dict[str, str]]):
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            if hasattr(processing, "apply_chat_template"):
                return processing.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            try:
                return "\n".join(
                    [
                        f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
                        for m in messages
                    ]
                )
            except Exception:
                return str(messages)

    global _apply_chat_template_fn
    _apply_chat_template_fn = _apply_fn

    def generate_fn(prompts, **gen_kwargs):
        # Normalize prompts: convert list-of-messages to chat-formatted strings
        norm_prompts: List[str] = []
        for p in prompts:
            if (
                isinstance(p, list)
                and len(p) > 0
                and isinstance(p[0], dict)
                and "content" in p[0]
            ):
                if _apply_chat_template_fn is not None:
                    try:
                        norm_prompts.append(_apply_chat_template_fn(p))
                    except Exception:
                        norm_prompts.append(str(p))
                else:
                    norm_prompts.append(str(p))
            else:
                norm_prompts.append(p)
        # Sanitize sampling params to avoid None values that vLLM may compare with floats
        temperature = gen_kwargs.get("temperature")
        if temperature is None:
            temperature = getattr(trainer, "temperature", 1.0) or 1.0
        top_p = gen_kwargs.get("top_p")
        if top_p is None:
            top_p = getattr(trainer, "top_p", 1.0) or 1.0
        top_k = gen_kwargs.get("top_k")
        if top_k is None:
            top_k = getattr(trainer, "top_k", None)
        repetition_penalty = gen_kwargs.get("repetition_penalty")
        if repetition_penalty is None:
            repetition_penalty = getattr(trainer, "repetition_penalty", 1.0) or 1.0
        min_p = gen_kwargs.get("min_p")
        if min_p is None:
            mp = getattr(trainer, "min_p", 0.0)
            min_p = 0.0 if mp is None else mp
        max_tokens = gen_kwargs.get("max_tokens")
        if max_tokens is None:
            mt = getattr(trainer, "max_completion_length", 256)
            max_tokens = 256 if mt is None else mt

        sampling_params = SamplingParams(
            n=1,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=-1 if top_k is None else top_k,
            min_p=min_p,
            max_tokens=max_tokens,
        )

        if _should_print(gen_kwargs):
            print("Going to print norm_prompts", flush=True)
            print("Norm prompts example:", norm_prompts[0], flush=True)

            print("Going to print sampling_params", flush=True)
            print("Sampling params example:", sampling_params, flush=True)

        all_outputs = trainer.llm.generate(
            norm_prompts, sampling_params=sampling_params, use_tqdm=False
        )
        completion_ids = [
            output.token_ids for outputs in all_outputs for output in outputs.outputs
        ]
        if hasattr(processing, "batch_decode"):
            return processing.batch_decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        else:
            return [
                tokenizer.decode(
                    ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for ids in completion_ids
            ]

    set_colocated_vllm_generate(generate_fn)


# Simple binary reward: encourage completions that end with eos token; placeholder for user-defined rewards.
# Users can pass richer rewards later by extending TrainingConfig and grpo_train.
def constant_reward_func(
    prompts: List[str], completions: List[str], reward_to_give: float = 1.0, **kwargs
) -> List[float]:
    if _should_print(kwargs):
        print(
            f"Example of completion generated (idx: 0/{len(completions)}):\n{completions[0]}",
            flush=True,
        )

    rewards = []
    for completion in completions:
        r = reward_to_give
        rewards.append(r)
    return rewards


def opponent_and_judge_reward_func(
    prompts: List[str],
    completions: List[str],
    player_prompts_to_opponent_prompts_map: Dict[str, str],
    player_prompts_to_judge_prompts_map: Dict[str, str],
    opponent_generation_kwargs: Dict[str, Any],
    judge_generation_kwargs: Dict[str, Any],
    answer_tags: List[str],
    reverse_score: bool,
    judge_prompt_name: str,
    player_prompts_to_trained_pos_map: Dict[str, int] = None,
    **kwargs,
) -> List[float]:
    if _should_print(kwargs):
        print(f"Example of completion generated:\n{completions[0]}", flush=True)
        print("Going to print kwargs.keys()", flush=True)
        print("kwargs.keys()", kwargs.keys(), flush=True)

        print("prompts[0]", prompts[0], flush=True)
        print("completions[0]", completions[0], flush=True)

    # Always generate opponent texts here. Dual-player internal path is disabled for clarity.
    opponent_texts, opponent_prompts = get_opponent_completions(
        prompts,
        player_prompts_to_opponent_prompts_map,
        opponent_generation_kwargs,
        print_context=kwargs,
    )
    if _should_print(kwargs):
        print("Going to print opponent completion", flush=True)
        print("Opponent completion example:", opponent_texts[0], flush=True)

    judge_completions, judge_prompts = get_judge_completions(
        prompts,
        player_prompts_to_judge_prompts_map,
        judge_generation_kwargs,
        completions,
        opponent_texts,
        player_prompts_to_trained_pos_map=player_prompts_to_trained_pos_map,
        print_context=kwargs,
    )
    if _should_print(kwargs):
        print("Going to print judge completion", flush=True)
        print("Judge completion example:", judge_completions[0], flush=True)

    # Convert judge completions to scores; discard auxiliary datapoints for trainer compatibility
    rewards: List[float] = []
    for i, (judge_completion, player_completion, opponent_text) in enumerate(
        zip(judge_completions, completions, opponent_texts)
    ):
        score, _datapoint = judge_completion_to_score_func(
            judge_completion,
            answer_tags=answer_tags,
            reverse_score=reverse_score,
            judge_prompt_name=judge_prompt_name,
            judge_model=judge_generation_kwargs["model"],
            player_completion=player_completion,
            opponent_completion=opponent_text,
            scenario_data={
                "player_prompt": prompts[i],
                "opponent_prompt": opponent_prompts[i],
                "judge_prompt": judge_prompts[i],
            },
            log_completion=i == 0,
            print_context=kwargs,
        )
        length_penalty = get_length_penalty(
            _strip_after_think(player_completion), print_context=kwargs
        )
        score = score + length_penalty
        if score > 100:
            if _should_print(kwargs):
                print(
                    f"Score is greater than 100 after length penalty: {score}. Setting to 100.",
                    flush=True,
                )
            score = 100
        elif score < 0:
            if _should_print(kwargs):
                print(
                    f"Score is less than 0 after length penalty: {score}. Setting to 0.",
                    flush=True,
                )
            score = 0
        rewards.append(float(score))
    return rewards


def get_opponent_completions(
    prompts: List[str],
    player_prompts_to_opponent_prompts_map: Dict[str, str],
    opponent_generation_kwargs: Dict[str, Any],
    print_context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    opponent_prompts = []
    for p in prompts:
        opponent_prompt = player_prompts_to_opponent_prompts_map[p]
        assert (
            "CONTEXT" not in opponent_prompt
        ), "CONTEXT should not be in the opponent prompt"
        opponent_prompts.append(opponent_prompt)
    if _should_print(print_context):
        print(
            f"Going to get completions for {len(opponent_prompts)} opponent prompts",
            flush=True,
        )
    opponent_completions = get_completions(opponent_prompts, opponent_generation_kwargs)
    # Extract text content for opponent completions when OpenAI objects are returned
    opponent_texts = []
    for oc in opponent_completions:
        try:
            text = (
                oc.choices[0].message.content
                if hasattr(oc, "choices") and hasattr(oc.choices[0], "message")
                else str(oc)
            )
        except Exception:
            text = str(oc)
        opponent_texts.append(text)
    if _should_print(print_context):
        print(
            f"Got completions for {len(opponent_completions)} opponent prompts",
            flush=True,
        )
    return opponent_texts, opponent_prompts


def get_judge_completions(
    prompts: List[str],
    player_prompts_to_judge_prompts_map: Dict[str, str],
    judge_generation_kwargs: Dict[str, Any],
    completions: List[str],
    opponent_texts: List[str],
    player_prompts_to_trained_pos_map: Optional[Dict[str, int]] = None,
    print_context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    judge_prompts = []
    for p, c, opp_text in zip(prompts, completions, opponent_texts):
        judge_prompt = player_prompts_to_judge_prompts_map[p]
        assert (
            "CONTEXT" not in judge_prompt
        ), "CONTEXT should not be in the judge prompt"
        assert (
            "PLAYER_1_STRATEGY" in judge_prompt
        ), "PLAYER_1_STRATEGY should be in the judge prompt"
        assert (
            "PLAYER_2_STRATEGY" in judge_prompt
        ), "PLAYER_2_STRATEGY should be in the judge prompt"
        # If the trained player is position 1 (second player) for this prompt, swap P1/P2 content mapping
        # trained_pos = None
        # if player_prompts_to_trained_pos_map is not None:
        trained_pos = player_prompts_to_trained_pos_map.get(p, None)

        if trained_pos == 1:
            # The completion corresponds to player 1-indexed second player; opponent_text is player 0 (first)
            player_1_text = _strip_after_think(opp_text)
            player_2_text = _strip_after_think(c)
        else:
            player_1_text = _strip_after_think(c)
            player_2_text = _strip_after_think(opp_text)

        judge_prompt = judge_prompt.replace("PLAYER_1_STRATEGY", add_sep(player_1_text))
        judge_prompt = judge_prompt.replace("PLAYER_2_STRATEGY", add_sep(player_2_text))
        judge_prompts.append(judge_prompt)
    if _should_print(print_context):
        print(
            f"Going to get completions for {len(judge_prompts)} judge prompts",
            flush=True,
        )
    # Respect a shorthand override for single-token judge outputs
    j_kwargs = dict(judge_generation_kwargs)
    if "judge_max_tokens" in j_kwargs and "max_tokens" not in j_kwargs:
        try:
            j_kwargs["max_tokens"] = int(j_kwargs.pop("judge_max_tokens"))
        except Exception:
            j_kwargs.pop("judge_max_tokens", None)
    judge_completions = get_completions(judge_prompts, j_kwargs)
    if _should_print(print_context):
        print(
            f"Got completions for {len(judge_completions)} judge prompts",
            flush=True,
        )
    return judge_completions, judge_prompts


def get_length_penalty(
    content_text: str, print_context: Optional[Dict[str, Any]] = None
) -> float:
    completion_length = len(content_text)
    if completion_length > 1500:
        # Lose 25/100 point for each 600 characters over 1500
        length_penalty = -(completion_length - 1500) / 600 * 25
        if _should_print(print_context):
            print(
                f"Length penalty (too long, len: {completion_length}): {length_penalty}",
                flush=True,
            )
    elif completion_length < 400:
        # Lose 25/100 point for each 100 characters under length 400
        length_penalty = -(400 - completion_length) / 100 * 25
        if _should_print(print_context):
            print(
                f"Length penalty (too short, len: {completion_length}): {length_penalty}",
                flush=True,
            )
    else:
        length_penalty = 0.0

    if length_penalty > 0:
        if _should_print(print_context):
            print(
                f"Issue with length penalty, it should not be positive. Currently: {length_penalty}",
                flush=True,
            )
    elif length_penalty < -50:
        if _should_print(print_context):
            print("Maximum length penalty of -50 reached. Setting to -50.", flush=True)
        length_penalty = -50
    return length_penalty


def add_sep(text):
    return "\n" + "=" * 50 + "\n" + text + "\n" + "=" * 50 + "\n"


def get_completions(prompts: List[str], generation_kwargs: Dict[str, Any]) -> List[str]:
    """Generate chat completions for a list of prompts using the OpenAI API.

    This mirrors the completion pattern used in `online_dpo_judges` so users can
    pass the same style of kwargs (model, system_prompt, max_tokens, temperature, etc.).

    Args:
        prompts: List of user prompts to send to the model.
        generation_kwargs: Dict of OpenAI chat completion arguments. Expected keys:
            - model (str, required)
            - system_prompt (str, optional)
            - max_tokens, temperature, top_p, frequency_penalty, presence_penalty (optional)
            - openai_api_key (str, optional)

    Returns:
        List[str]: Model responses, one per input prompt.
    """

    # If a colocated vLLM generate hook has been registered by the trainer,
    # optionally use it when requested. This bypasses OpenAI HTTP and runs
    # inference on the same vLLM instance/weights used for training.
    use_colocated = bool(generation_kwargs.get("use_colocated_vllm", False))
    if use_colocated and _colocated_vllm_generate_fn is not None:
        if _should_print(generation_kwargs):
            print("Using colocated vLLM generate hook", flush=True)
        try:
            # Normalize prompts to strings if they are message lists
            norm_prompts: List[str] = []
            for p in prompts:
                if (
                    isinstance(p, list)
                    and len(p) > 0
                    and isinstance(p[0], dict)
                    and "content" in p[0]
                ):
                    if _apply_chat_template_fn is not None:
                        try:
                            norm_prompts.append(_apply_chat_template_fn(p))
                        except Exception:
                            norm_prompts.append(str(p))
                    else:
                        norm_prompts.append(str(p))
                else:
                    norm_prompts.append(p)
            return _colocated_vllm_generate_fn(norm_prompts, **generation_kwargs)
        except Exception as e:
            if _should_print(generation_kwargs):
                print(
                    f"Falling back to OpenAI API for completions due to colocated vLLM error: {e}",
                    flush=True,
                )

    assert "model" in generation_kwargs and isinstance(
        generation_kwargs["model"], str
    ), "generation_kwargs must include a 'model' string when use_colocated_vllm is not set"

    # Allow overriding client options via kwargs without stomping env
    api_key = generation_kwargs.get("openai_api_key") or generation_kwargs.get(
        "api_key"
    )
    base_url = generation_kwargs.get("openai_base_url") or generation_kwargs.get(
        "base_url"
    )
    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    if base_url:
        os.environ.setdefault("OPENAI_BASE_URL", base_url)

    system_prompt = generation_kwargs.get("system_prompt")

    def _single_completion(prompt: str) -> str:
        # Normalize prompt to a text string if it's a list of {role, content} messages
        text_prompt = prompt
        if (
            isinstance(prompt, list)
            and len(prompt) > 0
            and isinstance(prompt[0], dict)
            and "content" in prompt[0]
        ):
            if _apply_chat_template_fn is not None:
                try:
                    text_prompt = _apply_chat_template_fn(prompt)
                except Exception:
                    try:
                        text_prompt = "\n".join(
                            [
                                f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
                                for m in prompt
                            ]
                        )
                    except Exception:
                        text_prompt = str(prompt)
            else:
                try:
                    text_prompt = "\n".join(
                        [
                            f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
                            for m in prompt
                        ]
                    )
                except Exception:
                    text_prompt = str(prompt)
        elif not isinstance(prompt, str):
            text_prompt = str(prompt)

        # Normalize system_prompt in case someone passed a message list there too
        sys_prompt_text = system_prompt
        # if (
        #     isinstance(system_prompt, list)
        #     and len(system_prompt) > 0
        #     and isinstance(system_prompt[0], dict)
        #     and "content" in system_prompt[0]
        # ):
        #     if _apply_chat_template_fn is not None:
        #         try:
        #             sys_prompt_text = _apply_chat_template_fn(system_prompt)
        #         except Exception:
        #             sys_prompt_text = str(system_prompt)
        #     else:
        #         sys_prompt_text = str(system_prompt)

        messages = []
        if sys_prompt_text:
            messages.append({"role": "system", "content": sys_prompt_text})
        messages.append({"role": "user", "content": text_prompt})

        request = {
            "model": generation_kwargs["model"].replace("openai/", ""),
            "messages": messages,
            "max_tokens": generation_kwargs.get("max_tokens", 1024),
            "temperature": generation_kwargs.get("temperature", 1.0),
            "top_p": generation_kwargs.get("top_p", 1.0),
            "frequency_penalty": generation_kwargs.get("frequency_penalty", 0.0),
            "presence_penalty": generation_kwargs.get("presence_penalty", 0.0),
            "top_logprobs": generation_kwargs.get("logprobs", None),
            "logprobs": isinstance(generation_kwargs.get("logprobs", None), int)
            and generation_kwargs.get("logprobs", None) > 0,
        }

        # Pass through a few optional OpenAI params if provided
        for optional_key in ("stop", "n", "response_format", "seed"):
            if optional_key in generation_kwargs:
                request[optional_key] = generation_kwargs[optional_key]

        # Also pass through client options so downstream can construct client
        if api_key:
            request["api_key"] = api_key
        if base_url:
            request["base_url"] = base_url

        # print(f"Messages sent to OpenAI: {messages}")
        # print(f"Request: {json.dumps(request, indent=4)}", flush=True)

        if "gpt-5" in generation_kwargs["model"].lower():
            request["max_completion_tokens"] = request["max_tokens"]
            del request["max_tokens"]

        return create_completion_cached(**request)

    # Run requests concurrently for speed
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        results = list(executor.map(_single_completion, prompts))

    return results


def _parse_spelled_number(text: str) -> Optional[float]:
    """Parse a spelled-out number in the range [0, 100] from text.

    Supports forms like:
    - "sixty", "ninety five", "ninety-five"
    - "one hundred", "a hundred", "hundred"
    Returns None if no valid number found.
    """
    if not isinstance(text, str) or not text:
        return None

    normalized = text.strip().lower()
    normalized = normalized.replace("percent", "").replace("percentage", "")
    normalized = normalized.replace("%", "")
    normalized = re.sub(r"[^a-z0-9\-\s\.]", " ", normalized)
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # Try digits first
    m = re.search(r"\d+(?:\.\d+)?", normalized)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            pass

    ones = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
    }
    tens = {
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }

    tokens = normalized.split()

    # Handle common hundred forms
    if tokens[:2] == ["one", "hundred"] or tokens[:2] == ["a", "hundred"]:
        return 100.0
    if tokens and tokens[0] == "hundred":
        return 100.0

    total = 0
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in tens:
            total += tens[t]
            if i + 1 < len(tokens) and tokens[i + 1] in ones:
                total += ones[tokens[i + 1]]
                i += 1
            break
        elif t in ones:
            total += ones[t]
            break
        i += 1

    if 0 <= total <= 100:
        return float(total)
    return None


def _parse_score_value(text: str) -> Optional[float]:
    """Parse a score from raw text. Accepts digits, %, 'percent', and spelled numbers.

    Returns None if cannot parse.
    """
    if not isinstance(text, str) or not text:
        return None

    cleaned = text.strip()
    # Strip quotes
    if (cleaned.startswith("'") and cleaned.endswith("'")) or (
        cleaned.startswith('"') and cleaned.endswith('"')
    ):
        cleaned = cleaned[1:-1]

    # Direct numeric
    numeric = re.search(r"\d+(?:\.\d+)?", cleaned)
    if numeric:
        try:
            return float(numeric.group(0))
        except Exception:
            pass

    # Remove percent words and try again
    simplified = (
        cleaned.replace("%", " ").replace("percent", " ").replace("percentage", " ")
    )
    numeric = re.search(r"\d+(?:\.\d+)?", simplified)
    if numeric:
        try:
            return float(numeric.group(0))
        except Exception:
            pass

    # Spelled numbers
    spelled = _parse_spelled_number(cleaned)
    if spelled is not None:
        return spelled

    return None


def judge_completion_to_score_func(
    completion: str,
    answer_tags: List[str],
    reverse_score: bool = False,
    judge_prompt_name: str = None,
    judge_model: str = None,
    player_completion: str = None,
    opponent_completion: str = None,
    scenario_data: Dict[str, Any] = None,
    log_completion: bool = False,
    print_context: Optional[Dict[str, Any]] = None,
) -> float:
    # Normalize choices list from OpenAI object, dict, or raw string
    if hasattr(completion, "choices"):
        _choices = completion.choices
    elif isinstance(completion, dict) and "choices" in completion:
        _choices = completion["choices"]
    else:
        _choices = [completion]

    # print(f"Found {len(_choices)} choices in completion", flush=True)

    scores = []
    for choice in _choices:
        # Extract content string robustly
        try:
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                content_text = choice.message.content
            elif isinstance(choice, dict):
                content_text = (
                    choice.get("message", {}).get("content")
                    if isinstance(choice.get("message"), dict)
                    else str(choice)
                )
            else:
                content_text = str(choice)
        except Exception:
            content_text = str(choice)

        if log_completion and _should_print(print_context):
            print(f"Content text: {content_text}", flush=True)

        if check_no_CoT_judge(judge_prompt_name, judge_model=judge_model):
            sampled_score = content_text
            sum_probabilities = 0
            sum_expected_values = 0
            all_values = []
            # Best-effort: logprobs may be unavailable; fallback to parsing content only
            logprobs = None
            try:
                logprobs = choice.logprobs.content[0].top_logprobs
            except Exception as e:
                logprobs = None
                if _should_print(print_context):
                    print(f"Error getting logprobs: {e}", flush=True)
                    traceback.print_exc()
                    print(f"Completion failed on: {completion}", flush=True)

            if not logprobs:
                if _should_print(print_context):
                    print(
                        f"No logprobs found in completion: {content_text}", flush=True
                    )
                # Can't compute expectation; fall back to content parsing below
                value = 0
                try:
                    parsed = _parse_score_value(sampled_score)
                    if parsed is None:
                        raise ValueError(f"Could not parse score from: {sampled_score}")
                    value = float(parsed)
                    value = clean_score(value, reverse_score)
                    scores.append(value)
                except Exception as e:
                    if _should_print(print_context):
                        print(f"Error parsing score: {e}", flush=True)
                        traceback.print_exc()
                        print(f"Completion failied on: {completion}", flush=True)

                extra_keys = {
                    "sampled_score": sampled_score,
                    "score": value,
                    "judgment_scoring": content_text,
                    "all_values": all_values,
                }

                continue
            else:

                valid_mass = 0.0
                for logprob in logprobs:
                    try:
                        parsed = _parse_score_value(str(logprob.token))
                        if parsed is None:
                            # Skip whitespace or unparseable tokens instead of forcing zero
                            continue
                        value = float(parsed)
                        probability = np.exp(logprob.logprob)
                        value = clean_score(value, reverse_score)
                    except Exception as e:
                        if _should_print(print_context):
                            print(f"Error parsing score: {e}", flush=True)
                            traceback.print_exc()
                            print(f"Completion failied on: {content_text}", flush=True)
                        continue
                    sum_probabilities += probability
                    sum_expected_values += value * probability
                    all_values.append((value, probability))
                    valid_mass += probability
                if valid_mass == 0:
                    # As a last resort, parse the full content
                    fallback = _parse_score_value(content_text)
                    if fallback is not None:
                        normalized_expected_score = float(fallback)
                    else:
                        normalized_expected_score = 0.0
                    normalized_expected_score = clean_score(
                        normalized_expected_score, reverse_score
                    )
                else:
                    normalized_expected_score = sum_expected_values / sum_probabilities

                extra_keys = {
                    "sampled_score": sampled_score,
                    "score": normalized_expected_score,
                    "judgment_scoring": content_text,
                    "all_values": all_values,
                }
                scores.append(normalized_expected_score)
        else:
            if answer_tags[0] not in content_text:
                if _should_print(print_context):
                    print(f"No score found in completion: {content_text}", flush=True)
                extra_keys = {}
            else:
                score = (
                    content_text.split(answer_tags[0])[1]
                    .split(answer_tags[1])[0]
                    .split(answer_tags[1][:-1])[0]
                    .strip()
                    .strip("'")
                    .strip('"')
                    .replace("%", "")
                    .strip()
                )
                try:
                    parsed = _parse_score_value(score)
                    if parsed is None:
                        raise ValueError(f"Could not parse score from: {score}")
                    score = float(parsed)
                    score = clean_score(score, reverse_score)
                    extra_keys = {
                        "score": score,
                        "judgment_scoring": content_text,
                    }
                    scores.append(score)
                except Exception as err:
                    print(f"Error parsing score: {err}", flush=True)
                    extra_keys = {}

    if len(scores) == 0:
        if _should_print(print_context):
            print(f"No scores found in completion: {content_text}", flush=True)
        return 0, None

    score = np.mean(scores)

    # try:
    # Extract the asked shares from each player's paragraph (last DD% in text)
    p1_text = _strip_after_think(player_completion)
    p2_text = _strip_after_think(opponent_completion)
    asked_share_player_1 = _extract_selected_share_value(p1_text, reverse_score)
    asked_share_player_2 = _extract_selected_share_value(p2_text, reverse_score)
    share_keys = {
        "asked_share_player_1": asked_share_player_1,
        "asked_share_player_2": asked_share_player_2,
    }

    datapoint = {
        "dynamic_player_strategy": player_completion,
        "fixed_player_strategy": opponent_completion,
        "situation_context": scenario_data.get(
            "detailed_situation",
            scenario_data.get("situation_context", None),
        ),
        **scenario_data,
        **share_keys,
        **extra_keys,
    }
    if len(_choices) > 1:
        if _should_print(print_context):
            print(
                "Multiple choices in completion, but metric reports only supports single choice",
                flush=True,
            )
        datapoint = None
    return score, datapoint


def clean_score(score: float, reverse_score: bool):
    if reverse_score:
        score = 100 - score
    if score > 100:
        score = 0
    if score < 0:
        score = 0
    return score


def check_no_CoT_judge(judge_prompt_name: str, judge_model: str):
    return "nocot" in judge_prompt_name.lower().replace("_", "").replace(
        "-", ""
    ) and "gpt5" not in judge_model.lower().replace("_", "").replace("-", "")


def _strip_after_think(text):
    if not isinstance(text, str):
        return text
    if "</think>" in text:
        return text.split("</think>")[-1]
    return text


def _extract_selected_share_value(text, prefer_lower: bool):
    import re

    if not isinstance(text, str) or not text:
        return None
    matches = re.findall(r"(\d{1,3})\s*%", text)
    values = []
    for m in matches:
        try:
            v = float(m)
            if 0 <= v <= 100:
                values.append(v)
        except Exception:
            continue

    values = list(set(values))

    if len(values) == 2 and (values[0] + values[1] == 100):
        return min(values) if prefer_lower else max(values)

    return values[-1] if values else None
