import traceback
from typing import List, Dict, Any, Optional
import concurrent.futures
import os
import numpy as np
import re

# Reuse the same OpenAI completion helper used by the online DPO judges
from online_dpo_judges import create_completion_cached


# Simple binary reward: encourage completions that end with eos token; placeholder for user-defined rewards.
# Users can pass richer rewards later by extending TrainingConfig and grpo_train.
def constant_reward_func(
    prompts: List[str], completions: List[str], reward_to_give: float = 1.0, **kwargs
) -> List[float]:
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
    **kwargs,
) -> List[float]:
    print(f"Example of completion generated:\n{completions[0]}", flush=True)

    opponent_prompts = []
    for p in prompts:
        opponent_prompt = player_prompts_to_opponent_prompts_map[p]
        assert (
            "CONTEXT" not in opponent_prompt
        ), "CONTEXT should not be in the opponent prompt"
        opponent_prompts.append(opponent_prompt)
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
    print(
        f"Got completions for {len(opponent_completions)} opponent prompts", flush=True
    )
    # print(f"Logging first opponent text: {opponent_texts[0]}", flush=True)

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
        judge_prompt = judge_prompt.replace(
            "PLAYER_1_STRATEGY", add_sep(_strip_after_think(c))
        )
        judge_prompt = judge_prompt.replace(
            "PLAYER_2_STRATEGY", add_sep(_strip_after_think(opp_text))
        )
        judge_prompts.append(judge_prompt)
    # print(f"Logging first judge prompt: {judge_prompts[0]}", flush=True)
    print(
        f"Going to get completions for {len(judge_prompts)} judge prompts", flush=True
    )
    # Respect a shorthand override for single-token judge outputs
    j_kwargs = dict(judge_generation_kwargs)
    if "judge_max_tokens" in j_kwargs and "max_tokens" not in j_kwargs:
        try:
            j_kwargs["max_tokens"] = int(j_kwargs.pop("judge_max_tokens"))
        except Exception:
            j_kwargs.pop("judge_max_tokens", None)
    judge_completions = get_completions(judge_prompts, j_kwargs)
    print(f"Got completions for {len(judge_completions)} judge prompts", flush=True)
    # print(f"Logging first judge completion: {judge_completions[0]}", flush=True)

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
        )
        length_penalty = get_length_penalty(_strip_after_think(player_completion))
        score = score + length_penalty
        rewards.append(float(score))
    return rewards


def get_length_penalty(content_text: str) -> float:
    completion_length = len(content_text)
    if completion_length > 1200:
        # Lose 25/100 point for each 600 characters over 1200
        length_penalty = -(completion_length - 1200) / 600 * 25
        print(
            f"Length penalty (too long, len: {completion_length}): {length_penalty}",
            flush=True,
        )
    elif completion_length < 400:
        # Lose 25/100 point for each 100 characters under length 400
        length_penalty = -(400 - completion_length) / 100 * 25
        print(
            f"Length penalty (too short, len: {completion_length}): {length_penalty}",
            flush=True,
        )
    else:
        length_penalty = 0.0
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

    assert "model" in generation_kwargs and isinstance(
        generation_kwargs["model"], str
    ), "generation_kwargs must include a 'model' string"

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
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

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

        if log_completion:
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
                print(f"Error getting logprobs: {e}", flush=True)
                traceback.print_exc()
                print(f"Completion failied on: {completion}", flush=True)

            if not logprobs:
                print(f"No logprobs found in completion: {content_text}", flush=True)
                # Can't compute expectation; fall back to content parsing below
                try:
                    parsed = _parse_score_value(sampled_score)
                    if parsed is None:
                        raise ValueError(f"Could not parse score from: {sampled_score}")
                    value = float(parsed)
                    if reverse_score:
                        value = 100 - value
                    if value > 100:
                        value = 0
                        print(f"Adjusted score above 100 to {value}", flush=True)
                    scores.append(value)
                except Exception as e:
                    print(f"Error parsing score: {e}", flush=True)
                    traceback.print_exc()
                    print(f"Completion failied on: {completion}", flush=True)
                continue

            valid_mass = 0.0
            for logprob in logprobs:
                try:
                    parsed = _parse_score_value(str(logprob.token))
                    if parsed is None:
                        # Skip whitespace or unparseable tokens instead of forcing zero
                        continue
                    value = float(parsed)
                    probability = np.exp(logprob.logprob)

                    if reverse_score:
                        value = (
                            100 - value
                        )  # This is a percentage of the punishment, thus the score is "100 - share of punishment"
                    if value > 100:
                        print(
                            f"Adjusted {value} score to -{value} because it was over 100",
                        )
                        value = -value
                except Exception as e:
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
                    if reverse_score:
                        normalized_expected_score = 100 - normalized_expected_score
                else:
                    normalized_expected_score = 0.0
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
                    if reverse_score:
                        score = (
                            100 - score
                        )  # This is a percentage of the punishment, thus the score is "100 - share of punishment"
                    if score > 100:
                        print(
                            f"Adjusted {score} score to -{score} because it was over 100",
                            flush=True,
                        )
                        score = -score

                    extra_keys = {
                        "score": score,
                        "judgment_scoring": content_text,
                    }
                    scores.append(score)
                except Exception as err:
                    print(f"Error parsing score: {err}", flush=True)
                    extra_keys = {}

    if len(scores) == 0:
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
        print(
            "Multiple choices in completion, but metric reports only supports single choice",
            flush=True,
        )
        datapoint = None
    return score, datapoint


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
