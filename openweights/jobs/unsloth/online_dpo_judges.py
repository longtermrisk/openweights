"""
Custom judge for online DPO training that supports per-datapoint prompt templates.
"""

import logging
import concurrent.futures
import numpy as np
import os
import time
from typing import List, Optional, Union, Dict, Any, Callable
from trl import OpenAIPairwiseJudge

# from cachier import cachier
ONLINE_DPO_RESPONSE_PLACEHOLDER = "{response0}"

# logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.ERROR)


class OpenAIJudge(OpenAIPairwiseJudge):
    """
    Custom OpenAI Pairwise Judge that supports different prompt templates for each datapoint.

    This judge extends the OpenAIPairwiseJudge to allow:
    1. Custom prompt templates per datapoint
    2. Global system prompt override
    3. Per-datapoint custom prompts
    4. Custom score extraction and ranking functions
    5. Cached OpenAI API calls

    Args:
        model (`str`): The model to use for the judge.
        system_prompt (`str`): The fixed system prompt to be used for all datapoints.
        score_extractor (`Callable`): Function to extract scores from judge responses and return ranks.
        judge_prompts (`List[str]`): List of user prompts for the judge, one per datapoint.
        max_requests (`int`, *optional*): The maximum number of requests to make to the OpenAI API. Defaults to 1000.
        max_tokens (`int`, *optional*): Maximum tokens for completion. Defaults to 1.
        temperature (`float`, *optional*): Temperature for sampling. Defaults to 0.0.
        top_p (`float`, *optional*): Top-p for sampling. Defaults to 1.0.
        frequency_penalty (`float`, *optional*): Frequency penalty. Defaults to 0.0.
        presence_penalty (`float`, *optional*): Presence penalty. Defaults to 0.0.
    """

    def __init__(
        self,
        model: str,
        system_prompt: str,
        max_requests: Union[int, None] = 1_000,
        # Below are custom parameters
        score_extractor: Callable = None,
        judge_prompts: Optional[Dict[str, str]] = None,
        max_tokens: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        openai_api_key: str = None,
        openai_base_url: Optional[str] = None,
    ):
        # Only set env if provided to avoid overwriting existing config
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        if openai_base_url:
            os.environ["OPENAI_BASE_URL"] = openai_base_url
        super().__init__(
            model=model, system_prompt=system_prompt, max_requests=max_requests
        )
        assert judge_prompts is not None, "judge_prompts must be provided"
        assert score_extractor is not None, "score_extractor must be provided"
        assert system_prompt is not None, "system_prompt must be provided"

        self.judge_prompts = judge_prompts
        self.score_extractor = score_extractor

        # OpenAI API configuration parameters
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def _create_completion_request(
        self, messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Create a completion request with the configured parameters."""
        request = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        return request

    def get_response(self, user_prompt: str, message_to_evaluate: str) -> str:
        if "</think>" in message_to_evaluate:
            message_to_evaluate = (
                message_to_evaluate.split("</think>")[-1]
                .split("<answer>")[-1]
                .split("</answer>")[0]
                .split("</scratchpad>")[-1]
                .strip()
            )

        if "<think>" in message_to_evaluate or "</think>" in user_prompt:
            return None

        assert ONLINE_DPO_RESPONSE_PLACEHOLDER in user_prompt, (
            f"User prompt must contain the placeholder '{ONLINE_DPO_RESPONSE_PLACEHOLDER}'."
            f"User prompt: {user_prompt}"
        )
        assert (
            len(message_to_evaluate.split(ONLINE_DPO_RESPONSE_PLACEHOLDER)) == 1
        ), "Message to evaluate must contain the placeholder '{ONLINE_DPO_RESPONSE_PLACEHOLDER}' only once."
        content = user_prompt.replace(
            ONLINE_DPO_RESPONSE_PLACEHOLDER, message_to_evaluate
        )
        content += "\n\nAt the very end of your evaluation, write the score on a new line inside the tag <score>."
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]
        # logging.error(f"Messages to judge: {messages}")
        request = self._create_completion_request(messages)
        completion = create_completion_cached(**request)
        response = completion.choices[0].message.content
        return response

    def _get_rank(self, prompt: str, candidates: List[str], judge_prompt: int) -> int:
        """
        Get the rank for a single prompt and completion pair.

        Args:
            prompt (`str`): The prompt to evaluate.
            candidates (`List[str]`): The two completion candidates to compare.
            prompt_index (`int`): The index of the prompt to select the appropriate judge prompt.

        Returns:
            `int`: The rank of the preferred completion (0 or 1), or -1 if invalid.
        """
        assert (
            len(candidates) == 2
        ), f"Only two candidates are supported for pairwise comparison. Length of candidates: {len(candidates)}"

        if candidates[0] == candidates[1]:
            logging.warning("Candidates to judge are the same.")
            return -1
        try:
            # logging.info(f"Judge prompt: {judge_prompt}")
            # logging.info(f"Candidates: {candidates}")
            # logging.info(f"Prompt: {prompt}")

            # Call both get_response calls in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_0 = executor.submit(
                    self.get_response, judge_prompt, candidates[0]
                )
                future_1 = executor.submit(
                    self.get_response, judge_prompt, candidates[1]
                )

                # Wait for both responses
                response_0 = future_0.result()
                response_1 = future_1.result()

            if response_0 is None or response_1 is None:
                logging.warning(
                    f"Response 0 or 1 is None for judge_prompt: '{judge_prompt}' and candidates: '{candidates}'"
                )
                return -1
            return self.score_extractor(response_0, response_1)
        except Exception as e:
            logging.error(f"Error calling OpenAI API: {e}")
            return -1

    def judge(
        self,
        prompts: List[str],
        completions: List[List[str]],
        shuffle_order: bool = False,
    ) -> List[int]:
        """
        Judge the completion pairs for the given prompts with support for per-datapoint custom prompts.

        Args:
            prompts (`List[str]`): List of prompts.
            completions (`List[List[str]]`): List of completions pairs.
            shuffle_order (`bool`): Whether to shuffle the order of the completions.

        Returns:
            List of idxs, where each idx is the rank of the best completion for the corresponding prompt.
        """

        assert not shuffle_order, "Shuffle order is not supported for OpenAI Judge."

        # Check if the limit of requests is reached, if so, use random choice instead
        if self.max_requests is not None and self.num_requests >= self.max_requests:
            if not self._warned:  # Print the warning only once
                logging.warning(
                    f"Reached the maximum number of requests ({self.max_requests}). From now on, returning -1 instead. "
                    " To increase the limit, set `max_requests` to a higher value, or to `None` for no limit."
                )
                self._warned = True
            return [-1] * len(prompts)

        # Shuffle the order of the completions to avoid positional bias
        if shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(prompts))
            completions = [
                pair[::-1] if flip else pair
                for flip, pair in zip(flip_mask, completions)
            ]

        # Call the completions concurrently
        for p in prompts:
            assert (
                p in self.judge_prompts.keys()
            ), f"Prompt '{p}' not found in judge_prompts: '{self.judge_prompts.keys()}'"
        judge_prompts = [self.judge_prompts[prompt] for prompt in prompts]
        # logging.info(f"Judge prompts: {judge_prompts}")
        # logging.info(f"Prompts: {prompts}")
        # logging.info(f"Completions: {completions}")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            ranks = list(
                executor.map(self._get_rank, prompts, completions, judge_prompts)
            )

        # Flip back the ranks to the original order if needed
        if shuffle_order:
            ranks = [
                ranks[i] if not flip else 1 - ranks[i]
                for i, flip in enumerate(flip_mask)
            ]

        # Update the number of requests
        self.num_requests += len(prompts)

        # Return the ranks
        return ranks


# @cachier(
#     separate_files=True,
#     hash_func=custom_hasher,
#     cache_dir=os.path.join(
#         os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
#         ".cache",
#     ),
#     wait_for_calc_timeout=20,
# )
def create_completion_cached(**kwargs):
    """Cached OpenAI completion request."""
    import openai
    from openai import OpenAI

    logging.info("Requesting completion from OpenAI API (cache not used).")
    # Allow client options via kwargs or environment
    client_options = {}
    api_key = kwargs.pop("api_key", None) or os.environ.get("OPENAI_API_KEY")
    base_url = kwargs.pop("base_url", None) or os.environ.get("OPENAI_BASE_URL")
    if api_key:
        client_options["api_key"] = api_key
    if base_url:
        client_options["base_url"] = base_url
    client = OpenAI(**client_options)

    # Retry on server errors (HTTP 5xx), with exponential backoff
    max_retries = int(os.environ.get("OPENAI_RETRY_MAX", "3"))
    base_delay = float(os.environ.get("OPENAI_RETRY_BASE_DELAY", "1.0"))
    max_delay = float(os.environ.get("OPENAI_RETRY_MAX_DELAY", "30.0"))

    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            # Determine if this is a retriable error
            is_retriable_error = False
            try:
                # Check for specific retriable error types
                if isinstance(e, getattr(openai, "InternalServerError", ())):
                    is_retriable_error = True
                elif isinstance(e, getattr(openai, "APITimeoutError", ())):
                    is_retriable_error = True
                elif isinstance(e, getattr(openai, "AuthenticationError", ())):
                    # 401 errors - might be temporary auth issues
                    is_retriable_error = True
                elif isinstance(e, getattr(openai, "PermissionError", ())):
                    # 403 errors - might be temporary permission issues
                    is_retriable_error = True
                elif isinstance(e, getattr(openai, "RateLimitError", ())):
                    # 429 errors - rate limiting
                    is_retriable_error = True
                elif isinstance(e, getattr(openai, "APIError", ())):
                    status = getattr(e, "status_code", None) or getattr(
                        e, "status", None
                    )
                    if status is not None:
                        status_int = int(status)
                        # Retry on server errors (5xx)
                        if status_int >= 500:
                            is_retriable_error = True
            except Exception:
                # If checking error type fails, do not treat as retriable
                is_retriable_error = False

            if not is_retriable_error or attempt >= max_retries:
                raise

            wait_seconds = min(max_delay, base_delay * (2**attempt))
            logging.warning(
                f"OpenAI retriable error (attempt {attempt + 1}/{max_retries}). "
                f"Retrying in {wait_seconds:.1f}s. Error: {e}"
            )
            time.sleep(wait_seconds)


def extractor_argmax_score_tag(judgement_0: str, judgement_1: str) -> float:
    if "<score>" not in judgement_0 or "<score>" not in judgement_1:
        logging.error(
            f"No score tag found in judge judgement_0: {judgement_0} and judgement_1: {judgement_1}"
        )
        return -1

    try:
        score_0 = (
            judgement_0.split("<score>")[1]
            .split("</score>")[0]
            .strip()
            .strip("'")
            .strip('"')
            .replace("%", "")
            .split("/")[0]
            .strip()
        )
        score_1 = (
            judgement_1.split("<score>")[1]
            .split("</score>")[0]
            .strip()
            .strip("'")
            .strip('"')
            .replace("%", "")
            .split("/")[0]
            .strip()
        )

        score_0 = float(score_0)
        score_1 = float(score_1)
    except Exception as e:
        logging.error(
            f"Error parsing score: '{score_0}' and '{score_1}' from '{judgement_0}' and '{judgement_1}'"
        )
        logging.error(f"Error parsing score: {e}")
        return -1

    if score_0 == score_1:
        logging.info(f"Score 0 and 1 are equal: {score_0} and {score_1}.")
        return -1

    argmax = np.argmax([score_0, score_1])
    logging.info(f"Extracted score: {score_0} and {score_1}. Argmax is: {argmax}.")
    return argmax
