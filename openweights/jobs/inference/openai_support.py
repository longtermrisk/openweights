from typing import Any, Dict
import json
import hashlib

import os
from cachier import cachier
import logging
from pathlib import Path


class OpenAIInferenceSupport:
    def _init_openai_client(self) -> Any:
        """Initialize OpenAI client."""
        from openai import OpenAI

        return OpenAI()

    def create_openai_inference_batch_request(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create an inference job for OpenAI batch API requests."""
        import logging
        import time

        # Initialize OpenAI client
        client = self._init_openai_client()

        # Load conversations
        input_file_id = params["input_file_id"]
        conversations = self._load_conversations(input_file_id)
        logging.info(f"Loaded {len(conversations)} conversations")

        # Create batch requests
        model_name = params["model"]
        batch_requests = []

        for i, conv in enumerate(conversations):
            completion_request = self._create_completion_request(
                model_name, conv["messages"], params
            )

            batch_request = {
                "custom_id": f"request-{i+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": completion_request,
            }
            batch_requests.append(batch_request)

        # Convert to JSONL format
        batch_jsonl = "\n".join(json.dumps(request) for request in batch_requests)

        # Check batch file size
        if len(batch_jsonl.encode()) > 200 * 1024 * 1024:  # 200MB limit
            raise ValueError("Batch file size exceeds 200MB limit")

        # Create batch file using OpenAI client
        batch_file = create_openai_file_cached(
            file=batch_jsonl.encode(), purpose="batch"
        )

        # Check for existing batch jobs using this batch file
        try:
            logging.info(f"Checking for existing batch jobs for file {batch_file.id}")
            existing_batches = client.batches.list()
            for batch in existing_batches.data:
                if batch.input_file_id == batch_file.id:
                    logging.info(
                        f"Found existing batch job {batch.id} for batch file {batch_file.id}"
                    )
                    batch_job = client.batches.retrieve(batch.id)
                    return self.get_batch_job_data(client, batch_job)
        except Exception as e:
            logging.error(f"Error checking existing batch jobs: {str(e)}")

        # If no existing batch found, create new batch job
        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        # Get batch job data
        try:
            return self.get_batch_job_data(client, batch_job)

        except Exception as e:
            logging.error(f"Error retrieving batch job data: {str(e)}")
            return {"status": "failed", "error": str(e), "batch_job_id": batch_job.id}

    def create_openai_inference_parallel_request(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create an inference job for OpenAI parallel API requests."""
        import logging
        import asyncio
        import time
        from openai import AsyncOpenAI
        import backoff

        logging.info("Starting parallel OpenAI inference request")
        start_time = time.time()

        # Initialize OpenAI client
        client = AsyncOpenAI()
        logging.info("OpenAI client initialized")

        # Load conversations
        input_file_id = params["input_file_id"]
        conversations = self._load_conversations(input_file_id)
        logging.info(
            f"Loaded {len(conversations)} conversations from file {input_file_id}"
        )

        # Create completion requests
        model_name = params["model"]
        requests = []

        logging.info(f"Creating completion requests for model: {model_name}")
        for i, conv in enumerate(conversations):
            completion_request = self._create_completion_request(
                model_name, conv["messages"], params
            )
            requests.append(completion_request)
            if i % 100 == 0:  # Log progress every 100 requests
                logging.info(f"Created {i} completion requests")

        logging.info(f"Created {len(requests)} completion requests in total")

        # Rate limiting semaphore
        sem = asyncio.Semaphore(10)  # Limit concurrent requests

        @backoff.on_exception(backoff.expo, (Exception), max_tries=3, max_time=30)
        async def process_request(request):
            """Process a single request with rate limiting and retries."""
            async with sem:  # Rate limiting
                try:
                    # Use the cached version for the actual API call
                    return await asyncio.to_thread(create_completion_cached, **request)
                except Exception as e:
                    logging.error(f"Error processing request: {str(e)}")
                    raise

        async def process_requests():
            """Process all requests concurrently with rate limiting."""
            logging.info("Starting parallel request processing")

            # Create all tasks
            tasks = [process_request(request) for request in requests]
            logging.info(f"Created {len(tasks)} async tasks")

            # Process all tasks concurrently and maintain order
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert exceptions to None and log errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.error(f"Request at index {i} failed: {str(result)}")
                    results[i] = None
                elif i % 100 == 0:
                    logging.info(f"Processed request at index {i}")

            return results

        # Run async processing
        logging.info("Running async processing")
        results = asyncio.run(process_requests())
        logging.info(f"Completed {len(results)} requests")

        # Format results to match OpenAI's response format
        logging.info("Formatting results")
        formatted_results = []
        for i, result in enumerate(results):
            if result is None:
                logging.warning(f"Request {i} failed and was skipped")
                continue

            formatted_result = {
                # OpenAI API response
                "id": result.id,
                "object": result.object,
                "created": result.created,
                "model": result.model,
                "choices": [
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content,
                            "function_call": choice.message.function_call,
                            "tool_calls": choice.message.tool_calls,
                        },
                        "finish_reason": choice.finish_reason,
                        "logprobs": [
                            {
                                "logprob": logprob_content.logprob,
                                "token": logprob_content.token,
                                "top_logprobs": [
                                    {
                                        "token": top_logprob.token,
                                        "logprob": top_logprob.logprob,
                                    }
                                    for top_logprob in logprob_content.top_logprobs
                                ],
                            }
                            for logprob_content in choice.logprobs.content
                        ],
                    }
                    for choice in result.choices
                ],
                "usage": (
                    {
                        "prompt_tokens": result.usage.prompt_tokens,
                        "completion_tokens": result.usage.completion_tokens,
                        "total_tokens": result.usage.total_tokens,
                    }
                    if result.usage
                    else None
                ),
                # OpenWeights response
                "completion": result.choices[0].message.content,
                "completions": [choice.message.content for choice in result.choices],
                "logprobs": [
                    [
                        {
                            "decoded_token": top_logprob.token,
                            "logprob": top_logprob.logprob,
                        }
                        # Over different tokens at each position
                        for top_logprob in logprob_content.top_logprobs
                    ]
                    # Over the sequence of tokens
                    for logprob_content in result.choices[0].logprobs.content
                ],
            }
            formatted_results.append(formatted_result)
            if i % 100 == 0:  # Log progress every 100 results
                logging.info(f"Formatted {i} results")

        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Completed parallel inference in {duration:.2f} seconds")
        logging.info(f"Average time per request: {duration/len(requests):.2f} seconds")

        return {
            "status": "completed",
            "results": formatted_results,
        }

    def check_use_openai_api(self, model: str) -> bool:
        return model.lower().startswith("openai/")

    def convert_to_openai_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenWeights parameters to OpenAI API parameters."""

        params["model"] = params["model"].replace("openai/", "")
        if "logprobs" in params:
            if isinstance(params["logprobs"], int):
                params["top_logprobs"] = params["logprobs"]
                params["logprobs"] = bool(params["logprobs"])
            else:
                assert params["logprobs"] is None
        else:
            assert "top_logprobs" not in params

        if "n_completions_per_prompt" in params:
            params["n"] = params["n_completions_per_prompt"]
            del params["n_completions_per_prompt"]

        return params

    def get_batch_job_data(self, openai_client, batch_job):
        logging.info(f"Retrieving batch job {batch_job.id}")
        batch_data = openai_client.batches.retrieve(batch_job.id)
        logging.info(f"Batch job status: {batch_data.status}")
        if batch_data.status == "completed":
            logging.info(f"Retrieving results for file {batch_data.output_file_id}")
            file_data = openai_client.files.retrieve(batch_data.output_file_id)
            return {
                "status": "completed",
                "results": json.loads(file_data.content),
                "batch_job_info": json.loads(json.dumps(batch_data.model_dump())),
            }
        else:
            logging.info(f"New batch job {batch_job.id} status: {batch_data.status}")
            return {
                "status": batch_data.status,
                "batch_job_info": json.loads(json.dumps(batch_data.model_dump())),
            }

    def _load_conversations(self, input_file_id: str) -> list:
        """Load and parse conversations from input file."""
        content = self.client.files.content(input_file_id).decode()
        return [json.loads(line) for line in content.split("\n") if line.strip()]

    def _create_completion_request(
        self, model_name: str, messages: list, params: dict
    ) -> dict:
        """Create a single completion request with optional parameters."""
        request = {"model": model_name, "messages": messages}

        # Add optional parameters if provided
        optional_params = [
            "max_tokens",
            "temperature",
            "top_p",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "n",
            "logprobs",
            "top_logprobs",
        ]
        for param in optional_params:
            if param in params:
                request[param] = params[param]

        return request


def custom_hasher(args, kwargs):
    """Hash function for caching that handles nested data structures deterministically.

    Args:
        args: Positional arguments (should be empty)
        kwargs: Keyword arguments to hash

    Returns:
        MD5 hash of the processed arguments
    """

    assert len(args) == 0, f"Unexpected args: {args}"
    kwargs = kwargs.copy()
    kwargs.pop("gateway", None)

    def normalize_value(v):
        """Normalize values for consistent hashing."""
        if v is None:
            return None
        elif isinstance(v, (str, int, bool)):
            return v
        elif isinstance(v, float):
            # Round floats to avoid precision issues
            return round(v, 10)
        elif isinstance(v, Path):
            # Normalize paths to absolute strings
            return str(v.resolve())
        elif isinstance(v, (list, tuple)):
            # Preserve order for lists/tuples - don't sort!
            return [normalize_value(x) for x in v]
        elif isinstance(v, dict):
            # Sort dict keys for consistency
            return {k: normalize_value(v[k]) for k in sorted(v.keys())}
        elif isinstance(v, set):
            # Sets are unordered, so sort them
            return sorted([normalize_value(x) for x in v])
        else:
            # For other types, use their string representation
            return str(v)

    # Normalize all kwargs
    normalized_kwargs = {k: normalize_value(v) for k, v in kwargs.items()}

    # Use JSON for deterministic string representation
    # sort_keys ensures dict keys are in consistent order
    # separators removes whitespace variations
    # ensure_ascii handles unicode consistently
    json_str = json.dumps(
        normalized_kwargs, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )

    # Generate hash
    hash_value = hashlib.md5(json_str.encode("utf-8")).hexdigest()

    # Optional: Log the hash value for debugging
    logging.info(f"Generated hash: {hash_value} when hashing JSON: {json_str[:200]}...")

    return hash_value


@cachier(
    separate_files=True,
    hash_func=custom_hasher,
    cache_dir=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        ".cache",
    ),
    wait_for_calc_timeout=60,
)
def create_openai_file_cached(**kwargs):
    from openai import OpenAI

    return OpenAI().files.create(**kwargs)


@cachier(
    separate_files=True,
    hash_func=custom_hasher,
    cache_dir=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        ".cache",
    ),
    wait_for_calc_timeout=60,
)
def create_completion_cached(**kwargs):
    from openai import OpenAI

    return OpenAI().chat.completions.create(**kwargs)
