import asyncio
from collections import defaultdict

import openai

from openweights.client.decorators import openai_retry
from openweights.client.temporary_api import APIS, TemporaryApi

DEPLOYMENT_QUEUE = []
STARTING = []


class AsyncChatCompletions:
    """This class is a wrapper around the OpenAI Chat API that handles deployment of models,
    request caching (when seeds are provided), and rate limiting.

    Args:
        ow: OpenWeights client
        deploy_kwargs: kwargs to pass to ow.multi_deploy
        request_timeout: timeout for requests in seconds
        per_token_timeout: computes a timeout based on max_tokens * per_token_timeout for each request

        When both timeouts are set, the maximum of the two is used.
    """

    def __init__(self, ow, deploy_kwargs={}, request_timeout=300, per_token_timeout=1):
        self.ow = ow
        self.completions = self
        self.deploy_kwargs = deploy_kwargs
        self.request_timeout = request_timeout
        self.per_token_timeout = per_token_timeout

    async def create(self, model: str, **kwargs):
        async def cached_create(model, **kwargs):
            return await self._create(model, **kwargs)

        return await cached_create(model, **kwargs)

    async def _create(self, model, **kwargs):
        api = await self._get_api(model)
        async with api.sem:
            return await self._create_with_backoff(api, model, **kwargs)

    @openai_retry()
    async def _create_with_backoff(self, api, model, **kwargs):
        timeout = kwargs.pop("timeout", None) or max(
            self.request_timeout,
            kwargs.get("max_tokens", 1) * self.per_token_timeout,
        )
        return await api.async_client.chat.completions.create(
            model=model, timeout=timeout, **kwargs
        )

    async def _get_api(self, model):
        """If the model is not yet deployed, we add it to a queue of to-be-deployed models and wait for 5 seconds
        to group them at once, which is more efficient if the multiple lora finetunes of the same model are deployed.
        """
        if model in APIS:
            return APIS[model]
        if looks_like_openai(model):
            return OpenAiApi()
        if model not in DEPLOYMENT_QUEUE and model not in STARTING:
            print(f"Adding {model} to deployment queue")
            DEPLOYMENT_QUEUE.append(model)
            # Create a task to deploy the model in 5 seconds
            asyncio.create_task(self._wait_and_deploy_queue())
        # Poll until model is deployed
        while model not in APIS:
            await asyncio.sleep(1)
        return APIS[model]

    async def _wait_and_deploy_queue(self, seconds_to_wait=5):
        for _ in range(seconds_to_wait):
            await asyncio.sleep(1)
            if len(DEPLOYMENT_QUEUE) == 0:
                return
        # Move all models from the queue to a list of models to be deployed, such that DEPLOYMENT_QUEUE is empty after this
        models_to_deploy = DEPLOYMENT_QUEUE.copy()
        DEPLOYMENT_QUEUE.clear()
        STARTING.extend(models_to_deploy)
        print(f"Deploying {models_to_deploy}")
        # Deploy the models
        import openweights.jobs.vllm

        apis = self.ow.api.multi_deploy(models_to_deploy, **self.deploy_kwargs)
        # Wait for apis to be up and move each model to APIS as soon as its API is ready
        print(f"Waiting for {models_to_deploy} to be up")
        api_to_models = defaultdict(list)
        for model, api in apis.items():
            api_to_models[api].append(model)

        async def handle_api(api, models):
            await api.async_up()
            for model in models:
                APIS[model] = api
                STARTING.remove(model)

        await asyncio.gather(
            *[handle_api(api, models) for api, models in api_to_models.items()]
        )

    def kill(self, model_id):
        api = APIS.pop(model_id, None)
        if api is not None and api not in APIS.values():
            api.down()
        self.ow.jobs.cancel(api.job_id)


class ChatCompletions(AsyncChatCompletions):
    def create(self, **kwargs):
        assert (
            kwargs.get("stream", False) is False
        ), "ow.chat.completions.create does only support stream=True in async mode"
        response = asyncio.run(super().create(**kwargs))
        return response


def looks_like_openai(model):
    return any(model.lower().startswith(i) for i in ["gpt", "o1", "o3"])


class OpenAiApi(TemporaryApi):
    def __init__(self, concurrents=20, base_url=None, api_key=None, models=[]):
        self.concurrents = concurrents
        self.sem = asyncio.Semaphore(concurrents)
        if base_url is None:
            self.async_client = openai.AsyncOpenAI()
            self.sync_client = openai.OpenAI()
        else:
            self.async_client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
            self.sync_client = openai.OpenAI(base_url=base_url, api_key=api_key)

        for model in models:
            APIS[model] = self
