# Deploy a model as a temporary Openai-like API

You can deploy models as openai-like APIs in one of the following ways (sorted from highest to lowest level of abstraction)
- create chat completions via `ow.chat.completions.create` or `await ow.async_chat.completions.create` - this will deploy models when needed. This queues to-be-deployed models for 5 seconds and then deploys them via `ow.multi_deploy`. This client is optimized to not overload the vllm server it is talking to.
- pass a list of models to deploy to `ow.multi_deploy` - this takes a list of models or lora adapters, groups them by `base_model`, and deploys all lora adapters of the same base model on one API to save runpod resources. Calls `ow.deploy` for each single deployment job. [Example](gradio_ui.py)
- `ow.api.deploy` - takes a single model and optionally a list of lora adapters, then creates a job of type `api`. Returns a `openweights.client.temporary_api.TemporaryAPI` object. [Example](../example/gradio_ui_with_temporary_api.py)

API jobs can never complete, they stop either because they are canceled or failed. API jobs have a timeout 15 minutes in the future when they are being created, and while a `TemporaryAPI` is alive (after `api.up()` and before `api.down()` has been called), it resets the timeout every minute. This ensures that an API is alive while the process that created it is running, at that it will automatically shut down later - but not immediately so that during debugging you don't always have to wait for deployment.
