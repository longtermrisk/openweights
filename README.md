This repo is research code. Please use github issues or contact me via email (niels dot warncke at gmail dot com) or slack when you encounter issues.

# OpenWeights
An openai-like sdk with the flexibility of working on a local GPU: finetune, inference, API deployments and custom workloads on managed runpod instances.


## Installation
Run `pip install openweights` or install from source via `pip install -e .`
Then add your `$OPENWEIGHTS_API_KEY` to the `.env`. You can create one via the [dashboard](https://yzxz5i6z2x2f0y-8124.proxy.runpod.net/).

---

## Quickstart
```python
from openweights import OpenWeights

ow = OpenWeights()

training_file = ow.files.upload("data/train.jsonl", purpose="conversations")["id"]
job = ow.fine_tuning.create(
    model="unsloth/Qwen3-4B",
    training_file=training_file,
    loss="sft",
    epochs=1,
    learning_rate=1e-4,
    r=32,
)
```
For more examples, checkout the [cookbook](cookbook).

# Overview

`openweights` lets you submit jobs that will be run on managed runpod instances. It supports a range of built-in jobs out-of-the-box, but is built for custom workloads.

## Custom jobs
A custom job lets you run a script that you would normally run on one GPU as a job.

Example:
```python
from openweights import OpenWeights, register, Jobs
ow = OpenWeights()

@register('my_custom_job')
class MyCustomJob(Jobs):
    mount = {
        'local/path/to/script.py': 'script.py',
        'local/path/to/dir/': 'dirname/'
    }
    params: Type[BaseModel] = MyParams  # Your Pydantic model for params
    requires_vram_gb: int = 24
    base_image: str = 'nielsrolf/ow-default' # optional

    def get_entrypoint(self, validated_params: BaseModel) -> str:
        # Get the entrypoint command for the job.
        return f'python script.py {json.dumps(validated_params.model_dump())}'
```

[More details](cookbook/custom_job/)


## Built-in jobs

### Inference
```python
from openweights import OpenWeights
ow = OpenWeights()

file = ow.files.create(
  file=open("mydata.jsonl", "rb"),
  purpose="conversations"
)

job = ow.inference.create(
    model=model,
    input_file_id=file['id'],
    max_tokens=1000,
    temperature=1,
    min_tokens=600,
)

# Wait or poll until job is done, then:
if job.status == 'completed':
    output_file_id = job['outputs']['file']
    output = ow.files.content(output_file_id).decode('utf-8')
    print(output)
```
[More details](cookbook/inference/)

### OpenAI-like vllm API
```py
from openweights import OpenWeights

ow = OpenWeights()

model = 'unsloth/llama-3-8b-Instruct'

# async with ow.api.deploy(model) also works
with ow.api.deploy(model):            # async with ow.api.deploy(model) also works
    # entering the context manager is equivalent to temp_api = ow.api.deploy(model) ; api.up()
    completion = ow.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "is 9.11 > 9.9?"}]
    )
    print(completion.choices[0].message)       # when this context manager exits, it calls api.down()
```
[More details](cookbook/api-deployment/)


### Inspect-AI
```python
from openweights import OpenWeights
ow = OpenWeights()

job = ow.inspect_ai.create(
    model='meta-llama/Llama-3.3-70B-Instruct',
    eval_name='inspect_evals/gpqa_diamond',
    options='--top-p 0.9', # Can be any options that `inspect eval` accepts - we simply pass them on without validation
)

if job.status == 'completed':
    job.download(f"{args.local_save_dir}")
```

---


## General notes

### Job and file IDs are content hashes
The `job_id` is based on the params hash, which means that if you submit the same job many times, it will only run once. If you resubmit a failed or canceled job, it will reset the job status to `pending`.

### Running a dev pod
Start a pod in dev mode - that allows ssh'ing into it without starting a worker automatically. This is useful to debug the worker.
```sh
python openweights/cluster/start_runpod.py A6000 finetuning --dev_mode=true
```
