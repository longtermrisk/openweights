# OpenWeights
An openai-like sdk with the flexibility of working on a local GPU: finetune, inference, API deployments and custom workloads on managed runpod instances.

# Installation
Clone the repo and run `pip install -e .`.
Then add your `$OPENWEIGHTS_API_KEY` to the `.env`. You can create one via the [dashboard](https://yzxz5i6z2x2f0y-8124.proxy.runpod.net/).

# Quickstart
```python
from openweights import OpenWeights
import openweights.jobs.unsloth     # This import makes ow.fine_tuning available
ow = OpenWeights()

with open('tests/preference_dataset.jsonl', 'rb') as file:
    file = ow.files.create(file, purpose="preference")

job = ow.fine_tuning.create(
    model='unsloth/llama-3-8b-Instruct',
    training_file=file['id'],
    loss='dpo'
)
```
Currently supported are sft, dpo and orpo on models up to 32B in bf16 or 70B in 4bit. More info: [Fine-tuning Options](docs/finetuning.md)

# Overview

A bunch of things work out of the box: for example lora finetuning, API deployments, batch inference jobs, or running MMLU-pro and inspect-ai evals. However, the best and most useful and coolest feature is that you can very easily [create your own jobs](example/custom_job/) or modify existing ones: all built-in jobs can just as well live outside of this repo. For example, you can copy and modify [the finetuning code](openweights/jobs/unsloth): when a job is created, the necessary source code is uploaded as part of the job and therefore does not need to be part of this repo.

## Inference
```python
from openweights import OpenWeights
import openweights.jobs.inference     # This import makes ow.inference available
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
    output = client.files.content(output_file_id).decode('utf-8')
    print(output)
```
Code: [`openweights/jobs/inference`](openweights/jobs/inference)

## OpenAI-like vllm API
```py
from openweights import OpenWeights
import openweights.jobs.vllm        # this makes ow.api available

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
Code: [`openweights/jobs/vllm`](openweights/jobs/vllm)


API jobs can never complete, they stop either because they are canceled or failed. API jobs have a timeout 15 minutes in the future when they are being created, and while a `TemporaryAPI` is alive (after `api.up()` and before `api.down()` has been called), it resets the timeout every minute. This ensures that an API is alive while the process that created it is running, at that it will automatically shut down later - but not immediately so that during debugging you don't always have to wait for deployment.

## `ow.chat.completions`
We implement an efficient chat client that handles local caching on disk when a seed is provided as well as concurrency management and backpressure. It also deploys models when they are not openai models and not already deployed. We make many guesses that are probably suboptimal for many use cases when we automatically deploy models - for those cases you should explicitly use `ow.api.deploy`.

## Inspect-AI
```python

from openweights import OpenWeights
import openweights.jobs.inspect_ai     # this makes ow.inspect_ai available
ow = OpenWeights()

job = ow.inspect_ai.create(
    model='meta-llama/Llama-3.3-70B-Instruct',
    eval_name='inspect_evals/gpqa_diamond',
    options='--top-p 0.9', # Can be any options that `inspect eval` accepts - we simply pass them on without validation
)

if job.status == 'completed':
    job.download(f"{args.local_save_dir}")
```


## MMLU-pro
```python
from openweights import OpenWeights
import openweights.jobs.mmlu_pro        # this makes ow.mmlu_pro available
ow = OpenWeights()

job = ow.mmlu_pro.create(
    model=args.model,
    ntrain=args.ntrain,
    selected_subjects=args.selected_subjects,
    save_dir=args.save_dir,
    global_record_file=args.global_record_file,
    gpu_util=args.gpu_util
)

if job.status == 'completed':
    job.download(f"{args.local_save_dir}")
```

# General notes

## Job and file IDs are content hashes
The `job_id` is based on the params hash, which means that if you submit the same job many times, it will only run once. If you resubmit a failed or canceled job, it will reset the job status to `pending`.

## More docs
- [Fine-tuning Options](docs/finetuning.md)
- [APIs](docs/api.md)
- [Custom jobs](example/custom_job/)

## Development
Start a pod in dev mode - that allows ssh'ing into it without starting a worker automatically. This is useful to debug the worker.
```sh
python openweights/cluster/start_runpod.py A6000 finetuning --dev_mode=true
```

## Architecture Overview

### Core Components

1. **Client Layer** (`openweights/client/`):
   - `OpenWeights` class: Main client entry point with organization-based authentication
   - `Jobs`: Base class for all job types with mounting, validation, and execution
   - `Files`: File upload/download management with content hashing
   - `Events`: Job monitoring and metrics collection
   - `TemporaryApi`: Manages temporary API deployments with automatic timeout

2. **Job System** (`openweights/jobs/`):
   - Jobs are Python classes that inherit from `Jobs` base class
   - Each job type registers itself using the `@register("name")` decorator
   - Jobs define: mounted source files, Docker image, VRAM requirements, and entrypoint commands
   - Built-in job types:
     - `fine_tuning` (unsloth): SFT, DPO, ORPO fine-tuning with LoRA
     - `inference`: Batch inference with OpenAI API compatibility
     - `api` (vllm): Deploy models as OpenAI-compatible APIs
     - `inspect_ai`: Run Inspect-AI evaluations
     - `mmlu_pro`: MMLU-Pro benchmark evaluations

3. **Cluster Management** (`openweights/cluster/`):
   - `start_runpod.py`: Provisions RunPod instances
   - `supervisor.py`: Manages job execution on workers
   - `org_manager.py`: Organization-level resource management

4. **Worker System** (`openweights/worker/`):
   - Runs on RunPod instances to execute jobs
   - Downloads mounted files and executes job scripts
   - Reports progress and results back to the central system

### Key Patterns

- **Content-based IDs**: Job and file IDs are SHA256 hashes of their content, enabling automatic deduplication
- **Modular Job System**: All job types follow the same pattern and can be easily extended or replaced
- **Automatic VRAM Estimation**: Jobs can guess required VRAM based on model size and quantization
- **LoRA Support**: First-class support for LoRA adapters in both training and inference
- **OpenAI Compatibility**: Inference and API jobs provide OpenAI-compatible interfaces

### Data Flow

1. User creates job via client SDK
2. Job parameters are validated and source files are uploaded
3. Job is queued in the database with computed content hash as ID
4. RunPod worker picks up the job and downloads mounted files
5. Worker executes the job script with validated parameters
6. Results are uploaded and job status is updated

### Custom Jobs

Creating custom jobs is straightforward:
```python
@register('my_custom_job')
class MyCustomJob(Jobs):
    mount = {'local/script.py': 'script.py'}  # Files to upload
    base_image = 'nielsrolf/ow-default'       # Docker image
    requires_vram_gb = 24                     # VRAM requirement

    def get_entrypoint(self, params):
        return f'python script.py {json.dumps(params.model_dump())}'
```

## Important Implementation Details

- Job IDs are deterministic based on parameters and mounted files
- Organization-based multi-tenancy with Supabase authentication
- Automatic model deployment grouping for efficient resource usage
- Built-in request caching (when seeds are provided) and rate limiting
- Support for both sync and async client interfaces
- Automatic timeout management for API deployments

## File Organization

- `openweights/`: Main package
  - `client/`: Core client logic and API interfaces
  - `jobs/`: Job implementations organized by type
  - `cluster/`: RunPod and resource management
  - `worker/`: Job execution runtime
  - `dashboard/`: Web UI (React frontend + FastAPI backend)
- `docs/`: Additional documentation
- `example/`: Usage examples including custom job creation
