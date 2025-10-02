# Development
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


# TTL (Time To Live) Feature of `openweights/cluster/start_runpod.py`

The TTL feature provides automatic pod termination to prevent runaway costs and ensure resource cleanup.

## Overview

- **Default TTL**: 24 hours for all pods
- **Automatic termination**: Pods self-terminate when TTL expires
- **Extensible**: TTL can be extended from within the pod
- **Dev mode support**: TTL monitoring runs for both dev and worker instances

## Usage

### Starting pods with custom TTL

```bash
# Start dev instance with default 24-hour TTL
python openweights/cluster/start_runpod.py A100 default --dev_mode=true

# Start dev instance with 2-hour TTL
python openweights/cluster/start_runpod.py A100 default --dev_mode=true --ttl_hours=2

# Start worker with 12-hour TTL
python openweights/cluster/start_runpod.py A100 finetuning --ttl_hours=12
```

### Managing TTL from within a pod

Once inside a pod, use the TTL manager utility:

```bash
# Check current TTL status
python openweights/worker/services/ttl_manager.py --check

# Extend TTL by 5 more hours
python openweights/worker/services/ttl_manager.py --extend 5

# Set TTL to 10 hours from now
python openweights/worker/services/ttl_manager.py --set 10
```

### Manual TTL management

You can also manually update the TTL by editing `~/shutdown.txt`:

```bash
python3 -c "
import datetime
with open('~/shutdown.txt', 'w') as f:
    new_time = datetime.datetime.now() + datetime.timedelta(hours=48)
    f.write(new_time.isoformat())
print(f'TTL extended to {new_time}')
"
```

## How it works

1. **TTL Setup**: When a pod starts, the TTL monitor service calculates the shutdown time and writes it to `~/shutdown.txt`
2. **Monitoring**: A background service checks the shutdown time every minute
3. **Termination**: When the current time exceeds the shutdown time, the service terminates the pod using the RunPod API
4. **Extension**: Jobs or users can extend the TTL by updating the shutdown time in the file

## Architecture

- **TTL Monitor Service**: `openweights/worker/services/ttl_monitor.py`
- **TTL Manager Utility**: `openweights/worker/services/ttl_manager.py`
- **Configuration**: TTL passed via `TTL_HOURS` environment variable
- **Shutdown File**: `~/shutdown.txt` contains ISO format datetime

## Environment Variables

- `TTL_HOURS`: Number of hours for TTL (default: 24)
- `RUNPOD_API_KEY`: RunPod API key for pod termination
- `OW_DEV`: Indicates if running in dev mode (affects other services, not TTL)

## Notes

- TTL monitoring runs for both dev and worker instances
- This provides an additional safety net especially for dev instances
- Pod ID is automatically detected from RunPod metadata API
- Failed termination attempts are retried every minute
- TTL can be reset/extended unlimited times before expiration
