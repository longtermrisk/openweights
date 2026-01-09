# OpenWeights Architecture

## Overview

OpenWeights is a Python SDK for running distributed compute jobs on managed RunPod GPU infrastructure. It provides a simple, OpenAI-like API with full flexibility for custom workloads including fine-tuning, inference, evaluations, and arbitrary Python scripts.

**Key Features:**
- Simple Python SDK with OpenAI-compatible interfaces
- Full flexibility to define custom jobs with arbitrary Docker images and entrypoints
- Automated management of RunPod GPU infrastructure
- Multi-tenancy with organization-based isolation
- Content-addressable job and file IDs for deduplication

## Core Concepts

### What is a Job?

A job is the fundamental unit of work in OpenWeights. It consists of three components:

1. **Docker Image**: The container environment (e.g., `nielsrolf/ow-default`, custom images)
2. **Mounted Files**: Files uploaded to Supabase storage and mounted into the container
3. **Entrypoint**: The command/script to execute (e.g., `python train.py --model=llama`)

Jobs can be:
- **Built-in jobs**: Pre-configured templates for common tasks (fine-tuning with Unsloth, inference with vLLM, Inspect AI evaluations)
- **Custom jobs**: User-defined jobs using the `@register` decorator and `Jobs` base class

### Job Lifecycle States

Jobs progress through the following states:
- `pending`: Job is queued, waiting for a worker
- `in_progress`: Job is currently executing on a worker
- `completed`: Job finished successfully
- `failed`: Job encountered an error
- `canceled`: Job was manually canceled or timed out

### Jobs, Runs, and Events

**Jobs** are reusable templates that define what to execute:
- Identified by content hash of their parameters (e.g., `unsloth-abc123def456`)
- If you submit the same job twice, it uses the existing job (deduplication)
- Contain: docker image, script/entrypoint, parameters, VRAM requirements, hardware constraints

**Runs** are individual executions of a job:
- Each job can have multiple runs (e.g., if restarted after failure)
- Track execution status, assigned worker, and log file
- Created when a worker picks up a job or when using `ow.run` context

**Events** are structured logs/outputs during a run:
- Store arbitrary JSON data (metrics, checkpoints, errors)
- Can reference uploaded files (model checkpoints, outputs)
- Used to track progress and collect results

**Relationship:**
```
Job (1) ──< (many) Runs (1) ──< (many) Events
```

## System Architecture

OpenWeights follows a queue-based architecture with three main components:

### 1. Job Queue (Supabase)

**Database Tables:**
- `jobs`: Job definitions and status
- `runs`: Execution records linking jobs to workers
- `events`: Structured logs and outputs from runs
- `files`: File metadata (actual files stored in Supabase Storage)
- `worker`: Worker registration and health tracking
- `organizations`: Multi-tenant isolation
- `organization_secrets`: API keys and credentials (HF_TOKEN, RUNPOD_API_KEY, etc.)
- `service_account_tokens`: JWT tokens for API authentication

**Key Features:**
- Row Level Security (RLS) ensures organization isolation
- Atomic job acquisition using PostgreSQL functions (`acquire_job`, `update_job_status_if_in_progress`)
- Content-addressable IDs prevent duplicate jobs and files

### 2. Cluster Manager

**Architecture:**
- **Supervisor** (`cluster/supervisor.py`): Top-level process that spawns one manager per organization
- **Organization Manager** (`cluster/org_manager.py`): Manages GPU workers for a single organization

**Responsibilities:**
1. Monitor job queue for pending jobs
2. Provision RunPod workers when jobs arrive
3. Scale workers based on demand (up to MAX_WORKERS per org)
4. Terminate idle workers (idle > 5 minutes)
5. Clean up unresponsive workers (no ping > 2 minutes)
6. Match jobs to hardware based on VRAM requirements and `allowed_hardware` constraints

**Worker Provisioning:**
- Determines GPU type based on job's `requires_vram_gb` and `allowed_hardware`
- Supports multi-GPU configurations (1x, 2x, 4x, 8x GPUs)
- Creates worker record in database with `status='starting'`
- Launches RunPod pod with appropriate Docker image and environment variables
- Updates worker record with `pod_id` when pod is ready

### 3. Workers

**Worker Lifecycle:**
1. **Initialization** (`worker/main.py`):
   - Detects GPU configuration (type, count, VRAM)
   - Runs GPU health checks
   - Registers in database with hardware specs
   - Starts health check background thread

2. **Job Acquisition:**
   - Polls database for pending jobs matching its Docker image
   - Filters by hardware compatibility (VRAM or `allowed_hardware`)
   - Prefers jobs with cached models
   - Uses `acquire_job()` RPC for atomic job claiming

3. **Job Execution:**
   - Downloads mounted files from Supabase Storage
   - Creates temporary directory for job execution
   - Runs job script with `OPENWEIGHTS_RUN_ID` environment variable
   - Streams logs to local file and stdout
   - Monitors for cancellation signals

4. **Result Collection:**
   - Uploads log file to Supabase Storage
   - Uploads files from `/uploads` directory as results
   - Creates events with file references
   - Updates job status atomically

5. **Health Monitoring:**
   - Pings database every 5 seconds
   - Checks for job cancellation or timeout
   - Listens for shutdown signal from cluster manager

6. **Shutdown:**
   - Reverts in-progress jobs to pending (if worker dies)
   - Uploads final logs
   - Terminates RunPod pod

## Authentication & Authorization

### User Authentication Flow

1. **Sign Up**: Users create accounts via Supabase Auth in the dashboard
2. **Organization Creation**: Users create organizations in the dashboard UI
3. **API Key Generation**:
   - Users create API tokens via the CLI: `ow token create --name "my-token"`
   - API tokens are prefixed with `ow_` and stored securely in the `api_tokens` table
   - Tokens can optionally have expiration dates and can be revoked
   - Format: `ow_` followed by a randomly generated secure token

### Authorization Mechanism

**Client-Side:**
```python
ow = OpenWeights(auth_token=os.getenv("OPENWEIGHTS_API_KEY"))
```

The client:
- Accepts an OpenWeights API token (starting with `ow_`)
- Automatically exchanges the API token for a short-lived JWT using `exchange_api_token_for_jwt()` RPC
- Passes the JWT in the `Authorization` header to Supabase
- Extracts organization ID from the JWT using `get_organization_from_token()` RPC
- Supports backwards compatibility: if the token is already a JWT (doesn't start with `ow_`), it uses it directly

**Database-Side:**
- Supabase Row Level Security (RLS) policies automatically filter queries
- Policies check `organization_id` column against the authenticated token's org
- Ensures users can only access their organization's jobs, runs, events, files, workers

**Key RLS Policies:**
- Jobs: Can only query/insert/update jobs where `organization_id` matches token
- Files: Can only access files stored under `organizations/{org_id}/` path
- Workers: Can only view workers belonging to their organization
- Events/Runs: Accessible through their parent job's organization

### Worker Authentication

Workers can operate in two modes:

1. **User-Provided Token**: Uses the organization's service account token from environment
2. **Auto-Generated Token**: Worker creates its own service account token at startup using `create_service_account_token()` RPC

Both approaches leverage RLS to ensure workers can only access their organization's data.

## Client SDK (`openweights/client/`)

### Main Components

**`OpenWeights` class** (`__init__.py`):
- Entry point for SDK
- Initializes Supabase client with auth token
- Provides accessors for jobs, runs, events, files, chat
- Supports custom job registration via `@register` decorator

**`Jobs` class** (`jobs.py`):
- Base class for job definitions
- Handles file uploads and mounting
- Computes content-addressable job IDs
- Implements `get_or_create_or_reset()` for job deduplication

**`Run` class** (`run.py`):
- Represents a single job execution
- Created automatically when jobs execute
- Provides logging and file upload from within jobs
- Can be used standalone for script-based jobs

**`Files` class** (`files.py`):
- Content-addressable file storage
- Format: `{purpose}:file-{hash[:12]}`
- Validates conversation/preference datasets
- Handles organization-specific storage paths

**`Events` class** (`events.py`):
- Structured logging for runs
- Supports file attachments
- Provides `latest()` to extract most recent metric values

## Built-in Jobs

### Fine-Tuning (`openweights/jobs/unsloth/`)

**Jobs:**
- SFT (Supervised Fine-Tuning)
- DPO (Direct Preference Optimization)
- ORPO (Odds Ratio Preference Optimization)
- Weighted SFT (token-level loss weighting)

**Features:**
- Built on Unsloth for memory-efficient training
- Automatic model upload to Hugging Face
- Support for LoRA/QLoRA
- Checkpoint tracking via events
- Log probability tracking

### Inference (`openweights/jobs/inference/`)

**Backend:** vLLM

**Features:**
- Batch inference on JSONL datasets
- OpenAI-compatible API endpoints
- Support for conversation and text completion formats
- Automatic result file upload

### Evaluation (`openweights/jobs/inspect_ai.py`)

**Backend:** Inspect AI framework

**Features:**
- Run evaluations from the Inspect AI library
- Automatic result download
- Flexible eval options pass-through

### Custom Jobs

Users can define custom jobs:

```python
from openweights import OpenWeights, register, Jobs
from pydantic import BaseModel

@register('my_job')
class MyCustomJob(Jobs):
    mount = {'local/script.py': 'script.py'}
    params = MyParamsModel  # Pydantic model
    requires_vram_gb = 24
    base_image = 'nielsrolf/ow-default'

    def get_entrypoint(self, params):
        return f'python script.py --arg={params.arg}'
```

## Default Jobs Directory

The `openweights/jobs/` directory contains several built-in job implementations:
- `unsloth/`: Fine-tuning jobs
- `weighted_sft/`: Token-weighted SFT
- `inference/`: vLLM inference
- `vllm/`: vLLM configuration
- `inspect_ai.py`: Inspect AI evaluations
- `mmlu_pro/`: MMLU Pro evaluation

**Important:** These are simply convenient job definitions included in the repository. There is nothing architecturally special about them—they could just as easily live in external repositories or be defined by users in their own codebases.

## Dashboard (`openweights/dashboard/`)

**Backend** (`backend/main.py`): FastAPI service
- REST API for job/run/worker management
- Proxies Supabase with additional business logic
- Token management endpoints
- File content serving

**Frontend** (`frontend/src/`): React + TypeScript
- Job/run/worker list and detail views
- Real-time log streaming
- Metrics visualization
- Organization management
- Token creation and management

## Storage Architecture

**Supabase Storage** (`files` bucket):
- Organization-scoped paths: `organizations/{org_id}/{file_id}`
- Files are content-addressed with purpose prefix: `{purpose}:file-{hash[:12]}`
- RLS policies enforce organization boundaries

**File Types:**
- `conversations`: Training datasets (validated JSONL)
- `preference`: Preference datasets for DPO/ORPO
- `result`: Job outputs (model checkpoints, predictions)
- `log`: Execution logs
- `custom_job_file`: Mounted files for custom jobs

## Hardware Management

**GPU Selection:**
- Jobs specify `requires_vram_gb` (default: 24)
- Optionally specify `allowed_hardware` list (e.g., `["2x A100", "4x H100"]`)
- Cluster manager determines GPU type and count from `HARDWARE_CONFIG` mapping
- Workers register their exact hardware type (e.g., "2x L40")

**Supported GPUs:**
- NVIDIA L40, A100, A100S, H100N, H100S, H200
- Multi-GPU: 1x, 2x, 4x, 8x configurations
- Configurable in `cluster/start_runpod.py`

**Worker Matching:**
- Workers filter jobs by Docker image first
- Then by hardware compatibility (VRAM or `allowed_hardware` match)
- Prefer jobs with cached models

## Fault Tolerance

**Job Atomicity:**
- `acquire_job()`: Atomically transitions job from pending → in_progress
- `update_job_status_if_in_progress()`: Only updates if still assigned to worker
- Prevents race conditions when multiple workers or managers interact

**Worker Failure Handling:**
1. **Unresponsive Workers** (no ping > 2 min):
   - Cluster manager reverts their in-progress jobs to pending
   - Terminates RunPod pod
   - Marks worker as terminated

2. **Worker Crashes**:
   - `atexit` handler attempts to revert jobs to pending
   - Cluster manager's health check catches missed cases

3. **Repeated Failures**:
   - Workers track last 5 job outcomes
   - Self-terminate if all 5 failed (likely bad worker)

## Content Addressing

**Job IDs:**
```python
job_id = f"{job_type}-{sha256(params + org_id).hex()[:12]}"
```
- Deterministic based on parameters and organization
- Resubmitting identical job returns existing job
- Optional suffix for manual job variants

**File IDs:**
```python
file_id = f"{purpose}:file-{sha256(content + org_id).hex()[:12]}"
```
- Automatic deduplication within organization
- Content changes = new file ID

## Scaling & Performance

**Horizontal Scaling:**
- One organization manager per organization
- Managers provision workers dynamically
- Workers execute jobs concurrently

**Cost Optimization:**
- Idle workers terminated after 5 minutes
- Content addressing prevents redundant work
- Workers prefer cached models to reduce download time

**Limits:**
- `MAX_WORKERS_PER_ORG`: Default 8 (configurable per org)
- Worker TTL: 24 hours (configurable, extendable from within pod)

## Monitoring & Observability

**Worker Health:**
- Ping every 5 seconds
- GPU health checks at startup
- Log aggregation via Supabase Storage

**Job Progress:**
- Events table for structured logging
- Real-time log streaming in dashboard
- Metrics visualization (loss curves, accuracy, etc.)

**System State:**
- Database tables provide complete audit trail
- Worker status: starting, active, shutdown, terminated
- Job status: pending, in_progress, completed, failed, canceled
