from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from datetime import datetime

class Job(BaseModel):
    id: str
    type: str
    status: str
    model: Optional[str] = None
    script: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    requires_vram_gb: Optional[int] = None
    docker_image: Optional[str] = None
    created_at: datetime

class Run(BaseModel):
    id: int
    job_id: str
    worker_id: Optional[str]
    status: str
    log_file: Optional[str]
    created_at: datetime

class Worker(BaseModel):
    id: str
    status: str
    gpu_type: Optional[str]
    gpu_count: Optional[int]
    vram_gb: Optional[int]
    docker_image: Optional[str]
    cached_models: Optional[List[str]]
    pod_id: Optional[str]
    ping: Optional[datetime]
    created_at: datetime

class JobWithRuns(Job):
    runs: List[Run]

class RunWithJobAndWorker(Run):
    job: Job
    worker: Optional[Worker]

class WorkerWithRuns(Worker):
    runs: List[Run]