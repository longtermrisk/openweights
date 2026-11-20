from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Organization(BaseModel):
    id: str
    name: str
    created_at: datetime


class OrganizationCreate(BaseModel):
    name: str
    secrets: Dict[str, str]


class OrganizationUpdate(BaseModel):
    name: str


class OrganizationSecret(BaseModel):
    """A stored secret as the dashboard may see it: never the value itself."""

    name: str
    # "secret": a credential, write-only. "env": an environment variable of the workers.
    kind: str = "env"
    plain_value: Optional[str] = None  # only for settings that are not secrets
    last_four: Optional[str] = None  # only for secrets of 12+ characters
    updated_at: Optional[datetime] = None


class OrganizationSecretValue(BaseModel):
    value: str = Field(min_length=1)


class Member(BaseModel):
    user_id: str
    email: Optional[str] = None
    role: str


class MemberInvite(BaseModel):
    email: str
    role: str = "user"


class MemberRoleUpdate(BaseModel):
    role: str


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


class JobPage(BaseModel):
    items: List[Job]
    total: int


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
    logfile: Optional[str]  # New field for saved logs
    created_at: datetime


class JobWithRuns(Job):
    runs: List[Run]


class RunWithJobAndWorker(Run):
    job: Job
    worker: Optional[Worker]


class WorkerWithRuns(Worker):
    runs: List[Run]


class TokenCreate(BaseModel):
    name: str
    expires_in_days: Optional[int] = None  # None means no expiration
    spending_limit_usd: Optional[Decimal] = Field(
        default=None, ge=0, allow_inf_nan=False
    )


class Token(BaseModel):
    id: str
    name: str
    expires_at: Optional[datetime] = None
    created_at: datetime
    revoked_at: Optional[datetime] = None
    access_token: Optional[str] = None  # Only included when token is first created
