import hashlib
import json
import logging
import os
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any, Dict, List, Type

from postgrest.exceptions import APIError
from pydantic import BaseModel

from openweights.client.decorators import supabase_retry
from openweights.cluster.start_runpod import normalize_hardware_type
from openweights.images import OW_UNSLOTH_IMAGE

logger = logging.getLogger(__name__)

# Jobs rows together with their priority, which lives in the job_priority table
# (see supabase/migrations/20261112000000_job_priority.sql).
JOB_SELECT = "*, job_priority(priority)"


def validate_priority(priority) -> int:
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise ValueError(f"priority must be an integer, got {priority!r}")
    if not -(2**31) <= priority < 2**31:
        raise ValueError(f"priority out of range: {priority}")
    return priority


def with_priority(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return a jobs row fetched with JOB_SELECT with the embedded job_priority
    record flattened into an integer ``priority`` (0 if the job has none)."""
    row = dict(row)
    embedded = row.pop("job_priority", None)
    if isinstance(embedded, list):
        embedded = embedded[0] if embedded else None
    if embedded:
        row["priority"] = embedded.get("priority") or 0
    else:
        row["priority"] = row.get("priority") or 0
    return row


def priority_sort_key(job: Dict[str, Any]):
    """Scheduling order of pending jobs: highest priority first, then largest VRAM
    requirement, then oldest."""
    return (
        -(job.get("priority") or 0),
        -(job.get("requires_vram_gb") or 0),
        job.get("created_at") or "",
    )


@dataclass
class Job:
    id: str
    type: str
    status: str
    model: str
    requires_vram_gb: int | None
    docker_image: str
    script: str
    params: Dict[str, Any] | None
    outputs: Dict[str, Any] | None
    organization_id: str
    created_at: datetime
    updated_at: datetime
    worker_id: str | None
    timeout: datetime | None
    allowed_hardware: List[str] | None = None
    priority: int = 0

    submitted_by: str | None = None
    api_token_id: str | None = None

    _manager: "Jobs" = None

    @classmethod
    def from_row(cls, row: Dict[str, Any], manager: "Jobs") -> "Job":
        """Build a Job from a database row, ignoring columns this client does not
        know about, so adding a column to the jobs table does not break clients."""
        known = {f.name for f in fields(cls)}
        row = with_priority(row)
        return cls(**{k: v for k, v in row.items() if k in known}, _manager=manager)

    def _update(self, job):
        self.__dict__.update(job.__dict__)
        return self

    def cancel(self):
        return self._update(self._manager.cancel(self.id))

    def restart(self):
        return self._update(self._manager.restart(self.id))

    def set_priority(self, priority: int):
        """Change the priority of this job. Pending jobs with a higher priority
        are started first; the default is 0."""
        return self._update(self._manager.set_priority(self.id, priority))

    def __getitem__(self, key):
        return getattr(self, key)

    @property
    def runs(self):
        return self._manager._ow.runs.list(job_id=self.id)

    def download(self, target_dir: str, only_last_run: bool = True):
        if only_last_run:
            self.runs[-1].download(target_dir)
        else:
            for run in self.runs:
                run.download(f"{target_dir}/{run.id}")

    def refresh(self):
        """Refresh the job status and details"""
        return self._update(self._manager.retrieve(self.id))


class Jobs:
    mount: Dict[str, str] = {}  # source path -> target path mapping
    params: Type[BaseModel] = BaseModel  # Pydantic model for parameter validation
    base_image: str = OW_UNSLOTH_IMAGE
    requires_vram_gb: int | None = 24  # Required VRAM in GB

    def __init__(self, ow_instance):
        self._ow = ow_instance

    @property
    def id_predix(self):
        return self.__class__.__name__.lower()

    @property
    def _org_id(self):
        return self._ow.organization_id

    def get_entrypoint(self, validated_params: BaseModel) -> str:
        """Get the entrypoint command for the job.

        Args:
            validated_params: The validated parameters as a Pydantic model instance

        Returns:
            The command to run as a string
        """
        raise NotImplementedError("Subclasses must implement get_entrypoint")

    def _upload_mounted_files(self, extra_files=None) -> Dict[str, str]:
        """Upload all mounted files and return mapping of target paths to file IDs."""
        uploaded_files = {}

        mount = self.mount.copy()
        if extra_files:
            mount.update(extra_files)
        for source_path, target_path in mount.items():
            # Handle both files and directories
            if os.path.isfile(source_path):
                with open(source_path, "rb") as f:
                    file_response = self._ow.files.create(f, purpose="custom_job_file")
                uploaded_files[target_path] = file_response["id"]
            elif os.path.isdir(source_path):
                # For directories, upload each file maintaining the structure
                for root, _, files in os.walk(source_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, source_path)
                        target_file_path = os.path.join(target_path, rel_path)

                        with open(full_path, "rb") as f:
                            file_response = self._ow.files.create(
                                f, purpose="custom_job_file"
                            )
                        uploaded_files[target_file_path] = file_response["id"]
            else:
                raise ValueError(f"Mount source path does not exist: {source_path}")

        return uploaded_files

    @supabase_retry()
    def list(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List jobs"""
        result = (
            self._ow._supabase.table("jobs")
            .select(JOB_SELECT)
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
        return [Job.from_row(row, self) for row in result.data]

    @supabase_retry()
    def retrieve(self, job_id: str) -> Dict[str, Any]:
        """Get job details"""
        result = (
            self._ow._supabase.table("jobs")
            .select(JOB_SELECT)
            .eq("id", job_id)
            .single()
            .execute()
        )
        return Job.from_row(result.data, self)

    @supabase_retry()
    def cancel(self, job_id: str) -> Dict[str, Any]:
        """Cancel a job"""
        logger.info(f"Canceling job: {job_id}")
        result = (
            self._ow._supabase.table("jobs")
            .update({"status": "canceled"})
            .eq("id", job_id)
            .execute()
        )
        logger.info(f"Job canceled: {job_id}")
        return self.retrieve(job_id)

    @supabase_retry()
    def restart(self, job_id: str) -> Dict[str, Any]:
        """Restart a job"""
        logger.info(f"Restarting job: {job_id}")
        result = (
            self._ow._supabase.table("jobs")
            .update({"status": "pending"})
            .eq("id", job_id)
            .execute()
        )
        logger.info(f"Job restarted: {job_id}")
        return self.retrieve(job_id)

    @supabase_retry()
    def set_priority(self, job_id: str, priority: int) -> Job:
        """Set the priority of a job. Pending jobs with a higher priority are
        started first; jobs without an explicit priority have priority 0."""
        priority = validate_priority(priority)
        self._ow._supabase.table("job_priority").upsert(
            {"job_id": job_id, "priority": priority}, on_conflict="job_id"
        ).execute()
        return self.retrieve(job_id)

    def compute_id(self, data: Dict[str, Any], docker_image: str = None) -> str:
        """Compute job ID from data"""
        docker_image = docker_image or data.get("docker_image") or self.base_image
        hash_data = {k: v for k, v in data.items() if k != "docker_image"}
        job_id = f"{self.id_predix}-{hashlib.sha256(json.dumps(hash_data).encode() + docker_image.encode() + self._org_id.encode()).hexdigest()[:12]}"
        if (
            data.get("validated_params", None) is not None
            and data["validated_params"].get("job_id_suffix", None) is not None
        ):
            job_id += f"-{data['validated_params']['job_id_suffix']}"
        elif (
            data.get("params", None) is not None
            and data["params"].get("validated_params", None) is not None
            and data["params"]["validated_params"].get("job_id_suffix", None)
            is not None
        ):
            job_id += f"-{data['params']['validated_params']['job_id_suffix']}"
        return job_id

    @supabase_retry()
    def get_or_create_or_reset(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """If job exists and is [pending, in_progress, completed] return it.
        If job exists and is [failed, canceled] reset it to pending and return it.
        If job doesn't exist, create it and return it.
        """
        # The priority is not part of the job's identity and is stored in its own
        # table. None keeps the priority of an existing job (0 for a new one).
        priority = data.get("priority")
        if priority is not None:
            priority = validate_priority(priority)
        job, current_priority = self._get_or_create_or_reset(
            {k: v for k, v in data.items() if k != "priority"}
        )
        if priority is not None and priority != current_priority:
            return self.set_priority(job.id, priority)
        job.priority = current_priority
        return job

    def _get_or_create_or_reset(self, data: Dict[str, Any]):
        """Returns the job and the priority it has in the database."""
        # Normalize allowed_hardware (e.g. '1x A100 SXM 80GB' -> '1x A100S') before
        # hashing, because workers match it literally against their hardware type.
        if data.get("allowed_hardware") is not None:
            data["allowed_hardware"] = [
                normalize_hardware_type(hardware)
                for hardware in data["allowed_hardware"]
            ]

        data["id"] = data.get("id", self.compute_id(data))
        data["organization_id"] = self._org_id

        try:
            result = (
                self._ow._supabase.table("jobs")
                .select(JOB_SELECT)
                .eq("id", data["id"])
                .single()
                .execute()
            )
        except APIError as e:
            if "contains 0 rows" in str(e):
                logger.info(
                    f"Creating new job: {data['id']} (type: {data.get('type', 'unknown')})"
                )
                result = self._ow._supabase.table("jobs").insert(data).execute()
                logger.info(f"Job created: {data['id']}")
                return Job.from_row(result.data[0], self), 0
            else:
                raise
        job = with_priority(result.data)
        current_priority = job["priority"]
        logger.info(f"Job already exists: {data['id']} (status: {job['status']})")

        # Check if any of the key fields have changed and need updating
        fields_to_sync = [
            "allowed_hardware",
            "requires_vram_gb",
            "docker_image",
            "script",
        ]
        needs_update = any(
            data.get(field) != job.get(field) for field in fields_to_sync
        )

        if job["status"] in ["failed", "canceled"]:
            # Reset job to pending
            logger.info(f"Resetting job {data['id']} from {job['status']} to pending")
            data["status"] = "pending"
            data["outputs"] = None
            result = (
                self._ow._supabase.table("jobs")
                .update(data)
                .eq("id", data["id"])
                .execute()
            )
            return Job.from_row(result.data[0], self), current_priority
        elif job["status"] in ["pending", "in_progress", "completed"]:
            # Update fields if they've changed
            if needs_update:
                logger.info(f"Updating job fields for {data['id']}")
                update_data = {
                    field: data[field] for field in fields_to_sync if field in data
                }
                result = (
                    self._ow._supabase.table("jobs")
                    .update(update_data)
                    .eq("id", data["id"])
                    .execute()
                )
                return Job.from_row(result.data[0], self), current_priority
            return Job.from_row(job, self), current_priority
        else:
            raise ValueError(f"Invalid job status: {job['status']}")

    @supabase_retry()
    def find(self, **params) -> List[Dict[str, Any]]:
        """Find jobs by their JSON values in job.params
        Example:
            jobs = ow.jobs.find(training_file='result:file-abc123')
            jobs = ow.jobs.find(meta={'group': 'hparams'})
        """
        query = self._ow._supabase.table("jobs").select(JOB_SELECT)

        for key, value in params.items():
            if isinstance(value, dict):
                # For nested dictionary values, use containedBy operator
                query = query.contains(f"params->{key}", value)
            elif isinstance(value, bool):
                # Convert boolean to lowercase string for JSON comparison
                query = query.eq(f"params->>{key}", str(value).lower())
            else:
                # For simple values, use eq operator with ->> for JSON text extraction
                query = query.eq(f"params->>{key}", value)

        data = query.execute().data

        return [Job.from_row(row, self) for row in data]

    def create(self, **params) -> Dict[str, Any]:
        """Create and submit a custom job.

        Args:
            **params: Parameters for the job, will be validated against self.params
            allowed_hardware: Optional list of allowed hardware configurations (e.g. ['2x A100', '4x H100'])
            docker_image: Optional override for the Docker image to use
            priority: Optional integer; pending jobs with a higher priority are started first (default 0)

        Returns:
            The created job object
        """
        # Extract allowed_hardware and docker_image if provided
        allowed_hardware = params.pop("allowed_hardware", None)
        docker_image = params.pop("docker_image", None) or self.base_image
        priority = params.pop("priority", None)

        # Validate parameters
        validated_params = self.params(**params)

        # Upload mounted files
        mounted_files = self._upload_mounted_files()

        # Get entrypoint command
        entrypoint = self.get_entrypoint(validated_params)

        # Create job
        job_data = {
            "type": "custom",
            "docker_image": docker_image,
            "requires_vram_gb": params.get("requires_vram_gb", self.requires_vram_gb),
            "script": entrypoint,
            "allowed_hardware": allowed_hardware,
            "priority": priority,
            "params": {
                "validated_params": validated_params.model_dump(),
                "mounted_files": mounted_files,
            },
        }

        return self.get_or_create_or_reset(job_data)
