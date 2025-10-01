import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Type

from postgrest.exceptions import APIError
from pydantic import BaseModel

from openweights.client.decorators import supabase_retry
from openweights.cluster.start_runpod import GPUs


@dataclass
class Job:
    id: str
    type: str
    status: str
    model: str
    requires_vram_gb: int
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

    _manager: "Jobs" = None

    def _update(self, job):
        self.__dict__.update(job.__dict__)
        return self

    def cancel(self):
        return self._update(self._manager.cancel(self.id))

    def restart(self):
        return self._update(self._manager.restart(self.id))

    def __getitem__(self, key):
        return getattr(self, key)

    @property
    def runs(self):
        return self._manager.client.runs.list(job_id=self.id)

    def download(self, target_dir: str, only_last_run: bool = True):
        if only_last_run:
            self.runs[-1].download(target_dir)
        else:
            for run in self.runs:
                run.download(f"{target_dir}/{run.id}")

    def refresh(self):
        """Refresh the job status and details"""
        if self._manager is None:
            breakpoint()
        return self._update(self._manager.retrieve(self.id))


class Jobs:
    mount: Dict[str, str] = {}  # source path -> target path mapping
    params: Type[BaseModel] = BaseModel  # Pydantic model for parameter validation
    # base_image: str = 'nielsrolf/ow-inference-v2'  # Base Docker image to use
    base_image: str = "nielsrolf/ow-default:v0.6"
    requires_vram_gb: int = 24  # Required VRAM in GB

    def __init__(self, client):
        """Initialize the custom job.
        `client` should be an instance of `openweights.OpenWeights`."""
        self.client = client

    @property
    def id_predix(self):
        return self.__class__.__name__.lower()

    @property
    def _supabase(self):
        return self.client._supabase

    @property
    def _org_id(self):
        return self.client.organization_id

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
                    file_response = self.client.files.create(
                        f, purpose="custom_job_file"
                    )
                uploaded_files[target_path] = file_response["id"]
            elif os.path.isdir(source_path):
                # For directories, upload each file maintaining the structure
                for root, _, files in os.walk(source_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, source_path)
                        target_file_path = os.path.join(target_path, rel_path)

                        with open(full_path, "rb") as f:
                            file_response = self.client.files.create(
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
            self._supabase.table("jobs")
            .select("*")
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
        return [Job(**row, _manager=self) for row in result.data]

    @supabase_retry()
    def retrieve(self, job_id: str) -> Dict[str, Any]:
        """Get job details"""
        result = (
            self._supabase.table("jobs").select("*").eq("id", job_id).single().execute()
        )
        return Job(**result.data, _manager=self)

    @supabase_retry()
    def cancel(self, job_id: str) -> Dict[str, Any]:
        """Cancel a job"""
        result = (
            self._supabase.table("jobs")
            .update({"status": "canceled"})
            .eq("id", job_id)
            .execute()
        )
        return Job(**result.data[0], _manager=self)

    @supabase_retry()
    def restart(self, job_id: str) -> Dict[str, Any]:
        """Restart a job"""
        result = (
            self._supabase.table("jobs")
            .update({"status": "pending"})
            .eq("id", job_id)
            .execute()
        )
        return Job(**result.data[0], _manager=self)

    def compute_id(self, data: Dict[str, Any]) -> str:
        """Compute job ID from data"""
        job_id = f"{self.id_predix}-{hashlib.sha256(json.dumps(data).encode() + self._org_id.encode()).hexdigest()[:12]}"
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
        data["id"] = data.get("id", self.compute_id(data))
        data["organization_id"] = self._org_id

        # Validate allowed_hardware if provided
        if "allowed_hardware" in data and data["allowed_hardware"] is not None:
            valid_suffixes = list(GPUs.keys())
            for hardware in data["allowed_hardware"]:
                if not any(hardware.endswith(suffix) for suffix in valid_suffixes):
                    raise ValueError(
                        f"Invalid hardware configuration: '{hardware}'. "
                        f"Each entry must end with one of: {', '.join(valid_suffixes)}"
                    )

        try:
            result = (
                self._supabase.table("jobs")
                .select("*")
                .eq("id", data["id"])
                .single()
                .execute()
            )
        except APIError as e:
            if "contains 0 rows" in str(e):
                result = self._supabase.table("jobs").insert(data).execute()
                return Job(**result.data[0], _manager=self)
            else:
                raise
        job = result.data

        if job["status"] in ["failed", "canceled"]:
            # Reset job to pending
            data["status"] = "pending"
            result = (
                self._supabase.table("jobs").update(data).eq("id", data["id"]).execute()
            )
            return Job(**result.data[0], _manager=self)
        elif job["status"] in ["pending", "in_progress", "completed"]:
            return Job(**job, _manager=self)
        else:
            raise ValueError(f"Invalid job status: {job['status']}")

    @supabase_retry()
    def find(self, **params) -> List[Dict[str, Any]]:
        """Find jobs by their JSON values in job.params
        Example:
            jobs = client.jobs.find(training_file='result:file-abc123')
            jobs = client.jobs.find(meta={'group': 'hparams'})
        """
        query = self._supabase.table("jobs").select("*")

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

        return [Job(**row, _manager=self) for row in data]

    def create(self, **params) -> Dict[str, Any]:
        """Create and submit a custom job.

        Args:
            **params: Parameters for the job, will be validated against self.params
            allowed_hardware: Optional list of allowed hardware configurations (e.g. ['2x A100', '4x H100'])

        Returns:
            The created job object
        """
        # Extract allowed_hardware if provided
        allowed_hardware = params.pop("allowed_hardware", None)

        # Validate parameters
        validated_params = self.params(**params)

        # Upload mounted files
        mounted_files = self._upload_mounted_files()

        # Get entrypoint command
        entrypoint = self.get_entrypoint(validated_params)

        # Create job
        job_data = {
            "type": "custom",
            "docker_image": self.base_image,
            "requires_vram_gb": params.get("requires_vram_gb", self.requires_vram_gb),
            "script": entrypoint,
            "params": {
                "validated_params": validated_params.model_dump(),
                "mounted_files": mounted_files,
            },
        }

        # Add allowed_hardware if specified
        if allowed_hardware is not None:
            job_data["allowed_hardware"] = allowed_hardware

        return self.get_or_create_or_reset(job_data)
