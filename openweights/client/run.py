import hashlib
import os
import sys
from datetime import datetime
from typing import Any, BinaryIO, Dict, List, Optional

from postgrest.exceptions import APIError

from openweights.client.decorators import supabase_retry
from supabase import Client


class Run:
    def __init__(
        self,
        client: "OpenWeights",
        job_id: Optional[str] = None,
        worker_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        self.client = client
        self._supabase = client._supabase
        self.organization_id = organization_id
        self.id = run_id or os.getenv("OPENWEIGHTS_RUN_ID")
        if self.id:
            self._fetch_and_init_run(job_id, worker_id)
        else:
            # Create new run
            data = {"status": "in_progress"}

            if job_id:
                data["job_id"] = job_id
            else:
                # Create a new script job
                command = " ".join(sys.argv)
                job_data = {
                    "id": f"sjob-{hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:12]}",
                    "type": "script",
                    "script": command,
                    "status": "in_progress",
                    "organization_id": self.organization_id,
                }
                job_result = self._supabase.table("jobs").insert(job_data).execute()
                data["job_id"] = job_result.data[0]["id"]

            if worker_id:
                data["worker_id"] = worker_id

            # Get organization_id from job
            job = self._get_job_org_id_with_retry(data["job_id"])
            if not job.data:
                raise ValueError(f"Job {data['job_id']} not found")

            result = self._supabase.table("runs").insert(data).execute()
            self._load_data(result.data[0])

    @supabase_retry()
    def _fetch_and_init_run(self, job_id, worker_id):
        """Fetch run data and initialize"""
        try:
            result = (
                self._supabase.table("runs")
                .select("*")
                .eq("id", self.id)
                .single()
                .execute()
            )
        except APIError as e:
            if "contains 0 rows" in str(e):
                raise ValueError(f"Run with ID {self.id} not found")
            raise

        run_data = result.data
        if job_id and run_data["job_id"] != job_id:
            raise ValueError(
                f"Run {self.id} is associated with job {run_data['job_id']}, not {job_id}"
            )

        if worker_id and run_data["worker_id"] != worker_id:
            # reassign run to self
            run_data["worker_id"] = worker_id
            result = (
                self._supabase.table("runs")
                .update(run_data)
                .eq("id", self.id)
                .execute()
            )
            run_data = result.data[0]

        self._load_data(run_data)

    @supabase_retry()
    def _get_job_org_id_with_retry(self, job_id):
        """Get job organization ID with retry logic"""
        return (
            self._supabase.table("jobs")
            .select("organization_id")
            .eq("id", job_id)
            .single()
            .execute()
        )

    def _load_data(self, data: Dict[str, Any]):
        self.id = data["id"]
        self.job_id = data["job_id"]
        self.worker_id = data.get("worker_id")
        self.status = data["status"]
        self.log_file = data.get("log_file")
        self.created_at = data["created_at"]

    @staticmethod
    def get(supabase: Client, run_id: int) -> "Run":
        """Get a run by ID"""
        run = Run(supabase)
        run.id = run_id
        try:
            result = (
                supabase.table("runs").select("*").eq("id", run_id).single().execute()
            )
        except APIError as e:
            if "contains 0 rows" in str(e):
                raise ValueError(f"Run with ID {run_id} not found")
            raise
        run._load_data(result.data)
        return run

    @supabase_retry(return_on_exhaustion=None)
    def update(self, status: Optional[str] = None, logfile: Optional[str] = None):
        """Update run status and/or logfile"""
        data = {}
        if status:
            data["status"] = status
        if logfile:
            data["log_file"] = logfile

        if data:
            result = (
                self._supabase.table("runs").update(data).eq("id", self.id).execute()
            )
            self._load_data(result.data[0])

    @supabase_retry(return_on_exhaustion=None)
    def log(self, event_data: Dict[str, Any], file: Optional[BinaryIO] = None):
        """Log an event for this run"""
        if file:
            file_id = self._supabase.files.create(file, purpose="event")["id"]
        else:
            file_id = None
        data = {"run_id": self.id, "data": event_data, "file": file_id}
        self._supabase.table("events").insert(data).execute()

    @property
    def events(self) -> List[Dict[str, Any]]:
        """Get all events for this run"""
        result = (
            self._supabase.table("events").select("*").eq("run_id", self.id).execute()
        )
        return result.data

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def download(self, target_dir):
        """Download artifacts for this run"""
        os.makedirs(target_dir, exist_ok=True)
        # Logs
        if self.log_file is not None:
            log = self.client.files.content(self.log_file)
            with open(f"{target_dir}/{self.id}.log", "wb") as f:
                f.write(log)
        events = self.events
        for i, event in enumerate(events):
            if event["data"].get("file"):
                file = self.client.files.content(event["data"]["file"])
                rel_path = (
                    event["data"]["path"]
                    if "path" in event["data"]
                    else event["data"]["file"]
                )
                path = f"{target_dir}/{rel_path}"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(file)


class Runs:
    def __init__(self, client: "OpenWeights"):
        self.client = client
        self._supabase = client._supabase

    @supabase_retry()
    def list(
        self,
        job_id: Optional[str] = None,
        worker_id: Optional[str] = None,
        limit: int = 10,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List runs by job_id or worker_id"""
        query = self._supabase.table("runs").select("*").limit(limit)
        if job_id:
            query = query.eq("job_id", job_id)
        if worker_id:
            query = query.eq("worker_id", worker_id)
        if status:
            query = query.eq("status", status)
        result = query.execute()
        return [Run(self.client, run_id=row["id"]) for row in result.data]
