"""
Organization-specific cluster manager.
"""

import io
import logging
import os
import signal
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
import runpod
from dotenv import load_dotenv

from openweights.client import (
    _SUPABASE_ANON_KEY,
    _SUPABASE_URL,
    ApiTokenError,
    OpenWeights,
)
from openweights.client.decorators import supabase_retry
from openweights.cluster.start_runpod import (
    HARDWARE_REGISTRY,
    is_spending_limit_error,
    parse_hardware_config,
    populate_hardware_config,
)
from openweights.cluster.start_runpod import start_worker as runpod_start_worker

# Load environment variables
load_dotenv()

# Constants
POLL_INTERVAL = 15
IDLE_THRESHOLD = 300
STARTUP_THRESHOLD = 600
UNRESPONSIVE_THRESHOLD = 120
MAX_WORKERS = os.environ.get("MAX_WORKERS", 8)

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def format_cooldown_remaining(until: float, now: Optional[float] = None) -> str:
    """Format human-readable time remaining until epoch ``until``.

    Args:
        until: Unix timestamp (seconds) when the cooldown ends.
        now: Reference ``time.time()``; defaults to the current wall time.

    Returns:
        A short duration string such as ``42m15s`` or ``3h0m0s``.
    """
    t_now = time.time() if now is None else now
    rem = max(0.0, until - t_now)
    whole = int(rem)
    if whole >= 3600:
        h, r = divmod(whole, 3600)
        m, s = divmod(r, 60)
        return f"{h}h{m}m{s}s"
    if whole >= 60:
        m, s = divmod(whole, 60)
        return f"{m}m{s}s"
    if rem >= 1.0:
        return f"{whole}s"
    return f"{rem:.1f}s"


def determine_gpu_type(required_vram, allowed_hardware=None, runpod_client=None):
    """Determine the best GPU type and count for the required VRAM.

    Args:
        required_vram: Required VRAM in GB
        allowed_hardware: List of allowed hardware configurations (e.g. ['2x A100', '4x H100'])

    Returns:
        Tuple of (gpu_type, count)
    """
    candidates = HARDWARE_REGISTRY.get_candidate_hardware(
        required_vram,
        allowed_hardware,
        runpod_client=runpod_client,
    )
    if candidates:
        count, gpu = parse_hardware_config(candidates[0])
        return gpu, count
    raise ValueError(
        f"No suitable GPU configuration found for VRAM requirement {required_vram}"
    )


class OrganizationManager:
    def __init__(self):
        self._ow = OpenWeights()
        self.org_id = self._ow.organization_id
        print("org name", self._ow.org_name)
        self.shutdown_flag = False

        # Set up RunPod client
        runpod.api_key = os.environ["RUNPOD_API_KEY"]
        self.hardware_registry = HARDWARE_REGISTRY
        populate_hardware_config(runpod, force=True)

        # Register signal handlers
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)

    @property
    def worker_env(self):
        secrets = self.get_secrets()
        # Remove SUPABASE_URL and SUPABASE_ANON_KEY from secrets if present
        # to avoid duplicate keyword argument error
        secrets.pop("SUPABASE_URL", None)
        secrets.pop("SUPABASE_ANON_KEY", None)
        return dict(
            SUPABASE_URL=_SUPABASE_URL,
            SUPABASE_ANON_KEY=_SUPABASE_ANON_KEY,
            **secrets,
        )

    @supabase_retry()
    def get_secrets(self) -> Dict[str, str]:
        """Get organization secrets from the database, with local environment overrides.

        When running a self-managed cluster, secrets from the local environment
        are used as base values, and secrets from the database (if any) override them.
        This allows users to run their own cluster without submitting secrets to the service.
        """
        # Start with local environment variables
        secrets = {}

        # Common secret keys that might be needed
        secret_keys = [
            "RUNPOD_API_KEY",
            "HF_TOKEN",
            "WANDB_API_KEY",
            "MAX_WORKERS",
            "OPENAI_API_KEY",
            "OPENWEIGHTS_API_KEY",
        ]

        # If custom env vars were provided via env file, add them to the list
        # This is communicated via the _OW_CUSTOM_ENV_VARS environment variable
        if "_OW_CUSTOM_ENV_VARS" in os.environ:
            custom_vars = os.environ["_OW_CUSTOM_ENV_VARS"].split(",")
            # Add custom vars to secret_keys, avoiding duplicates
            for var in custom_vars:
                if var and var not in secret_keys:
                    secret_keys.append(var)

        for key in secret_keys:
            if key in os.environ:
                secrets[key] = os.environ[key]

        # Try to get overrides from database (optional)
        try:
            result = (
                self._ow._supabase.table("organization_secrets")
                .select("name, value")
                .eq("organization_id", self.org_id)
                .execute()
            )

            # Override with database values if present
            for secret in result.data:
                secrets[secret["name"]] = secret["value"]
        except Exception as e:
            # If database query fails, just use environment variables
            logger.warning(f"Could not fetch secrets from database: {e}")
            logger.info("Using only local environment variables for secrets")

        return secrets

    def handle_shutdown(self, _signum, _frame):
        """Handle shutdown signals gracefully."""
        logger.info(
            f"Received shutdown signal, cleaning up organization {self.org_id}..."
        )
        self.shutdown_flag = True

    @supabase_retry()
    def get_running_workers(self):
        """Get all active and starting workers for this organization."""
        return (
            self._ow._supabase.table("worker")
            .select("*")
            .eq("organization_id", self.org_id)
            .in_("status", ["active", "starting", "shutdown"])
            .execute()
            .data
        )

    @supabase_retry()
    def get_pending_jobs(self):
        """Get all pending jobs for this organization."""
        return (
            self._ow._supabase.table("jobs")
            .select("*")
            .eq("organization_id", self.org_id)
            .eq("status", "pending")
            .order("requires_vram_gb", desc=True)
            .order("created_at", desc=False)
            .execute()
            .data
        )

    @supabase_retry()
    def get_idle_workers(self, running_workers):
        """Returns a list of idle workers."""
        idle_workers = []
        current_time = time.time()

        for worker in running_workers:
            # Skip if the worker is not a pod
            if not worker.get("pod_id"):
                continue
            # If the worker was started less than STARTUP_THRESHOLD minutes ago, skip it
            worker_created_at = datetime.fromisoformat(
                worker["created_at"].replace("Z", "+00:00")
            ).timestamp()
            if current_time - worker_created_at < STARTUP_THRESHOLD:
                continue

            # Find the latest run associated with the worker
            runs = (
                self._ow._supabase.table("runs")
                .select("*")
                .eq("worker_id", worker["id"])
                .execute()
                .data
            )
            if runs:
                # Sort by created_at to get the most recent run
                last_run = max(runs, key=lambda r: r["updated_at"])
                last_run_updated_at = datetime.fromisoformat(
                    last_run["updated_at"].replace("Z", "+00:00")
                ).timestamp()
                if (
                    last_run["status"] != "in_progress"
                    and current_time - last_run_updated_at > IDLE_THRESHOLD
                ):
                    idle_workers.append(worker)
            else:
                # If no runs found for this worker, consider it idle
                idle_workers.append(worker)

        return idle_workers

    @supabase_retry()
    def fetch_and_save_worker_logs(self, worker):
        """Fetch logs from a worker and save them to a file."""
        try:
            if not worker["pod_id"]:
                return None

            # Fetch logs from worker
            response = requests.get(
                f"https://{worker['pod_id']}-10101.proxy.runpod.net/logs"
            )
            if response.status_code != 200:
                logger.error(
                    f"Failed to fetch logs for worker {worker['id']}: HTTP {response.status_code}"
                )
                return None

            # Save logs to a file using OpenWeights client
            logs = response.text
            file_id = self._ow.files.create(
                file=io.BytesIO(logs.encode("utf-8")), purpose="logs"
            )

            # Update worker record with logfile ID
            self._ow._supabase.table("worker").update({"logfile": file_id}).eq(
                "id", worker["id"]
            ).execute()

            return file_id
        except Exception as e:
            logger.error(f"Error saving logs for worker {worker['id']}: {e}")
            return None

    @supabase_retry()
    def clean_up_unresponsive_workers(self, workers):
        """
        Clean up workers that haven't pinged in more than UNRESPONSIVE_THRESHOLD seconds
        and safely revert their in-progress jobs.
        """
        current_time = datetime.now(timezone.utc)

        for worker in workers:
            try:
                # Parse ping time as UTC
                last_ping = datetime.fromisoformat(
                    worker["ping"].replace("Z", "+00:00")
                ).astimezone(timezone.utc)
                time_since_ping = (current_time - last_ping).total_seconds()
                # Fresh pods can spend several minutes pulling large images before
                # the worker process starts pinging. Reuse STARTUP_THRESHOLD here
                # so cold boots are not killed as "unresponsive" prematurely.
                threshold = (
                    STARTUP_THRESHOLD
                    if worker["status"] == "starting"
                    else UNRESPONSIVE_THRESHOLD
                )
                is_unresponsive = time_since_ping > threshold
            except Exception as e:
                # If parsing ping time fails, treat worker as unresponsive
                is_unresponsive = True
                time_since_ping = "unknown"

            if is_unresponsive:
                logger.info(
                    f"Worker {worker['id']} hasn't pinged for {time_since_ping} seconds. Cleaning up..."
                )

                # Save worker logs before termination (if applicable)
                self.fetch_and_save_worker_logs(worker)

                # 1) Find any runs currently 'in_progress' for this worker.
                runs = (
                    self._ow._supabase.table("runs")
                    .select("*")
                    .eq("worker_id", worker["id"])
                    .eq("status", "in_progress")
                    .execute()
                    .data
                )

                # 2) For each run, set run to 'failed' (or 'canceled'), and
                #    revert the job to 'pending' *only if* it's still in_progress for THIS worker.
                for run in runs:
                    # Mark the run as failed
                    self._ow._supabase.table("runs").update({"status": "failed"}).eq(
                        "id", run["id"]
                    ).execute()

                    # Safely revert the job to 'pending' using your RPC that only updates
                    # if status='in_progress' for the same worker_id.
                    try:
                        self._ow._supabase.rpc(
                            "update_job_status_if_in_progress",
                            {
                                "_job_id": run["job_id"],
                                "_new_status": "pending",  # Must be valid enum label
                                "_worker_id": worker["id"],
                                "_job_outputs": None,
                                "_job_script": None,
                            },
                        ).execute()
                    except Exception as e:
                        logger.error(
                            f"Error reverting job {run['job_id']} to pending: {e}"
                        )

                # 3) If this worker has a RunPod pod, terminate it
                if worker.get("pod_id"):
                    try:
                        logger.info(f"Terminating pod {worker['pod_id']}")
                        runpod.terminate_pod(worker["pod_id"])
                    except Exception as e:
                        logger.error(f"Failed to terminate pod {worker['pod_id']}: {e}")

                # 4) Finally, mark the worker as 'terminated' in the DB
                self._ow._supabase.table("worker").update({"status": "terminated"}).eq(
                    "id", worker["id"]
                ).execute()

    def group_jobs_by_hardware_requirements(self, pending_jobs):
        """Group jobs by their hardware requirements."""
        job_groups = {}

        for job in pending_jobs:
            # Create a key based on allowed_hardware
            if job["allowed_hardware"]:
                # Sort the allowed hardware to ensure consistent grouping
                key = tuple(sorted(job["allowed_hardware"]))
            else:
                # Jobs with no hardware requirements can run on any hardware
                key = None

            if key not in job_groups:
                job_groups[key] = []

            job_groups[key].append(job)

        return job_groups

    @supabase_retry()
    def scale_workers(self, running_workers, pending_jobs):
        """Scale workers according to pending jobs and limits."""
        # Skip provisioning entirely if we're in a spending-limit pause
        if self.hardware_registry.is_spending_limit_paused():
            pause_until = self.hardware_registry.spending_limit_pause_until()
            remaining = int(pause_until - time.time())
            logger.warning(
                "Provisioning paused due to spending limit — %d s remaining. "
                "Jobs will stay pending.",
                max(remaining, 0),
            )
            return

        # Group active workers by docker image
        print("@@@@ Scaling workers")
        running_workers_by_image = {}
        for worker in running_workers:
            image = worker["docker_image"]
            if image not in running_workers_by_image:
                running_workers_by_image[image] = []
            running_workers_by_image[image].append(worker)

        # Group pending jobs by docker image
        pending_jobs_by_image = {}
        for job in pending_jobs:
            image = job["docker_image"]
            if image not in pending_jobs_by_image:
                pending_jobs_by_image[image] = []
            pending_jobs_by_image[image].append(job)

        # Process each docker image type separately
        for docker_image, image_pending_jobs in pending_jobs_by_image.items():
            active_count = len(running_workers_by_image.get(docker_image, []))
            starting_count = len(
                [
                    w
                    for w in running_workers
                    if w["status"] == "starting" and w["docker_image"] == docker_image
                ]
            )

            if len(image_pending_jobs) > 0:
                available_slots = MAX_WORKERS - len(running_workers)
                print(
                    f"available slots: {MAX_WORKERS - len(running_workers)}, MAX_WORKERS: {MAX_WORKERS}, running: {len(running_workers)}, active: {active_count}, starting: {starting_count}, pending jobs for image {docker_image}: {len(image_pending_jobs)}"
                )

                # Group jobs by hardware requirements
                job_groups = self.group_jobs_by_hardware_requirements(
                    image_pending_jobs
                )

                # Process each hardware group separately
                for hardware_key, hardware_jobs in job_groups.items():
                    # Calculate how many workers to start for this hardware group
                    group_num_to_start = min(
                        len(hardware_jobs) - starting_count, available_slots
                    )

                    if group_num_to_start <= 0:
                        continue

                    logging.info(
                        f"Available slots: {available_slots} | Pending jobs for hardware {hardware_key}: {len(hardware_jobs)} | Starting: {starting_count}"
                    )
                    logging.info(
                        f"=> Starting {group_num_to_start} workers for hardware {hardware_key}"
                    )

                    # Sort jobs by VRAM requirement descending
                    hardware_jobs.sort(
                        key=lambda job: job["requires_vram_gb"] or 0, reverse=True
                    )

                    # Split jobs for each worker
                    jobs_batches = [
                        hardware_jobs[i::group_num_to_start]
                        for i in range(group_num_to_start)
                    ]

                    for jobs_batch in jobs_batches:
                        max_vram_required = max(
                            job["requires_vram_gb"] or 0 for job in jobs_batch
                        )
                        try:
                            allowed_hardware = jobs_batch[0]["allowed_hardware"]
                            candidate_hardware = (
                                self.hardware_registry.get_candidate_hardware(
                                    max_vram_required,
                                    allowed_hardware,
                                    runpod_client=runpod,
                                )
                            )
                            if not candidate_hardware:
                                if allowed_hardware:
                                    cooldowns = self.hardware_registry.get_active_cooldown_end_times(
                                        allowed_hardware
                                    )
                                    now_ts = time.time()
                                    ladder_min = "→".join(
                                        str(s // 60)
                                        for s in self.hardware_registry.cooldown_ladder_seconds
                                    )
                                    sched = ", ".join(
                                        f"{hw} escalation_level="
                                        f"{self.hardware_registry.get_cooldown_escalation_level(hw)} "
                                        f"~{format_cooldown_remaining(ts, now_ts)} left (until "
                                        f"{datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()})"
                                        for hw, ts in sorted(cooldowns.items())
                                    )
                                    logger.warning(
                                        "Cannot start worker batch: every allowed_hardware "
                                        "type is on provisioning failure cooldown "
                                        "(failure_threshold=%s, cooldown_ladder_min=%s). Active cooldowns: "
                                        "%s | image=%s requires_vram_gb=%s allowed_hardware=%s",
                                        self.hardware_registry.failure_threshold,
                                        ladder_min,
                                        sched
                                        or "(no active cooldown rows; possible race)",
                                        docker_image,
                                        max_vram_required,
                                        allowed_hardware,
                                    )
                                else:
                                    logger.warning(
                                        "Cannot start worker batch: no RunPod hardware "
                                        "candidates for VRAM autoselect (requires_vram_gb=%s, "
                                        "image=%s). Check RunPod catalog vs VERIFIED_GPUs.",
                                        max_vram_required,
                                        docker_image,
                                    )
                                continue

                            worker_id = f"{self.org_id}-{uuid.uuid4().hex[:8]}"
                            worker_created = False
                            pod = None
                            last_error = None

                            for hardware_type in candidate_hardware:
                                count, gpu = parse_hardware_config(hardware_type)
                                worker_insert_data = {
                                    "status": "starting",
                                    "ping": datetime.now(timezone.utc).isoformat(),
                                    "vram_gb": 0,
                                    "gpu_type": gpu,
                                    "gpu_count": count,
                                    "hardware_type": hardware_type,
                                    "docker_image": docker_image,
                                    "id": worker_id,
                                    "organization_id": self.org_id,
                                    "pod_id": None,
                                }
                                worker_update_data = {
                                    key: value
                                    for key, value in worker_insert_data.items()
                                    if key not in {"id", "organization_id"}
                                }

                                logger.info(
                                    "Starting a new worker - VRAM: %s, Hardware: %s, Image: %s",
                                    max_vram_required,
                                    hardware_type,
                                    docker_image,
                                )

                                if not worker_created:
                                    self._ow._supabase.table("worker").insert(
                                        worker_insert_data
                                    ).execute()
                                    worker_created = True
                                else:
                                    self._ow._supabase.table("worker").update(
                                        worker_update_data
                                    ).eq("id", worker_id).execute()

                                try:
                                    pod = runpod_start_worker(
                                        gpu=gpu,
                                        count=count,
                                        worker_id=worker_id,
                                        image=docker_image,
                                        env=self.worker_env,
                                        name=f"{self._ow.org_name}-{time.time()}-ow-1day",
                                        runpod_client=runpod,
                                    )
                                    self.hardware_registry.record_success(hardware_type)
                                    self._ow._supabase.table("worker").update(
                                        {"pod_id": pod["id"]}
                                    ).eq("id", worker_id).execute()
                                    break
                                except Exception as e:
                                    last_error = e
                                    cooldown_applied = (
                                        self.hardware_registry.record_failure(
                                            hardware_type, e
                                        )
                                    )

                                    # Spending-limit errors are account-wide —
                                    # stop trying other hardware types immediately.
                                    if is_spending_limit_error(e):
                                        logger.warning(
                                            "Spending limit hit while starting %s: %s. "
                                            "Pausing all provisioning.",
                                            hardware_type,
                                            e,
                                        )
                                        break

                                    cooldown_until = (
                                        self.hardware_registry.get_cooldown_info(
                                            hardware_type
                                        )
                                    )
                                    if cooldown_applied and cooldown_until is not None:
                                        logger.error(
                                            "Failed to start worker on %s; cooling down "
                                            "~%s remaining until %s (UTC), "
                                            "escalation_level=%s: %s",
                                            hardware_type,
                                            format_cooldown_remaining(cooldown_until),
                                            datetime.fromtimestamp(
                                                cooldown_until, timezone.utc
                                            ).isoformat(),
                                            self.hardware_registry.get_cooldown_escalation_level(
                                                hardware_type
                                            ),
                                            e,
                                        )
                                    else:
                                        logger.error(
                                            "Failed to start worker on %s: %s",
                                            hardware_type,
                                            e,
                                        )

                            if pod is None and worker_created:
                                self._ow._supabase.table("worker").update(
                                    {"status": "terminated"}
                                ).eq("id", worker_id).execute()
                                if last_error is not None:
                                    logger.error(
                                        "Exhausted hardware candidates for image %s, VRAM %s: %s",
                                        docker_image,
                                        max_vram_required,
                                        last_error,
                                    )
                        except Exception as e:
                            traceback.print_exc()
                            logger.error(
                                f"Failed to start worker for VRAM {max_vram_required} and image {docker_image}: {e}"
                            )
                            continue

    @supabase_retry()
    def set_shutdown_flags(self, idle_workers):
        for idle_worker in idle_workers:
            logger.info(f"Setting shutdown flag for idle worker: {idle_worker['id']}")
            try:
                # Save logs before marking for shutdown
                self.fetch_and_save_worker_logs(idle_worker)
                self._ow._supabase.table("worker").update({"status": "shutdown"}).eq(
                    "id", idle_worker["id"]
                ).execute()
            except Exception as e:
                logger.error(
                    f"Failed to set shutdown flag for worker {idle_worker['id']}: {e}"
                )

    def manage_cluster(self):
        """Main loop for managing the organization's cluster."""
        logger.info(f"Starting cluster management for organization {self.org_id}")

        global MAX_WORKERS

        while not self.shutdown_flag:
            worker_env = self.worker_env
            MAX_WORKERS = int(worker_env.get("MAX_WORKERS", MAX_WORKERS))
            runpod.api_key = worker_env["RUNPOD_API_KEY"]
            # try:
            # Get active workers and pending jobs
            running_workers = self.get_running_workers()
            pending_jobs = self.get_pending_jobs()

            # Log status
            status_counts: dict[str, int] = {}
            for w in running_workers:
                status_counts[w["status"]] = status_counts.get(w["status"], 0) + 1
            status_breakdown = (
                ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items()))
                or "none"
            )
            logger.info(
                f"[org={self._ow.org_name} ({self.org_id})] "
                f"workers: {len(running_workers)}/{MAX_WORKERS} ({status_breakdown}), "
                f"pending jobs: {len(pending_jobs)}"
            )
            # Scale workers if needed
            if pending_jobs:
                self.scale_workers(running_workers, pending_jobs)

            # Clean up unresponsive workers
            self.clean_up_unresponsive_workers(running_workers)

            # Handle idle workers
            active_and_starting_workers = [
                w for w in running_workers if w["status"] in ["active", "starting"]
            ]
            idle_workers = self.get_idle_workers(active_and_starting_workers)
            self.set_shutdown_flags(idle_workers)

            time.sleep(POLL_INTERVAL)

        logger.info(f"Shutting down cluster management for organization {self.org_id}")


API_TOKEN_RETRY_INTERVAL_S = 60


def main():
    """Run the org manager, surviving API-token rejections.

    If ``OPENWEIGHTS_API_KEY`` is rejected by the server (expired / revoked /
    invalid), we don't crash the process:

    * During construction (no manager yet), we retry building the
      ``OrganizationManager`` after a backoff.
    * After construction, ``manage_cluster`` absorbs the error in-place
      without rebuilding the manager.
    """
    org_id = os.environ.get("ORGANIZATION_ID", "<unknown>")
    while True:
        try:
            manager = OrganizationManager()
        except ApiTokenError as exc:
            logger.warning(
                "OPENWEIGHTS_API_KEY rejected for org %s during startup: %s. "
                "Rotate the token or clear api_tokens.expires_at. "
                "Retrying in %ds.",
                org_id,
                exc,
                API_TOKEN_RETRY_INTERVAL_S,
            )
            time.sleep(API_TOKEN_RETRY_INTERVAL_S)
            continue
        manager.manage_cluster()
        return


if __name__ == "__main__":
    main()
