"""
Organization-specific cluster manager.
"""

import io
import logging
import os
import random
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List

import requests
import runpod
from dotenv import load_dotenv

from openweights.client import OpenWeights
from openweights.cluster.start_runpod import HARDWARE_CONFIG, populate_hardware_config
from openweights.cluster.start_runpod import start_worker as runpod_start_worker

# Load environment variables
load_dotenv()

# Constants
POLL_INTERVAL = 15
IDLE_THRESHOLD = 300  # 5 minutes = 300 seconds
UNRESPONSIVE_THRESHOLD = 120  # 2 minutes = 120 seconds
MAX_WORKERS = int(os.getenv("MAX_WORKERS_PER_ORG", 8))
# MAX_WORKERS = 20

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def determine_gpu_type(required_vram, allowed_hardware=None, choice=None):
    """Determine the best GPU type and count for the required VRAM.

    Args:
        required_vram: Required VRAM in GB
        allowed_hardware: List of allowed hardware configurations (e.g. ['2x A100', '4x H100'])
        choice: Random seed for hardware selection

    Returns:
        Tuple of (gpu_type, count)
    """
    vram_options = sorted(HARDWARE_CONFIG.keys())

    # If allowed_hardware is specified, filter GPU options to only include allowed configurations
    if allowed_hardware:
        hardware_config = random.choice(allowed_hardware)
        count, gpu = hardware_config.split("x ")
        return gpu.strip(), int(count)

    # If no allowed_hardware specified, use the original logic
    for vram in vram_options:
        if required_vram <= vram:
            if choice is None:
                choice = random.choice(HARDWARE_CONFIG[vram])
            else:
                choice = HARDWARE_CONFIG[vram][choice % len(HARDWARE_CONFIG[vram])]
            count, gpu = choice.split("x ")
            return gpu.strip(), int(count)

    raise ValueError(
        f"No suitable GPU configuration found for VRAM requirement {required_vram}"
    )


class OrganizationManager:
    def __init__(self):
        self.openweights = OpenWeights()
        self.org_id = self.openweights.organization_id
        print("org name", self.openweights.org_name)
        self.supabase = self.openweights._supabase
        self.shutdown_flag = False

        # Set up RunPod client
        runpod.api_key = os.environ["RUNPOD_API_KEY"]
        populate_hardware_config(runpod)

        # Register signal handlers
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)

    @property
    def worker_env(self):
        secrets = self.get_secrets()
        runpod.api_key = secrets["RUNPOD_API_KEY"]
        global MAX_WORKERS
        MAX_WORKERS = secrets.get("MAX_WORKERS", MAX_WORKERS)
        return dict(
            SUPABASE_URL=os.environ["SUPABASE_URL"],
            SUPABASE_ANON_KEY=os.environ["SUPABASE_ANON_KEY"],
            **secrets,
        )

    def get_secrets(self) -> Dict[str, str]:
        """Get organization secrets from the database."""
        result = (
            self.supabase.table("organization_secrets")
            .select("name, value")
            .eq("organization_id", self.org_id)
            .execute()
        )

        return {secret["name"]: secret["value"] for secret in result.data}

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(
            f"Received shutdown signal, cleaning up organization {self.org_id}..."
        )
        self.shutdown_flag = True

    def get_running_workers(self):
        """Get all active and starting workers for this organization."""
        return (
            self.supabase.table("worker")
            .select("*")
            .eq("organization_id", self.org_id)
            .in_("status", ["active", "starting", "shutdown"])
            .execute()
            .data
        )

    def get_pending_jobs(self):
        """Get all pending jobs for this organization."""
        return (
            self.supabase.table("jobs")
            .select("*")
            .eq("organization_id", self.org_id)
            .eq("status", "pending")
            .order("requires_vram_gb", desc=True)
            .order("created_at", desc=False)
            .execute()
            .data
        )

    def get_idle_workers(self, running_workers):
        """Returns a list of idle workers."""
        idle_workers = []
        current_time = time.time()

        for worker in running_workers:
            # Skip if the worker is not a pod
            if not worker.get("pod_id"):
                continue
            # If the worker was started less than 5 minutes ago, skip it
            worker_created_at = datetime.fromisoformat(
                worker["created_at"].replace("Z", "+00:00")
            ).timestamp()
            if current_time - worker_created_at < IDLE_THRESHOLD:
                continue

            # Find the latest run associated with the worker
            runs = (
                self.supabase.table("runs")
                .select("*")
                .eq("worker_id", worker["id"])
                .is_("pod_id", "not.is", None)
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
            file_id = self.openweights.files.create(
                file=io.BytesIO(logs.encode("utf-8")), purpose="logs"
            )

            # Update worker record with logfile ID
            self.supabase.table("worker").update({"logfile": file_id}).eq(
                "id", worker["id"]
            ).execute()

            return file_id
        except Exception as e:
            logger.error(f"Error saving logs for worker {worker['id']}: {e}")
            return None

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
                # If status is 'starting', give it more time before calling it unresponsive
                threshold = (
                    UNRESPONSIVE_THRESHOLD * 3
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
                    self.supabase.table("runs")
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
                    self.supabase.table("runs").update({"status": "failed"}).eq(
                        "id", run["id"]
                    ).execute()

                    # Safely revert the job to 'pending' using your RPC that only updates
                    # if status='in_progress' for the same worker_id.
                    try:
                        self.supabase.rpc(
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
                self.supabase.table("worker").update({"status": "terminated"}).eq(
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

    def scale_workers(self, running_workers, pending_jobs):
        """Scale workers according to pending jobs and limits."""
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

        print(
            f"Running workers by image: {len(running_workers_by_image)} | Pending jobs by image: {len(pending_jobs_by_image)}"
        )

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
                        key=lambda job: job["requires_vram_gb"], reverse=True
                    )

                    # Split jobs for each worker
                    jobs_batches = [
                        hardware_jobs[i::group_num_to_start]
                        for i in range(group_num_to_start)
                    ]

                    for jobs_batch in jobs_batches:
                        max_vram_required = max(
                            job["requires_vram_gb"] for job in jobs_batch
                        )
                        try:
                            # Get allowed hardware from the first job in the batch
                            allowed_hardware = jobs_batch[0]["allowed_hardware"]

                            gpu, count = determine_gpu_type(
                                max_vram_required, allowed_hardware
                            )
                            hardware_type = f"{count}x {gpu}"

                            logger.info(
                                f"Starting a new worker - VRAM: {max_vram_required}, Hardware: {hardware_type}, Image: {docker_image}"
                            )

                            # Create worker in database with status 'starting'
                            worker_id = f"{self.org_id}-{uuid.uuid4().hex[:8]}"
                            worker_data = {
                                "status": "starting",
                                "ping": datetime.now(timezone.utc).isoformat(),
                                "vram_gb": 0,
                                "gpu_type": gpu,
                                "gpu_count": count,
                                "hardware_type": hardware_type,
                                "docker_image": docker_image,
                                "id": worker_id,
                                "organization_id": self.org_id,
                            }
                            self.supabase.table("worker").insert(worker_data).execute()

                            try:
                                # Start the worker
                                pod = runpod_start_worker(
                                    gpu=gpu,
                                    count=count,
                                    worker_id=worker_id,
                                    image=docker_image,
                                    env=self.worker_env,
                                    name=f"{self.openweights.org_name}-{time.time()}-ow-1day",
                                )
                                # Update worker with pod_id
                                assert pod is not None
                                self.supabase.table("worker").update(
                                    {"pod_id": pod["id"]}
                                ).eq("id", worker_id).execute()
                            except Exception as e:
                                # We mark all jobs as failed that have a single entry in allowed_hardware which corresponds to the current hardware and remove this hardware configuratoin from all jobs that have a list of allowed_hardware with more than one entry
                                # For each job in jobs_batch:
                                #   - If allowed_hardware has only one entry and it matches the current hardware, mark job as failed.
                                #   - If allowed_hardware has more than one entry, remove the current hardware from allowed_hardware and update the job.
                                for job in jobs_batch:
                                    allowed_hw = job.get("allowed_hardware", [])
                                    current_hw = hardware_type
                                    if isinstance(allowed_hw, str):
                                        # Defensive: convert to list if needed
                                        allowed_hw = [allowed_hw]
                                    if (
                                        len(allowed_hw) == 1
                                        and allowed_hw[0] == current_hw
                                    ):
                                        # Mark job as failed
                                        self.supabase.table("jobs").update(
                                            {
                                                "status": "failed",
                                                "outputs": {
                                                    "error": f"Error starting worker with GPU {current_hw}"
                                                    + str(e)
                                                },
                                            }
                                        ).eq("id", job["id"]).execute()
                                    elif (
                                        current_hw in allowed_hw and len(allowed_hw) > 1
                                    ):
                                        # Remove this hardware from allowed_hardware and update job
                                        new_allowed_hw = [
                                            hw for hw in allowed_hw if hw != current_hw
                                        ]
                                        self.supabase.table("jobs").update(
                                            {"allowed_hardware": new_allowed_hw}
                                        ).eq("id", job["id"]).execute()
                                logger.error(f"Failed to start worker: {e}")
                                # If worker creation fails, clean up the worker
                                self.supabase.table("worker").update(
                                    {"status": "terminated"}
                                ).eq("id", worker_id).execute()
                        except Exception as e:
                            logger.error(
                                f"Failed to start worker for VRAM {max_vram_required} and image {docker_image}: {e}"
                            )
                            continue

    def manage_cluster(self):
        """Main loop for managing the organization's cluster."""
        logger.info(f"Starting cluster management for organization {self.org_id}")

        while not self.shutdown_flag:
            # try:
            # Get active workers and pending jobs
            running_workers = self.get_running_workers()
            pending_jobs = self.get_pending_jobs()

            # Log status
            logger.info(
                f"Status: {len(running_workers)} active workers, {len(pending_jobs)} pending jobs"
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
            for idle_worker in idle_workers:
                logger.info(
                    f"Setting shutdown flag for idle worker: {idle_worker['id']}"
                )
                try:
                    # Save logs before marking for shutdown
                    self.fetch_and_save_worker_logs(idle_worker)
                    self.supabase.table("worker").update({"status": "shutdown"}).eq(
                        "id", idle_worker["id"]
                    ).execute()
                except Exception as e:
                    logger.error(
                        f"Failed to set shutdown flag for worker {idle_worker['id']}: {e}"
                    )

            # except Exception as e:
            #     logger.error(f"Error in management loop: {e}")

            time.sleep(POLL_INTERVAL)

        logger.info(f"Shutting down cluster management for organization {self.org_id}")


def main():
    manager = OrganizationManager()
    manager.manage_cluster()


if __name__ == "__main__":
    main()
