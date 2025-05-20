from typing import Dict
import atexit
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
import signal
import jwt
import random

import runpod
import torch
from dotenv import load_dotenv
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions

from openweights.client import Files, OpenWeights, Run
from openweights.worker.gpu_health_check import GPUHealthCheck

# Load environment variables
load_dotenv()
openweights = OpenWeights()

# Set up logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

runpod.api_key = os.environ.get('RUNPOD_API_KEY')


def maybe_read(path):
    try:
        with open(path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


GPUs = {
    # References found at https://rest.runpod.io/v1/docs#v-0-106
    # GPUs for compute-intensive tasks (e.g. LoRAfinetuning)
    # "NVIDIA RTX 6000 Ada Generation": "6000Ada",  # Best compute-efficiency, to use by default
    # "NVIDIA RTX 4000 Ada Generation": "4000Ada",
    "NVIDIA RTX 6000 Ada Generation": "A6000",  # Best compute-efficiency, to use by default
    "NVIDIA RTX 4000 Ada Generation": "A4000",
    "NVIDIA L40": "L40",
    "NVIDIA L40S": "L40S",
    "NVIDIA A30": "A30",
    #
    # Belows, GPUs are only good for high-memory task (e.g., pretraining and vanilla finetuning)
    "NVIDIA A100 80GB PCIe": "A100",  # Default A100 - 80GB
    "NVIDIA A100 80GB PCIe": "A100_80",
    "NVIDIA A100-SXM4-40GB": "A100_40",
    "NVIDIA A100-SXM4-80GB": "A100S",
    "NVIDIA H100 PCIe": "H100",
    "NVIDIA H100 NVL": "H100N",
    "NVIDIA H100 80GB HBM3": "H100S",
    "NVIDIA H200": "H200",
    "NVIDIA B200": "B200",
    #
    # Below, GPUs are cost inefficient
    # "NVIDIA GeForce RTX 4080": "RTX4080",
    # "NVIDIA GeForce RTX 3090": "RTX3090",
    # "NVIDIA GeForce RTX 3090 Ti": "RTX3090Ti",
    # "Tesla V100-SXM2-32GB": "V100",  # Default V100 - 32GB
    # "Tesla V100-SXM2-32GB": "V100_32",
    # "Tesla V100-SXM2-16GB": "V100_16",
    # "Tesla V100-FHHL-16GB": "V100_16_FHHL",
    # "Tesla V100-PCIE-16GB": "V100_16_PCIE",
    # "NVIDIA GeForce RTX 4070 Ti": "RTX4070Ti",
    # "NVIDIA RTX 4000 SFF Ada Generation": "A4000_SFF",
    # "NVIDIA RTX 5000 Ada Generation": "A5000_ADA",
    # "AMD Instinct MI300X OAM": "MI300X",
    # "NVIDIA RTX 2000 Ada Generation": "2000Ada",
    # "NVIDIA RTX A6000": "A6000",
    # "NVIDIA RTX A4000": "A4000",
    # "NVIDIA RTX A2000": "A2000",
    # "NVIDIA GeForce RTX 4090": "RTX4090",
    # "NVIDIA RTX A5000": "A5000",
    # "NVIDIA A40": "A40",
    # "NVIDIA RTX A4500": "A4500",
    # "NVIDIA GeForce RTX 3080": "RTX3080",
    # "NVIDIA GeForce RTX 3070": "RTX3070",
    # "NVIDIA GeForce RTX 3080 Ti": "RTX3080Ti",
    # "NVIDIA L4": "L4",
}


class Worker:
    def __init__(self):
        logging.info("Initializing worker")
        self.supabase = openweights._supabase
        self.organization_id = openweights.organization_id
        self.auth_token = openweights.auth_token
        self.files = Files(self.supabase, self.organization_id)
        self.cached_models = []
        self.current_job = None
        self.current_process = None
        self.shutdown_flag = False
        self.current_run = None
        self.worker_id = os.environ.get('WORKER_ID', f'unmanaged-worker-{datetime.now().timestamp()}')
        self.docker_image = os.environ.get('DOCKER_IMAGE', 'dev')
        self.pod_id = None
        self.past_job_status = []
        self.hardware_type = None

        # Create a service account token for the worker if needed
        if not os.environ.get('OPENWEIGHTS_API_KEY'):
            result = self.supabase.rpc(
                'create_service_account_token',
                {
                    'org_id': self.organization_id,
                    'token_name': f'worker-{self.worker_id}',
                    'created_by': self.get_user_id_from_token()
                }
            ).execute()
            
            if not result.data:
                raise ValueError("Failed to create service account token")
            
            token = result.data[1]  # Get the JWT token
            os.environ['OPENWEIGHTS_API_KEY'] = token
            # Reinitialize supabase client with new token
            self.supabase = create_client(
                openweights.supabase_url,
                openweights.supabase_key,
                ClientOptions(
                    schema="public",
                    headers={"Authorization": f"Bearer {token}"},
                    auto_refresh_token=False,
                    persist_session=False
                )
            )

        # Detect GPU info
        try:
            logging.info("Detecting GPU info")
            self.gpu_count = torch.cuda.device_count()
            self.vram_gb = (
                torch.cuda.get_device_properties(0).total_memory // (1024**3)
            ) * self.gpu_count

            # Determine hardware type based on GPU info
            gpu_name = torch.cuda.get_device_name(0)
            self.hardware_type = None

            def clean_gpu_name(gpu_name):
                return gpu_name.lower().replace("nvidia ", "").strip()

            # Start with exact match
            for gpu_name_pattern, gpu_type in GPUs.items():
                if clean_gpu_name(gpu_name_pattern) == clean_gpu_name(gpu_name):
                    self.hardware_type = f"{self.gpu_count}x {gpu_type}"
                    break

            # If no exact match, use include match
            if self.hardware_type is None:
                for gpu_name_pattern, gpu_type in GPUs.items():
                    if clean_gpu_name(gpu_name_pattern) in clean_gpu_name(gpu_name):
                        self.hardware_type = f"{self.gpu_count}x {gpu_type}"
                        break

            if self.hardware_type is None:
                logging.info(f"GPU {gpu_name} not found in GPUs ({GPUs.keys()}).")
                raise ValueError(f"GPU {gpu_name} not found in GPUs ({GPUs.keys()}).")

                self.hardware_type = f"{self.gpu_count}x {gpu_type}"
                logging.info(f"Inferred hardware type: {self.hardware_type}")

        except:
            logging.warning("Failed to retrieve VRAM, registering with 0 VRAM")
            self.gpu_count = 0
            self.vram_gb = 0
            self.hardware_type = None
        
        # GPU health check
        if self.gpu_count > 0:
            is_healthy, errors, diagnostics = GPUHealthCheck.check_gpu_health()
            if not is_healthy:
                # Log complete diagnostic information
                logging.info(
                    f"GPU Health Check Results: {json.dumps(diagnostics, indent=2)}"
                )
                for error in errors:
                    logging.error(f"GPU health check failed: {error}")
                self.supabase.table("worker").update({"status": "shutdown"}).eq(
                    "id", self.worker_id
                ).execute()
                self.shutdown_flag = True

        # Register or read existing worker
        logging.debug(f"Registering worker {self.worker_id} with VRAM {self.vram_gb} GB and hardware {self.hardware_type}")
        data = self.supabase.table('worker').select('*').eq('id', self.worker_id).execute().data
        if data:
            self.pod_id = data[0].get('pod_id')

        if data and data[0]['status'] == 'shutdown':
            logging.info(f"Worker {self.worker_id} is already registered with status 'shutdown'. Exiting...")
            self.shutdown_flag = True
        else:
            self.supabase.table('worker').upsert({
                'id': self.worker_id,
                'status': 'active',
                'vram_gb': self.vram_gb,
                'ping': datetime.now(timezone.utc).isoformat(),
                'organization_id': self.organization_id,
                'hardware_type': self.hardware_type,
            }).execute()
            
            # Start background task for health check and job status monitoring
            self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
            self.health_check_thread.start()
        
        os.makedirs('logs', exist_ok=True)
        
        atexit.register(self.shutdown_handler)

    def get_user_id_from_token(self):
        """Extract user ID from JWT token."""
        if not self.auth_token:
            raise ValueError("No authentication token provided")
        
        try:
            payload = jwt.decode(self.auth_token, options={"verify_signature": False})
            return payload.get('sub')  # 'sub' is the user ID in Supabase JWTs
        except Exception as e:
            raise ValueError(f"Invalid authentication token: {str(e)}")

    def _health_check_loop(self):
        """Background task that updates worker ping and checks job status."""
        while not self.shutdown_flag:
            try:
                # Update ping timestamp
                result = self.supabase.table('worker').update({
                    'ping': datetime.now(timezone.utc).isoformat(),
                }).eq('id', self.worker_id).execute()

                # Check if worker status is 'shutdown'
                if result.data and result.data[0]['status'] == 'shutdown':
                    self.shutdown_flag = True

                # Check if current job is canceled or timed out
                if self.current_job:
                    job = self.supabase.table('jobs').select('status', 'timeout').eq('id', self.current_job['id']).single().execute().data
                    
                    should_cancel = False
                    if job['status'] == 'canceled' or self.shutdown_flag:
                        should_cancel = True
                        logging.info(f"Job {self.current_job['id']} was canceled, stopping execution")
                    elif job['timeout'] and datetime.fromisoformat(job['timeout'].replace('Z', '+00:00')) <= datetime.now(timezone.utc):
                        should_cancel = True
                        logging.info(f"Job {self.current_job['id']} has timed out, stopping execution")
                        # Update job status to canceled
                        self.supabase.table('jobs').update({'status': 'canceled'}).eq('id', self.current_job['id']).execute()
                    
                    if should_cancel:
                        # Wait for logs to propagate
                        logging.info("Waiting for logs to propagate...")
                        time.sleep(20)
                        logging.info("Waiting for logs to propagate more...")
                        time.sleep(20)
                        logging.info("Waiting for logs to propagate more...")
                        time.sleep(20)
                        logging.info("Waiting for logs to propagate done.")
                        if self.current_process:
                            try:
                                os.killpg(os.getpgid(self.current_process.pid), signal.SIGTERM)
                            except:
                                # Fallback: try harder to kill everything
                                subprocess.run(['pkill', '-f', 'vllm'], check=False)
                            self.current_process = None
                        if self.current_run:
                            self.current_run.update(status='canceled')
                        self.current_job = None

            except Exception as e:
                logging.error(f"Error in health check loop: {e}")

            time.sleep(5)  # Wait for 5 seconds before next check

        self.shutdown_handler()

    def shutdown_handler(self):
        """Clean up resources and update status on shutdown."""
        logging.info(f"Shutting down worker {self.worker_id}.")
        self.shutdown_flag = True
        
        # If we have a current job, revert it to 'pending' ONLY if it's still in_progress
        # (We do that by calling our same update_job_status_if_in_progress function.)
        if self.current_job:
            try:
                # Upload final logs if they exist
                log_file_path = os.path.join("logs", self.current_run.id)
                if os.path.exists(log_file_path):
                    with open(log_file_path, 'rb') as log_file:
                        log_response = self.files.create(log_file, purpose='log')
                        if self.current_run:
                            self.current_run.update(logfile=log_response['id'])
                
                # Update worker record with logfile ID
                with open("logs/main", 'rb') as log_file:
                    log_response = self.files.create(log_file, purpose='logs')
                    self.supabase.table('worker').update({
                        'logfile': log_response['id']
                    }).eq('id', self.worker_id).execute()

                # Mark job as 'pending' only if it is still 'in_progress' by this worker
                self.update_job_status_if_in_progress(
                    self.current_job['id'],
                    'pending',
                    None,  # job_outputs
                    None   # job_script
                )
                if self.current_run:
                    self.current_run.update(status='canceled')
            except Exception as e:
                logging.error(f"Error updating job status during shutdown: {e}")

        # Update worker status
        try:
            result = self.supabase.table('worker').update({'status': 'shutdown'}).eq('id', self.worker_id).execute()
            # If the worker has a pod_id, terminate the pod
            if result.data and result.data[0].get('pod_id'):
                runpod.terminate_pod(result.data[0]['pod_id'])
        except Exception as e:
            logging.error(f"Error updating worker status during shutdown: {e}")

    def find_and_execute_job(self):
        while not self.shutdown_flag:
            logging.info("Worker is looking for jobs...")
            job = None
            try:
                job = self._find_job()
            except Exception as e:
                logging.error(f"Error finding job: {e}")
                traceback.print_exc()
            if not job:
                logging.info(
                    "No suitable job found. Checking again in a few seconds..."
                )
                time.sleep(5)
                continue

            try:
                logging.info(f"Attempting to acquire job {job['id']}")
                acquired_job = self.acquire_job(job["id"])
                if not acquired_job:
                    # Another worker took it in the meantime
                    logging.info(f"Job {job['id']} was taken by another worker.")
                    time.sleep(2)
                    continue

                logging.info(
                    f"Worker {self.worker_id} acquired job {job['id']}, executing..."
                )
                self._execute_job(acquired_job)
            except KeyboardInterrupt:
                logging.info("Worker interrupted by user. Shutting down...")
                break
            except Exception as e:
                logging.error(f"Failed to execute job {job['id']}: {e}")
                traceback.print_exc()

            if self.shutdown_flag:
                break

    def _find_job(self):
        """Fetch pending jobs for this docker image, sorted by required VRAM desc, then oldest first."""
        logging.info("Fetching jobs from the database...")
        jobs = (
            self.supabase.table("jobs")
            .select("*")
            .eq("status", "pending")
            .eq("docker_image", self.docker_image)
            .order("requires_vram_gb", desc=True)
            .order("created_at", desc=False)
            .execute()
            .data
        )

        logging.info(f"Fetched {len(jobs)} pending jobs from the database")

        # Filter jobs by VRAM requirements
        logging.info(
            f"VRAM requirements per job: {[j['requires_vram_gb'] for j in jobs]} GB"
        )
        logging.info(f"Hardware type: {self.hardware_type}")
        logging.info(f"VRAM available: {self.vram_gb} GB")
        logging.info(
            f"Number of jobs existing before filtering them by VRAM: {len(jobs)}"
        )
        suitable_jobs = [j for j in jobs if j["requires_vram_gb"] <= self.vram_gb]
        logging.info(f"Found {len(suitable_jobs)} suitable jobs based on VRAM criteria")

        # Further filter jobs by hardware requirements
        if self.hardware_type:
            hardware_suitable_jobs = []
            for job in suitable_jobs:
                # If job doesn't specify allowed_hardware, it can run on any hardware
                if not job["allowed_hardware"]:
                    hardware_suitable_jobs.append(job)
                # If job specifies allowed_hardware, check if this worker's hardware is allowed
                elif self.hardware_type in job["allowed_hardware"]:
                    hardware_suitable_jobs.append(job)
                else:
                    logging.info(
                        f"""Job {job["id"]} is not suitable for this worker's hardware {self.hardware_type}. 
                        Allowed hardware: {job["allowed_hardware"]}"""
                    )

            suitable_jobs = hardware_suitable_jobs
            logging.info(
                f"Found {len(suitable_jobs)} suitable jobs after hardware filtering"
            )

        # Shuffle suitable jobs to get different workers to cache different models
        random.shuffle(suitable_jobs)

        if not suitable_jobs:
            return None
        
        # Prefer a job with a cached model
        for j in suitable_jobs:
            if j['model'] in self.cached_models:
                logging.debug(f"Selecting job {j['id']} with cached model {j['model']}")
                return j
        return suitable_jobs[0]

    def _execute_job(self, job):
        """Execute the job and update status in the database."""
        self.current_job = job
        self.current_run = Run(openweights, job['id'], self.worker_id)
        
        # Create a temporary directory for job execution
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.makedirs(f"{tmp_dir}/uploads", exist_ok=True)
            log_file_path = os.path.join("logs", str(self.current_run.id))

            # We already acquired the job via RPC, so it should be in 'in_progress' state for us now.

            outputs = None
            status = 'canceled'
            script = None  # We'll store the actual script string we ran.

            try:
                if 'mounted_files' in job['params']:
                    self._setup_custom_job_files(tmp_dir, job['params']['mounted_files'])
                script = job['script']

                with open(log_file_path, 'w') as log_file:
                    env = os.environ.copy()
                    env['OPENWEIGHTS_RUN_ID'] = str(self.current_run.id)
                    env['N_GPUS'] = str(self.gpu_count)

                    self.current_process = subprocess.Popen(
                        script, 
                        shell=True, 
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        cwd=tmp_dir, 
                        env=env,
                        preexec_fn=os.setsid,  # Allow us to send signals to the process group
                        bufsize=1,  # Line buffered
                        universal_newlines=True  # Text mode
                    )

                    # Stream logs to both file and stdout
                    for line in iter(self.current_process.stdout.readline, ''):
                        print(line.rstrip('\n'), flush=True)  # Immediate stdout flush
                        log_file.write(line)
                        log_file.flush()  # Force immediate write to file

                    self.current_process.wait()

                    if self.current_process is None:
                        logging.info(f"Job {job['id']} was canceled", extra={'run_id': self.current_run.id})
                    elif self.current_process.returncode == 0:
                        status = 'completed'
                        logging.info(f"Completed job {job['id']}", extra={'run_id': self.current_run.id})
                        # We now have the model cached
                        cached_model = '/'.join(job['model'].split('/')[:2])
                        if cached_model not in self.cached_models:
                            self.cached_models.append(cached_model)
                    else:
                        status = 'failed'
                        logging.error(f"Job {job['id']} failed with return code {self.current_process.returncode}", 
                                      extra={'run_id': self.current_run.id})
                        self.current_run.log({'error': f"Process exited with return code {self.current_process.returncode}"})
            except Exception as e:
                logging.error(f"Job {job['id']} failed: {e}", extra={'run_id': self.current_run.id})
                status = 'failed'
                self.current_run.log({'error': str(e)})
            finally:
                # Upload log file to Supabase
                with open(log_file_path, 'rb') as log_file:
                    log_response = self.files.create(log_file, purpose='log')
                
                # Then upload any files from /uploads as results
                upload_dir = os.path.join(tmp_dir, "uploads")
                for root, _, files in os.walk(upload_dir):
                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        try:
                            with open(file_path, 'rb') as file:
                                file_response = self.files.create(file, purpose='result')
                            # Log the uploaded file to the run
                            rel_path = os.path.relpath(file_path, upload_dir)
                            self.current_run.log({'path': rel_path, 'file': file_response['id']})
                        except Exception as e:
                            logging.error(f"Failed to upload file {file_name}: {e}")

                # Attempt to fetch the latest events for outputs
                outputs = openweights.events.latest('*', job_id=job['id'])

                # Use your RPC to update the job status only if it's still 'in_progress' for you
                self.update_job_status_if_in_progress(
                    job['id'],
                    status,
                    outputs,
                    script
                )
                # Also update the run object
                self.current_run.update(status=status, logfile=log_response['id'])

                self.past_job_status.append(status)
                
                # Clear current job/run/process
                self.current_job = None
                self.current_run = None
                self.current_process = None

                # If last 5 jobs have failed, shutdown the worker
                if len(self.past_job_status) >= 5 and all([s == 'failed' for s in self.past_job_status[-5:]]):
                    logging.info("Last 5 jobs have failed. Shutting down worker.")
                    self.shutdown_flag = True
                    self.shutdown_handler()

    def _setup_custom_job_files(self, tmp_dir: str, mounted_files: Dict[str, str]):
        """Download and set up mounted files for a custom job."""
        for target_path, file_id in mounted_files.items():
            full_path = os.path.join(tmp_dir, target_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            content = self.files.content(file_id)
            with open(full_path, 'wb') as f:
                f.write(content)

    def acquire_job(self, job_id: str):
        """
        Attempts to set a job from 'pending' to 'in_progress' for this worker 
        using the Postgres acquire_job() function. Returns the updated row if 
        successful, or None if not acquired.
        """
        result = self.supabase.rpc("acquire_job", {
            "_job_id": job_id,
            "_worker_id": self.worker_id
        }).execute()
        
        # acquire_job() returns SETOF jobs; might be empty if not acquired
        if not result.data:
            return None
        return result.data[0]

    def update_job_status_if_in_progress(self, job_id: str, new_status: str,
                                         job_outputs: dict = None,
                                         job_script: str = None):
        """
        Updates the job status ONLY if the job is still in 'in_progress' 
        and still assigned to this worker (atomic check inside the function).
        """
        self.supabase.rpc("update_job_status_if_in_progress", {
            "_job_id": job_id,
            "_new_status": new_status,
            "_worker_id": self.worker_id,
            "_job_outputs": job_outputs,
            "_job_script": job_script
        }).execute()


if __name__ == "__main__":
    worker = Worker()
    print("Worker initialized with ID:", worker.worker_id)
    worker.find_and_execute_job()