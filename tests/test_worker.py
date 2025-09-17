import os
import time
from datetime import datetime
from multiprocessing import Process

import pytest

from openweights.client import OpenWeights
from openweights.worker.main import Worker


@pytest.fixture(scope="module")
def client():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    return OpenWeights(supabase_url, supabase_key)


@pytest.fixture(scope="module")
def setup_worker(client):
    worker_process = Process(target=start_worker)
    worker_process.start()
    yield
    worker_process.terminate()


def start_worker():
    worker = Worker(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    worker.find_and_execute_job()


def test_worker_executes_job_with_zero_vram(client, setup_worker):
    # Insert a job with 0 VRAM requirements
    job = client.jobs.create(**{"requires_vram_gb": 0, "script": "date"})
    job_id = job["id"]

    # Allow some time for the worker to pick up the job
    start_time = time.time()
    timeout = 60  # seconds
    job_executed = False

    while time.time() - start_time < timeout:
        # Check the job status
        job = client.jobs.retrieve(job_id)

        if job["status"] == "completed":
            job_executed = True
            break

        time.sleep(5)

    assert job_executed, "Worker did not complete the job in time"
