import logging
import os
import time
from datetime import datetime
from multiprocessing import Process

import pytest

from openweights import OpenWeights
from openweights.worker.main import Worker
from supabase import create_client


# Set up logging configuration
def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


valid_sft_file = os.path.join(os.path.dirname(__file__), "sft_dataset.jsonl")
valid_pref_file = os.path.join(os.path.dirname(__file__), "preference_dataset.jsonl")


# Function to start worker process
def start_worker_process():
    setup_logging()
    worker = Worker()
    worker.find_and_execute_job()


@pytest.fixture(scope="module")
def client():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    openweights_api_key = os.getenv("OPENWEIGHTS_API_KEY")
    return OpenWeights(supabase_url, supabase_key, openweights_api_key)


@pytest.fixture(scope="module")
def worker():
    # Create a worker process
    worker_process = Process(target=start_worker_process)
    worker_process.start()
    yield
    worker_process.terminate()


def test_file_upload(client):
    # Test uploading a file and check database entries
    file_content = f"This is a test file.{datetime.now().timestamp()}".encode()
    with open("/tmp/test_file.txt", "wb") as file:
        file.write(file_content)

    with open("/tmp/test_file.txt", "rb") as file:
        try:
            logging.debug("Attempting to upload file")
            response = client.files.create(file, purpose="result")
            logging.debug(f"Upload response: {response}")
        except Exception as e:
            logging.error(f"File upload failed: {e}")
            raise

    assert response["object"] == "file"
    assert response["purpose"] == "result"
    assert response["bytes"] == len(file_content)

    # Validate file hash
    file_id = response["id"]

    # Retrieve and validate file content
    retrieved_content = client.files.content(file_id)
    assert retrieved_content == file_content


def test_file_validation(client):
    # Test file validation
    with open(valid_sft_file, "rb") as file:
        response = client.files.create(file, purpose="conversations")
    assert response["purpose"] == "conversations"

    # Attempt to validate as preference dataset
    with open(valid_sft_file, "rb") as file:
        with pytest.raises(Exception):
            client.files.create(file, purpose="preference")

    # Validate valid preference dataset
    with open(valid_pref_file, "rb") as file:
        response = client.files.create(file, purpose="preference")
    assert response["purpose"] == "preference"


def test_list_jobs(client):
    response = client.jobs.list(limit=5)
    assert isinstance(response, list)
    assert len(response) <= 5


def test_create_fine_tuning_job(client):
    # Use a real file for fine-tuning
    with open(valid_sft_file, "rb") as file:
        response = client.files.create(file, purpose="conversations")
    file_id = response["id"]

    params = {"training_file": file_id, "requires_vram_gb": 0, "loss": "sft"}
    response = client.fine_tuning.create(model="test-model", **params)
    assert response["type"] == "fine-tuning"
    assert response["status"] == "pending"


def test_create_inference_job(client):
    with open(valid_sft_file, "rb") as file:
        response = client.files.create(file, purpose="conversations")
    file_id = response["id"]

    params = {}
    response = client.inference.create(
        input_file_id=file_id, model="test-model", **params
    )
    assert response["type"] == "inference"
    assert response["status"] == "pending"


def test_cancel_job(client):
    with open(valid_pref_file, "rb") as file:
        response = client.files.create(file, purpose="preference")
    file_id = response["id"]

    params = {"training_file": file_id}
    job_response = client.fine_tuning.create(model="test-model", **params)
    job_id = job_response["id"]

    # Attempt to cancel the newly created job
    response = client.jobs.cancel(job_id)
    assert response["status"] == "canceled"


def test_job_cancellation(client):
    # Create a script that counts from 0 to 300 with 1s intervals
    script_content = f"""
    # {time.ctime()}
    for i in $(seq 0 300); do
        echo "Count: $i"
        sleep 1
    done
    """

    # Create the job
    job = client.jobs.create(script=script_content, requires_vram_gb=0)
    job_id = job["id"]
    print(job)

    # Wait for job to be in progress (poll every second for up to 30s)
    start_time = time.time()
    job_started = False
    while time.time() - start_time < 30:
        job = client.jobs.retrieve(job_id)
        if job["status"] == "in_progress":
            job_started = True
            break
        time.sleep(1)
    print(job)
    assert job_started, "Job did not start within 30 seconds"

    # Cancel the job
    client.jobs.cancel(job_id)

    # Wait for job to be canceled (poll every second for up to 60s)
    start_time = time.time()
    while time.time() - start_time < 60:
        runs = client.runs.list(job_id=job_id)
        assert len(runs) == 1, "Expected exactly one run for the job"
        run = runs[0]
        if run["status"] == "canceled":
            run_canceled = True
            break
        time.sleep(1)
    print(job)
    assert run_canceled, "Run was not canceled within 60 seconds"

    # Wait short time to upload logs
    time.sleep(3)
    run = client.runs.list(job_id=job_id)[0]

    # Check the run logs
    assert run["log_file"] is not None, "Run should have a log file"

    # Verify logs are not empty and contain some count output
    log_content = client.files.content(run["log_file"]).decode("utf-8")
    assert len(log_content) > 0, "Log file should not be empty"
    assert "Count:" in log_content, "Log should contain count output"


def test_list_runs(client):
    with open(valid_sft_file, "rb") as file:
        response = client.files.create(file, purpose="conversations")
    file_id = response["id"]

    job = client.jobs.create(script=f"{time.ctime()}\ndate", requires_vram_gb=0)
    print(job)
    job_id = job["id"]

    # Allow some time for the worker to pick up the job
    time.sleep(10)

    # Retrieve runs for the job
    runs = client.runs.list(job_id=job_id)
    print(runs)

    assert isinstance(runs, list)
    assert len(runs) > 0
    run = runs[-1]
    assert run["job_id"] == job_id
    # Retrieve and check logs
    log_content = client.files.content(run["log_file"])
    assert len(log_content) > 0


def test_script_job_execution(client):
    # Create a script job with a simple echo command
    script_content = "echo hello world!"
    job = client.jobs.create(script=script_content, requires_vram_gb=0)
    job_id = job["id"]

    # Allow some time for the worker to pick up and execute the job
    time.sleep(10)
    # Retrieve the runs for the job
    runs = client.runs.list(job_id=job_id)

    run = runs[-1]

    # Check the logfile for the expected output
    log_content = client.files.content(run["log_file"]).decode()
    assert "hello world" in log_content
