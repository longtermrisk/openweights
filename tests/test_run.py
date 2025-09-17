import os
import time
from datetime import datetime

import pytest

from openweights.client import Run
from supabase import create_client


@pytest.fixture(scope="module")
def supabase():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    return create_client(supabase_url, supabase_key)


def test_create_run_with_new_job(supabase):
    # Create a new run without a job_id (should create a script job)
    run = Run(supabase)

    assert run.id is not None
    assert run.job_id is not None
    assert run.status == "in_progress"

    # Verify the job was created
    job = (
        supabase.table("jobs").select("*").eq("id", run.job_id).single().execute().data
    )
    assert job is not None
    assert job["type"] == "script"
    assert job["status"] == "in_progress"


def test_create_run_with_existing_job(supabase):
    # First create a job
    job_id = f"testjob-{datetime.now().timestamp()}"
    job_data = {
        "id": job_id,
        "type": "script",
        "script": "test script",
        "status": "pending",
    }
    supabase.table("jobs").insert(job_data).execute()

    # Create a run for this job
    run = Run(supabase, job_id=job_id)

    assert run.id is not None
    assert run.job_id == job_id
    assert run.status == "in_progress"


def test_get_existing_run(supabase):
    # First create a run
    original_run = Run(supabase)
    run_id = original_run.id

    # Get the run using the static method
    retrieved_run = Run.get(supabase, run_id)

    assert retrieved_run.id == original_run.id
    assert retrieved_run.job_id == original_run.job_id
    assert retrieved_run.status == original_run.status


def test_update_run(supabase):
    run = Run(supabase)

    # Update status
    run.update(status="completed")
    assert run.status == "completed"

    # Verify in database
    db_run = supabase.table("runs").select("*").eq("id", run.id).single().execute().data
    assert db_run["status"] == "completed"

    # Update logfile
    run.update(logfile="testlog-123")
    assert run.log_file == "testlog-123"

    # Verify in database
    db_run = supabase.table("runs").select("*").eq("id", run.id).single().execute().data
    assert db_run["log_file"] == "testlog-123"


def test_log_events(supabase):
    run = Run(supabase)

    # Log some events
    test_events = [
        {"loss": 0.5, "step": 1},
        {"loss": 0.3, "step": 2},
        {"loss": 0.1, "step": 3},
    ]

    for event in test_events:
        run.log(event)

    # Verify events in database
    events = supabase.table("events").select("*").eq("run_id", run.id).execute().data
    assert len(events) == len(test_events)

    # Verify event data
    event_data = [event["data"] for event in events]
    for test_event in test_events:
        assert test_event in event_data


def test_run_with_environment_variable(supabase):
    # First create a run to get its ID
    original_run = Run(supabase)
    run_id = original_run.id

    # Set environment variable
    os.environ["OPENWEIGHTS_RUN_ID"] = str(run_id)

    try:
        # Create new run instance - should use existing run
        run = Run(supabase)

        assert run.id == run_id
        assert run.job_id == original_run.job_id
        assert run.status == original_run.status
    finally:
        # Clean up environment
        del os.environ["OPENWEIGHTS_RUN_ID"]


def test_run_with_mismatched_job_id(supabase):
    # Create a run
    original_run = Run(supabase)
    run_id = original_run.id

    # Set environment variable
    os.environ["OPENWEIGHTS_RUN_ID"] = str(run_id)

    try:
        # Try to create new run with different job_id
        with pytest.raises(ValueError) as exc_info:
            Run(supabase, job_id="different-job-id")

        assert "Run" in str(exc_info.value)
        assert "associated with job" in str(exc_info.value)
    finally:
        # Clean up environment
        del os.environ["OPENWEIGHTS_RUN_ID"]


def test_get_nonexistent_run(supabase):
    with pytest.raises(ValueError) as exc_info:
        Run.get(supabase, 999999)

    assert "Run with ID" in str(exc_info.value)
    assert "not found" in str(exc_info.value)
