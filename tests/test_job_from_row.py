from openweights.client.jobs import Job


def test_job_from_row_ignores_unknown_columns():
    row = {
        "id": "custom-abc",
        "type": "custom",
        "status": "pending",
        "model": None,
        "requires_vram_gb": 24,
        "docker_image": "nielsrolf/ow-unsloth:v0.14.0",
        "script": "echo hi",
        "params": {},
        "outputs": None,
        "organization_id": "org",
        "created_at": "2026-10-12T00:00:00Z",
        "updated_at": "2026-10-12T00:00:00Z",
        "worker_id": None,
        "timeout": None,
        "submitted_by": "user",
        "api_token_id": None,
        "column_added_later": 1,
    }
    job = Job.from_row(row, manager=None)
    assert job.id == "custom-abc"
    assert job.submitted_by == "user"
    assert not hasattr(job, "column_added_later")
