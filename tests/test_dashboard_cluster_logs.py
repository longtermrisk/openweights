"""Cluster tab: org manager log endpoint."""

import importlib

import pytest
from fastapi.testclient import TestClient

from test_dashboard_jobs import dashboard  # noqa: F401

ORG_ID = "1ef0e036-5a2b-428c-8673-c135746d3655"


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OW_ORG_MANAGER_LOG_DIR", str(tmp_path))
    return tmp_path


def test_returns_tail_without_prefixes_and_with_secrets_redacted(dashboard, log_dir):
    app, database = dashboard
    database.verify_organization_access.return_value = True
    (log_dir / f"org_{ORG_ID}_stderr.log.1").write_text(
        "2026-10-06 09:00:00,000 - org_manager.py      :629  2026-10-06 09:00:00,000 old\n"
    )
    (log_dir / f"org_{ORG_ID}_stderr.log").write_text(
        "2026-10-06 10:00:00,000 - org_manager.py      :629  2026-10-06 10:00:00,000 "
        "Failed to start worker on 1x A100 80GB\n"
        "2026-10-06 10:00:01,000 - Traceback key=rpa_ABCDEFGHIJKLMNOPQRSTUV\n"
    )

    response = TestClient(app).get(
        f"/organizations/{ORG_ID}/cluster/logs", params={"lines": 2}
    )

    assert response.status_code == 200
    assert response.text.splitlines() == [
        "2026-10-06 10:00:00,000 Failed to start worker on 1x A100 80GB",
        "2026-10-06 10:00:01,000 - Traceback key=[redacted]",
    ]
    database.verify_organization_access.assert_called_once_with(ORG_ID)


def test_reads_rotated_file_first(dashboard, log_dir):
    app, database = dashboard
    database.verify_organization_access.return_value = True
    (log_dir / f"org_{ORG_ID}_stderr.log.1").write_text("old\n")
    (log_dir / f"org_{ORG_ID}_stderr.log").write_text("new\n")

    response = TestClient(app).get(f"/organizations/{ORG_ID}/cluster/logs")

    assert response.text == "old\nnew"


def test_missing_log_is_empty(dashboard, log_dir):
    app, database = dashboard
    database.verify_organization_access.return_value = True
    response = TestClient(app).get(f"/organizations/{ORG_ID}/cluster/logs")
    assert response.status_code == 200
    assert response.text == ""


def test_requires_org_access(dashboard, log_dir):
    app, database = dashboard
    database.verify_organization_access.return_value = False
    (log_dir / f"org_{ORG_ID}_stderr.log").write_text("secret org log\n")
    response = TestClient(app).get(f"/organizations/{ORG_ID}/cluster/logs")
    assert response.status_code == 403


def test_rejects_non_uuid_org_ids(dashboard, log_dir):
    cluster_logs = importlib.import_module("cluster_logs")
    with pytest.raises(ValueError):
        cluster_logs.read_cluster_log("../../etc", 10)
