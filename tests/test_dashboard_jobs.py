"""Dashboard pagination, RLS and nonblocking request regression tests."""

import asyncio
import importlib
import json
import threading
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from test_cost_tracking import ORG, OTHER, ROOT, USER, db  # noqa: F401


@pytest.fixture
def dashboard(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "openweights/dashboard/backend"))
    main = importlib.import_module("main")
    database = Mock()
    main.app.dependency_overrides[main.get_db] = lambda: database
    yield main.app, database
    main.app.dependency_overrides.clear()


def test_page_endpoint_returns_only_summary_fields(dashboard):
    app, database = dashboard
    database.get_jobs_page.return_value = {
        "items": [
            {
                "id": "job-1",
                "type": "custom",
                "status": "pending",
                "created_at": "2026-09-25T00:00:00Z",
            }
        ],
        "total": 1234,
    }
    with TestClient(app) as client:
        response = client.get(
            "/organizations/org/jobs/page",
            params=[
                ("status", "pending"),
                ("status", "in_progress"),
                ("search", "a,b%"),
                ("limit", 5),
                ("offset", 15),
            ],
        )
    assert response.status_code == 200
    assert response.json()["total"] == 1234
    assert "script" not in response.json()["items"][0]
    database.get_jobs_page.assert_called_once_with(
        "org", ["pending", "in_progress"], "a,b%", 5, 15
    )


@pytest.mark.parametrize(
    "params", [{"limit": 0}, {"limit": 101}, {"offset": -1}, {"status": "invalid"}]
)
def test_invalid_page_rejected_before_database(dashboard, params):
    app, database = dashboard
    with TestClient(app) as client:
        assert (
            client.get("/organizations/org/jobs/page", params=params).status_code == 422
        )
    database.get_jobs_page.assert_not_called()


def test_page_access_denied(dashboard):
    app, database = dashboard
    database.get_jobs_page.side_effect = ValueError("No access to this organization")
    with TestClient(app) as client:
        assert client.get("/organizations/other/jobs/page").status_code == 403


@pytest.mark.asyncio
async def test_slow_logs_do_not_block_jobs(dashboard):
    app, database = dashboard
    started, release = threading.Event(), threading.Event()

    def slow_logs(*args):
        started.set()
        assert release.wait(5), "Concurrent request was blocked by log download"
        return "logs"

    database.get_log_file_content.side_effect = slow_logs
    database.get_jobs_page.return_value = {"items": [], "total": 0}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        task = asyncio.create_task(client.get("/organizations/org/runs/1/logs"))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            response = await asyncio.wait_for(
                client.get("/organizations/org/jobs/page"), 2
            )
            assert response.status_code == 200
            assert not release.is_set()
        finally:
            release.set()
            await task


@pytest.fixture
def jobs_db(db):
    db(
        (
            ROOT / "supabase/migrations/20260925000000_dashboard_jobs_pagination.sql"
        ).read_text()
    )
    db(f"""
        INSERT INTO jobs(id,type,status,organization_id,created_at,params,outputs,script)
        SELECT 'job-' || lpad(n::text,4,'0'), 'custom',
               CASE WHEN n % 2 = 0 THEN 'pending'::job_status ELSE 'completed'::job_status END,
               '{ORG}', '2026-09-25', '{{"needle": "param-match"}}', '{{"result": "output-match"}}', repeat('x',10000)
        FROM generate_series(1,1100) n;
        INSERT INTO jobs(id,type,status,organization_id) VALUES ('private', 'custom', 'pending', '{OTHER}');
    """)
    return db


def test_sql_pagination_search_counts_and_isolation(jobs_db):
    def page(args="", claims=None):
        return json.loads(
            jobs_db(
                f"SELECT get_dashboard_jobs('{ORG}' {args})",
                claims=claims or {"organization_id": ORG},
            )
        )

    first = page(", page_limit => 5")
    second = page(", page_limit => 5, page_offset => 5")
    assert first["total"] == 1100
    assert len(first["items"]) == 5
    assert first["items"][0]["id"] == "job-1100"
    assert not {j["id"] for j in first["items"]} & {j["id"] for j in second["items"]}
    assert set(first["items"][0]) == {
        "id",
        "type",
        "status",
        "model",
        "docker_image",
        "created_at",
    }
    assert page(", statuses => ARRAY['pending']::job_status[]")["total"] == 550
    assert page(", statuses => ARRAY[]::job_status[]") == {"items": [], "total": 0}
    for search in ["PARAM-MATCH", "output-match"]:
        assert page(f", search_text => '{search}'")["total"] == 1100
    assert (
        page(", search_text => '%'")["total"] == 0
    )  # Literal search, not SQL wildcards.
    assert page(", page_offset => 9999") == {"items": [], "total": 1100}
    assert page(claims={"sub": USER})["total"] == 1100
    assert page(claims={"organization_id": OTHER}) == {"items": [], "total": 0}
