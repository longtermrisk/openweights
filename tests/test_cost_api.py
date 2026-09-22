"""Dashboard cost endpoint validation, authorization and SDK contract tests."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from postgrest.exceptions import APIError


@pytest.fixture
def client(monkeypatch):
    backend = Path(__file__).resolve().parents[1] / "openweights/dashboard/backend"
    monkeypatch.syspath_prepend(str(backend))
    main = importlib.import_module("main")
    db = SimpleNamespace(client=Mock())
    main.app.dependency_overrides[main.get_db] = lambda: db
    with TestClient(main.app) as http:
        yield http, db.client
    main.app.dependency_overrides.clear()


def test_report_is_returned_from_scoped_rpc(client):
    http, db = client
    db.rpc.return_value.execute.return_value.data = {"total_usd": 1.5}
    response = http.get("/organizations/org/costs")
    assert response.json() == {"total_usd": 1.5}
    db.rpc.assert_called_once_with(
        "get_cost_report", {"org_id": "org", "row_limit": 100, "row_offset": 0}
    )


@pytest.mark.parametrize("amount", [-1, "NaN", "Infinity", "not-a-number"])
def test_invalid_amount_never_reaches_database(client, amount):
    http, db = client
    assert (
        http.put(
            "/organizations/org/costs/limits/key", json={"amount_usd": amount}
        ).status_code
        == 422
    )
    db.rpc.assert_not_called()


def test_clear_limit(client):
    http, db = client
    assert (
        http.put(
            "/organizations/org/costs/limits/key", json={"amount_usd": None}
        ).status_code
        == 200
    )
    db.rpc.assert_called_once_with(
        "set_spending_limit", {"org_id": "org", "key_id": "key", "amount_usd": None}
    )


def test_database_permission_error_stays_forbidden(client):
    http, db = client
    db.rpc.return_value.execute.side_effect = APIError(
        {
            "message": "Organization access denied",
            "code": "42501",
            "details": None,
            "hint": None,
        }
    )
    assert http.get("/organizations/org/costs").status_code == 403
    assert (
        http.put(
            "/organizations/org/costs/limits/key", json={"amount_usd": 1}
        ).status_code
        == 403
    )


def test_dashboard_bootstrap_uses_public_runtime_configuration(client, monkeypatch):
    import json

    main = importlib.import_module("main")
    monkeypatch.setattr(main, "_SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setattr(main, "_SUPABASE_ANON_KEY", "public-anon-key")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "must-never-reach-browser")
    http, db = client
    response = http.get("/config.js")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-type"].startswith("application/javascript")
    config = json.loads(
        response.text.removeprefix("window.__OPENWEIGHTS_CONFIG__ = ").removesuffix(";")
    )
    assert config == {
        "supabaseUrl": "https://example.supabase.co",
        "supabaseAnonKey": "public-anon-key",
    }
    assert "must-never-reach-browser" not in response.text
    db.rpc.assert_not_called()


@pytest.mark.parametrize("amount", [-1, "NaN", "Infinity", "invalid"])
def test_invalid_initial_token_limit(client, amount):
    http, db = client
    response = http.post(
        "/organizations/org/tokens", json={"name": "test", "spending_limit_usd": amount}
    )
    assert response.status_code == 422
    db.rpc.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("amount", [None, "0", "12.34"])
async def test_database_token_creation_uses_atomic_budget(client, amount):
    from database import Database
    from models import TokenCreate

    db = Database.__new__(Database)
    db.client = Mock()
    db.set_organization_id = Mock()
    db.client.rpc.return_value.execute.return_value.data = [
        {"token_id": "key", "token": "ow_test"}
    ]
    result = await db.create_token(
        "org", TokenCreate(name="test", spending_limit_usd=amount)
    )
    assert result.access_token == "ow_test"
    params = {"org_id": "org", "token_name": "test", "expires_at": None}
    if amount is not None:
        params["spending_limit_usd"] = amount
    db.client.rpc.assert_called_once_with(
        "create_api_token_with_limit" if amount is not None else "create_api_token",
        params,
    )
