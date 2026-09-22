"""Creation-time budgets validate before contacting the server and use one RPC."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openweights.cli import token


@pytest.mark.parametrize("amount", ["-1", "NaN", "Infinity", "-Infinity", "invalid"])
def test_cli_invalid_limit_creates_nothing(monkeypatch, amount):
    factory = Mock()
    monkeypatch.setattr(token, "get_openweights_client", factory)
    assert token.handle_token_create(SimpleNamespace(spending_limit_usd=amount)) == 1
    factory.assert_not_called()


@pytest.mark.parametrize("amount", [None, "0", "12.34"])
def test_cli_creates_token_with_atomic_budget(monkeypatch, amount, capsys):
    db = Mock()
    db.rpc.return_value.execute.return_value.data = [
        {"token_id": "key", "token": "ow_test"}
    ]
    monkeypatch.setattr(
        token,
        "get_openweights_client",
        lambda: SimpleNamespace(organization_id="org", _supabase=db),
    )
    args = SimpleNamespace(name="test", expires_in_days=None, spending_limit_usd=amount)
    assert token.handle_token_create(args) == 0
    params = {"org_id": "org", "token_name": "test", "expires_at": None}
    if amount is not None:
        params["spending_limit_usd"] = amount
    db.rpc.assert_called_once_with(
        "create_api_token_with_limit" if amount is not None else "create_api_token",
        params,
    )
