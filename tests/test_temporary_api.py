"""TemporaryApi.up() must not depend on OPENAI_API_KEY being set."""

from types import SimpleNamespace

import pytest

from openweights.client import temporary_api
from openweights.client.temporary_api import TemporaryApi


class _Table:
    def __init__(self, row):
        self._row = row

    def __getattr__(self, name):  # select/eq/single chain
        return lambda *a, **k: self

    def execute(self):
        return SimpleNamespace(data=self._row)


@pytest.mark.parametrize(
    "params_api_key,expected", [(None, "api_key"), ("secret", "secret")]
)
def test_up_uses_job_api_key_for_all_clients(monkeypatch, params_api_key, expected):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    params = {"model": "m", "max_num_seqs": 4}
    if params_api_key is not None:
        params["api_key"] = params_api_key
    job = {"status": "in_progress", "worker_id": "w", "params": params}
    ow = SimpleNamespace(
        jobs=SimpleNamespace(retrieve=lambda job_id: job),
        _supabase=SimpleNamespace(table=lambda name: _Table({"pod_id": "pod123"})),
    )
    constructed = []

    class FakeClient:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

    monkeypatch.setattr(temporary_api, "OpenAI", FakeClient)
    monkeypatch.setattr(temporary_api, "AsyncOpenAI", FakeClient)
    monkeypatch.setattr(TemporaryApi, "wait_until_ready", lambda self, c, m: None)
    monkeypatch.setattr(TemporaryApi, "_manage_timeout", lambda self: None)

    api = TemporaryApi(ow, "apijob-test")
    client = api.up()

    assert api.api_key == expected
    assert api.base_url == "https://pod123-8000.proxy.runpod.net/v1"
    assert client is api.sync_client
    assert constructed and all(c["api_key"] == expected for c in constructed)
