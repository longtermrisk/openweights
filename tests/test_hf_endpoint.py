"""model_exists must follow HF_ENDPOINT instead of a hard-coded huggingface.co."""

from unittest import mock

from openweights.client import utils


def _head_url(monkeypatch, endpoint):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    if endpoint is None:
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
    else:
        monkeypatch.setenv("HF_ENDPOINT", endpoint)
    utils.model_exists.cache_clear()
    with mock.patch.object(utils.requests, "head") as head:
        head.return_value.status_code = 200
        assert utils.model_exists("org/model") is True
    return head.call_args.args[0], head.call_args.kwargs["headers"]


def test_default_endpoint(monkeypatch):
    url, headers = _head_url(monkeypatch, None)
    assert url == "https://huggingface.co/api/models/org/model"
    assert headers == {"Authorization": "Bearer hf_test"}


def test_hf_endpoint_is_honoured(monkeypatch):
    url, _ = _head_url(monkeypatch, "http://broker.internal:14141/api/broker/hf/")
    assert url == "http://broker.internal:14141/api/broker/hf/api/models/org/model"
