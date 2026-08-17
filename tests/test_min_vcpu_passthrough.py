import pytest

from openweights.cluster import start_runpod


class FakeRunpodClient:
    """Records the kwargs create_pod was called with."""

    def __init__(self):
        self.create_calls = []

    def create_pod(self, *args, **kwargs):
        self.create_calls.append(kwargs)
        return {"id": "fake-pod-id"}

    def terminate_pod(self, pod_id):
        pass


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("USER", "tester")
    monkeypatch.setattr(start_runpod, "RUNPOD_MIN_VCPU_COUNT", None)
    monkeypatch.setattr(start_runpod, "RUNPOD_MIN_MEMORY_GB", None)
    return FakeRunpodClient()


def start(client, **overrides):
    params = dict(
        gpu="H200",
        image="nielsrolf/ow-unsloth:v0.11",
        count=4,
        dev_mode=False,
        env={},
        runpod_client=client,
    )
    params.update(overrides)
    start_runpod.start_worker(**params)
    assert client.create_calls, "create_pod was never called"
    return client.create_calls[0]


def test_min_vcpu_and_memory_are_forwarded_to_create_pod(client):
    kwargs = start(client, min_vcpu_count=96, min_memory_in_gb=500)

    assert kwargs["min_vcpu_count"] == 96
    assert kwargs["min_memory_in_gb"] == 500
    assert kwargs["gpu_count"] == 4


def test_fields_are_omitted_when_unset(client):
    kwargs = start(client)

    assert kwargs["min_vcpu_count"] is None
    assert kwargs["min_memory_in_gb"] is None


def test_environment_variables_provide_defaults(client, monkeypatch):
    monkeypatch.setattr(start_runpod, "RUNPOD_MIN_VCPU_COUNT", "64")
    monkeypatch.setattr(start_runpod, "RUNPOD_MIN_MEMORY_GB", "256")

    kwargs = start(client)

    assert kwargs["min_vcpu_count"] == 64
    assert kwargs["min_memory_in_gb"] == 256


def test_explicit_arguments_override_environment_defaults(client, monkeypatch):
    monkeypatch.setattr(start_runpod, "RUNPOD_MIN_VCPU_COUNT", "64")

    kwargs = start(client, min_vcpu_count=8)

    assert kwargs["min_vcpu_count"] == 8
