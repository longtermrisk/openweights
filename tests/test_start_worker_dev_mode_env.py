import pytest

from openweights.cluster import start_runpod


class FakeRunpodClient:
    """Captures what would have been sent to RunPod."""

    def __init__(self):
        self.create_kwargs = None

    def create_pod(self, *args, **kwargs):
        self.create_kwargs = kwargs
        return {"id": "pod-abc123"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        start_runpod, "get_ip_and_port", lambda pod_id, client: ("10.0.0.1", 2222)
    )
    return FakeRunpodClient()


def _start(client, env):
    start_runpod.start_worker(
        gpu="A100",
        image="some-image",
        count=1,
        name="test-worker",
        dev_mode=True,
        env=env,
        runpod_client=client,
    )
    return client.create_kwargs["env"]


def test_dev_mode_forwards_caller_env_to_the_pod(client, monkeypatch):
    monkeypatch.setenv("OPENWEIGHTS_API_KEY", "ow-key")

    env = _start(client, {"WANDB_PROJECT": "my-project", "MAX_JOBS": "16"})

    # Vars outside the credential list used to be dropped on the floor.
    assert env["WANDB_PROJECT"] == "my-project"
    assert env["MAX_JOBS"] == "16"
    # ...while the credentials from the caller's shell still get forwarded.
    assert env["OPENWEIGHTS_API_KEY"] == "ow-key"


def test_dev_mode_caller_env_beats_the_shell(client, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from-shell")

    env = _start(client, {"HF_TOKEN": "from-env-file"})

    assert env["HF_TOKEN"] == "from-env-file"


def test_dev_mode_still_sets_operational_vars(client, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")

    env = _start(client, {"WANDB_PROJECT": "my-project"})

    assert env["OW_DEV"] == "true"
    assert env["TTL_HOURS"] == "24"
