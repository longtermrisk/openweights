import pytest

from openweights.cli import common, ssh as ssh_cli
from openweights.cli.common import RemoteBootstrapError


class FakeStartResult:
    def __init__(self, pod_id="pod-xyz"):
        self.provider_meta = {"pod_id": pod_id} if pod_id else {}
        self.terminated = 0

    def terminate(self):
        self.terminated += 1


def test_machine_we_created_gets_terminated():
    start_res = FakeStartResult("pod-xyz")

    ssh_cli._terminate_after_failure(start_res, "boom")

    assert start_res.terminated == 1


def test_existing_machine_is_left_alone():
    """--existing pods belong to someone else; a failure must not kill them."""
    start_res = FakeStartResult(pod_id=None)

    ssh_cli._terminate_after_failure(start_res, "boom")

    assert start_res.terminated == 0


def test_failing_terminate_still_names_the_pod(capsys):
    class Exploding(FakeStartResult):
        def terminate(self):
            raise RuntimeError("runpod unreachable")

    ssh_cli._terminate_after_failure(Exploding("pod-xyz"), "boom")

    err = capsys.readouterr().err
    assert "could not terminate pod-xyz" in err
    assert "billing" in err


def test_bootstrap_failure_raises_instead_of_exiting(monkeypatch):
    """It used to sys.exit, which skipped the caller's terminate and leaked the pod."""
    monkeypatch.setattr(common, "scp_text", lambda *a, **k: None)
    monkeypatch.setattr(common, "ssh_exec", lambda *a, **k: 3)

    with pytest.raises(RemoteBootstrapError) as excinfo:
        common.bootstrap_remote(
            ssh=None, remote_cwd="/workspace", do_editable_install=False
        )

    assert excinfo.value.returncode == 3
