from types import SimpleNamespace

import pytest

from openweights.cluster.org_manager import OrganizationManager
from openweights.cluster.start_runpod import GPUs, normalize_hardware_type


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1x A100", "1x A100"),
        ("2x H100S", "2x H100S"),
        ("A100S", "1x A100S"),
        ("1x A100 80GB", "1x A100"),
        ("1x A100 SXM 80GB", "1x A100S"),
        ("1x NVIDIA A100-SXM4-80GB", "1x A100S"),
        ("4x NVIDIA H100 80GB HBM3", "4x H100S"),
        ("1x h100 nvl", "1x H100N"),
        ("8x 6000Ada", "8x 6000Ada"),
    ],
)
def test_normalize_hardware_type(raw, expected):
    assert normalize_hardware_type(raw) == expected


@pytest.mark.parametrize("raw", ["1x A100 40GB", "1x TPU", "0x A100", "", "1x "])
def test_normalize_hardware_type_rejects_unknown(raw):
    with pytest.raises(ValueError, match="Invalid hardware configuration"):
        normalize_hardware_type(raw)


def test_every_gpu_key_is_canonical():
    for key in GPUs:
        assert normalize_hardware_type(f"1x {key}") == f"1x {key}"


class FakeTable:
    def __init__(self, calls):
        self.calls = calls

    def update(self, data):
        self.calls.append(("update", data))
        return self

    def eq(self, column, value):
        self.calls.append(("eq", column, value))
        return self

    def execute(self):
        return SimpleNamespace(data=[])


def test_manager_normalizes_aliases_and_fails_unknown_hardware():
    calls = []
    manager = OrganizationManager.__new__(OrganizationManager)
    manager._ow = SimpleNamespace(
        _supabase=SimpleNamespace(table=lambda name: FakeTable(calls))
    )
    jobs = [
        {"id": "a", "allowed_hardware": ["1x A100 80GB", "1x A100 SXM 80GB"]},
        {"id": "b", "allowed_hardware": ["1x A100"]},
        {"id": "c", "allowed_hardware": None},
        {"id": "d", "allowed_hardware": ["1x TPU"]},
    ]

    valid = manager.normalize_allowed_hardware(jobs)

    assert [job["id"] for job in valid] == ["a", "b", "c"]
    assert valid[0]["allowed_hardware"] == ["1x A100", "1x A100S"]
    updates = [call[1] for call in calls if call[0] == "update"]
    assert updates[0] == {"allowed_hardware": ["1x A100", "1x A100S"]}
    assert updates[1]["status"] == "failed"
    assert "1x TPU" in updates[1]["outputs"]["error"]
    assert len(updates) == 2
