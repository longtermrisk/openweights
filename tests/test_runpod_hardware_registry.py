from openweights.cluster.start_runpod import RunpodHardwareRegistry

import pytest


class FakeTime:
    def __init__(self, initial=0):
        self.current = initial

    def now(self):
        return self.current

    def advance(self, seconds):
        self.current += seconds


class FakeRunpodClient:
    def __init__(self, gpus):
        self._gpus = gpus
        self.calls = 0

    def get_gpus(self):
        self.calls += 1
        return list(self._gpus)


def test_refresh_populates_hardware_config_from_runpod_inventory():
    fake_time = FakeTime()
    registry = RunpodHardwareRegistry(now_fn=fake_time.now)
    client = FakeRunpodClient(
        [
            {"id": "NVIDIA L40", "memoryInGb": 48},
            {"id": "NVIDIA A100 80GB PCIe", "memoryInGb": 80},
            {"id": "NVIDIA GeForce RTX 4090", "memoryInGb": 24},
        ]
    )

    hardware_config = registry.refresh(client, force=True)

    assert hardware_config == {
        43: ["1x L40"],
        75: ["1x A100"],
    }
    assert client.calls == 1


def test_registry_cools_down_gpu_after_repeated_failures_and_readds_after_expiry():
    fake_time = FakeTime()
    registry = RunpodHardwareRegistry(
        failure_threshold=2,
        cooldown_ladder_seconds=(10,),
        now_fn=fake_time.now,
    )
    client = FakeRunpodClient([{"id": "NVIDIA L40", "memoryInGb": 48}])

    registry.refresh(client, force=True)
    assert registry.get_candidate_hardware(24) == ["1x L40"]

    assert registry.record_failure("1x L40", "first failure") is False
    assert registry.get_candidate_hardware(24) == ["1x L40"]

    assert registry.record_failure("1x L40", "second failure") is True
    assert registry.get_candidate_hardware(24) == []

    fake_time.advance(11)
    assert registry.get_candidate_hardware(24) == ["1x L40"]


def test_allowed_hardware_respects_cooldowns_without_mutating_job_preferences():
    fake_time = FakeTime()
    registry = RunpodHardwareRegistry(
        failure_threshold=1,
        cooldown_ladder_seconds=(10,),
        now_fn=fake_time.now,
    )
    client = FakeRunpodClient(
        [
            {"id": "NVIDIA L40", "memoryInGb": 48},
            {"id": "NVIDIA A100 80GB PCIe", "memoryInGb": 80},
        ]
    )
    allowed_hardware = ["1x L40", "1x A100"]

    registry.refresh(client, force=True)
    assert registry.get_candidate_hardware(24, allowed_hardware=allowed_hardware) == [
        "1x L40",
        "1x A100",
    ]

    registry.record_failure("1x L40", "capacity issue")
    assert registry.get_cooldown_escalation_level("1x L40") == 1

    assert registry.get_candidate_hardware(24, allowed_hardware=allowed_hardware) == [
        "1x A100"
    ]
    assert allowed_hardware == ["1x L40", "1x A100"]


def test_cooldown_duration_escalates_per_provisioning_failure_wave() -> None:
    """Later cooldown waves use later ladder rungs until the final rung repeats."""
    fake_time = FakeTime()
    registry = RunpodHardwareRegistry(
        failure_threshold=1,
        cooldown_ladder_seconds=(100, 200, 400),
        now_fn=fake_time.now,
    )

    assert registry.record_failure("1x L40", "wave1") is True
    assert registry.get_cooldown_escalation_level("1x L40") == 1
    until_first = registry.get_cooldown_info("1x L40")
    assert until_first is not None
    assert until_first == pytest.approx(100.0)

    fake_time.advance(101)
    assert registry.get_cooldown_info("1x L40") is None

    assert registry.record_failure("1x L40", "wave2") is True
    assert registry.get_cooldown_escalation_level("1x L40") == 2
    until_second = registry.get_cooldown_info("1x L40")
    assert until_second is not None
    assert until_second == pytest.approx(301.0)

    fake_time.advance(201)
    assert registry.get_cooldown_info("1x L40") is None
    assert registry.record_failure("1x L40", "wave3") is True
    assert registry.get_cooldown_escalation_level("1x L40") == 3
    until_third = registry.get_cooldown_info("1x L40")
    assert until_third is not None
    assert until_third == pytest.approx(702.0)

    fake_time.advance(401)
    assert registry.get_cooldown_info("1x L40") is None
    assert registry.record_failure("1x L40", "wave4") is True
    assert registry.get_cooldown_escalation_level("1x L40") == 4
    until_fourth = registry.get_cooldown_info("1x L40")
    assert until_fourth is not None
    assert until_fourth == pytest.approx(1103.0)
