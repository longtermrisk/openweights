"""The manager pauses provisioning while the RunPod account is out of credits.

No RunPod account and no database: the balance and the pod starts are fakes.
"""

import logging

import pytest

from openweights.cluster.runpod_credits import (
    RunpodCreditMonitor,
    hourly_cost_usd,
    is_out_of_credits_error,
)
from test_credential_safety import pending_job, scaling_manager
from test_runpod_hardware_registry import FakeTime

NO_INSTANCES = (
    "There are no longer any instances available with the requested specifications."
)


class Account:
    """A RunPod account whose balance the test sets."""

    def __init__(self, balance):
        self.balance = balance
        self.reads = 0

    def fetch(self):
        self.reads += 1
        if isinstance(self.balance, Exception):
            raise self.balance
        return self.balance


def make_monitor(balance, **kwargs):
    clock = FakeTime(1000)
    account = Account(balance)
    monitor = RunpodCreditMonitor(
        account.fetch,
        min_balance_usd=5,
        check_interval_seconds=60,
        paused_check_interval_seconds=0,
        probe_interval_seconds=300,
        now_fn=clock.now,
        **kwargs,
    )
    return monitor, account, clock


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Insufficient balance to deploy this pod", True),
        ("Your balance is too low. Please add funds.", True),
        ("You do not have enough credits: not enough credit", True),
        (NO_INSTANCES, False),
        ("Something went wrong. Please try again later", False),
        # A spending limit is not a balance; it has its own pause.
        ("You have exceeded your spending limit", False),
    ],
)
def test_out_of_credits_error_patterns(message, expected):
    assert is_out_of_credits_error(message) is expected


def test_hourly_cost_of_unknown_hardware_is_zero_not_a_reason_to_pause():
    assert hourly_cost_usd("2x A100") == pytest.approx(2 * hourly_cost_usd("1x A100"))
    assert hourly_cost_usd("1x NoSuchGpu") == 0.0
    assert hourly_cost_usd("nonsense") == 0.0


def test_low_balance_pauses_before_any_pod_start_and_top_up_resumes_next_cycle(
    caplog,
):
    monitor, account, clock = make_monitor(200.0)
    assert monitor.can_provision()

    # The balance is not read every cycle while all is well...
    clock.advance(15)
    assert monitor.can_provision() and account.reads == 1

    # ...so a balance that ran out is noticed within the check interval.
    account.balance = 0.4
    clock.advance(60)
    with caplog.at_level(logging.ERROR):
        assert not monitor.can_provision()
    assert monitor.paused and "out of credits" in caplog.text

    # While paused it is read every cycle.
    reads = account.reads
    for _ in range(3):
        clock.advance(15)
        assert not monitor.can_provision()
    assert account.reads == reads + 3

    # Topped up: the very next cycle provisions again.
    account.balance = 150.0
    clock.advance(15)
    with caplog.at_level(logging.INFO):
        assert monitor.can_provision()
    assert not monitor.paused and "credits are back" in caplog.text


def test_refusal_with_a_balance_below_one_hour_of_the_pod_is_a_credit_failure():
    # RunPod wants one hour of the pod on the account. $6 is above the minimum
    # here but does not pay for 8 A100s, whatever the wording of the refusal.
    monitor, account, clock = make_monitor(6.0)
    assert monitor.can_provision()
    assert monitor.record_failure("8x A100", "Something went wrong.")
    assert monitor.paused


def test_refusal_with_enough_balance_is_left_to_the_hardware_cooldowns():
    monitor, account, clock = make_monitor(200.0)
    assert monitor.can_provision()
    assert not monitor.record_failure("1x A100", NO_INSTANCES)
    assert not monitor.paused and monitor.can_provision()


def test_failure_reads_the_balance_at_once():
    # The last regular read still saw credits; the refusal must not wait for the
    # next one, or the hardware would collect failures in the meantime.
    monitor, account, clock = make_monitor(200.0)
    assert monitor.can_provision()
    account.balance = 0.0
    clock.advance(5)
    assert monitor.record_failure("1x A100", NO_INSTANCES)


def test_refusal_by_message_pauses_until_top_up_or_successful_probe():
    # RunPod refuses for lack of credits at a balance that looks sufficient here.
    monitor, account, clock = make_monitor(40.0)
    assert monitor.can_provision()
    assert monitor.record_failure("1x A100", "Insufficient balance to deploy")
    assert not monitor.can_provision()

    # No top-up: one start is tried after the probe interval, and only one.
    clock.advance(300)
    assert monitor.can_provision()
    assert monitor.record_failure("1x A100", "Insufficient balance to deploy")
    assert not monitor.can_provision()

    # A probe that works ends the pause.
    clock.advance(300)
    assert monitor.can_provision()
    monitor.record_success()
    assert not monitor.paused and monitor.can_provision()

    # A top-up ends it without waiting for a probe.
    assert monitor.record_failure("1x A100", "Insufficient balance to deploy")
    clock.advance(15)
    assert not monitor.can_provision()
    account.balance = 240.0
    clock.advance(15)
    assert monitor.can_provision() and not monitor.paused


def test_unreadable_balance_never_pauses_by_itself(caplog):
    # A key without access to the balance: as before this module, plus the
    # recognition by message.
    monitor, account, clock = make_monitor(RuntimeError("forbidden"))
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            assert monitor.can_provision()
            clock.advance(60)
    assert caplog.text.count("Could not read the RunPod balance") == 1
    assert not monitor.record_failure("1x A100", NO_INSTANCES)
    assert monitor.record_failure("1x A100", "insufficient funds")
    assert not monitor.can_provision()
    clock.advance(300)
    assert monitor.can_provision()  # probe


def out_of_credits_manager(monkeypatch, account):
    manager, supabase, started = scaling_manager(
        monkeypatch, lambda worker_id: "ow_worker_token"
    )
    clock = FakeTime(1000)
    manager.credit_monitor = RunpodCreditMonitor(
        account.fetch,
        min_balance_usd=5,
        check_interval_seconds=60,
        paused_check_interval_seconds=0,
        now_fn=clock.now,
    )
    return manager, supabase, started, clock


def test_manager_starts_nothing_while_out_of_credits_and_resumes_on_top_up(
    monkeypatch,
):
    account = Account(0.0)
    manager, supabase, started, clock = out_of_credits_manager(monkeypatch, account)
    jobs = [
        pending_job("job-1", "nielsrolf/ow-unsloth:v0.16.0"),
        pending_job("job-2", "nielsrolf/ow-vllm:v0.16.0"),
    ]

    for _ in range(3):
        manager.scale_workers([], jobs)
        clock.advance(15)
    # No pod, no worker row, no job touched, and no GPU type blamed.
    assert started == [] and supabase.inserts == [] and supabase.updates == []
    assert manager.hardware_registry.failures == []

    account.balance = 100.0
    manager.scale_workers([], jobs)
    assert len(started) == 2


def test_refusal_for_lack_of_credits_stops_the_cycle_without_blaming_hardware(
    monkeypatch,
):
    from openweights.cluster import org_manager as org_manager_module

    account = Account(100.0)
    manager, supabase, started, clock = out_of_credits_manager(monkeypatch, account)
    attempts = []

    def refuse(**kwargs):
        attempts.append(kwargs)
        account.balance = 0.2  # the credits ran out since the last read
        raise RuntimeError(f"Failed to start RunPod worker: {NO_INSTANCES}")

    monkeypatch.setattr(org_manager_module, "runpod_start_worker", refuse)
    jobs = [
        pending_job("job-1", "nielsrolf/ow-unsloth:v0.16.0"),
        pending_job("job-2", "nielsrolf/ow-vllm:v0.16.0"),
    ]
    manager.scale_workers([], jobs)

    # One attempt: not the second GPU type, not the second image.
    assert len(attempts) == 1
    assert manager.hardware_registry.failures == []
    assert manager.credit_monitor.paused
    # The worker row made for the attempt does not stay 'starting'.
    ((_, row),) = supabase.inserts
    assert ("worker", {"status": "terminated"}, {"id": row["id"]}) in supabase.updates
    assert not [update for update in supabase.updates if update[0] == "jobs"]
