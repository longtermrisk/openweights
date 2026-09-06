"""Unit tests for worker GPU reclaim policy (no GPU required)."""

import logging

import pytest

from openweights.worker import gpu_reclaim
from openweights.worker.gpu_reclaim import (
    ForeignGpuHolderError,
    GpuProcess,
    reclaim_gpu,
)


def _patch(monkeypatch, holders, free_fraction, visible=frozenset()):
    monkeypatch.setattr(gpu_reclaim, "_query_gpu_processes", lambda: holders)
    monkeypatch.setattr(gpu_reclaim, "_query_free_fraction", lambda: free_fraction)
    monkeypatch.setattr(gpu_reclaim, "_pid_is_visible", lambda pid: pid in visible)
    killed = []
    monkeypatch.setattr(gpu_reclaim.os, "kill", lambda pid, sig: killed.append(pid))
    return killed


def test_no_holders_is_a_noop(monkeypatch):
    killed = _patch(monkeypatch, [], 1.0)
    reclaim_gpu()
    assert killed == []


def test_small_foreign_holder_is_ignored_when_enough_vram_is_free(monkeypatch, caplog):
    # The RunPod signature observed in production: a host-side PID holding
    # ~700 MiB of a 94 GB card. Must not abort the job or shut the worker down.
    killed = _patch(monkeypatch, [GpuProcess(pid=1191391, used_mib=708)], 0.992)
    with caplog.at_level(logging.WARNING):
        reclaim_gpu()
    assert killed == []
    assert "ignoring foreign GPU holder" in caplog.text


def test_large_foreign_holder_still_fails(monkeypatch):
    _patch(monkeypatch, [GpuProcess(pid=4242, used_mib=60000)], 0.35)
    with pytest.raises(ForeignGpuHolderError) as excinfo:
        reclaim_gpu()
    assert "pid=4242" in str(excinfo.value)
    assert "0.350" in str(excinfo.value)


def test_foreign_holder_with_unknown_free_memory_fails_conservatively(monkeypatch):
    _patch(monkeypatch, [GpuProcess(pid=4242, used_mib=708)], None)
    with pytest.raises(ForeignGpuHolderError):
        reclaim_gpu()


def test_visible_leftover_holder_is_killed(monkeypatch):
    killed = _patch(
        monkeypatch, [GpuProcess(pid=777, used_mib=30000)], 0.99, visible={777}
    )
    reclaim_gpu(timeout_s=1, poll_interval_s=0)
    assert killed == [777]
