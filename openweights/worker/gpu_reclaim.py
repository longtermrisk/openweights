"""Reclaim GPU memory between jobs by killing leftover processes we own.

This runs *between* jobs on the same worker pod. It addresses the case where
a previous job's subprocess (notably vLLM's ``EngineCore``) is still holding
VRAM when the next job starts. Cross-container ghosts are *not* killable from
inside the container (PID namespace isolation); we surface those as a hard
failure so the orchestrator can reschedule the job elsewhere.

Tools used (must be present in the worker image): ``nvidia-smi``, ``ps``.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuProcess:
    """A process reported by ``nvidia-smi`` as holding GPU memory."""

    pid: int
    used_mib: int


class ForeignGpuHolderError(RuntimeError):
    """A GPU is held by a PID not visible from inside this container.

    This indicates either a sibling container on the same host, or a host-side
    daemon. It cannot be resolved from within the worker; the caller should
    fail the job so the scheduler can move it.
    """


def _query_gpu_processes() -> list[GpuProcess]:
    """Return the list of compute apps currently holding GPU memory.

    Returns an empty list if ``nvidia-smi`` is unavailable or reports none.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("nvidia-smi query failed: %s", exc)
        return []

    procs: list[GpuProcess] = []
    for line in (line.strip() for line in out.splitlines()):
        if not line:
            continue
        pid_str, mem_str = (part.strip() for part in line.split(","))
        procs.append(GpuProcess(pid=int(pid_str), used_mib=int(mem_str)))
    return procs


def _query_free_fraction() -> float | None:
    """Return min(free/total) across all GPUs, or ``None`` if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("nvidia-smi memory query failed: %s", exc)
        return None

    fractions: list[float] = []
    for line in (line.strip() for line in out.splitlines()):
        if not line:
            continue
        free_str, total_str = (part.strip() for part in line.split(","))
        free, total = int(free_str), int(total_str)
        if total <= 0:
            continue
        fractions.append(free / total)
    return min(fractions) if fractions else None


def _pid_is_visible(pid: int) -> bool:
    """Whether ``pid`` exists in our PID namespace (signal 0 probe)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _describe_pid(pid: int) -> str:
    """Best-effort human description of a PID via ``ps``."""
    try:
        return subprocess.check_output(
            ["ps", "-o", "pid=,user=,cmd=", "-p", str(pid)],
            text=True,
            timeout=5,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return f"pid={pid} (unavailable)"


def reclaim_gpu(
    *,
    min_free_fraction: float = 0.85,
    timeout_s: float = 60.0,
    poll_interval_s: float = 1.0,
) -> None:
    """Ensure each GPU has at least ``min_free_fraction`` free VRAM.

    1. Enumerate GPU-holding processes via ``nvidia-smi``.
    2. For each holder not owned by the current process tree, SIGKILL it if
       it is visible in our PID namespace.
    3. If any holder is *not* visible (cross-container/host process), raise
       :class:`ForeignGpuHolderError` immediately — retrying is futile.
    4. Poll until VRAM is reclaimed or ``timeout_s`` elapses.

    Raises:
        ForeignGpuHolderError: A foreign PID is holding the GPU.
        RuntimeError: VRAM not reclaimed within the timeout.
    """
    self_tree = {os.getpid(), os.getppid()}
    holders = _query_gpu_processes()

    if not holders:
        logger.info("reclaim_gpu: no GPU processes reported by nvidia-smi")
        return

    foreign: list[GpuProcess] = []
    killed: list[GpuProcess] = []
    skipped_self: list[GpuProcess] = []

    for proc in holders:
        if proc.pid in self_tree:
            skipped_self.append(proc)
            continue
        if not _pid_is_visible(proc.pid):
            foreign.append(proc)
            continue
        description = _describe_pid(proc.pid)
        logger.warning(
            "reclaim_gpu: killing leftover GPU holder %s (used=%d MiB)",
            description,
            proc.used_mib,
        )
        try:
            os.kill(proc.pid, signal.SIGKILL)
            killed.append(proc)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            logger.error(
                "reclaim_gpu: cannot kill pid=%s (%d MiB): %s",
                proc.pid,
                proc.used_mib,
                exc,
            )
            foreign.append(proc)

    if foreign:
        details = ", ".join(f"pid={p.pid} used={p.used_mib}MiB" for p in foreign)
        raise ForeignGpuHolderError(
            f"GPU is held by process(es) outside this container's PID namespace: {details}. "
            "Cannot reclaim from inside the worker; reschedule the job on a different pod."
        )

    if not killed:
        logger.info(
            "reclaim_gpu: nothing to kill (self=%d, foreign=0)", len(skipped_self)
        )
        return

    deadline = time.monotonic() + timeout_s
    while True:
        free_frac = _query_free_fraction()
        if free_frac is None:
            logger.warning("reclaim_gpu: cannot read VRAM, giving up wait")
            return
        if free_frac >= min_free_fraction:
            logger.info(
                "reclaim_gpu: VRAM reclaimed (min free fraction=%.3f >= %.3f)",
                free_frac,
                min_free_fraction,
            )
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"reclaim_gpu: VRAM not reclaimed within {timeout_s:.0f}s "
                f"(min free fraction={free_frac:.3f} < {min_free_fraction:.3f})"
            )
        time.sleep(poll_interval_s)
