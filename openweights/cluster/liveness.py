"""Decide whether a worker with a stale heartbeat is really gone.

The cluster manager reaps a worker once its ``ping`` column is older than
``UNRESPONSIVE_THRESHOLD`` seconds. That heartbeat is written by a thread inside
the worker process, and it can stall while the job the worker is running is
perfectly healthy: on 2026-09-10, -14 and -15 the manager terminated four pods
whose training logs were still advancing mid-step, each exactly two minutes after
the last ping, and it fetched those logs live from the pod at the moment it killed
it. The pod's own log endpoint is therefore a second, independent liveness signal,
and this module is the rule that combines the two.

Pure functions, no I/O, so the rule is unit-testable without a database or a pod.
"""
from typing import Optional


def worker_looks_alive(
    time_since_ping: float,
    seconds_since_log_grew: Optional[float],
    *,
    threshold: float,
    grace: float,
    hard_limit: float,
) -> bool:
    """True when the manager should leave the worker alone.

    ``time_since_ping``: age of the worker's last heartbeat, seconds.
    ``seconds_since_log_grew``: how long ago the pod's log last grew, or None when
    the log could not be read (endpoint down, no pod), which is treated as no
    evidence of life.
    ``threshold``: the heartbeat age past which the worker counts as unresponsive.
    ``grace``: how long a stale-heartbeat worker may go without its log growing.
    ``hard_limit``: heartbeat age past which the worker is reaped no matter what the
    log does, so a broken heartbeat cannot keep a pod alive indefinitely.
    """
    if time_since_ping <= threshold:
        return True
    if seconds_since_log_grew is None:
        return False
    if time_since_ping > hard_limit:
        return False
    return seconds_since_log_grew < grace
