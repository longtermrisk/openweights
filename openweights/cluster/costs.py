"""Snapshot the entire pod's hourly compute rate, without multiplying it twice."""

from decimal import Decimal, InvalidOperation

from openweights.cluster.start_runpod import GPU_COST_PER_HOUR


def worker_cost_fields(pod, gpu, count, started_at):
    rate = pod.get("costPerHr")
    source = "runpod"
    try:
        rate = Decimal(str(rate))
        if not rate.is_finite() or rate < 0:
            raise ValueError("Invalid pod rate")
    except (InvalidOperation, ValueError):
        estimate = GPU_COST_PER_HOUR.get(gpu)
        rate = Decimal(str(estimate)) * count if estimate is not None else None
        source = "hardware_estimate" if rate is not None else "unknown"
    return {
        "pod_id": pod["id"],
        "hourly_cost_usd": str(rate) if rate is not None else None,
        "cost_rate_source": source,
        "billing_started_at": started_at,
    }


def terminate_worker_pod(pod_id, runpod_client):
    """Confirm termination even when a worker already deleted its own pod.

    RunPod may reject repeat termination with Unauthorized. A successful lookup
    returning None confirms absence; lookup failures or a surviving pod must keep
    accounting open so the manager can retry.
    """
    try:
        runpod_client.terminate_pod(pod_id)
    except Exception:
        if runpod_client.get_pod(pod_id) is not None:
            raise
