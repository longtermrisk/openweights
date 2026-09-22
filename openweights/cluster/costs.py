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
