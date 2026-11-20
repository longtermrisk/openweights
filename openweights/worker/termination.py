"""How a worker gets its pod terminated without holding the RunPod API key.

The organization's RunPod key can terminate every pod of the account, so it does not
belong on a pod that runs user code. A worker instead sets
``worker.termination_requested_at`` on its own row; the cluster manager, which holds
the key, terminates the pod on its next loop.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from postgrest.exceptions import APIError

TERMINATION_COLUMN = "termination_requested_at"

logger = logging.getLogger(__name__)


def _column_is_missing(error: APIError) -> bool:
    """The database predates the migration that adds the column."""
    # PGRST204: column not in PostgREST's schema cache; 42703: undefined column.
    return error.code in ("PGRST204", "42703")


def request_termination(supabase, worker_id: str, status: Optional[str] = None) -> bool:
    """Ask the cluster manager to terminate this worker's pod.

    ``status`` is written in the same update (a worker that has shut down passes
    ``"shutdown"``; the TTL monitor leaves the status alone).

    Returns False if the database has no ``termination_requested_at`` column yet. The
    status is still written then, and the manager falls back to terminating the pod
    once the worker has stopped pinging.
    """
    update = {TERMINATION_COLUMN: datetime.now(timezone.utc).isoformat()}
    if status is not None:
        update["status"] = status
    try:
        supabase.table("worker").update(update).eq("id", worker_id).execute()
        return True
    except APIError as error:
        if not _column_is_missing(error):
            raise
    logger.warning(
        "worker.%s does not exist yet (migration not applied); the cluster manager "
        "will terminate this pod once it stops pinging",
        TERMINATION_COLUMN,
    )
    if status is not None:
        supabase.table("worker").update({"status": status}).eq(
            "id", worker_id
        ).execute()
    return False
