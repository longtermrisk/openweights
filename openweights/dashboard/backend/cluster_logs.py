"""Read an organization's cluster manager log for the dashboard's Cluster tab.

The supervisor (`ow cluster --super`) writes each org manager's output to
`$OW_ORG_MANAGER_LOG_DIR/org_<org_id>_{stdout,stderr}.log` with one rotated
backup `.1`. The logging output (provisioning decisions, cooldowns, errors) goes
to stderr; stdout only has debug prints, so we serve stderr.
"""

import os
import re
import uuid
from pathlib import Path

# The supervisor's own "<asctime> - " prefix, followed by the org manager's
# "<file>.py   :<line>  " prefix. Both are noise next to the manager's own timestamp.
_PREFIX = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} - \S+\.py\s*:\d+\s+")
_SECRET = re.compile(
    r"\b(?:rpa_|hf_|ow_|sk-|ghp_|gho_)[A-Za-z0-9_\-]{16,}"
    r"|\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"
)


def log_dir() -> Path:
    return Path(os.environ.get("OW_ORG_MANAGER_LOG_DIR", "logs"))


def read_cluster_log(organization_id: str, max_lines: int) -> str:
    """Return the last `max_lines` lines of the org manager log, oldest first."""
    org_id = str(uuid.UUID(organization_id))  # never build a path from raw input
    current = log_dir() / f"org_{org_id}_stderr.log"
    lines: list[str] = []
    for path in (current.with_name(current.name + ".1"), current):
        if path.exists():
            lines.extend(path.read_text(errors="replace").splitlines())
    return "\n".join(
        _SECRET.sub("[redacted]", _PREFIX.sub("", line)) for line in lines[-max_lines:]
    )
