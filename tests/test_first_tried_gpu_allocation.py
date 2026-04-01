"""Tests for deterministic GPU allocation (first entry in allowed_hardware).

We load `determine_gpu_type` directly from the source file to avoid pulling
in the full openweights package (which needs DB credentials and has
import-time side effects).
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Load org_manager.py in isolation by stubbing its heavy dependencies.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

# Stub modules that org_manager imports at the top level
_stubs = {}
for mod_name in [
    "openweights",
    "openweights.client",
    "openweights.client.decorators",
    "openweights.cluster",
    "openweights.cluster.start_runpod",
    "requests",
    "runpod",
    "dotenv",
]:
    _stubs[mod_name] = sys.modules.get(mod_name, None)
    sys.modules[mod_name] = MagicMock()

# Make supabase_retry a passthrough decorator
sys.modules["openweights.client.decorators"].supabase_retry = lambda *a, **kw: (lambda f: f)

# Provide a real dict for HARDWARE_CONFIG so sorted() works on it
MOCK_HARDWARE_CONFIG = {
    43: ["1x L40"],       # 48 GB - 5
    75: ["1x A100"],      # 80 GB - 5
    136: ["1x H200"],     # 141 GB - 5
}
sys.modules["openweights.cluster.start_runpod"].HARDWARE_CONFIG = MOCK_HARDWARE_CONFIG
sys.modules["openweights.cluster.start_runpod"].populate_hardware_config = MagicMock()

# Now import the module under test
spec = importlib.util.spec_from_file_location(
    "org_manager",
    ROOT / "openweights" / "cluster" / "org_manager.py",
)
org_manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(org_manager)

determine_gpu_type = org_manager.determine_gpu_type


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDetermineGpuType:
    """Tests for determine_gpu_type with allowed_hardware."""

    def test_picks_first_entry(self):
        """Should always pick the first entry, not a random one."""
        gpu, count = determine_gpu_type(0, allowed_hardware=["1x L40", "1x A100", "1x H200"])
        assert gpu == "L40"
        assert count == 1

    def test_multi_gpu_config(self):
        """Should parse multi-GPU entries correctly."""
        gpu, count = determine_gpu_type(0, allowed_hardware=["2x A100", "1x H200"])
        assert gpu == "A100"
        assert count == 2

    def test_no_allowed_hardware_falls_through(self):
        """When allowed_hardware is None, should fall through to VRAM-based logic."""
        gpu, count = determine_gpu_type(40, allowed_hardware=None)
        assert gpu == "L40"
        assert count == 1
