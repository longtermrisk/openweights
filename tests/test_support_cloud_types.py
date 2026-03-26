"""Tests for cloud_type support.

Verifies that:
- Jobs are grouped by (cloud_type, allowed_hardware)
- cloud_type defaults to "SECURE" when absent from params
- Different cloud_type values produce separate groups
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Load org_manager.py in isolation (same stub approach as other test files).
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

_original_modules = {}
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
    _original_modules[mod_name] = sys.modules.get(mod_name, None)
    sys.modules[mod_name] = MagicMock()

sys.modules["openweights.client.decorators"].supabase_retry = lambda *a, **kw: (lambda f: f)
sys.modules["openweights.cluster.start_runpod"].HARDWARE_CONFIG = {}
sys.modules["openweights.cluster.start_runpod"].populate_hardware_config = MagicMock()

spec = importlib.util.spec_from_file_location(
    "org_manager",
    ROOT / "openweights" / "cluster" / "org_manager.py",
)
org_manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(org_manager)

OrganizationManager = org_manager.OrganizationManager


def _make_job(cloud_type=None, allowed_hardware=None):
    """Create a minimal fake job dict."""
    job = {
        "allowed_hardware": allowed_hardware,
        "requires_vram_gb": 48,
        "params": {},
    }
    if cloud_type is not None:
        job["params"]["cloud_type"] = cloud_type
    return job


class TestGroupJobsByCloudType:
    """Tests for group_jobs_by_hardware_requirements with cloud_type."""

    def _group(self, jobs):
        """Call the grouping method without a real OrganizationManager instance."""
        return OrganizationManager.group_jobs_by_hardware_requirements(None, jobs)

    def test_same_cloud_type_same_hardware_grouped(self):
        """Jobs with same cloud_type and hardware should be in one group."""
        jobs = [
            _make_job(cloud_type="SECURE", allowed_hardware=["1x L40"]),
            _make_job(cloud_type="SECURE", allowed_hardware=["1x L40"]),
        ]
        groups = self._group(jobs)
        assert len(groups) == 1
        key = list(groups.keys())[0]
        assert key[0] == "SECURE"
        assert len(groups[key]) == 2

    def test_different_cloud_type_creates_separate_groups(self):
        """Jobs with different cloud_type but same hardware should be separate."""
        jobs = [
            _make_job(cloud_type="SECURE", allowed_hardware=["1x A100"]),
            _make_job(cloud_type="COMMUNITY", allowed_hardware=["1x A100"]),
        ]
        groups = self._group(jobs)
        assert len(groups) == 2
        cloud_types = {k[0] for k in groups.keys()}
        assert cloud_types == {"SECURE", "COMMUNITY"}

    def test_default_cloud_type_is_secure(self):
        """Jobs without cloud_type in params should default to SECURE."""
        jobs = [
            _make_job(cloud_type=None, allowed_hardware=["1x L40"]),
        ]
        groups = self._group(jobs)
        key = list(groups.keys())[0]
        assert key[0] == "SECURE"

    def test_missing_params_defaults_to_secure(self):
        """Jobs with params=None should default to SECURE."""
        job = {
            "allowed_hardware": ["1x L40"],
            "requires_vram_gb": 48,
            "params": None,
        }
        groups = self._group([job])
        key = list(groups.keys())[0]
        assert key[0] == "SECURE"

    def test_hardware_order_irrelevant_for_grouping(self):
        """allowed_hardware is sorted, so order shouldn't matter."""
        jobs = [
            _make_job(cloud_type="SECURE", allowed_hardware=["1x A100", "1x L40"]),
            _make_job(cloud_type="SECURE", allowed_hardware=["1x L40", "1x A100"]),
        ]
        groups = self._group(jobs)
        assert len(groups) == 1
        assert len(list(groups.values())[0]) == 2
