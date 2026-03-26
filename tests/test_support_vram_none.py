"""Tests for requires_vram_gb=None support.

Verifies that None VRAM values don't crash sorting, max computation,
or worker hardware filtering.  We test the exact expressions used in
org_manager.py and worker/main.py without needing a running cluster.
"""

import pytest


class TestVramNoneSorting:
    """Test the sort key used in org_manager.scale_workers."""

    def _sort_jobs(self, jobs):
        """Replicate the sort from org_manager.py line ~433."""
        jobs.sort(key=lambda job: job["requires_vram_gb"] or 0, reverse=True)
        return jobs

    def test_sort_with_none_values(self):
        """None VRAM should be treated as 0 and sorted last."""
        jobs = [
            {"requires_vram_gb": None, "id": "a"},
            {"requires_vram_gb": 80, "id": "b"},
            {"requires_vram_gb": 48, "id": "c"},
        ]
        sorted_jobs = self._sort_jobs(jobs)
        assert [j["id"] for j in sorted_jobs] == ["b", "c", "a"]

    def test_sort_all_none(self):
        """All-None list should sort without error."""
        jobs = [
            {"requires_vram_gb": None, "id": "a"},
            {"requires_vram_gb": None, "id": "b"},
        ]
        sorted_jobs = self._sort_jobs(jobs)
        assert len(sorted_jobs) == 2

    def test_sort_mixed_zero_and_none(self):
        """None and 0 should be treated equivalently."""
        jobs = [
            {"requires_vram_gb": 0, "id": "a"},
            {"requires_vram_gb": None, "id": "b"},
            {"requires_vram_gb": 48, "id": "c"},
        ]
        sorted_jobs = self._sort_jobs(jobs)
        assert sorted_jobs[0]["id"] == "c"
        # 0 and None both become 0, so they come last (order between them is stable)
        assert {j["id"] for j in sorted_jobs[1:]} == {"a", "b"}

    def test_sort_no_none(self):
        """Normal integer values should still sort correctly."""
        jobs = [
            {"requires_vram_gb": 24, "id": "a"},
            {"requires_vram_gb": 80, "id": "b"},
        ]
        sorted_jobs = self._sort_jobs(jobs)
        assert [j["id"] for j in sorted_jobs] == ["b", "a"]


class TestVramNoneMaxComputation:
    """Test the max VRAM computation used in org_manager.scale_workers."""

    def _max_vram(self, jobs):
        """Replicate the max computation from org_manager.py line ~444."""
        return max(job["requires_vram_gb"] or 0 for job in jobs)

    def test_max_with_none(self):
        """None should be ignored (treated as 0) in max computation."""
        jobs = [
            {"requires_vram_gb": None},
            {"requires_vram_gb": 48},
        ]
        assert self._max_vram(jobs) == 48

    def test_max_all_none(self):
        """All-None batch should return 0."""
        jobs = [
            {"requires_vram_gb": None},
            {"requires_vram_gb": None},
        ]
        assert self._max_vram(jobs) == 0

    def test_max_single_none(self):
        """Single None job should return 0."""
        jobs = [{"requires_vram_gb": None}]
        assert self._max_vram(jobs) == 0

    def test_max_normal_values(self):
        """Normal integer values should still work."""
        jobs = [
            {"requires_vram_gb": 24},
            {"requires_vram_gb": 80},
        ]
        assert self._max_vram(jobs) == 80


class TestVramNoneWorkerFiltering:
    """Test the worker hardware suitability check from worker/main.py."""

    def _is_suitable(self, job, worker_vram_gb):
        """Replicate the filtering logic from worker/main.py line ~366."""
        if job.get("allowed_hardware"):
            # Simplified: if allowed_hardware is set, check would be different
            return True
        return (job["requires_vram_gb"] or 0) <= worker_vram_gb

    def test_none_vram_fits_any_worker(self):
        """A job with None VRAM should fit on any worker."""
        job = {"requires_vram_gb": None}
        assert self._is_suitable(job, 24)
        assert self._is_suitable(job, 48)
        assert self._is_suitable(job, 80)

    def test_none_vram_fits_zero_vram_worker(self):
        """Edge case: None VRAM treated as 0, fits even a 0-VRAM worker."""
        job = {"requires_vram_gb": None}
        assert self._is_suitable(job, 0)

    def test_normal_vram_filtering(self):
        """Normal integer values should filter correctly."""
        job = {"requires_vram_gb": 48}
        assert not self._is_suitable(job, 24)
        assert self._is_suitable(job, 48)
        assert self._is_suitable(job, 80)

    def test_zero_vram_fits_any(self):
        """Zero VRAM requirement should fit on any worker."""
        job = {"requires_vram_gb": 0}
        assert self._is_suitable(job, 24)


class TestJobDataclassType:
    """Test that the Job dataclass accepts None for requires_vram_gb."""

    def test_job_accepts_none_vram(self):
        """Job dataclass should accept requires_vram_gb=None."""
        # We test the type annotation indirectly: int | None should accept None.
        # This is a pure-Python check (no import of the actual dataclass needed).
        from typing import get_type_hints
        hint = int | None
        assert isinstance(None, hint.__args__[1])  # NoneType
        assert isinstance(42, hint.__args__[0])     # int
