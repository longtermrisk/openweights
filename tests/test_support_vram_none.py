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
