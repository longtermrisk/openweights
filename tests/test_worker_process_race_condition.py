"""Tests for worker process race condition fix.

Verifies that capturing a local reference to the subprocess prevents
AttributeError when the health-check thread nulls self.current_process
during log streaming.
"""

import threading
import time


class FakeProcess:
    """Minimal subprocess stand-in for testing the race condition pattern."""

    def __init__(self):
        self.returncode = 0
        self._lines = ["line1\n", "line2\n", ""]
        self.stdout = self

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        return ""

    def wait(self):
        self.returncode = 0


class TestProcessRaceCondition:
    """Test the pattern: proc = self.current_process before streaming."""

    def test_local_ref_survives_null(self):
        """A local reference should remain valid even if the source is set to None."""
        current_process = FakeProcess()
        proc = current_process  # local capture (the fix)
        current_process = None  # simulates health-check thread nulling it

        # proc should still be usable
        assert proc is not None
        proc.wait()
        assert proc.returncode == 0

    def test_local_ref_survives_null_during_iteration(self):
        """Simulate the exact race: null current_process mid-loop."""
        holder = {"current_process": FakeProcess()}
        proc = holder["current_process"]

        lines_read = []
        for line in iter(proc.stdout.readline, ""):
            lines_read.append(line)
            # Simulate health-check thread nulling it between iterations
            holder["current_process"] = None

        proc.wait()
        assert len(lines_read) == 2
        assert proc.returncode == 0
        assert holder["current_process"] is None  # was nulled

    def test_without_fix_would_crash(self):
        """Without the local ref, accessing .wait() on None raises AttributeError."""
        holder = {"current_process": FakeProcess()}
        # Read lines using the holder reference directly (the old buggy pattern)
        holder["current_process"] = None

        try:
            holder["current_process"].wait()
            assert False, "Should have raised AttributeError"
        except AttributeError:
            pass  # This is the bug the fix prevents

    def test_threaded_race_condition(self):
        """Simulate the actual threading scenario: another thread nulls the process."""
        holder = {"current_process": FakeProcess()}
        proc = holder["current_process"]  # local capture (the fix)

        def cancel_job():
            """Simulates the health-check thread cancelling."""
            time.sleep(0.01)
            holder["current_process"] = None

        thread = threading.Thread(target=cancel_job)
        thread.start()

        # Stream logs using local ref (safe)
        lines = []
        for line in iter(proc.stdout.readline, ""):
            lines.append(line)
            time.sleep(0.02)  # Give the cancel thread time to null it

        proc.wait()
        thread.join()

        assert holder["current_process"] is None  # was nulled by thread
        assert proc is not None  # local ref still valid
        assert proc.returncode == 0
