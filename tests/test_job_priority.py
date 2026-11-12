"""Job priority: stored in the job_priority table, honored by workers and the cluster manager."""

import random
import sys
import types

import pytest
from postgrest.exceptions import APIError

from openweights.client.jobs import (
    JOB_SELECT,
    Job,
    Jobs,
    priority_sort_key,
    with_priority,
)


def job_row(job_id, priority=None, vram=24, created_at="2026-11-12T00:00:00Z", **kw):
    row = {
        "id": job_id,
        "type": "custom",
        "status": "pending",
        "model": None,
        "requires_vram_gb": vram,
        "docker_image": "img",
        "script": "echo hi",
        "params": {},
        "outputs": None,
        "organization_id": "org",
        "created_at": created_at,
        "updated_at": created_at,
        "worker_id": None,
        "timeout": None,
        "allowed_hardware": None,
        "job_priority": None if priority is None else {"priority": priority},
    }
    row.update(kw)
    return row


def test_with_priority_flattens_embedded_record():
    assert with_priority(job_row("a"))["priority"] == 0
    assert with_priority(job_row("a", 5))["priority"] == 5
    # PostgREST returns a list when it does not detect the one-to-one relationship
    assert with_priority(job_row("a", job_priority=[{"priority": -2}]))["priority"] == -2
    assert with_priority(job_row("a", job_priority=[]))["priority"] == 0
    assert "job_priority" not in with_priority(job_row("a", 5))
    # rows from an insert/update or an RPC have no embedded record at all
    row = job_row("a")
    del row["job_priority"]
    assert with_priority(row)["priority"] == 0


def test_job_from_row_reads_priority():
    assert Job.from_row(job_row("a", 3), manager=None).priority == 3
    assert Job.from_row(job_row("a"), manager=None).priority == 0


def test_scheduling_order():
    jobs = [
        with_priority(job_row("old", created_at="2026-11-01T00:00:00Z")),
        with_priority(job_row("big", vram=80)),
        with_priority(job_row("low", -1, vram=80)),
        with_priority(job_row("high", 10, created_at="2026-11-12T10:00:00Z")),
        with_priority(job_row("new", created_at="2026-11-12T10:00:00Z")),
    ]
    random.shuffle(jobs)
    assert [j["id"] for j in sorted(jobs, key=priority_sort_key)] == [
        "high",
        "big",
        "old",
        "new",
        "low",
    ]


# --------------------------------------------------------------------------- client


class FakeQuery:
    def __init__(self, db, table):
        self.db, self.table, self.op, self.payload, self.job_id = (
            db,
            table,
            "select",
            None,
            None,
        )
        self.columns = None

    def select(self, columns):
        self.columns = columns
        return self

    def insert(self, payload):
        self.op, self.payload = "insert", payload
        return self

    def update(self, payload):
        self.op, self.payload = "update", payload
        return self

    def upsert(self, payload, on_conflict=None):
        assert on_conflict == "job_id"
        self.op, self.payload = "upsert", payload
        return self

    def eq(self, column, value):
        assert column == "id"
        self.job_id = value
        return self

    def single(self):
        return self

    def execute(self):
        db = self.db
        db.calls.append((self.table, self.op))
        if self.table == "job_priority":
            assert self.op == "upsert"
            assert self.payload["job_id"] in db.jobs  # FK
            db.priorities[self.payload["job_id"]] = self.payload["priority"]
            return types.SimpleNamespace(data=[self.payload])
        if self.op == "insert":
            assert "priority" not in self.payload
            db.jobs[self.payload["id"]] = {**job_row(self.payload["id"]), **self.payload}
            row = dict(db.jobs[self.payload["id"]])
            del row["job_priority"]
            return types.SimpleNamespace(data=[row])
        if self.op == "update":
            assert "priority" not in self.payload
            db.jobs[self.job_id].update(self.payload)
            row = dict(db.jobs[self.job_id])
            del row["job_priority"]
            return types.SimpleNamespace(data=[row])
        assert self.columns == JOB_SELECT
        if self.job_id not in db.jobs:
            raise APIError({"message": "JSON object requested, contains 0 rows"})
        row = dict(db.jobs[self.job_id])
        priority = db.priorities.get(self.job_id)
        row["job_priority"] = None if priority is None else {"priority": priority}
        return types.SimpleNamespace(data=row)


class FakeDb:
    def __init__(self):
        self.jobs, self.priorities, self.calls = {}, {}, []

    def table(self, name):
        return FakeQuery(self, name)


@pytest.fixture
def jobs():
    db = FakeDb()
    ow = types.SimpleNamespace(_supabase=db, organization_id="org")
    manager = Jobs(ow)
    manager.db = db
    return manager


def data(**kw):
    return {"type": "custom", "docker_image": "img", "script": "echo hi", **kw}


def test_priority_does_not_change_the_job_id_and_is_not_a_jobs_column(jobs):
    plain = jobs.get_or_create_or_reset(data())
    assert plain.priority == 0
    assert jobs.db.priorities == {}  # no row needed for the default

    prioritized = jobs.get_or_create_or_reset(data(priority=7))
    assert prioritized.id == plain.id
    assert prioritized.priority == 7
    assert jobs.db.priorities == {plain.id: 7}


def test_new_job_with_priority(jobs):
    job = jobs.get_or_create_or_reset(data(priority=3))
    assert job.priority == 3
    assert jobs.db.priorities == {job.id: 3}


def test_resubmitting_without_priority_keeps_it(jobs):
    job = jobs.get_or_create_or_reset(data(priority=3))
    again = jobs.get_or_create_or_reset(data())
    assert again.priority == 3
    again = jobs.get_or_create_or_reset(data(priority=None))
    assert again.priority == 3
    # also when the job is reset from failed to pending
    jobs.db.jobs[job.id]["status"] = "failed"
    reset = jobs.get_or_create_or_reset(data())
    assert reset.status == "pending" and reset.priority == 3
    # an explicit 0 resets it
    assert jobs.get_or_create_or_reset(data(priority=0)).priority == 0
    assert jobs.db.priorities == {job.id: 0}


def test_set_priority(jobs):
    job = jobs.get_or_create_or_reset(data())
    assert job.set_priority(-5) is job
    assert job.priority == -5
    assert jobs.retrieve(job.id).priority == -5
    for bad in (1.5, "1", True, 2**40):
        with pytest.raises(ValueError):
            jobs.set_priority(job.id, bad)
        with pytest.raises(ValueError):
            jobs.get_or_create_or_reset(data(priority=bad))


def test_custom_job_create_accepts_priority(jobs):
    from pydantic import BaseModel

    class Params(BaseModel):
        class Config:
            extra = "forbid"

        x: int

    class MyJob(Jobs):
        params = Params

        def get_entrypoint(self, validated_params):
            return f"echo {validated_params.x}"

    manager = MyJob(jobs._ow)
    job = manager.create(x=1, priority=2)
    assert job.priority == 2
    assert manager.create(x=1).id == job.id


# --------------------------------------------------------------------------- worker


@pytest.fixture
def worker(monkeypatch):
    # The worker module imports torch and runpod at import time; neither is needed here.
    for name in ("torch", "runpod", "jwt"):
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    from openweights.worker.main import Worker

    w = Worker.__new__(Worker)
    w.hardware_type = "1x L40"
    w.vram_gb = 48
    w.cached_models = []
    w.docker_image = "img"
    return w


def test_worker_takes_highest_priority_job_it_can_run(worker):
    rows = [
        job_row("normal-1"),
        job_row("normal-2", vram=40),
        job_row("high", 5),
        job_row("too-big", 100, vram=80),
        job_row("other-hardware", 100, allowed_hardware=["1x H200"]),
        job_row("low", -1),
    ]
    worker._fetch_pending_jobs = lambda: rows
    for _ in range(20):
        assert worker._find_job()["id"] == "high"

    # A cached model only breaks ties within the highest priority
    worker.cached_models = ["cached"]
    rows[0]["model"] = "cached"
    assert worker._find_job()["id"] == "high"
    rows[2]["job_priority"] = None
    for _ in range(20):
        assert worker._find_job()["id"] == "normal-1"

    # Without the prioritized jobs, the rest is taken before the low priority one
    worker.cached_models = []
    picked = {worker._find_job()["id"] for _ in range(50)}
    assert picked == {"normal-1", "normal-2", "high"}

    worker._fetch_pending_jobs = lambda: [job_row("low", -1)]
    assert worker._find_job()["id"] == "low"
    worker._fetch_pending_jobs = lambda: []
    assert worker._find_job() is None
