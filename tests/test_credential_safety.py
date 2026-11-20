"""Credential safety: write-only secrets in the dashboard, RunPod key only on the manager.

The database tests need the disposable PostgreSQL container of test_cost_tracking.py
(OW_COST_TEST_CONTAINER=ow-cost-tests) and are skipped without it.
"""

import json
import sys
import types
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from postgrest.exceptions import APIError

from openweights.cluster import org_manager as org_manager_module
from openweights.cluster import start_runpod
from openweights.cluster.org_manager import (
    OrganizationManager,
    worker_image_needs_runpod_key,
)
from openweights.worker import termination
from openweights.worker.services import ttl_monitor
from test_cost_tracking import KEY, ORG, OTHER, ROOT, USER, claims, db  # noqa: F401
from test_dashboard_jobs import dashboard  # noqa: F401

SECRET = "hf_abcdefghijklmnopqrstuvwxyz1234"
MIGRATION = ROOT / "supabase/migrations/20261120000000_credential_safety.sql"


class FakeSupabase:
    """Records table updates and RPCs; answers selects from `rows`."""

    def __init__(self, rows=None, fail_update=None):
        self.rows = rows or {}
        self.fail_update = fail_update  # callable(update dict) -> exception or None
        self.updates = []  # (table, data, filters)
        self.rpcs = []

    def table(self, name):
        return FakeQuery(self, name)

    from_ = table

    def rpc(self, name, params):
        self.rpcs.append((name, params))
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=True))


class FakeQuery:
    def __init__(self, supabase, table):
        self.supabase, self.table = supabase, table
        self.data, self.filters = None, {}

    def select(self, *_columns):
        return self

    def update(self, data):
        self.data = data
        return self

    def eq(self, column, value):
        self.filters[column] = value
        return self

    def in_(self, column, values):
        self.filters[column] = list(values)
        return self

    def order(self, *_args, **_kwargs):
        return self

    def execute(self):
        if self.data is None:
            return SimpleNamespace(data=self.supabase.rows.get(self.table, []))
        error = self.supabase.fail_update and self.supabase.fail_update(self.data)
        if error:
            raise error
        self.supabase.updates.append((self.table, self.data, self.filters))
        return SimpleNamespace(data=[self.data])


# --- 1. The dashboard never returns a stored secret value -----------------------


@pytest.fixture
def database_module(dashboard):  # noqa: F811 - the fixture puts the backend on sys.path
    import database

    return database


def make_database(database_module, client):
    database = database_module.Database.__new__(database_module.Database)
    database.client = client
    database.set_organization_id = lambda org_id: None
    return database


def test_mask_shows_last_four_of_secrets_and_all_of_plain_settings(database_module):
    mask = database_module.mask_secret_row
    assert mask({"name": "HF_TOKEN", "value": SECRET}) == {
        "name": "HF_TOKEN",
        "plain_value": None,
        "last_four": "1234",
        "updated_at": None,
    }
    assert mask({"name": "HF_ORG", "value": "my-org"})["plain_value"] == "my-org"
    assert mask({"name": "HF_ORG", "value": "my-org"})["last_four"] is None
    # Four characters of a short secret would be most of it.
    assert mask({"name": "PIN", "value": "12345678"})["last_four"] is None


def test_list_uses_masked_rpc(database_module):
    client = Mock()
    client.rpc.return_value.execute.return_value.data = [
        {
            "name": "HF_TOKEN",
            "plain_value": None,
            "last_four": "1234",
            "updated_at": None,
        }
    ]
    secrets = make_database(database_module, client).list_organization_secrets(ORG)
    client.rpc.assert_called_once_with(
        "list_organization_secrets_masked", {"org_id": ORG}
    )
    client.from_.assert_not_called()
    assert secrets[0].last_four == "1234"


def test_list_masks_in_the_backend_before_the_migration(database_module):
    """Without the RPC (PGRST202) rows are read as before, and masked before leaving."""
    client = FakeSupabase(
        rows={
            "organization_secrets": [
                {"name": "HF_ORG", "value": "my-org", "updated_at": None},
                {"name": "HF_TOKEN", "value": SECRET, "updated_at": None},
            ]
        }
    )
    client.rpc = Mock(side_effect=APIError({"code": "PGRST202", "message": "no rpc"}))
    secrets = make_database(database_module, client).list_organization_secrets(ORG)
    dumped = json.dumps([secret.model_dump(mode="json") for secret in secrets])
    assert SECRET not in dumped
    assert '"last_four": "1234"' in dumped and '"plain_value": "my-org"' in dumped


def test_list_refuses_non_admins(database_module):
    client = Mock()
    client.rpc.side_effect = APIError({"code": "42501", "message": "denied"})
    with pytest.raises(ValueError, match="admins"):
        make_database(database_module, client).list_organization_secrets(ORG)


def test_secret_model_has_no_field_that_could_carry_a_value(dashboard):  # noqa: F811
    import models

    assert set(models.OrganizationSecret.model_fields) == {
        "name",
        "plain_value",
        "last_four",
        "updated_at",
    }
    # Extra keys (such as a raw "value" from a careless query) are dropped.
    leaked = models.OrganizationSecret(name="HF_TOKEN", value=SECRET, last_four="1234")
    assert SECRET not in leaked.model_dump_json()


def test_secrets_endpoint_returns_masked_rows_only(dashboard):  # noqa: F811
    import models

    app, database = dashboard
    database.list_organization_secrets.return_value = [
        models.OrganizationSecret(name="HF_TOKEN", last_four="1234"),
        models.OrganizationSecret(name="HF_ORG", plain_value="my-org"),
    ]
    response = TestClient(app).get(f"/organizations/{ORG}/secrets")
    assert response.status_code == 200
    assert response.json() == [
        {
            "name": "HF_TOKEN",
            "plain_value": None,
            "last_four": "1234",
            "updated_at": None,
        },
        {
            "name": "HF_ORG",
            "plain_value": "my-org",
            "last_four": None,
            "updated_at": None,
        },
    ]


def test_secrets_are_replaced_and_deleted_one_at_a_time(dashboard):  # noqa: F811
    app, database = dashboard
    client = TestClient(app)

    response = client.put(
        f"/organizations/{ORG}/secrets/HF_TOKEN", json={"value": SECRET}
    )
    assert response.status_code == 200 and SECRET not in response.text
    database.set_organization_secret.assert_called_once_with(ORG, "HF_TOKEN", SECRET)

    assert (
        client.delete(f"/organizations/{ORG}/secrets/WANDB_API_KEY").status_code == 200
    )
    database.delete_organization_secret.assert_called_once_with(ORG, "WANDB_API_KEY")

    assert (
        client.put(
            f"/organizations/{ORG}/secrets/HF_TOKEN", json={"value": ""}
        ).status_code
        == 422
    )
    # The bulk endpoint needed the browser to hold every value; it is gone.
    assert (
        client.put(f"/organizations/{ORG}/secrets", json={"A": "b"}).status_code == 405
    )


def test_set_secret_validates_name_and_uses_the_rpc(database_module, monkeypatch):
    client = FakeSupabase(
        rows={"organization_secrets": [{"name": "HF_USER", "value": "me"}]}
    )
    database = make_database(database_module, client)
    seen = {}
    monkeypatch.setattr(
        database_module,
        "validate_organization_secrets",
        lambda secrets: seen.update(secrets) or (True, ""),
    )

    with pytest.raises(ValueError, match="letters"):
        database.set_organization_secret(ORG, "bad name", "x")
    database.set_organization_secret(ORG, "HF_TOKEN", SECRET)

    # The new token is checked against the stored, readable HF settings.
    assert seen == {"HF_USER": "me", "HF_TOKEN": SECRET}
    assert client.rpcs == [
        (
            "manage_organization_secret",
            {"org_id": ORG, "secret_name": "HF_TOKEN", "secret_value": SECRET},
        )
    ]


def test_delete_protects_the_managers_api_key(database_module):
    client = FakeSupabase()
    database = make_database(database_module, client)
    with pytest.raises(ValueError, match="cannot be deleted"):
        database.delete_organization_secret(ORG, "OPENWEIGHTS_API_KEY")
    database.delete_organization_secret(ORG, "WANDB_API_KEY")
    assert client.rpcs == [
        ("delete_organization_secret", {"org_id": ORG, "secret_name": "WANDB_API_KEY"})
    ]


# --- 1b. Row level security (real PostgreSQL) -----------------------------------


@pytest.fixture
def secrets_db(db):  # noqa: F811
    db(MIGRATION.read_text())
    db(f"""INSERT INTO organization_secrets(organization_id,name,value) VALUES
        ('{ORG}','HF_TOKEN','{SECRET}'), ('{ORG}','RUNPOD_API_KEY','rpa_short'),
        ('{ORG}','HF_ORG','my-org'), ('{OTHER}','HF_TOKEN','other-org-secret-value')""")
    return db


def test_signed_in_admin_cannot_read_secret_values_from_the_table(secrets_db):
    user = {"sub": USER}
    assert secrets_db("SELECT name, value FROM organization_secrets", claims=user) == (
        "HF_ORG|my-org"
    )
    # ... nor change or delete them behind the RPCs' back.
    secrets_db(
        "UPDATE organization_secrets SET value='x' WHERE name='HF_TOKEN'", claims=user
    )
    secrets_db("DELETE FROM organization_secrets WHERE name='HF_TOKEN'", claims=user)
    secrets_db(
        f"INSERT INTO organization_secrets(organization_id,name,value) VALUES ('{ORG}','NEW','x')",
        claims=user,
        error=True,
    )
    assert (
        secrets_db(
            f"SELECT value FROM organization_secrets WHERE organization_id='{ORG}' AND name='HF_TOKEN'"
        )
        == SECRET
    )


def test_masked_listing_shows_last_four_and_plain_settings(secrets_db):
    listing = secrets_db(
        f"SELECT name, plain_value, last_four FROM list_organization_secrets_masked('{ORG}')",
        claims={"sub": USER},
    )
    assert listing.splitlines() == [
        "HF_ORG|my-org|",
        "HF_TOKEN||1234",
        "RUNPOD_API_KEY||",
    ]
    assert SECRET not in secrets_db(
        f"SELECT * FROM list_organization_secrets_masked('{ORG}')", claims={"sub": USER}
    )


def test_masked_listing_is_for_admins_of_that_organization(secrets_db):
    for caller in ({"sub": OTHER}, claims(org=OTHER), {}):
        error = secrets_db(
            f"SELECT * FROM list_organization_secrets_masked('{ORG}')",
            claims=caller,
            error=True,
        )
        assert "Only organization admins" in error
    assert "permission denied" in secrets_db(
        f"SET ROLE anon; SELECT * FROM list_organization_secrets_masked('{ORG}')",
        error=True,
    )


def test_signed_in_admin_still_replaces_and_deletes_through_rpcs(secrets_db):
    user = {"sub": USER}
    secrets_db(
        f"SELECT manage_organization_secret('{ORG}','HF_TOKEN','hf_replacement_value_9876')",
        claims=user,
    )
    assert "HF_TOKEN||9876" in secrets_db(
        f"SELECT name, plain_value, last_four FROM list_organization_secrets_masked('{ORG}')",
        claims=user,
    )
    assert (
        secrets_db(
            f"SELECT delete_organization_secret('{ORG}','HF_TOKEN')", claims=user
        )
        == "t"
    )
    assert (
        secrets_db(
            f"SELECT count(*) FROM organization_secrets WHERE organization_id='{ORG}'"
        )
        == "2"
    )


def test_api_keys_keep_direct_access_for_the_cluster_manager(secrets_db):
    """The org manager and `ow env` read and write the table with an API key."""
    assert (
        secrets_db(
            "SELECT value FROM organization_secrets WHERE name='HF_TOKEN'",
            claims=claims(),
        )
        == SECRET
    )
    secrets_db(
        f"INSERT INTO organization_secrets(organization_id,name,value) VALUES ('{ORG}','WANDB_API_KEY','w')",
        claims=claims(),
    )
    secrets_db(
        "UPDATE organization_secrets SET value='w2' WHERE name='WANDB_API_KEY'",
        claims=claims(),
    )
    assert (
        secrets_db("SELECT count(*) FROM organization_secrets", claims=claims()) == "4"
    )
    # A key with a spending limit is not an admin and sees nothing, as before.
    secrets_db(
        f"INSERT INTO spending_limits(organization_id,api_token_id,limit_usd) VALUES ('{ORG}','{KEY}',10)"
    )
    assert (
        secrets_db("SELECT count(*) FROM organization_secrets", claims=claims()) == "0"
    )


def test_worker_can_request_termination_in_the_database(secrets_db):
    secrets_db(
        f"INSERT INTO worker(id,organization_id,status) VALUES ('w','{ORG}','active')"
    )
    secrets_db(
        "UPDATE worker SET status='shutdown', termination_requested_at=now() WHERE id='w'",
        claims=claims(),
    )
    assert (
        secrets_db(
            "SELECT status FROM worker WHERE termination_requested_at IS NOT NULL"
        )
        == "shutdown"
    )


# --- 2. Workers ask the manager for termination ---------------------------------


def test_request_termination_sets_flag_and_status():
    supabase = FakeSupabase()
    assert termination.request_termination(supabase, "worker-1", status="shutdown")
    ((table, data, filters),) = supabase.updates
    assert table == "worker" and filters == {"id": "worker-1"}
    assert data["status"] == "shutdown"
    assert datetime.fromisoformat(data["termination_requested_at"]).tzinfo


def test_request_termination_without_status_leaves_status_alone():
    supabase = FakeSupabase()
    termination.request_termination(supabase, "worker-1")
    assert list(supabase.updates[0][1]) == ["termination_requested_at"]


@pytest.mark.parametrize("code", ["PGRST204", "42703"])
def test_request_termination_before_the_migration_still_sets_status(code):
    """Unknown column: fall back to status only; the manager's ping check does the rest."""
    supabase = FakeSupabase(
        fail_update=lambda data: "termination_requested_at" in data
        and APIError({"code": code, "message": "no such column"})
    )
    assert not termination.request_termination(supabase, "worker-1", status="shutdown")
    assert supabase.updates == [("worker", {"status": "shutdown"}, {"id": "worker-1"})]


def test_request_termination_reports_other_errors():
    supabase = FakeSupabase(
        fail_update=lambda data: APIError({"code": "42501", "message": "denied"})
    )
    with pytest.raises(APIError):
        termination.request_termination(supabase, "worker-1", status="shutdown")


@pytest.fixture
def worker_main(monkeypatch):
    # The worker module imports torch at import time; it is not needed here.
    for name in ("torch", "jwt"):
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    from openweights.worker import main

    return main


def test_worker_shutdown_requests_termination_and_never_calls_runpod(
    worker_main, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    assert not hasattr(worker_main, "runpod")
    supabase = FakeSupabase()
    worker = worker_main.Worker.__new__(worker_main.Worker)
    worker.worker_id = "worker-1"
    worker.current_job = worker.current_run = None
    worker._ow = SimpleNamespace(_supabase=supabase, files=Mock())

    worker.shutdown_handler()

    (update,) = [data for table, data, _ in supabase.updates if table == "worker"]
    assert update["status"] == "shutdown" and update["termination_requested_at"]


@pytest.mark.parametrize(
    "image, needs_key",
    [
        ("nielsrolf/ow-unsloth:v0.16.0", False),
        ("nielsrolf/ow-vllm:v0.16.1", False),
        ("nielsrolf/ow-unsloth:v1.0.0", False),
        ("nielsrolf/ow-unsloth:v0.16.0-candidate", False),
        ("nielsrolf/ow-unsloth:v0.15.0", True),
        ("nielsrolf/ow-vllm:v0.9.12", True),
        ("nielsrolf/ow-unsloth:latest", True),
        ("someone/custom-image:v0.16.0", True),
        ("runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04", True),
        (None, True),
    ],
)
def test_only_images_that_predate_manager_termination_get_the_key(image, needs_key):
    assert worker_image_needs_runpod_key(image) is needs_key


def make_manager(supabase=None, secrets=None):
    manager = OrganizationManager.__new__(OrganizationManager)
    manager.org_id = ORG
    manager._ow = SimpleNamespace(_supabase=supabase or FakeSupabase(), files=Mock())
    manager.get_secrets = lambda: dict(
        secrets
        or {"RUNPOD_API_KEY": "rp", "HF_TOKEN": "hf", "OPENWEIGHTS_API_KEY": "ow"}
    )
    manager.fetch_and_save_worker_logs = Mock()
    return manager


def test_manager_keeps_the_runpod_key_out_of_new_worker_pods():
    manager = make_manager()
    env = manager.worker_env_for("nielsrolf/ow-unsloth:v0.16.0")
    assert "RUNPOD_API_KEY" not in env
    assert env["HF_TOKEN"] == "hf" and env["OPENWEIGHTS_API_KEY"] == "ow"
    # Old images terminate their own pod and still need it.
    assert (
        manager.worker_env_for("nielsrolf/ow-unsloth:v0.15.0")["RUNPOD_API_KEY"] == "rp"
    )
    # The manager itself keeps using it.
    assert manager.worker_env["RUNPOD_API_KEY"] == "rp"


def test_current_sdk_images_are_keyless():
    from openweights.images import OW_UNSLOTH_IMAGE, OW_VLLM_IMAGE

    assert not worker_image_needs_runpod_key(OW_UNSLOTH_IMAGE)
    assert not worker_image_needs_runpod_key(OW_VLLM_IMAGE)


class FakeRunpodClient:
    def __init__(self, pod_exists=False, terminate_error=None):
        self.create_kwargs = None
        self.terminated = []
        self.pod_exists, self.terminate_error = pod_exists, terminate_error

    def create_pod(self, *args, **kwargs):
        self.create_kwargs = kwargs
        return {"id": "pod-abc123"}

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)
        if self.terminate_error:
            raise self.terminate_error

    def get_pod(self, pod_id):
        return {"id": pod_id} if self.pod_exists else None


def test_start_worker_does_not_copy_the_callers_runpod_key_to_the_pod(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    client = FakeRunpodClient()
    start_runpod.start_worker(
        gpu="A100",
        image="nielsrolf/ow-unsloth:v0.16.0",
        worker_id="worker-1",
        name="test-worker",
        env={"HF_TOKEN": "hf"},
        runpod_client=client,
    )
    env = client.create_kwargs["env"]
    assert "RUNPOD_API_KEY" not in env
    assert env["WORKER_ID"] == "worker-1" and env["HF_TOKEN"] == "hf"


def test_dev_pods_keep_the_callers_own_key_for_their_ttl(monkeypatch):
    """`ow ssh` pods have no manager; the caller's key is what ends them at TTL."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    monkeypatch.setattr(start_runpod, "get_ip_and_port", lambda *_: ("10.0.0.1", 22))
    client = FakeRunpodClient()
    start_runpod.start_worker(
        gpu="A100", image="img", name="dev", dev_mode=True, env={}, runpod_client=client
    )
    assert client.create_kwargs["env"]["RUNPOD_API_KEY"] == "rp-key"


NOW = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)


def worker_row(status="active", ping_age=0, now=NOW, **extra):
    return {
        "id": "worker-1",
        "status": status,
        "pod_id": "pod-1",
        "ping": (now - timedelta(seconds=ping_age)).isoformat(),
        "termination_requested_at": None,
        **extra,
    }


@pytest.mark.parametrize(
    "worker, terminated",
    [
        # The worker asked for it: immediately, whatever its status and ping.
        (worker_row("shutdown", termination_requested_at=NOW.isoformat()), True),
        (worker_row("active", termination_requested_at=NOW.isoformat()), True),
        # Healthy workers are left alone.
        (worker_row("active", ping_age=30), False),
        (worker_row("starting", ping_age=300), False),
        # Asked to shut down by the manager, still finishing (uploading logs).
        (worker_row("shutdown", ping_age=30), False),
        # Workers that exit without setting the flag stop pinging: a crashed process,
        # an old image that failed to terminate itself, a pod that never came up.
        (worker_row("active", ping_age=121), True),
        (worker_row("shutdown", ping_age=121), True),
        (worker_row("starting", ping_age=601), True),
        (worker_row("active", ping=None), True),
        # A row from before the migration has no such key at all.
        ({"id": "w", "status": "active", "ping": NOW.isoformat()}, False),
    ],
)
def test_manager_terminates_on_request_or_when_pings_stop(worker, terminated):
    reason = make_manager().termination_reason(worker, NOW)
    assert bool(reason) is terminated


def test_manager_terminates_pod_reverts_jobs_and_marks_worker(monkeypatch):
    runpod_client = FakeRunpodClient()
    monkeypatch.setattr(org_manager_module, "runpod", runpod_client)
    supabase = FakeSupabase(rows={"runs": [{"id": 7, "job_id": "job-1"}]})
    manager = make_manager(supabase)
    worker = worker_row("shutdown", termination_requested_at=NOW.isoformat())

    healthy = worker_row(ping_age=1, now=datetime.now(timezone.utc)) | {"id": "ok"}
    assert manager.clean_up_workers([worker, healthy]) == ["worker-1"]

    assert runpod_client.terminated == ["pod-1"]
    assert ("runs", {"status": "failed"}, {"id": 7}) in supabase.updates
    assert supabase.rpcs[0][0] == "update_job_status_if_in_progress"
    assert supabase.rpcs[0][1]["_new_status"] == "pending"
    assert supabase.updates[-1] == (
        "worker",
        {"status": "terminated"},
        {"id": "worker-1"},
    )


def test_failed_pod_termination_is_retried_not_marked_terminated(monkeypatch):
    runpod_client = FakeRunpodClient(pod_exists=True, terminate_error=RuntimeError("x"))
    monkeypatch.setattr(org_manager_module, "runpod", runpod_client)
    supabase = FakeSupabase()
    manager = make_manager(supabase)
    worker = worker_row("shutdown", termination_requested_at=NOW.isoformat())

    assert manager.clean_up_workers([worker]) == []
    assert not any(data == {"status": "terminated"} for _, data, _ in supabase.updates)


def test_pod_an_old_worker_already_deleted_counts_as_terminated(monkeypatch):
    """Old images terminate their own pod; RunPod then rejects the manager's call."""
    runpod_client = FakeRunpodClient(
        pod_exists=False, terminate_error=RuntimeError("x")
    )
    monkeypatch.setattr(org_manager_module, "runpod", runpod_client)
    manager = make_manager()
    stale = worker_row("shutdown", ping_age=500, now=datetime.now(timezone.utc))
    assert manager.clean_up_workers([stale]) == ["worker-1"]


def test_worker_without_pod_is_only_marked_terminated(monkeypatch):
    runpod_client = FakeRunpodClient()
    monkeypatch.setattr(org_manager_module, "runpod", runpod_client)
    supabase = FakeSupabase()
    worker = worker_row(
        "shutdown", pod_id=None, termination_requested_at=NOW.isoformat()
    )
    assert make_manager(supabase).clean_up_workers([worker]) == ["worker-1"]
    assert runpod_client.terminated == []


# --- TTL monitor -----------------------------------------------------------------


def test_expired_managed_pod_asks_the_manager(monkeypatch):
    requested = []
    monkeypatch.setenv("WORKER_ID", "worker-1")
    monkeypatch.setenv("OPENWEIGHTS_API_KEY", "ow_test")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr(
        termination,
        "request_termination",
        lambda supabase, worker_id, status=None: requested.append((worker_id, status))
        or True,
    )
    monkeypatch.setitem(
        sys.modules,
        "openweights",
        SimpleNamespace(OpenWeights=lambda: SimpleNamespace(_supabase=object())),
    )

    # Not "done": the monitor keeps running (and asking) until the pod is gone.
    assert ttl_monitor.terminate_expired_pod() is False
    assert requested == [("worker-1", None)]


def test_expired_dev_pod_terminates_itself_with_the_callers_key(monkeypatch):
    calls = []
    monkeypatch.delenv("WORKER_ID", raising=False)
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-1")
    monkeypatch.setitem(
        sys.modules,
        "runpod",
        SimpleNamespace(
            terminate_pod=lambda pod_id: calls.append(pod_id), api_key=None
        ),
    )
    assert ttl_monitor.terminate_expired_pod() is True
    assert calls == ["pod-1"]


def test_expired_pod_without_any_means_keeps_retrying(monkeypatch):
    for name in ("WORKER_ID", "OPENWEIGHTS_API_KEY", "RUNPOD_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert ttl_monitor.terminate_expired_pod() is False
