"""Unit tests and real PostgreSQL accounting/RLS tests.

Integration: start disposable postgres:16 container named ow-cost-tests and run
OW_COST_TEST_CONTAINER=ow-cost-tests pytest tests/test_cost_tracking.py.
Each test gets its own database; no configured Supabase credentials are used.
"""

import json
import os
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openweights.client.costs import Costs
from openweights.cluster.costs import terminate_worker_pod, worker_cost_fields

ROOT = Path(__file__).resolve().parents[1]
ORG = "00000000-0000-0000-0000-000000000001"
OTHER = "00000000-0000-0000-0000-000000000002"
USER = "00000000-0000-0000-0000-000000000003"
KEY = "00000000-0000-0000-0000-000000000004"


def test_provider_rate_is_per_pod_not_per_gpu():
    assert worker_cost_fields({"id": "p", "costPerHr": "4.25"}, "A100", 4, "start") == {
        "pod_id": "p",
        "hourly_cost_usd": "4.25",
        "cost_rate_source": "runpod",
        "billing_started_at": "start",
    }


@pytest.mark.parametrize("rate", [None, "NaN", "Infinity", -1, "invalid"])
def test_invalid_rate_uses_hardware_estimate(rate):
    fields = worker_cost_fields({"id": "p", "costPerHr": rate}, "A100", 2, "start")
    assert fields["hourly_cost_usd"] == "2.78"
    assert fields["cost_rate_source"] == "hardware_estimate"


def test_unknown_hardware_is_not_free():
    assert (
        worker_cost_fields({"id": "p"}, "unknown", 1, "start")["hourly_cost_usd"]
        is None
    )


@pytest.mark.parametrize("amount", [-1, float("nan"), float("inf")])
def test_sdk_rejects_invalid_limit(amount):
    with pytest.raises(ValueError):
        Costs(None).set_limit(KEY, amount)


def test_sdk_report_and_limit():
    client = Mock()
    costs = Costs(SimpleNamespace(_supabase=client, organization_id=ORG))
    costs.report()
    client.rpc.assert_called_with(
        "get_cost_report", {"org_id": ORG, "row_limit": 100, "row_offset": 0}
    )
    costs.set_limit(KEY, "0.01")
    client.rpc.assert_called_with(
        "set_spending_limit", {"org_id": ORG, "key_id": KEY, "amount_usd": "0.01"}
    )
    costs.set_limit(KEY)
    assert client.rpc.call_args.args[1]["amount_usd"] is None


@pytest.fixture
def db():
    container = os.environ.get("OW_COST_TEST_CONTAINER")
    if not container:
        pytest.skip(
            "Set OW_COST_TEST_CONTAINER to opt into disposable PostgreSQL tests"
        )
    if container != "ow-cost-tests":
        pytest.fail("Only the disposable ow-cost-tests container is supported")
    database = "cost_test_" + uuid.uuid4().hex

    def sql(statement, *, claims=None, error=False, dbname=database):
        if claims is not None:
            statement = (
                "SET ROLE authenticated; SET request.jwt.claims = '"
                + json.dumps(claims)
                + "'; "
                + statement
            )
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                container,
                "psql",
                "-X",
                "-qAt",
                "-v",
                "ON_ERROR_STOP=1",
                "-U",
                "postgres",
                "-d",
                dbname,
            ],
            input=statement,
            text=True,
            capture_output=True,
        )
        if error:
            assert result.returncode != 0, result.stdout
            return result.stderr
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    sql(f"CREATE DATABASE {database}", dbname="postgres")
    try:
        bootstrap = """
        DO $$ BEGIN CREATE ROLE anon; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
        DO $$ BEGIN CREATE ROLE authenticated; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
        DO $$ BEGIN CREATE ROLE service_role; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
        CREATE SCHEMA auth;
        CREATE TABLE auth.users(id uuid PRIMARY KEY, email text);
        CREATE FUNCTION auth.uid() RETURNS uuid LANGUAGE sql STABLE AS $$
        SELECT (nullif(current_setting('request.jwt.claims', true), '')::jsonb ->> 'sub')::uuid $$;
        GRANT USAGE ON SCHEMA auth TO authenticated;
        """
        # Use the actual production core schema, excluding unavailable Supabase extensions.
        schema = (ROOT / "supabase/migrations/20250116_v1_schema.sql").read_text()
        schema = "\n".join(
            line
            for line in schema.splitlines()
            if not line.startswith("CREATE EXTENSION")
        )
        sql(bootstrap + schema)
        sql(
            "CREATE SCHEMA IF NOT EXISTS extensions; CREATE EXTENSION pgcrypto WITH SCHEMA extensions;"
        )
        sql((ROOT / "supabase/migrations/20260922000000_cost_tracking.sql").read_text())
        sql(
            (
                ROOT / "supabase/migrations/20260922180000_token_creation_limit.sql"
            ).read_text()
        )
        sql(f"""
            INSERT INTO organizations(id,name) VALUES ('{ORG}','test'), ('{OTHER}','other');
            INSERT INTO auth.users(id) VALUES ('{USER}');
            INSERT INTO organization_members(organization_id,user_id,role) VALUES ('{ORG}','{USER}','admin');
            INSERT INTO api_tokens(id,organization_id,name,token_prefix,token_hash,created_by)
            VALUES ('{KEY}','{ORG}','key','ow_test','hash','{USER}');
        """)
        yield sql
    finally:
        sql(f"DROP DATABASE {database} WITH (FORCE)", dbname="postgres")


def claims(org=ORG, key=KEY):
    return {"organization_id": org, "api_token_id": key}


def add_job(db, job="job", key=KEY):
    db(
        f"INSERT INTO jobs(id,type,organization_id,submitted_by) VALUES ('{job}','script','{ORG}','{OTHER}')",
        claims=claims(key=key),
    )


def add_worker(db, worker="worker", rate="2"):
    db(
        f"""INSERT INTO worker(id,organization_id,pod_id,status,hardware_type,hourly_cost_usd,cost_rate_source,billing_started_at)
        VALUES ('{worker}','{ORG}','pod','active','1x A100',{rate},'runpod',now()-interval '4 hours')"""
    )


def report(db, org=ORG):
    return json.loads(db(f"SELECT get_cost_report('{org}')", claims=claims(org)))


def test_equal_overhead_per_distinct_job_not_retry(db):
    add_worker(db)
    add_job(db, "a")
    add_job(db, "b")
    for run_id, job, start, end in [(1, "a", 3, 2), (2, "a", 2, 1.5), (3, "b", 1, 0.5)]:
        db(
            f"INSERT INTO runs(id,job_id,worker_id,status) VALUES ({run_id},'{job}','worker','completed')"
        )
        db(
            f"UPDATE cost_runs SET started_at=now()-interval '{start} hours', ended_at=now()-interval '{end} hours' WHERE run_id={run_id}"
        )
    db("UPDATE worker SET status='terminated' WHERE id='worker'")
    r = report(db)
    jobs = {j["job_id"]: j for j in r["jobs"]}
    assert jobs["a"]["direct_usd"] == pytest.approx(3)
    assert jobs["b"]["direct_usd"] == pytest.approx(1)
    assert jobs["a"]["overhead_usd"] == pytest.approx(2, abs=0.01)
    assert jobs["b"]["overhead_usd"] == pytest.approx(2, abs=0.01)
    assert r["total_usd"] == pytest.approx(sum(j["total_usd"] for j in r["jobs"]))
    assert r["api_keys"][0]["total_usd"] == r["users"][0]["total_usd"]
    assert r["users"][0]["user_id"] == USER  # forged submitting user was ignored


def test_idle_worker_overhead_unallocated(db):
    add_worker(db)
    r = report(db)
    assert r["total_usd"] == pytest.approx(8, abs=0.01)
    assert r["unallocated_overhead_usd"] == r["total_usd"]
    assert r["jobs"] == []


def test_cost_history_survives_job_and_key_deletion(db):
    add_worker(db)
    add_job(db)
    db("INSERT INTO runs(job_id,worker_id,status) VALUES ('job','worker','completed')")
    db("UPDATE worker SET status='terminated' WHERE id='worker'")
    before = report(db)["total_usd"]
    db(f"DELETE FROM jobs; DELETE FROM api_tokens WHERE id='{KEY}'")
    r = report(db)
    assert r["total_usd"] == before
    assert r["api_keys"][0]["api_token_id"] == KEY
    assert r["users"][0]["user_id"] == USER


def test_frozen_end_timestamps_ignore_log_updates(db):
    add_worker(db)
    add_job(db)
    db(
        "INSERT INTO runs(job_id,worker_id,status) VALUES ('job','worker','in_progress')"
    )
    db("UPDATE runs SET status='failed'")
    before = db("SELECT ended_at FROM cost_runs")
    db("UPDATE runs SET log_file='late-upload'")
    assert db("SELECT ended_at FROM cost_runs") == before
    db("UPDATE worker SET status='terminated'")
    before = report(db)["total_usd"]
    db("UPDATE worker SET logfile='late-upload',hourly_cost_usd=100")
    assert report(db)["total_usd"] == before


def test_failed_shutdown_still_accrues(db):
    add_worker(db)
    db("UPDATE worker SET status='shutdown'")
    assert db("SELECT ended_at IS NULL FROM cost_workers") == "t"
    db("UPDATE worker SET status='terminated'")
    assert db("SELECT ended_at IS NOT NULL FROM cost_workers") == "t"


def test_tenant_and_ledger_access(db):
    assert "Organization access denied" in db(
        f"SELECT get_cost_report('{OTHER}')", claims=claims(), error=True
    )
    for table in ["cost_workers", "cost_runs", "cost_job_amounts", "spending_limits"]:
        assert "permission denied" in db(
            f"SELECT * FROM {table}", claims=claims(), error=True
        )
    assert "permission denied" in db(
        f"SELECT key_budget_exhausted('{ORG}','{KEY}')", claims=claims(), error=True
    )
    assert "permission denied" in db(
        f"SET ROLE anon; SELECT get_cost_report('{ORG}')", error=True
    )


def test_only_human_admin_can_change_limits(db):
    assert "Only a signed-in" in db(
        f"SELECT set_spending_limit('{ORG}','{KEY}',10)", claims=claims(), error=True
    )
    db(f"SELECT set_spending_limit('{ORG}','{KEY}',10)", claims={"sub": USER})
    assert report(db)["limits"][0]["limit_usd"] == 10
    db(f"SELECT set_spending_limit('{ORG}','{KEY}',NULL)", claims={"sub": USER})
    assert report(db)["limits"] == []
    for value in ["-1", "'NaN'", "'Infinity'"]:
        assert "check constraint" in db(
            f"SELECT set_spending_limit('{ORG}','{KEY}',{value})",
            claims={"sub": USER},
            error=True,
        )


def test_zero_limit_rejects_submission_restart_and_acquire(db):
    add_job(db)
    add_worker(db)
    db(f"SELECT set_spending_limit('{ORG}','{KEY}',0)", claims={"sub": USER})
    assert "spending limit reached" in db(
        f"INSERT INTO jobs(id,type,organization_id) VALUES ('new','script','{ORG}')",
        claims=claims(),
        error=True,
    )
    assert "spending limit reached" in db(
        "SELECT acquire_job('job','worker')", claims=claims(), error=True
    )
    db("UPDATE jobs SET status='canceled' WHERE id='job'", claims=claims())
    assert "spending limit reached" in db(
        "UPDATE jobs SET status='pending' WHERE id='job'", claims=claims(), error=True
    )


def test_limit_cancels_running_and_pending_work(db):
    add_worker(db)
    add_job(db)
    add_job(db, "pending")
    db("SELECT acquire_job('job','worker')", claims=claims())
    db(
        "INSERT INTO runs(job_id,worker_id,status) VALUES ('job','worker','in_progress')",
        claims=claims(),
    )
    db(f"SELECT set_spending_limit('{ORG}','{KEY}',1)", claims={"sub": USER})
    assert db(f"SELECT enforce_spending_limits('{ORG}')", claims=claims()) == "2"
    # Accounting continues until the process actually stops, not cancellation request time.
    assert db("SELECT ended_at IS NULL FROM cost_runs") == "t"
    assert db("SELECT count(*) FROM jobs WHERE status='canceled'") == "2"


def test_budgeted_job_cannot_run_on_unknown_price(db):
    add_worker(db, rate="NULL")
    add_job(db)
    db(f"SELECT set_spending_limit('{ORG}','{KEY}',10)", claims={"sub": USER})
    assert "unpriced workers" in db(
        "SELECT acquire_job('job','worker')", claims=claims(), error=True
    )
    assert report(db)["unpriced_workers"] == 1


def test_multiple_workers_and_unpriced_run(db):
    add_job(db)
    add_worker(db, "w1")
    add_worker(db, "w2", rate="3")
    db(
        "INSERT INTO runs(job_id,worker_id,status) VALUES ('job','w1','completed'),('job','w2','completed'),('job',NULL,'completed')"
    )
    r = report(db)
    assert r["total_usd"] == pytest.approx(20, abs=0.01)
    assert r["jobs"][0]["total_usd"] == pytest.approx(r["total_usd"])
    assert r["unknown_runs"] == 1


def test_cross_org_worker_and_overlapping_runs_rejected(db):
    add_worker(db)
    add_job(db)
    db(
        "INSERT INTO runs(job_id,worker_id,status) VALUES ('job','worker','in_progress')"
    )
    assert "already has an active run" in db(
        "INSERT INTO runs(job_id,worker_id,status) VALUES ('job','worker','in_progress')",
        error=True,
    )
    db(f"INSERT INTO worker(id,organization_id) VALUES ('other','{OTHER}')")
    assert "organizations must match" in db(
        "INSERT INTO runs(job_id,worker_id,status) VALUES ('job','other','in_progress')",
        error=True,
    )


def test_pagination_does_not_change_totals(db):
    for n in range(3):
        add_worker(db, worker=f"worker{n}")
        add_job(db, job=f"job{n}")
        db(
            f"INSERT INTO runs(job_id,worker_id,status) VALUES ('job{n}','worker{n}','completed')"
        )
    page = json.loads(db(f"SELECT get_cost_report('{ORG}',1,1)", claims=claims()))
    full = report(db)
    assert page["job_count"] == 3 and page["worker_count"] == 3
    assert len(page["jobs"]) == len(page["workers"]) == 1
    assert page["total_usd"] == pytest.approx(full["total_usd"], abs=0.01)
    assert len(page["api_keys"]) == 1
    assert "Invalid cost report pagination" in db(
        f"SELECT get_cost_report('{ORG}',0,0)", claims=claims(), error=True
    )


def test_already_deleted_pod_finishes_accounting():
    provider = Mock()
    provider.terminate_pod.side_effect = RuntimeError("Unauthorized")
    provider.get_pod.return_value = None
    terminate_worker_pod("pod", provider)
    provider.get_pod.assert_called_once_with("pod")


def test_failed_termination_of_existing_pod_is_retried():
    provider = Mock()
    provider.terminate_pod.side_effect = RuntimeError("termination failed")
    provider.get_pod.return_value = {"id": "pod"}
    with pytest.raises(RuntimeError, match="termination failed"):
        terminate_worker_pod("pod", provider)


def test_failed_lookup_does_not_stop_cost_clock():
    provider = Mock()
    provider.terminate_pod.side_effect = RuntimeError("termination failed")
    provider.get_pod.side_effect = RuntimeError("lookup failed")
    with pytest.raises(RuntimeError, match="lookup failed"):
        terminate_worker_pod("pod", provider)


@pytest.mark.parametrize("limit", ["0", "12.34"])
def test_token_and_budget_created_together(db, limit):
    key = db(
        f"SELECT token_id FROM create_api_token_with_limit('{ORG}','limited',{limit})",
        claims={"sub": USER},
    )
    assert (
        db(f"SELECT limit_usd FROM spending_limits WHERE api_token_id='{key}'") == limit
    )
    assert db(f"SELECT name FROM api_tokens WHERE id='{key}'") == "limited"


@pytest.mark.parametrize("limit", ["NULL", "-1", "'NaN'", "'Infinity'", "'-Infinity'"])
def test_invalid_initial_limit_does_not_create_token(db, limit):
    db(
        f"SELECT * FROM create_api_token_with_limit('{ORG}','invalid',{limit})",
        claims={"sub": USER},
        error=True,
    )
    assert db("SELECT count(*) FROM api_tokens WHERE name='invalid'") == "0"


def test_api_key_can_choose_initial_budget_but_not_change_it(db):
    key = db(
        f"SELECT token_id FROM create_api_token_with_limit('{ORG}','limited',10)",
        claims=claims(),
    )
    assert (
        db(f"SELECT limit_usd FROM spending_limits WHERE api_token_id='{key}'") == "10"
    )
    assert "signed-in organization admin" in db(
        f"SELECT set_spending_limit('{ORG}','{key}',100)", claims=claims(), error=True
    )


def test_initial_limit_denial_creates_no_token(db):
    db(
        f"SELECT * FROM create_api_token_with_limit('{OTHER}','foreign',10)",
        claims=claims(),
        error=True,
    )
    assert db("SELECT count(*) FROM api_tokens WHERE name='foreign'") == "0"
    db(f"UPDATE organization_members SET role='user' WHERE user_id='{USER}'")
    db(
        f"SELECT * FROM create_api_token_with_limit('{ORG}','denied',10)",
        claims={"sub": USER},
        error=True,
    )
    assert db("SELECT count(*) FROM api_tokens WHERE name='denied'") == "0"


def test_failure_to_save_budget_rolls_back_created_token(db):
    db("""CREATE FUNCTION reject_test_budget() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN RAISE EXCEPTION 'test budget failure'; END $$;
    CREATE TRIGGER reject_test_budget BEFORE INSERT ON spending_limits
    FOR EACH ROW EXECUTE FUNCTION reject_test_budget();""")
    assert "test budget failure" in db(
        f"SELECT * FROM create_api_token_with_limit('{ORG}','rollback',10)",
        claims=claims(),
        error=True,
    )
    assert db("SELECT count(*) FROM api_tokens WHERE name='rollback'") == "0"


def test_hundred_jobs_exhausted_after_sixty_preserves_completed_and_other_keys(db):
    db(
        f"INSERT INTO jobs(id,type,organization_id) SELECT 'batch-'||i,'script','{ORG}' FROM generate_series(1,100) i",
        claims=claims(),
    )
    db("UPDATE jobs SET status='completed' WHERE substring(id from 7)::integer <= 60")
    add_worker(db)
    db(
        "INSERT INTO runs(job_id,worker_id,status) VALUES ('batch-1','worker','completed')"
    )
    db("UPDATE cost_runs SET started_at=now()-interval '1 hour', ended_at=now()")
    other_key = db(
        f"SELECT token_id FROM create_api_token('{ORG}','other-key')", claims=claims()
    )
    add_job(db, "other-key-job", key=other_key)
    db(f"SELECT set_spending_limit('{ORG}','{KEY}',1)", claims={"sub": USER})
    assert db(f"SELECT enforce_spending_limits('{ORG}')", claims=claims()) == "40"
    assert db("SELECT count(*) FROM jobs WHERE status='completed'") == "60"
    assert db("SELECT count(*) FROM jobs WHERE status='canceled'") == "40"
    assert db("SELECT status FROM jobs WHERE id='other-key-job'") == "pending"
    assert "spending limit reached" in db(
        f"INSERT INTO jobs(id,type,organization_id) VALUES ('new','script','{ORG}')",
        claims=claims(),
        error=True,
    )
