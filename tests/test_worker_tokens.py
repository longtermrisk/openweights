"""A worker gets an OpenWeights token of its own instead of the organization's key.

The database tests need the disposable PostgreSQL container of test_cost_tracking.py
(OW_COST_TEST_CONTAINER=ow-cost-tests) and are skipped without it.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from postgrest.exceptions import APIError

from openweights.cluster.org_manager import WORKER_TOKEN_LIFETIME_HOURS
from test_cost_tracking import KEY, ORG, OTHER, ROOT, USER, claims, db  # noqa: F401
from test_credential_safety import (  # noqa: F401
    FakeSupabase,
    database_module,
    make_database,
    make_manager,
)
from test_dashboard_jobs import dashboard  # noqa: F401

MIGRATIONS = [
    ROOT / "supabase/migrations/20261120000000_credential_safety.sql",
    ROOT / "supabase/migrations/20261121000000_secret_metadata.sql",
    ROOT / "supabase/migrations/20261122000000_worker_tokens.sql",
    ROOT / "supabase/migrations/20261123000000_worker_token_broker_settings.sql",
]
WORKER = "worker-1"


# --- The cluster manager -----------------------------------------------------------


class TokenSupabase(FakeSupabase):
    def __init__(self, error=None):
        super().__init__()
        self.error = error

    def rpc(self, name, params):
        self.rpcs.append((name, params))

        def execute():
            if self.error:
                raise self.error
            return SimpleNamespace(
                data=[{"token_id": "id", "token": "ow_worker_token"}]
            )

        return SimpleNamespace(execute=execute)


def test_manager_asks_for_a_token_that_outlives_the_pod_by_a_margin():
    supabase = TokenSupabase()
    manager = make_manager(supabase=supabase)

    assert manager.create_worker_token(WORKER) == "ow_worker_token"

    ((name, params),) = supabase.rpcs
    assert name == "create_worker_token"
    assert params["org_id"] == ORG and params["for_worker_id"] == WORKER
    lifetime = datetime.fromisoformat(params["expires_at"]) - datetime.now(timezone.utc)
    assert (
        timedelta(hours=24) < lifetime <= timedelta(hours=WORKER_TOKEN_LIFETIME_HOURS)
    )


def test_worker_gets_its_own_token_in_place_of_the_organization_key():
    manager = make_manager()
    env = manager.worker_env_for("nielsrolf/ow-unsloth:v0.16.0", "ow_worker_token")
    assert env["OPENWEIGHTS_API_KEY"] == "ow_worker_token"
    assert "ow" not in env.values()  # the organization's key, see make_manager
    # Any image works with it: old ones read the same variable.
    old = manager.worker_env_for("nielsrolf/ow-unsloth:v0.14.0", "ow_worker_token")
    assert old["OPENWEIGHTS_API_KEY"] == "ow_worker_token"


def test_before_the_migration_the_worker_keeps_the_organization_key():
    missing = APIError({"code": "PGRST202", "message": "function not found"})
    manager = make_manager(supabase=TokenSupabase(error=missing))
    assert manager.create_worker_token(WORKER) is None
    env = manager.worker_env_for("nielsrolf/ow-unsloth:v0.16.0", None)
    assert env["OPENWEIGHTS_API_KEY"] == "ow"


def test_other_database_errors_are_not_swallowed():
    denied = APIError({"code": "42501", "message": "Only organization admins"})
    manager = make_manager(supabase=TokenSupabase(error=denied))
    with pytest.raises(APIError):
        manager.create_worker_token(WORKER)


def test_dashboard_does_not_list_worker_tokens_as_api_keys(database_module):
    row = {"name": "n", "created_at": "2026-11-20T00:00:00+00:00"}
    client = FakeSupabase(
        rows={
            "api_tokens": [
                {**row, "id": "a", "worker_id": None},
                {**row, "id": "b", "worker_id": WORKER, "token_hash": "h"},
                {**row, "id": "c"},  # a database from before worker tokens
            ]
        }
    )
    database = make_database(database_module, client)
    assert [token.id for token in database.list_tokens(ORG)] == ["a", "c"]


def test_manager_writes_the_litellm_address_onto_the_token(monkeypatch):
    supabase = TokenSupabase()
    manager = make_manager(supabase=supabase)
    monkeypatch.setattr(
        manager, "broker_settings", lambda: {"litellm_base_url": "https://l.example"}
    )
    assert manager.create_worker_token(WORKER) == "ow_worker_token"
    ((_, params),) = supabase.rpcs
    assert params["broker_settings"] == {"litellm_base_url": "https://l.example"}


def test_a_database_before_broker_settings_still_gives_a_worker_token(monkeypatch):
    class OldDatabase(TokenSupabase):
        def rpc(self, name, params):
            self.error = (
                APIError({"code": "PGRST202", "message": "function not found"})
                if "broker_settings" in params
                else None
            )
            return super().rpc(name, params)

    supabase = OldDatabase()
    manager = make_manager(supabase=supabase)
    monkeypatch.setattr(
        manager, "broker_settings", lambda: {"litellm_base_url": "https://l.example"}
    )
    # Not None: falling back to the organization's key would be a step backwards.
    assert manager.create_worker_token(WORKER) == "ow_worker_token"
    assert ["broker_settings" in params for _, params in supabase.rpcs] == [True, False]


# --- The database (real PostgreSQL) --------------------------------------------------


@pytest.fixture
def token_db(db):  # noqa: F811
    for migration in MIGRATIONS:
        db(migration.read_text())
    db(f"""INSERT INTO worker(id,organization_id,status) VALUES
            ('{WORKER}','{ORG}','starting'), ('other-org-worker','{OTHER}','starting');
        INSERT INTO organization_secrets(organization_id,name,value) VALUES
            ('{ORG}','WANDB_API_KEY','an-environment-variable'), ('{ORG}','HF_USER','me');
        INSERT INTO organization_secret_metadata(organization_id,name,last_four)
            VALUES ('{ORG}','HF_TOKEN','1234');
        INSERT INTO jobs(id,type,organization_id) VALUES ('job','script','{ORG}')""")
    return db


def create(db, worker=WORKER, hours=25, caller=None, error=False):  # noqa: F811
    result = db(
        f"SELECT token_id, token FROM create_worker_token('{ORG}','{worker}', now() + interval '{hours} hours')",
        claims=claims() if caller is None else caller,
        error=error,
    )
    return result if error else result.split("|")


def test_manager_creates_a_token_bound_to_the_worker(token_db):
    token_id, token = create(token_db)
    assert token.startswith("ow_")
    assert (
        token_db(f"""SELECT worker_id, name, revoked_at IS NULL,
                   expires_at BETWEEN now() + interval '24 hours' AND now() + interval '26 hours',
                   token_hash = encode(extensions.digest('{token}', 'sha256'), 'hex')
            FROM api_tokens WHERE id = '{token_id}'""")
        == (f"{WORKER}|worker {WORKER}|t|t|t")
    )


def test_worker_token_is_a_member_but_not_an_admin(token_db):
    token_id, _ = create(token_db)
    worker = claims(key=token_id)

    assert token_db(f"SELECT is_organization_admin('{ORG}')", claims=worker) == "f"
    assert token_db(f"SELECT is_organization_member('{ORG}')", claims=worker) == "t"
    # The organization's key, for comparison, is an admin.
    assert token_db(f"SELECT is_organization_admin('{ORG}')", claims=claims()) == "t"

    # What a worker does:
    assert token_db("SELECT id FROM jobs", claims=worker) == "job"
    token_db(
        f"UPDATE worker SET status='active', ping=now() WHERE id='{WORKER}'",
        claims=worker,
    )
    assert token_db(f"SELECT status FROM worker WHERE id='{WORKER}'") == "active"

    # What it must not see: settings, environment variables, which secrets exist.
    assert token_db("SELECT count(*) FROM organization_secrets", claims=worker) == "0"
    assert (
        token_db("SELECT count(*) FROM organization_secret_metadata", claims=worker)
        == "0"
    )
    assert "Only organization admins" in token_db(
        f"SELECT * FROM list_organization_secrets_masked('{ORG}')",
        claims=worker,
        error=True,
    )


@pytest.mark.parametrize(
    "call",
    [
        f"create_api_token('{ORG}','escape')",
        f"create_api_token_with_limit('{ORG}','escape',1)",
        f"create_worker_token('{ORG}','{WORKER}', now() + interval '7 days')",
        f"revoke_api_token('{KEY}')",
        f"set_spending_limit('{ORG}','{KEY}',0)",
        f"manage_organization_secret('{ORG}','HF_USER','x')",
        f"delete_organization_secret('{ORG}','HF_USER')",
        f"record_organization_secret('{ORG}','HF_TOKEN','0000')",
    ],
)
def test_worker_token_cannot_make_keys_or_change_settings(token_db, call):
    token_id, _ = create(token_db)
    error = token_db(f"SELECT {call}", claims=claims(key=token_id), error=True)
    assert "admin" in error.lower()


def test_worker_token_cannot_turn_itself_into_an_ordinary_key(token_db):
    token_id, _ = create(token_db)
    worker = claims(key=token_id)
    for statement in (
        f"UPDATE api_tokens SET worker_id = NULL WHERE id = '{token_id}'",
        f"UPDATE api_tokens SET expires_at = NULL WHERE id = '{token_id}'",
    ):
        token_db(statement, claims=worker)  # no row is visible for writing
    assert "permission denied" in token_db(
        "DELETE FROM spending_limits", claims=worker, error=True
    )
    token_db(
        f"""INSERT INTO api_tokens(organization_id,name,token_prefix,token_hash,created_by)
            VALUES ('{ORG}','escape','ow_x','hash','{USER}')""",
        claims=worker,
        error=True,
    )
    assert token_db(
        f"SELECT worker_id, expires_at IS NOT NULL FROM api_tokens WHERE id='{token_id}'"
    ) == (f"{WORKER}|t")
    assert token_db("SELECT count(*) FROM api_tokens WHERE name='escape'") == "0"


@pytest.mark.parametrize("ending", ["terminated", "deleted"])
def test_token_is_revoked_when_its_worker_ends(token_db, ending):
    token_id, _ = create(token_db)
    other_id = token_db(f"SELECT id FROM api_tokens WHERE id='{KEY}'")
    token_db(f"UPDATE worker SET status='active' WHERE id='{WORKER}'")
    assert (
        token_db(f"SELECT revoked_at IS NULL FROM api_tokens WHERE id='{token_id}'")
        == "t"
    )
    if ending == "terminated":
        # As the manager does it, with the organization's key.
        token_db(
            f"UPDATE worker SET status='terminated' WHERE id='{WORKER}'",
            claims=claims(),
        )
    else:
        token_db(f"DELETE FROM worker WHERE id='{WORKER}'", claims=claims())
    assert (
        token_db(f"SELECT revoked_at IS NULL FROM api_tokens WHERE id='{token_id}'")
        == "f"
    )
    # Other tokens of the organization are untouched.
    assert (
        token_db(f"SELECT revoked_at IS NULL FROM api_tokens WHERE id='{other_id}'")
        == "t"
    )


def test_a_worker_has_one_live_token(token_db):
    first, _ = create(token_db)
    second, _ = create(token_db)
    assert token_db(
        f"SELECT id FROM api_tokens WHERE worker_id='{WORKER}' AND revoked_at IS NULL"
    ) == (second)
    assert first != second


def test_only_the_organizations_admin_key_creates_worker_tokens(token_db):
    for caller in ({"sub": OTHER}, claims(org=OTHER), {}):
        assert "Only organization admins" in create(token_db, caller=caller, error=True)
    # A key with a spending limit is not an admin either.
    token_db(
        f"INSERT INTO spending_limits(organization_id,api_token_id,limit_usd) VALUES ('{ORG}','{KEY}',10)"
    )
    assert "Only organization admins" in create(token_db, error=True)
    assert "permission denied" in token_db(
        f"SET ROLE anon; SELECT create_worker_token('{ORG}','{WORKER}', now() + interval '1 hour')",
        error=True,
    )


def test_worker_tokens_are_for_live_workers_of_the_organization_and_expire(token_db):
    assert "No running worker" in create(
        token_db, worker="other-org-worker", error=True
    )
    assert "No running worker" in create(token_db, worker="no-such-worker", error=True)
    assert "must expire" in create(token_db, hours=24 * 9, error=True)
    assert "must expire" in create(token_db, hours=-1, error=True)
    assert "must expire" in token_db(
        f"SELECT create_worker_token('{ORG}','{WORKER}', NULL)",
        claims=claims(),
        error=True,
    )
    token_db(f"UPDATE worker SET status='terminated' WHERE id='{WORKER}'")
    assert "No running worker" in create(token_db, error=True)


def test_existing_keys_keep_their_rights(token_db):
    assert token_db(f"SELECT is_organization_admin('{ORG}')", claims=claims()) == "t"
    assert (
        token_db(f"SELECT is_organization_admin('{ORG}')", claims={"sub": USER}) == "t"
    )
    assert token_db("SELECT count(*) FROM organization_secrets", claims=claims()) == "2"
    assert (
        token_db("SELECT count(*) FROM api_tokens WHERE worker_id IS NOT NULL") == "0"
    )


def create_with_settings(db, settings, error=False):  # noqa: F811
    return db(
        f"SELECT token_id FROM create_worker_token('{ORG}','{WORKER}', now() + interval '25 hours', '{settings}'::jsonb)",
        claims=claims(),
        error=error,
    )


def test_manager_can_write_the_litellm_address_onto_a_worker_token(token_db):
    token_id = create_with_settings(
        token_db, '{"litellm_base_url": "https://litellm.example/proxy"}'
    )
    # The broker reads it with the worker token's own JWT.
    assert token_db(
        f"SELECT worker_id, broker_settings->>'litellm_base_url' FROM api_tokens WHERE id='{token_id}'",
        claims=claims(key=token_id),
    ) == (f"{WORKER}|https://litellm.example/proxy")
    # The worker cannot point the broker somewhere else.
    token_db(
        f"""UPDATE api_tokens SET broker_settings = '{{"litellm_base_url": "https://evil.example"}}' WHERE id='{token_id}'""",
        claims=claims(key=token_id),
    )
    assert (
        token_db(
            f"SELECT broker_settings->>'litellm_base_url' FROM api_tokens WHERE id='{token_id}'"
        )
        == "https://litellm.example/proxy"
    )


@pytest.mark.parametrize(
    "settings",
    [
        '{"litellm_base_url": "http://litellm.example"}',
        '{"litellm_base_url": "https://a b"}',
        '{"litellm_base_url": "https://ok.example", "extra": 1}',
        '{"something_else": "https://ok.example"}',
        '["https://ok.example"]',
        '"https://ok.example"',
    ],
)
def test_broker_settings_hold_one_https_address_and_nothing_else(token_db, settings):
    assert "broker_settings may only hold" in create_with_settings(
        token_db, settings, error=True
    )


def test_three_argument_call_of_older_managers_still_works(token_db):
    token_id, _ = create(token_db)
    assert (
        token_db(
            f"SELECT broker_settings IS NULL FROM api_tokens WHERE id='{token_id}'"
        )
        == "t"
    )


def test_litellm_address_is_a_setting_the_dashboard_may_show(token_db):
    token_db(
        f"SELECT manage_organization_secret('{ORG}','LITELLM_BASE_URL','https://litellm.example')",
        claims={"sub": USER},
    )
    assert "LITELLM_BASE_URL|https://litellm.example|" in token_db(
        f"SELECT name, plain_value, last_four FROM list_organization_secrets_masked('{ORG}')",
        claims={"sub": USER},
    )
