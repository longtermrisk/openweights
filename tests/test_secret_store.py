"""Secrets are kept in a secret store, environment variables in the database.

The database tests need the disposable PostgreSQL container of test_cost_tracking.py
(OW_COST_TEST_CONTAINER=ow-cost-tests) and are skipped without it.
"""

import base64
import json
from types import SimpleNamespace

import httpx
import pytest

from openweights import secret_store
from openweights.cluster.supervisor import ManagerSupervisor
from openweights.secret_store import (
    GoogleSecretManagerStore,
    InMemorySecretStore,
    ManagedSecretReader,
    SecretStoreError,
    secret_store_from_env,
)
from openweights.secret_store_migration import migrate
from test_cost_tracking import KEY, ORG, OTHER, ROOT, USER, claims, db  # noqa: F401
from test_credential_safety import (  # noqa: F401
    FakeSupabase,
    database_module,
    make_database,
    make_manager,
)
from test_dashboard_jobs import dashboard  # noqa: F401

SECRET = "hf_abcdefghijklmnopqrstuvwxyz1234"
MIGRATIONS = [
    ROOT / "supabase/migrations/20261120000000_credential_safety.sql",
    ROOT / "supabase/migrations/20261121000000_secret_metadata.sql",
]


# --- The Google Secret Manager store ---------------------------------------------


class FakeSecretManager:
    """Just enough of the Secret Manager REST API, in memory."""

    def __init__(self):
        self.secrets = {}  # secret path -> list of {"name", "state", "data"}
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-token"
        path = request.url.path.removeprefix("/v1/")
        self.requests.append((request.method, path))
        body = json.loads(request.content) if request.content else {}
        if request.method == "POST" and path.endswith("/secrets"):
            secret = f"{path}/{request.url.params['secretId']}"
            if secret in self.secrets:
                return httpx.Response(409, json={"error": "exists"})
            self.secrets[secret] = []
            return httpx.Response(200, json={"name": secret})
        if path.endswith(":addVersion"):
            secret = path.removesuffix(":addVersion")
            versions = self.secrets[secret]
            version = {
                "name": f"{secret}/versions/{len(versions) + 1}",
                "state": "ENABLED",
                "data": body["payload"]["data"],
            }
            versions.append(version)
            return httpx.Response(200, json={"name": version["name"]})
        if path.endswith("/versions/latest:access"):
            versions = [
                v
                for v in self.secrets.get(path.split("/versions/")[0], [])
                if v["state"] == "ENABLED"
            ]
            if not versions:
                return httpx.Response(404, json={"error": "not found, value hidden"})
            return httpx.Response(200, json={"payload": {"data": versions[-1]["data"]}})
        if request.method == "GET" and path.endswith("/versions"):
            versions = self.secrets[path.removesuffix("/versions")]
            return httpx.Response(
                200,
                json={
                    "versions": [
                        {"name": v["name"]} for v in versions if v["state"] == "ENABLED"
                    ]
                },
            )
        if path.endswith(":destroy"):
            secret = path.split("/versions/")[0]
            for version in self.secrets[secret]:
                if version["name"] == path.removesuffix(":destroy"):
                    version["state"], version["data"] = "DESTROYED", None
            return httpx.Response(200, json={})
        if request.method == "DELETE":
            return httpx.Response(
                200 if self.secrets.pop(path, None) is not None else 404
            )
        return httpx.Response(500, text="unexpected request")


@pytest.fixture
def gcp():
    fake = FakeSecretManager()
    store = GoogleSecretManagerStore(
        "my-project-123",
        client=httpx.Client(transport=httpx.MockTransport(fake)),
        token_source=lambda _client: ("test-token", 3600),
    )
    return store, fake


def test_gcp_store_round_trip_and_one_live_version(gcp):
    store, fake = gcp
    assert store.get(ORG, "HF_TOKEN") is None
    store.set(ORG, "HF_TOKEN", SECRET)
    assert store.get(ORG, "HF_TOKEN") == SECRET
    store.set(ORG, "HF_TOKEN", "hf_replacement_value_9876")
    assert store.get(ORG, "HF_TOKEN") == "hf_replacement_value_9876"

    # The earlier value is destroyed, not merely superseded.
    (versions,) = fake.secrets.values()
    assert [v["state"] for v in versions] == ["DESTROYED", "ENABLED"]
    assert SECRET not in json.dumps(fake.secrets)
    assert base64.b64decode(versions[1]["data"]).decode() == "hf_replacement_value_9876"

    store.delete(ORG, "HF_TOKEN")
    store.delete(ORG, "HF_TOKEN")  # deleting what is not there is fine
    assert store.get(ORG, "HF_TOKEN") is None


def test_gcp_store_keeps_organizations_and_names_apart(gcp):
    store, fake = gcp
    store.set(ORG, "HF_TOKEN", "a" * 20)
    store.set(OTHER, "HF_TOKEN", "b" * 20)
    store.set(ORG, "RUNPOD_API_KEY", "c" * 20)
    assert len(fake.secrets) == 3
    assert store.get(OTHER, "HF_TOKEN") == "b" * 20
    assert f"projects/my-project-123/secrets/ow-org-{ORG}-HF_TOKEN" in fake.secrets


def test_gcp_store_takes_only_managed_secrets_of_real_organizations(gcp):
    store, fake = gcp
    with pytest.raises(ValueError, match="not a managed secret"):
        store.set(ORG, "WANDB_API_KEY", "x")
    with pytest.raises(ValueError, match="UUID"):
        store.get("../../other-project/secrets/x", "HF_TOKEN")
    with pytest.raises(ValueError, match="project id"):
        GoogleSecretManagerStore("Not A Project")
    assert fake.requests == []


def test_gcp_store_errors_do_not_quote_the_store(gcp):
    store, _fake = gcp
    store._client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(403, text="denied: " + SECRET)
        )
    )
    with pytest.raises(SecretStoreError) as error:
        store.get(ORG, "HF_TOKEN")
    assert "403" in str(error.value) and SECRET not in str(error.value)


def test_store_is_selected_by_environment():
    assert secret_store_from_env({}) is None
    store = secret_store_from_env(
        {"OW_SECRET_STORE": "gcp-secret-manager://my-project-123"}
    )
    assert isinstance(store, GoogleSecretManagerStore)
    assert store.project == "my-project-123"
    with pytest.raises(ValueError, match="must look like"):
        secret_store_from_env({"OW_SECRET_STORE": "vault://x"})


# --- Reading on the manager side --------------------------------------------------


class CountingStore(InMemorySecretStore):
    def __init__(self):
        super().__init__()
        self.reads = 0

    def get(self, organization_id, name):
        self.reads += 1
        return super().get(organization_id, name)


def metadata(*rows):
    return FakeSupabase(
        rows={
            "organization_secret_metadata": [
                {"name": name, "updated_at": updated_at} for name, updated_at in rows
            ]
        }
    )


def test_reader_asks_the_store_only_after_a_change():
    store = CountingStore()
    store.set(ORG, "HF_TOKEN", "first-value-000001")
    reader = ManagedSecretReader(store)

    assert reader.read(metadata(("HF_TOKEN", "t1")), ORG) == {
        "HF_TOKEN": "first-value-000001"
    }
    reader.read(metadata(("HF_TOKEN", "t1")), ORG)
    assert store.reads == 1

    store.set(ORG, "HF_TOKEN", "second-value-00002")
    assert reader.read(metadata(("HF_TOKEN", "t2")), ORG) == {
        "HF_TOKEN": "second-value-00002"
    }
    assert store.reads == 2
    # A deleted secret is gone on the next read, not served from the cache.
    assert reader.read(metadata(), ORG) == {}


def test_reader_skips_what_is_recorded_but_missing_and_what_is_not_a_secret():
    reader = ManagedSecretReader(InMemorySecretStore())
    assert reader.read(metadata(("HF_TOKEN", "t1"), ("WANDB_API_KEY", "t1")), ORG) == {}


def test_supervisor_prefers_the_store_over_a_leftover_row():
    store = InMemorySecretStore()
    store.set(ORG, "HF_TOKEN", "from-the-store-0001")
    supervisor = ManagerSupervisor.__new__(ManagerSupervisor)
    supervisor.managed_secrets = ManagedSecretReader(store)
    supervisor.supabase = FakeSupabase(
        rows={
            "organization_secrets": [
                {"name": "HF_TOKEN", "value": "stale-row"},
                {"name": "HF_USER", "value": "me"},
            ],
            "organization_secret_metadata": [{"name": "HF_TOKEN", "updated_at": "t"}],
        }
    )
    assert supervisor.get_org_secrets(ORG) == {
        "HF_TOKEN": "from-the-store-0001",
        "HF_USER": "me",
    }
    supervisor.managed_secrets = None  # a deployment without a store
    assert supervisor.get_org_secrets(ORG)["HF_TOKEN"] == "stale-row"


def test_workers_do_not_get_secrets_they_have_no_use_for():
    manager = make_manager(
        secrets={
            "RUNPOD_API_KEY": "rp",
            "HF_TOKEN": "hf",
            "OPENWEIGHTS_API_KEY": "ow",
            "LITELLM_API_KEY": "sk-litellm",
            "WANDB_API_KEY": "an-env-variable",
        }
    )
    env = manager.worker_env_for("nielsrolf/ow-unsloth:v0.16.0")
    assert "LITELLM_API_KEY" not in env and "RUNPOD_API_KEY" not in env
    # Environment variables are passed on as before.
    assert env["WANDB_API_KEY"] == "an-env-variable"
    # Still passed until workers have tokens of their own.
    assert env["HF_TOKEN"] == "hf" and env["OPENWEIGHTS_API_KEY"] == "ow"
    assert "LITELLM_API_KEY" not in manager.worker_env_for(
        "nielsrolf/ow-unsloth:v0.15.0"
    )


# --- The dashboard backend ---------------------------------------------------------


class RpcSupabase(FakeSupabase):
    """Like FakeSupabase, with a chosen answer per RPC."""

    def __init__(self, answers=None, **kwargs):
        super().__init__(**kwargs)
        self.answers = answers or {}

    def rpc(self, name, params):
        self.rpcs.append((name, params))
        data = self.answers.get(name, True)
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=data))


def test_dashboard_writes_a_secret_to_the_store_and_only_a_hint_to_the_database(
    database_module, monkeypatch
):
    monkeypatch.setattr(
        database_module, "validate_organization_secrets", lambda secrets: (True, "")
    )
    client = RpcSupabase()
    database = make_database(database_module, client)
    database.secret_store = InMemorySecretStore()

    database.set_organization_secret(ORG, "HF_TOKEN", SECRET)

    assert database.secret_store.get(ORG, "HF_TOKEN") == SECRET
    assert client.rpcs == [
        ("is_organization_admin", {"org_id": ORG}),
        (
            "record_organization_secret",
            {"org_id": ORG, "secret_name": "HF_TOKEN", "secret_last_four": "1234"},
        ),
    ]
    assert SECRET not in json.dumps(client.rpcs)


def test_dashboard_keeps_environment_variables_in_the_database(database_module):
    client = RpcSupabase()
    database = make_database(database_module, client)
    database.secret_store = InMemorySecretStore()

    database.set_organization_secret(ORG, "WANDB_PROJECT", "my-project")

    assert database.secret_store.values == {}
    assert [name for name, _ in client.rpcs] == ["manage_organization_secret"]


def test_dashboard_asks_the_database_who_is_admin_before_touching_the_store(
    database_module,
):
    client = RpcSupabase(answers={"is_organization_admin": False})
    database = make_database(database_module, client)
    database.secret_store = InMemorySecretStore()
    database.secret_store.set(ORG, "LITELLM_API_KEY", "sk-existing-value-01")

    with pytest.raises(ValueError, match="Only organization admins"):
        database.set_organization_secret(ORG, "LITELLM_API_KEY", "sk-attacker-value-99")
    with pytest.raises(ValueError, match="Only organization admins"):
        database.delete_organization_secret(ORG, "LITELLM_API_KEY")

    assert database.secret_store.get(ORG, "LITELLM_API_KEY") == "sk-existing-value-01"
    assert [name for name, _ in client.rpcs] == ["is_organization_admin"] * 2


def test_dashboard_deletes_a_secret_from_the_store(database_module):
    client = RpcSupabase()
    database = make_database(database_module, client)
    database.secret_store = InMemorySecretStore()
    database.secret_store.set(ORG, "LITELLM_API_KEY", "sk-existing-value-01")

    database.delete_organization_secret(ORG, "LITELLM_API_KEY")

    assert database.secret_store.values == {}
    assert [name for name, _ in client.rpcs] == [
        "is_organization_admin",
        "forget_organization_secret",
    ]


def test_dashboard_delete_falls_back_to_a_row_from_before_the_store(database_module):
    client = RpcSupabase(answers={"forget_organization_secret": False})
    database = make_database(database_module, client)
    database.secret_store = InMemorySecretStore()
    database.delete_organization_secret(ORG, "LITELLM_API_KEY")
    assert client.rpcs[-1][0] == "delete_organization_secret"


def test_dashboard_reports_store_failures_without_details(database_module):
    class BrokenStore(InMemorySecretStore):
        def set(self, *_args):
            raise SecretStoreError("Secret Manager POST x failed with status 403")

    client = RpcSupabase()
    database = make_database(database_module, client)
    database.secret_store = BrokenStore()
    with pytest.raises(ValueError, match="status 403"):
        database.set_organization_secret(ORG, "LITELLM_API_KEY", "sk-some-value-00001")
    # Nothing is recorded for a secret that was not stored.
    assert [name for name, _ in client.rpcs] == ["is_organization_admin"]


def test_listing_says_which_entries_are_secrets(database_module):
    client = RpcSupabase(
        answers={
            "list_organization_secrets_masked": [
                {"name": "HF_TOKEN", "plain_value": None, "last_four": "1234"},
                {"name": "HF_USER", "plain_value": "me", "last_four": None},
                {"name": "OPENAI_API_KEY", "plain_value": None, "last_four": "abcd"},
            ]
        }
    )
    database = make_database(database_module, client)
    listing = database.list_organization_secrets(ORG)
    assert {entry.name: entry.kind for entry in listing} == {
        "HF_TOKEN": "secret",
        "HF_USER": "env",
        "OPENAI_API_KEY": "env",
    }


# --- Moving existing rows into the store -------------------------------------------


class MigrationSupabase:
    def __init__(self, rows):
        self.tables = {"organization_secrets": rows, "organization_secret_metadata": []}

    def table(self, name):
        return MigrationQuery(self, name)


class MigrationQuery:
    def __init__(self, supabase, name):
        self.rows, self.filters, self.action = supabase.tables[name], [], "select"

    def select(self, *_columns):
        return self

    def in_(self, column, values):
        self.filters.append(lambda row: row[column] in values)
        return self

    def eq(self, column, value):
        self.filters.append(lambda row: row[column] == value)
        return self

    def upsert(self, row):
        self.action, self.row = "upsert", row
        return self

    def delete(self):
        self.action = "delete"
        return self

    def execute(self):
        matching = [row for row in self.rows if all(f(row) for f in self.filters)]
        if self.action == "upsert":
            self.rows.append(self.row)
        elif self.action == "delete":
            for row in matching:
                self.rows.remove(row)
        return SimpleNamespace(data=matching)


def stored_rows():
    return [
        {"organization_id": ORG, "name": "HF_TOKEN", "value": SECRET},
        {"organization_id": ORG, "name": "HF_USER", "value": "me"},
        {"organization_id": ORG, "name": "OPENAI_API_KEY", "value": "sk-env-var"},
        {
            "organization_id": OTHER,
            "name": "RUNPOD_API_KEY",
            "value": "rpa_" + "x" * 20,
        },
    ]


def test_migration_dry_run_changes_nothing_and_prints_no_values():
    supabase, store, lines = MigrationSupabase(stored_rows()), InMemorySecretStore(), []
    assert migrate(supabase, store, apply=False, out=lines.append) == 2
    assert store.values == {} and len(supabase.tables["organization_secrets"]) == 4
    assert SECRET not in "\n".join(lines)


def test_migration_moves_secrets_and_leaves_environment_variables():
    supabase, store, lines = MigrationSupabase(stored_rows()), InMemorySecretStore(), []
    assert migrate(supabase, store, apply=True, out=lines.append) == 2

    assert store.get(ORG, "HF_TOKEN") == SECRET
    assert store.get(OTHER, "RUNPOD_API_KEY") == "rpa_" + "x" * 20
    assert sorted(r["name"] for r in supabase.tables["organization_secrets"]) == [
        "HF_USER",
        "OPENAI_API_KEY",
    ]
    recorded = supabase.tables["organization_secret_metadata"]
    assert {(r["organization_id"], r["name"], r["last_four"]) for r in recorded} == {
        (ORG, "HF_TOKEN", "1234"),
        (OTHER, "RUNPOD_API_KEY", "xxxx"),
    }
    assert SECRET not in "\n".join(lines) and SECRET not in json.dumps(recorded)
    # Running it again finds nothing left to move.
    assert migrate(supabase, store, apply=True, out=lines.append) == 0


def test_migration_keeps_the_row_when_the_store_does_not_hold_the_value():
    class ForgetfulStore(InMemorySecretStore):
        def get(self, *_args):
            return None

    supabase = MigrationSupabase(stored_rows())
    with pytest.raises(RuntimeError, match="left in place"):
        migrate(supabase, ForgetfulStore(), apply=True, out=lambda _line: None)
    assert len(supabase.tables["organization_secrets"]) == 4


# --- The database (real PostgreSQL) --------------------------------------------------


@pytest.fixture
def metadata_db(db):  # noqa: F811
    for migration in MIGRATIONS:
        db(migration.read_text())
    db(f"""INSERT INTO organization_secrets(organization_id,name,value) VALUES
        ('{ORG}','HF_TOKEN','{SECRET}'), ('{ORG}','HF_ORG','my-org'),
        ('{ORG}','OPENAI_API_KEY','sk-an-environment-variable')""")
    return db


LISTING = f"SELECT name, plain_value, last_four FROM list_organization_secrets_masked('{ORG}')"


def test_recording_a_stored_secret_removes_its_row(metadata_db):
    user = {"sub": USER}
    metadata_db(
        f"SELECT record_organization_secret('{ORG}','HF_TOKEN','9876')", claims=user
    )
    assert (
        metadata_db("SELECT name FROM organization_secrets ORDER BY name")
        == "HF_ORG\nOPENAI_API_KEY"
    )
    # The dashboard lists it like before, from what was recorded.
    assert metadata_db(LISTING, claims=user) == (
        "HF_ORG|my-org|\nHF_TOKEN||9876\nOPENAI_API_KEY||able"
    )
    assert metadata_db(
        "SELECT name, last_four FROM organization_secret_metadata", claims=user
    ) == ("HF_TOKEN|9876")


def test_only_secrets_can_be_recorded_and_only_by_admins(metadata_db):
    error = metadata_db(
        f"SELECT record_organization_secret('{ORG}','OPENAI_API_KEY','abcd')",
        claims={"sub": USER},
        error=True,
    )
    assert "not a secret" in error
    for caller in ({"sub": OTHER}, claims(org=OTHER), {}):
        for call in (
            f"record_organization_secret('{ORG}','HF_TOKEN','9876')",
            f"forget_organization_secret('{ORG}','HF_TOKEN')",
        ):
            assert "Only organization admins" in metadata_db(
                f"SELECT {call}", claims=caller, error=True
            )
    assert "permission denied" in metadata_db(
        f"SET ROLE anon; SELECT record_organization_secret('{ORG}','HF_TOKEN','9876')",
        error=True,
    )


def test_metadata_is_read_by_admins_and_the_manager_key_and_written_by_nobody(
    metadata_db,
):
    metadata_db(
        f"SELECT record_organization_secret('{ORG}','HF_TOKEN','9876')",
        claims={"sub": USER},
    )
    count = "SELECT count(*) FROM organization_secret_metadata"
    assert metadata_db(count, claims={"sub": USER}) == "1"
    assert metadata_db(count, claims=claims()) == "1"  # the cluster manager's key
    assert metadata_db(count, claims=claims(org=OTHER)) == "0"
    assert metadata_db(count, claims={"sub": OTHER}) == "0"
    for statement in (
        "UPDATE organization_secret_metadata SET last_four='0000'",
        "DELETE FROM organization_secret_metadata",
        f"INSERT INTO organization_secret_metadata(organization_id,name) VALUES ('{ORG}','RUNPOD_API_KEY')",
    ):
        assert "permission denied" in metadata_db(
            statement, claims={"sub": USER}, error=True
        )
    # A key with a spending limit is not an admin.
    metadata_db(
        f"INSERT INTO spending_limits(organization_id,api_token_id,limit_usd) VALUES ('{ORG}','{KEY}',10)"
    )
    assert metadata_db(count, claims=claims()) == "0"


def test_a_stored_secret_cannot_get_a_copy_in_the_table(metadata_db):
    metadata_db(
        f"SELECT record_organization_secret('{ORG}','HF_TOKEN','9876')",
        claims={"sub": USER},
    )
    # `ow env import` with an API key, and the dashboard's RPC:
    for statement, caller in (
        (
            f"INSERT INTO organization_secrets(organization_id,name,value) VALUES ('{ORG}','HF_TOKEN','copy')",
            claims(),
        ),
        (
            f"SELECT manage_organization_secret('{ORG}','HF_TOKEN','copy')",
            {"sub": USER},
        ),
    ):
        assert "kept in the secret store" in metadata_db(
            statement, claims=caller, error=True
        )
    # Environment variables are unaffected.
    metadata_db(
        f"SELECT manage_organization_secret('{ORG}','WANDB_PROJECT','p')",
        claims={"sub": USER},
    )
    # Once forgotten, the name is free again (a deployment without a store).
    assert (
        metadata_db(
            f"SELECT forget_organization_secret('{ORG}','HF_TOKEN')",
            claims={"sub": USER},
        )
        == "t"
    )
    metadata_db(
        f"SELECT manage_organization_secret('{ORG}','HF_TOKEN','back-in-the-table')",
        claims={"sub": USER},
    )


def test_metadata_never_holds_more_than_four_characters(metadata_db):
    assert "violates check constraint" in metadata_db(
        f"SELECT record_organization_secret('{ORG}','HF_TOKEN','{SECRET}')",
        claims={"sub": USER},
        error=True,
    )
