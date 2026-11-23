"""The credential broker: workers reach the Hugging Face Hub without the real token."""

import asyncio
import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from openweights.broker import HF_PREFIX, LITELLM_PREFIX, hf_stand_in_token
from openweights.broker.app import (
    OrganizationSecrets,
    UpstreamNotAllowed,
    checked_upstream,
    WorkerIdentity,
    WorkerTokenValidator,
    create_app,
    hub_url_via_broker,
    presented_worker_token,
)
from openweights.secret_store import InMemorySecretStore
from test_cost_tracking import ORG
from test_credential_safety import make_manager

WORKER_TOKEN = "ow_" + "ab" * 24
STAND_IN = hf_stand_in_token(WORKER_TOKEN)
REAL = "hf_REALorganizationTOKEN1234567890"
BROKER = "https://broker.example"


class FakeValidator:
    def __init__(
        self, identity=WorkerIdentity(ORG, "worker-1", "token-id"), error=None
    ):
        self.identity, self.error, self.seen = identity, error, []

    async def validate(self, worker_token):
        self.seen.append(worker_token)
        if self.error:
            raise self.error
        return self.identity if worker_token == WORKER_TOKEN else None


class Hub:
    """A stand-in for huggingface.co that records what reaches it."""

    def __init__(self, respond=None):
        self.requests = []
        self.respond = respond or (
            lambda request: httpx.Response(200, json={"ok": True})
        )

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        return self.respond(request)


LITELLM = "https://litellm.example/proxy"
LITELLM_KEY = "sk-REALlitellmKEY0987654321"


def make_broker(hub=None, validator=None, store_values=None, check_upstream=None):
    hub = hub or Hub()
    store = InMemorySecretStore()
    values = {"HF_TOKEN": REAL} if store_values is None else store_values
    for name, value in values.items():
        store.set(ORG, name, value)
    secrets = OrganizationSecrets(store)
    app = create_app(
        validator or FakeValidator(),
        secrets,
        upstream_client=httpx.AsyncClient(
            transport=httpx.MockTransport(hub), follow_redirects=False
        ),
        public_url=BROKER,
        # Tests do not resolve names; the check itself is tested on its own.
        check_upstream=check_upstream or (lambda url: url.rstrip("/")),
    )
    client = TestClient(app, follow_redirects=False)
    client.headers.pop("accept-encoding", None)
    return client, hub, store, secrets


AUTH = {"Authorization": f"Bearer {STAND_IN}"}


def test_worker_token_is_found_in_the_forms_clients_send_it():
    basic = (
        lambda text: "Basic " + base64.b64encode(text.encode()).decode()
    )  # noqa: E731
    assert presented_worker_token(f"Bearer {STAND_IN}") == WORKER_TOKEN
    assert presented_worker_token(f"Bearer {WORKER_TOKEN}") == WORKER_TOKEN
    assert presented_worker_token(basic(f"user:{STAND_IN}")) == WORKER_TOKEN
    assert presented_worker_token(basic(f"{STAND_IN}:x-oauth-basic")) == WORKER_TOKEN
    for other in (
        None,
        "",
        "Bearer hf_some_real_looking_token",
        "Bearer ow_short",
        "garbage",
    ):
        assert presented_worker_token(other) is None


def test_request_goes_to_the_hub_with_the_real_token_and_comes_back_without_it():
    client, hub, _, _ = make_broker()
    response = client.get(
        f"{HF_PREFIX}/api/models/org/model?full=true&x=a%20b",
        headers={
            **AUTH,
            "X-Forwarded-For": "1.2.3.4",
            "Cookie": "c=1",
            "User-Agent": "hub/1",
        },
    )
    assert response.status_code == 200 and response.json() == {"ok": True}

    (sent,) = hub.requests
    assert (
        str(sent.url) == "https://huggingface.co/api/models/org/model?full=true&x=a%20b"
    )
    assert sent.headers["authorization"] == f"Bearer {REAL}"
    assert sent.headers["user-agent"] == "hub/1"
    assert "cookie" not in sent.headers and "x-forwarded-for" not in sent.headers
    assert WORKER_TOKEN not in str(sent.headers)
    assert REAL not in response.text and REAL not in str(response.headers)


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer hf_a_real_token_someone_tried"},
        {"Authorization": "Bearer ow_" + "cd" * 24},
    ],
)
def test_without_a_valid_worker_token_nothing_reaches_the_hub(headers):
    client, hub, _, _ = make_broker()
    response = client.get(f"{HF_PREFIX}/api/whoami-v2", headers=headers)
    assert response.status_code == 401
    assert hub.requests == [] and REAL not in response.text


def test_failures_behind_the_broker_are_reported_without_details():
    client, hub, _, _ = make_broker(
        validator=FakeValidator(error=httpx.ConnectError("db down"))
    )
    assert client.get(f"{HF_PREFIX}/api/whoami-v2", headers=AUTH).status_code == 503
    assert hub.requests == []

    client, hub, _, _ = make_broker(store_values={})
    response = client.get(f"{HF_PREFIX}/api/whoami-v2", headers=AUTH)
    assert response.status_code == 403 and "no HF_TOKEN" in response.json()["error"]
    assert hub.requests == []

    def unreachable(request):
        raise httpx.ConnectError("no route to " + REAL)

    client, _, _, _ = make_broker(hub=Hub(unreachable))
    response = client.get(f"{HF_PREFIX}/api/whoami-v2", headers=AUTH)
    assert response.status_code == 502 and REAL not in response.text


def test_redirects_are_not_followed_and_only_hub_urls_are_pointed_back():
    locations = {
        "to-cdn": "https://cdn-lfs.hf.co/repos/ab/cd/file?X-Amz-Signature=sig",
        "relative": "/api/models/new/name",
        "absolute": "https://huggingface.co/new/name/resolve/main/config.json",
    }

    def respond(request):
        return httpx.Response(
            302, headers={"location": locations[request.url.path.split("/")[-1]]}
        )

    client, hub, _, _ = make_broker(hub=Hub(respond))
    got = {
        name: client.get(f"{HF_PREFIX}/x/{name}", headers=AUTH).headers["location"]
        for name in locations
    }
    # File bytes come from signed storage, directly and without any token.
    assert got["to-cdn"] == locations["to-cdn"]
    assert got["relative"] == "/hf/api/models/new/name"
    assert got["absolute"] == "/hf/new/name/resolve/main/config.json"
    assert len(hub.requests) == 3  # one each: the broker itself followed nothing


def test_head_keeps_the_size_and_the_file_headers_huggingface_hub_reads():
    def respond(request):
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(
            200,
            headers={
                "content-length": "123456",
                "etag": '"abc"',
                "x-repo-commit": "deadbeef",
                "x-linked-size": "999",
                "set-cookie": "session=1",
            },
        )

    client, _, _, _ = make_broker(hub=Hub(respond))
    response = client.head(
        f"{HF_PREFIX}/org/model/resolve/main/config.json",
        headers={**AUTH, "Accept-Encoding": "identity"},
    )
    assert response.headers["content-length"] == "123456"
    assert response.headers["x-repo-commit"] == "deadbeef"
    assert response.headers["x-linked-size"] == "999"
    assert "set-cookie" not in response.headers


def test_lfs_batch_links_and_xet_routes_lead_back_to_the_broker():
    batch = {
        "objects": [
            {
                "oid": "1",
                "actions": {
                    "upload": {"href": "https://s3.amazonaws.com/bucket/obj?sig=1"},
                    "verify": {
                        "href": "https://huggingface.co/org/model.git/info/lfs/objects/verify"
                    },
                },
            }
        ]
    }

    def respond(request):
        if request.url.path.endswith("/info/lfs/objects/batch"):
            return httpx.Response(200, json=batch)
        return httpx.Response(
            200,
            json=[],
            headers={
                "link": '<https://huggingface.co/api/models?cursor=2>; rel="next"',
                "x-xet-refresh-route": "/api/models/org/model/xet-read-token/main",
            },
        )

    client, hub, _, _ = make_broker(hub=Hub(respond))
    answer = client.post(
        f"{HF_PREFIX}/org/model.git/info/lfs/objects/batch",
        headers=AUTH,
        json={"objects": []},
    ).json()
    actions = answer["objects"][0]["actions"]
    assert actions["upload"]["href"] == "https://s3.amazonaws.com/bucket/obj?sig=1"
    assert (
        actions["verify"]["href"]
        == f"{BROKER}/hf/org/model.git/info/lfs/objects/verify"
    )

    listing = client.get(f"{HF_PREFIX}/api/models", headers=AUTH)
    assert listing.headers["link"] == f'<{BROKER}/hf/api/models?cursor=2>; rel="next"'
    assert listing.headers["x-xet-refresh-route"] == (
        f"{BROKER}/hf/api/models/org/model/xet-read-token/main"
    )


def test_bodies_are_passed_on_with_their_length_and_git_gets_basic_auth():
    client, hub, _, _ = make_broker()
    payload = b'{"summary": "upload"}\n' * 100
    client.post(
        f"{HF_PREFIX}/api/models/org/model/commit/main",
        headers={**AUTH, "Content-Type": "application/x-ndjson"},
        content=payload,
    )
    client.post(
        f"{HF_PREFIX}/org/model.git/git-receive-pack", headers=AUTH, content=b"pack"
    )
    commit, push = hub.requests
    assert commit.content == payload
    assert commit.headers["content-length"] == str(len(payload))
    assert commit.headers["content-type"] == "application/x-ndjson"
    assert (
        push.headers["authorization"]
        == "Basic " + base64.b64encode(f"user:{REAL}".encode()).decode()
    )


def test_a_replaced_token_is_picked_up_after_the_hub_refuses_the_old_one():
    def respond(request):
        ok = (
            request.headers["authorization"]
            == "Bearer hf_NEWorganizationTOKEN0987654321"
        )
        return httpx.Response(200 if ok else 401, json={})

    client, hub, store, _ = make_broker(hub=Hub(respond))
    store.set(ORG, "HF_TOKEN", REAL)
    assert client.get(f"{HF_PREFIX}/api/whoami-v2", headers=AUTH).status_code == 401
    store.set(ORG, "HF_TOKEN", "hf_NEWorganizationTOKEN0987654321")
    assert client.get(f"{HF_PREFIX}/api/whoami-v2", headers=AUTH).status_code == 200


def test_only_hub_urls_are_rewritten():
    hub = "https://huggingface.co"
    assert hub_url_via_broker("https://huggingface.co.evil.example/x", hub, "/hf") == (
        "https://huggingface.co.evil.example/x"
    )
    assert hub_url_via_broker("//evil.example/x", hub, "/hf") == "//evil.example/x"
    assert hub_url_via_broker(hub, hub, "/hf") == "/hf"
    assert hub_url_via_broker(hub + "?a=1", hub, "/hf") == "/hf?a=1"


def test_secrets_are_read_from_the_store_once_per_period():
    class Counting(InMemorySecretStore):
        reads = 0

        def get(self, *args):
            Counting.reads += 1
            return super().get(*args)

    now = [0.0]
    store = Counting()
    store.set(ORG, "HF_TOKEN", REAL)
    secrets = OrganizationSecrets(store, ttl=300, clock=lambda: now[0])

    async def read():
        return await secrets.get(ORG, "HF_TOKEN")

    assert asyncio.run(read()) == REAL and asyncio.run(read()) == REAL
    assert Counting.reads == 1
    now[0] = 301
    asyncio.run(read())
    assert Counting.reads == 2


# --- LiteLLM ------------------------------------------------------------------------


def litellm_broker(settings={"litellm_base_url": LITELLM}, **kwargs):  # noqa: B006
    validator = FakeValidator(WorkerIdentity(ORG, "worker-1", "token-id", settings))
    kwargs.setdefault("store_values", {"LITELLM_API_KEY": LITELLM_KEY})
    return make_broker(validator=validator, **kwargs)


def test_litellm_request_goes_to_the_organizations_proxy_with_its_key():
    def respond(request):
        return httpx.Response(
            200,
            content=b'data: {"delta": "Hel"}\n\ndata: {"delta": "lo"}\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream"},
        )

    client, hub, _, _ = litellm_broker(hub=Hub(respond))
    body = {"model": "m", "messages": [], "stream": True}
    response = client.post(
        f"{LITELLM_PREFIX}/v1/chat/completions?x=1",
        headers={
            "Authorization": f"Bearer {WORKER_TOKEN}",
            "Metadata-Flavor": "Google",
            "X-Stainless-Lang": "python",
        },
        json=body,
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.count("data:") == 3

    (sent,) = hub.requests
    assert str(sent.url) == f"{LITELLM}/v1/chat/completions?x=1"
    assert sent.headers["authorization"] == f"Bearer {LITELLM_KEY}"
    assert json.loads(sent.content) == body
    assert sent.headers["x-stainless-lang"] == "python"
    # Never forwarded: the header a cloud metadata server insists on.
    assert "metadata-flavor" not in sent.headers
    assert WORKER_TOKEN not in str(sent.headers)
    assert LITELLM_KEY not in response.text and LITELLM_KEY not in str(response.headers)


def test_litellm_accepts_the_worker_token_as_x_api_key_and_does_not_pass_it_on():
    client, hub, _, _ = litellm_broker()
    response = client.get(
        f"{LITELLM_PREFIX}/v1/models", headers={"x-api-key": WORKER_TOKEN}
    )
    assert response.status_code == 200
    (sent,) = hub.requests
    assert "x-api-key" not in sent.headers
    assert sent.headers["authorization"] == f"Bearer {LITELLM_KEY}"


@pytest.mark.parametrize(
    "settings, store_values, message",
    [
        ({}, {"LITELLM_API_KEY": LITELLM_KEY}, "no LITELLM_BASE_URL"),
        ({"litellm_base_url": LITELLM}, {}, "no LITELLM_API_KEY"),
        # Having a Hugging Face token does not open the LiteLLM route.
        ({"litellm_base_url": LITELLM}, {"HF_TOKEN": REAL}, "no LITELLM_API_KEY"),
    ],
)
def test_litellm_needs_both_the_key_and_the_address(settings, store_values, message):
    client, hub, _, _ = litellm_broker(settings=settings, store_values=store_values)
    response = client.get(
        f"{LITELLM_PREFIX}/v1/models",
        headers={"Authorization": f"Bearer {WORKER_TOKEN}"},
    )
    assert response.status_code == 403 and message in response.json()["error"]
    assert hub.requests == []


def test_litellm_refuses_an_address_that_leads_inside_and_sends_nothing():
    def refuse(url):
        raise UpstreamNotAllowed("the address must be a public host")

    client, hub, _, _ = litellm_broker(check_upstream=refuse)
    response = client.post(
        f"{LITELLM_PREFIX}/v1/chat/completions",
        headers={"Authorization": f"Bearer {WORKER_TOKEN}"},
        json={},
    )
    assert response.status_code == 403 and "not usable" in response.json()["error"]
    assert hub.requests == []


def test_litellm_does_not_hand_redirects_or_cookies_to_the_client():
    def respond(request):
        return httpx.Response(
            307, headers={"location": "https://elsewhere.example/", "set-cookie": "a=1"}
        )

    client, hub, _, _ = litellm_broker(hub=Hub(respond))
    response = client.get(
        f"{LITELLM_PREFIX}/v1/models",
        headers={"Authorization": f"Bearer {WORKER_TOKEN}"},
    )
    assert "location" not in response.headers and "set-cookie" not in response.headers
    assert len(hub.requests) == 1  # and the broker did not follow it


def test_without_a_valid_worker_token_litellm_is_closed():
    client, hub, _, _ = litellm_broker()
    for headers in (
        {},
        {"Authorization": f"Bearer {LITELLM_KEY}"},
        {"x-api-key": "ow_short"},
    ):
        assert (
            client.get(f"{LITELLM_PREFIX}/v1/models", headers=headers).status_code
            == 401
        )
    assert hub.requests == []


def resolver(*addresses):
    return lambda host, port: [
        (None, None, None, "", (address, port)) for address in addresses
    ]


def test_upstream_must_be_a_public_https_host():
    public = resolver("93.184.216.34")
    assert checked_upstream("https://litellm.example/proxy/", public) == (
        "https://litellm.example/proxy"
    )
    for url in (
        "http://litellm.example",  # not https
        "https://user:pw@litellm.example",  # credentials in the address
        "https://169.254.169.254/computeMetadata/v1/",  # the metadata server
        "https://10.0.0.5",
        "https://[::1]:8443",
        "https://localhost",
        "https://metadata.google.internal",
        "https://service.local",
        "ftp://litellm.example",
        "https://",
    ):
        with pytest.raises(UpstreamNotAllowed):
            checked_upstream(url, public)
    # A public-looking name that resolves inside is refused too, even if only one
    # of its addresses does.
    for inside in (
        "169.254.169.254",
        "10.1.2.3",
        "127.0.0.1",
        "192.168.1.1",
        "::1",
        "fd00::1",
    ):
        with pytest.raises(UpstreamNotAllowed):
            checked_upstream(
                "https://sneaky.example", resolver("93.184.216.34", inside)
            )

    def unresolvable(host, port):
        raise OSError("no such host")

    with pytest.raises(UpstreamNotAllowed):
        checked_upstream("https://nowhere.example", unresolvable)


def test_health_answers_on_both_paths():
    client, _, _, _ = make_broker()
    assert client.get("/health").json() == {"ok": True}
    assert client.get("/healthz").json() == {"ok": True}


# --- Checking a worker token with the database -------------------------------------


def jwt_for(token_id, org=ORG):
    payload = (
        base64.urlsafe_b64encode(
            json.dumps({"api_token_id": token_id, "organization_id": org}).encode()
        )
        .decode()
        .rstrip("=")
    )
    return f"header.{payload}.signature"


class Database:
    """The two PostgREST calls the validator makes."""

    def __init__(self):
        self.tokens = {WORKER_TOKEN: ("worker-token-id", "worker-1")}
        self.personal = {"ow_" + "ee" * 24: ("personal-token-id", None)}
        self.exchanges = 0
        self.status = None
        self.settings = {"litellm_base_url": LITELLM}
        self.has_settings_column = True

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["apikey"] == "anon-key"
        if self.status:
            return httpx.Response(self.status, json={"message": "unavailable"})
        if request.url.path.endswith("/rpc/exchange_api_token_for_jwt"):
            self.exchanges += 1
            token = json.loads(request.content)["api_token"]
            known = {**self.tokens, **self.personal}.get(token)
            if not known:
                return httpx.Response(
                    400, json={"message": "API token has been revoked"}
                )
            return httpx.Response(200, json=jwt_for(known[0]))
        assert request.url.path.endswith("/api_tokens")
        token_id = request.url.params["id"].removeprefix("eq.")
        assert request.headers["authorization"] == f"Bearer {jwt_for(token_id)}"
        worker = {v[0]: v[1] for v in {**self.tokens, **self.personal}.values()}[
            token_id
        ]
        if "broker_settings" in request.url.params["select"]:
            if not self.has_settings_column:
                return httpx.Response(400, json={"message": "column does not exist"})
            return httpx.Response(
                200, json=[{"worker_id": worker, "broker_settings": self.settings}]
            )
        return httpx.Response(200, json=[{"worker_id": worker}])


def make_validator(database, now):
    return WorkerTokenValidator(
        "https://db.example",
        "anon-key",
        client=httpx.AsyncClient(transport=httpx.MockTransport(database)),
        ttl=60,
        clock=lambda: now[0],
    )


def test_validator_accepts_a_live_worker_token_and_learns_whose_it_is():
    validator = make_validator(Database(), [0.0])
    identity = asyncio.run(validator.validate(WORKER_TOKEN))
    assert (identity.organization_id, identity.worker_id) == (ORG, "worker-1")
    assert identity.settings == {"litellm_base_url": LITELLM}


def test_validator_works_with_a_database_from_before_broker_settings():
    database = Database()
    database.has_settings_column = False
    identity = asyncio.run(make_validator(database, [0.0]).validate(WORKER_TOKEN))
    assert identity.worker_id == "worker-1" and identity.settings == {}


def test_validator_refuses_personal_keys_and_unknown_tokens():
    database = Database()
    validator = make_validator(database, [0.0])
    # A valid API key of a person is not a way to use the organization's secrets here.
    assert asyncio.run(validator.validate("ow_" + "ee" * 24)) is None
    assert asyncio.run(validator.validate("ow_" + "00" * 24)) is None


def test_a_revoked_token_stops_working_within_the_cache_period():
    database, now = Database(), [0.0]
    validator = make_validator(database, now)
    assert asyncio.run(validator.validate(WORKER_TOKEN)) is not None
    del database.tokens[WORKER_TOKEN]  # the worker was terminated

    now[0] = 59
    assert asyncio.run(validator.validate(WORKER_TOKEN)) is not None  # still cached
    assert database.exchanges == 1
    now[0] = 61
    assert asyncio.run(validator.validate(WORKER_TOKEN)) is None


def test_a_database_outage_is_an_error_not_a_refusal():
    database = Database()
    database.status = 503
    validator = make_validator(database, [0.0])
    with pytest.raises(httpx.HTTPError):
        asyncio.run(validator.validate(WORKER_TOKEN))
    # Nothing was cached: it works again as soon as the database does.
    database.status = None
    assert asyncio.run(validator.validate(WORKER_TOKEN)) is not None


# --- The cluster manager points workers at the broker ---------------------------------


def brokered_manager(monkeypatch, broker_url=BROKER + "/", from_store=("HF_TOKEN",)):
    if broker_url:
        monkeypatch.setenv("OW_BROKER_URL", broker_url)
    else:
        monkeypatch.delenv("OW_BROKER_URL", raising=False)
    manager = make_manager(
        secrets={
            "RUNPOD_API_KEY": "rp",
            "HF_TOKEN": REAL,
            "OPENWEIGHTS_API_KEY": "ow",
            "HF_USER": "me",
        }
    )
    manager.secrets_from_store = frozenset(from_store)
    return manager


def test_worker_gets_a_stand_in_and_the_brokers_address_not_the_real_token(monkeypatch):
    manager = brokered_manager(monkeypatch)
    env = manager.worker_env_for(WORKER_TOKEN)
    assert env["HF_TOKEN"] == STAND_IN
    assert env["HF_ENDPOINT"] == f"{BROKER}/hf"
    assert REAL not in env.values()
    assert env["HF_USER"] == "me"  # environment variables are untouched


@pytest.mark.parametrize(
    "broker_url, from_store",
    [
        (None, ("HF_TOKEN",)),  # a deployment without a broker
        (BROKER, ()),  # the token is not in the store: nothing to broker
    ],
)
def test_without_broker_or_store_the_worker_keeps_the_real_token(
    monkeypatch, broker_url, from_store
):
    manager = brokered_manager(monkeypatch, broker_url, from_store)
    env = manager.worker_env_for(WORKER_TOKEN)
    assert env["HF_TOKEN"] == REAL and "HF_ENDPOINT" not in env


def litellm_manager(monkeypatch, url="https://litellm.example/proxy/", in_store=True):
    monkeypatch.setenv("OW_BROKER_URL", BROKER)
    manager = make_manager(
        secrets={
            "RUNPOD_API_KEY": "rp",
            "OPENWEIGHTS_API_KEY": "ow",
            "LITELLM_API_KEY": LITELLM_KEY,
            **({"LITELLM_BASE_URL": url} if url else {}),
        }
    )
    manager.secrets_from_store = frozenset({"LITELLM_API_KEY"} if in_store else ())
    return manager


def test_worker_reaches_litellm_through_the_broker_with_its_own_token(monkeypatch):
    manager = litellm_manager(monkeypatch)
    assert manager.broker_settings() == {"litellm_base_url": LITELLM}
    env = manager.worker_env_for(WORKER_TOKEN)
    assert env["LITELLM_API_KEY"] == WORKER_TOKEN
    assert env["LITELLM_BASE_URL"] == f"{BROKER}/litellm"
    assert LITELLM_KEY not in env.values()


@pytest.mark.parametrize(
    "url, in_store",
    [(None, True), ("http://litellm.example", True), (LITELLM, False)],
)
def test_without_address_or_stored_key_the_worker_gets_no_litellm_key(
    monkeypatch, url, in_store
):
    manager = litellm_manager(monkeypatch, url=url, in_store=in_store)
    assert manager.broker_settings() is None
    env = manager.worker_env_for(WORKER_TOKEN)
    assert "LITELLM_API_KEY" not in env and LITELLM_KEY not in env.values()
    assert env.get("LITELLM_BASE_URL") != f"{BROKER}/litellm"


def test_no_broker_settings_on_a_deployment_without_a_broker(monkeypatch):
    manager = litellm_manager(monkeypatch)
    monkeypatch.delenv("OW_BROKER_URL")
    assert manager.broker_settings() is None
