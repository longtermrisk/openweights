"""The broker's HTTP service. What it is for: see the package docstring."""

import base64
import hashlib
import json
import logging
import re
import time
import ipaddress
import socket
from typing import Callable, Dict, Optional, Tuple
from urllib.parse import urlsplit

import anyio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("openweights.broker")

from openweights.broker import HF_PREFIX, HF_TOKEN_PREFIX, LITELLM_PREFIX

HF_UPSTREAM = "https://huggingface.co"
WORKER_TOKEN_PATTERN = re.compile(r"ow_[0-9a-f]{48}")
# Request bodies up to this size are read whole; larger ones (rare: the Hub takes
# large files at signed storage URLs, not here) are streamed.
MAX_BUFFERED_BODY = 32 * 1024 * 1024

# Never forwarded upstream: transport internals, the incoming credentials (replaced),
# and traces of the proxies in front of the broker.
STRIP_REQUEST_HEADERS = {
    "host",
    "connection",
    "content-length",
    "transfer-encoding",
    "accept-encoding",
    "keep-alive",
    "upgrade",
    "expect",
    "authorization",
    "x-api-key",
    "api-key",
    "cookie",
    # The cloud metadata server answers only requests that carry this header. No
    # upstream of ours needs it, and a request must never be able to supply it.
    "metadata-flavor",
    "forwarded",
    "via",
    "traceparent",
}
STRIP_REQUEST_PREFIXES = ("x-forwarded-", "x-cloud-trace", "x-goog-", "cf-")
# httpx has already decoded the body, so the original framing would corrupt the copy.
STRIP_RESPONSE_HEADERS = {
    "content-encoding",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "set-cookie",
}


def presented_worker_token(authorization: Optional[str]) -> Optional[str]:
    """The worker token in an Authorization header, or None.

    Clients send "Bearer hf_ow_…"; git sends it as either half of Basic credentials.
    """
    if not authorization:
        return None
    candidates = []
    bearer = re.fullmatch(r"(?:Bearer|token)\s+(\S+)", authorization.strip(), re.I)
    if bearer:
        candidates.append(bearer.group(1))
    basic = re.fullmatch(r"Basic\s+([A-Za-z0-9+/=]+)", authorization.strip(), re.I)
    if basic:
        try:
            decoded = base64.b64decode(basic.group(1)).decode("utf-8")
        except Exception:
            decoded = ""
        candidates.extend(decoded.split(":", 1))
    for candidate in candidates:
        if candidate.startswith(HF_TOKEN_PREFIX):
            candidate = candidate[len(HF_TOKEN_PREFIX) :]
        if WORKER_TOKEN_PATTERN.fullmatch(candidate):
            return candidate
    return None


def hub_auth_header(path: str, token: str) -> str:
    """git's smart-HTTP endpoints take the token as a Basic password, the rest as Bearer."""
    parts = path.strip("/").split("/")
    last, before_last = parts[-1], parts[-2] if len(parts) > 1 else ""
    if last in ("git-upload-pack", "git-receive-pack") or (
        before_last == "info" and last == "refs"
    ):
        return "Basic " + base64.b64encode(f"user:{token}".encode()).decode()
    return f"Bearer {token}"


def hub_url_via_broker(url: str, upstream: str, broker_prefix: str) -> str:
    """Point a URL from a Hub answer back at the broker if it leads to the Hub.

    The client would otherwise send its stand-in token straight to huggingface.co,
    which has never heard of it. URLs of other hosts (signed storage URLs) are
    returned unchanged: the client goes there directly, without a token.
    """
    root = upstream.rstrip("/")
    if url.startswith("/") and not url.startswith("//"):
        return broker_prefix + url
    if url == root:
        return broker_prefix
    if url.startswith(root + "/") or url.startswith(root + "?"):
        return broker_prefix + url[len(root) :]
    return url


def is_lfs_batch(method: str, path: str) -> bool:
    return method.upper() == "POST" and path.rstrip("/").endswith(
        "/info/lfs/objects/batch"
    )


def rewrite_lfs_batch_hrefs(text: str, upstream: str, absolute_prefix: str) -> str:
    """The LFS batch answer with every href that leads to the Hub pointed at the broker."""
    try:
        parsed = json.loads(text)
    except ValueError:
        return text

    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
        elif isinstance(node, dict):
            for key, value in node.items():
                if key == "href" and isinstance(value, str):
                    # The client resolves a relative href against the batch URL itself.
                    if not value.startswith("/"):
                        node[key] = hub_url_via_broker(value, upstream, absolute_prefix)
                else:
                    visit(value)

    visit(parsed)
    return json.dumps(parsed)


class WorkerIdentity:
    def __init__(
        self,
        organization_id: str,
        worker_id: str,
        token_id: str,
        settings: Optional[dict] = None,
    ):
        self.organization_id = organization_id
        self.worker_id = worker_id
        self.token_id = token_id
        # What the cluster manager wrote onto the token for the broker
        # (api_tokens.broker_settings), e.g. where the organization's LiteLLM is.
        self.settings = settings or {}


class WorkerTokenValidator:
    """Asks the OpenWeights database whether a worker token is valid right now.

    The token is exchanged for a JWT exactly as the SDK does it, which fails for a
    revoked or expired token; with that JWT the broker reads the token's own row to
    learn which worker it belongs to. Only worker tokens are accepted: a personal
    API key is not a way to use the organization's secrets through the broker.

    An answer is cached for ``ttl`` seconds, which bounds how long a revoked token
    keeps working here.
    """

    def __init__(
        self,
        supabase_url: str,
        anon_key: str,
        client: Optional[httpx.AsyncClient] = None,
        ttl: float = 60,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.supabase_url = supabase_url.rstrip("/")
        self.anon_key = anon_key
        self._client = client or httpx.AsyncClient(timeout=20)
        self.ttl = ttl
        self._clock = clock
        self._cache: Dict[str, Tuple[float, Optional[WorkerIdentity]]] = {}

    async def validate(self, worker_token: str) -> Optional[WorkerIdentity]:
        key = hashlib.sha256(worker_token.encode()).hexdigest()
        cached = self._cache.get(key)
        if cached and cached[0] > self._clock():
            return cached[1]
        identity = await self._ask_database(worker_token)
        if len(self._cache) > 5000:
            self._cache.clear()
        self._cache[key] = (self._clock() + self.ttl, identity)
        return identity

    async def _ask_database(self, worker_token: str) -> Optional[WorkerIdentity]:
        exchange = await self._client.post(
            f"{self.supabase_url}/rest/v1/rpc/exchange_api_token_for_jwt",
            headers={
                "apikey": self.anon_key,
                "authorization": f"Bearer {self.anon_key}",
            },
            json={"api_token": worker_token},
        )
        if exchange.status_code >= 500:
            raise httpx.HTTPError("the OpenWeights database is not answering")
        if exchange.status_code != 200:
            return None  # invalid, revoked or expired
        jwt = exchange.json()
        try:
            payload = jwt.split(".")[1]
            claims = json.loads(
                base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
            )
            token_id, organization_id = (
                claims["api_token_id"],
                claims["organization_id"],
            )
        except Exception:
            return None
        row = await self._read_token_row(token_id, jwt, "worker_id,broker_settings")
        if row.status_code == 400:
            # A database from before broker settings has no such column.
            row = await self._read_token_row(token_id, jwt, "worker_id")
        if row.status_code >= 500:
            raise httpx.HTTPError("the OpenWeights database is not answering")
        rows = row.json() if row.status_code == 200 else []
        worker_id = rows[0].get("worker_id") if rows else None
        if not worker_id:
            return None  # a valid key, but not a worker's
        return WorkerIdentity(
            organization_id, worker_id, token_id, rows[0].get("broker_settings")
        )

    async def _read_token_row(self, token_id: str, jwt: str, columns: str):
        return await self._client.get(
            f"{self.supabase_url}/rest/v1/api_tokens",
            params={"id": f"eq.{token_id}", "select": columns},
            headers={"apikey": self.anon_key, "authorization": f"Bearer {jwt}"},
        )


class OrganizationSecrets:
    """An organization's secrets from the secret store, remembered for ``ttl`` seconds."""

    def __init__(
        self, store, ttl: float = 300, clock: Callable[[], float] = time.monotonic
    ):
        self.store = store
        self.ttl = ttl
        self._clock = clock
        self._cache: Dict[Tuple[str, str], Tuple[float, Optional[str]]] = {}

    async def get(self, organization_id: str, name: str) -> Optional[str]:
        key = (organization_id, name)
        cached = self._cache.get(key)
        if cached and cached[0] > self._clock():
            return cached[1]
        value = await anyio.to_thread.run_sync(self.store.get, organization_id, name)
        self._cache[key] = (self._clock() + self.ttl, value)
        return value

    def forget(self, organization_id: str, name: str) -> None:
        self._cache.pop((organization_id, name), None)


class UpstreamNotAllowed(ValueError):
    """An upstream address the broker refuses to send a credential to."""


def checked_upstream(url: str, resolve=socket.getaddrinfo) -> str:
    """An organization-chosen upstream base URL, or UpstreamNotAllowed.

    The address comes from an organization's settings, and the broker sends that
    organization's key there, which is the organization's own business. What must
    not happen is that the address leads *inside*: to the broker's own host, its
    cloud metadata server (which hands out the broker's credentials) or a private
    network. So: https only, a host name (no literal address), and every address the
    name resolves to must be a public one.
    """
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.scheme != "https" or not host or parts.username or parts.password:
        raise UpstreamNotAllowed("the address must be an https URL without credentials")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise UpstreamNotAllowed("the address must use a host name")
    if host == "localhost" or host.endswith((".internal", ".local", ".localhost")):
        raise UpstreamNotAllowed("the address must be a public host")
    try:
        addresses = {info[4][0] for info in resolve(host, parts.port or 443)}
    except OSError:
        raise UpstreamNotAllowed("the host name does not resolve")
    for address in addresses:
        if not ipaddress.ip_address(address.split("%")[0]).is_global:
            raise UpstreamNotAllowed("the address must be a public host")
    return url.rstrip("/")


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": f"OpenWeights broker: {message}"}, status_code=status)


def create_app(
    validator: WorkerTokenValidator,
    secrets: OrganizationSecrets,
    upstream_client: Optional[httpx.AsyncClient] = None,
    hf_upstream: str = HF_UPSTREAM,
    public_url: Optional[str] = None,
    check_upstream: Callable[[str], str] = checked_upstream,
) -> FastAPI:
    """The broker as an ASGI app. ``public_url`` is the address workers reach it at."""
    app = FastAPI(title="OpenWeights credential broker", docs_url=None, redoc_url=None)
    # No redirects: following one could send the real token to a host nobody vetted.
    client = upstream_client or httpx.AsyncClient(
        timeout=httpx.Timeout(60, read=600), follow_redirects=False
    )
    hf_root = hf_upstream.rstrip("/")

    def origin_of(request: Request) -> str:
        if public_url:
            return public_url.rstrip("/")
        scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
        return f"{scheme}://{request.headers.get('host', request.url.netloc)}"

    async def worker_and_secret(request: Request, secret_name: str):
        """(worker, the organization's secret), or (a refusal to return, None)."""
        worker_token = presented_worker_token(
            request.headers.get("authorization")
        ) or presented_worker_token(
            "Bearer "
            + (request.headers.get("x-api-key") or request.headers.get("api-key") or "")
        )
        if worker_token is None:
            return _error(401, "send the worker's own token as the API key."), None
        try:
            identity = await validator.validate(worker_token)
        except httpx.HTTPError:
            return _error(503, "cannot check the worker token right now."), None
        if identity is None:
            return _error(401, "this worker token is not valid (any more)."), None
        try:
            value = await secrets.get(identity.organization_id, secret_name)
        except Exception as error:  # the store's own message names no secret
            logger.error(
                "secret store failed for org %s: %s", identity.organization_id, error
            )
            return _error(503, f"cannot read the organization's {secret_name}."), None
        if not value:
            return (
                _error(
                    403,
                    f"the organization has no {secret_name} secret; set one in the dashboard.",
                ),
                None,
            )
        return identity, value

    def forwardable_headers(request: Request) -> Dict[str, str]:
        return {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in STRIP_REQUEST_HEADERS
            and not name.lower().startswith(STRIP_REQUEST_PREFIXES)
        }

    async def request_content(request: Request):
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return None
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) <= MAX_BUFFERED_BODY:
            # Sent on with its length: some endpoints refuse a chunked body.
            return await request.body()
        return request.stream()

    @app.get("/healthz")
    @app.get("/health")  # Cloud Run's front end keeps /healthz for itself
    async def healthz():
        return {"ok": True}

    @app.api_route(
        HF_PREFIX + "/{path:path}",
        methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def huggingface(path: str, request: Request):
        identity, real_token = await worker_and_secret(request, "HF_TOKEN")
        if real_token is None:
            return identity  # the refusal
        headers = forwardable_headers(request)
        headers["authorization"] = hub_auth_header(path, real_token)
        # The Hub's sizes are only right for an unencoded answer, which is why
        # huggingface_hub asks for "identity" on HEAD: pass that wish on.
        if request.headers.get("accept-encoding", "").strip().lower() == "identity":
            headers["accept-encoding"] = "identity"

        url = f"{hf_root}/{path}"
        if request.url.query:
            url += f"?{request.url.query}"
        upstream_request = client.build_request(
            request.method, url, headers=headers, content=await request_content(request)
        )
        started = time.monotonic()
        try:
            upstream = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as error:
            logger.warning(
                "hf %s /%s failed upstream: %s",
                request.method,
                path,
                type(error).__name__,
            )
            return _error(502, "Hugging Face is unreachable.")

        if upstream.status_code == 401:
            # The stored token may have been replaced: read it again next time.
            secrets.forget(identity.organization_id, "HF_TOKEN")

        response_headers = {
            name: value
            for name, value in upstream.headers.items()
            if name.lower() not in STRIP_RESPONSE_HEADERS
        }
        # An unencoded answer has a true Content-Length, and huggingface_hub needs it
        # (file size on HEAD, progress and consistency checks on GET).
        length = upstream.headers.get("content-length")
        if length and not upstream.headers.get("content-encoding"):
            response_headers["content-length"] = length
        origin = origin_of(request)
        if "location" in response_headers:
            # Relative, so it is right whatever address the client used to reach us.
            response_headers["location"] = hub_url_via_broker(
                response_headers["location"], hf_root, HF_PREFIX
            )
        if "link" in response_headers:
            response_headers["link"] = re.sub(
                r"<([^>]+)>",
                lambda match: "<"
                + hub_url_via_broker(
                    match.group(1),
                    hf_root,
                    HF_PREFIX if match.group(1).startswith("/") else origin + HF_PREFIX,
                )
                + ">",
                response_headers["link"],
            )
        if "x-xet-refresh-route" in response_headers:
            response_headers["x-xet-refresh-route"] = hub_url_via_broker(
                response_headers["x-xet-refresh-route"], hf_root, origin + HF_PREFIX
            )

        logger.info(
            "hf %s /%s -> %s (%d ms) worker=%s org=%s",
            request.method,
            path,
            upstream.status_code,
            (time.monotonic() - started) * 1000,
            identity.worker_id,
            identity.organization_id,
        )

        if upstream.is_success and is_lfs_batch(request.method, path):
            body = (await upstream.aread()).decode("utf-8", "replace")
            await upstream.aclose()
            response_headers.pop("content-length", None)
            return Response(
                rewrite_lfs_batch_hrefs(body, hf_root, origin + HF_PREFIX),
                status_code=upstream.status_code,
                headers=response_headers,
            )
        if request.method == "HEAD":
            await upstream.aclose()
            return Response(status_code=upstream.status_code, headers=response_headers)

        async def body():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            body(), status_code=upstream.status_code, headers=response_headers
        )

    @app.api_route(
        LITELLM_PREFIX + "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def litellm(path: str, request: Request):
        """The organization's LiteLLM proxy, with the organization's key put in."""
        identity, real_key = await worker_and_secret(request, "LITELLM_API_KEY")
        if real_key is None:
            return identity  # the refusal
        configured = identity.settings.get("litellm_base_url")
        if not configured:
            return _error(
                403,
                "the organization has no LITELLM_BASE_URL setting; set one in the dashboard.",
            )
        try:
            base = await anyio.to_thread.run_sync(check_upstream, configured)
        except UpstreamNotAllowed as refusal:
            return _error(403, f"LITELLM_BASE_URL is not usable: {refusal}.")

        headers = forwardable_headers(request)
        headers["authorization"] = f"Bearer {real_key}"
        url = f"{base}/{path}"
        if request.url.query:
            url += f"?{request.url.query}"
        started = time.monotonic()
        try:
            upstream = await client.send(
                client.build_request(
                    request.method,
                    url,
                    headers=headers,
                    content=await request_content(request),
                ),
                stream=True,
            )
        except httpx.HTTPError as error:
            logger.warning(
                "litellm %s /%s failed upstream: %s",
                request.method,
                path,
                type(error).__name__,
            )
            return _error(502, "the LiteLLM proxy is unreachable.")
        if upstream.status_code == 401:
            secrets.forget(identity.organization_id, "LITELLM_API_KEY")
        logger.info(
            "litellm %s /%s -> %s (%d ms) worker=%s org=%s",
            request.method,
            path,
            upstream.status_code,
            (time.monotonic() - started) * 1000,
            identity.worker_id,
            identity.organization_id,
        )
        response_headers = {
            name: value
            for name, value in upstream.headers.items()
            if name.lower() not in STRIP_RESPONSE_HEADERS and name.lower() != "location"
        }

        async def body():
            try:
                # Chunks as they arrive: answers are often token streams.
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            body(), status_code=upstream.status_code, headers=response_headers
        )

    return app


def app_from_env() -> FastAPI:
    """The app as `ow broker` runs it: configured by the environment."""
    import os

    from openweights.client import _SUPABASE_ANON_KEY, _SUPABASE_URL
    from openweights.secret_store import secret_store_from_env

    store = secret_store_from_env()
    if store is None:
        raise RuntimeError(
            "The broker needs a secret store to read the real tokens from: set "
            "OW_SECRET_STORE (for example gcp-secret-manager://<project id>)."
        )
    return create_app(
        WorkerTokenValidator(_SUPABASE_URL, _SUPABASE_ANON_KEY),
        OrganizationSecrets(store),
        public_url=os.environ.get("OW_BROKER_URL"),
    )
