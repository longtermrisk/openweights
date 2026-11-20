"""Where an organization's secrets are kept, as opposed to its environment variables.

An organization has two kinds of settings:

* **Environment variables** (``HF_USER``, ``MAX_WORKERS``, anything custom) are rows of
  ``organization_secrets`` and are passed to every worker, where job code can read them.
* **Secrets** (``MANAGED_SECRETS``) are credentials. On a deployment with a secret
  store they live in that store and nowhere in the database; the database only keeps
  what may be displayed about them (``organization_secret_metadata``: the last four
  characters and the time of the last change).

The store is selected with ``OW_SECRET_STORE``:

    OW_SECRET_STORE=gcp-secret-manager://<project id>

Unset means there is no store: secrets stay rows of ``organization_secrets`` as before.
That is the case for a self-managed cluster started with ``ow cluster --env-file``,
which takes its credentials from the env file.

Who may do what is decided by the store's own access control, not here. On Google
Cloud the dashboard's service account can add and destroy versions but has no
``secretmanager.versions.access``; the cluster manager's (and later the broker's) can
read. So the dashboard cannot return a stored value even if asked to.
"""

import base64
import logging
import os
import re
import time
from typing import Callable, Dict, Iterable, Optional

import httpx

logger = logging.getLogger(__name__)

# Credentials that belong in the secret store. Mirrored in
# is_managed_organization_secret() in the database and in the dashboard frontend.
MANAGED_SECRETS = frozenset(
    {"HF_TOKEN", "OPENWEIGHTS_API_KEY", "RUNPOD_API_KEY", "LITELLM_API_KEY"}
)
# Managed secrets the cluster manager still copies into a worker's environment. This
# list shrinks as workers move to their own tokens and to the broker; a secret that
# was never passed to workers (LITELLM_API_KEY) does not start now.
SECRETS_STILL_PASSED_TO_WORKERS = frozenset({"HF_TOKEN", "OPENWEIGHTS_API_KEY"})

SECRET_STORE_ENV = "OW_SECRET_STORE"
GCP_SCHEME = "gcp-secret-manager://"
GCP_API = "https://secretmanager.googleapis.com/v1"
GCP_METADATA_TOKEN_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/"
    "service-accounts/default/token"
)


def is_managed_secret(name: str) -> bool:
    return name in MANAGED_SECRETS


def last_four(value: str) -> Optional[str]:
    """The part of a secret that may be displayed: nothing for short values."""
    return value[-4:] if len(value) >= 12 else None


class SecretStoreError(RuntimeError):
    """The secret store refused or failed a request. Never carries a secret value."""


class InMemorySecretStore:
    """A store for tests."""

    def __init__(self):
        self.values: Dict[tuple, str] = {}

    def get(self, organization_id: str, name: str) -> Optional[str]:
        return self.values.get((organization_id, name))

    def set(self, organization_id: str, name: str, value: str) -> None:
        self.values[(organization_id, name)] = value

    def delete(self, organization_id: str, name: str) -> None:
        self.values.pop((organization_id, name), None)


def _metadata_server_token(client: httpx.Client) -> tuple:
    """An access token for the service account this process runs as, and its lifetime."""
    response = client.get(GCP_METADATA_TOKEN_URL, headers={"Metadata-Flavor": "Google"})
    response.raise_for_status()
    body = response.json()
    return body["access_token"], float(body.get("expires_in", 300))


class GoogleSecretManagerStore:
    """Secrets in Google Secret Manager, one secret per organization and name.

    Uses the REST API with the token of the service account the process runs as
    (metadata server), so it needs no Google client library.
    """

    def __init__(
        self,
        project: str,
        client: Optional[httpx.Client] = None,
        token_source: Optional[Callable[[httpx.Client], tuple]] = None,
    ):
        if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", project):
            raise ValueError(f"Not a Google Cloud project id: {project!r}")
        self.project = project
        self._client = client or httpx.Client(timeout=20)
        self._token_source = token_source or _metadata_server_token
        self._token: Optional[str] = None
        self._token_expires_at = 0.0

    @staticmethod
    def secret_id(organization_id: str, name: str) -> str:
        if not re.fullmatch(r"[0-9a-fA-F-]{36}", organization_id):
            raise ValueError("organization_id must be a UUID")
        if not is_managed_secret(name):
            raise ValueError(f"{name} is not a managed secret")
        return f"ow-org-{organization_id.lower()}-{name}"

    def _secret_path(self, organization_id: str, name: str) -> str:
        return (
            f"projects/{self.project}/secrets/{self.secret_id(organization_id, name)}"
        )

    def _headers(self) -> Dict[str, str]:
        if self._token is None or time.time() > self._token_expires_at - 60:
            self._token, lifetime = self._token_source(self._client)
            self._token_expires_at = time.time() + lifetime
        return {"Authorization": f"Bearer {self._token}"}

    def _request(self, method: str, path: str, ok: Iterable[int] = (200,), **kwargs):
        response = self._client.request(
            method, f"{GCP_API}/{path}", headers=self._headers(), **kwargs
        )
        if response.status_code not in ok:
            # The body of a failed call describes the request, not a secret, but it
            # is the store's text and not ours: report only the status.
            raise SecretStoreError(
                f"Secret Manager {method} {path.split('?')[0]} failed with status "
                f"{response.status_code}"
            )
        return response

    def get(self, organization_id: str, name: str) -> Optional[str]:
        path = self._secret_path(organization_id, name)
        response = self._request("GET", f"{path}/versions/latest:access", ok=(200, 404))
        if response.status_code == 404:
            return None
        data = response.json()["payload"]["data"]
        return base64.b64decode(data).decode("utf-8")

    def set(self, organization_id: str, name: str, value: str) -> None:
        """Store a new value and destroy the versions that held earlier ones."""
        path = self._secret_path(organization_id, name)
        created = self._request(
            "POST",
            f"projects/{self.project}/secrets"
            f"?secretId={self.secret_id(organization_id, name)}",
            ok=(200, 409),
            json={
                "replication": {"automatic": {}},
                "labels": {"managed-by": "openweights"},
            },
        )
        payload = base64.b64encode(value.encode("utf-8")).decode("ascii")
        added = self._request(
            "POST", f"{path}:addVersion", json={"payload": {"data": payload}}
        )
        if created.status_code == 409:
            self._destroy_versions(path, keep=added.json()["name"])

    def _destroy_versions(self, path: str, keep: str) -> None:
        listed = self._request(
            "GET", f"{path}/versions?filter=state:ENABLED&pageSize=100"
        )
        for version in listed.json().get("versions", []):
            if version["name"] != keep:
                self._request("POST", f"{version['name']}:destroy", json={})

    def delete(self, organization_id: str, name: str) -> None:
        self._request("DELETE", self._secret_path(organization_id, name), ok=(200, 404))


def secret_store_from_env(environ=None):
    """The configured secret store, or None when the deployment has none."""
    setting = (environ if environ is not None else os.environ).get(SECRET_STORE_ENV)
    if not setting:
        return None
    if setting.startswith(GCP_SCHEME):
        return GoogleSecretManagerStore(setting[len(GCP_SCHEME) :].strip("/"))
    raise ValueError(
        f"{SECRET_STORE_ENV} must look like {GCP_SCHEME}<project id>, got {setting!r}"
    )


class ManagedSecretReader:
    """Reads an organization's managed secrets, asking the store only after a change.

    ``organization_secret_metadata`` says which secrets exist and when each was last
    set, so one cheap database query per loop decides whether the store has to be
    read at all.
    """

    def __init__(self, store):
        self.store = store
        self._cache: Dict[tuple, tuple] = {}

    def read(self, supabase, organization_id: str) -> Dict[str, str]:
        rows = (
            supabase.table("organization_secret_metadata")
            .select("name, updated_at")
            .eq("organization_id", organization_id)
            .execute()
            .data
            or []
        )
        secrets = {}
        for row in rows:
            name = row["name"]
            if not is_managed_secret(name):
                continue
            key = (organization_id, name)
            cached = self._cache.get(key)
            if cached is None or cached[0] != row["updated_at"]:
                value = self.store.get(organization_id, name)
                if value is None:
                    logger.warning(
                        "Secret %s of organization %s is recorded but not in the store",
                        name,
                        organization_id,
                    )
                    continue
                cached = (row["updated_at"], value)
                self._cache[key] = cached
            secrets[name] = cached[1]
        for key in [k for k in self._cache if k[0] == organization_id]:
            if key[1] not in secrets:
                del self._cache[key]
        return secrets
