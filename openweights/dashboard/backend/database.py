import os
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import jwt
import requests
from dotenv import load_dotenv
from models import (
    Job,
    JobWithRuns,
    Member,
    Organization,
    OrganizationCreate,
    OrganizationSecret,
    Run,
    RunWithJobAndWorker,
    Token,
    TokenCreate,
    Worker,
    WorkerWithRuns,
)
from postgrest.exceptions import APIError
from utils import (
    clean_ansi,
    validate_huggingface_secrets,
    validate_organization_secrets,
)

from openweights.client import _SUPABASE_ANON_KEY, _SUPABASE_URL, OpenWeights
from openweights.secret_store import (
    SecretStoreError,
    is_managed_secret,
    last_four,
    secret_store_from_env,
)
from supabase import Client, create_client

# Settings stored next to the secrets that are not credentials and may be shown.
# Mirrors is_plain_organization_setting() in the database.
PLAIN_SETTINGS = frozenset({"HF_USER", "HF_ORG", "OW_MANAGED", "MAX_WORKERS"})
HF_SECRET_NAMES = frozenset({"HF_TOKEN", "HF_USER", "HF_ORG"})
SECRET_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def secret_kind(name: str) -> str:
    """"secret": a credential, write-only. "env": passed to every worker as is."""
    return "secret" if is_managed_secret(name) else "env"


def mask_secret_row(row: dict) -> dict:
    """What may leave the backend about a stored secret: never the value itself.

    Same rule as list_organization_secrets_masked() in the database.
    """
    name, value = row["name"], row["value"] or ""
    plain = name in PLAIN_SETTINGS
    return {
        "name": name,
        "kind": secret_kind(name),
        "plain_value": value if plain else None,
        # Four characters of a short value would be most of it.
        "last_four": value[-4:] if not plain and len(value) >= 12 else None,
        "updated_at": row.get("updated_at"),
    }


class Database:
    secret_store = None

    def __init__(self, auth_token: Optional[str] = None):
        self.supabase_url = _SUPABASE_URL
        self.supabase_anon_key = _SUPABASE_ANON_KEY
        self.auth_token = auth_token
        self._current_org_id = None
        # None on a deployment without a secret store: secrets then stay rows of
        # organization_secrets, like environment variables.
        self.secret_store = secret_store_from_env()

        if not self.supabase_url or not self.supabase_anon_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")

        # Initialize regular client for normal operations
        self.client = create_client(self.supabase_url, self.supabase_anon_key)
        if auth_token:
            self.client.postgrest.auth(auth_token)

    @property
    def ow_client(self) -> OpenWeights:
        """Get an OpenWeights client for the current organization."""
        if not self._current_org_id:
            raise ValueError("No organization set. Call set_organization_id first.")
        return OpenWeights(
            self.supabase_url,
            self.supabase_anon_key,
            self.auth_token,
            organization_id=self._current_org_id,
        )

    def set_organization_id(self, org_id: str):
        """Set the current organization ID after verifying access."""
        if not self.verify_organization_access(org_id):
            raise ValueError("No access to this organization")
        self._current_org_id = org_id

    def get_user_id_from_token(self) -> str:
        """Extract user ID from JWT token."""
        if not self.auth_token:
            raise ValueError("No authentication token provided")

        try:
            # JWT tokens have three parts: header.payload.signature
            # We only need the payload
            payload = jwt.decode(self.auth_token, options={"verify_signature": False})
            return payload.get("sub")  # 'sub' is the user ID in Supabase JWTs
        except Exception as e:
            raise ValueError(f"Invalid authentication token: {str(e)}")

    def verify_organization_access(self, organization_id: str) -> bool:
        """Verify that the current user has access to the organization."""
        try:
            result = self.client.rpc(
                "is_organization_member", {"org_id": organization_id}
            ).execute()
            return result.data
        except Exception as e:
            print(f"Error verifying organization access: {e}")
            return False

    def create_organization(self, org_data: OrganizationCreate) -> Organization:
        """Create a new organization with the given name and secrets."""
        # First validate all secrets
        is_valid, error_message = validate_organization_secrets(org_data.secrets)
        if not is_valid:
            raise ValueError(f"Invalid secrets: {error_message}")

        try:
            # Create organization using RPC (handles organization creation and membership)
            org_response = self.client.rpc(
                "create_organization", {"org_name": org_data.name}
            ).execute()

            if not org_response.data:
                raise ValueError("Failed to create organization")

            org_id = org_response.data

            # Create a default API token for the organization
            token = self.create_token(org_id, TokenCreate(name="Default API Key"))

            # Store the API token as an organization secret
            self._write_setting(org_id, "OPENWEIGHTS_API_KEY", token.access_token)

            for secret_name, secret_value in org_data.secrets.items():
                self._write_setting(org_id, secret_name, secret_value)

            # Fetch the created organization to return
            org_result = (
                self.client.from_("organizations")
                .select("*")
                .eq("id", org_id)
                .single()
                .execute()
            )

            return Organization(**org_result.data)

        except Exception as e:
            raise ValueError(f"Failed to create organization: {str(e)}")

    def _readable_secret_values(
        self, organization_id: str, names: List[str]
    ) -> Dict[str, str]:
        """Values of the named secrets that this caller may read directly.

        Once migration 20261120000000_credential_safety.sql is applied, a signed-in
        user can read only the plain settings (HF_USER, HF_ORG, ...) this way; before
        it, an admin gets every name asked for.
        """
        result = (
            self.client.from_("organization_secrets")
            .select("name, value")
            .eq("organization_id", organization_id)
            .in_("name", names)
            .execute()
        )
        return {row["name"]: row["value"] for row in (result.data or [])}

    def set_organization_secret(
        self, organization_id: str, secret_name: str, secret_value: str
    ) -> None:
        """Create a secret or replace its value (admin-only, enforced by the RPC)."""
        self.set_organization_id(organization_id)
        if not SECRET_NAME_PATTERN.fullmatch(secret_name):
            raise ValueError(
                "Secret names may contain only letters, digits and underscores"
            )
        if not secret_value:
            raise ValueError("Secret value must not be empty")

        if secret_name in HF_SECRET_NAMES:
            # Validate the Hugging Face settings together, as far as the stored
            # ones are readable (the stored HF_TOKEN no longer is).
            hf_secrets = self._readable_secret_values(
                organization_id, sorted(HF_SECRET_NAMES)
            )
            hf_secrets[secret_name] = secret_value
            is_valid, error_message = validate_organization_secrets(hf_secrets)
            if not is_valid:
                raise ValueError(f"Invalid secret configuration: {error_message}")

        self._write_setting(organization_id, secret_name, secret_value)

    def _stores_in_secret_store(self, name: str) -> bool:
        return self.secret_store is not None and is_managed_secret(name)

    def _require_admin(self, organization_id: str) -> None:
        """The secret store does not know our users: ask the database first."""
        is_admin = self.client.rpc(
            "is_organization_admin", {"org_id": organization_id}
        ).execute()
        if is_admin.data is not True:
            raise ValueError("Only organization admins can manage secrets")

    def _write_setting(self, organization_id: str, name: str, value: str) -> None:
        """Write a secret to the secret store, anything else to the database."""
        try:
            if self._stores_in_secret_store(name):
                self._require_admin(organization_id)
                self.secret_store.set(organization_id, name, value)
                # Records what may be displayed and removes a copy of this secret
                # that an earlier version stored in organization_secrets.
                self.client.rpc(
                    "record_organization_secret",
                    {
                        "org_id": organization_id,
                        "secret_name": name,
                        "secret_last_four": last_four(value),
                    },
                ).execute()
                return
            result = self.client.rpc(
                "manage_organization_secret",
                {
                    "org_id": organization_id,
                    "secret_name": name,
                    "secret_value": value,
                },
            ).execute()
        except APIError as e:
            raise ValueError(f"Failed to update {name}: {e.message}")
        except SecretStoreError as e:
            raise ValueError(f"Failed to update {name}: {e}")
        if not result.data:
            raise ValueError(f"Failed to update {name}")

    def delete_organization_secret(
        self, organization_id: str, secret_name: str
    ) -> None:
        """Delete a secret (admin-only, enforced by the RPC)."""
        self.set_organization_id(organization_id)
        if secret_name == "OPENWEIGHTS_API_KEY":
            raise ValueError(
                "OPENWEIGHTS_API_KEY cannot be deleted: the cluster manager needs it"
            )
        try:
            if self._stores_in_secret_store(secret_name):
                self._require_admin(organization_id)
                self.secret_store.delete(organization_id, secret_name)
                forgotten = self.client.rpc(
                    "forget_organization_secret",
                    {"org_id": organization_id, "secret_name": secret_name},
                ).execute()
                if forgotten.data:
                    return
                # Not in the store: it may still be a row from before the store.
            result = self.client.rpc(
                "delete_organization_secret",
                {"org_id": organization_id, "secret_name": secret_name},
            ).execute()
        except APIError as e:
            raise ValueError(f"Failed to delete secret: {e.message}")
        except SecretStoreError as e:
            raise ValueError(f"Failed to delete secret: {e}")
        if not result.data:
            raise ValueError(f"Secret not found: {secret_name}")

    def get_organization(self, organization_id: str) -> Organization:
        """Fetch a single organization by id (RLS-enforced)."""
        self.set_organization_id(organization_id)
        result = (
            self.client.from_("organizations")
            .select("*")
            .eq("id", organization_id)
            .single()
            .execute()
        )
        if not result.data:
            raise ValueError("Organization not found")
        return Organization(**result.data)

    def update_organization(self, organization_id: str, name: str) -> Organization:
        """Rename an organization (admin-only, enforced by RPC)."""
        self.set_organization_id(organization_id)
        result = self.client.rpc(
            "update_organization", {"org_id": organization_id, "new_name": name}
        ).execute()
        if not result.data:
            raise ValueError("Failed to update organization")
        return self.get_organization(organization_id)

    def get_organization_members(self, organization_id: str) -> List[Member]:
        """List members of an organization."""
        self.set_organization_id(organization_id)
        result = self.client.rpc(
            "get_organization_members", {"org_id": organization_id}
        ).execute()
        return [
            Member(
                user_id=m["user_id"],
                role=m["role"],
                email=m.get("email"),
            )
            for m in (result.data or [])
        ]

    def invite_organization_member(
        self, organization_id: str, email: str, role: str
    ) -> Member:
        """Invite an existing auth user to the organization."""
        self.set_organization_id(organization_id)
        result = self.client.rpc(
            "invite_organization_member",
            {
                "org_id": organization_id,
                "member_email": email,
                "member_role": role,
            },
        ).execute()
        if not result.data:
            raise ValueError("Failed to invite member")
        row = result.data[0] if isinstance(result.data, list) else result.data
        return Member(user_id=row["user_id"], email=row.get("email"), role=row["role"])

    def update_organization_member_role(
        self, organization_id: str, user_id: str, role: str
    ) -> None:
        """Change a member's role (admin-only, enforced by RLS)."""
        self.set_organization_id(organization_id)
        result = (
            self.client.from_("organization_members")
            .update({"role": role})
            .eq("organization_id", organization_id)
            .eq("user_id", user_id)
            .execute()
        )
        if not result.data:
            raise ValueError("Failed to update member role")

    def remove_organization_member(
        self, organization_id: str, user_id: str
    ) -> None:
        """Remove a member from the organization."""
        self.set_organization_id(organization_id)
        result = self.client.rpc(
            "remove_organization_member",
            {"org_id": organization_id, "member_id": user_id},
        ).execute()
        if not result.data:
            raise ValueError("Failed to remove member")

    def list_organization_secrets(
        self, organization_id: str
    ) -> List[OrganizationSecret]:
        """List an organization's secrets without their values (admin-only).

        A secret is described by its name, its last four characters and when it was
        last changed. Only the plain settings come back in full.
        """
        self.set_organization_id(organization_id)
        try:
            rows = (
                self.client.rpc(
                    "list_organization_secrets_masked", {"org_id": organization_id}
                )
                .execute()
                .data
            )
        except APIError as e:
            if e.code == "42501":
                raise ValueError("Only organization admins can list secrets")
            if e.code != "PGRST202":
                raise
            # The database predates migration 20261120000000_credential_safety.sql:
            # read the rows as before and mask them here.
            rows = [
                mask_secret_row(row)
                for row in (
                    self.client.from_("organization_secrets")
                    .select("name, value, updated_at")
                    .eq("organization_id", organization_id)
                    .order("name")
                    .execute()
                    .data
                    or []
                )
            ]
        return [
            OrganizationSecret(**{**row, "kind": secret_kind(row["name"])})
            for row in (rows or [])
        ]

    def create_token(
        self, organization_id: str, token_data: TokenCreate
    ) -> Token:
        """Create a new API token with optional expiration."""
        self.set_organization_id(organization_id)

        # Calculate expiration time if specified
        expires_at = None
        if token_data.expires_in_days is not None:
            expires_at = datetime.now(timezone.utc) + timedelta(
                days=token_data.expires_in_days
            )

        params = {
            "org_id": organization_id,
            "token_name": token_data.name,
            "expires_at": expires_at.isoformat() if expires_at else None,
        }
        rpc = "create_api_token"
        if token_data.spending_limit_usd is not None:
            rpc = "create_api_token_with_limit"
            params["spending_limit_usd"] = str(token_data.spending_limit_usd)
        result = self.client.rpc(rpc, params).execute()

        if not result.data or len(result.data) == 0:
            raise ValueError("Failed to create token")

        # Response is array of objects with token_id and token
        token_data_result = result.data[0]
        token_id = token_data_result["token_id"]
        api_token = token_data_result["token"]

        return Token(
            id=token_id,
            name=token_data.name,
            expires_at=expires_at,
            created_at=datetime.now(timezone.utc),
            access_token=api_token,
        )

    def list_tokens(self, organization_id: str) -> List[Token]:
        """List all API tokens for an organization."""
        self.set_organization_id(organization_id)
        result = (
            self.client.table("api_tokens")
            .select("id, name, expires_at, created_at, revoked_at")
            .eq("organization_id", organization_id)
            .order("created_at", desc=True)
            .execute()
        )
        return [Token(**token) for token in result.data]

    def delete_token(self, organization_id: str, token_id: str):
        """Revoke an API token."""
        self.set_organization_id(organization_id)

        # Use the revoke_api_token RPC instead of direct deletion
        result = self.client.rpc("revoke_api_token", {"token_id": token_id}).execute()

        if not result.data:
            raise ValueError("Failed to revoke token or token not found")

    def get_jobs(self, organization_id: str, status: Optional[str] = None) -> List[Job]:
        self.set_organization_id(organization_id)
        query = (
            self.client.table("jobs")
            .select("*")
            .eq("organization_id", organization_id)
            .order("created_at", desc=True)
        )
        if status:
            query = query.eq("status", status)
        result = query.execute()
        return [Job(**job) for job in result.data]

    def get_jobs_page(
        self,
        organization_id: str,
        statuses: Optional[List[str]],
        search: str,
        limit: int,
        offset: int,
    ) -> dict:
        self.set_organization_id(organization_id)
        return self.client.rpc(
            "get_dashboard_jobs",
            {
                "org_id": organization_id,
                "statuses": statuses,
                "search_text": search,
                "page_limit": limit,
                "page_offset": offset,
            },
        ).execute().data

    def get_job(self, organization_id: str, job_id: str) -> JobWithRuns:
        self.set_organization_id(organization_id)
        job = (
            self.client.table("jobs")
            .select("*")
            .eq("id", job_id)
            .eq("organization_id", organization_id)
            .single()
            .execute()
        )
        runs = (
            self.client.table("runs")
            .select("*")
            .eq("job_id", job_id)
            .order("created_at", desc=True)
            .execute()
        )
        return JobWithRuns(**job.data, runs=[Run(**run) for run in runs.data])

    def cancel_job(self, organization_id: str, job_id: str) -> Job:
        """Cancel a job."""
        self.set_organization_id(organization_id)
        result = (
            self.client.table("jobs")
            .update({"status": "canceled"})
            .eq("id", job_id)
            .eq("organization_id", organization_id)
            .execute()
        )

        if not result.data:
            raise ValueError("Job not found or no access")

        return Job(**result.data[0])

    def restart_job(self, organization_id: str, job_id: str) -> Job:
        """Restart a job."""
        self.set_organization_id(organization_id)
        result = (
            self.client.table("jobs")
            .update({"status": "pending", "worker_id": None})  # Clear worker assignment
            .eq("id", job_id)
            .eq("organization_id", organization_id)
            .execute()
        )

        if not result.data:
            raise ValueError("Job not found or no access")

        return Job(**result.data[0])

    def get_runs(self, organization_id: str, status: Optional[str] = None) -> List[Run]:
        self.set_organization_id(organization_id)
        # First get all jobs for this organization
        jobs_result = (
            self.client.from_("jobs")
            .select("id")
            .eq("organization_id", organization_id)
            .execute()
        )
        job_ids = [job["id"] for job in jobs_result.data]

        # Then get all runs for these jobs
        query = self.client.from_("runs").select(
            "id, job_id, worker_id, status, log_file, created_at"
        )
        if job_ids:
            query = query.in_("job_id", job_ids)
        if status:
            query = query.eq("status", status)

        result = query.execute()
        return [Run(**run) for run in result.data]

    def get_run(self, organization_id: str, run_id: str) -> RunWithJobAndWorker:
        self.set_organization_id(organization_id)
        run = self.client.table("runs").select("*").eq("id", run_id).single().execute()
        job = (
            self.client.table("jobs")
            .select("*")
            .eq("id", run.data["job_id"])
            .eq("organization_id", organization_id)
            .single()
            .execute()
        )
        worker = None
        if run.data.get("worker_id"):
            worker = (
                self.client.table("worker")
                .select("*")
                .eq("id", run.data["worker_id"])
                .single()
                .execute()
            )
            worker = Worker(**worker.data) if worker.data else None
        return RunWithJobAndWorker(**run.data, job=Job(**job.data), worker=worker)

    def get_run_events(self, organization_id: str, run_id: str) -> List[dict]:
        """Get events for a run."""
        self.set_organization_id(organization_id)
        try:
            return self.ow_client.events.list(run_id=run_id)
        except Exception as e:
            print(f"Error getting run events: {e}")
            raise ValueError(f"Error getting run events: {str(e)}")

    def get_workers(
        self, organization_id: str, status: Optional[str] = None
    ) -> List[Worker]:
        self.set_organization_id(organization_id)
        query = (
            self.client.table("worker")
            .select("*")
            .eq("organization_id", organization_id)
            .order("created_at", desc=True)
        )
        if status:
            query = query.eq("status", status)
        result = query.execute()
        return [Worker(**worker) for worker in result.data]

    def get_worker(self, organization_id: str, worker_id: str) -> WorkerWithRuns:
        self.set_organization_id(organization_id)
        worker = (
            self.client.table("worker")
            .select("*")
            .eq("id", worker_id)
            .eq("organization_id", organization_id)
            .single()
            .execute()
        )
        runs = (
            self.client.table("runs")
            .select("*")
            .eq("worker_id", worker_id)
            .order("created_at", desc=True)
            .execute()
        )
        return WorkerWithRuns(**worker.data, runs=[Run(**run) for run in runs.data])

    def get_worker_logs(self, organization_id: str, worker_id: str) -> str:
        """Get logs from a worker's pod or saved logfile."""
        self.set_organization_id(organization_id)
        worker = (
            self.client.table("worker")
            .select("*")
            .eq("id", worker_id)
            .eq("organization_id", organization_id)
            .single()
            .execute()
        )

        if not worker.data:
            raise ValueError("Worker not found or no access")

        # If worker is terminated or shutdown, try to get logs from saved file
        if worker.data["status"] in ["terminated", "shutdown"] and worker.data.get(
            "logfile"
        ):
            try:
                return clean_ansi(
                    self.ow_client.files.content(worker.data["logfile"]).decode("utf-8")
                )
            except Exception as e:
                return f"Error fetching saved logs: {str(e)}"

        # If worker is active/starting, try to get logs from pod
        pod_id = worker.data.get("pod_id")
        if not pod_id:
            return "No pod ID available for this worker"

        try:
            response = requests.get(
                f"https://{pod_id}-10101.proxy.runpod.net/logs", timeout=(5, 10)
            )
            if response.status_code == 200:
                return clean_ansi(response.text)
            else:
                return f"Error fetching logs: HTTP {response.status_code}"
        except Exception as e:
            return f"Error fetching logs: {str(e)}"

    def shutdown_worker(self, organization_id: str, worker_id: str) -> Worker:
        """Set the shutdown flag for a worker."""
        self.set_organization_id(organization_id)
        result = (
            self.client.table("worker")
            .update({"status": "shutdown"})
            .eq("id", worker_id)
            .eq("organization_id", organization_id)
            .execute()
        )

        if not result.data:
            raise ValueError("Worker not found or no access")

        return Worker(**result.data[0])

    def get_file_content(self, organization_id: str, file_id: str) -> str:
        """Get the content of a file."""
        self.set_organization_id(organization_id)
        try:
            content = self.ow_client.files.content(file_id)
            if isinstance(content, bytes):
                return content.decode("utf-8")
            return str(content)
        except Exception as e:
            raise Exception(f"Error getting file content: {str(e)}")

    def get_log_file_content(self, organization_id: str, run_id: str) -> str:
        self.set_organization_id(organization_id)
        try:
            # First try to get the run
            run_result = (
                self.client.table("runs").select("*").eq("id", run_id).execute()
            )
            if not run_result.data:
                return f"Run {run_id} not found in database"

            run = run_result.data[0]
            # If the run is in progress, try to get logs from worker
            if run["status"] == "in_progress" and run.get("worker_id"):
                worker_result = (
                    self.client.table("worker")
                    .select("*")
                    .eq("id", run["worker_id"])
                    .execute()
                )
                if worker_result.data:
                    worker = worker_result.data[0]
                    pod_id = worker.get("pod_id")
                    if not pod_id:
                        return "No pod ID available for this worker"

                    try:
                        response = requests.get(
                            f"https://{pod_id}-10101.proxy.runpod.net/{run_id}", timeout=(5, 10)
                        )
                        if response.status_code == 200:
                            return clean_ansi(response.text)
                        else:
                            return f"Error fetching logs: HTTP {response.status_code}"
                    except Exception as e:
                        return f"Error fetching logs: {str(e)}"

            if not run.get("log_file"):
                return f"No log file associated with run {run_id}"

            # Get log content using OpenWeights client
            try:
                log_content = self.ow_client.files.content(run["log_file"]).decode(
                    "utf-8"
                )
                return clean_ansi(log_content)
            except Exception as e:
                print(f"Error getting log content via OpenWeights: {str(e)}")
                return f"Error getting log content via OpenWeights: {str(e)}"

        except Exception as e:
            return f"Error processing request: {str(e)}"
