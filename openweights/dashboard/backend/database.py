import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import jwt
import requests
from dotenv import load_dotenv
from models import (
    Job,
    JobWithRuns,
    Organization,
    OrganizationCreate,
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

from openweights import OpenWeights
from supabase import Client, create_client


class Database:
    def __init__(self, auth_token: Optional[str] = None):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
        self.supabase_service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.auth_token = auth_token
        self._current_org_id = None

        if (
            not self.supabase_url
            or not self.supabase_anon_key
            or not self.supabase_service_key
        ):
            raise ValueError(
                "SUPABASE_URL, SUPABASE_ANON_KEY, and SUPABASE_SERVICE_ROLE_KEY must be set"
            )

        # Initialize regular client for normal operations
        self.client = create_client(self.supabase_url, self.supabase_anon_key)
        if auth_token:
            self.client.postgrest.auth(auth_token)

        # Initialize admin client for operations requiring service role
        self.admin_client = create_client(self.supabase_url, self.supabase_service_key)

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

    async def create_organization(self, org_data: OrganizationCreate) -> Organization:
        """Create a new organization with the given name and secrets."""
        # First validate all secrets
        is_valid, error_message = validate_organization_secrets(org_data.secrets)
        if not is_valid:
            raise ValueError(f"Invalid secrets: {error_message}")

        try:
            # Create organization using admin client to bypass RLS
            result = (
                self.admin_client.from_("organizations")
                .insert({"name": org_data.name})
                .execute()
            )

            if not result.data:
                raise ValueError("Failed to create organization")

            org_id = result.data[0]["id"]

            # Add the creator as an admin using admin client
            user_id = self.get_user_id_from_token()
            member_result = (
                self.admin_client.from_("organization_members")
                .insert(
                    {"organization_id": org_id, "user_id": user_id, "role": "admin"}
                )
                .execute()
            )

            if not member_result.data:
                raise ValueError("Failed to add admin member")

            # Add secrets using admin client
            token = await self.create_token(
                org_id, TokenCreate(name="OpenWeights API Key")
            )
            org_data.secrets["OPENWEIGHTS_API_KEY"] = token.access_token
            for secret_name, secret_value in org_data.secrets.items():
                secret_result = (
                    self.admin_client.from_("organization_secrets")
                    .insert(
                        {
                            "organization_id": org_id,
                            "name": secret_name,
                            "value": secret_value,
                        }
                    )
                    .execute()
                )

                if not secret_result.data:
                    raise ValueError(f"Failed to add secret: {secret_name}")

            return Organization(**result.data[0])

        except Exception as e:
            raise ValueError(f"Failed to create organization: {str(e)}")

    async def update_organization_secret(
        self, organization_id: str, secret_name: str, secret_value: str
    ) -> bool:
        """Update or create an organization secret."""
        self.set_organization_id(organization_id)

        # Get existing secrets to validate together
        try:
            secrets_result = (
                self.client.from_("organization_secrets")
                .select("name, value")
                .eq("organization_id", organization_id)
                .execute()
            )

            # Create a dictionary of current secrets
            current_secrets = {
                secret["name"]: secret["value"] for secret in secrets_result.data
            }

            # Update with new secret value
            current_secrets[secret_name] = secret_value

            # Validate all secrets together
            is_valid, error_message = validate_organization_secrets(current_secrets)
            if not is_valid:
                raise ValueError(f"Invalid secret configuration: {error_message}")

            # If validation passes, update the secret
            result = self.client.rpc(
                "manage_organization_secret",
                {
                    "org_id": organization_id,
                    "secret_name": secret_name,
                    "secret_value": secret_value,
                },
            ).execute()

            return bool(result.data)
        except Exception as e:
            raise ValueError(f"Failed to update secret: {str(e)}")

    async def update_organization_secrets(
        self, organization_id: str, secrets: Dict[str, str]
    ) -> bool:
        """Update all organization secrets together. Deletes secrets not in the input dict."""
        self.set_organization_id(organization_id)

        try:
            # First validate all secrets together
            is_valid, error_message = validate_organization_secrets(secrets)
            if not is_valid:
                raise ValueError(f"Invalid secret configuration: {error_message}")

            # Get current secrets
            current_secrets_result = (
                self.client.from_("organization_secrets")
                .select("name")
                .eq("organization_id", organization_id)
                .execute()
            )

            current_secret_names = {
                secret["name"] for secret in current_secrets_result.data
            }
            new_secret_names = set(secrets.keys())

            # Delete secrets that are not in the new set
            secrets_to_delete = current_secret_names - new_secret_names
            for secret_name in secrets_to_delete:
                delete_result = (
                    self.client.from_("organization_secrets")
                    .delete()
                    .eq("organization_id", organization_id)
                    .eq("name", secret_name)
                    .execute()
                )

                if not delete_result.data:
                    print(f"Warning: Failed to delete secret {secret_name}")

            # Update or insert new secrets
            for secret_name, secret_value in secrets.items():
                result = self.client.rpc(
                    "manage_organization_secret",
                    {
                        "org_id": organization_id,
                        "secret_name": secret_name,
                        "secret_value": secret_value,
                    },
                ).execute()

                if not result.data:
                    raise ValueError(f"Failed to update secret: {secret_name}")

            return True
        except Exception as e:
            raise ValueError(f"Failed to update secrets: {str(e)}")

    async def create_token(
        self, organization_id: str, token_data: TokenCreate
    ) -> Token:
        """Create a new token with optional expiration."""
        self.set_organization_id(organization_id)

        # Get user ID from token
        user_id = self.get_user_id_from_token()

        # Check if user is an admin of the organization
        admin_check = self.client.rpc(
            "is_organization_admin", {"org_id": organization_id}
        ).execute()

        if not admin_check.data:
            raise ValueError("User is not an admin of this organization")

        # Calculate expiration time if specified
        expires_at = None
        if token_data.expires_in_days is not None:
            expires_at = datetime.utcnow() + timedelta(days=token_data.expires_in_days)

        # Create service account token using admin client
        result = self.admin_client.rpc(
            "create_service_account_token",
            {
                "org_id": organization_id,
                "token_name": token_data.name,
                "created_by": user_id,
                "expires_at": expires_at.isoformat() if expires_at else None,
            },
        ).execute()

        if not result.data:
            raise ValueError("Failed to create token")

        token_id = result.data[0]["token_id"]
        jwt_token = result.data[0]["jwt_token"]

        return Token(
            id=token_id,
            name=token_data.name,
            expires_at=expires_at,
            created_at=datetime.utcnow(),
            access_token=jwt_token,
        )

    async def list_tokens(self, organization_id: str) -> List[Token]:
        """List all tokens for an organization."""
        self.set_organization_id(organization_id)
        result = (
            self.client.table("tokens")
            .select("*")
            .eq("organization_id", organization_id)
            .execute()
        )
        return [Token(**token) for token in result.data]

    async def delete_token(self, organization_id: str, token_id: str):
        """Delete a token."""
        self.set_organization_id(organization_id)
        self.client.table("tokens").delete().eq("id", token_id).execute()

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
            response = requests.get(f"https://{pod_id}-10101.proxy.runpod.net/logs")
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
                            f"https://{pod_id}-10101.proxy.runpod.net/{run_id}"
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
