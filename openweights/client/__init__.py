import logging
import os
from typing import Any, Dict, Optional

# from supabase.lib.client_options import ClientOptions
from supabase import ClientOptions

from openweights.client.chat import AsyncChatCompletions, ChatCompletions
from openweights.client.decorators import supabase_retry
from openweights.client.events import Events
from openweights.client.files import (
    Files,
    validate_messages,
    validate_preference_dataset,
)
from openweights.client.jobs import Job, Jobs
from openweights.client.run import Run, Runs
from openweights.client.temporary_api import TemporaryApi
from openweights.client.utils import get_lora_rank, group_models_or_adapters_by_model
from supabase import create_client

logger = logging.getLogger(__name__)

# Reduce noise to only warnings+errors
for name in ["httpx", "httpx._client", "postgrest", "gotrue", "supabase"]:
    logging.getLogger(name).setLevel(logging.WARNING)


def exchange_api_token_for_jwt(
    supabase_url: str, supabase_anon_key: str, api_token: str
) -> str:
    """Exchange an OpenWeights API token for a short-lived JWT.

    Args:
        supabase_url: Supabase project URL
        supabase_anon_key: Supabase anon key
        api_token: OpenWeights API token (starts with 'ow_')

    Returns:
        JWT token for authenticating with Supabase
    """
    logger.info("Exchanging API token for JWT")
    # Create temporary client without auth
    temp_client = create_client(supabase_url, supabase_anon_key)

    # Call the exchange function
    response = temp_client.rpc(
        "exchange_api_token_for_jwt", {"api_token": api_token}
    ).execute()

    if not response.data:
        raise ValueError("Failed to exchange API token for JWT")
    logger.info("Successfully exchanged API token for JWT")
    return response.data


def create_authenticated_client(
    supabase_url: str, supabase_anon_key: str, auth_token: Optional[str] = None
):
    """Create a Supabase client with authentication.

    Args:
        supabase_url: Supabase project URL
        supabase_anon_key: Supabase anon key
        auth_token: Either a JWT token or an OpenWeights API token (starting with 'ow_')
    """
    if not auth_token:
        raise ValueError("No auth_token provided")

    # If it's an API token (starts with 'ow_'), exchange for JWT
    if auth_token.startswith("ow_"):
        jwt_token = exchange_api_token_for_jwt(
            supabase_url, supabase_anon_key, auth_token
        )
    else:
        # Assume it's already a JWT
        jwt_token = auth_token

    headers = {"Authorization": f"Bearer {jwt_token}"}

    options = ClientOptions(
        schema="public",
        headers=headers,
        auto_refresh_token=False,
        persist_session=False,
    )

    return create_client(supabase_url, supabase_anon_key, options)


_REGISTERED_JOBS = {}


def register(name: str):
    """Decorator to register a custom job class"""

    def register_job(cls):
        _REGISTERED_JOBS[name] = cls
        for ow in OpenWeights._INSTANCES:
            setattr(ow, name, cls(ow))
        return cls

    return register_job


_SUPABASE_URL = os.environ.get(
    "SUPABASE_URL", "https://cmaguqyuzweixkrqjvnf.supabase.co"
)
_SUPABASE_ANON_KEY = os.environ.get(
    "SUPABASE_ANON_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNtYWd1cXl1endlaXhrcnFqdm5mIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjE2ODA1ODIsImV4cCI6MjA3NzI1NjU4Mn0.SlD0g3sWHsc3_SKEofR6Y6H01oWiEYBlmXYQiw0379s",
)


class OpenWeights:
    _INSTANCES = []

    def __init__(
        self,
        supabase_url: Optional[str] = _SUPABASE_URL,
        supabase_key: Optional[str] = _SUPABASE_ANON_KEY,
        auth_token: Optional[str] = None,
        organization_id: Optional[str] = None,
        use_async: bool = False,
        deploy_kwargs: Dict[str, Any] = {"max_model_len": 2048},
    ):
        """Initialize OpenWeights client

        Args:
            supabase_url: Supabase project URL (or SUPABASE_URL env var)
            supabase_key: Supabase anon key (or SUPABASE_ANON_KEY env var)
            auth_token: Authentication token (or OPENWEIGHTS_API_KEY env var)
                       Can be either a session token or a service account JWT token
        """
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.auth_token = auth_token or os.getenv("OPENWEIGHTS_API_KEY")
        self.deploy_kwargs = deploy_kwargs

        if not self.auth_token:
            raise ValueError(
                "Authentication token must be provided either as argument or OPENWEIGHTS_API_KEY environment variable"
            )

        logger.info("Creating authenticated Supabase client")
        self._supabase = create_authenticated_client(
            self.supabase_url, self.supabase_key, self.auth_token
        )

        # Store reference to this instance in the supabase client for token refresh
        self._supabase._ow_instance = self

        # Get organization ID from token
        self.organization_id = organization_id or self.get_organization_id()
        self.org_name = self.get_organization_name()
        logger.info(
            f"Initialized OpenWeights client for organization: {self.org_name} (ID: {self.organization_id})"
        )

        # Initialize components with organization ID
        self.files = Files(self, self.organization_id)
        self.jobs = Jobs(self)
        self.runs = Runs(self)
        self.events = Events(self)
        self.async_chat = AsyncChatCompletions(self, deploy_kwargs=self.deploy_kwargs)
        self.sync_chat = ChatCompletions(self, deploy_kwargs=self.deploy_kwargs)
        self.chat = self.async_chat if use_async else self.sync_chat

        self._current_run = None
        self.hf_org = self.get_hf_org()

        for name, cls in _REGISTERED_JOBS.items():
            setattr(self, name, cls(self))
        OpenWeights._INSTANCES.append(self)

    @supabase_retry()
    def get_organization_id(self) -> str:
        """Get the organization ID associated with the current token"""
        result = self._supabase.rpc("get_organization_from_token").execute()
        if not result.data:
            raise ValueError("Could not determine organization ID from token")
        return result.data

    @supabase_retry()
    def get_organization_name(self):
        """Get the organization ID associated with the current token"""
        result = (
            self._supabase.table("organizations")
            .select("*")
            .eq("id", self.organization_id)
            .single()
            .execute()
        )
        return result.data["name"]

    @supabase_retry()
    def get_hf_org(self):
        """Get organization secrets from the database."""
        result = (
            self._supabase.table("organization_secrets")
            .select("value")
            .eq("organization_id", self.organization_id)
            .eq("name", "HF_ORG")
            .execute()
        )
        if not result.data or len(result.data) == 0:
            return os.environ.get("HF_ORG") or os.environ.get("HF_USER")
        return result.data[0]["value"]

    def _refresh_jwt(self) -> None:
        """Refresh the JWT token by exchanging the API token again.

        This is called automatically by @supabase_retry when a 401 error occurs.
        """
        # Only refresh if we have an ow_ API token (not a raw JWT)
        if not self.auth_token.startswith("ow_"):
            raise ValueError("Cannot refresh JWT: auth_token is not an ow_ API token")

        logger.info("Refreshing JWT token")
        # Get new client
        self._supabase = create_authenticated_client(
            self.supabase_url, self.supabase_key, self.auth_token
        )

        # Restore reference to this instance in the supabase client for future token refresh
        self._supabase._ow_instance = self
        logger.info("JWT token refreshed successfully")

    @property
    def run(self):
        if not self._current_run:
            self._current_run = Run(self, organization_id=self.organization_id)
        return self._current_run
