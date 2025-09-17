import logging
import os
from typing import Any, Dict, Optional

from supabase.lib.client_options import ClientOptions

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

# Reduce noise to only warnings+errors
for name in ["httpx", "httpx._client", "postgrest", "gotrue", "supabase"]:
    logging.getLogger(name).setLevel(logging.WARNING)


def create_authenticated_client(
    supabase_url: str, supabase_anon_key: str, auth_token: Optional[str] = None
):
    """Create a Supabase client with authentication.

    Args:
        supabase_url: Supabase project URL
        supabase_anon_key: Supabase anon key
        auth_token: Session token from Supabase auth (optional)
        api_key: OpenWeights API key starting with 'ow_' (optional)
    """
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    else:
        raise ValueError("No auth_token provided")

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


class OpenWeights:
    _INSTANCES = []

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
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
        self.supabase_url = supabase_url or os.environ.get(
            "SUPABASE_URL", "https://taofkfabrhpgtohaikst.supabase.co"
        )
        self.supabase_key = supabase_key or os.environ.get(
            "SUPABASE_ANON_KEY",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRhb2ZrZmFicmhwZ3RvaGFpa3N0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzE5MjkyMjcsImV4cCI6MjA0NzUwNTIyN30.KRufleTgprt16mfm0_91YjKIFZAne1-IW8buMxWVMeE",
        )
        self.auth_token = auth_token or os.getenv("OPENWEIGHTS_API_KEY")
        self.deploy_kwargs = deploy_kwargs

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Supabase URL and key must be provided either as arguments or environment variables"
            )

        if not self.auth_token:
            raise ValueError(
                "Authentication token must be provided either as argument or OPENWEIGHTS_API_KEY environment variable"
            )

        self._supabase = create_authenticated_client(
            self.supabase_url, self.supabase_key, self.auth_token
        )

        # Get organization ID from token
        self.organization_id = organization_id or self.get_organization_id()
        self.org_name = self.get_organization_name()
        print("Connected to org: ", self.org_name)

        # Initialize components with organization ID
        self.files = Files(self._supabase, self.organization_id)
        self.jobs = Jobs(self)
        self.runs = Runs(self)
        self.events = Events(self._supabase)
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
            .single()
            .execute()
        )
        if not result.data:
            raise ValueError("Could not determine organization ID from token")
        return result.data["value"]

    @property
    def run(self):
        if not self._current_run:
            self._current_run = Run(self, organization_id=self.organization_id)
        return self._current_run
