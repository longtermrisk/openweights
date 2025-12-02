import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta, timezone

from openai import AsyncOpenAI, OpenAI

from openweights.client.decorators import openai_retry

logger = logging.getLogger(__name__)

APIS = {}


class TemporaryApi:
    def __init__(self, ow, job_id):
        self._ow = ow
        self.job_id = job_id

        self.pod_id = None
        self.base_url = None
        self.api_key = None
        self._timeout_thread = None
        self._stop_timeout_thread = False

        self.sem = None

    def up(self):
        logger.info(f"Starting API for job: {self.job_id}")
        self._stop_timeout_thread = False
        self._timeout_thread = threading.Thread(
            target=self._manage_timeout, daemon=True
        )
        self._timeout_thread.start()

        # Poll until status is 'in_progress'
        while True:
            job = self._ow.jobs.retrieve(self.job_id)
            if job["status"] == "in_progress":
                break
            elif job["status"] in ["failed", "canceled"]:
                logger.warning(f"Job {self.job_id} {job['status']}, restarting...")
                return self.up()
            time.sleep(5)
        # Get worker
        worker = (
            self._ow._supabase.table("worker")
            .select("*")
            .eq("id", job["worker_id"])
            .single()
            .execute()
            .data
        )
        self.pod_id = worker["pod_id"]
        self.base_url = f"https://{self.pod_id}-8000.proxy.runpod.net/v1"
        logger.info(f"API endpoint: {self.base_url}")
        openai = OpenAI(api_key="no-api-key-required", base_url=self.base_url)
        self.wait_until_ready(openai, job["params"]["model"])
        APIS[job["params"]["model"]] = self

        self.sem = asyncio.Semaphore(job["params"]["max_num_seqs"])
        self.async_client = AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, max_retries=1
        )
        self.sync_client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, max_retries=1
        )
        logger.info(f"API ready for job: {self.job_id}")
        return self.sync_client

    def __enter__(self):
        return self.up()

    @openai_retry(interval=10, max_time=3600, max_tries=3600)
    def wait_until_ready(self, openai, model):
        logger.info(f"Waiting for model API to be ready: {model}")
        openai.chat.completions.create(
            model=model, messages=[dict(role="user", content="Hello")], max_tokens=200
        )
        logger.info(f"Model API ready: {model}")

    @openai_retry(interval=1, max_tries=10)
    async def async_up(self):
        logger.info(f"Starting async API for job: {self.job_id}")
        self._stop_timeout_thread = False
        self._timeout_thread = threading.Thread(
            target=self._manage_timeout, daemon=True
        )
        self._timeout_thread.start()
        while True:
            job = self._ow.jobs.retrieve(self.job_id)
            if job["status"] == "in_progress":
                break
            elif job["status"] in ["failed", "canceled"]:
                # Reset to pending and try again
                logger.warning(f"Job {self.job_id} {job['status']}, restarting...")
                self._ow.jobs.restart(self.job_id)
                return self.up()
            await asyncio.sleep(5)
        # Get worker
        worker = (
            self._ow._supabase.table("worker")
            .select("*")
            .eq("id", job["worker_id"])
            .single()
            .execute()
            .data
        )
        self.pod_id = worker["pod_id"]
        self.base_url = f"https://{self.pod_id}-8000.proxy.runpod.net/v1"
        self.api_key = job["params"].get("api_key", "api_key")
        logger.info(f"API endpoint: {self.base_url}")

        openai = OpenAI(api_key=self.api_key, base_url=self.base_url)
        await self.async_wait_until_ready(openai, job["params"]["model"])
        APIS[job["params"]["model"]] = self
        logger.info(f"API ready: {self.base_url}")

        self.sem = asyncio.Semaphore(job["params"]["max_num_seqs"])
        self.async_client = AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, max_retries=1, timeout=1800
        )
        self.sync_client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, max_retries=1, timeout=1800
        )
        return self.async_client

    async def async_wait_until_ready(self, openai, model):
        logger.info(f"Waiting for model to be ready: {model}")
        for _ in range(120):
            await asyncio.sleep(10)
            try:
                openai.chat.completions.create(
                    model=model, messages=[dict(role="user", content="Hello")]
                )
                logger.info(f"Model is ready: {model}")
                return
            except Exception as e:
                if "<!DOCTYPE html>" in str(e) and "<title>" in str(e):
                    title_content = str(e).split("<title>")[1].split("</title>")[0]
                    logger.debug(f"Waiting for API, error: {title_content}")
                else:
                    logger.debug(f"Waiting for API, error: {' '.join(str(e).split()[:20])}")

    async def __aenter__(self):
        return await self.async_up()

    def _manage_timeout(self):
        """Background thread to update job timeout and monitor job health."""
        while not self._stop_timeout_thread:
            try:
                # Set timeout to 15 minutes from now
                new_timeout = datetime.now(timezone.utc) + timedelta(minutes=15)
                logger.debug(f"Updating job timeout to {new_timeout}")
                response = (
                    self._ow._supabase.table("jobs")
                    .update({"timeout": new_timeout.isoformat()})
                    .eq("id", self.job_id)
                    .execute()
                )
                job = response.data[0]

                # Check job status and handle failures
                if job["status"] in ["failed", "canceled"]:
                    logger.warning(
                        f"Job {self.job_id} is in {job['status']} state. Attempting to restart..."
                    )
                    try:
                        # Reset the job to pending state
                        self._ow.jobs.restart(self.job_id)
                        # Call up() to reinitialize the API
                        self.up()
                        logger.info(f"Successfully restarted job {self.job_id}")
                    except Exception as e:
                        logger.error(f"Error restarting job {self.job_id}: {e}")
                elif job["status"] == "completed":
                    logger.warning(
                        f"Job {self.job_id} is marked as completed but should be running. Restarting..."
                    )
                    try:
                        self._ow.jobs.restart(self.job_id)
                        self.up()
                        logger.info(f"Successfully restarted completed job {self.job_id}")
                    except Exception as e:
                        logger.error(f"Error restarting completed job {self.job_id}: {e}")

            except Exception as e:
                logger.error(f"Error in timeout management thread: {e}")
            time.sleep(60)

    def down(self):
        logger.info(f"Shutting down API for job: {self.job_id}")
        self._stop_timeout_thread = True
        if self._timeout_thread:
            self._timeout_thread.join(
                timeout=1.0
            )  # Wait for thread to finish with timeout
        self._ow.jobs.cancel(self.job_id)

    def __exit__(self, exc_type, exc_value, traceback):
        self.down()

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.down()
