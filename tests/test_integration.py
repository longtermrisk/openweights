"""
Integration tests for OpenWeights.

These tests run against a live Supabase database and test the full stack:
- User signup and token management
- Worker execution
- Docker image building
- Cluster management with cookbook examples

Usage:
    python tests/test_integration.py
    python tests/test_integration.py --skip-until test_cluster_and_cookbook

Requirements:
    - .env.worker file must exist with SUPABASE_URL, SUPABASE_ANON_KEY, etc.
    - Access to dev Supabase database
    - Docker installed and running
    - RunPod API key configured
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field

from openweights import Jobs, OpenWeights, register
from openweights.images import OW_CLUSTER_IMAGE, OW_UNSLOTH_IMAGE, OW_VLLM_IMAGE

# Ordered list of cookbook examples to run sequentially.
# Cheaper/faster examples first, expensive ones later.
# Paths are relative to the cookbook/ directory.
COOKBOOK_EXAMPLES = [
    "sft/lora_qwen3_5_0_8b.py",
    "sft/lora_qwen3_4b.py",
    "sft/lora_gemma3_4b.py",
    "sft/lora_olmo3_7b.py",
    "sft/lora_qwen3_5_35b_a3b.py",
    "sft/qlora_llama3_70b.py",
    "sft/logprob_tracking.py",
    "sft/sampling_callback.py",
    "sft/token_level_weighted_sft.py",
    "preference_learning/llama3_dpo.py",
    "preference_learning/llama3_orpo.py",
    "inference/run_inference.py",
    "inference/qwen36_inference.py",
    "custom_job/client_side.py",
    "api-deployment/context_manager_api.py",
]


class TestResult:
    """Track test results"""

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error: Optional[str] = None
        self.duration: float = 0.0
        self.start_time = time.time()

    def mark_passed(self):
        self.passed = True
        self.duration = time.time() - self.start_time

    def mark_failed(self, error: str):
        self.passed = False
        self.error = error
        self.duration = time.time() - self.start_time

    def __str__(self):
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        result = f"{status} - {self.name} ({self.duration:.2f}s)"
        if self.error:
            result += f"\n  Error: {self.error}"
        return result


class IntegrationTestRunner:
    """Run integration tests for OpenWeights"""

    def __init__(self, debug: bool = False):
        self.results: List[TestResult] = []
        self.env_backup: Optional[str] = None
        self.test_token: Optional[str] = None
        self.initial_token: Optional[str] = None
        self.debug = debug
        self.python_executable = sys.executable

        # Paths
        self.repo_root = Path(__file__).parent.parent
        self.env_worker_path = self.repo_root / ".env.worker"
        self.env_backup_path = self.repo_root / ".env.worker.backup"
        self.env_test_path = self.repo_root / ".env.test"
        self.logs_dir = self.repo_root / "logs"

        # Create logs directory structure
        self.logs_dir.mkdir(exist_ok=True)
        (self.logs_dir / "cookbook").mkdir(exist_ok=True)

    DOCKER_BUILD_TIMEOUT_SECONDS = 3600

    def backup_env(self):
        """Backup .env.worker file"""
        if self.env_worker_path.exists():
            shutil.copy(self.env_worker_path, self.env_backup_path)
            print(f"Backed up .env.worker to {self.env_backup_path}")

    def restore_env(self):
        """Restore .env.worker file"""
        if self.env_backup_path.exists():
            shutil.copy(self.env_backup_path, self.env_worker_path)
            self.env_backup_path.unlink()
            print(f"Restored .env.worker from backup")

    def save_test_state(self):
        """Save current test state to .env.test for resumption"""
        if self.env_worker_path.exists():
            shutil.copy(self.env_worker_path, self.env_test_path)
            print(f"Saved test state to {self.env_test_path}")

    def load_test_state(self):
        """Load test state from .env.test when skipping tests"""
        if self.env_test_path.exists():
            shutil.copy(self.env_test_path, self.env_worker_path)
            print(f"Loaded test state from {self.env_test_path}")

            # Extract tokens from the loaded env
            from dotenv import load_dotenv

            load_dotenv(self.env_worker_path, override=True)

            # Try to extract the token
            import re

            env_content = self.env_worker_path.read_text()
            token_match = re.search(
                r"OPENWEIGHTS_API_KEY=(ow_[a-f0-9]{48})", env_content
            )
            if token_match:
                self.initial_token = token_match.group(1)
                print(
                    f"Loaded initial token from test state: {self.initial_token[:20]}..."
                )
            else:
                print("Warning: Could not extract token from .env.test")
        else:
            raise FileNotFoundError(
                f".env.test not found at {self.env_test_path}. "
                "You must run the full test suite first before using --skip-until."
            )

    def update_env_token(self, token: str):
        """Update OPENWEIGHTS_API_KEY in .env.worker"""
        if not self.env_worker_path.exists():
            raise FileNotFoundError(f".env.worker not found at {self.env_worker_path}")

        lines = self.env_worker_path.read_text().splitlines()
        updated_lines = []
        found = False

        for line in lines:
            if line.startswith("OPENWEIGHTS_API_KEY="):
                updated_lines.append(f"OPENWEIGHTS_API_KEY={token}")
                found = True
            else:
                updated_lines.append(line)

        if not found:
            updated_lines.append(f"OPENWEIGHTS_API_KEY={token}")

        self.env_worker_path.write_text("\n".join(updated_lines) + "\n")
        print(f"Updated OPENWEIGHTS_API_KEY in .env.worker")

    def _get_env_with_token(self) -> Dict[str, str]:
        """Get environment dict with current token from .env.worker"""
        from dotenv import dotenv_values

        env_from_file = dotenv_values(self.env_worker_path)

        cmd_env = os.environ.copy()
        cmd_env.update(env_from_file)

        return cmd_env

    def _prompt_manual_execution(
        self, command_desc: str, cwd: Optional[Path] = None
    ) -> bool:
        """Prompt user to run command manually in debug mode

        Returns:
            True if user wants to run manually, False to auto-run
        """
        if not self.debug:
            return False

        print("\n" + "=" * 80)
        print("DEBUG MODE - SUBPROCESS CONTROL")
        print("=" * 80)
        print(f"About to start: {command_desc}")
        print(f"Working directory: {cwd or self.repo_root}")
        print("\nOptions:")
        print("  [m] Run MANUALLY in a separate terminal (you control it)")
        print("  [a] AUTO-START as background subprocess (default)")
        print("=" * 80)

        response = input("Your choice [m/a]: ").strip().lower()
        return response in ["m", "manual"]

    def _start_subprocess(
        self,
        command: List[str],
        log_name: str,
        command_desc: str,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Optional[subprocess.Popen]:
        """Start a subprocess with logging or prompt for manual execution

        Args:
            command: Command to run (e.g., ["python", "-m", "openweights.cli", "worker"])
            log_name: Name for log file (e.g., "worker", "cluster", "cookbook/custom_job/client_side")
            command_desc: Human-readable command description (e.g., "ow worker")
            cwd: Working directory for the command

        Returns:
            Popen object if subprocess was started, None if running manually
        """
        # Check if user wants to run manually
        if self._prompt_manual_execution(command_desc, cwd):
            print("\n" + ">" * 80)
            print("MANUAL EXECUTION MODE")
            print(">" * 80)
            print(f"Please run the following command in a separate terminal:\n")
            print(f"  cd {cwd or self.repo_root}")
            print(f"  {command_desc}\n")
            print(">" * 80)
            print(
                "IMPORTANT: Start the command above, then press Enter here to continue..."
            )
            print(">" * 80)
            input("\nPress Enter after you've started the command: ")
            print("✓ Continuing with test (assuming manual process is running)...\n")
            return None

        # Auto-start mode: Create log file
        log_path = self.logs_dir / f"{log_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n✓ Auto-starting: {command_desc}")
        print(f"  Logging to: {log_path}\n")

        log_file = open(log_path, "w")

        process_env = self._get_env_with_token()
        if env:
            process_env.update(env)

        process = subprocess.Popen(
            command,
            cwd=cwd or self.repo_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=process_env,
        )

        # Store log file handle so it stays open
        process._log_file = log_file  # type: ignore

        return process

    def _cleanup_subprocess(
        self, process: Optional[subprocess.Popen], timeout: int = 10
    ):
        """Clean up a subprocess gracefully

        Args:
            process: Process to clean up (None if running manually)
            timeout: Seconds to wait before force killing
        """
        if process is None:
            # Manual mode - ask user to stop
            print("\n" + ">" * 80)
            print("MANUAL PROCESS CLEANUP")
            print(">" * 80)
            print("Please STOP the manually-run process (Ctrl+C in that terminal)")
            print(">" * 80)
            input("Press Enter after you've stopped the process: ")
            print("✓ Continuing...\n")
            return

        # Auto mode - terminate subprocess
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            print("⚠ Had to forcefully kill process")

        # Close log file if it exists
        if hasattr(process, "_log_file"):
            process._log_file.close()

    def run_cli_command(
        self,
        command: List[str],
        capture_output: bool = True,
        env: Optional[Dict[str, str]] = None,
    ) -> subprocess.CompletedProcess:
        """Run an ow CLI command"""
        full_command = [self.python_executable, "-m", "openweights.cli"] + command
        command_desc = "ow " + " ".join(command)

        # In debug mode, ask if user wants to run manually
        if self._prompt_manual_execution(command_desc, self.repo_root):
            print("\n" + ">" * 80)
            print("MANUAL EXECUTION MODE - CLI COMMAND")
            print(">" * 80)
            print(f"Please run the following command in a separate terminal:\n")
            print(f"  cd {self.repo_root}")
            print(f"  {command_desc}\n")
            print(">" * 80)
            print(
                "IMPORTANT: Run the command above, then press Enter here to continue..."
            )
            print(">" * 80)
            input("\nPress Enter after you've run the command: ")
            print("✓ Continuing with test (assuming command completed)...\n")

            # Return a mock result for manual execution
            # The test will continue but won't have actual output
            return subprocess.CompletedProcess(
                args=full_command,
                returncode=0,
                stdout="[Manual execution - no output captured]",
                stderr="",
            )

        # Auto mode - run normally
        print(f"Running: {' '.join(full_command)}")

        # Get environment with token from .env.worker
        cmd_env = self._get_env_with_token()

        # Apply any additional env overrides
        if env:
            cmd_env.update(env)

        result = subprocess.run(
            full_command,
            cwd=self.repo_root,
            capture_output=capture_output,
            text=True,
            timeout=300,  # 5 minute timeout
            env=cmd_env,
        )

        if result.stdout:
            print(f"STDOUT: {result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")

        return result

    def test_signup_and_tokens(self) -> TestResult:
        """Test signup, token creation, token usage, and token revocation"""
        result = TestResult("Signup and Token Management")

        try:
            print("\n" + "=" * 80)
            print("TEST: Signup and Token Management")
            print("=" * 80)

            # Step 1: Sign up (or login if already exists)
            print("\n1. Testing 'ow signup'...")

            # Generate random email for testing
            import secrets

            test_email = f"test-{secrets.token_hex(8)}@openweights.test"

            # Load env to get SUPABASE credentials
            from dotenv import load_dotenv

            load_dotenv(self.env_worker_path, override=True)

            signup_result = self.run_cli_command(
                [
                    "signup",
                    test_email,
                ]
            )

            if signup_result.returncode != 0:
                raise Exception(f"Signup failed: {signup_result.stderr}")

            # Extract initial token from output
            output = signup_result.stdout + signup_result.stderr

            # Try to extract token (format: ow_...)
            import re

            token_match = re.search(r"(ow_[a-f0-9]{48})", output)
            if token_match:
                self.initial_token = token_match.group(1)
                print(f"Extracted initial token: {self.initial_token[:20]}...")
            else:
                raise Exception("Could not extract initial token from signup output")

            # Update env with initial token
            self.update_env_token(self.initial_token)

            # Step 2: Create a new token
            print("\n2. Testing 'ow token create'...")
            token_create_result = self.run_cli_command(
                ["token", "create", "--name", "integration-test-token"]
            )

            if token_create_result.returncode != 0:
                raise Exception(f"Token creation failed: {token_create_result.stderr}")

            # Extract the new token
            token_match = re.search(r"(ow_[a-f0-9]{48})", token_create_result.stdout)
            if token_match:
                self.test_token = token_match.group(1)
                print(f"Created test token: {self.test_token[:20]}...")
            else:
                raise Exception("Could not extract test token from output")

            # Update env with test token
            self.update_env_token(self.test_token)

            # Step 3: Test token by listing jobs
            print("\n3. Testing 'ow ls' with test token...")
            ls_result = self.run_cli_command(["ls"])

            if ls_result.returncode != 0:
                raise Exception(f"'ow ls' failed with test token: {ls_result.stderr}")

            print("✓ Successfully listed jobs with test token")

            # Step 4: Test env import
            print("\n4. Testing 'ow env import MAX_WORKERS=2'...")

            # Create a temporary .env file for testing
            test_env_file = self.repo_root / ".env.test"
            test_env_file.write_text("MAX_WORKERS=2\n")

            try:
                # Use -y flag to skip confirmation prompt
                env_import_result = self.run_cli_command(
                    ["env", "import", str(test_env_file), "-y"]
                )

                if env_import_result.returncode != 0:
                    raise Exception(
                        f"'ow env import' failed: {env_import_result.stderr}"
                    )

                print("✓ Successfully imported environment variable")

                # Verify the env was actually imported by checking 'ow env show'
                print("\n4b. Verifying environment variable was imported...")
                env_show_result = self.run_cli_command(["env", "show"])

                if env_show_result.returncode != 0:
                    raise Exception(f"'ow env show' failed: {env_show_result.stderr}")

                # Check if MAX_WORKERS=2 appears in the output
                if "MAX_WORKERS=2" not in env_show_result.stdout:
                    raise Exception(
                        f"MAX_WORKERS=2 not found in 'ow env show' output. Output was:\n{env_show_result.stdout}"
                    )

                print("✓ Verified MAX_WORKERS=2 in organization secrets")

            finally:
                # Clean up test file
                if test_env_file.exists():
                    test_env_file.unlink()

            # Step 5: Get token ID for revocation
            print("\n5. Getting token ID for revocation...")

            # Query to get token ID
            from dotenv import load_dotenv

            load_dotenv(self.env_worker_path, override=True)
            ow = OpenWeights()

            tokens_result = (
                ow._supabase.table("api_tokens")
                .select("id")
                .eq("organization_id", ow.organization_id)
                .order("created_at", desc=True)
                .execute()
            )

            test_token_id = None
            for token_record in tokens_result.data:
                # The test token is the most recent one
                if token_record["id"]:
                    test_token_id = token_record["id"]
                    break

            if not test_token_id:
                raise Exception("Could not find test token ID")

            print(f"Found test token ID: {test_token_id}")

            # Step 6: Revoke the test token
            print("\n6. Testing 'ow token revoke'...")
            # Note: token revoke requires stdin confirmation, we'll need to handle that separately
            revoke_result = subprocess.run(
                [
                    self.python_executable,
                    "-m",
                    "openweights.cli",
                    "token",
                    "revoke",
                    "--token-id",
                    test_token_id,
                ],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                input="yes\n",
                timeout=30,
                env=self._get_env_with_token(),
            )

            if revoke_result.returncode != 0:
                raise Exception(f"Token revocation failed: {revoke_result.stderr}")

            print("✓ Successfully revoked test token")

            # Step 6: Verify token is invalid by trying to use it
            print("\n6. Verifying revoked token is invalid...")
            ls_result_after_revoke = self.run_cli_command(["ls"])

            if ls_result_after_revoke.returncode == 0:
                raise Exception(
                    "'ow ls' succeeded with revoked token (should have failed)"
                )

            print("✓ Confirmed revoked token is invalid")

            # Restore initial token for remaining tests
            self.update_env_token(self.initial_token)

            result.mark_passed()

        except Exception as e:
            result.mark_failed(str(e))

        self.results.append(result)
        return result

    def test_worker_execution(self) -> TestResult:
        """Test worker execution with addition job"""
        result = TestResult("Worker Execution")

        try:
            print("\n" + "=" * 80)
            print("TEST: Worker Execution")
            print("=" * 80)

            # Ensure we're using the initial token
            if not self.initial_token:
                raise Exception("Initial token not available")

            self.update_env_token(self.initial_token)

            # Define addition job inline
            from dotenv import load_dotenv

            load_dotenv(self.env_worker_path, override=True)

            ow = OpenWeights()

            class AdditionParams(BaseModel):
                a: float = Field(..., description="First number")
                b: float = Field(..., description="Second number")

            @register("addition")
            class AdditionJob(Jobs):
                mount = {
                    str(
                        self.repo_root / "cookbook/custom_job/worker_side.py"
                    ): "worker_side.py"
                }
                params = AdditionParams
                requires_vram_gb = 0

                def get_entrypoint(self, validated_params: AdditionParams) -> str:
                    params_json = json.dumps(validated_params.model_dump())
                    return f"python worker_side.py '{params_json}'"

            # Submit job
            print("\n1. Submitting addition job (5 + 9)...")
            job = ow.addition.create(a=5, b=9)
            job_id = job["id"]
            print(f"Created job: {job_id}")

            # Start worker in background
            print("\n2. Starting worker...")
            worker_process = self._start_subprocess(
                command=[self.python_executable, "-m", "openweights.cli", "worker"],
                log_name="worker",
                command_desc="ow worker",
                env={"DOCKER_IMAGE": Jobs.base_image},
            )

            # Wait for job completion (timeout after 5 minutes)
            print("\n3. Waiting for job completion...")
            max_wait = 300  # 5 minutes
            start_time = time.time()

            while time.time() - start_time < max_wait:
                job_status = ow.jobs.retrieve(job_id)
                status = job_status["status"]
                print(f"Job status: {status}")

                if status == "completed":
                    print("✓ Job completed successfully")
                    break
                elif status == "failed":
                    raise Exception(f"Job failed: {job_status}")

                time.sleep(5)
            else:
                raise Exception("Job did not complete within timeout")

            # Verify result
            print("\n4. Verifying job result...")
            events = ow.events.list(job_id=job_id)

            result_found = False
            for event in events:
                if event["data"].get("result") == 14.0:
                    result_found = True
                    print(f"✓ Found expected result: {event['data']['result']}")
                    break

            if not result_found:
                raise Exception("Expected result (14.0) not found in events")

            # Clean up worker
            print("\n5. Stopping worker...")
            self._cleanup_subprocess(worker_process)

            result.mark_passed()

        except Exception as e:
            result.mark_failed(str(e))

        self.results.append(result)
        return result

    def test_docker_build_and_push(self) -> TestResult:
        """Test building and pushing Docker images"""
        result = TestResult("Docker Build and Push")

        def build_and_push(image: str, dockerfile: Optional[str] = None) -> None:
            build_command = [
                "docker",
                "buildx",
                "build",
                "--progress=plain",
                "--platform",
                "linux/amd64",
            ]
            if dockerfile is not None:
                build_command.extend(["-f", dockerfile])
            build_command.extend(["-t", image, "--load", "."])

            build_result = subprocess.run(
                build_command,
                cwd=self.repo_root,
                timeout=self.DOCKER_BUILD_TIMEOUT_SECONDS,
            )
            if build_result.returncode != 0:
                raise Exception(f"Docker build failed for {image}")

            last_returncode = 1
            for attempt in range(1, 4):
                push_result = subprocess.run(
                    ["docker", "push", image],
                    cwd=self.repo_root,
                    timeout=self.DOCKER_BUILD_TIMEOUT_SECONDS,
                )
                last_returncode = push_result.returncode
                if push_result.returncode == 0:
                    break
                if attempt < 3:
                    print(f"Docker push failed for {image}; retrying ({attempt}/3)")
                    time.sleep(10)
            if last_returncode != 0:
                raise Exception(f"Docker push failed for {image}")

        def validate_cluster_image() -> None:
            validation_script = """
import importlib.metadata as metadata
import os
import sys

from openweights.client import (
    _SUPABASE_ANON_KEY,
    _SUPABASE_URL,
    create_authenticated_client,
)
from supabase import ClientOptions, create_client

print(f"supabase=={metadata.version('supabase')}")
options = ClientOptions(
    schema="public",
    headers={"Authorization": "Bearer fake.jwt.token"},
    auto_refresh_token=False,
    persist_session=False,
)
if not hasattr(options, "storage"):
    raise RuntimeError("Supabase ClientOptions is missing the storage attribute")

create_client(_SUPABASE_URL, _SUPABASE_ANON_KEY, options)
create_authenticated_client(_SUPABASE_URL, _SUPABASE_ANON_KEY, "fake.jwt.token")

import openweights.cluster.supervisor  # noqa: F401,E402

backend_path = os.path.join(os.getcwd(), "openweights", "dashboard", "backend")
sys.path.insert(0, backend_path)
import main  # noqa: F401,E402

print("cluster runtime validation ok")
""".strip()

            validation_result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--platform",
                    "linux/amd64",
                    "--entrypoint",
                    "python",
                    OW_CLUSTER_IMAGE,
                    "-c",
                    validation_script,
                ],
                cwd=self.repo_root,
                timeout=120,
            )
            if validation_result.returncode != 0:
                raise Exception(
                    f"Cluster image runtime validation failed for {OW_CLUSTER_IMAGE}"
                )

        try:
            print("\n" + "=" * 80)
            print("TEST: Docker Build and Push")
            print("=" * 80)

            # Get version from Jobs.base_image
            from openweights.client.jobs import Jobs

            version = Jobs.base_image.split(":")[-1]
            print(f"Using version: {version}")

            # Check if Docker is running
            print("\n1. Checking Docker...")
            docker_check = subprocess.run(
                ["docker", "info"], capture_output=True, timeout=10
            )

            if docker_check.returncode != 0:
                raise Exception("Docker is not running or not accessible")

            print("✓ Docker is running")

            # Build ow-unsloth image for AMD64
            print(f"\n2. Building and pushing ow-unsloth:{version} for AMD64...")
            build_and_push(OW_UNSLOTH_IMAGE)
            print(f"✓ Successfully built and pushed {OW_UNSLOTH_IMAGE}")

            # Build ow-vllm image for AMD64
            print(f"\n3. Building and pushing ow-vllm:{version} for AMD64...")
            build_and_push(OW_VLLM_IMAGE, "Dockerfile.vllm")
            print(f"✓ Successfully built and pushed {OW_VLLM_IMAGE}")

            # Build ow-cluster image for AMD64
            print(f"\n4. Building and pushing ow-cluster:{version} for AMD64...")
            build_and_push(OW_CLUSTER_IMAGE, "Dockerfile.cluster")
            print(f"✓ Successfully built and pushed {OW_CLUSTER_IMAGE}")

            print(f"\n5. Validating ow-cluster:{version} runtime imports...")
            validate_cluster_image()
            print(f"✓ Cluster runtime validation passed for {OW_CLUSTER_IMAGE}")

            print(
                f"\n✓ Docker build, push, and validation completed for version {version}"
            )

            result.mark_passed()

        except Exception as e:
            result.mark_failed(str(e))

        self.results.append(result)
        return result

    def _fetch_run_logs(self, ow, job_id: str, max_lines: int = 100) -> str:
        """Fetch the latest run's log file for a job.

        Returns the last `max_lines` lines as a string, or an error message.
        """
        try:
            runs = ow.runs.list(job_id=job_id)
            if not runs:
                return "(no runs found)"
            latest_run = runs[-1]  # most recent run (list is oldest-first)
            log_file = latest_run.log_file
            if not log_file:
                return "(no log file on run)"
            content = ow.files.content(log_file)
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            lines = content.splitlines()
            if len(lines) > max_lines:
                lines = lines[-max_lines:]
            return "\n".join(lines)
        except Exception as e:
            return f"(error fetching logs: {e})"

    def _cleanup_stale_cluster_workers(
        self, ow, stale_after_seconds: int = 120
    ) -> None:
        """Terminate leftover workers before cookbook runs.

        Interrupted integration runs can leave old workers behind. For the isolated
        integration-test org, it is safer to terminate any existing worker rows and
        their pods up front so retries always use a fresh image.
        """
        workers = (
            ow._supabase.table("worker")
            .select("id, status, ping, pod_id")
            .eq("organization_id", ow.organization_id)
            .in_("status", ["starting", "active", "shutdown"])
            .execute()
            .data
        )

        if not workers:
            return

        now = time.time()
        stale_worker_ids = []
        stale_pod_ids = []
        for worker in workers:
            if worker["status"] == "active":
                stale_worker_ids.append(worker["id"])
                if worker.get("pod_id"):
                    stale_pod_ids.append(worker["pod_id"])
                continue
            try:
                last_ping = worker.get("ping")
                if not last_ping:
                    stale_worker_ids.append(worker["id"])
                    if worker.get("pod_id"):
                        stale_pod_ids.append(worker["pod_id"])
                    continue
                last_ping_ts = (
                    datetime.fromisoformat(last_ping.replace("Z", "+00:00"))
                    .astimezone(timezone.utc)
                    .timestamp()
                )
            except ValueError:
                stale_worker_ids.append(worker["id"])
                if worker.get("pod_id"):
                    stale_pod_ids.append(worker["pod_id"])
                continue

            if now - last_ping_ts > stale_after_seconds:
                stale_worker_ids.append(worker["id"])
                if worker.get("pod_id"):
                    stale_pod_ids.append(worker["pod_id"])

        if not stale_worker_ids:
            return

        print(f"Cleaning up stale workers before cluster run: {stale_worker_ids}")
        if stale_pod_ids and os.getenv("RUNPOD_API_KEY"):
            import runpod

            runpod.api_key = os.environ["RUNPOD_API_KEY"]
            for pod_id in stale_pod_ids:
                try:
                    runpod.terminate_pod(pod_id)
                except Exception as exc:
                    print(f"Warning: failed to terminate stale pod {pod_id}: {exc}")
        (
            ow._supabase.table("worker")
            .update({"status": "terminated"})
            .in_("id", stale_worker_ids)
            .execute()
        )

    def _cleanup_stale_cluster_processes(self) -> None:
        """Terminate orphaned local cluster-manager processes from prior runs."""
        cmd = [
            "ps",
            "-axo",
            "pid=,command=",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        stale_pids: List[int] = []
        repo_root_str = str(self.repo_root)
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            pid_str, _, command = line.partition(" ")
            if (
                "openweights.cli cluster" in command
                and repo_root_str in command
                and int(pid_str) != os.getpid()
            ):
                stale_pids.append(int(pid_str))

        if not stale_pids:
            return

        print(f"Stopping stale cluster manager processes: {stale_pids}")
        for pid in stale_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue

        deadline = time.time() + 5
        while time.time() < deadline:
            alive = []
            for pid in stale_pids:
                try:
                    os.kill(pid, 0)
                    alive.append(pid)
                except ProcessLookupError:
                    continue
            if not alive:
                return
            time.sleep(0.2)

        for pid in stale_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue

    def _run_single_cookbook_example(
        self, ow, example_path: Path, cookbook_dir: Path
    ) -> Dict[str, Any]:
        """Run a single cookbook example and wait for its job to complete.

        Returns a result dict with keys:
            example, job_id, status, logs, duration, error
        """
        rel = example_path.relative_to(cookbook_dir)
        example_key = str(rel.with_suffix(""))

        start_time = time.time()

        # Snapshot job IDs + statuses + updated_at before running the script
        jobs_before = (
            ow._supabase.table("jobs")
            .select("id, status, updated_at")
            .eq("organization_id", ow.organization_id)
            .execute()
        )
        jobs_before_map = {
            j["id"]: {"status": j["status"], "updated_at": j["updated_at"]}
            for j in jobs_before.data
        }

        # Launch cookbook script as subprocess
        log_name = f"cookbook/{example_key}"
        process = self._start_subprocess(
            command=[self.python_executable, str(example_path)],
            log_name=log_name,
            command_desc=f"python cookbook/{rel}",
            cwd=example_path.parent,
        )

        # Poll for the job ID by diffing the job table
        job_id = None
        job_match_timeout = 60
        match_start = time.time()
        check_interval = 2

        while time.time() - match_start < job_match_timeout:
            jobs_after = (
                ow._supabase.table("jobs")
                .select("id, status, updated_at")
                .eq("organization_id", ow.organization_id)
                .execute()
            )
            jobs_after_map = {
                j["id"]: {"status": j["status"], "updated_at": j["updated_at"]}
                for j in jobs_after.data
            }

            # Detect new jobs, reset jobs, or jobs touched by the script
            candidates = []
            for jid, info in jobs_after_map.items():
                if jid not in jobs_before_map:
                    # Brand new job
                    candidates.append(jid)
                elif jobs_before_map[jid]["status"] in (
                    "completed",
                    "failed",
                    "canceled",
                ) and info["status"] in ("pending", "in_progress"):
                    # Job was reset from terminal state
                    candidates.append(jid)
                elif info["updated_at"] != jobs_before_map[jid]["updated_at"]:
                    # Job was touched (e.g. already completed, script re-submitted)
                    candidates.append(jid)

            if len(candidates) == 1:
                job_id = candidates[0]
                print(f"  Matched -> {job_id}")
                break
            elif len(candidates) > 1:
                job_id = candidates[0]
                print(f"  Multiple candidates for {example_key}: {candidates}")
                print(f"  Using {job_id}")
                break

            # Fallback: parse the subprocess log for a job ID even while it is
            # still running. Examples that reuse an existing pending job can
            # block waiting for completion before the DB row changes.
            if not candidates:
                log_path = self.logs_dir / f"{log_name}.log"
                if log_path and log_path.exists():
                    import re

                    log_text = log_path.read_text(errors="replace")
                    # Match job IDs from "Job already exists: ID" or "Job created: ID" log lines
                    matches = re.findall(
                        r"Job (?:already exists|created): (\S+)", log_text
                    )
                    if matches:
                        job_id = matches[
                            -1
                        ]  # use last match (most likely the submitted job)
                        print(f"  Matched (from subprocess log) -> {job_id}")
                        break

            elapsed = time.time() - match_start
            print(f"  Waiting for job to appear... ({elapsed:.1f}s)")
            time.sleep(check_interval)

        if job_id is None:
            duration = time.time() - start_time
            if process is not None:
                self._cleanup_subprocess(process, timeout=5)
            return {
                "example": example_key,
                "job_id": None,
                "status": "no_job_found",
                "logs": "",
                "duration": duration,
                "error": f"No job found after {job_match_timeout}s",
            }

        # Poll database for terminal status
        max_wait = 7200  # 2 hours per job
        poll_start = time.time()
        poll_interval = 30

        while time.time() - poll_start < max_wait:
            try:
                job_data = ow.jobs.retrieve(job_id)
                status = job_data["status"]

                if status in ("completed", "failed", "canceled"):
                    duration = time.time() - start_time

                    # "canceled" is expected for API deployment examples
                    # where the context manager cancels the job after use.
                    # Treat as success if the subprocess exited cleanly.
                    subprocess_ok = (
                        process is not None
                        and process.poll() is not None
                        and process.returncode == 0
                    )
                    is_success = status == "completed" or (
                        status == "canceled" and subprocess_ok
                    )

                    logs = ""
                    if not is_success:
                        logs = self._fetch_run_logs(ow, job_id)

                    if process is not None and process.poll() is None:
                        self._cleanup_subprocess(process, timeout=5)

                    return {
                        "example": example_key,
                        "job_id": job_id,
                        "status": "completed" if is_success else status,
                        "logs": logs,
                        "duration": duration,
                        "error": None if is_success else f"Job {status}",
                    }

                elapsed = time.time() - poll_start
                print(f"  [{elapsed:.0f}s] {example_key}: {status}")
            except Exception as e:
                print(f"  Error polling {example_key}: {e}")

            time.sleep(poll_interval)

        # Timeout
        duration = time.time() - start_time
        if process is not None and process.poll() is None:
            self._cleanup_subprocess(process, timeout=5)
        return {
            "example": example_key,
            "job_id": job_id,
            "status": "timeout",
            "logs": self._fetch_run_logs(ow, job_id),
            "duration": duration,
            "error": "Job did not complete within 2 hours",
        }

    def _print_cookbook_summary(self, completed_results: List[Dict[str, Any]]):
        """Print a summary table of cookbook example results."""
        print("\n" + "=" * 80)
        print("COOKBOOK EXAMPLES SUMMARY")
        print("=" * 80)

        print(f"\n{'Example':<45} {'Job ID':<25} {'Status':<12} {'Duration':<10}")
        print("-" * 92)

        for r in completed_results:
            example_name = r["example"][:44]
            job_id = (r.get("job_id") or "N/A")[:24]
            status = r["status"]
            duration = f"{r['duration']:.0f}s"

            if status == "completed":
                status_display = f"✓ {status}"
            elif status in ("failed", "canceled", "timeout", "no_job_found"):
                status_display = f"✗ {status}"
            else:
                status_display = f"⧗ {status}"

            print(
                f"{example_name:<45} {job_id:<25} {status_display:<12} {duration:<10}"
            )

        succeeded = [r for r in completed_results if r["status"] == "completed"]
        failed = [r for r in completed_results if r["status"] != "completed"]

        print("-" * 92)
        print(
            f"Total: {len(completed_results)} | Completed: {len(succeeded)} | Failed: {len(failed)}"
        )
        print("=" * 80)

    def test_cluster_and_cookbook(
        self, skip_until_cookbook: Optional[str] = None
    ) -> TestResult:
        """Test cluster management with cookbook examples.

        Runs examples sequentially with fail-fast behavior. If skip_until_cookbook
        is provided, skips examples until the matching key (e.g. "sft/lora_qwen3_4b").
        """
        result = TestResult("Cluster and Cookbook Examples")

        cluster_process = None
        completed_results: List[Dict[str, Any]] = []

        try:
            print("\n" + "=" * 80)
            print("TEST: Cluster and Cookbook Examples")
            print("=" * 80)

            # Ensure we're using the initial token
            if not self.initial_token:
                raise Exception("Initial token not available")

            self.update_env_token(self.initial_token)

            from dotenv import load_dotenv

            load_dotenv(self.env_worker_path, override=True)

            ow = OpenWeights()
            cookbook_dir = self.repo_root / "cookbook"

            # Resolve example paths from the ordered list
            examples = []
            for rel in COOKBOOK_EXAMPLES:
                path = cookbook_dir / rel
                if not path.exists():
                    print(f"Warning: cookbook example not found: {rel}")
                    continue
                examples.append(path)

            print(f"\nWill run {len(examples)} cookbook examples sequentially")
            for ex in examples:
                print(f"  - {ex.relative_to(cookbook_dir)}")

            # Handle skip_until_cookbook
            skip_mode = skip_until_cookbook is not None
            if skip_mode:
                print(f"\nSkipping until: {skip_until_cookbook}")

            self._cleanup_stale_cluster_processes()
            self._cleanup_stale_cluster_workers(ow)

            # Start cluster manager ONCE
            print("\n1. Starting cluster manager...")
            cluster_process = self._start_subprocess(
                command=[self.python_executable, "-m", "openweights.cli", "cluster"],
                log_name="cluster",
                command_desc="ow cluster",
            )

            # Run examples sequentially
            print("\n2. Running cookbook examples sequentially...")
            for example_path in examples:
                example_key = str(
                    example_path.relative_to(cookbook_dir).with_suffix("")
                )

                # Skip logic
                if skip_mode:
                    if example_key == skip_until_cookbook:
                        skip_mode = False
                        print(f"\nResuming from: {example_key}")
                    else:
                        print(f"Skipping: {example_key}")
                        continue

                print(f"\n{'—' * 60}")
                print(f"Running: {example_key}")
                print(f"{'—' * 60}")

                r = self._run_single_cookbook_example(ow, example_path, cookbook_dir)
                completed_results.append(r)

                if r["status"] == "completed":
                    print(f"✓ {example_key} completed in {r['duration']:.0f}s")
                else:
                    # Fail fast: print logs, show resume command, raise
                    print(f"\n✗ {example_key} {r['status']}")
                    if r["logs"]:
                        print(f"\n--- Run logs (last 100 lines) ---")
                        print(r["logs"])
                        print(f"--- End logs ---\n")

                    self._print_cookbook_summary(completed_results)

                    # Find the index of the current example to suggest resume
                    all_keys = [
                        str(Path(rel).with_suffix("")) for rel in COOKBOOK_EXAMPLES
                    ]
                    try:
                        idx = all_keys.index(example_key)
                        if idx + 1 < len(all_keys):
                            next_key = all_keys[idx + 1]
                            print(f"\nTo resume from the NEXT example:")
                            print(
                                f"  python tests/test_integration.py --skip-until-cookbook {next_key}"
                            )
                        print(f"\nTo retry THIS example:")
                        print(
                            f"  python tests/test_integration.py --skip-until-cookbook {example_key}"
                        )
                    except ValueError:
                        pass

                    raise Exception(
                        f"Cookbook example {example_key} {r['status']}: {r.get('error', '')}"
                    )

            # All examples passed
            self._print_cookbook_summary(completed_results)

            # Wait for workers to terminate
            print("\n3. Waiting for workers to terminate (up to 10 minutes)...")
            worker_termination_start = time.time()
            max_worker_wait = 600

            while time.time() - worker_termination_start < max_worker_wait:
                active_workers = (
                    ow._supabase.table("worker")
                    .select("id, status")
                    .eq("organization_id", ow.organization_id)
                    .in_("status", ["starting", "active"])
                    .execute()
                )

                if not active_workers.data:
                    elapsed = time.time() - worker_termination_start
                    print(f"✓ All workers terminated after {elapsed:.1f} seconds")
                    break

                elapsed = time.time() - worker_termination_start
                print(
                    f"[{elapsed:.0f}s] Waiting for {len(active_workers.data)} worker(s) to terminate..."
                )
                time.sleep(30)
            else:
                print(f"⚠ Warning: Some workers still active after 10 minutes")

            result.mark_passed()

        except Exception as e:
            result.mark_failed(str(e))

        finally:
            # Always clean up cluster manager
            if cluster_process is not None:
                print("\nStopping cluster manager...")
                self._cleanup_subprocess(cluster_process)

        self.results.append(result)
        return result

    def run_all_tests(
        self,
        skip_until: Optional[str] = None,
        skip_until_cookbook: Optional[str] = None,
    ):
        """Run all integration tests

        Args:
            skip_until: Optional test name to skip to. Will load state from .env.test
            skip_until_cookbook: Optional cookbook example key to skip to (e.g. "sft/lora_qwen3_4b").
                                Implies skip_until="test_cluster_and_cookbook".
        """
        # --skip-until-cookbook implies --skip-until test_cluster_and_cookbook
        if skip_until_cookbook and not skip_until:
            skip_until = "test_cluster_and_cookbook"

        print("\n" + "=" * 80)
        print("OPENWEIGHTS INTEGRATION TEST SUITE")
        print("=" * 80)
        print(f"Repository: {self.repo_root}")
        print(f"Environment: {self.env_worker_path}")
        if skip_until:
            print(f"Skipping until: {skip_until}")
        if skip_until_cookbook:
            print(f"Skipping cookbook until: {skip_until_cookbook}")
        print("\n")

        # Define test methods in order (cookbook test gets skip_until_cookbook kwarg)
        tests = [
            ("test_signup_and_tokens", self.test_signup_and_tokens, {}),
            ("test_worker_execution", self.test_worker_execution, {}),
            ("test_docker_build_and_push", self.test_docker_build_and_push, {}),
            (
                "test_cluster_and_cookbook",
                self.test_cluster_and_cookbook,
                {"skip_until_cookbook": skip_until_cookbook},
            ),
        ]

        # Validate skip_until if provided
        if skip_until:
            test_names = [name for name, _, _ in tests]
            if skip_until not in test_names:
                print(f"Error: Invalid test name '{skip_until}'")
                print(f"Valid test names: {', '.join(test_names)}")
                sys.exit(1)

        # Validate skip_until_cookbook if provided
        if skip_until_cookbook:
            valid_keys = [str(Path(rel).with_suffix("")) for rel in COOKBOOK_EXAMPLES]
            if skip_until_cookbook not in valid_keys:
                print(f"Error: Invalid cookbook example '{skip_until_cookbook}'")
                print(f"Valid examples: {', '.join(valid_keys)}")
                sys.exit(1)

        try:
            # Backup environment (unless we're skipping)
            if not skip_until:
                self.backup_env()
            else:
                # Load test state from .env.test
                self.load_test_state()

            # Run tests
            skip_mode = skip_until is not None
            for test_name, test_method, kwargs in tests:
                # If in skip mode, wait until we reach the target test
                if skip_mode:
                    if test_name == skip_until:
                        print(f"\n{'=' * 80}")
                        print(f"Resuming from: {test_name}")
                        print(f"{'=' * 80}\n")
                        skip_mode = False  # Start running tests from here
                    else:
                        print(f"Skipping: {test_name}")
                        continue

                # Run the test
                result = test_method(**kwargs)

                # Save test state after each successful test
                if result.passed:
                    self.save_test_state()

                # Stop immediately if a test fails before cluster test
                # (cluster test is the last one, so we can run it even if it might fail)
                if not result.passed and test_name != "test_cluster_and_cookbook":
                    print(f"\n{'=' * 80}")
                    print(f"STOPPING: {test_name} failed")
                    print(f"{'=' * 80}\n")
                    print(f"To resume from the next test, run:")
                    print(
                        f"  python tests/test_integration.py --skip-until <test_name>"
                    )
                    print(f"\nTest state saved to: {self.env_test_path}")
                    break

        finally:
            # Always restore environment (unless we're in skip mode and should keep test state)
            if not skip_until:
                self.restore_env()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80 + "\n")

        for result in self.results:
            print(result)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        print("\n" + "=" * 80)
        print(f"TOTAL: {passed}/{total} tests passed")
        print("=" * 80 + "\n")

        if passed == total:
            print("✓ All tests passed!")
            sys.exit(0)
        else:
            print("✗ Some tests failed")
            sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="OpenWeights Integration Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python tests/test_integration.py

  # Skip to a specific test (requires .env.test from previous run)
  python tests/test_integration.py --skip-until test_cluster_and_cookbook

  # Skip to a specific cookbook example (implies --skip-until test_cluster_and_cookbook)
  python tests/test_integration.py --skip-until-cookbook sft/lora_qwen3_4b

  # Run in debug mode (manually run subprocesses)
  python tests/test_integration.py --debug

Available tests (in order):
  - test_signup_and_tokens
  - test_worker_execution
  - test_docker_build_and_push
  - test_cluster_and_cookbook

Available cookbook examples (in order):
  """
        + "\n  ".join(f"- {str(Path(r).with_suffix(''))}" for r in COOKBOOK_EXAMPLES)
        + """
        """,
    )
    parser.add_argument(
        "--skip-until",
        type=str,
        help="Skip tests until the specified test name. Requires .env.test from a previous run.",
    )
    parser.add_argument(
        "--skip-until-cookbook",
        type=str,
        help="Skip cookbook examples until the specified key (e.g. 'sft/lora_qwen3_4b'). "
        "Implies --skip-until test_cluster_and_cookbook.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: prompt to run subprocesses manually in separate terminals.",
    )

    args = parser.parse_args()

    runner = IntegrationTestRunner(debug=args.debug)
    runner.run_all_tests(
        skip_until=args.skip_until,
        skip_until_cookbook=args.skip_until_cookbook,
    )


if __name__ == "__main__":
    main()
