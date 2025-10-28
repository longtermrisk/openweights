"""
Integration tests for OpenWeights.

These tests run against a live Supabase database and test the full stack:
- User signup and token management
- Worker execution
- Docker image building
- Cluster management with cookbook examples

Usage:
    python tests/test_integration.py

Requirements:
    - .env.worker file must exist with SUPABASE_URL, SUPABASE_ANON_KEY, etc.
    - Access to dev Supabase database
    - Docker installed and running
    - RunPod API key configured
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field

from openweights import Jobs, OpenWeights, register


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

    def __init__(self):
        self.results: List[TestResult] = []
        self.env_backup: Optional[str] = None
        self.test_token: Optional[str] = None
        self.initial_token: Optional[str] = None

        # Paths
        self.repo_root = Path(__file__).parent.parent
        self.env_worker_path = self.repo_root / ".env.worker"
        self.env_backup_path = self.repo_root / ".env.worker.backup"

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

    def run_cli_command(
        self,
        command: List[str],
        capture_output: bool = True,
        env: Optional[Dict[str, str]] = None,
    ) -> subprocess.CompletedProcess:
        """Run an ow CLI command"""
        full_command = ["python", "-m", "openweights.cli"] + command
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
                    "--supabase-url",
                    os.getenv("SUPABASE_URL"),
                    "--supabase-key",
                    os.getenv("SUPABASE_ANON_KEY"),
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
                    "python",
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
            worker_process = subprocess.Popen(
                ["python", "-m", "openweights.cli", "worker"],
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
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
            worker_process.terminate()
            try:
                worker_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker_process.kill()

            result.mark_passed()

        except Exception as e:
            result.mark_failed(str(e))

        self.results.append(result)
        return result

    def test_docker_build_and_push(self) -> TestResult:
        """Test building and pushing Docker images"""
        result = TestResult("Docker Build and Push")

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

            # Build ow-default image (worker image) for AMD64
            print(f"\n2. Building and pushing ow-default:{version} for AMD64...")
            build_result = subprocess.run(
                [
                    "docker",
                    "buildx",
                    "build",
                    "--platform",
                    "linux/amd64",
                    "-t",
                    f"nielsrolf/ow-default:{version}",
                    "--push",
                    ".",
                ],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=1200,  # 20 minute timeout for build+push
            )

            if build_result.returncode != 0:
                raise Exception(f"Docker build failed: {build_result.stderr}")

            print(f"✓ Successfully built and pushed ow-default:{version}")

            # Build ow-cluster image for AMD64
            print(f"\n3. Building and pushing ow-cluster:{version} for AMD64...")
            cluster_build_result = subprocess.run(
                [
                    "docker",
                    "buildx",
                    "build",
                    "--platform",
                    "linux/amd64",
                    "-f",
                    "Dockerfile.cluster",
                    "-t",
                    f"nielsrolf/ow-cluster:{version}",
                    "--push",
                    ".",
                ],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=1200,  # 20 minute timeout for build+push
            )

            if cluster_build_result.returncode != 0:
                raise Exception(
                    f"Cluster Docker build failed: {cluster_build_result.stderr}"
                )

            print(f"✓ Successfully built and pushed ow-cluster:{version}")

            print(f"\n✓ Docker build and push completed for version {version}")

            result.mark_passed()

        except Exception as e:
            result.mark_failed(str(e))

        self.results.append(result)
        return result

    def test_cluster_and_cookbook(self) -> TestResult:
        """Test cluster management with cookbook examples"""
        result = TestResult("Cluster and Cookbook Examples")

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

            # Find all cookbook examples
            cookbook_dir = self.repo_root / "cookbook"
            cookbook_examples = []

            for py_file in cookbook_dir.rglob("*.py"):
                # Skip gradio_ui as requested
                if py_file.name == "gradio_ui.py":
                    print(f"Skipping {py_file.relative_to(self.repo_root)}")
                    continue

                # Skip worker_side.py (helper file)
                if py_file.name == "worker_side.py":
                    continue

                cookbook_examples.append(py_file)

            print(f"\nFound {len(cookbook_examples)} cookbook examples to test")
            for example in cookbook_examples:
                print(f"  - {example.relative_to(self.repo_root)}")

            # Submit cookbook jobs one by one, matching each to its job ID
            print("\n1. Submitting cookbook jobs and matching to job IDs...")
            submitted_jobs = []

            for example in cookbook_examples:
                try:
                    print(f"\nSubmitting {example.name}...")

                    # Get job IDs before submission
                    jobs_before = (
                        ow._supabase.table("jobs")
                        .select("id")
                        .eq("organization_id", ow.organization_id)
                        .execute()
                    )
                    job_ids_before = {job["id"] for job in jobs_before.data}

                    # Start the process in background - it will wait for job completion
                    process = subprocess.Popen(
                        ["python", str(example)],
                        cwd=example.parent,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )

                    # Wait a moment for job to be created
                    time.sleep(5)

                    # Get job IDs after submission
                    jobs_after = (
                        ow._supabase.table("jobs")
                        .select("id")
                        .eq("organization_id", ow.organization_id)
                        .execute()
                    )
                    job_ids_after = {job["id"] for job in jobs_after.data}

                    # Find the new job ID (should be exactly one)
                    new_job_ids = job_ids_after - job_ids_before

                    if len(new_job_ids) == 1:
                        job_id = list(new_job_ids)[0]
                        submitted_jobs.append(
                            {
                                "example": example.name,
                                "process": process,
                                "job_id": job_id,
                                "completed": False,
                            }
                        )
                        print(f"  ✓ Matched {example.name} -> {job_id}")
                    elif len(new_job_ids) == 0:
                        print(
                            f"  ⚠ No new job found for {example.name}, waiting longer..."
                        )
                        time.sleep(10)
                        # Try again
                        jobs_after = (
                            ow._supabase.table("jobs")
                            .select("id")
                            .eq("organization_id", ow.organization_id)
                            .execute()
                        )
                        job_ids_after = {job["id"] for job in jobs_after.data}
                        new_job_ids = job_ids_after - job_ids_before

                        if len(new_job_ids) == 1:
                            job_id = list(new_job_ids)[0]
                            submitted_jobs.append(
                                {
                                    "example": example.name,
                                    "process": process,
                                    "job_id": job_id,
                                    "completed": False,
                                }
                            )
                            print(f"  ✓ Matched {example.name} -> {job_id}")
                        else:
                            print(f"  ✗ Still couldn't find job for {example.name}")
                            process.terminate()
                    else:
                        print(
                            f"  ⚠ Multiple new jobs found for {example.name}: {new_job_ids}"
                        )
                        # Take the most recent one
                        job_id = list(new_job_ids)[0]
                        submitted_jobs.append(
                            {
                                "example": example.name,
                                "process": process,
                                "job_id": job_id,
                                "completed": False,
                            }
                        )
                        print(f"  Using {job_id}")

                except Exception as e:
                    print(f"  ✗ Error with {example.name}: {e}")

            print(f"\nSuccessfully submitted and matched {len(submitted_jobs)} jobs")

            # Start cluster manager
            print("\n2. Starting cluster manager...")
            cluster_process = subprocess.Popen(
                ["python", "-m", "openweights.cli", "cluster"],
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Monitor job completion by checking job status in database
            # Some cookbook examples wait for completion, others just submit and exit
            print("\n3. Monitoring job completion...")
            print("Note: This may take several hours for fine-tuning jobs...")

            max_wait = 7200  # 2 hours max
            start_time = time.time()
            check_interval = 30  # Check every 30 seconds

            while time.time() - start_time < max_wait:
                all_done = True

                for job_info in submitted_jobs:
                    if job_info["completed"]:
                        continue

                    # Check if submission process has finished (for examples that wait)
                    if (
                        job_info["process"].poll() is not None
                        and "returncode" not in job_info
                    ):
                        job_info["returncode"] = job_info["process"].returncode

                    # Check actual job status in database (for all examples)
                    try:
                        job_status = ow.jobs.retrieve(job_info["job_id"])
                        current_status = job_status["status"]
                        job_info["current_status"] = current_status

                        if current_status in ["completed", "failed", "canceled"]:
                            if not job_info["completed"]:
                                job_info["completed"] = True
                                job_info["final_status"] = current_status
                                returncode_info = (
                                    f" (exit code: {job_info.get('returncode', 'N/A')})"
                                    if "returncode" in job_info
                                    else ""
                                )
                                print(
                                    f"✓ {job_info['example']}: {current_status}{returncode_info}"
                                )
                        else:
                            all_done = False

                    except Exception as e:
                        print(f"⚠ Error checking {job_info['example']}: {e}")
                        all_done = False

                if all_done:
                    print("\n✓ All jobs completed!")
                    break

                completed_count = sum(1 for j in submitted_jobs if j["completed"])
                pending_count = sum(
                    1
                    for j in submitted_jobs
                    if not j["completed"] and j.get("current_status") == "pending"
                )
                in_progress_count = sum(
                    1
                    for j in submitted_jobs
                    if not j["completed"] and j.get("current_status") == "in_progress"
                )

                print(
                    f"Progress: {completed_count}/{len(submitted_jobs)} completed | {in_progress_count} in progress | {pending_count} pending"
                )

                time.sleep(check_interval)
            else:
                print("\n⚠ Timeout reached, some jobs may still be running")
                # Kill any remaining submission processes
                for job_info in submitted_jobs:
                    if job_info["process"].poll() is None:
                        print(
                            f"Terminating submission process for {job_info['example']}..."
                        )
                        job_info["process"].terminate()

            # Wait for workers to terminate (cluster manager should terminate idle workers after 5 min)
            print("\n4. Waiting for workers to terminate (up to 10 minutes)...")
            print(
                "Cluster manager should terminate idle workers after 5 minutes of inactivity..."
            )
            worker_termination_start = time.time()
            max_worker_wait = 600  # 10 minutes

            while time.time() - worker_termination_start < max_worker_wait:
                # Check active workers
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
                print(
                    f"⚠ Warning: {len(active_workers.data)} worker(s) still active after 10 minutes"
                )

            # Now clean up cluster manager
            print("\n5. Stopping cluster manager...")
            cluster_process.terminate()
            try:
                cluster_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cluster_process.kill()
                print("⚠ Had to forcefully kill cluster manager")

            # Print summary table
            print("\n" + "=" * 80)
            print("COOKBOOK EXAMPLES SUMMARY")
            print("=" * 80)

            # Create a formatted table
            print(f"\n{'Example':<40} {'Job ID':<25} {'Status':<12} {'Exit Code':<10}")
            print("-" * 87)

            for job in submitted_jobs:
                example_name = job["example"][:39]  # Truncate if too long
                job_id = job.get("job_id", "N/A")[:24]
                status = job.get("final_status", job.get("current_status", "unknown"))
                exit_code = str(job.get("returncode", "N/A"))

                # Color coding
                if status == "completed":
                    status_display = f"✓ {status}"
                elif status in ["failed", "canceled"]:
                    status_display = f"✗ {status}"
                else:
                    status_display = f"⧗ {status}"

                print(
                    f"{example_name:<40} {job_id:<25} {status_display:<12} {exit_code:<10}"
                )

            # Summary counts
            completed = [
                j for j in submitted_jobs if j.get("final_status") == "completed"
            ]
            failed = [j for j in submitted_jobs if j.get("final_status") == "failed"]
            canceled = [
                j for j in submitted_jobs if j.get("final_status") == "canceled"
            ]
            pending = [j for j in submitted_jobs if not j.get("completed", False)]

            print("-" * 87)
            print(
                f"Total: {len(submitted_jobs)} | Completed: {len(completed)} | Failed: {len(failed)} | Canceled: {len(canceled)} | Pending: {len(pending)}"
            )
            print("=" * 80)

            # Mark test as passed if at least some jobs completed
            if len(completed) > 0:
                result.mark_passed()
            else:
                raise Exception("No cookbook jobs completed successfully")

        except Exception as e:
            result.mark_failed(str(e))

        self.results.append(result)
        return result

    def run_all_tests(self):
        """Run all integration tests"""
        print("\n" + "=" * 80)
        print("OPENWEIGHTS INTEGRATION TEST SUITE")
        print("=" * 80)
        print(f"Repository: {self.repo_root}")
        print(f"Environment: {self.env_worker_path}")
        print("\n")

        try:
            # Backup environment
            self.backup_env()

            # Run tests
            self.test_signup_and_tokens()
            self.test_worker_execution()
            self.test_docker_build_and_push()
            self.test_cluster_and_cookbook()

        finally:
            # Always restore environment
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
    runner = IntegrationTestRunner()
    runner.run_all_tests()


if __name__ == "__main__":
    main()
