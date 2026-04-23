"""
GPU Compatibility Test Suite
=============================

Spawns RunPod pods *directly* (bypassing the cluster manager) to verify which
GPU types can successfully run our Docker images for fine-tuning and inference.

Usage:
    # Run all GPU tests (default: both finetuning and inference)
    python tests/gpu_compatibility_check.py

    # Run only inference tests
    python tests/gpu_compatibility_check.py --job-types inference

    # Run only finetuning tests
    python tests/gpu_compatibility_check.py --job-types finetuning

    # Test a subset of GPUs
    python tests/gpu_compatibility_check.py --gpus L40 A100 H200

    # Custom timeout and polling interval
    python tests/gpu_compatibility_check.py --timeout 1800 --poll-interval 30

    # Use a specific model for testing
    python tests/gpu_compatibility_check.py --model unsloth/Llama-3.2-1B-Instruct

Environment variables required:
    OPENWEIGHTS_API_KEY   OpenWeights API token
    RUNPOD_API_KEY        RunPod API key
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

import runpod

# ---------------------------------------------------------------------------
# Add project root to path so we can import openweights
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from openweights import OpenWeights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU registry — maps short names to RunPod GPU IDs
# ---------------------------------------------------------------------------
TEST_GPUS: Dict[str, str] = {
    # Currently verified
    "L40": "NVIDIA L40",
    "A100": "NVIDIA A100 80GB PCIe",
    "A100S": "NVIDIA A100-SXM4-80GB",
    "H100N": "NVIDIA H100 NVL",
    "H100S": "NVIDIA H100 80GB HBM3",
    "H200": "NVIDIA H200",
    # Candidates to test
    "6000Ada": "NVIDIA RTX 6000 Ada Generation",
    "A4000": "NVIDIA RTX A4000",
    "A40": "NVIDIA A40",
    "A6000": "NVIDIA RTX A6000",
    "L40S": "NVIDIA L40S",
    "L4": "NVIDIA L4",
    "4000Ada": "NVIDIA RTX 4000 Ada Generation",
    "RTX4090": "NVIDIA GeForce RTX 4090",
    "A4500": "NVIDIA RTX A4500",  # RTX PRO 4500
    "2000Ada": "NVIDIA RTX 2000 Ada Generation",
    "A5000": "NVIDIA RTX A5000",
    "RTX3090": "NVIDIA GeForce RTX 3090",
    "RTX5090": "NVIDIA GeForce RTX 5090",
}

DOCKER_IMAGE = "nielsrolf/ow-default:unsloth2026.3.17-pt2.9.0-vllm-0.16.0-cu12.8-studio-release-v0.1.3-beta"
ALLOWED_CUDA_VERSIONS = ["12.8"]

# Small model that fits on any GPU with ≥16 GB VRAM
DEFAULT_TEST_MODEL = "unsloth/Llama-3.2-1B-Instruct"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _make_sft_dataset() -> str:
    """Return a minimal SFT dataset as a JSONL string (3 examples)."""
    examples = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Say hello {i}."},
                {"role": "assistant", "content": f"Hello {i}!"},
            ]
        }
        for i in range(3)
    ]
    return "\n".join(json.dumps(ex) for ex in examples)


def _make_inference_dataset() -> str:
    """Return a minimal inference input as a JSONL string (3 prompts)."""
    examples = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"What is {i} + {i}?"},
            ]
        }
        for i in range(3)
    ]
    return "\n".join(json.dumps(ex) for ex in examples)


def upload_dataset(ow: OpenWeights, content: str, purpose: str) -> str:
    """Upload a JSONL string as a file and return the file ID."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(content)
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as fh:
            file_obj = ow.files.create(fh, purpose=purpose)
        logger.info(f"Uploaded {purpose} dataset: {file_obj['id']}")
        return file_obj["id"]
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

JobType = Literal["finetuning", "inference"]


@dataclass
class GPUTestResult:
    gpu_short: str
    gpu_runpod_id: str
    job_type: JobType
    job_id: Optional[str] = None
    pod_id: Optional[str] = None
    status: str = (
        "not_started"  # not_started | pod_created | job_pending | in_progress | completed | failed | pod_error | timeout
    )
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def duration_s(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None


@dataclass
class TestSession:
    results: List[GPUTestResult] = field(default_factory=list)
    pod_ids: List[str] = field(default_factory=list)  # Track for cleanup

    def add(self, result: GPUTestResult):
        self.results.append(result)
        if result.pod_id:
            self.pod_ids.append(result.pod_id)

    def print_summary(self):
        """Print a human-readable results table."""
        print("\n" + "=" * 90)
        print("GPU COMPATIBILITY TEST RESULTS")
        print("=" * 90)
        print(f"{'GPU':<15} {'Job Type':<14} {'Status':<14} {'Duration':<12} {'Error'}")
        print("-" * 90)
        for r in sorted(self.results, key=lambda x: (x.job_type, x.gpu_short)):
            dur = f"{r.duration_s:.0f}s" if r.duration_s else "-"
            err = (r.error or "")[:40]
            print(f"{r.gpu_short:<15} {r.job_type:<14} {r.status:<14} {dur:<12} {err}")
        print("=" * 90)

        # Summary counts
        for jtype in ["finetuning", "inference"]:
            subset = [r for r in self.results if r.job_type == jtype]
            if not subset:
                continue
            passed = sum(1 for r in subset if r.status == "completed")
            failed = sum(
                1 for r in subset if r.status in ("failed", "pod_error", "timeout")
            )
            pending = sum(
                1
                for r in subset
                if r.status not in ("completed", "failed", "pod_error", "timeout")
            )
            print(f"\n{jtype}: {passed} passed, {failed} failed, {pending} other")

    def save_json(self, path: str):
        """Save results to a JSON file."""
        data = []
        for r in self.results:
            data.append(
                {
                    "gpu_short": r.gpu_short,
                    "gpu_runpod_id": r.gpu_runpod_id,
                    "job_type": r.job_type,
                    "job_id": r.job_id,
                    "pod_id": r.pod_id,
                    "status": r.status,
                    "error": r.error,
                    "duration_s": r.duration_s,
                }
            )
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# Pod management — directly via RunPod SDK
# ---------------------------------------------------------------------------


def create_test_pod(
    gpu_short: str,
    gpu_runpod_id: str,
    runpod_api_key: str,
    image: str = DOCKER_IMAGE,
) -> dict:
    """Create a RunPod pod for testing, bypassing the cluster manager."""
    worker_id = f"gpu-test-{gpu_short.lower()}-{uuid.uuid4().hex[:6]}"
    name = f"gpu-compat-test-{gpu_short.lower()}-{uuid.uuid4().hex[:4]}"

    env = {
        "WORKER_ID": worker_id,
        "DOCKER_IMAGE": image,
        "OW_DEV": "false",
        "TTL_HOURS": "1",  # Short TTL — these are throwaway pods
        "RUNPOD_API_KEY": runpod_api_key,
    }

    logger.info(f"Creating pod for GPU {gpu_short} ({gpu_runpod_id})...")
    pod = runpod.create_pod(
        name,
        image,
        gpu_runpod_id,
        cloud_type="ALL",
        support_public_ip=True,
        container_disk_in_gb=100,  # Minimal disk — we don't save anything
        volume_in_gb=0,
        volume_mount_path="/workspace",
        gpu_count=1,
        allowed_cuda_versions=ALLOWED_CUDA_VERSIONS,
        ports="8000/http,10101/http,22/tcp",
        start_ssh=True,
        env=env,
    )
    logger.info(f"Pod created for {gpu_short}: pod_id={pod['id']}")
    return pod


def terminate_pod(pod_id: str):
    """Terminate a RunPod pod. Logs but does not raise on failure."""
    try:
        runpod.terminate_pod(pod_id)
        logger.info(f"Terminated pod {pod_id}")
    except Exception as e:
        logger.warning(f"Failed to terminate pod {pod_id}: {e}")


# ---------------------------------------------------------------------------
# Job creation helpers — create jobs targeted at specific workers
# ---------------------------------------------------------------------------


def create_finetuning_job(
    ow: OpenWeights,
    model: str,
    training_file_id: str,
    allowed_hardware: List[str],
    gpu_short: str,
) -> dict:
    """Create a minimal fine-tuning job that targets a specific hardware type."""
    # job_id_suffix ensures each GPU gets a unique job (OpenWeights deduplicates
    # by content hash, so without this every GPU would reuse the first completed job).
    job = ow.fine_tuning.create(
        model=model,
        training_file=training_file_id,
        loss="sft",
        max_steps=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        logging_steps=1,
        save_steps=999999,  # Never save checkpoints
        merge_before_push=False,
        push_to_private=True,
        r=2,
        use_rslora=True,
        max_seq_length=128,
        requires_vram_gb=0,  # Disable VRAM filter — we use allowed_hardware
        allowed_hardware=allowed_hardware,
        job_id_suffix=f"gpu-test-{gpu_short}",
    )
    logger.info(f"Created finetuning job: {job['id']} targeting {allowed_hardware}")
    return job


def create_inference_job(
    ow: OpenWeights,
    model: str,
    input_file_id: str,
    allowed_hardware: List[str],
    gpu_short: str,
) -> dict:
    """Create a minimal inference job that targets a specific hardware type."""
    # job_id_suffix ensures each GPU gets a unique job.
    job = ow.inference.create(
        model=model,
        input_file_id=input_file_id,
        max_tokens=10,
        temperature=0.0,
        max_model_len=512,
        requires_vram_gb=0,  # Disable VRAM filter — we use allowed_hardware
        allowed_hardware=allowed_hardware,
        job_id_suffix=f"gpu-test-{gpu_short}",
    )
    logger.info(f"Created inference job: {job['id']} targeting {allowed_hardware}")
    return job


# ---------------------------------------------------------------------------
# Polling helpers
# ---------------------------------------------------------------------------


def wait_for_job(
    ow: OpenWeights,
    job_id: str,
    timeout_s: int = 900,
    poll_interval_s: int = 15,
) -> dict:
    """Poll a job until it reaches a terminal state or times out."""
    start = time.time()
    while time.time() - start < timeout_s:
        job = ow.jobs.retrieve(job_id)
        status = job["status"]
        if status in ("completed", "failed", "canceled"):
            return job
        time.sleep(poll_interval_s)
    return {"id": job_id, "status": "timeout"}


# ---------------------------------------------------------------------------
# Main test orchestration
# ---------------------------------------------------------------------------


def run_tests(
    gpus: List[str],
    job_types: List[JobType],
    model: str = DEFAULT_TEST_MODEL,
    timeout_s: int = 900,
    poll_interval_s: int = 15,
) -> TestSession:
    """
    Run GPU compatibility tests.

    For each GPU × job_type combination:
      1. Create a RunPod pod directly with that GPU type
      2. Create a job (finetuning or inference) targeting that pod's hardware type
      3. Wait for the job to complete or fail
      4. Terminate the pod
      5. Record the result

    We do NOT retry failed GPUs — the whole point is to discover which ones fail.
    """
    # --- Validate environment ---
    assert os.environ.get(
        "OPENWEIGHTS_API_KEY"
    ), "Missing required environment variable: OPENWEIGHTS_API_KEY"

    ow = OpenWeights()

    # RUNPOD_API_KEY: prefer env var, fall back to organization secrets in DB
    runpod_api_key = os.environ.get("RUNPOD_API_KEY")
    if not runpod_api_key:
        logger.info(
            "RUNPOD_API_KEY not in environment, fetching from organization secrets..."
        )
        result = (
            ow._supabase.table("organization_secrets")
            .select("name, value")
            .eq("organization_id", ow.organization_id)
            .execute()
        )
        for secret in result.data:
            if secret["name"] == "RUNPOD_API_KEY":
                runpod_api_key = secret["value"]
                break
        assert (
            runpod_api_key
        ), "RUNPOD_API_KEY not found in environment or organization secrets"
    runpod.api_key = runpod_api_key

    session = TestSession()

    # --- Upload test datasets once ---
    training_file_id = None
    inference_file_id = None

    if "finetuning" in job_types:
        logger.info("Uploading SFT training dataset...")
        training_file_id = upload_dataset(ow, _make_sft_dataset(), "conversations")

    if "inference" in job_types:
        logger.info("Uploading inference input dataset...")
        inference_file_id = upload_dataset(
            ow, _make_inference_dataset(), "conversations"
        )

    # --- Run tests sequentially per GPU ---
    # (Sequential because each GPU needs its own pod, and we want to clean up
    #  promptly to avoid unnecessary RunPod charges.)
    for gpu_short in gpus:
        gpu_runpod_id = TEST_GPUS.get(gpu_short)
        if gpu_runpod_id is None:
            logger.error(f"Unknown GPU short name: {gpu_short}. Skipping.")
            continue

        for job_type in job_types:
            result = GPUTestResult(
                gpu_short=gpu_short,
                gpu_runpod_id=gpu_runpod_id,
                job_type=job_type,
            )
            result.started_at = time.time()

            # 1. Create pod
            try:
                pod = create_test_pod(gpu_short, gpu_runpod_id, runpod_api_key)
                result.pod_id = pod["id"]
                result.status = "pod_created"
                session.add(result)
            except Exception as e:
                result.status = "pod_error"
                result.error = str(e)
                result.finished_at = time.time()
                session.add(result)
                logger.error(f"[{gpu_short}/{job_type}] Pod creation failed: {e}")
                continue

            # 2. Create job
            # The worker determines its hardware_type by matching the GPU name
            # against the GPUs dict keys. The allowed_hardware values must use
            # the short names from the GPUs dict (e.g. "1x A6000", not the full
            # RunPod ID). GPUs not in the GPUs dict won't be matchable by the
            # worker, so we skip job creation for them and report the issue.
            try:
                from openweights.cluster.start_runpod import GPUs as KNOWN_GPUS

                if gpu_short not in KNOWN_GPUS:
                    result.status = "failed"
                    result.error = (
                        f"GPU '{gpu_short}' not in GPUs dict — worker cannot "
                        f"match it. Add it to openweights/cluster/start_runpod.py first."
                    )
                    result.finished_at = time.time()
                    terminate_pod(result.pod_id)
                    logger.error(f"[{gpu_short}/{job_type}] {result.error}")
                    continue

                hardware_types = [f"1x {gpu_short}"]
                if job_type == "finetuning":
                    assert training_file_id is not None
                    job = create_finetuning_job(
                        ow, model, training_file_id, hardware_types, gpu_short
                    )
                else:
                    assert inference_file_id is not None
                    job = create_inference_job(
                        ow, model, inference_file_id, hardware_types, gpu_short
                    )
                result.job_id = job["id"]
                result.status = "job_pending"
            except Exception as e:
                result.status = "failed"
                result.error = f"Job creation failed: {e}"
                result.finished_at = time.time()
                terminate_pod(result.pod_id)
                logger.error(f"[{gpu_short}/{job_type}] Job creation failed: {e}")
                continue

            # 3. Wait for job
            try:
                logger.info(
                    f"[{gpu_short}/{job_type}] Waiting for job {result.job_id} "
                    f"(timeout={timeout_s}s)..."
                )
                final_job = wait_for_job(ow, result.job_id, timeout_s, poll_interval_s)
                result.status = final_job["status"]
                if result.status == "failed":
                    # Try to get error info from the job
                    result.error = _extract_job_error(ow, final_job)
                elif result.status == "timeout":
                    result.error = f"Job did not finish within {timeout_s}s"
            except Exception as e:
                result.status = "failed"
                result.error = f"Polling error: {e}"
            finally:
                result.finished_at = time.time()
                # 4. Always terminate the pod
                terminate_pod(result.pod_id)

            dur = result.duration_s or 0
            logger.info(
                f"[{gpu_short}/{job_type}] Result: {result.status} "
                f"(took {dur:.0f}s)"
            )

    return session


def _extract_job_error(ow: OpenWeights, job: dict) -> Optional[str]:
    """Try to extract a useful error message from a failed job."""
    try:
        # Check if there's a run with logs
        runs = (
            ow._supabase.table("runs")
            .select("*")
            .eq("job_id", job["id"])
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data
        )
        if runs and runs[0].get("log_file"):
            # Return the log file ID so the user can inspect it
            return f"log_file={runs[0]['log_file']}"
    except Exception:
        pass
    # Job can be a Job dataclass or a plain dict (timeout sentinel)
    if isinstance(job, dict):
        return job.get("error") or "Unknown error (check logs)"
    return getattr(job, "error", None) or "Unknown error (check logs)"


# ---------------------------------------------------------------------------
# Cleanup safety net
# ---------------------------------------------------------------------------


def cleanup_test_pods(session: TestSession):
    """Terminate any pods that might still be running."""
    for pod_id in set(session.pod_ids):
        terminate_pod(pod_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test GPU compatibility with OpenWeights docker images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=list(TEST_GPUS.keys()),
        help=f"GPU short names to test. Available: {', '.join(TEST_GPUS.keys())}",
    )
    parser.add_argument(
        "--job-types",
        nargs="+",
        default=["finetuning", "inference"],
        choices=["finetuning", "inference"],
        help="Which job types to test (default: both)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_TEST_MODEL,
        help=f"HuggingFace model to use for tests (default: {DEFAULT_TEST_MODEL})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Max seconds to wait per job (default: 900)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=15,
        help="Seconds between job status polls (default: 15)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save JSON results (default: results/gpu_compat_<timestamp>.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate GPU names
    unknown = [g for g in args.gpus if g not in TEST_GPUS]
    if unknown:
        logger.error(
            f"Unknown GPU names: {unknown}. "
            f"Available: {', '.join(sorted(TEST_GPUS.keys()))}"
        )
        sys.exit(1)

    logger.info(f"GPUs to test: {args.gpus}")
    logger.info(f"Job types: {args.job_types}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Timeout per job: {args.timeout}s")

    session = None
    try:
        session = run_tests(
            gpus=args.gpus,
            job_types=args.job_types,
            model=args.model,
            timeout_s=args.timeout,
            poll_interval_s=args.poll_interval,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted! Cleaning up pods...")
        if session:
            cleanup_test_pods(session)
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error during test run")
        if session:
            cleanup_test_pods(session)
        sys.exit(1)

    # Print results
    session.print_summary()

    # Save JSON results
    output_path = args.output
    if output_path is None:
        os.makedirs("results", exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = f"results/gpu_compat_{ts}.json"
    session.save_json(output_path)


if __name__ == "__main__":
    main()
