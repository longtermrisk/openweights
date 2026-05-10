"""
Usage:
    python start_runpod.py --gpu A6000 --container_disk_in_gb 25 --volume_in_gb 30 --ttl_hours 24

TTL (Time To Live) Feature:
    - All pods have a default TTL of 24 hours to prevent runaway costs
    - TTL can be customized with --ttl_hours parameter
    - TTL can be extended from within the pod by updating ~/shutdown.txt with a new timestamp
    - Example to extend TTL from within pod:
      python3 -c "
      import datetime
      with open('~/shutdown.txt', 'w') as f:
          new_time = datetime.datetime.now() + datetime.timedelta(hours=48)
          f.write(new_time.isoformat())
      "

Note: possible unknown error with echo when running the script.
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from threading import RLock
from typing import Callable, Dict, List, Optional

import backoff
import fire
import paramiko
import runpod
from dotenv import load_dotenv
from scp import SCPClient

from openweights.images import OW_UNSLOTH_IMAGE, OW_VLLM_IMAGE

logger = logging.getLogger(__name__)

IMAGES = {
    "default": OW_UNSLOTH_IMAGE,
    "inference": OW_VLLM_IMAGE,
    "inference-debugging": OW_VLLM_IMAGE,
    "finetuning": OW_UNSLOTH_IMAGE,
    "torch": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
}

GPUs = {
    # References found at https://rest.runpod.io/v1/docs#v-0-106
    # GPUs for compute-intensive tasks (e.g. LoRAfinetuning)
    "6000Ada": "NVIDIA RTX 6000 Ada Generation",  # Not available with cuda 12.8
    "4000Ada": "NVIDIA RTX 4000 Ada Generation",
    "L40": "NVIDIA L40",
    "L40S": "NVIDIA L40S",  # not available with cuda 12.8
    "A30": "NVIDIA A30",  # not available with cuda 12.8
    # Belows, GPUs are only good for high-memory task (e.g., pretraining and vanilla finetuning)
    "A100": "NVIDIA A100 80GB PCIe",  # Default A100 - 80GB
    "A100S": "NVIDIA A100-SXM4-80GB",
    # "H100": "NVIDIA H100 PCIe", # not available with cuda 12.8
    "H100N": "NVIDIA H100 NVL",
    "H100S": "NVIDIA H100 80GB HBM3",
    "H200": "NVIDIA H200",
    "B200": "NVIDIA B200",  # CUDA error: CUDA error (/__w/xformers/xformers/third_party/flash-attention/hopper/flash_fwd_launch_template.h:175): no kernel image is available for execution on the device
    # Below, GPUs are cost inefficient
    "RTX4080": "NVIDIA GeForce RTX 4080",
    "RTX3090": "NVIDIA GeForce RTX 3090",
    "RTX3090Ti": "NVIDIA GeForce RTX 3090 Ti",
    "RTX4070Ti": "NVIDIA GeForce RTX 4070 Ti",
    "A4000_SFF": "NVIDIA RTX 4000 SFF Ada Generation",
    "A5000_ADA": "NVIDIA RTX 5000 Ada Generation",
    "MI300X": "AMD Instinct MI300X OAM",
    "2000Ada": "NVIDIA RTX 2000 Ada Generation",
    "A6000": "NVIDIA RTX A6000",
    "A4000": "NVIDIA RTX A4000",
    "A2000": "NVIDIA RTX A2000",
    "RTX4090": "NVIDIA GeForce RTX 4090",
    "A5000": "NVIDIA RTX A5000",
    "A40": "NVIDIA A40",
    "A4500": "NVIDIA RTX A4500",
    "RTX3080": "NVIDIA GeForce RTX 3080",
    "RTX3070": "NVIDIA GeForce RTX 3070",
    "RTX3080Ti": "NVIDIA GeForce RTX 3080 Ti",
    "L4": "NVIDIA L4",
}


VERIFIED_GPUs = {
    # References found at https://rest.runpod.io/v1/docs#v-0-106
    # GPUs for compute-intensive tasks (e.g. LoRAfinetuning)
    "6000Ada": "NVIDIA RTX 6000 Ada Generation",
    # "4000Ada": "NVIDIA RTX 4000 Ada Generation",  # untested (no RunPod stock)
    # L40 pods currently pass RunPod availability checks but frequently disappear
    # before the worker process starts on the CUDA 12.8 v0.10 images.
    # Keep L40 in GPUS for explicit allowed_hardware requests, but do not select it
    # automatically for production workers.
    # "L40": "NVIDIA L40",
    # "L40S": "NVIDIA L40S", # not available with cuda 12.8
    # "A30": "NVIDIA A30", # not available with cuda 12.8
    #
    # Belows, GPUs are only good for high-memory task (e.g., pretraining and vanilla finetuning)
    "A100": "NVIDIA A100 80GB PCIe",  # Default A100 - 80GB
    "A100S": "NVIDIA A100-SXM4-80GB",
    # "H100": "NVIDIA H100 PCIe", # not available with cuda 12.8
    "H100N": "NVIDIA H100 NVL",
    "H100S": "NVIDIA H100 80GB HBM3",
    "H200": "NVIDIA H200",
    # "B200": "NVIDIA B200",  CUDA error: CUDA error (/__w/xformers/xformers/third_party/flash-attention/hopper/flash_fwd_launch_template.h:175): no kernel image is available for execution on the device
    #
    # Below, GPUs are cost inefficient
    # "RTX4080": "NVIDIA GeForce RTX 4080",
    "RTX3090": "NVIDIA GeForce RTX 3090",
    # "RTX3090Ti": "NVIDIA GeForce RTX 3090 Ti",
    # "V100": "Tesla V100-SXM2-32GB",  # Default V100 - 32GB
    # "V100_32": "Tesla V100-SXM2-32GB",
    # "V100_16": "Tesla V100-SXM2-16GB",
    # "V100_16_FHHL": "Tesla V100-FHHL-16GB",
    # "V100_16_PCIE": "Tesla V100-PCIE-16GB",
    # "RTX4070Ti": "NVIDIA GeForce RTX 4070 Ti",
    # "A4000_SFF": "NVIDIA RTX 4000 SFF Ada Generation",
    # "A5000_ADA": "NVIDIA RTX 5000 Ada Generation",
    # "MI300X": "AMD Instinct MI300X OAM",
    # "2000Ada": "NVIDIA RTX 2000 Ada Generation",
    # "A6000": "NVIDIA RTX A6000",
    # "A4000": "NVIDIA RTX A4000",
    # "A2000": "NVIDIA RTX A2000",
    # "RTX4090": "NVIDIA GeForce RTX 4090",
    # "A5000": "NVIDIA RTX A5000",
    "A40": "NVIDIA A40",
    "A4500": "NVIDIA RTX A4500",
    # "RTX3080": "NVIDIA GeForce RTX 3080",
    # "RTX3070": "NVIDIA GeForce RTX 3070",
    # "RTX3080Ti": "NVIDIA GeForce RTX 3080 Ti",
    # "L4": "NVIDIA L4",
}

# Approximate RunPod on-demand cost per GPU-hour (USD).
# Used to sort candidate GPUs cheapest-first within the same VRAM tier.
# GPUs not listed here get a high default cost so they sort last.
GPU_COST_PER_HOUR: Dict[str, float] = {
    "L40": 0.99,
    "A100": 1.39,
    "A100S": 1.49,
    "H100S": 2.69,
    "H100N": 3.07,
    "H200": 3.59,
    "B200": 4.99,
}
_DEFAULT_GPU_COST = 999.0  # fallback for unlisted GPUs
GPU_COUNT = 1
allowed_cuda_versions = ["12.8"]
HARDWARE_REFRESH_INTERVAL_SECONDS = int(
    os.getenv("OW_RUNPOD_HARDWARE_REFRESH_INTERVAL_SECONDS", "900")
)
HARDWARE_FAILURE_THRESHOLD = int(os.getenv("OW_RUNPOD_HARDWARE_FAILURE_THRESHOLD", "3"))
# Escalating provisioning-failure cooldowns (seconds). After each cooldown triggers,
# the next uses the following rung; the final rung repeats for all later cooldowns.
DEFAULT_HARDWARE_COOLDOWN_LADDER_SECONDS: tuple[int, ...] = (
    1 * 60,
    2 * 60,
    5 * 60,
    10 * 60,
    20 * 60,
    40 * 60,
    80 * 60,
    160 * 60,
    360 * 60,
)


def _env_cooldown_ladder_seconds() -> tuple[int, ...]:
    """Parse ``OW_RUNPOD_HARDWARE_COOLDOWN_LADDER_SECONDS`` (comma-separated seconds)."""
    raw = os.getenv("OW_RUNPOD_HARDWARE_COOLDOWN_LADDER_SECONDS")
    if raw is None or not raw.strip():
        return DEFAULT_HARDWARE_COOLDOWN_LADDER_SECONDS
    parts = [int(p.strip()) for p in raw.split(",") if p.strip()]
    if not parts:
        return DEFAULT_HARDWARE_COOLDOWN_LADDER_SECONDS
    return tuple(parts)


HARDWARE_COOLDOWN_LADDER_SECONDS = _env_cooldown_ladder_seconds()
# How long to pause all provisioning after a spending-limit error
SPENDING_LIMIT_PAUSE_SECONDS = int(
    os.getenv("OW_RUNPOD_SPENDING_LIMIT_PAUSE_SECONDS", "300")  # 5 minutes
)
# Patterns in RunPod error messages that indicate a spending limit
SPENDING_LIMIT_ERROR_PATTERNS = [
    "spending limit",
    "spend limit",
    "budget limit",
    "exceeded your",
]
RUNPOD_CLOUD_TYPE = os.getenv("OW_RUNPOD_CLOUD_TYPE", "ALL").upper()
RUNPOD_SUPPORT_PUBLIC_IP = (
    os.getenv("OW_RUNPOD_SUPPORT_PUBLIC_IP", "true").lower() == "true"
)
RUNPOD_MIN_DOWNLOAD = os.getenv("OW_RUNPOD_MIN_DOWNLOAD")
RUNPOD_MIN_UPLOAD = os.getenv("OW_RUNPOD_MIN_UPLOAD")
RUNPOD_DATA_CENTER_ID = os.getenv("OW_RUNPOD_DATA_CENTER_ID")
RUNPOD_COUNTRY_CODE = os.getenv("OW_RUNPOD_COUNTRY_CODE")


# Check that GPU name mapping is unique in both directions
gpu_full = list(GPUs.values())
if not len(gpu_full) == len(set(gpu_full)):
    # print duplicates
    from collections import Counter

    counts = Counter(gpu_full)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    raise ValueError(f"Duplicate GPU full names found: {duplicates}")


# Build map of memory -> hardware configu
HARDWARE_CONFIG = {}


def parse_hardware_config(hardware_type: str) -> tuple[int, str]:
    count, gpu = hardware_type.split("x ", maxsplit=1)
    return int(count), gpu.strip()


def is_spending_limit_error(error: Exception | str) -> bool:
    """Check whether an error message indicates a RunPod spending limit."""
    msg = str(error).lower()
    return any(pattern in msg for pattern in SPENDING_LIMIT_ERROR_PATTERNS)


def _gpu_cost_sort_key(hardware_type: str) -> float:
    """Return the approximate $/hr for a hardware type (e.g. '1x L40' or '2x A100').

    Multi-GPU configs scale linearly.  GPUs missing from GPU_COST_PER_HOUR
    sort last so newly-added GPUs are never silently preferred over known-cheap ones.
    """
    count, gpu_name = parse_hardware_config(hardware_type)
    per_gpu = GPU_COST_PER_HOUR.get(gpu_name, _DEFAULT_GPU_COST)
    return count * per_gpu


@dataclass
class HardwareFailureState:
    consecutive_failures: int = 0
    cooldown_until: Optional[float] = None
    last_failure_at: Optional[float] = None
    last_success_at: Optional[float] = None
    last_error: Optional[str] = None
    #: Increments each time a cooldown is applied; drives ladder duration selection.
    cooldown_escalation_level: int = 0


class RunpodHardwareRegistry:
    """RunPod GPU inventory cache with escalating failure cooldowns per hardware string."""

    def __init__(
        self,
        refresh_interval_seconds: int = HARDWARE_REFRESH_INTERVAL_SECONDS,
        failure_threshold: int = HARDWARE_FAILURE_THRESHOLD,
        spending_limit_pause_seconds: int = SPENDING_LIMIT_PAUSE_SECONDS,
        cooldown_ladder_seconds: Optional[tuple[int, ...]] = None,
        now_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        """Build a registry.

        Args:
            refresh_interval_seconds: Minimum interval between RunPod inventory refreshes.
            failure_threshold: Consecutive provisioning failures before a cooldown applies.
            cooldown_ladder_seconds: Cooldown durations (seconds) per escalation level;
                defaults to env / built-in ladder. The last entry repeats for later levels.
            now_fn: Injectable clock for tests (returns unix time as float).
        """
        self.refresh_interval_seconds = refresh_interval_seconds
        self.failure_threshold = failure_threshold
        self.spending_limit_pause_seconds = spending_limit_pause_seconds
        resolved_ladder = (
            cooldown_ladder_seconds
            if cooldown_ladder_seconds is not None
            else HARDWARE_COOLDOWN_LADDER_SECONDS
        )
        if not resolved_ladder:
            raise ValueError("cooldown_ladder_seconds must be non-empty")
        self.cooldown_ladder_seconds: tuple[int, ...] = resolved_ladder
        self.now_fn = now_fn or time.time
        self._last_refresh_at = 0.0
        self._discovered_config: Dict[int, List[str]] = {}
        self._hardware_config: Dict[int, List[str]] = {}
        self._failure_state: Dict[str, HardwareFailureState] = {}
        self._lock = RLock()
        # Global pause: when a spending-limit error is detected, all provisioning
        # is paused until this timestamp.
        self._spending_limit_pause_until: float = 0.0

    def _now(self) -> float:
        return self.now_fn()

    def _is_on_cooldown(self, hardware_type: str, now: Optional[float] = None) -> bool:
        now = self._now() if now is None else now
        state = self._failure_state.get(hardware_type)
        if state is None or state.cooldown_until is None:
            return False
        if now >= state.cooldown_until:
            state.cooldown_until = None
            state.consecutive_failures = 0
            state.last_error = None
            return False
        return True

    def _rebuild_hardware_config(self, discovered_config: Dict[int, List[str]]) -> None:
        now = self._now()
        filtered: Dict[int, List[str]] = {}
        for memory_gb, hardware_types in discovered_config.items():
            available = [
                hardware_type
                for hardware_type in sorted(set(hardware_types), key=_gpu_cost_sort_key)
                if not self._is_on_cooldown(hardware_type, now=now)
            ]
            if available:
                filtered[memory_gb] = available
        self._hardware_config = filtered
        HARDWARE_CONFIG.clear()
        HARDWARE_CONFIG.update(filtered)

    def refresh(self, runpod_client, force: bool = False) -> Dict[int, List[str]]:
        with self._lock:
            now = self._now()
            if (
                force
                or not self._hardware_config
                or now - self._last_refresh_at >= self.refresh_interval_seconds
            ):
                runpod_gpus = runpod_client.get_gpus()
                discovered_config: Dict[int, List[str]] = {}
                for gpu_short, gpu_full in VERIFIED_GPUs.items():
                    for gpu in runpod_gpus:
                        if gpu["id"] != gpu_full:
                            continue
                        memory_gb = int(gpu["memoryInGb"]) - 5
                        discovered_config.setdefault(memory_gb, []).append(
                            f"1x {gpu_short}"
                        )
                self._last_refresh_at = now
                self._discovered_config = discovered_config
                self._rebuild_hardware_config(self._discovered_config)
            else:
                self._rebuild_hardware_config(self._discovered_config)
            return dict(self._hardware_config)

    def get_candidate_hardware(
        self,
        required_vram: int,
        allowed_hardware: Optional[List[str]] = None,
        runpod_client=None,
    ) -> List[str]:
        with self._lock:
            if runpod_client is not None:
                self.refresh(runpod_client)
            else:
                self._rebuild_hardware_config(self._discovered_config)

            now = self._now()
            if allowed_hardware:
                return [
                    hw
                    for hw in allowed_hardware
                    if not self._is_on_cooldown(hw, now=now)
                ]

            candidates: List[str] = []
            for vram in sorted(self._hardware_config.keys()):
                if required_vram <= vram:
                    candidates.extend(self._hardware_config[vram])
            return candidates

    def record_success(self, hardware_type: str) -> None:
        with self._lock:
            state = self._failure_state.setdefault(
                hardware_type, HardwareFailureState()
            )
            state.consecutive_failures = 0
            state.cooldown_until = None
            state.cooldown_escalation_level = 0
            state.last_success_at = self._now()
            state.last_error = None
            self._rebuild_hardware_config(self._discovered_config)

    def record_failure(self, hardware_type: str, error: Exception | str) -> bool:
        with self._lock:
            now = self._now()

            # If this is a spending-limit error, apply a global pause instead of
            # penalising the individual hardware type.
            if is_spending_limit_error(error):
                self._spending_limit_pause_until = (
                    now + self.spending_limit_pause_seconds
                )
                # Don't count spending-limit errors toward the per-hardware
                # failure threshold — they are account-wide and transient.
                return False

            state = self._failure_state.setdefault(
                hardware_type, HardwareFailureState()
            )
            state.consecutive_failures += 1
            state.last_failure_at = now
            state.last_error = str(error)
            cooldown_applied = False
            if state.consecutive_failures >= self.failure_threshold:
                ladder = self.cooldown_ladder_seconds
                idx = min(state.cooldown_escalation_level, len(ladder) - 1)
                duration_s = ladder[idx]
                until_ts = state.last_failure_at + duration_s
                state.cooldown_until = until_ts
                state.consecutive_failures = 0
                state.cooldown_escalation_level += 1
                cooldown_applied = True
                escalation_level = state.cooldown_escalation_level
                logger.warning(
                    "RunPod provisioning cooldown triggered: hardware=%s "
                    "escalation_level=%s ladder_rung=%s/%s duration_s=%s until_utc=%s",
                    hardware_type,
                    escalation_level,
                    idx + 1,
                    len(ladder),
                    duration_s,
                    datetime.fromtimestamp(until_ts, tz=timezone.utc).isoformat(),
                )
            self._rebuild_hardware_config(self._discovered_config)
            return cooldown_applied

    def get_cooldown_info(self, hardware_type: str) -> Optional[float]:
        state = self._failure_state.get(hardware_type)
        if state is None:
            return None
        if not self._is_on_cooldown(hardware_type):
            return None
        return state.cooldown_until

    def is_spending_limit_paused(self) -> bool:
        """Return True if provisioning is globally paused due to a spending-limit error."""
        return self._now() < self._spending_limit_pause_until

    def spending_limit_pause_until(self) -> float:
        """Return the timestamp when the spending-limit pause expires (0 if not paused)."""
        return self._spending_limit_pause_until

    def get_cooldown_escalation_level(self, hardware_type: str) -> int:
        """Return how many provisioning cooldown waves have completed for this hardware.

        Resets to ``0`` on :meth:`record_success`. Increments each time a cooldown
        is applied in :meth:`record_failure`.
        """
        with self._lock:
            state = self._failure_state.get(hardware_type)
            if state is None:
                return 0
            return state.cooldown_escalation_level

    def get_active_cooldown_end_times(
        self, hardware_types: List[str]
    ) -> Dict[str, float]:
        """Return ``cooldown_until`` epoch timestamps for types currently on cooldown.

        Only includes entries from ``hardware_types`` that are still cooling down
        after repeated provisioning failures.
        """
        with self._lock:
            now = self._now()
            out: Dict[str, float] = {}
            for hw in hardware_types:
                if not self._is_on_cooldown(hw, now=now):
                    continue
                state = self._failure_state.get(hw)
                if state is None or state.cooldown_until is None:
                    continue
                out[hw] = state.cooldown_until
            return out


HARDWARE_REGISTRY = RunpodHardwareRegistry()


def populate_hardware_config(runpod_client, force: bool = False):
    return HARDWARE_REGISTRY.refresh(runpod_client, force=force)


def wait_for_pod(pod, runpod_client):
    while pod.get("runtime") is None:
        time.sleep(1)
        pod = runpod_client.get_pod(pod["id"])
    return pod


@lru_cache
@backoff.on_exception(
    backoff.constant,
    TypeError,
    interval=1,
    max_time=600,
    max_tries=600,
    logger=None,
)
def get_ip_and_port(pod_id, runpod_client):
    pod = runpod_client.get_pod(pod_id)
    for ip_and_port in pod["runtime"]["ports"]:
        if ip_and_port["privatePort"] == 22:
            ip = ip_and_port["ip"]
            port = ip_and_port["publicPort"]
            return ip, port


def create_ssh_client(pod, runpod_client=None):
    key_file = os.path.expanduser("~/.ssh/id_ed25519")
    user = "root"
    ip, port = get_ip_and_port(pod["id"], runpod_client)
    print(f"Connecting to {ip}:{port}")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for _ in range(10):
        try:
            ssh.connect(ip, port=port, username=user, key_filename=key_file)
            return ssh
        except Exception as e:
            print(e)
            time.sleep(1)
            continue
    print("Failed to connect to pod. Shutting down pod")
    runpod_client.terminate_pod(pod["id"])


def copy_to_pod(pod, src, dst, runpod_client=None):
    if not os.path.exists(src):
        # Assume src is relative to __file__
        src = os.path.join(os.path.dirname(__file__), src)
        assert os.path.exists(src), f"File {src} does not exist"
    ssh = create_ssh_client(pod, runpod_client)
    with SCPClient(ssh.get_transport()) as scp:
        scp.put(src, dst)


def run_on_pod(pod, cmd, runpod_client=None):
    ssh = create_ssh_client(pod, runpod_client)
    stdin, stdout, stderr = ssh.exec_command(cmd)

    while True:
        line = stdout.readline()
        if not line:
            break
        print(line, end="")

    while True:
        error_line = stderr.readline()
        if not error_line:
            break
        print(error_line, end="")

    stdin.close()
    stdout.close()
    stderr.close()
    ssh.close()


def run_on_pod_interactive(pod, cmd, runpod_client=None):
    ssh = create_ssh_client(pod, runpod_client)
    channel = ssh.get_transport().open_session()
    channel.get_pty()
    channel.exec_command(cmd)
    output_buffer = b""
    logs = ""

    while True:
        if channel.recv_ready():
            output_buffer += channel.recv(1024)
            try:
                output = output_buffer.decode()
                print(output, end="")
                logs += output
                output_buffer = b""
                if (
                    "password" in output.lower()
                ):  # Check for password prompt or other interactive input requests
                    password = input("Enter the required input: ")
                    channel.send(password + "\n")
            except UnicodeDecodeError:
                pass  # Ignore decode errors and continue receiving data

        if channel.recv_stderr_ready():
            error = channel.recv_stderr(1024).decode(errors="ignore")
            print(error, end="")

        if channel.exit_status_ready():
            break

    channel.close()
    ssh.close()
    return logs


def check_correct_cuda(pod, allowed=allowed_cuda_versions, runpod_client=None):
    cmd = "nvidia-smi"
    logs = run_on_pod_interactive(pod, cmd, runpod_client)
    return any([f"CUDA Version: {i}" in logs for i in allowed])


@backoff.on_exception(backoff.expo, Exception, max_time=60, max_tries=5, logger=None)
def _start_worker(
    gpu,
    image,
    count=GPU_COUNT,
    name=None,
    container_disk_in_gb=500,
    volume_in_gb=500,
    worker_id=None,
    dev_mode=False,
    ttl_hours=24,
    pending_workers=None,
    env=None,
    runpod_client=None,
):
    client = runpod_client or runpod
    gpu = GPUs[gpu]
    # default name: <username>-worker-<timestamp>
    name = name or f"{os.environ['USER']}-worker-{int(time.time())}"
    image = IMAGES.get(image, image)

    if pending_workers is None:
        pending_workers = []

    env = env or {}
    env.update(
        {
            "WORKER_ID": worker_id,
            "DOCKER_IMAGE": image,
            "OW_DEV": "true" if dev_mode else "false",
            "TTL_HOURS": str(ttl_hours),
            "RUNPOD_API_KEY": os.getenv("RUNPOD_API_KEY"),
        }
    )
    if worker_id is None:
        worker_id = uuid.uuid4().hex[:8]
    pod = client.create_pod(
        name,
        image,
        gpu,
        cloud_type=RUNPOD_CLOUD_TYPE,
        support_public_ip=RUNPOD_SUPPORT_PUBLIC_IP,
        container_disk_in_gb=container_disk_in_gb,
        volume_in_gb=volume_in_gb,
        volume_mount_path="/workspace",
        gpu_count=count,
        allowed_cuda_versions=allowed_cuda_versions,
        data_center_id=RUNPOD_DATA_CENTER_ID,
        country_code=RUNPOD_COUNTRY_CODE,
        min_download=int(RUNPOD_MIN_DOWNLOAD) if RUNPOD_MIN_DOWNLOAD else None,
        min_upload=int(RUNPOD_MIN_UPLOAD) if RUNPOD_MIN_UPLOAD else None,
        ports="8000/http,10101/http,22/tcp",
        start_ssh=True,
        env=env,
    )
    pending_workers.append(pod["id"])

    if dev_mode:
        ip, port = get_ip_and_port(pod["id"], client)
        print(f"ssh root@{ip} -p {port} -i ~/.ssh/id_ed25519")
    pending_workers.remove(pod["id"])
    return pod


def start_worker(
    gpu,
    image,
    count=GPU_COUNT,
    name=None,
    container_disk_in_gb=500,
    volume_in_gb=500,
    worker_id=None,
    dev_mode=False,
    ttl_hours=24,
    env=None,
    runpod_client=None,
):
    pending_workers = []
    if dev_mode:
        env = {
            var: os.environ.get(var)
            for var in [
                "OPENWEIGHTS_API_KEY",
                "RUNPOD_API_KEY",
                "HF_TOKEN",
                "HF_USER",
                "HF_ORG",
            ]
        }
    if runpod_client is None:
        runpod.api_key = os.getenv("RUNPOD_API_KEY")
        runpod_client = runpod
    try:
        pod = _start_worker(
            gpu,
            image,
            count,
            name,
            container_disk_in_gb,
            volume_in_gb,
            worker_id,
            dev_mode,
            ttl_hours,
            pending_workers,
            env,
            runpod_client,
        )
        if pod is None:
            raise RuntimeError("RunPod create_pod returned no pod")
        return pod
    except Exception as e:
        raise RuntimeError(f"Failed to start RunPod worker: {e}") from e
    finally:
        print("Pending workers: ", pending_workers)
        for pod_id in pending_workers:
            print(f"Shutting down pod {pod_id}")
            runpod_client.terminate_pod(pod_id)


if __name__ == "__main__":
    fire.Fire(start_worker)
