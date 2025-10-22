"""Common utilities for CLI commands."""

import base64
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import runpod
from dotenv import dotenv_values

from openweights.cluster import start_runpod

# -------- Provider Abstraction ------------------------------------------------


@dataclass
class SSHSpec:
    host: str
    port: int
    user: str
    key_path: str


@dataclass
class StartResult:
    ssh: SSHSpec
    terminate: Callable[[], None]  # call to terminate the machine
    provider_meta: Dict


class Provider:
    def start(
        self, image: str, gpu: str, count: int, env: Dict[str, str]
    ) -> StartResult:
        raise NotImplementedError


class RunpodProvider(Provider):
    """
    Thin wrapper around your start_runpod.py module.
    - Expects RUNPOD_API_KEY in env (or via the shell environment).
    - Uses key at key_path.
    """

    def __init__(self, key_path: str):
        self.key_path = os.path.expanduser(key_path)

    def start(
        self, image: str, gpu: str, count: int, env: Dict[str, str]
    ) -> StartResult:
        if "RUNPOD_API_KEY" in env:
            os.environ["RUNPOD_API_KEY"] = env["RUNPOD_API_KEY"]
        runpod.api_key = os.getenv("RUNPOD_API_KEY")

        pod = start_runpod.start_worker(
            gpu=gpu,
            image=image,
            count=count,
            ttl_hours=int(env.get("TTL_HOURS", "24")),
            env=env,
            runpod_client=runpod,
            dev_mode=True,  # keep your current choice
        )
        assert pod is not None, "Runpod start_worker returned None"

        ip, port = start_runpod.get_ip_and_port(pod["id"], runpod)
        ssh = SSHSpec(host=ip, port=int(port), user="root", key_path=self.key_path)

        def _terminate():
            runpod.terminate_pod(pod["id"])

        return StartResult(
            ssh=ssh, terminate=_terminate, provider_meta={"pod_id": pod["id"]}
        )


# -------- Bidirectional Sync (Unison) ----------------------------------------


class UnisonSyncer:
    """
    Bidirectional sync using Unison in watch mode.
    - Quiet (minimal logs).
    - Initial one-shot sync uses a sentinel so the first prompt runs on up-to-date files.
    Requirements: `unison` available locally and on the remote image.
    """

    def __init__(
        self,
        local_dir: str,
        remote_dir: str,
        ssh: SSHSpec,
        ignore: List[str],
        label: str,
    ):
        self.local_dir = os.path.abspath(local_dir)
        self.remote_dir = remote_dir.rstrip("/")
        self.ssh = ssh
        self.ignore = ignore
        self.label = label
        self._proc: Optional[subprocess.Popen] = None

    def _sshargs(self) -> str:
        return (
            f"-p {self.ssh.port} "
            f"-i {shlex.quote(self.ssh.key_path)} "
            f"-o StrictHostKeyChecking=no "
            f"-o UserKnownHostsFile=/dev/null"
        )

    def _unison_base(self) -> List[str]:
        remote_root = (
            f"ssh://{self.ssh.user}@{self.ssh.host}//{self.remote_dir.lstrip('/')}"
        )
        cmd = [
            "unison",
            self.local_dir,
            remote_root,
            "-auto",
            "-batch",
            "-ui",
            "text",
            "-prefer",
            "newer",  # last-writer-wins
            "-copyonconflict",  # keep both if conflict
            "-sshargs",
            self._sshargs(),
            "-confirmbigdel=false",
        ]
        for ex in self.ignore:
            cmd += ["-ignore", f"Name {ex}"]
        # Always ignore our local sentinel content if it exists on either side
        cmd += ["-ignore", "Name .ow_sync"]
        return cmd

    def _initial_sync(self):
        # create busy sentinel locally so remote prompt blocks until first sync completes
        ow_sync = os.path.join(self.local_dir, ".ow_sync")
        os.makedirs(ow_sync, exist_ok=True)
        busy = os.path.join(ow_sync, "busy")
        with open(busy, "w") as f:
            f.write("1")

        try:
            # one-shot reconciliation
            subprocess.run(
                self._unison_base(),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        finally:
            if os.path.exists(busy):
                os.remove(busy)
            # mirror sentinel removal promptly
            subprocess.run(
                self._unison_base(),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def start(self):
        print(
            f"[ow] Initial sync (bidirectional via Unison): {self.label} <-> {self.remote_dir}"
        )
        self._initial_sync()
        # Use polling interval instead of watch mode for more reliable syncing
        # -repeat 2 means check every 2 seconds
        watch_cmd = self._unison_base() + ["-repeat", "2"]

        # Create a log file for debugging
        log_file = os.path.join(self.local_dir, ".ow_sync", f"unison_{self.label}.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Keep the log file open for the lifetime of the process
        self._log_file = open(log_file, "w")
        self._proc = subprocess.Popen(
            watch_cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )
        print(f"[ow] Watching (bi-dir): {self.local_dir} (label: {self.label})")
        print(f"[ow] Unison logs: {log_file}")

    def stop(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        # Close the log file if it exists
        if hasattr(self, "_log_file") and self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass


# -------- Remote bootstrap & shell glue --------------------------------------

REMOTE_INIT = r"""
set -euo pipefail

mkdir -p "$HOME/.ow_sync"

# require unison and rsync present (rsync not used now, but nice to have)
need_missing=0
for bin in unison; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "[ow] $bin not found on remote. Please install it in your image."
    need_missing=1
  fi
done
if [ "$need_missing" -ne 0 ]; then
  exit 1
fi

OW_RC="$HOME/.ow_sync/ow_prompt.sh"
cat > "$OW_RC" <<'EOF'
ow_sync_wait() {
  # stay quiet; block only if initial sentinel exists
  if [ -f "$HOME/.ow_sync/busy" ]; then
    while [ -f "$HOME/.ow_sync/busy" ]; do sleep 0.1; done
  fi
}
if [ -n "${PROMPT_COMMAND-}" ]; then
  PROMPT_COMMAND="ow_sync_wait;$PROMPT_COMMAND"
else
  PROMPT_COMMAND="ow_sync_wait"
fi
export PROMPT_COMMAND
EOF

BASH_RC="$HOME/.bashrc"
if [ -f "$BASH_RC" ]; then
  if ! grep -q ".ow_sync/ow_prompt.sh" "$BASH_RC"; then
    echo ". \"$OW_RC\"" >> "$BASH_RC"
  fi
else
  echo ". \"$OW_RC\"" > "$BASH_RC"
fi
"""


def ssh_exec(ssh: SSHSpec, remote_cmd: str) -> int:
    """Execute a command via SSH."""
    cmd = [
        "ssh",
        "-tt",
        "-p",
        str(ssh.port),
        "-i",
        ssh.key_path,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        f"{ssh.user}@{ssh.host}",
        remote_cmd,
    ]
    return subprocess.call(cmd)


def ssh_exec_quiet(ssh: SSHSpec, remote_cmd: str) -> int:
    """Execute a command via SSH without TTY."""
    cmd = [
        "ssh",
        "-p",
        str(ssh.port),
        "-i",
        ssh.key_path,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        f"{ssh.user}@{ssh.host}",
        remote_cmd,
    ]
    return subprocess.call(cmd)


def scp_text(ssh: SSHSpec, text: str, remote_path: str):
    """Copy arbitrary text to a remote file via SSH safely."""
    remote = f"{ssh.user}@{ssh.host}"
    remote_dir = os.path.dirname(remote_path)
    encoded = base64.b64encode(text.encode()).decode()
    cmd = (
        f"bash -lc 'mkdir -p {shlex.quote(remote_dir)} && "
        f"echo {shlex.quote(encoded)} | base64 -d > {shlex.quote(remote_path)}'"
    )
    subprocess.check_call(
        [
            "ssh",
            "-p",
            str(ssh.port),
            "-i",
            ssh.key_path,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            remote,
            cmd,
        ]
    )


def wait_for_ssh(ssh: SSHSpec, deadline_s: int = 180):
    """Poll until sshd accepts a connection."""
    start = time.time()
    cmd = [
        "ssh",
        "-p",
        str(ssh.port),
        "-i",
        ssh.key_path,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=2",
        f"{ssh.user}@{ssh.host}",
        "true",
    ]
    while True:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if proc.returncode == 0:
            return
        if time.time() - start > deadline_s:
            raise RuntimeError(
                f"SSH not reachable at {ssh.host}:{ssh.port} within {deadline_s}s"
            )
        time.sleep(2)


def bootstrap_remote(
    ssh: SSHSpec,
    remote_cwd: str,
    do_editable_install: bool,
    additional_dirs: List[str] = None,
):
    """Bootstrap the remote machine with necessary setup."""
    scp_text(ssh, REMOTE_INIT, "/root/.ow_sync/remote_init.sh")
    rc = ssh_exec(ssh, "bash ~/.ow_sync/remote_init.sh")
    if rc != 0:
        sys.exit(rc)

    # Create remote_cwd and any additional mount directories
    dirs_to_create = [remote_cwd]
    if additional_dirs:
        dirs_to_create.extend(additional_dirs)

    for dir_path in dirs_to_create:
        rc = ssh_exec(ssh, f"mkdir -p {shlex.quote(dir_path)}")
        if rc != 0:
            sys.exit(rc)

    if do_editable_install:
        check_cmd = f"bash -lc 'cd {shlex.quote(remote_cwd)} && if [ -f pyproject.toml ]; then python3 -m pip install -e .; else echo \"[ow] no pyproject.toml\"; fi'"
        rc = ssh_exec(ssh, check_cmd)
        if rc != 0:
            sys.exit(rc)


def open_interactive_shell(
    ssh: SSHSpec, remote_cwd: str, env_pairs: Dict[str, str]
) -> int:
    """Open an interactive shell on the remote machine."""
    parts = []
    if env_pairs:
        exports = " ".join(
            [f"export {k}={shlex.quote(v)}" for k, v in env_pairs.items()]
        )
        parts.append(exports)
    parts.append(f"cd {shlex.quote(remote_cwd)}")
    parts.append("exec bash")
    remote_cmd = f"bash -lc {shlex.quote(' ; '.join(parts))}"
    cmd = [
        "ssh",
        "-tt",
        "-p",
        str(ssh.port),
        "-i",
        ssh.key_path,
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=120",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        f"{ssh.user}@{ssh.host}",
        remote_cmd,
    ]
    rc = subprocess.call(cmd)
    # ensure trailing newline to keep local tty pretty
    try:
        sys.stdout.write("\n")
        sys.stdout.flush()
    except Exception:
        pass
    return rc


def load_env_file(path: Optional[str]) -> Dict[str, str]:
    """Load environment variables from a file."""
    if not path:
        return {}
    p = os.path.abspath(path)
    if not os.path.exists(p):
        raise SystemExit(f"--env-file path not found: {p}")
    vals = dotenv_values(p) or {}
    return {k: (v if v is not None else "") for k, v in vals.items()}
