#!/usr/bin/env python3
import argparse
import base64
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import runpod
from dotenv import dotenv_values  # hard requirement

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
        watch_cmd = self._unison_base() + ["-repeat", "watch"]
        self._proc = subprocess.Popen(
            watch_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[ow] Watching (bi-dir): {self.local_dir} (label: {self.label})")

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


def _ssh_exec(ssh: SSHSpec, remote_cmd: str) -> int:
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


def _scp_text(ssh: SSHSpec, text: str, remote_path: str):
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


def wait_for_ssh(ssh, deadline_s: int = 180):
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


def bootstrap_remote(ssh: SSHSpec, remote_cwd: str, do_editable_install: bool):
    _scp_text(ssh, REMOTE_INIT, "/root/.ow_sync/remote_init.sh")
    rc = _ssh_exec(ssh, "bash ~/.ow_sync/remote_init.sh")
    if rc != 0:
        sys.exit(rc)

    rc = _ssh_exec(ssh, f"mkdir -p {shlex.quote(remote_cwd)}")
    if rc != 0:
        sys.exit(rc)

    if do_editable_install:
        check_cmd = f"bash -lc 'cd {shlex.quote(remote_cwd)} && if [ -f pyproject.toml ]; then python3 -m pip install -e .; else echo \"[ow] no pyproject.toml\"; fi'"
        rc = _ssh_exec(ssh, check_cmd)
        if rc != 0:
            sys.exit(rc)


def open_interactive_shell(ssh: SSHSpec, remote_cwd: str, env_pairs: Dict[str, str]):
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


# -------- CLI ----------------------------------------------------------------


def parse_mounts(
    mounts: List[str], cwd_remote: Optional[str]
) -> List[Tuple[str, str, str]]:
    parsed = []
    if not mounts:
        local = os.getcwd()
        remote = cwd_remote or "~/workspace"
        parsed.append((local, remote, "cwd"))
        return parsed
    for i, m in enumerate(mounts):
        if ":" not in m:
            raise SystemExit(f"--mount must be LOCAL:REMOTE (got: {m})")
        local, remote = m.split(":", 1)
        parsed.append((os.path.abspath(local), remote, f"mount{i+1}"))
    return parsed


def load_env_file(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    p = os.path.abspath(path)
    if not os.path.exists(p):
        raise SystemExit(f"--env-file path not found: {p}")
    vals = dotenv_values(p) or {}
    return {k: (v if v is not None else "") for k, v in vals.items()}


def main():
    ap = argparse.ArgumentParser(
        prog="ow",
        description="Remote GPU shell with live, bidirectional sync (fail-fast).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sshp = sub.add_parser(
        "ssh", help="Start or attach to a remote shell with live file sync."
    )
    sshp.add_argument(
        "--mount",
        action="append",
        default=[],
        help="LOCAL:REMOTE (repeatable). Defaults to CWD:~/workspace",
    )
    sshp.add_argument(
        "--env-file", default=None, help="Path to .env to export and pass to provider."
    )
    sshp.add_argument(
        "--image", default="nielsrolf/ow-default:v0.7", help="Provider image name."
    )
    sshp.add_argument("--gpu", default="L40", help="GPU type for provider.")
    sshp.add_argument("--count", type=int, default=1, help="GPU count.")
    sshp.add_argument(
        "--remote-cwd",
        default="/workspace",
        help="Remote working directory for the main mount.",
    )
    sshp.add_argument(
        "--provider", default="runpod", choices=["runpod"], help="Machine provider."
    )
    sshp.add_argument(
        "--key-path", default="~/.ssh/id_ed25519", help="SSH private key path."
    )
    sshp.add_argument(
        "--exclude",
        action="append",
        default=[".git", "__pycache__", ".mypy_cache", ".venv", ".env"],
        help="Ignore patterns (Unison Name filters, repeatable).",
    )
    sshp.add_argument(
        "--no-editable-install",
        action="store_true",
        help="Skip `pip install -e .` if pyproject.toml exists.",
    )
    sshp.add_argument(
        "--no-terminate-prompt",
        action="store_true",
        help="Don’t ask to terminate the machine on exit.",
    )
    args = ap.parse_args()

    if args.cmd == "ssh":
        env_from_file = load_env_file(args.env_file)
        provider_env = dict(env_from_file)  # only pass what's in --env-file

        if args.provider == "runpod":
            provider = RunpodProvider(key_path=args.key_path)
        else:
            raise SystemExit(f"Unknown provider: {args.provider}")

        print("[ow] Starting/allocating machine...")
        start_res = provider.start(
            image=args.image, gpu=args.gpu, count=args.count, env=provider_env
        )
        ssh = start_res.ssh
        print(f"[ow] SSH: {ssh.user}@{ssh.host}:{ssh.port} using key {ssh.key_path}")

        print("[ow] Waiting for sshd to become ready...")
        wait_for_ssh(ssh)
        print(f"[ow] SSH: {ssh.user}@{ssh.host}:{ssh.port} using key {ssh.key_path}")

        mounts = parse_mounts(args.mount, args.remote_cwd)

        do_editable = not args.no_editable_install
        bootstrap_remote(ssh, remote_cwd=mounts[0][1], do_editable_install=do_editable)

        # Start bidirectional syncers
        syncers: List[UnisonSyncer] = []
        for local, remote, label in mounts:
            s = UnisonSyncer(
                local_dir=local,
                remote_dir=remote,
                ssh=ssh,
                ignore=args.exclude,
                label=label,
            )
            s.start()
            syncers.append(s)

        try:
            print("[ow] Opening interactive shell. Type `exit` or Ctrl-D to leave.")
            exit_code = open_interactive_shell(
                ssh, remote_cwd=mounts[0][1], env_pairs=env_from_file
            )
        finally:
            for s in syncers:
                s.stop()

        if not args.no_terminate_prompt:
            ans = input("Terminate the machine? [y/N] ").strip().lower()
            if ans in ("y", "yes"):
                print("[ow] Terminating machine...")
                start_res.terminate()
            else:
                print("[ow] Leaving machine running.")

        sys.exit(exit_code)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()
