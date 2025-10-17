"""Exec command implementation - execute commands on remote GPU."""

import os
import shlex
import sys
from typing import List

from openweights.cli.common import (
    RunpodProvider,
    UnisonSyncer,
    bootstrap_remote,
    load_env_file,
    ssh_exec,
    wait_for_ssh,
)


def add_exec_parser(parser):
    """Add arguments for the exec command."""
    parser.add_argument(
        "command", nargs="+", help="Command to execute on the remote machine"
    )
    parser.add_argument(
        "--mount",
        action="append",
        default=[],
        help="LOCAL:REMOTE (repeatable). Defaults to CWD:/workspace",
    )
    parser.add_argument(
        "--env-file", default=None, help="Path to .env to export and pass to provider."
    )
    parser.add_argument(
        "--image", default="nielsrolf/ow-default:v0.7", help="Provider image name."
    )
    parser.add_argument("--gpu", default="L40", help="GPU type for provider.")
    parser.add_argument("--count", type=int, default=1, help="GPU count.")
    parser.add_argument(
        "--remote-cwd", default="/workspace", help="Remote working directory."
    )
    parser.add_argument(
        "--provider", default="runpod", choices=["runpod"], help="Machine provider."
    )
    parser.add_argument(
        "--key-path", default="~/.ssh/id_ed25519", help="SSH private key path."
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[".git", "__pycache__", ".mypy_cache", ".venv", ".env"],
        help="Ignore patterns (Unison Name filters, repeatable).",
    )
    parser.add_argument(
        "--no-editable-install",
        action="store_true",
        help="Skip `pip install -e .` if pyproject.toml exists.",
    )
    parser.add_argument(
        "--no-terminate",
        action="store_true",
        help="Don't terminate the machine after execution.",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Don't sync files before execution (faster but files won't be up to date).",
    )


def handle_exec(args) -> int:
    """Handle the exec command."""
    env_from_file = load_env_file(args.env_file)
    provider_env = dict(env_from_file)

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
    print(f"[ow] SSH ready: {ssh.user}@{ssh.host}:{ssh.port}")

    # Parse mounts
    mounts = []
    if not args.mount:
        local = os.getcwd()
        remote = args.remote_cwd
        mounts.append((local, remote, "cwd"))
    else:
        for i, m in enumerate(args.mount):
            if ":" not in m:
                raise SystemExit(f"--mount must be LOCAL:REMOTE (got: {m})")
            local, remote = m.split(":", 1)
            mounts.append((os.path.abspath(local), remote, f"mount{i+1}"))

    do_editable = not args.no_editable_install
    bootstrap_remote(ssh, remote_cwd=mounts[0][1], do_editable_install=do_editable)

    # Sync files if not disabled
    syncers: List[UnisonSyncer] = []
    if not args.no_sync:
        print("[ow] Syncing files...")
        for local, remote, label in mounts:
            s = UnisonSyncer(
                local_dir=local,
                remote_dir=remote,
                ssh=ssh,
                ignore=args.exclude,
                label=label,
            )
            # Do initial sync only (no watch mode)
            s._initial_sync()
        print("[ow] File sync complete.")

    try:
        # Build the command to execute
        cmd_str = " ".join(shlex.quote(arg) for arg in args.command)

        # Build environment exports
        env_exports = ""
        if env_from_file:
            env_exports = " ".join(
                [f"export {k}={shlex.quote(v)}" for k, v in env_from_file.items()]
            )
            env_exports += " && "

        remote_cmd = (
            f"bash -lc '{env_exports}cd {shlex.quote(mounts[0][1])} && {cmd_str}'"
        )

        print(f"[ow] Executing: {cmd_str}")
        exit_code = ssh_exec(ssh, remote_cmd)

    finally:
        # Stop any running syncers (shouldn't be any in exec mode)
        for s in syncers:
            s.stop()

        # Terminate machine unless --no-terminate is set
        if not args.no_terminate:
            print("[ow] Terminating machine...")
            start_res.terminate()
        else:
            print("[ow] Leaving machine running.")

    return exit_code
