"""SSH command implementation."""

import os
import shlex
import sys
from typing import List, Optional, Tuple

from openweights.cli.common import (
    RunpodProvider,
    SSHSpec,
    UnisonSyncer,
    bootstrap_remote,
    load_env_file,
    open_interactive_shell,
    ssh_exec,
    wait_for_ssh,
)


def parse_existing_ssh(existing: str, key_path: str) -> SSHSpec:
    """Parse existing SSH connection string.

    Format: user@host:port
    Example: root@206.41.93.58:52206
    """
    if "@" not in existing:
        raise SystemExit(
            f"--existing must be in format user@host:port (got: {existing})"
        )

    user_host, port_str = existing.rsplit(":", 1)
    user, host = user_host.split("@", 1)

    try:
        port = int(port_str)
    except ValueError:
        raise SystemExit(f"Invalid port in --existing: {port_str}")

    return SSHSpec(
        host=host, port=port, user=user, key_path=os.path.expanduser(key_path)
    )


def parse_mounts(
    mounts: List[str], cwd_remote: Optional[str]
) -> List[Tuple[str, str, str]]:
    """Parse mount specifications."""
    parsed = []
    if not mounts:
        local = os.getcwd()
        # Default: mount CWD to the same absolute path on remote
        remote = local
        parsed.append((local, remote, "cwd"))
        return parsed
    for i, m in enumerate(mounts):
        if ":" in m:
            # User provided LOCAL:REMOTE format
            local, remote = m.split(":", 1)
            parsed.append((os.path.abspath(local), remote, f"mount{i+1}"))
        else:
            # No colon: mount to identical path as local
            local = os.path.abspath(m)
            parsed.append((local, local, f"mount{i+1}"))
    return parsed


def add_ssh_parser(parser):
    """Add arguments for the ssh command."""
    parser.add_argument(
        "command",
        nargs="*",
        help="Command to execute on the remote machine (non-interactive mode)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Start interactive session with live file sync",
    )
    parser.add_argument(
        "--mount",
        action="append",
        default=[],
        help="LOCAL:REMOTE or just LOCAL (repeatable). If no colon, mounts to identical path as local. Defaults to CWD mounted at same path on remote",
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
        "--remote-cwd",
        default="/workspace",
        help="Remote working directory for the main mount.",
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
        "--no-terminate-prompt",
        action="store_true",
        help="Don't ask to terminate the machine on exit.",
    )
    parser.add_argument(
        "--existing",
        help="Use existing SSH connection (format: user@host:port). Skips machine provisioning.",
    )


def handle_ssh(args) -> int:
    """Handle the ssh command."""
    env_from_file = load_env_file(args.env_file)
    provider_env = dict(env_from_file)  # only pass what's in --env-file

    # Handle existing connection
    if args.existing:
        ssh = parse_existing_ssh(args.existing, args.key_path)
        print(f"[ow] Using existing connection: {ssh.user}@{ssh.host}:{ssh.port}")

        # Create a dummy terminate function for consistency
        def noop_terminate():
            pass

        class DummyStartResult:
            def __init__(self, ssh_spec):
                self.ssh = ssh_spec
                self.terminate = noop_terminate

        start_res = DummyStartResult(ssh)
    else:
        # Normal provisioning flow
        if args.provider == "runpod":
            provider = RunpodProvider(key_path=args.key_path)
        else:
            raise SystemExit(f"Unknown provider: {args.provider}")

        print("[ow] Starting/allocating machine...")
        start_res = provider.start(
            image=args.image, gpu=args.gpu, count=args.count, env=provider_env
        )
        ssh = start_res.ssh

        print("[ow] Waiting for sshd to become ready...")
        wait_for_ssh(ssh)
        print(f"[ow] SSH ready: {ssh.user}@{ssh.host}:{ssh.port}")

    # Determine mode: command execution, connection string only, or interactive with sync
    has_command = bool(args.command)
    sync_mode = args.sync

    # Mode 1: Non-interactive command execution (ow ssh cmd)
    if has_command and not sync_mode:
        print(f"[ow] Executing command: {' '.join(args.command)}")
        mounts = parse_mounts(args.mount, args.remote_cwd)
        do_editable = not args.no_editable_install
        # Collect all remote directories that need to be created
        remote_dirs = [remote for _, remote, _ in mounts]
        bootstrap_remote(
            ssh,
            remote_cwd=mounts[0][1],
            do_editable_install=do_editable,
            additional_dirs=remote_dirs,
        )

        # Execute the command in the remote working directory
        cmd_str = " ".join(args.command)
        remote_cmd = f"bash -lc 'cd {shlex.quote(mounts[0][1])} && {cmd_str}'"
        exit_code = ssh_exec(ssh, remote_cmd)

        if not args.no_terminate_prompt:
            ans = input("Terminate the machine? [y/N] ").strip().lower()
            if ans in ("y", "yes"):
                print("[ow] Terminating machine...")
                start_res.terminate()
            else:
                print("[ow] Leaving machine running.")

        return exit_code

    # Mode 2: Just print connection string (ow ssh)
    if not has_command and not sync_mode:
        print(f"ssh -p {ssh.port} -i {ssh.key_path} {ssh.user}@{ssh.host}")
        return 0

    # Mode 3: Interactive with sync (ow ssh --sync or default behavior when command given with --sync)
    mounts = parse_mounts(args.mount, args.remote_cwd)
    do_editable = not args.no_editable_install
    # Collect all remote directories that need to be created
    remote_dirs = [remote for _, remote, _ in mounts]
    bootstrap_remote(
        ssh,
        remote_cwd=mounts[0][1],
        do_editable_install=do_editable,
        additional_dirs=remote_dirs,
    )

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

    return exit_code
