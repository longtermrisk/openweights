"""SSH command implementation."""

import os
import sys
from typing import List, Optional, Tuple

from openweights.cli.common import (
    RunpodProvider,
    UnisonSyncer,
    bootstrap_remote,
    load_env_file,
    open_interactive_shell,
    wait_for_ssh,
)


def parse_mounts(
    mounts: List[str], cwd_remote: Optional[str]
) -> List[Tuple[str, str, str]]:
    """Parse mount specifications."""
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


def add_ssh_parser(parser):
    """Add arguments for the ssh command."""
    parser.add_argument(
        "--mount",
        action="append",
        default=[],
        help="LOCAL:REMOTE (repeatable). Defaults to CWD:~/workspace",
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


def handle_ssh(args) -> int:
    """Handle the ssh command."""
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
    print(f"[ow] SSH ready: {ssh.user}@{ssh.host}:{ssh.port}")

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

    return exit_code
