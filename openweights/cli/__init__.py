#!/usr/bin/env python3
"""OpenWeights CLI entry point."""
import argparse
import signal
import sys


def main():
    """Main entry point for the ow CLI."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    ap = argparse.ArgumentParser(
        prog="ow", description="OpenWeights CLI for remote GPU operations"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ssh command
    from openweights.cli.ssh import add_ssh_parser, handle_ssh

    ssh_parser = sub.add_parser(
        "ssh", help="Start or attach to a remote shell with live file sync."
    )
    add_ssh_parser(ssh_parser)

    # exec command
    from openweights.cli.exec import add_exec_parser, handle_exec

    exec_parser = sub.add_parser(
        "exec", help="Execute a command on a remote GPU with file sync."
    )
    add_exec_parser(exec_parser)

    # signup command
    from openweights.cli.signup import add_signup_parser, handle_signup

    signup_parser = sub.add_parser(
        "signup", help="Create a new user, organization, and API key."
    )
    add_signup_parser(signup_parser)

    # cluster command
    from openweights.cli.cluster import add_cluster_parser, handle_cluster

    cluster_parser = sub.add_parser(
        "cluster", help="Run the cluster manager locally with your own infrastructure."
    )
    add_cluster_parser(cluster_parser)

    # worker command
    from openweights.cli.worker import add_worker_parser, handle_worker

    worker_parser = sub.add_parser(
        "worker", help="Run a worker to execute jobs from the queue."
    )
    add_worker_parser(worker_parser)

    # token command
    from openweights.cli.token import add_token_parser, handle_token

    token_parser = sub.add_parser("token", help="Manage API tokens for organizations.")
    add_token_parser(token_parser)

    args = ap.parse_args()

    if args.cmd == "ssh":
        sys.exit(handle_ssh(args))
    elif args.cmd == "exec":
        sys.exit(handle_exec(args))
    elif args.cmd == "signup":
        sys.exit(handle_signup(args))
    elif args.cmd == "cluster":
        sys.exit(handle_cluster(args))
    elif args.cmd == "worker":
        sys.exit(handle_worker(args))
    elif args.cmd == "token":
        sys.exit(handle_token(args))
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
