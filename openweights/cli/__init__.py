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

    # ls command
    from openweights.cli.ls import add_ls_parser, handle_ls

    ls_parser = sub.add_parser("ls", help="List job IDs.")
    add_ls_parser(ls_parser)

    # cancel command
    from openweights.cli.cancel import add_cancel_parser, handle_cancel

    cancel_parser = sub.add_parser("cancel", help="Cancel jobs by ID.")
    add_cancel_parser(cancel_parser)

    # logs command
    from openweights.cli.logs import add_logs_parser, handle_logs

    logs_parser = sub.add_parser("logs", help="Display logs for a job.")
    add_logs_parser(logs_parser)

    # fetch command
    from openweights.cli.fetch import add_fetch_parser, handle_fetch

    fetch_parser = sub.add_parser("fetch", help="Fetch file content by ID.")
    add_fetch_parser(fetch_parser)

    # serve command
    from openweights.cli.serve import add_serve_parser, handle_serve

    serve_parser = sub.add_parser("serve", help="Start the dashboard backend server.")
    add_serve_parser(serve_parser)

    # deploy command
    from openweights.cli.deploy import add_deploy_parser, handle_deploy

    deploy_parser = sub.add_parser(
        "deploy", help="Deploy a cluster instance on RunPod."
    )
    add_deploy_parser(deploy_parser)

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
    elif args.cmd == "ls":
        sys.exit(handle_ls(args))
    elif args.cmd == "cancel":
        sys.exit(handle_cancel(args))
    elif args.cmd == "logs":
        sys.exit(handle_logs(args))
    elif args.cmd == "fetch":
        sys.exit(handle_fetch(args))
    elif args.cmd == "serve":
        sys.exit(handle_serve(args))
    elif args.cmd == "deploy":
        sys.exit(handle_deploy(args))
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
