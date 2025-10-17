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

    args = ap.parse_args()

    if args.cmd == "ssh":
        sys.exit(handle_ssh(args))
    elif args.cmd == "exec":
        sys.exit(handle_exec(args))
    elif args.cmd == "signup":
        sys.exit(handle_signup(args))
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
