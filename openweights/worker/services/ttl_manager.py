#!/usr/bin/env python3
"""
TTL management utility for extending or checking pod TTL
"""
import argparse
import datetime
import os


def get_shutdown_time():
    """Get the current shutdown time"""
    shutdown_file = os.path.expanduser("~/shutdown.txt")
    if not os.path.exists(shutdown_file):
        return None

    with open(shutdown_file, "r") as f:
        shutdown_time_str = f.read().strip()

    try:
        return datetime.datetime.fromisoformat(shutdown_time_str)
    except ValueError:
        return None


def set_shutdown_time(shutdown_time):
    """Set a new shutdown time"""
    shutdown_file = os.path.expanduser("~/shutdown.txt")
    with open(shutdown_file, "w") as f:
        f.write(shutdown_time.isoformat())


def extend_ttl(hours):
    """Extend TTL by specified hours from now"""
    new_shutdown_time = datetime.datetime.now() + datetime.timedelta(hours=hours)
    set_shutdown_time(new_shutdown_time)
    return new_shutdown_time


def set_ttl(hours):
    """Set TTL to specified hours from now"""
    return extend_ttl(hours)


def check_ttl():
    """Check current TTL status"""
    shutdown_time = get_shutdown_time()
    if not shutdown_time:
        return "No TTL set"

    current_time = datetime.datetime.now()
    time_left = shutdown_time - current_time

    if time_left.total_seconds() <= 0:
        return f"TTL expired at {shutdown_time}"
    else:
        hours_left = time_left.total_seconds() / 3600
        return f"TTL expires at {shutdown_time} ({hours_left:.1f} hours remaining)"


def main():
    parser = argparse.ArgumentParser(description="Manage pod TTL (Time To Live)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Check current TTL status")
    group.add_argument(
        "--extend", type=float, help="Extend TTL by specified hours from now"
    )
    group.add_argument("--set", type=float, help="Set TTL to specified hours from now")

    args = parser.parse_args()

    if args.check:
        print(check_ttl())
    elif args.extend:
        new_time = extend_ttl(args.extend)
        print(f"TTL extended by {args.extend} hours. New shutdown time: {new_time}")
    elif args.set:
        new_time = set_ttl(args.set)
        print(f"TTL set to {args.set} hours from now. Shutdown time: {new_time}")


if __name__ == "__main__":
    main()
