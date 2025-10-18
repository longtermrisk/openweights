"""List jobs CLI command."""

import os
import sys

from openweights import OpenWeights


def add_ls_parser(parser):
    """Add arguments for the ls command."""
    parser.add_argument(
        "--status",
        type=str,
        choices=["pending", "in_progress", "completed", "failed", "canceled"],
        help="Filter jobs by status",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of job IDs to return (default: 100)",
    )


def get_openweights_client():
    """Get OpenWeights client with authentication."""
    auth_token = os.getenv("OPENWEIGHTS_API_KEY")

    if not auth_token:
        print(
            "Error: OPENWEIGHTS_API_KEY environment variable not set", file=sys.stderr
        )
        print("Please set your API key:", file=sys.stderr)
        print("  export OPENWEIGHTS_API_KEY=your_token_here", file=sys.stderr)
        sys.exit(1)

    try:
        return OpenWeights(auth_token=auth_token)
    except Exception as e:
        print(f"Error initializing OpenWeights client: {str(e)}", file=sys.stderr)
        sys.exit(1)


def handle_ls(args) -> int:
    """Handle the ls command - list job IDs only."""
    try:
        ow = get_openweights_client()

        # Use the existing jobs.list() method with RLS
        jobs = ow.jobs.list(limit=args.limit)

        # Filter by status if provided
        if args.status:
            jobs = [job for job in jobs if job.status == args.status]

        # Print only job IDs, one per line (for piping to other commands)
        for job in jobs:
            print(job.id)

        return 0

    except Exception as e:
        print(f"Error listing jobs: {str(e)}", file=sys.stderr)
        return 1
