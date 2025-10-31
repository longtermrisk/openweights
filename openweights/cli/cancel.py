"""Cancel jobs CLI command."""

import os
import sys

from openweights import OpenWeights


def add_cancel_parser(parser):
    """Add arguments for the cancel command."""
    parser.add_argument(
        "job_ids",
        nargs="*",
        help="Job IDs to cancel (if not provided, reads from stdin for piping)",
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


def handle_cancel(args) -> int:
    """Handle the cancel command - cancel jobs by ID."""
    try:
        ow = get_openweights_client()

        # Get job IDs from args or stdin
        if args.job_ids:
            job_ids = args.job_ids
        else:
            # Read from stdin for piping (e.g., ow ls | ow cancel)
            job_ids = [line.strip() for line in sys.stdin if line.strip()]

        if not job_ids:
            print("Error: No job IDs provided", file=sys.stderr)
            print("Usage: ow cancel <job_id> [<job_id> ...]", file=sys.stderr)
            print("   or: ow ls | ow cancel", file=sys.stderr)
            return 1

        # Cancel each job
        canceled_count = 0
        skipped_count = 0
        error_count = 0

        for job_id in job_ids:
            try:
                # Retrieve the job to check its status
                job = ow.jobs.retrieve(job_id)

                # Only cancel if in pending or in_progress status
                if job.status in ["pending", "in_progress"]:
                    ow.jobs.cancel(job_id)
                    print(f"Canceled: {job_id}", file=sys.stderr)
                    canceled_count += 1
                else:
                    print(f"Skipped: {job_id} (status: {job.status})", file=sys.stderr)
                    skipped_count += 1

            except Exception as e:
                print(f"Error canceling {job_id}: {str(e)}", file=sys.stderr)
                error_count += 1

        # Print summary
        print(
            f"\nSummary: {canceled_count} canceled, {skipped_count} skipped, {error_count} errors",
            file=sys.stderr,
        )

        return 0 if error_count == 0 else 1

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1
