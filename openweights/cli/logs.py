"""Display job logs CLI command."""

import os
import sys

from openweights import OpenWeights


def add_logs_parser(parser):
    """Add arguments for the logs command."""
    parser.add_argument(
        "job_id",
        help="Job ID to display logs for",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        help="Specific run ID to show logs for (default: latest run)",
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


def handle_logs(args) -> int:
    """Handle the logs command - display job logs."""
    try:
        ow = get_openweights_client()

        # Get the job
        try:
            job = ow.jobs.retrieve(args.job_id)
        except Exception as e:
            print(f"Error: Job {args.job_id} not found: {str(e)}", file=sys.stderr)
            return 1

        # Get runs for this job
        runs = job.runs

        if not runs:
            print(f"Error: No runs found for job {args.job_id}", file=sys.stderr)
            return 1

        # Select the run to display
        if args.run_id:
            run = next((r for r in runs if r.id == args.run_id), None)
            if not run:
                print(
                    f"Error: Run {args.run_id} not found for job {args.job_id}",
                    file=sys.stderr,
                )
                return 1
        else:
            # Use the latest run (last in the list)
            run = runs[-1]

        # Check if log file exists
        if not run.log_file:
            print(f"Error: No log file available for run {run.id}", file=sys.stderr)
            print(f"Run status: {run.status}", file=sys.stderr)
            return 1

        # Fetch and print the log content
        try:
            log_content = ow.files.content(run.log_file)
            # Print the log content to stdout (as bytes, decoded)
            print(log_content.decode("utf-8"), end="")
            return 0
        except Exception as e:
            print(f"Error fetching log file: {str(e)}", file=sys.stderr)
            return 1

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1
