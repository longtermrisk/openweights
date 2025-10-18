"""Worker CLI command."""

import os
import sys
from pathlib import Path


def add_worker_parser(parser):
    """Add worker subcommand arguments."""
    parser.add_argument(
        "--env-file",
        type=str,
        help="Path to .env file to load environment variables from",
    )


def handle_worker(args):
    """Handle the worker command."""
    # Load environment variables from .env file if provided
    if args.env_file:
        env_file = Path(args.env_file)
        if not env_file.exists():
            print(f"Error: Environment file not found: {env_file}", file=sys.stderr)
            return 1

        # Load the env file
        from dotenv import load_dotenv

        load_dotenv(env_file)
        print(f"Loaded environment variables from {env_file}")

    # Import and run the worker
    try:
        from openweights.worker.main import Worker

        # Set IS_LOCAL environment variable to indicate this is a local worker
        os.environ["IS_LOCAL"] = "true"

        worker = Worker()
        print(f"Worker initialized with ID: {worker.worker_id}")
        worker.find_and_execute_job()
        return 0
    except KeyboardInterrupt:
        print("\nWorker stopped by user")
        return 0
    except Exception as e:
        print(f"Error running worker: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
