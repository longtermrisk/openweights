"""Cluster command for running the organization manager locally."""

import os
import sys
from pathlib import Path


def add_cluster_parser(parser):
    """Add arguments for the cluster command."""
    parser.add_argument(
        "--env-file",
        type=str,
        help="Path to .env file with environment variables (optional)",
    )
    parser.add_argument(
        "--super",
        action="store_true",
        dest="supervisor",
        help="Run the supervisor instead of organization manager (requires SUPABASE_SERVICE_ROLE_KEY)",
    )


def load_env_file(env_path: str) -> dict:
    """Load environment variables from a .env file."""
    env_vars = {}
    env_file = Path(env_path)

    if not env_file.exists():
        print(f"Error: .env file not found at {env_path}")
        sys.exit(1)

    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Parse KEY=VALUE
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                env_vars[key] = value

    return env_vars


def handle_cluster(args) -> int:
    """Handle the cluster command."""

    # Load environment variables from file if provided
    if args.env_file:
        print(f"Loading environment from {args.env_file}...")
        env_vars = load_env_file(args.env_file)

        if not env_vars:
            print("Warning: No environment variables found in file")
        else:
            # Update os.environ with loaded variables
            os.environ.update(env_vars)
            print(f"Loaded {len(env_vars)} environment variables")

            # Store the list of custom env var keys so org_manager can pass them to workers
            # We use a special env var to communicate which vars came from the env file
            os.environ["_OW_CUSTOM_ENV_VARS"] = ",".join(env_vars.keys())

    # Validate required environment variables based on mode
    if args.supervisor:
        # Supervisor mode requires service role key
        required_vars = ["SUPABASE_SERVICE_ROLE_KEY"]
        mode_name = "supervisor"
    else:
        # Organization manager mode requires standard keys
        required_vars = ["OPENWEIGHTS_API_KEY"]
        mode_name = "organization manager"

    missing_vars = [var for var in required_vars if var not in os.environ]

    if missing_vars:
        print(f"Error: Missing required environment variables for {mode_name}:")
        for var in missing_vars:
            print(f"  - {var}")
        print()
        print("These can be set in your environment or provided via --env-file")
        return 1

    # Check for RunPod API key (not required for supervisor)
    if not args.supervisor and "RUNPOD_API_KEY" not in os.environ:
        print("Warning: RUNPOD_API_KEY not set")
        print(
            "The cluster manager will attempt to fetch it from the database, "
            "but for self-managed clusters you should set it in your environment."
        )

    print(f"Starting OpenWeights {mode_name}...")
    print()

    try:
        if args.supervisor:
            # Import and run the supervisor
            from openweights.cluster.supervisor import ManagerSupervisor

            supervisor = ManagerSupervisor()
            supervisor.supervise()
        else:
            # Import and run the organization manager
            from openweights.cluster.org_manager import OrganizationManager

            manager = OrganizationManager()
            manager.manage_cluster()
        return 0

    except KeyboardInterrupt:
        print(f"\n{mode_name.capitalize()} stopped by user")
        return 0
    except Exception as e:
        print(f"Error running {mode_name}: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1
