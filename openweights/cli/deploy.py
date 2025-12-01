"""Deploy command implementation."""

import os
import sys
from pathlib import Path

import runpod

from openweights.cluster import start_runpod


def load_env_file(env_path: str) -> dict:
    """Load environment variables from a .env file."""
    env_vars = {}
    env_file = Path(env_path)

    if not env_file.exists():
        print(f"[ow] Error: .env file not found at {env_path}", file=sys.stderr)
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


def add_deploy_parser(parser):
    """Add arguments for the deploy command."""
    parser.add_argument(
        "--image",
        default="nielsrolf/ow-cluster:v0.7",
        help="Docker image for the cluster.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Name for the pod (defaults to auto-generated).",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env file with environment variables (optional)",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Don't start the dashboard backend",
    )
    parser.add_argument(
        "--no-cluster",
        action="store_true",
        help="Don't start the cluster manager",
    )
    parser.add_argument(
        "--super",
        action="store_true",
        dest="supervisor",
        help="Run cluster manager in supervisor mode",
    )


def handle_deploy(args) -> int:
    """Handle the deploy command."""
    import requests

    # Load environment variables from file if provided
    env_vars = {}
    if args.env_file:
        print(f"[ow] Loading environment from {args.env_file}...")
        env_vars = load_env_file(args.env_file)
        print(f"[ow] Loaded {len(env_vars)} environment variables")

    # Validate required environment variables
    required_vars = [
        "OPENWEIGHTS_API_KEY",
        "HF_ORG",
        "HF_TOKEN",
        "HF_USER",
    ]
    missing_vars = []
    for var in required_vars:
        if var not in env_vars and var not in os.environ:
            missing_vars.append(var)

    if missing_vars:
        print("[ow] Error: Missing required environment variables:", file=sys.stderr)
        for var in missing_vars:
            print(f"[ow]   - {var}", file=sys.stderr)
        print(
            "[ow] These can be set in your environment or provided via --env-file",
            file=sys.stderr,
        )
        return 1

    # Check for RunPod API key
    runpod_api_key = env_vars.get("RUNPOD_API_KEY") or os.environ.get("RUNPOD_API_KEY")
    if not runpod_api_key:
        print(
            "[ow] Error: RUNPOD_API_KEY not found in environment or env-file",
            file=sys.stderr,
        )
        return 1

    # Determine OW_CMD based on flags
    if args.no_serve and args.no_cluster:
        print(
            "[ow] Error: Cannot specify both --no-serve and --no-cluster",
            file=sys.stderr,
        )
        return 1
    elif args.no_serve:
        ow_cmd = "cluster"
    elif args.no_cluster:
        ow_cmd = "serve"
    else:
        ow_cmd = "both"

    # Build environment variables for the pod
    pod_env = {}

    # Add all environment variables from env-file
    pod_env.update(env_vars)

    # Add environment variables from current environment if not in env-file
    for var in required_vars:
        if var not in pod_env and var in os.environ:
            pod_env[var] = os.environ[var]

    # Add OW_CMD
    pod_env["OW_CMD"] = ow_cmd

    # Add _OW_CUSTOM_ENV_VARS to track which vars came from env-file
    if env_vars:
        pod_env["_OW_CUSTOM_ENV_VARS"] = ",".join(env_vars.keys())

    # Add --super flag if specified
    if args.supervisor:
        pod_env["OW_CLUSTER_FLAGS"] = "--super"

    # Prepare RunPod API request
    API = "https://rest.runpod.io/v1/pods"
    headers = {
        "Authorization": f"Bearer {runpod_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "computeType": "CPU",
        "name": args.name or "ow-cluster",
        "imageName": args.image,
        "vcpuCount": 4,
        "cpuFlavorIds": ["cpu3c"],
        "containerDiskInGb": 20,
        "ports": ["8124/http"],
        "env": pod_env,
    }

    # Deploy the pod
    print(f"[ow] Deploying pod with image {args.image}...")
    print(f"[ow] OW_CMD: {ow_cmd}")
    if args.supervisor:
        print("[ow] Running in supervisor mode")

    try:
        r = requests.post(API, json=payload, headers=headers)
        r.raise_for_status()
        result = r.json()

        pod_id = result.get("id")
        print(f"[ow] Successfully deployed pod: {pod_id}")
        print(f"[ow] Pod name: {payload['name']}")
        print(f"[ow] Dashboard (if running): https://{pod_id}-8124.proxy.runpod.net")
        return 0
    except requests.exceptions.RequestException as e:
        print(f"[ow] Error deploying pod: {e}", file=sys.stderr)
        if hasattr(e, "response") and e.response is not None:
            print(f"[ow] Response: {e.response.text}", file=sys.stderr)
        return 1
