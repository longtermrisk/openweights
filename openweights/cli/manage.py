"""Cluster management control CLI commands."""

import os
import sys

from openweights import OpenWeights


def add_manage_parser(parser):
    """Add arguments for the manage command."""
    subparsers = parser.add_subparsers(dest="manage_cmd", required=True)

    # start command
    subparsers.add_parser(
        "start", help="Enable managed cluster infrastructure (sets OW_MANAGED=true)"
    )

    # stop command
    subparsers.add_parser(
        "stop", help="Disable managed cluster infrastructure (sets OW_MANAGED=false)"
    )


def get_openweights_client():
    """Get OpenWeights client with authentication."""
    auth_token = os.getenv("OPENWEIGHTS_API_KEY")

    if not auth_token:
        print("Error: OPENWEIGHTS_API_KEY environment variable not set")
        print("Please set your API key:")
        print("  export OPENWEIGHTS_API_KEY=your_token_here")
        sys.exit(1)

    try:
        return OpenWeights(auth_token=auth_token)
    except Exception as e:
        print(f"Error initializing OpenWeights client: {str(e)}")
        sys.exit(1)


def check_required_secrets(ow, org_id) -> tuple[bool, list[str]]:
    """Check if all required secrets are present.

    Returns:
        (all_present, missing_secrets)
    """
    required_secrets = [
        "HF_TOKEN",
        "HF_ORG",
        "HF_USER",
        "OPENWEIGHTS_API_KEY",
        "RUNPOD_API_KEY",
    ]

    # Query organization_secrets table
    result = (
        ow._supabase.table("organization_secrets")
        .select("name")
        .eq("organization_id", org_id)
        .execute()
    )

    existing_names = {s["name"] for s in result.data}
    missing = [s for s in required_secrets if s not in existing_names]

    return (len(missing) == 0, missing)


def handle_manage_start(args) -> int:
    """Handle the manage start command."""
    try:
        ow = get_openweights_client()
        org_id = ow.organization_id

        # Check if required secrets are present
        all_present, missing = check_required_secrets(ow, org_id)

        if not all_present:
            print("Error: Cannot enable cluster management - missing required secrets:")
            for secret in missing:
                print(f"  - {secret}")
            print("\nPlease import your environment variables first:")
            print("  ow env import path/to/.env")
            print("\nOr check which secrets are configured:")
            print("  ow env show")
            return 1

        # Set OW_MANAGED to true in organization_secrets
        existing = (
            ow._supabase.table("organization_secrets")
            .select("id")
            .eq("organization_id", org_id)
            .eq("name", "OW_MANAGED")
            .execute()
        )

        if existing.data:
            # Update existing
            ow._supabase.table("organization_secrets").update({"value": "true"}).eq(
                "organization_id", org_id
            ).eq("name", "OW_MANAGED").execute()
        else:
            # Insert new
            ow._supabase.table("organization_secrets").insert(
                {
                    "organization_id": org_id,
                    "name": "OW_MANAGED",
                    "value": "true",
                }
            ).execute()

        print(f"Cluster management enabled for organization {org_id}")
        print(
            "\nThe supervisor will now automatically provision and manage workers for your jobs."
        )
        print(
            "Workers will be created on RunPod using your RUNPOD_API_KEY when jobs are submitted."
        )

        return 0

    except Exception as e:
        print(f"Error enabling cluster management: {str(e)}")
        return 1


def handle_manage_stop(args) -> int:
    """Handle the manage stop command."""
    try:
        ow = get_openweights_client()
        org_id = ow.organization_id

        # Set OW_MANAGED to false in organization_secrets
        existing = (
            ow._supabase.table("organization_secrets")
            .select("id")
            .eq("organization_id", org_id)
            .eq("name", "OW_MANAGED")
            .execute()
        )

        if existing.data:
            # Update existing
            ow._supabase.table("organization_secrets").update({"value": "false"}).eq(
                "organization_id", org_id
            ).eq("name", "OW_MANAGED").execute()
        else:
            # Insert new with false value
            ow._supabase.table("organization_secrets").insert(
                {
                    "organization_id": org_id,
                    "name": "OW_MANAGED",
                    "value": "false",
                }
            ).execute()

        print(f"Cluster management disabled for organization {org_id}")
        print(
            "\nThe supervisor will stop provisioning new workers and terminate the org_manager."
        )
        print("Existing workers will complete their current jobs before shutting down.")

        return 0

    except Exception as e:
        print(f"Error disabling cluster management: {str(e)}")
        return 1


def handle_manage(args) -> int:
    """Handle the manage command."""
    if args.manage_cmd == "start":
        return handle_manage_start(args)
    elif args.manage_cmd == "stop":
        return handle_manage_stop(args)
    else:
        print(f"Unknown manage command: {args.manage_cmd}")
        return 1
