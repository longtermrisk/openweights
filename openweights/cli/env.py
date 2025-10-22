"""Environment/secrets management CLI commands."""

import os
import sys
from pathlib import Path

from dotenv import dotenv_values

from openweights import OpenWeights


def add_env_parser(parser):
    """Add arguments for the env command."""
    subparsers = parser.add_subparsers(dest="env_cmd", required=True)

    # import command
    import_parser = subparsers.add_parser(
        "import", help="Import environment variables from a .env file"
    )
    import_parser.add_argument(
        "env_file",
        type=str,
        help="Path to .env file to import",
    )

    # show command
    subparsers.add_parser(
        "show", help="Show organization secrets (environment variables)"
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


def handle_env_import(args) -> int:
    """Handle the env import command."""
    try:
        env_file = Path(args.env_file)
        if not env_file.exists():
            print(f"Error: File not found: {env_file}")
            return 1

        # Load environment variables from file
        env_vars = dotenv_values(env_file)
        if not env_vars:
            print(f"Warning: No environment variables found in {env_file}")
            return 0

        # Print warning about security
        print("\n" + "=" * 80)
        print("WARNING: SECURITY NOTICE")
        print("=" * 80)
        print(
            "You are about to upload sensitive credentials to the OpenWeights database."
        )
        print(
            "These secrets will be stored in the organization_secrets table and used by"
        )
        print("the cluster manager to provision workers on your behalf.")
        print()
        print("Secrets to be uploaded:")
        for key in env_vars.keys():
            print(f"  - {key}")
        print()
        print("IMPORTANT: Only proceed if you trust the OpenWeights infrastructure and")
        print(
            "understand that these credentials will be accessible to cluster managers."
        )
        print("=" * 80)

        response = input("\nType 'yes' to confirm upload: ")
        if response.lower() != "yes":
            print("Import canceled.")
            return 0

        # Get OpenWeights client
        ow = get_openweights_client()
        org_id = ow.organization_id

        # Upload each secret to the database
        uploaded_count = 0
        for name, value in env_vars.items():
            if value is None:
                continue

            try:
                # Check if secret already exists
                existing = (
                    ow._supabase.table("organization_secrets")
                    .select("id")
                    .eq("organization_id", org_id)
                    .eq("name", name)
                    .execute()
                )

                if existing.data:
                    # Update existing secret
                    ow._supabase.table("organization_secrets").update(
                        {"value": value}
                    ).eq("organization_id", org_id).eq("name", name).execute()
                    print(f"Updated: {name}")
                else:
                    # Insert new secret
                    ow._supabase.table("organization_secrets").insert(
                        {
                            "organization_id": org_id,
                            "name": name,
                            "value": value,
                        }
                    ).execute()
                    print(f"Added: {name}")

                uploaded_count += 1

            except Exception as e:
                print(f"Error uploading {name}: {str(e)}")
                continue

        print(f"\nSuccessfully uploaded {uploaded_count} secret(s)")

        # Check if all required secrets are present
        required_secrets = [
            "HF_TOKEN",
            "HF_ORG",
            "HF_USER",
            "OPENWEIGHTS_API_KEY",
            "RUNPOD_API_KEY",
        ]
        missing = [s for s in required_secrets if s not in env_vars]

        if missing:
            print("\nWarning: Missing required secrets for cluster management:")
            for secret in missing:
                print(f"  - {secret}")
            print(
                "\nCluster manager will not be able to provision workers until all required"
            )
            print("secrets are provided.")
        else:
            print(print("\nTo start the cluster manager, run: \now manage start`"))

        return 0

    except Exception as e:
        print(f"Error importing environment: {str(e)}")
        return 1


def handle_env_show(args) -> int:
    """Handle the env show command."""
    try:
        ow = get_openweights_client()
        org_id = ow.organization_id

        # Query organization_secrets table
        result = (
            ow._supabase.table("organization_secrets")
            .select("name, value")
            .eq("organization_id", org_id)
            .order("name")
            .execute()
        )

        if not result.data:
            print(f"No secrets found for organization {org_id}")
            print("\nTo import secrets, use:")
            print("  ow env import path/to/.env")
            return 0

        print(f"\nOrganization Secrets (org_id: {org_id}):")
        print("-" * 80)

        for secret in result.data:
            # Print in FOO=bar format
            print(f"{secret['name']}={secret['value']}")

        print("-" * 80)
        print(f"\nTotal: {len(result.data)} secret(s)")

        # Check if all required secrets are present
        required_secrets = [
            "HF_TOKEN",
            "HF_ORG",
            "HF_USER",
            "OPENWEIGHTS_API_KEY",
            "RUNPOD_API_KEY",
        ]
        existing_names = {s["name"] for s in result.data}
        missing = [s for s in required_secrets if s not in existing_names]

        if missing:
            print("\nWarning: Missing required secrets for cluster management:")
            for secret in missing:
                print(f"  - {secret}")
        else:
            print("\nAll required secrets are present.")
            print("You can enable cluster management with: ow manage start")

        return 0

    except Exception as e:
        print(f"Error showing environment: {str(e)}")
        return 1


def handle_env(args) -> int:
    """Handle the env command."""
    if args.env_cmd == "import":
        return handle_env_import(args)
    elif args.env_cmd == "show":
        return handle_env_show(args)
    else:
        print(f"Unknown env command: {args.env_cmd}")
        return 1
