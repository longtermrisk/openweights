"""Signup command for creating users, orgs, and API keys."""

import os
import sys
from pathlib import Path
from typing import Optional

from supabase import Client, create_client


def add_signup_parser(parser):
    """Add arguments for the signup command."""
    parser.add_argument("email", help="Email address for the new user")
    parser.add_argument(
        "--org-env",
        type=str,
        help="Path to .env file with organization secrets to import",
    )
    parser.add_argument(
        "--supabase-url",
        type=str,
        default=os.getenv("SUPABASE_URL"),
        help="Supabase project URL (or set SUPABASE_URL env var)",
    )
    parser.add_argument(
        "--supabase-key",
        type=str,
        default=os.getenv("SUPABASE_ANON_KEY"),
        help="Supabase anon key (or set SUPABASE_ANON_KEY env var)",
    )
    parser.add_argument(
        "--password",
        type=str,
        help="Password for the user (will be generated if not provided)",
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


def create_supabase_client(url: str, anon_key: str) -> Client:
    """Create a Supabase client with anon key."""
    return create_client(url, anon_key)


def handle_signup(args) -> int:
    """Handle the signup command."""

    # Validate required environment
    if not args.supabase_url or not args.supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_ANON_KEY must be set")
        print(
            "Either set them as environment variables or pass --supabase-url and --supabase-key"
        )
        return 1

    print(f"Creating user account for {args.email}...")

    # Create Supabase client with anon key
    supabase = create_supabase_client(args.supabase_url, args.supabase_key)

    # Generate password if not provided
    password = args.password
    if not password:
        import secrets
        import string

        alphabet = string.ascii_letters + string.digits + string.punctuation
        password = "".join(secrets.choice(alphabet) for _ in range(24))
        print(f"Generated password: {password}")
        print("(Save this password - it won't be shown again)")

    try:
        # Create user via standard sign_up (works with anon key)
        auth_response = supabase.auth.sign_up(
            {
                "email": args.email,
                "password": password,
            }
        )

        if not auth_response.user:
            print(f"Error: Failed to create user")
            return 1

        user_id = auth_response.user.id
        print(f"✓ User created: {args.email} (ID: {user_id})")

        # Check if email confirmation is required
        if not auth_response.session:
            print()
            print("=" * 60)
            print("Email confirmation required!")
            print(f"Please check {args.email} for a confirmation link.")
            print("After confirming, you can sign in with your password.")
            print("=" * 60)
            return 0

        session_token = auth_response.session.access_token

        # Create authenticated client for the user
        from supabase.lib.client_options import ClientOptions

        user_client = create_client(
            args.supabase_url,
            args.supabase_key,
            options=ClientOptions(headers={"Authorization": f"Bearer {session_token}"}),
        )

        # Create organization
        print(f"Creating organization '{args.email}'...")
        org_response = user_client.rpc(
            "create_organization", {"org_name": args.email}
        ).execute()

        if not org_response.data:
            print("Error: Failed to create organization")
            return 1

        org_id = org_response.data
        print(f"✓ Organization created: {args.email} (ID: {org_id})")

        # Create API token
        print("Creating API token...")
        token_response = user_client.rpc(
            "create_api_token",
            {"org_id": org_id, "token_name": "Default token", "expires_at": None},
        ).execute()

        if not token_response.data or len(token_response.data) == 0:
            print("Error: Failed to create API token")
            return 1

        # Response is array of objects with token_id and token
        token_data = token_response.data[0]
        api_token = token_data["token"]
        token_id = token_data["token_id"]

        print(f"✓ API token created (ID: {token_id})")
        print()
        print("=" * 60)
        print("Your OpenWeights API token:")
        print(api_token)
        print("=" * 60)
        print()
        print("Save this token securely - it won't be shown again!")
        print("Add it to your environment:")
        print(f"  export OPENWEIGHTS_API_KEY={api_token}")
        print()

        # Import organization secrets if provided
        if args.org_env:
            print(f"Importing secrets from {args.org_env}...")
            env_vars = load_env_file(args.org_env)

            if not env_vars:
                print("Warning: No environment variables found in file")
            else:
                for key, value in env_vars.items():
                    try:
                        user_client.rpc(
                            "manage_organization_secret",
                            {
                                "org_id": org_id,
                                "secret_name": key,
                                "secret_value": value,
                            },
                        ).execute()
                        print(f"  ✓ {key}")
                    except Exception as e:
                        print(f"  ✗ {key}: {str(e)}")

                print(f"✓ Imported {len(env_vars)} secrets")

        print()
        print("Setup complete! You can now use OpenWeights:")
        print(f"  export OPENWEIGHTS_API_KEY={api_token}")
        print("  python -c 'from openweights import OpenWeights; ow = OpenWeights()'")

        return 0

    except Exception as e:
        print(f"Error during signup: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1
