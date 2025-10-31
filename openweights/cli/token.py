"""Token management CLI commands."""

import os
import sys
from datetime import datetime, timedelta, timezone

from openweights import OpenWeights


def add_token_parser(parser):
    """Add arguments for the token command."""
    subparsers = parser.add_subparsers(dest="token_cmd", required=True)

    # ls command
    subparsers.add_parser("ls", help="List all tokens for the organization")

    # create command
    create_parser = subparsers.add_parser("create", help="Create a new API token")
    create_parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name/description for the token",
    )
    create_parser.add_argument(
        "--expires-in-days",
        type=int,
        help="Number of days until token expires (optional, default: no expiration)",
    )

    # revoke command
    revoke_parser = subparsers.add_parser("revoke", help="Revoke (delete) a token")
    revoke_parser.add_argument(
        "--token-id",
        type=str,
        required=True,
        help="Token ID to revoke",
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
        # Create OpenWeights client which will automatically get org_id from token
        return OpenWeights(auth_token=auth_token)
    except Exception as e:
        print(f"Error initializing OpenWeights client: {str(e)}")
        sys.exit(1)


def format_datetime(dt) -> str:
    """Format datetime for display."""
    if dt is None:
        return "Never"
    # Handle both datetime objects and strings
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except ValueError:
            return dt
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def handle_token_ls(args) -> int:
    """Handle the token ls command."""
    try:
        ow = get_openweights_client()
        org_id = ow.organization_id

        # Query api_tokens table
        result = (
            ow._supabase.table("api_tokens")
            .select("*")
            .eq("organization_id", org_id)
            .order("created_at", desc=True)
            .execute()
        )

        if not result.data:
            print(f"No tokens found for organization {org_id}")
            return 0

        print(f"\nTokens for organization {org_id}:")
        print("-" * 80)

        for token in result.data:
            print(f"\nToken ID:   {token['id']}")
            print(f"Name:       {token['name']}")
            print(f"Prefix:     {token['token_prefix']}")
            print(f"Created:    {format_datetime(token['created_at'])}")
            print(f"Expires:    {format_datetime(token.get('expires_at'))}")
            print(f"Last used:  {format_datetime(token.get('last_used_at'))}")

            # Check if revoked
            if token.get("revoked_at"):
                print(f"Status:     REVOKED ({format_datetime(token['revoked_at'])})")
            # Check if expired
            elif token.get("expires_at"):
                expires_at = token["expires_at"]
                if isinstance(expires_at, str):
                    expires_at = datetime.fromisoformat(
                        expires_at.replace("Z", "+00:00")
                    )

                if expires_at < datetime.now(timezone.utc):
                    print("Status:     EXPIRED")
                else:
                    days_left = (expires_at - datetime.now(timezone.utc)).days
                    print(f"Status:     Active ({days_left} days remaining)")
            else:
                print("Status:     Active (no expiration)")

        print("-" * 80)
        print(f"\nTotal: {len(result.data)} token(s)")

        return 0

    except Exception as e:
        print(f"Error listing tokens: {str(e)}")
        return 1


def handle_token_create(args) -> int:
    """Handle the token create command."""
    try:
        ow = get_openweights_client()
        org_id = ow.organization_id

        # Calculate expiration if specified
        expires_at = None
        if args.expires_in_days is not None:
            expires_at = (
                datetime.now(timezone.utc) + timedelta(days=args.expires_in_days)
            ).isoformat()

        # Call the create_api_token RPC function
        result = ow._supabase.rpc(
            "create_api_token",
            {
                "org_id": org_id,
                "token_name": args.name,
                "expires_at": expires_at,
            },
        ).execute()

        if not result.data or len(result.data) == 0:
            print("Error: Failed to create token")
            return 1

        token_data = result.data[0]

        print("\nToken created successfully!")
        print("-" * 80)
        print(f"Token ID:     {token_data['token_id']}")
        print(f"Name:         {args.name}")
        if expires_at:
            print(f"Expires:      {format_datetime(expires_at)}")
        else:
            print(f"Expires:      Never")
        print(f"\nAccess Token: {token_data['token']}")
        print("-" * 80)
        print("\nIMPORTANT: Save this token securely. It will not be shown again.")
        print("You can use it by setting: export OPENWEIGHTS_API_KEY=<token>")

        return 0

    except Exception as e:
        print(f"Error creating token: {str(e)}")
        return 1


def handle_token_revoke(args) -> int:
    """Handle the token revoke command."""
    try:
        ow = get_openweights_client()
        org_id = ow.organization_id

        # First, get token info to show what we're revoking
        token_result = (
            ow._supabase.table("api_tokens")
            .select("name, token_prefix")
            .eq("id", args.token_id)
            .eq("organization_id", org_id)
            .single()
            .execute()
        )

        if not token_result.data:
            print(f"Error: Token {args.token_id} not found in organization {org_id}")
            return 1

        # Confirm before revoking
        print(f"\nWARNING: You are about to revoke token:")
        print(f"  ID:     {args.token_id}")
        print(f"  Name:   {token_result.data['name']}")
        print(f"  Prefix: {token_result.data['token_prefix']}")
        print(
            "\nThis action cannot be undone. Any applications using this token will lose access."
        )

        response = input("\nType 'yes' to confirm: ")
        if response.lower() != "yes":
            print("Revocation canceled.")
            return 0

        # Call the revoke_api_token RPC function
        ow._supabase.rpc(
            "revoke_api_token",
            {
                "token_id": args.token_id,
            },
        ).execute()

        print(f"\nToken {args.token_id} has been revoked successfully.")

        return 0

    except Exception as e:
        print(f"Error revoking token: {str(e)}")
        return 1


def handle_token(args) -> int:
    """Handle the token command."""
    if args.token_cmd == "ls":
        return handle_token_ls(args)
    elif args.token_cmd == "create":
        return handle_token_create(args)
    elif args.token_cmd == "revoke":
        return handle_token_revoke(args)
    else:
        print(f"Unknown token command: {args.token_cmd}")
        return 1
