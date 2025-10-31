"""Quick test to check database state."""

import os

from supabase import create_client

# Get credentials from environment
supabase_url = os.getenv("SUPABASE_URL")
service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not supabase_url or not service_role_key:
    print("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
    exit(1)

# Create admin client
supabase = create_client(supabase_url, service_role_key)

# Check migrations
try:
    result = (
        supabase.table("supabase_migrations.schema_migrations")
        .select("version, name")
        .execute()
    )
    print("Applied migrations:")
    for row in result.data:
        print(f"  {row}")
except Exception as e:
    print(f"Error checking migrations: {e}")

# Check if is_organization_member exists
try:
    result = supabase.rpc(
        "is_organization_member", {"org_id": "fa25f56b-a8f8-442c-93a0-77a7161d386b"}
    ).execute()
    print(f"\nis_organization_member test: {result.data}")
except Exception as e:
    print(f"\nError calling is_organization_member: {e}")
