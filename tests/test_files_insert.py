"""Test files insert with API token."""

from dotenv import load_dotenv

load_dotenv(override=True)

from openweights import OpenWeights

ow = OpenWeights()
print(f"Connected to org: {ow.org_name}")
print(f"Organization ID: {ow.organization_id}")

# Try to insert a file record directly
try:
    result = (
        ow._supabase.table("files")
        .insert(
            {
                "id": "test:file-123",
                "filename": "test.txt",
                "purpose": "test",
                "bytes": 100,
                "organization_id": ow.organization_id,
            }
        )
        .execute()
    )
    print(f"✓ File insert succeeded: {result.data}")
except Exception as e:
    print(f"✗ File insert failed: {e}")

    # Check if we can call is_organization_member
    try:
        result = ow._supabase.rpc(
            "is_organization_member", {"org_id": ow.organization_id}
        ).execute()
        print(f"\nis_organization_member check: {result.data}")
    except Exception as e2:
        print(f"is_organization_member error: {e2}")
