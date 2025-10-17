"""Test JWT exchange and claims."""

import base64
import json
import os

from dotenv import load_dotenv

from supabase import create_client

# Reload environment variables
load_dotenv(override=True)

# Get credentials
supabase_url = os.getenv("SUPABASE_URL")
supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
api_token = "ow_2ba0326355c4f9496268fc08758dc63c4e49c04de1592b34"

if not all([supabase_url, supabase_anon_key, api_token]):
    print("Error: SUPABASE_URL, SUPABASE_ANON_KEY, and OPENWEIGHTS_API_KEY must be set")
    exit(1)

print(f"API Token: {api_token[:15]}...")

# Exchange token
temp_client = create_client(supabase_url, supabase_anon_key)
response = temp_client.rpc(
    "exchange_api_token_for_jwt", {"api_token": api_token}
).execute()

jwt = response.data
print(f"\nJWT: {jwt[:50]}...")

# Decode JWT (without verification, just to see claims)
parts = jwt.split(".")
if len(parts) == 3:
    # Decode payload (add padding if needed)
    payload = parts[1]
    payload += "=" * (4 - len(payload) % 4)
    decoded = base64.urlsafe_b64decode(payload)
    claims = json.loads(decoded)
    print(f"\nJWT Claims:")
    print(json.dumps(claims, indent=2))

    if "organization_id" in claims:
        print(f"\n✓ organization_id found: {claims['organization_id']}")
    else:
        print("\n✗ organization_id NOT found in JWT!")

# Test with authenticated client
from supabase.lib.client_options import ClientOptions

headers = {"Authorization": f"Bearer {jwt}"}
options = ClientOptions(schema="public", headers=headers)
auth_client = create_client(supabase_url, supabase_anon_key, options)

# Try to get org from token
try:
    result = auth_client.rpc("get_organization_from_token").execute()
    print(f"\nget_organization_from_token: {result.data}")
except Exception as e:
    print(f"\nError calling get_organization_from_token: {e}")

# Try to check membership
try:
    org_id = claims.get("organization_id")
    if org_id:
        result = auth_client.rpc("is_organization_member", {"org_id": org_id}).execute()
        print(f"is_organization_member({org_id}): {result.data}")
except Exception as e:
    print(f"Error calling is_organization_member: {e}")
