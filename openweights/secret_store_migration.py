"""Move secrets that are still rows of ``organization_secrets`` into the secret store.

    OW_SECRET_STORE=gcp-secret-manager://<project> SUPABASE_SERVICE_ROLE_KEY=... \
        python -m openweights.secret_store_migration [--apply] [--organization <id>]

Without ``--apply`` it only reports what it would move. It covers the organizations
the hosted cluster manager serves (``OW_MANAGED`` is true): a self-managed cluster
cannot read the deployment's secret store, so its organization's secrets stay in the
table. For every managed secret (``openweights.secret_store.MANAGED_SECRETS``) of
those organizations it

1. writes the value to the store and reads it back,
2. records the secret in ``organization_secret_metadata``,
3. deletes the row from ``organization_secrets``.

Run it only when every cluster manager reads the store (0.17.0 or later with
``OW_SECRET_STORE`` set): an older manager finds its secrets only in the table. It can
be run again at any time; secrets that are already in the store are left alone.

Values are never printed.
"""

import argparse
import os
import sys
from datetime import datetime, timezone

from openweights.secret_store import MANAGED_SECRETS, last_four, secret_store_from_env


def migrate(supabase, store, apply: bool, organization_id=None, out=print) -> int:
    """Move the rows; returns how many secrets were (or would be) moved."""
    query = (
        supabase.table("organization_secrets")
        .select("organization_id, name, value")
        .in_("name", sorted(MANAGED_SECRETS))
    )
    if organization_id:
        query = query.eq("organization_id", organization_id)
    rows = query.execute().data or []
    flags = (
        supabase.table("organization_secrets")
        .select("organization_id, name, value")
        .in_("name", ["OW_MANAGED"])
        .execute()
        .data
        or []
    )
    served = {
        flag["organization_id"]
        for flag in flags
        if (flag["value"] or "").lower() in ("true", "1", "yes")
    }
    moved = 0
    for row in rows:
        org_id, name, value = row["organization_id"], row["name"], row["value"]
        if org_id not in served:
            continue
        if not value:
            out(f"{org_id} {name}: empty, skipped")
            continue
        if not apply:
            out(f"{org_id} {name}: would move")
            moved += 1
            continue
        store.set(org_id, name, value)
        if store.get(org_id, name) != value:
            raise RuntimeError(
                f"{org_id} {name}: the store did not return what was written; "
                "the row was left in place"
            )
        supabase.table("organization_secret_metadata").upsert(
            {
                "organization_id": org_id,
                "name": name,
                "last_four": last_four(value),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ).execute()
        supabase.table("organization_secrets").delete().eq(
            "organization_id", org_id
        ).eq("name", name).execute()
        out(f"{org_id} {name}: moved")
        moved += 1
    return moved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--apply", action="store_true", help="move, not only report")
    parser.add_argument("--organization", help="only this organization id")
    args = parser.parse_args(argv)

    store = secret_store_from_env()
    if store is None:
        print("OW_SECRET_STORE is not set: there is no store to move secrets to.")
        return 2
    from supabase import create_client

    from openweights.client import _SUPABASE_URL

    supabase = create_client(
        os.environ.get("SUPABASE_URL", _SUPABASE_URL),
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )
    moved = migrate(supabase, store, args.apply, args.organization)
    print(f"{moved} secret(s) {'moved' if args.apply else 'to move (dry run)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
