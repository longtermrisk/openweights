"""Organization compute-cost estimates and lifetime API-key budgets (USD)."""

from decimal import Decimal

from openweights.client.decorators import supabase_retry


class Costs:
    def __init__(self, ow_instance):
        self._ow = ow_instance

    @supabase_retry()
    def report(self, limit=100, offset=0):
        """Return live costs by job, worker, API key and submitting user."""
        return (
            self._ow._supabase.rpc(
                "get_cost_report",
                {
                    "org_id": self._ow.organization_id,
                    "row_limit": limit,
                    "row_offset": offset,
                },
            )
            .execute()
            .data
        )

    @supabase_retry()
    def set_limit(self, api_token_id: str, amount_usd=None):
        """Set a lifetime key budget; None removes it. Requires an admin user JWT.

        Limits stop admission and cancel work after accrued spend reaches the
        threshold. They are not prepaid reservations or exact invoice caps.
        """
        if amount_usd is not None:
            amount = Decimal(str(amount_usd))
            if not amount.is_finite() or amount < 0:
                raise ValueError(
                    "Spending limit must be a finite nonnegative USD amount"
                )
            amount_usd = str(amount)
        self._ow._supabase.rpc(
            "set_spending_limit",
            {
                "org_id": self._ow.organization_id,
                "key_id": api_token_id,
                "amount_usd": amount_usd,
            },
        ).execute()
