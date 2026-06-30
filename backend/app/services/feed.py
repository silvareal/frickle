"""Read-side service: shape recent transactions for the decision feed."""

from __future__ import annotations

from ..schemas import TransactionRow
from ..state import AppState

_FIELDS = (
    "id", "event_timestamp", "payment_type", "product_category", "order_price",
    "billing_country", "assigned_label", "similarity_score", "is_fraud",
    "fraud_scenario", "created_at",
)


async def recent_transactions(state: AppState, limit: int = 50) -> list[TransactionRow]:
    rows = await state.db.recent_transactions(limit)
    return [TransactionRow(**{k: r[k] for k in _FIELDS}) for r in rows]
