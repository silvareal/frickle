from __future__ import annotations

from fastapi import APIRouter
from feature_pipeline.constants import (
    BILLING_COUNTRIES,
    FRAUD_SCENARIOS,
    PAYMENT_TYPES,
    PRODUCT_CATEGORIES,
)

router = APIRouter(tags=["meta"])


@router.get("/meta")
async def meta() -> dict[str, list[str]]:
    """Expose the frozen vocabularies so the frontend never hand-copies them."""
    return {
        "payment_types": PAYMENT_TYPES,
        "product_categories": PRODUCT_CATEGORIES,
        "billing_countries": BILLING_COUNTRIES,
        "fraud_scenarios": FRAUD_SCENARIOS,
    }
