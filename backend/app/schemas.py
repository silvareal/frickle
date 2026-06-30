"""Pydantic models for the decision service."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TransactionIn(BaseModel):
    event_timestamp: datetime
    payment_type: str
    product_category: str
    order_price: float = Field(ge=0)
    customer_job: str = "Unknown"
    customer_email: str = "synthetic@example.com"
    billing_city: str = "Synthville"
    billing_state: str = "NA"
    billing_country: str = "US"
    billing_zip: str = "00000"
    billing_latitude: float = 0.0
    billing_longitude: float = 0.0
    ip_address: str = "0.0.0.0"
    user_agent: str = "demo-console"
    merchant: str = "DemoMerchant"
    idempotency_key: str | None = None


class Neighbor(BaseModel):
    tx_id: int | None
    similarity: float
    label: str
    fraud_scenario: str | None = None
    order_price: float | None = None


class ReasonBreakdown(BaseModel):
    neighbor_evidence: list[str]
    transaction_features: list[str]
    under_flag_note: str | None = None


class VerdictOut(BaseModel):
    transaction_id: int
    verdict: str
    score: float
    fraud_votes: int
    k: int
    latency_ms: float
    neighbors: list[Neighbor]
    reasons: ReasonBreakdown


class TransactionRow(BaseModel):
    id: int
    event_timestamp: datetime
    payment_type: str
    product_category: str
    order_price: float
    billing_country: str
    assigned_label: str | None
    similarity_score: float | None
    is_fraud: bool
    fraud_scenario: str | None
    created_at: datetime
