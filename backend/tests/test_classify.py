"""Unit tests for the pure classification logic (no I/O)."""

from __future__ import annotations

from datetime import datetime, timezone

from app.classify import (
    CLEARED,
    FLAGGED,
    derive_reasons,
    majority_vote,
    transaction_feature_reasons,
)


def _n(label: str, sim: float, scenario: str | None = None) -> dict:
    return {"label": label, "similarity": sim, "fraud_scenario": scenario}


def test_majority_vote_flags_on_majority() -> None:
    neighbors = [_n(FLAGGED, 0.9, "card_testing")] * 3 + [_n(CLEARED, 0.5)] * 2
    verdict, votes, score = majority_vote(neighbors)
    assert verdict == FLAGGED
    assert votes == 3
    assert 0.0 < score <= 1.0


def test_majority_vote_clears_on_tie() -> None:
    neighbors = [_n(FLAGGED, 0.8)] * 2 + [_n(CLEARED, 0.4)] * 2
    verdict, votes, _ = majority_vote(neighbors)
    assert verdict == CLEARED and votes == 2


def test_empty_neighbors_clears() -> None:
    assert majority_vote([]) == (CLEARED, 0, 0.0)


def test_transaction_features_detect_structuring_and_offhours() -> None:
    tx = {
        "order_price": 9500.0,
        "billing_country": "NG",  # domestic in this demo — must NOT be flagged as foreign
        "payment_type": "wire",
        "event_timestamp": datetime(2025, 1, 1, 3, tzinfo=timezone.utc),
    }
    reasons = transaction_feature_reasons(tx)
    joined = " ".join(reasons)
    assert "Non-domestic" not in joined
    assert "structuring signature" in joined
    assert "Off-hours" in joined
    assert "Large wire" in joined


def test_non_domestic_country_flagged() -> None:
    tx = {"order_price": 100.0, "billing_country": "RU", "payment_type": "wire"}
    assert any("Non-domestic billing country (RU)" in r for r in transaction_feature_reasons(tx))


def test_micro_amount_card_testing() -> None:
    tx = {"order_price": 1.5, "billing_country": "US", "payment_type": "credit_card"}
    assert any("card-testing" in r for r in transaction_feature_reasons(tx))


def test_under_flag_note_when_odd_but_cleared() -> None:
    tx = {"order_price": 90000.0, "billing_country": "US", "payment_type": "credit_card"}
    neighbors = [_n(CLEARED, 0.6)] * 5
    reasons = derive_reasons(tx, neighbors, CLEARED, 0)
    assert reasons["under_flag_note"] is not None
    assert reasons["transaction_features"]  # whale-scale feature present


def test_no_under_flag_when_flagged() -> None:
    tx = {"order_price": 90000.0, "billing_country": "US", "payment_type": "credit_card"}
    neighbors = [_n(FLAGGED, 0.9, "whale_anomaly")] * 5
    reasons = derive_reasons(tx, neighbors, FLAGGED, 5)
    assert reasons["under_flag_note"] is None
    assert any("whale_anomaly" in r for r in reasons["neighbor_evidence"])
