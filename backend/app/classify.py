"""Pure classification logic: vote, score, derive reasons. No I/O — fully unit
testable. The verdict comes from k-NN similarity voting; reasons are derived to be
faithful to that decision and are split into two groups that must never be
conflated: neighbor evidence vs. intrinsic transaction features."""

from __future__ import annotations

from collections import Counter
from typing import Any

from feature_pipeline.constants import (
    DOMESTIC_COUNTRIES,
    HIGH_AMOUNT,
    HOME_COUNTRY,
    LARGE_TRANSFER,
    MICRO_AMOUNT,
    OFF_HOURS_END,
    STRUCTURING_HIGH,
    STRUCTURING_LOW,
)

_LARGE_TRANSFER_CHANNELS = {"wire", "crypto", "bank_transfer"}

FLAGGED = "flagged"
CLEARED = "cleared"


def majority_vote(neighbors: list[dict[str, Any]]) -> tuple[str, int, float]:
    """Return (verdict, fraud_votes, mean_similarity). A neighbor counts as a fraud
    vote when its assigned label is 'flagged'."""
    if not neighbors:
        return CLEARED, 0, 0.0
    fraud_votes = sum(1 for n in neighbors if n.get("label") == FLAGGED)
    mean_sim = sum(float(n.get("similarity", 0.0)) for n in neighbors) / len(neighbors)
    verdict = FLAGGED if fraud_votes * 2 > len(neighbors) else CLEARED
    return verdict, fraud_votes, round(mean_sim, 4)


def transaction_feature_reasons(tx: dict[str, Any]) -> list[str]:
    """Intrinsic oddities of the submitted transaction, independent of neighbors."""
    reasons: list[str] = []
    price = float(tx.get("order_price", 0.0))
    country = str(tx.get("billing_country", HOME_COUNTRY))
    hour = _hour(tx)
    payment = str(tx.get("payment_type", ""))

    if country not in DOMESTIC_COUNTRIES:
        reasons.append(f"Non-domestic billing country ({country})")
    if hour is not None and hour <= OFF_HOURS_END:
        reasons.append(f"Off-hours transaction ({hour:02d}:00)")
    if price >= HIGH_AMOUNT:
        reasons.append(f"Very high amount (${price:,.2f})")
    if price < MICRO_AMOUNT:
        reasons.append(f"Micro-amount (${price:,.2f}) — card-testing signature")
    if STRUCTURING_LOW <= price < STRUCTURING_HIGH:
        reasons.append(f"Amount just under $10k (${price:,.2f}) — structuring signature")
    if payment in _LARGE_TRANSFER_CHANNELS and price >= LARGE_TRANSFER:
        reasons.append(f"Large {payment} transfer (${price:,.2f})")
    return reasons


def neighbor_evidence_reasons(neighbors: list[dict[str, Any]], fraud_votes: int) -> list[str]:
    k = len(neighbors)
    if k == 0:
        return ["No neighbors available (store empty or unreachable)"]
    reasons = [f"{fraud_votes} of {k} nearest neighbors are flagged"]
    scenarios = Counter(
        n.get("fraud_scenario") for n in neighbors if n.get("label") == FLAGGED and n.get("fraud_scenario")
    )
    if scenarios:
        top, n_top = scenarios.most_common(1)[0]
        reasons.append(f"Flagged neighbors mostly resemble '{top}' ({n_top})")
    return reasons


def _hour(tx: dict[str, Any]) -> int | None:
    ts = tx.get("event_timestamp")
    if ts is None:
        return None
    if hasattr(ts, "hour"):
        return int(ts.hour)
    try:
        from datetime import datetime

        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).hour
    except ValueError:
        return None


def derive_reasons(
    tx: dict[str, Any], neighbors: list[dict[str, Any]], verdict: str, fraud_votes: int
) -> dict[str, Any]:
    feature_reasons = transaction_feature_reasons(tx)
    evidence = neighbor_evidence_reasons(neighbors, fraud_votes)
    under_flag = None
    # Honest surfacing: intrinsically odd but cleared on the neighbor vote. This is
    # where k-NN voting is weak on lone outliers (e.g. whale transactions).
    if verdict == CLEARED and feature_reasons:
        under_flag = (
            "Cleared by neighbor vote despite unusual features — k-NN under-flags "
            "rare lone outliers that have no similar flagged precedent."
        )
    return {
        "neighbor_evidence": evidence,
        "transaction_features": feature_reasons,
        "under_flag_note": under_flag,
    }
