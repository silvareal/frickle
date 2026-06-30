"""Online decision flow (orchestration only — no SQL, no gRPC message building).

Order: write Postgres first → vectorize off the event loop → k-NN query Ahnlich →
vote → derive reason → index the new vector back → update Postgres → return verdict.
If Ahnlich is unreachable the row is still persisted (label 'deferred') and a clear
response is returned rather than a 500."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..classify import derive_reasons, majority_vote
from ..schemas import Neighbor, ReasonBreakdown, TransactionIn, VerdictOut
from ..state import AppState

DEFERRED = "deferred"


async def process_transaction(state: AppState, payload: TransactionIn) -> VerdictOut:
    tx = payload.model_dump()
    tx_id = await state.db.insert_transaction(tx)

    artifacts = state.artifacts
    if artifacts is None:
        return _deferred_verdict(tx_id, reason="No fitted pipeline yet — run a sync to train.")

    loop = asyncio.get_running_loop()
    vector = await loop.run_in_executor(
        None, lambda: artifacts.pipeline.transform_online_event(tx)[0].tolist()
    )

    store = artifacts.manifest["active_store"]
    try:
        start = time.perf_counter()
        neighbors = await state.ahnlich.get_sim_n(store, vector, state.settings.knn_k)
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
    except Exception:
        logging.getLogger("decision_service").exception("ahnlich query failed for tx %s", tx_id)
        return _deferred_verdict(tx_id, reason="Ahnlich unreachable — row persisted, scoring deferred.")

    verdict, fraud_votes, score = majority_vote(neighbors)
    reasons = derive_reasons(tx, neighbors, verdict, fraud_votes)

    await _index_back(state, store, vector, tx_id, verdict, tx)
    await state.db.update_label(tx_id, verdict, score)

    return VerdictOut(
        transaction_id=tx_id,
        verdict=verdict,
        score=score,
        fraud_votes=fraud_votes,
        k=len(neighbors),
        latency_ms=latency_ms,
        neighbors=[Neighbor(**n) for n in neighbors],
        reasons=ReasonBreakdown(**reasons),
    )


async def _index_back(
    state: AppState, store: str, vector: list[float], tx_id: int, verdict: str, tx: dict[str, Any]
) -> None:
    meta = {
        "label": verdict,
        "tx_id": str(tx_id),
        "fraud_scenario": "",  # online events have no ground-truth scenario
        "order_price": str(tx["order_price"]),
    }
    try:
        await state.ahnlich.set_entries(store, [(vector, meta)])
    except Exception:
        pass  # indexing is best-effort; the row is already the SSOT in Postgres


def _deferred_verdict(tx_id: int, reason: str) -> VerdictOut:
    return VerdictOut(
        transaction_id=tx_id,
        verdict=DEFERRED,
        score=0.0,
        fraud_votes=0,
        k=0,
        latency_ms=0.0,
        neighbors=[],
        reasons=ReasonBreakdown(neighbor_evidence=[reason], transaction_features=[], under_flag_note=None),
    )
