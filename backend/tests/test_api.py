"""Integration test for the full /process path against real Postgres + Ahnlich,
plus the graceful-degradation path when Ahnlich is unreachable."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone

import pytest

from app.ahnlich import AhnlichClient
from app.config import Settings
from app.db import Database
from app.schemas import TransactionIn
from app.services.process import process_transaction
from app.state import AppState
from data.generate_transactions import generate
from worker.retrain import run_retrain

_DDL = open(__file__.replace("test_api.py", "schema.sql")).read()


@pytest.fixture
async def state(postgres_dsn, ahnlich_endpoint):
    host, port = ahnlich_endpoint
    tmp = tempfile.mkdtemp()
    settings = Settings(
        database_url=postgres_dsn, ahnlich_host=host, ahnlich_port=port, artifact_dir=tmp,
        store_name="itest",
    )
    db = Database(postgres_dsn)
    await db.connect()
    async with db.pool.acquire() as conn:
        await conn.execute(_DDL)
    rows = [r.__dict__ for r in generate(60, fraud_rate=0.15, seed=3)]
    for r in rows:
        r["event_timestamp"] = datetime.fromisoformat(r["event_timestamp"])
    await db.bulk_insert(rows)

    ahnlich = AhnlichClient(host, port)
    st = AppState(settings, db, ahnlich)
    yield st
    await ahnlich.close()
    await db.close()


async def test_full_process_path(state: AppState) -> None:
    summary = await run_retrain(state.db, state.ahnlich, state.settings)
    assert summary["dimension"] == 423
    assert summary["indexed"] == summary["trained_rows"]
    state.refresh_artifacts()

    payload = TransactionIn(
        event_timestamp=datetime(2025, 5, 1, 3, tzinfo=timezone.utc),
        payment_type="wire", product_category="jewelry", order_price=9500.0,
        billing_country="NG",
    )
    verdict = await process_transaction(state, payload)
    assert verdict.transaction_id > 0
    assert verdict.verdict in {"flagged", "cleared"}
    assert verdict.k == state.settings.knn_k
    assert verdict.neighbors
    assert verdict.reasons.transaction_features  # structuring + large wire present


async def test_degradation_when_ahnlich_down(state: AppState) -> None:
    await run_retrain(state.db, state.ahnlich, state.settings)
    state.refresh_artifacts()
    # Point the client at a dead port to simulate an unreachable store.
    broken = AppState(state.settings, state.db, AhnlichClient("127.0.0.1", 9))
    broken._artifacts = state._artifacts
    payload = TransactionIn(
        event_timestamp=datetime(2025, 5, 1, 12, tzinfo=timezone.utc),
        payment_type="credit_card", product_category="electronics", order_price=50.0,
    )
    verdict = await process_transaction(broken, payload)
    assert verdict.verdict == "deferred"
    assert verdict.transaction_id > 0  # row still persisted
    await broken.ahnlich.close()
