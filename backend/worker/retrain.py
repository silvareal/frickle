"""Offline retrain path: fit pipeline + Isolation Forest over Postgres history,
rebuild the Ahnlich store via atomic swap, write labels back to Postgres.

Atomic swap: build into the *inactive* slot, verify, then flip the manifest pointer.
A failed or partial run leaves the previous store and pointer untouched, so the
live decision path keeps serving the old store.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np
import pandas as pd
from feature_pipeline import UnifiedFeaturePipeline
from sklearn.ensemble import IsolationForest

from app.ahnlich import AhnlichClient
from app.artifacts import load_manifest, next_store_slot, save_artifacts
from app.classify import CLEARED, FLAGGED
from app.config import Settings, get_settings
from app.db import Database

_SET_CHUNK = 500


def _labels_from_model(model: IsolationForest, vectors: np.ndarray) -> list[str]:
    # IsolationForest: -1 == outlier. Outlier ≠ confirmed fraud — these are
    # statistical anomaly labels surfaced as 'flagged'.
    preds = model.predict(vectors)
    return [FLAGGED if p == -1 else CLEARED for p in preds]


async def _build_store(
    ahnlich: AhnlichClient, store: str, vectors: np.ndarray, rows: list[dict[str, Any]], labels: list[str]
) -> int:
    await ahnlich.drop_store(store)  # clear the inactive slot if a prior run left junk
    await ahnlich.create_store(store, dimension=int(vectors.shape[1]), error_if_exists=False)
    total = 0
    batch: list[tuple[list[float], dict[str, str]]] = []
    for vec, row, label in zip(vectors, rows, labels):
        meta = {
            "label": label,
            "tx_id": str(row["id"]),
            "fraud_scenario": str(row.get("fraud_scenario") or ""),
            "order_price": str(row["order_price"]),
        }
        batch.append((vec.tolist(), meta))
        if len(batch) >= _SET_CHUNK:
            total += await ahnlich.set_entries(store, batch)
            batch = []
    if batch:
        total += await ahnlich.set_entries(store, batch)
    return total


async def run_retrain(db: Database, ahnlich: AhnlichClient, settings: Settings) -> dict[str, Any]:
    rows = await db.fetch_all_for_training()
    if not rows:
        raise RuntimeError("no transactions to train on")

    df = pd.DataFrame(rows)
    pipeline = UnifiedFeaturePipeline()
    vectors = pipeline.fit_transform_history(df)
    assert pipeline.output_dim == vectors.shape[1], "pipeline dimension mismatch"

    # Fit the anomaly detector on the structured block only — fitting on the full
    # vector lets the 384-dim text track dominate, so it would flag textually-odd
    # but structurally-normal rows instead of the structured outliers we care about.
    structured = pipeline.structured_block(vectors)
    model = IsolationForest(contamination=settings.if_contamination, random_state=42)
    model.fit(structured)
    labels = _labels_from_model(model, structured)

    manifest = load_manifest(settings.artifact_dir)
    new_store, old_store = next_store_slot(manifest, settings.store_name)
    new_slot = int(new_store.rsplit("_", 1)[-1])

    indexed = await _build_store(ahnlich, new_store, vectors, rows, labels)
    if new_store not in await ahnlich.list_stores() or indexed != len(rows):
        raise RuntimeError("store verification failed; keeping previous store")

    # Verified — flip the pointer atomically, then write labels and drop the old store.
    save_artifacts(settings.artifact_dir, pipeline, model, active_store=new_store, slot=new_slot)
    for row, label in zip(rows, labels):
        await db.update_label(int(row["id"]), label, None)
    if old_store and old_store != new_store:
        await ahnlich.drop_store(old_store)

    flagged = labels.count(FLAGGED)
    return {
        "trained_rows": len(rows),
        "dimension": int(vectors.shape[1]),
        "active_store": new_store,
        "indexed": indexed,
        "flagged": flagged,
        "cleared": len(rows) - flagged,
    }


async def _main() -> None:
    settings = get_settings()
    db = Database(settings.database_url)
    await db.connect()
    ahnlich = AhnlichClient(settings.ahnlich_host, settings.ahnlich_port)
    try:
        result = await run_retrain(db, ahnlich, settings)
        print(json.dumps(result, indent=2))
    finally:
        await ahnlich.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(_main())
