"""Sync flow: run the offline retrain, then refresh the service's cached artifacts
so the new store/pointer takes effect on the online path."""

from __future__ import annotations

from typing import Any

from worker.retrain import run_retrain

from ..state import AppState


async def trigger_sync(state: AppState) -> dict[str, Any]:
    result = await run_retrain(state.db, state.ahnlich, state.settings)
    state.refresh_artifacts()
    return result
