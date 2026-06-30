from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_state
from ..state import AppState

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(state: AppState = Depends(get_state)) -> dict[str, bool]:
    return {
        "ahnlich": await state.ahnlich.ping(),
        "postgres": await state.db.healthy(),
        "pipeline_fitted": state.pipeline_fitted,
    }
