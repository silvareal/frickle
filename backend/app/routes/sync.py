from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from ..deps import get_state
from ..services.sync import trigger_sync
from ..state import AppState

router = APIRouter(tags=["sync"])


@router.post("/sync")
async def sync(state: AppState = Depends(get_state)) -> dict[str, Any]:
    return await trigger_sync(state)
