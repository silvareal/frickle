from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_state
from ..schemas import TransactionIn, TransactionRow, VerdictOut
from ..services.feed import recent_transactions
from ..services.process import process_transaction
from ..state import AppState

router = APIRouter(tags=["transactions"])


@router.post("/transaction/process", response_model=VerdictOut)
async def process(payload: TransactionIn, state: AppState = Depends(get_state)) -> VerdictOut:
    return await process_transaction(state, payload)


@router.get("/transactions", response_model=list[TransactionRow])
async def list_transactions(
    limit: int = 50, state: AppState = Depends(get_state)
) -> list[TransactionRow]:
    return await recent_transactions(state, limit)
