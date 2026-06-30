from fastapi import APIRouter

from . import health, meta, sync, transactions

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(meta.router)
api_router.include_router(transactions.router)
api_router.include_router(sync.router)

__all__ = ["api_router"]
