"""FastAPI app wiring: lifespan-managed pooled DB + one shared gRPC client, CORS,
correlation-id logging, and the API router. Routes stay thin — all logic lives in
the service layer."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .ahnlich import AhnlichClient
from .config import get_settings
from .db import Database
from .logging_mw import CorrelationLogMiddleware, configure_logging
from .routes import api_router
from .state import AppState


async def _warm_embedding_model(state: AppState) -> None:
    """Warm the text embedder at startup (off the event loop) so the first real
    request isn't penalised — loads the local model, or triggers the ahnlich-ai
    model load. No-op if no fitted pipeline exists yet (pre-seed)."""
    artifacts = state.artifacts
    if artifacts is None:
        return
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: artifacts.pipeline.embedder.encode(["warm up"]))
        logging.getLogger("decision_service").info("embedder warmed")
    except Exception:
        logging.getLogger("decision_service").warning("embedder warm-up skipped", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    configure_logging()
    settings = get_settings()
    db = Database(settings.database_url)
    await db.connect()
    ahnlich = AhnlichClient(settings.ahnlich_host, settings.ahnlich_port)
    state = AppState(settings, db, ahnlich)
    app.state.appstate = state
    await _warm_embedding_model(state)
    try:
        yield
    finally:
        await ahnlich.close()
        await db.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Ahnlich Anomaly Decision Service", lifespan=lifespan)
    app.add_middleware(CorrelationLogMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router)
    return app


app = create_app()
