"""Text embedder backed by the ahnlich-ai proxy, plus the factory that decides
which embedder the pipeline uses.

ahnlich-ai's `convert_store_input_to_embeddings` is a stateless text→vector call
(no store needed). Its all-MiniLM-L6-v2 output matches sentence-transformers to
~1e-7 and is unit-normalized, so it is a drop-in for the local model. The pipeline
calls `encode` synchronously (inside a worker thread or the retrain flow), so we run
the async gRPC call on a private event loop in a fresh thread to stay agnostic to
whatever loop the caller is on.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import numpy as np
from ahnlich_client_py.grpc.ai import query as ai_query
from ahnlich_client_py.grpc.ai.models import AiModel
from ahnlich_client_py.grpc.ai.preprocess import PreprocessAction
from ahnlich_client_py.grpc.keyval import StoreInput
from ahnlich_client_py.grpc.services.ai_service import AiServiceStub
from feature_pipeline import SentenceTransformerEmbedder, UnifiedFeaturePipeline
from grpclib.client import Channel

from .config import Settings

_BATCH = 256


def _run_sync(coro: Any) -> Any:
    """Run a coroutine to completion on a private loop in a fresh thread, so this
    works whether or not the caller already has a running event loop."""
    box: dict[str, Any] = {}

    def runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            box["value"] = loop.run_until_complete(coro)
        except BaseException as exc:  # propagate to the calling thread
            box["error"] = exc
        finally:
            loop.close()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


class AhnlichAIEmbedder:
    """Embeds text via the ahnlich-ai proxy instead of an in-process model."""

    def __init__(self, host: str, port: int, model: AiModel = AiModel.ALL_MINI_LM_L6_V2) -> None:
        self.host = host
        self.port = port
        self.model = model

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        channel = Channel(host=self.host, port=self.port)
        try:
            stub = AiServiceStub(channel)
            resp = await stub.convert_store_input_to_embeddings(
                ai_query.ConvertStoreInputToEmbeddings(
                    store_inputs=[StoreInput(raw_string=t) for t in texts],
                    preprocess_action=PreprocessAction.ModelPreprocessing,
                    model=self.model,
                )
            )
            return [list(v.single.embedding.key) for v in resp.values]
        finally:
            channel.close()

    def encode(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for start in range(0, len(texts), _BATCH):
            rows.extend(_run_sync(self._embed(texts[start : start + _BATCH])))
        return np.asarray(rows, dtype="float64")


def build_embedder(settings: Settings) -> Any:
    if settings.embedder == "ahnlich_ai":
        return AhnlichAIEmbedder(settings.ahnlich_ai_host, settings.ahnlich_ai_port)
    return SentenceTransformerEmbedder()


def attach_embedder(pipeline: UnifiedFeaturePipeline, settings: Settings) -> UnifiedFeaturePipeline:
    """Give a (constructed or freshly-loaded) pipeline the configured embedder."""
    pipeline.set_embedder(build_embedder(settings))
    return pipeline
