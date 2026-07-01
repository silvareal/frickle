"""Pluggable text embedders for the pipeline's text track.

The pipeline needs *some* object that turns a list of strings into an (N, 384)
array of unit-normalized embeddings. By default it runs sentence-transformers
locally; the backend can inject an alternative (e.g. one backed by the ahnlich-ai
proxy) without the pipeline package depending on the gRPC client. Both paths must
produce the same MiniLM vectors — verified to ~1e-7 — so the store stays consistent.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from .version import TEXT_MODEL_NAME


@runtime_checkable
class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (len(texts), 384) array of unit-normalized float embeddings."""
        ...


class SentenceTransformerEmbedder:
    """Default embedder: runs all-MiniLM-L6-v2 in-process. Model loads lazily and is
    never pickled (reloaded by name)."""

    def __init__(self, model_name: str = TEXT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype="float64")

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_model"] = None
        return state
