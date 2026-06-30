"""UnifiedFeaturePipeline — turns a transaction into one fixed-dimension vector.

Four tracks, concatenated in a fixed order:
  1. numerical   — StandardScaler over NUMERICAL_FEATURES (fitted)
  2. categorical — OneHotEncoder over FROZEN_CATEGORIES (frozen → stable dimension)
  3. timestamp   — year/month/day/hour/minute/second, fixed normalization (no fit)
  4. text        — sentence-transformers embedding of a composed text blob

The same private `_vectorize` runs for both the offline (history) and online
(single event) paths, which is what makes training/serving skew impossible: there
is one code path, exercised two ways. `test_skew.py` is the keystone guard.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import (
    CATEGORICAL_FEATURES,
    FROZEN_CATEGORIES,
    NUMERICAL_FEATURES,
    TIMESTAMP_FEATURE,
)
from .version import PIPELINE_VERSION, TEXT_MODEL_NAME

# Fixed divisors keep the timestamp track on a comparable scale to the other tracks
# without fitting state, so the online and offline paths are trivially identical.
_TS_PARTS = ["year", "month", "day", "hour", "minute", "second"]
_TS_DIVISORS = {"year": 1.0, "month": 12.0, "day": 31.0, "hour": 24.0, "minute": 60.0, "second": 60.0}
_TS_OFFSETS = {"year": 2020.0}

# The text track is 384 dims of descriptive free text (job/merchant/user-agent).
# Left at full magnitude it dominates cosine similarity and drowns out the
# structured signal (amount, channel, country, hour) that actually defines the
# typologies. Down-weight it so similarity is driven by the structured features —
# the whole point being "custom structured-data vectors". Applied identically
# offline and online, so the skew guarantee is unaffected.
TEXT_WEIGHT = 0.1

# Embedding width of all-MiniLM-L6-v2. The structured block (everything before the
# text track) is the first `output_dim - TEXT_DIM` columns.
TEXT_DIM = 384


def compose_text(row: dict[str, Any]) -> str:
    """Compose the free-text track from intrinsic descriptive fields."""
    job = str(row.get("customer_job", "") or "")
    merchant = str(row.get("merchant", "") or "")
    agent = str(row.get("user_agent", "") or "")
    return f"{job} | {merchant} | {agent}".strip()


class UnifiedFeaturePipeline:
    def __init__(self) -> None:
        self.version = PIPELINE_VERSION
        self.text_model_name = TEXT_MODEL_NAME
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(
            categories=FROZEN_CATEGORIES,
            handle_unknown="ignore",
            sparse_output=False,
        )
        self.fitted = False
        self.output_dim: int | None = None
        self._text_model: Any = None  # lazy; excluded from pickle

    # --- text model (lazy, not pickled) ----------------------------------------------
    @property
    def text_model(self) -> Any:
        if self._text_model is None:
            from sentence_transformers import SentenceTransformer

            self._text_model = SentenceTransformer(self.text_model_name)
        return self._text_model

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_text_model"] = None  # reload by name after deserialization
        return state

    # --- framing ---------------------------------------------------------------------
    @staticmethod
    def _to_frame(data: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.reset_index(drop=True).copy()
        if isinstance(data, dict):
            return pd.DataFrame([data])
        return pd.DataFrame(list(data))

    @staticmethod
    def _timestamp_track(df: pd.DataFrame) -> np.ndarray:
        ts = pd.to_datetime(df[TIMESTAMP_FEATURE], utc=True, errors="coerce")
        cols = []
        for part in _TS_PARTS:
            raw = getattr(ts.dt, part).astype("float64").to_numpy()
            cols.append((raw - _TS_OFFSETS.get(part, 0.0)) / _TS_DIVISORS[part])
        return np.column_stack(cols)

    def _text_track(self, df: pd.DataFrame) -> np.ndarray:
        texts = [compose_text(r) for r in df.to_dict("records")]
        emb = self.text_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype="float64") * TEXT_WEIGHT

    # --- core: one path used by both fit and online ----------------------------------
    def _vectorize(self, df: pd.DataFrame) -> np.ndarray:
        numerical = self.scaler.transform(df[NUMERICAL_FEATURES].astype("float64"))
        categorical = self.encoder.transform(df[CATEGORICAL_FEATURES].astype(str))
        timestamp = self._timestamp_track(df)
        text = self._text_track(df)
        return np.hstack([numerical, categorical, timestamp, text]).astype("float64")

    # --- public API ------------------------------------------------------------------
    def fit_transform_history(self, df: pd.DataFrame) -> np.ndarray:
        frame = self._to_frame(df)
        self.scaler.fit(frame[NUMERICAL_FEATURES].astype("float64"))
        self.encoder.fit(frame[CATEGORICAL_FEATURES].astype(str))
        self.fitted = True
        vectors = self._vectorize(frame)
        self.output_dim = int(vectors.shape[1])
        return vectors

    def transform_online_event(self, event: dict[str, Any]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("pipeline is not fitted")
        return self._vectorize(self._to_frame(event))

    @property
    def structured_dim(self) -> int:
        """Width of the structured block (numerical + categorical + timestamp),
        i.e. everything except the trailing text embedding."""
        return (self.output_dim or 0) - TEXT_DIM

    def structured_block(self, vectors: np.ndarray) -> np.ndarray:
        """The structured columns of already-computed vectors. The anomaly detector
        fits on these so its labels reflect transaction structure (amount, channel,
        country, time) rather than descriptive text noise."""
        return vectors[:, : self.structured_dim]

    def manifest(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "text_model": self.text_model_name,
            "output_dim": self.output_dim,
            "fitted": self.fitted,
            "tracks": {
                "numerical": len(NUMERICAL_FEATURES),
                "categorical": sum(len(c) for c in FROZEN_CATEGORIES),
                "timestamp": len(_TS_PARTS),
                "text": TEXT_DIM,
            },
        }
