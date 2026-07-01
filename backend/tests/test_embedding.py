"""Contract test for the ahnlich-ai embedder. Runs only when TEST_AHNLICH_AI_HOST
is set (a reachable ahnlich-ai proxy). Proves its all-MiniLM-L6-v2 output is the
right shape, deterministic, unit-normalized, and matches sentence-transformers —
the conditions that make offloading the text track safe."""

from __future__ import annotations

import os

import numpy as np
import pytest

from app.embedding import AhnlichAIEmbedder
from feature_pipeline import SentenceTransformerEmbedder

_HOST = os.getenv("TEST_AHNLICH_AI_HOST")
_PORT = int(os.getenv("TEST_AHNLICH_AI_PORT", "1370"))

pytestmark = pytest.mark.skipif(_HOST is None, reason="ahnlich-ai endpoint not provided")

_TEXTS = ["Accountant | ShopFast | Mozilla/5.0", "Trader | P2PCoin | curl/8.0"]


def test_shape_deterministic_and_normalized() -> None:
    emb = AhnlichAIEmbedder(_HOST, _PORT)
    a = emb.encode(_TEXTS)
    b = emb.encode(_TEXTS)
    assert a.shape == (2, 384)
    np.testing.assert_allclose(a, b, atol=0.0)  # deterministic
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), [1.0, 1.0], atol=1e-4)


def test_matches_sentence_transformers() -> None:
    ai = AhnlichAIEmbedder(_HOST, _PORT).encode(_TEXTS)
    st = SentenceTransformerEmbedder().encode(_TEXTS)
    assert np.abs(ai - st).max() < 1e-5
