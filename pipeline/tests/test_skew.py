"""Keystone test: the offline (history) and online (single event) paths must
produce identical vectors for the same input. If this ever fails, the change is
wrong — never weaken this test to make it pass."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_pipeline import UnifiedFeaturePipeline
from feature_pipeline.constants import BILLING_COUNTRIES


def _rows() -> list[dict]:
    return [
        {
            "event_timestamp": "2025-03-04T14:23:11Z",
            "payment_type": "credit_card",
            "product_category": "electronics",
            "order_price": 249.99,
            "billing_country": "US",
            "billing_latitude": 40.71,
            "billing_longitude": -74.0,
            "customer_job": "Accountant",
            "merchant": "ShopFast",
            "user_agent": "Mozilla/5.0",
        },
        {
            "event_timestamp": "2025-06-01T03:05:59Z",
            "payment_type": "crypto",
            "product_category": "jewelry",
            "order_price": 9500.0,
            "billing_country": "NG",
            "billing_latitude": 9.07,
            "billing_longitude": 7.49,
            "customer_job": "Engineer",
            "merchant": "GoldDirect",
            "user_agent": "curl/8.0",
        },
    ]


@pytest.fixture(scope="module")
def fitted_pipeline() -> UnifiedFeaturePipeline:
    pipe = UnifiedFeaturePipeline()
    pipe.fit_transform_history(pd.DataFrame(_rows()))
    return pipe


def test_offline_online_parity(fitted_pipeline: UnifiedFeaturePipeline) -> None:
    df = pd.DataFrame(_rows())
    offline = fitted_pipeline.fit_transform_history(df)
    for i, row in enumerate(_rows()):
        online = fitted_pipeline.transform_online_event(row)
        assert online.shape == (1, offline.shape[1])
        np.testing.assert_allclose(online[0], offline[i], rtol=1e-5, atol=1e-6)


def test_dimension_frozen_on_unseen_category(fitted_pipeline: UnifiedFeaturePipeline) -> None:
    base = _rows()[0]
    seen = fitted_pipeline.transform_online_event(base)
    unseen = dict(base, payment_type="totally_new_method", billing_country="ZZ")
    out = fitted_pipeline.transform_online_event(unseen)
    assert out.shape == seen.shape  # unseen categories must not change dimension


def test_manifest_dimension_matches(fitted_pipeline: UnifiedFeaturePipeline) -> None:
    # numerical(3) + onehot(payment 10 + category 10 + country 10 = 30) + ts(6) + text(384)
    assert len(BILLING_COUNTRIES) == 10
    assert fitted_pipeline.output_dim == 3 + 30 + 6 + 384 == 423
    assert fitted_pipeline.manifest()["output_dim"] == 423
