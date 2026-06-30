"""Contract test against a REAL Ahnlich container (not a mock). Proves the gRPC
call shapes — create_store / set / get_sim_n — are correct against the actual SDK
and server. This is the thing most likely to be wrong."""

from __future__ import annotations

import pytest

from app.ahnlich import AhnlichClient


@pytest.fixture
async def client(ahnlich_endpoint):
    host, port = ahnlich_endpoint
    c = AhnlichClient(host, port)
    yield c
    await c.close()


async def test_create_set_query_roundtrip(client: AhnlichClient) -> None:
    store = "contract"
    await client.drop_store(store)
    await client.create_store(store, dimension=4, error_if_exists=False)
    assert store in await client.list_stores()

    inserted = await client.set_entries(
        store,
        [
            ([1.0, 0.0, 0.0, 0.0], {"label": "flagged", "tx_id": "10", "fraud_scenario": "card_testing", "order_price": "2.0"}),
            ([0.0, 1.0, 0.0, 0.0], {"label": "cleared", "tx_id": "11", "fraud_scenario": "", "order_price": "75.0"}),
        ],
    )
    assert inserted == 2

    neighbors = await client.get_sim_n(store, [1.0, 0.05, 0.0, 0.0], k=2)
    assert len(neighbors) == 2
    top = neighbors[0]
    assert top["label"] == "flagged"
    assert top["tx_id"] == 10
    assert 0.0 <= top["similarity"] <= 1.0


async def test_ping(client: AhnlichClient) -> None:
    assert await client.ping() is True
