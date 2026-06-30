"""The only module that touches the Ahnlich gRPC client. All store operations
(create / set / get_sim_n / drop / list) live here; nothing else constructs gRPC
messages. Client is async gRPC (ahnlich-client-py==0.3.0): a grpclib Channel + a
DbServiceStub with typed request messages."""

from __future__ import annotations

from typing import Any

from ahnlich_client_py.grpc.algorithm.algorithms import Algorithm
from ahnlich_client_py.grpc.db import query
from ahnlich_client_py.grpc.keyval import DbStoreEntry, StoreKey, StoreValue
from ahnlich_client_py.grpc.metadata import MetadataValue
from ahnlich_client_py.grpc.services.db_service import DbServiceStub
from grpclib.client import Channel


def _to_value(metadata: dict[str, str]) -> StoreValue:
    return StoreValue(value={k: MetadataValue(raw_string=str(v)) for k, v in metadata.items()})


def _from_value(value: StoreValue) -> dict[str, str]:
    return {k: mv.raw_string for k, mv in value.value.items()}


class AhnlichClient:
    """Thin async wrapper around a single shared gRPC channel."""

    def __init__(self, host: str, port: int) -> None:
        self._channel = Channel(host=host, port=port)
        self._stub = DbServiceStub(self._channel)

    async def close(self) -> None:
        self._channel.close()

    async def ping(self) -> bool:
        try:
            await self._stub.ping(query.Ping())
            return True
        except Exception:
            return False

    async def list_stores(self) -> list[str]:
        resp = await self._stub.list_stores(query.ListStores())
        return [s.name for s in resp.stores]

    async def create_store(self, store: str, dimension: int, error_if_exists: bool = True) -> None:
        await self._stub.create_store(
            query.CreateStore(
                store=store,
                dimension=dimension,
                create_predicates=["label", "fraud_scenario"],
                non_linear_indices=[],
                error_if_exists=error_if_exists,
            )
        )

    async def drop_store(self, store: str) -> None:
        try:
            await self._stub.drop_store(query.DropStore(store=store, error_if_not_exists=False))
        except Exception:
            pass

    async def set_entries(self, store: str, entries: list[tuple[list[float], dict[str, str]]]) -> int:
        inputs = [
            DbStoreEntry(key=StoreKey(key=vec), value=_to_value(meta)) for vec, meta in entries
        ]
        await self._stub.set(query.Set(store=store, inputs=inputs))
        return len(inputs)

    async def get_sim_n(self, store: str, vector: list[float], k: int) -> list[dict[str, Any]]:
        resp = await self._stub.get_sim_n(
            query.GetSimN(
                store=store,
                search_input=StoreKey(key=vector),
                closest_n=k,
                algorithm=Algorithm.CosineSimilarity,
                condition=None,
            )
        )
        out: list[dict[str, Any]] = []
        for entry in resp.entries:
            meta = _from_value(entry.value)
            out.append(
                {
                    "similarity": float(entry.similarity.value),
                    "label": meta.get("label", "cleared"),
                    "fraud_scenario": meta.get("fraud_scenario") or None,
                    "tx_id": int(meta["tx_id"]) if meta.get("tx_id", "").isdigit() else None,
                    "order_price": float(meta["order_price"]) if meta.get("order_price") else None,
                }
            )
        return out
