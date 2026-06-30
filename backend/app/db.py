"""All SQL / asyncpg access. No SQL strings live anywhere else in the app."""

from __future__ import annotations

from typing import Any

import asyncpg

_TX_COLUMNS = (
    "event_timestamp, payment_type, product_category, order_price, customer_job, "
    "customer_email, billing_city, billing_state, billing_country, billing_zip, "
    "billing_latitude, billing_longitude, ip_address, user_agent, merchant, "
    "is_fraud, fraud_scenario"
)


class Database:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("database pool not initialized")
        return self._pool

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=10)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()

    async def healthy(self) -> bool:
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    async def insert_transaction(self, tx: dict[str, Any]) -> int:
        idem = tx.get("idempotency_key")
        async with self.pool.acquire() as conn:
            if idem:
                existing = await conn.fetchval(
                    "SELECT id FROM compliance_transactions WHERE idempotency_key = $1", idem
                )
                if existing is not None:
                    return int(existing)
            row = await conn.fetchrow(
                f"""
                INSERT INTO compliance_transactions ({_TX_COLUMNS}, idempotency_key)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)
                RETURNING id
                """,
                tx["event_timestamp"], tx["payment_type"], tx["product_category"],
                tx["order_price"], tx["customer_job"], tx["customer_email"],
                tx["billing_city"], tx["billing_state"], tx["billing_country"],
                tx["billing_zip"], tx["billing_latitude"], tx["billing_longitude"],
                tx["ip_address"], tx["user_agent"], tx["merchant"],
                tx.get("is_fraud", False), tx.get("fraud_scenario"), idem,
            )
            return int(row["id"])

    async def bulk_insert(self, rows: list[dict[str, Any]]) -> int:
        records = [
            (
                r["event_timestamp"], r["payment_type"], r["product_category"], r["order_price"],
                r["customer_job"], r["customer_email"], r["billing_city"], r["billing_state"],
                r["billing_country"], r["billing_zip"], r["billing_latitude"], r["billing_longitude"],
                r["ip_address"], r["user_agent"], r["merchant"], r.get("is_fraud", False),
                r.get("fraud_scenario"),
            )
            for r in rows
        ]
        async with self.pool.acquire() as conn:
            await conn.copy_records_to_table(
                "compliance_transactions",
                records=records,
                columns=[c.strip() for c in _TX_COLUMNS.split(",")],
            )
        return len(records)

    async def update_label(self, tx_id: int, label: str, score: float | None) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE compliance_transactions SET assigned_label=$2, similarity_score=$3 WHERE id=$1",
                tx_id, label, score,
            )

    async def fetch_all_for_training(self) -> list[dict[str, Any]]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM compliance_transactions ORDER BY id")
        return [dict(r) for r in rows]

    async def recent_transactions(self, limit: int = 50) -> list[dict[str, Any]]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM compliance_transactions ORDER BY created_at DESC, id DESC LIMIT $1",
                limit,
            )
        return [dict(r) for r in rows]

    async def truncate(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("TRUNCATE compliance_transactions RESTART IDENTITY")

    async def count(self) -> int:
        async with self.pool.acquire() as conn:
            return int(await conn.fetchval("SELECT COUNT(*) FROM compliance_transactions"))
