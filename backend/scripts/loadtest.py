"""Fire N transactions at the decision service and print measured p50/p99 latency.
This produces the number the README cites — it is never hardcoded."""

from __future__ import annotations

import argparse
import asyncio
import statistics
from datetime import datetime, timezone

import httpx

from data.generate_transactions import generate


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
    return ordered[k]


async def run(base_url: str, n: int) -> dict[str, float]:
    samples = generate(n, fraud_rate=0.1, seed=7)
    request_latencies: list[float] = []
    knn_latencies: list[float] = []
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        for tx in samples:
            payload = dict(tx.__dict__)
            payload.pop("is_fraud", None)
            payload.pop("fraud_scenario", None)
            payload["event_timestamp"] = datetime.fromisoformat(
                payload["event_timestamp"]
            ).astimezone(timezone.utc).isoformat()
            start = asyncio.get_event_loop().time()
            resp = await client.post("/api/v1/transaction/process", json=payload)
            request_latencies.append((asyncio.get_event_loop().time() - start) * 1000)
            if resp.status_code == 200:
                knn_latencies.append(resp.json().get("latency_ms", 0.0))

    return {
        "requests": float(len(request_latencies)),
        "request_p50_ms": round(_percentile(request_latencies, 50), 3),
        "request_p99_ms": round(_percentile(request_latencies, 99), 3),
        "knn_p50_ms": round(_percentile(knn_latencies, 50), 3),
        "knn_p99_ms": round(_percentile(knn_latencies, 99), 3),
        "knn_mean_ms": round(statistics.mean(knn_latencies), 3) if knn_latencies else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()
    result = asyncio.run(run(args.url, args.n))
    print("Measured decision-path latency (synthetic load):")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
