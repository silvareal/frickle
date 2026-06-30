# Behavioral Transaction Monitoring with Ahnlich

A demonstration of a **vector-similarity signal** built on **[Ahnlich DB](https://github.com/deven96/ahnlich)**,
shown on transaction monitoring. It shows that Ahnlich does fast, metadata-aware
similarity search over **custom structured-data vectors**.

> **What this is — and isn't.** This is a demonstration of a _similarity signal_,
> not a deployable fraud-detection system. All data is **synthetic** (Faker). Labels
> come from an unsupervised Isolation Forest — they are statistical outliers, and
> **outlier ≠ confirmed fraud**. The defensible claim is exactly: _fast,
> metadata-aware similarity search over custom structured-data vectors._ See
> [`docs/FRAUD_ENGINEER_NOTES.md`](docs/FRAUD_ENGINEER_NOTES.md) for the honest analysis.

![Ahnlich Anomaly Console — a flagged whale transaction with its neighbour basis and derived reasons](docs/images/console.png)

_The operator console: a $90k whale submitted (left) returns a **FLAGGED** verdict
(right) with the 5/5 neighbour vote, the nearest-neighbour basis, derived reasons,
and the measured Ahnlich k-NN query time. Live decision feed below._

## The problem

A rules engine only catches what someone already thought to encode. The idea here:
find transactions that _behave like_ past flagged cases — near neighbours in feature
space — even when no rule matches. We turn each transaction into one vector, store it
in Ahnlich with its label as metadata, and classify a new transaction by a k-NN vote
(k=5, cosine similarity) against that store.

## Run it in 30 seconds

```bash
cp .env.example .env
make up        # build + start postgres, ahnlich, backend, frontend
make seed      # load 2200 synthetic rows, fit the pipeline, build the store
open http://localhost:5173
```

`make up` waits on healthchecks; the backend runs migrations on boot. Check health:

```bash
curl localhost:8000/api/v1/health
# {"ahnlich":true,"postgres":true,"pipeline_fitted":true}
```

Other targets: `make sync` (retrain), `make test`, `make lint`, `make loadtest`, `make down`.

## Architecture

```
client → FastAPI Decision Service ──► PostgreSQL (source of truth)
                  │                         ▲
                  ├──► Ahnlich DB (vectors) │
                  │         ▲               │
                  └─ shared pipeline lib    │
                            │               │
                   Retrain Worker ──────────┘  (reads SSOT, rebuilds store)
```

- **PostgreSQL** — source of truth for every transaction, label, and score.
- **Ahnlich DB** — in-memory vector store; a **rebuildable cache**, never an SSOT.
  Its entire contents reconstruct from Postgres via the retrain worker. Losing the
  node is a non-event.
- **`pipeline/`** — the shared, versioned feature pipeline (four tracks: scaled
  numerics, frozen one-hot categoricals, timestamp decomposition, sentence-transformer
  text embedding). The _same code path_ runs offline and online, which is what
  prevents training/serving skew — guarded by `pipeline/tests/test_skew.py`. The text
  track is deliberately **down-weighted** (`TEXT_WEIGHT`) so cosine similarity is
  driven by the structured signal (amount, channel, country, time) rather than
  descriptive free text — this is, after all, a structured-data similarity demo.
- **Retrain worker** — fits the pipeline + Isolation Forest, rebuilds the store via
  **atomic swap** (build a new slot, verify, flip the pointer), writes labels back.
  The anomaly detector fits on the **structured block only** (not the 384-dim text
  track), so its outlier labels reflect transaction structure instead of textual
  oddities. `contamination` is a configurable operating threshold.
- **Decision service** — async FastAPI: write Postgres first → vectorize off the
  event loop → k-NN query Ahnlich → vote → derive a reason → index the vector back →
  update Postgres → return the verdict. Degrades gracefully if Ahnlich is down.

The online vector dimension always equals the pipeline's output dimension (423 =
3 numeric + 30 one-hot + 6 timestamp + 384 text), asserted at retrain time.

### Demonstration domains: card fraud + CBN (Nigeria)

The synthetic generator produces two legitimate regions — US and Nigerian (NG)
traffic with region-appropriate channels (NG: `bank_transfer`/NIP, `ussd`, `pos`,
`mobile_money`) and categories (`airtime`, `fx_remittance`). On top of that it
plants ten fraud typologies, round-robin:

- **General:** card-testing, structuring, whale anomaly, account takeover, odd-hour burst.
- **CBN-flavoured (Nigeria):** FX structuring (amounts just under the FX/BTA
  threshold), crypto FX-evasion (naira→crypto to bypass FX controls), POS-agent
  cash-out bursts, USSD micro-bursts, and SIM-swap takeovers.

Because NG is a first-class legitimate region, the intrinsic geo heuristic treats
both US and NG as domestic — legitimate Nigerian transactions are **not** painted
as suspicious; only billing outside the domestic set is. This keeps the
good-vs-fraud contrast honest rather than letting fraud separate trivially on
country. As always: synthetic data, unsupervised labels (outlier ≠ fraud), and a
similarity signal — not a real CBN/AML system.

## Measured latency

Latency is **measured by the running system**, never hardcoded. Reproduce with:

```bash
make loadtest   # fires 200 synthetic transactions, prints p50/p99
```

This reports both end-to-end request latency and the isolated Ahnlich k-NN query
time (the `latency_ms` field returned per decision). Run it on your machine and cite
the number it prints — the demo deliberately does not bake in a figure it can't back up.

A 200-transaction run measured on a developer laptop (Docker Desktop, Apple Silicon,
2200-row store) produced:

| metric                            | p50     | p99     |
| --------------------------------- | ------- | ------- |
| Ahnlich k-NN query (`latency_ms`) | ~7.8 ms | ~45 ms  |
| end-to-end `/process` request     | ~51 ms  | ~146 ms |

These are illustrative of one machine — regenerate your own with `make loadtest`. The
k-NN query is isolated from the rest of the request so the store's contribution is
visible on its own.

## How it maps to Ahnlich's strengths

- **Custom structured-data vectors.** The vectors are not generic image/text
  embeddings — they are purpose-built transaction features. Ahnlich stores and
  searches them directly.
- **Metadata-aware.** Each vector carries `{label, tx_id, fraud_scenario, order_price}`
  as metadata, returned with each neighbour so the decision is explainable.
- **Fast in-memory similarity.** The k-NN query time is isolated and reported per
  decision.
- **Rebuildable by design.** The store is reconstructed from Postgres on every
  retrain via atomic swap, so it is safe to treat as a cache.

## What a real bank deployment would add

Confirmed-fraud labels; model-risk governance and independent validation;
explainability for adverse decisions; audit trails; fair-lending testing;
data-residency/retention surviving retrain; ensemble context (supervised models,
rules, velocity, device/geo, graph); and case management with human review. This
similarity signal is one input among those — stating that gap is the point. The demo
also shows a real weakness on purpose: k-NN voting **under-flags lone outliers**
(e.g. a whale transaction with no similar flagged precedent). That behaviour is
surfaced, not hidden.

## Tests

```bash
make test
```

- `pipeline/tests/test_skew.py` — keystone: offline and online vectors are identical.
- `backend/tests/test_classify.py` — pure vote/score/reason logic.
- `backend/tests/test_pipeline_contract.py` — gRPC calls against a **real Ahnlich** container.
- `backend/tests/test_api.py` — full `/process` path against real Postgres + Ahnlich,
  plus the Ahnlich-down degradation path.
