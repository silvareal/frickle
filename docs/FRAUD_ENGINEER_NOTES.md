# Fraud-engineer notes — reading this demo honestly

**The hook:** find transactions that _behave like_ past flagged cases even when no
hand-written rule fires. A rules engine catches what you already thought to encode.
A similarity signal catches the next case that "rhymes" with a known-bad one — a
near neighbour in feature space — without anyone writing the rule first.

That is the whole claim here, and it is deliberately narrow: **fast, metadata-aware
similarity search over custom structured-data vectors.** Transaction monitoring is
the demonstration domain, not a product claim.

## Signal, not system

This is one _signal_, not a fraud-detection _system_. The verdict you see is a
k-NN vote (k=5, cosine similarity) over an Ahnlich vector store whose labels were
assigned by an unsupervised Isolation Forest. That pairing has two consequences a
reviewer should hold onto:

1. **Outlier ≠ fraud.** Isolation Forest labels statistical outliers. Some fraud is
   not anomalous (it mimics normal spend), and much anomaly is not fraud (a genuine
   large purchase). Every label in this demo is a _similarity/anomaly_ label, never
   a confirmed-fraud label.
2. **The vote inherits those labels.** A new transaction is "flagged" because it
   sits near transactions the unsupervised model called outliers — not because we
   have ground truth that those neighbours were fraud.

## Metrics that actually matter

Fraud is extreme class imbalance (well under 1% in the real world; inflated to
~5–6% here purely for demo visibility). Under that imbalance:

- **Accuracy is meaningless.** A model that flags nothing scores ~99% accuracy and
  catches zero fraud. Don't report it.
- **Use precision and recall at an operating threshold.** Precision = of what we
  flagged, how much was real. Recall = of real fraud, how much we caught.
- **Frame errors in cost terms.** A false positive declines a real customer
  (friction, churn, support cost). A false negative is money out the door plus
  downstream liability. The two are not symmetric, and the right trade-off is a
  **business decision**, not a modelling default. If you add a threshold, make it
  configurable and log it per decision so it is auditable.

## Where k-NN voting is strong vs. weak

- **Strong on dense typologies.** Card-testing produces bursts of near-identical
  micro-charges. Those cluster tightly, so a new card-testing charge lands among
  many flagged neighbours and the vote is decisive. Structuring and odd-hour bursts
  behave similarly — repeated, self-similar events.
- **Weak on lone outliers.** A one-off **whale** transaction has, by definition, no
  dense cluster of similar flagged precedents. Its five nearest neighbours are
  often ordinary high-value purchases, so the vote _clears_ it even though the
  transaction is intrinsically extreme. The demo surfaces this explicitly as an
  **under-flag note** rather than hiding it: the transaction-feature group still
  reports "very high amount", while the neighbour vote says cleared. That
  disagreement is the honest, instructive failure mode of similarity voting on rare
  events — do not rig the data or thresholds to mask it.

## CBN (Nigeria) demonstration typologies

The demo includes a Central-Bank-of-Nigeria-flavoured set of typologies alongside the
generic card-fraud ones: FX structuring (amounts parked just under the FX/BTA
threshold), crypto FX-evasion (naira→crypto to sidestep FX controls), POS-agent
cash-out bursts, USSD micro-bursts, and SIM-swap takeovers. These ride on
region-appropriate channels (`bank_transfer`/NIP, `ussd`, `pos`, `mobile_money`,
`airtime`, `fx_remittance`).

Two honesty points specific to this:

- **Legitimate Nigerian traffic must look legitimate.** Nigeria is a first-class
  region in the generator, with its own normal POS/USSD/transfer activity. The
  intrinsic geo heuristic therefore treats both US and NG as _domestic_ — if it
  flagged every NG transaction as "foreign", fraud would separate trivially on
  country and the signal would be meaningless. The good-vs-fraud contrast has to be
  earned on behaviour, not nationality.
- **These are still unsupervised outliers, not CBN/AML determinations.** Naming a
  typology "FX structuring" describes the _planted synthetic pattern_, not a
  regulatory finding. Nothing here is a compliant AML system; the FX/BTA thresholds
  used are illustrative, not the live CBN circular values.

## Explainability

Because the decision is "near these specific neighbours," it is inspectable: we can
show _which_ past cases drove the call, their similarity, their typology, and their
amounts. We separate the explanation into two groups that must never be conflated —
**neighbour evidence** (how the vote went) and **intrinsic transaction features**
(what is odd about this transaction on its own). Conflating them would dress up an
unsupervised vote as a feature-based rule, which it is not.

## What a real bank deployment additionally requires

The similarity signal is one input among many. A deployable system would add:

- **Confirmed-fraud labels** (chargebacks, investigations) — supervised ground truth.
- **Model-risk governance**: independent validation, documentation, monitoring,
  and challenger models.
- **Explainability for adverse decisions** sufficient for regulatory/customer needs.
- **Audit trails** for every decision and threshold.
- **Fair-lending / disparate-impact testing.**
- **Data-residency, retention, and right-to-erasure** that survive retrains.
- **Ensemble context**: supervised models, deterministic rules, velocity, device/
  geo, and graph features.
- **Case management and human review** for flagged items.

Stating this gap plainly is the credibility. This demo shows that Ahnlich can serve
the similarity signal quickly and with metadata; it does not pretend to be the rest.
