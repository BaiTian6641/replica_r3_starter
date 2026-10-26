# Acceptance-aware memory scoring v2

26 October 2026. `memory.acceptance.v2` separates raw top-one diagnostics from
usable, confidence-accepted retrieval. Original reports and metric definitions
remain unchanged. The new receipt is
`reports/memory_acceptance_20261026/native04_acceptance_v2_final.json`; the
source `rankings.jsonl` has 6,060 rows and unchanged bytes.

## Contract

- Raw top-one is diagnostic only.
- Accepted-correct uses eligible **fact** queries and requires every layer to be
  accepted and correct. Acceptance coverage uses any accepted layer.
- Accepted precision is accepted-correct divided by accepted retrievals.
- Zero vectors and ambiguous tied readouts receive no usable-retrieval credit.
- Abstentions, wrong accepted retrievals and unknown false recall remain in the
  denominator. Layer readouts are not independent for uncertainty; worlds are
  the uncertainty units.
- Address and context-removed arms stay separate from value-readout acceptance.
- Generated-answer accuracy is reported separately when measured. It is not
  inferred from vector rankings.

## Replayed results

| Variant | Raw top-one, query-any | Accepted correct | Accepted precision | Acceptance coverage | Abstention | Unknown false recall |
|---|---:|---:|---:|---:|---:|---:|
| Causal refresh | 118/282 = 41.84% | 42/240 = 17.50% | 54.55% | 77/282 = 27.30% | 72.70% | 7/42 = 16.67% |
| Shuffled values | 38/282 = 13.48% | 12/240 = 5.00% | 16.22% | 74/282 = 26.24% | 73.76% | 7/42 = 16.67% |
| Disabled | 44/282 = 15.60% | 0/240 = 0% | n/a | 0/282 = 0% | 100% | 0/42 = 0% |
| Full history | 36/282 = 12.77% | 18/240 = 7.50% | 17.31% | 104/282 = 36.88% | 63.12% | 19/42 = 45.24% |
| Gold-target oracle | 240/282 = 85.11% | 100/240 = 41.67% | 58.14% | 172/282 = 60.99% | 39.01% | 19/42 = 45.24% |

The causal layer-level view has 112/480 accepted-correct readouts at 23.33%,
with 88.89% precision. The stricter all-layer query view has 42/240 at 17.50%
and 54.55% precision. This distinction is intentional: a query is not accepted
as correct when another layer accepts a wrong value.

Causal correction is 4/24 and unaffected retention 4/60. Zero/tied readouts
account for 42 raw top-one layer-readouts, but receive no usable credit.
Disabled memory has 44 raw top-one query groups and zero accepted retrievals.

## Gates and disposition

Predeclared gates in `configs/memory_acceptance_v2.json` require accepted-correct
rate ≥0.85, accepted precision ≥0.95, correction ≥0.60, unaffected ≥0.70,
unknown false recall ≤0.10 and oracle raw top-one ≥0.85. The oracle raw bound
passes at 85.11%; every candidate learned-quality gate fails. Acceptance coverage
is report-only and cannot be used to disguise a low accepted-correct rate.

Focused tests: **9 passed**, covering zero vectors, empty memory, ties, rejected
confidence, correct/wrong acceptance, unknowns, layer/query aggregation and
missing coverage. Replay was CPU-only; no native training occurred.
