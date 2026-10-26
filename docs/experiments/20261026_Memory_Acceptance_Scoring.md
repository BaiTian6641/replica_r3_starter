# Acceptance-aware memory scoring v2

26 October 2026. `memory.acceptance.v2` separates raw top-one diagnostics from
usable, confidence-accepted retrieval. Original rankings and historical receipts
remain unchanged. The sole current receipt is
`reports/memory_acceptance_20261026/native04_acceptance_v2_reconciled_uncertainty_20261026.json`;
the source `rankings.jsonl` has 6,060 rows and unchanged bytes.

The receipt manifest
`reports/memory_acceptance_20261026/receipt_manifest.json` names exactly one
current receipt:

- `native04_acceptance_v2_reconciled_uncertainty_20261026.json` — status
  `current`, file SHA-256
  `f2c1d3f1c03c6669fee6ada1cc80beeafbf0a1419ec4114f8a864ca291d4b1ff`,
  canonical receipt digest
  `237f593734c41d5a9a36de0ead3058c817e84c596130ddca8b75766fc3e05c22`.
  Config SHA-256 `e8f0c1cb058295d5fb99df012ea0b13964e7b80cd51990a523a16c507d40280f`;
  source SHA-256 `fa13b963346dd96f3e57042d3587502111d0661ea05e1fba4582a7fabbc558f5`.
- `native04_acceptance_v2_final.json` — status `superseded`, preserved
  byte-for-byte (file SHA-256
  `e040d415f4e26f46ac22711052292ee2c27f2f3582948725a7a63c51ef47af66`,
  canonical digest
  `5cc528b0c2f81717e61e5153c493f87591ada81e1053e056c250c80b8eeab883`).
  Its counts, rates, gates and rejected verdict remain valid, but its
  accepted-precision intervals predate the estimable-world reset.
- `native04_acceptance_v2_superseded_pre_value_arms.json` — status
  `superseded`, preserved byte-for-byte from the former
  `native04_acceptance_v2.json` (file SHA-256
  `f18cb97ccd6d89c70f95982aef3726b6db317c53789a4a44479f0889cb933c9c`,
  canonical digest
  `6eddce32e25985e8c5aa1024bd7e931dde6734531df16a122a2121fcbf1e1e8a`).
  It was produced before `VALUE_ARMS` filtering and is historical only.

The pre-fix `native04_acceptance_v2.json` path is deliberately retired rather
than overwritten. The manifest records original paths, supersession reasons and
hashes for both historical receipts.

## Contract

- Raw top-one is diagnostic only.
- Accepted-correct uses eligible **fact** queries and requires every layer to be
  accepted and correct. Acceptance coverage uses any accepted layer.
- Accepted precision is accepted-correct divided by accepted retrievals.
- Zero vectors and ambiguous tied readouts receive no usable-retrieval credit.
- Abstentions, wrong accepted retrievals and unknown false recall remain in the
  denominator. Layer readouts are not independent for uncertainty; worlds are
  the uncertainty units.
- Each uncertainty metric bootstraps only the worlds where that metric is
  estimable. A world with no fact queries has no accepted-correct rate; a world
  with no accepted retrievals has no accepted precision. Those null values are
  not inherited from another metric. Seed, resample count and confidence are
  unchanged (`17`, `2000`, `0.95` in the declared config).
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

## Version and receipt selection

Consumers must validate the explicit receipt/config version; a `metric_version`
string alone is not sufficient. `replica_lab.memory_acceptance` exposes:

- `validate_config(config)` — requires the v2 config format and version plus the
  candidate, oracle, gates and bootstrap fields used by rescoring.
- `validate_receipt(receipt, config=..., superseded=..., require_current=...)`
  — requires the v2 receipt format and version, the explicit receipt keys,
  source hash shape, metrics, verdict and claims. It rejects v1, unsupported
  versions, malformed receipts, config/receipt candidate mismatches and, when
  `require_current=True`, any digest listed as superseded.
- `load_receipt(path, manifest_path=..., require_current=...)` — loads and
  validates one receipt. Current selection requires a manifest.
- `load_current_receipt(manifest_path, config=...)` — selects the manifest's
  sole `current` receipt, verifies its file SHA-256 and canonical receipt
  digest, and rejects superseded history.

`receipt_digest` is the canonical JSON digest used for supersession. The rescore
script validates its config and generated output, and can verify the current
receipt without rescoring:

```bash
python scripts/rescore_memory_acceptance.py \
  --check-current reports/memory_acceptance_20261026/receipt_manifest.json
```

Historical receipts remain readable without `require_current`, so valid v2
receipts stay backward-compatible while current authority is explicit.

## Gates and disposition

Predeclared gates in `configs/memory_acceptance_v2.json` require accepted-correct
rate ≥0.85, accepted precision ≥0.95, correction ≥0.60, unaffected ≥0.70,
unknown false recall ≤0.10 and oracle raw top-one ≥0.85. The oracle raw bound
passes at 85.11%; every candidate learned-quality gate fails. Acceptance coverage
is report-only and cannot be used to disguise a low accepted-correct rate.

Focused tests: **12 passed**, covering zero vectors, empty memory, ties,
rejected confidence, correct/wrong acceptance, unknowns, layer/query
aggregation, missing coverage, per-metric uncertainty resets, receipt
version/format/supersession validation and config version rejection. Replay was
CPU-only; no native training occurred.

Scope of this repair: the uncertainty reset, receipt/config validation API and
explicit supersession manifest. Gate thresholds, raw rankings, `VALUE_ARMS`,
metric definitions and the rejected disposition are unchanged. A corrected
receipt was generated by CPU rescore into a fresh versioned path; both historical
receipts and the immutable rankings remain intact.

The current receipt reproduces every count/rate metric, gate and rejected verdict
of `native04_acceptance_v2_final.json` exactly. Its uncertainty intervals now use
only estimable worlds: causal accepted precision uses 21 clusters rather than
inheriting values from another metric. The prior final receipt remains preserved
as the historical disposition record, not as current consumer input.
