# Development-only memory acceptance and address calibration

**26 October 2026 · CPU replay complete; acceptance calibration is rejected as
the sole boundary.** The protocol was committed at `5e98ce2`; implementation,
full-oracle gate handling and the derived receipt are committed at `92a8c12`.
No model, tokenizer, native job, writer update, relabeling or new generation was
used. Prepared04 heldout rows were not scored or used for confirmation.

## Declaration and evidence scope

`configs/memory_acceptance_calibration_v1.json` pins the immutable restart02
rankings at SHA-256
`e8c8a4a0338ebb1fcc1ec689372f1792350dcc7a8120714145d31c5e835f7952`
and prepared03 inventory at
`5994de0da6e7f863f4a697a40a25041e760d0d9e8e5c77f565713e1bdf763a93`.
The derived receipt is
`reports/memory_acceptance_20261026/calibration_dev01.json` (content SHA-256
`0d7df9f40ac76927d8aa695308a5cec92fe31cd909e5c238fc0cd4c72d2d9f46`).

The source contains 17,520 rows. The declared train/dev scope contains 11,680
rows across all five variants, four arms, 13/27 layers and all three phases;
5,840 heldout rows are present for inventory/identity validation but have
`heldout_rows_scored=0` and no reported metrics. The native 0.5 cosine / 0.05
margin rule reproduces the artifact acceptance flag on all 11,680 scored rows
with zero mismatches. This is an implementation audit, not a quality pass.

Threshold decisions use only runtime readout norm, top/second cosine and policy
metadata. Gold labels are used only after that decision for scoring. Zero and
tied readouts receive no credit, policy-blocked rows remain blocked, and
wrong-address categories are preserved. Value-frontier replay changes only
existing readout acceptance. Address-frontier replay is restricted to
`query_key_address`; it does not reselect a candidate, regenerate a refresh or
recompute downstream state.

## Native-point development metrics

These are train/dev metrics, not the earlier all-phase v4 denominators.

| Metric | Candidate `query_key_value` | Oracle target-refresh-last |
|---|---:|---:|
| Answerable-fact raw recoverability | 101/256 = 39.45% | 256/256 = 100% |
| Accepted-correct fact groups | 14/256 = 5.47% | 102/256 = 39.84% |
| Accepted precision | 12.61% | 62.58% |
| Acceptance coverage | 36.51% | 53.62% |
| Correction accepted-correct | 8/32 = 25.00% | 28/32 = 87.50% |
| Unaffected accepted-correct | 1/80 = 1.25% | 18/80 = 22.50% |
| Unknown false recall | 10/32 = 31.25% | 10/32 = 31.25% |
| Revoked false recall | 3/16 = 18.75% | 3/16 = 18.75% |

The candidate's accepted-wrong query groups include 34 `other_source`, 27
`independent_world`, 10 `unknown_address` and 3
`revoked_target_other_live`. Disabled memory accepts zero retrievals. Full
history and shuffled values remain separate controls; exact-key is a mixture
diagnostic, never blended into candidate quality or treated as an upper bound.

The oracle's raw answerable-fact gate passes, but its full acceptance gates do
not: accepted-correct, accepted precision, unaffected retention and unknown
abstention all fail, while correction and revoked false recall pass. Therefore
`oracle_meets_gates=false`; raw recoverability is not substituted for the full
oracle acceptance contract.

## Address-arm separation

At the native address threshold, the candidate `query_key_address` arm has:

- Raw address hits on **256/256** answerable-fact query groups.
- **58/304** accepted query groups, with only **4/256** strict all-layer
  accepted-and-correct fact groups (1.5625%) and 6.90% accepted precision.
- **62 accepted readout layers, all 62 address hits** (100% accepted-layer
  precision). Fifty-four accepted query groups fail the strict all-layer
  accepted-and-correct combination; this is an abstention/aggregation effect,
  not evidence that every accepted address layer is wrong.
- Zero accepted unknown or revoked query groups at the native point. Policy
  blocking, wrong-address categories and zero/tie rows remain separately
  reported.

This is why per-arm and per-layer views matter. The value arm has 111 accepted
query groups but only 14 accepted-correct; the address arm's raw recovery alone
cannot establish a useful value readout or a deployable selector.

## Declared frontier result

The value frontier contains 25 cosine/margin points over `query_key_value`,
`exact_key_value` and `context_removed`. The address frontier contains 25
points over `query_key_address` only.

| Diagnostic | Result |
|---|---:|
| Value points passing all candidate value gates | 0/25 |
| Value points passing all full oracle acceptance gates | 0/25 |
| Value points passing every gate | 0/25 |
| Best candidate accepted-correct rate | 18.75% at cosine 0.0 / margin 0.0 |
| Best candidate accepted precision | 16.00% at cosine 0.0 / margin 0.0 |
| Best candidate acceptance coverage | 98.68% at cosine 0.0 / margin 0.0 |
| Address points meeting descriptive 85% hit / 90% precision targets | 0/25 |
| Best address hit rate | 97.66% at cosine 0.0 / margin 0.0 |
| Best address precision among hit ≥85% | 87.50% at cosine 0.9 / margin 0.01, with 92.97% hit |

The address operating targets are descriptive diagnostics using the declared
quality thresholds, not a new acceptance gate. No value or address point
justifies a policy change. The best address tradeoff still misses the 90%
precision target, and full oracle acceptance fails across the value frontier.

The final disposition is **`oracle_cannot_meet_gates`**: acceptance calibration
is rejected as the sole boundary. Selection/refresh cannot be promoted from
this replay, and no writer training or native confirmation is permitted. A
candidate development pass would have required a new sealed fixture; it did not
occur.

## Verification and artifacts

The focused calibration suite passes **9 tests**, including runtime-only
threshold decisions, gold-label invariance of acceptance, zero/tie and policy
blocking, arm separation, exact-key isolation, frontier shape, tamper/scope
failures and the real train/dev replay. The H1/v4 suite passes **43 tests**;
H1's 26-check replay remains unchanged. The calibration CLI is CPU-only and
writes a deterministic 1,563,945-byte derived receipt. Raw rankings, inventory,
labels and prior v3/v4 receipts are unchanged.

**Implemented:** per-arm/layer/cohort metrics, runtime-only acceptance replay,
25-point value and address frontiers, source/scope audits, full-oracle gate
classification and joint address operating diagnostics. **Measured:** the
train/dev results above. **Rejected:** acceptance calibration as the sole
boundary and any threshold-based promotion. **Unrun:** native reselection,
refresh regeneration, writer training, generated-answer evaluation and any
fresh held-out confirmation. H1's literal-copy result remains separate and is
not memory retrieval evidence.
