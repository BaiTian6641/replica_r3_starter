# Memory acceptance v4 and current quality boundary

**26 October 2026 · v4 replay is implemented and verified; candidate quality is
rejected. H1's separate literal readout pass is not evidence of memory
retrieval.** The v4 protocol is `configs/memory_acceptance_v4.json`; v3 and v2
receipts remain historical and are not accepted as v4.

## Current receipts

The authoritative per-run receipts are:

- `reports/memory_acceptance_20261026/causal_fresh_native02_acceptance_v4.json`
  and `receipt_manifest_v4.native02.json` for the first-process phase scope.
- `reports/memory_acceptance_20261026/causal_fresh_restart02_acceptance_v4.json`
  and `receipt_manifest_v4.restart02.json` for all first-process, restart and
  revocation phases.

Each manifest selects its own run's current v4 receipt; there is no global v4
current-manifest claim. The restart receipt validates all 17,520 immutable
ranking rows against the prepared03 inventory pin
`5994de0da6e7f863f4a697a40a25041e760d0d9e8e5c77f565713e1bdf763a93`.
The rankings SHA-256 is
`e8c8a4a0338ebb1fcc1ec689372f1792350dcc7a8120714145d31c5e835f7952`.

v4 makes the oracle population explicit: raw valid top-1 on answerable `fact`
cases divided by `fact_n`. Unknown and revoked cases have separate abstention
and false-recall rates and do not enter that denominator. Total-group raw top-1
is descriptive only. Zero or tied readouts receive no recoverability or
acceptance credit, and layers are not independent uncertainty samples.

## Quality result

| Population | Candidate `query_key_value` | Oracle target-refresh-last |
|---|---:|---:|
| Raw valid top-1, all groups | 176/456 | 384/456 |
| Answerable-fact raw recoverability | 176/384 = 45.83% | 384/384 = 100% |
| Accepted-correct fact groups | 21/384 = 5.47% | 132/384 = 34.38% |
| Accepted precision | 15.22% | 60.83% |
| Acceptance coverage | 30.26% | 47.59% |
| Correction accepted-correct | 8/48 = 16.67% | 40/48 = 83.33% |
| Unaffected accepted-correct | 5/120 = 4.17% | 29/120 = 24.17% |
| Unknown false recall | 10/48 = 20.83% | 10/48 = 20.83% |
| Revoked false recall | 3/24 = 12.50% | 3/24 = 12.50% |

The candidate verdict is **rejected**. The answerable-fact oracle gate passes,
but even gold target identity does not make the acceptance/abstention rule
useful: oracle accepted-correct is only 34.38% and unaffected retention is
24.17%. Thus retrieval recovery and acceptance calibration are separate
boundaries. The oracle is a diagnostic population, never a legal policy or an
upper-bound claim.

Candidate accepted-wrong rows retain their causes: 34 `other_source`, 27
`independent_world`, 10 `unknown_address` and 3
`revoked_target_other_live`. Disabled memory accepts zero retrievals. Shuffled
values and full-history controls remain separate; `exact_key_value` is a
mixture diagnostic and is never blended into candidate quality.

The first-process native02 receipt has 8,400 rows and candidate accepted-correct
12/192 (6.25%) at 18.18% precision. Its answerable-fact oracle recoverability is
192/192 (100%). These phase-scoped results are consistent with the restart
boundary, not independent new evidence.

## Implemented controls and limits

Durable observation identity, query-local accepted refresh, wrong-address and
unknown/revoked controls, source/stale/revocation rejection, disabled and
shuffled-value arms, immutable rankings, fresh-process checkpoint parity and
zero old-observation/query re-encodes are already measured. They establish
mechanical behavior, not useful learned memory. Candidate generated-answer
accuracy is `not_measured` by v4; H1's separate sealed-MiniCPM literal-copy
result must not be substituted for it.

The v4 implementation and real-receipt replay tests pass. The receipt validator
rejects v1/v2/v3 inputs, missing/duplicate/extra rows, incorrect oracle
populations, invalid phase scopes and superseded current-receipt claims. No
native training or writer update was performed for this scoring work.

## Next diagnostic

`query_key_address` and `context_removed` rows exist in the immutable rankings
but are not included in the v4 primary-quality block. The next package is a
CPU-only, development-only acceptance/abstention and per-arm diagnostic over
those existing rows. It will use runtime scores only, keep wrong-address and
unknown/revoked controls separate, and will not regenerate a refresh state for
an offline address-threshold change. Any value-threshold frontier is a replay
of existing readouts, not a new native policy. Prepared04 heldout is already
spent and will not be reused for calibration or confirmation.

The stop rule is explicit: if no declared value-acceptance rule lets the oracle
meet the existing v4 quality gates on development rows, calibration is rejected
as the boundary; if the oracle passes while the candidate fails, selection or
refresh remains the boundary. A candidate development pass would require a new
sealed fixture before any native confirmation. No writer training, native rerun
or held-out retuning is authorized by this receipt.
