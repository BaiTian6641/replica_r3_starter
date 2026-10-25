# Verification repairs: authorization, parity, vision verdicts, and clean exports

**25 October 2026.** Four concrete verification defects are repaired. Mechanical
authorization and reload checks are now stricter; useful learned memory and
generalizable vision remain unproven.

## 1. Raw-page memory authorization

`PersistentFastMemory` now persists ordered page-to-record dependencies with
record ID, record hash, source hash and supersession metadata. Page text is never
scanned to guess provenance. Rebinding proves every dependency against the
authoritative ledger/view and rejects revoked, missing, changed or unconsumed-
superseded dependencies.

Omitted `records` means **unknown provenance**, not an empty dependency list:
the event may be inspected but cannot be source-authorized or saved. Only an
explicit `records=()` declares a deliberately opaque dependency-free observation.
Legacy raw snapshots without dependency metadata fail closed; record-backed
snapshots remain compatible.

The original sequence—append page without records, revoke a contained record,
attempt rebind/save—now fails before and after the unchanged page hash. Explicit
page-derived memory also rejects contained-record revocation before and after
reload. A verified current-view rebuild restores authorization.

Evidence: `f3de0e0`; focused memory suites **89 passed**.

## 2. Complete per-fact reload parity

`score_one_write` pairs rows by `(world_id, fact_id, layer)` for every read path.
Duplicate rows and missing before/after partners are rejected instead of being
overwritten by later facts.

A new regression zeroes an earlier fact's pre-save readout. Before the repair it
returned `one_write_storage_ok`; it now returns `fail_save_reload_parity`.
Multi-fact tests also reject duplicate and missing rows.

Immutable replay results:

| Artifact | Independent result | Parity rows per read path | Outcome |
|---|---|---:|---|
| One fact | `matched` | 16 | `one_write_storage_ok` |
| Two facts | `matched_with_full_fact_coverage` | 8 historical → 16 complete | `one_write_storage_ok` |
| Three facts | `matched_with_full_fact_coverage` | 8 historical → 24 complete | `one_write_storage_ok` |

All measured reload differences remain exactly zero. The repair changes coverage
counts, not historical outcomes. This remains mechanical observed-key evidence,
not natural-language addressing or useful answer generation.

Evidence: `f5a8325`; one-write tests **10 passed**. New files:
`native01.independent_score_v3.json`, `native02b.independent_score_v3.json`, and
`native03b.independent_score_v3.json` under `reports/`.

## 3. Conservative frozen-feature interpretation

The diagnostic now requires its declared incremental gain over both the best
single stream and the shuffled-image control before returning
`frozen_features_support_readout`. Otherwise it returns
`inconclusive_visual_contribution` and calls for a balanced paired diagnostic.
Even a supported result identifies the connector/training path only as a
candidate bottleneck, not a proven one.

The immutable 96-case artifact replays from `frozen_features_support_readout` to
**`inconclusive_visual_contribution`**. Raw measurements are unchanged:
question-only 43.75%, combined 41.67%, shuffled combined 37.50%, image-only
18.75%. Combined features underperform question-only, so the previous projector-
bottleneck conclusion was unsupported.

Evidence: `d323df8`; vision diagnostic tests **5 passed** and
`reports/vision_diagnostic_20261025/conservative_replay.json`.

## 4. Portable bounded-RSI tests

Bounded-RSI unit tests now construct deterministic synthetic perception and
rejected-memory evidence. They no longer read the ignored V0 report. One
explicitly marked integration test retains the historical native-evidence path
and skips with provisioning instructions when it is absent.

From `git archive HEAD` with no ignored native artifacts, the complete
verification-repair test set passes **147 tests with two explicitly reasoned
integration skips in 35.86 s**. Bounded RSI alone passes 11 tests with one skip;
it passes 12/12 in the provisioned environment. This is a focused clean-export
result, not a claim that the entire suite is artifact-free.

Evidence: `d86dc3b`, `79b5f54`, and
`reports/clean_export_verification_20261025.json`.

## Disposition and next boundary

The repairs close the reported verification defects; they do not change any
learned-capability verdict. The next native package is the fresh paired,
balanced image-dependence diagnostic with spatial and question-conditioned
readouts and a new final holdout. Matched-episode memory localization follows.
External corpus admission and Spark interface preparation proceed independently;
native FP8 remains unqualified.
