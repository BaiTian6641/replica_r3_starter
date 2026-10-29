# Spark scoped readout, recovery and control evidence

**29 October 2026 · repaired CPU engineering complete for this increment; native execution and useful learning remain open.**

## Scope and lineage

This increment connects the separately versioned tiered store to the existing Spark persistent-learner query path. It does not rewrite the sealed v1 protocol, sealed source, protected runtime pilot or earlier negative results.

| Binding | Value |
|---|---|
| Implementation commit | `11210d99f7391423a22d0a0dae7be78fc87e66c9` |
| Protocol | `spark_persistent_learner_v2` |
| Protocol SHA-256 | `49cea2ad9d24bd504f88a86595043abc016d4b5fadc5fb60a94d8bb1cf93c4bd` |
| Executable dependency binding | 38 files, including `ports.py`, `model_identity.py`, `training_state.py` and `preflight_spark_text_memory_v1.py` |
| Source identity | Official `XHToken/Spark-X2.6-4B-FP8`, revision `0318d6f6ab070c7b5b43052db571489b2508006c` |
| Native execution | Not run; exact FP8 backend evidence and external authorization are absent |
| Integrated output | `runs/spark_persistent_learner_v2/integrated06` |
| Evaluation boundary | Four fit events and twelve development cases per arm; sealed final data unmaterialized and unopened |

The integrated report has the same protocol hash and `status=complete`. The final no-launch review is [`review_final02.json`](../../reports/spark_persistent_learner_v2/review_final02.json): `model_loaded=false`, `tokenizer_loaded=false`, `labels_read=false`, `downloads=0`, `launch_allowed=false`, with exactly the two native bindings missing.

## Implemented and exercised

- Scope, time and verification filtering runs before semantic key reads; failed-check and quarantined records are excluded. Cold tickets bind the complete scope including `as_of` and revalidate time, verification, generation and payload identity after I/O.
- Ordinary arms use two-record semantic selection. The untouched baseline uses explicit `all_eligible` with at most 16 records/64 value slots and refuses truncation or incomplete authority. It is not a top-two semantic approximation.
- Restoration accepts only current-record provenance with matching record ID and canonical digest. Source epoch, validity, lineage, producer/representation/codec versions and correction links must match the neural producer and active mirror. Missing or conflicting authority is refused.
- Corrections preserve validated current provenance and an explicit predecessor. They never inherit the predecessor's producer authority. An explicit reconciliation path handles an inactive neural parent without reviving it or dropping `supersedes`; shuffled controls preserve the same links.
- Multiple cold tickets complete before one coherent final selection. Work from every attempt is aggregated, and any earlier decision boundary blocks generation. Empty or incomplete reads are refused unless an explicit empty control applies.
- Exact readout accounting includes candidate retrieval keys, admitted text/values and payload reads later rejected by validation. The in-process cache is not physical residency and does not eliminate internal selection reads.

## Verification and measurements

The integrated command and complete counter table are in the [v2 runbook](../../reports/spark_persistent_learner_v2/README.md). Raw trajectories, failures and artifact receipts remain under the output directory.

Supporting snapshots: [`control_baseline.json`](../../reports/spark_persistent_learner_v2/control_baseline.json) records the pre-fix trained-reader leak; [`control_views_after.json`](../../reports/spark_persistent_learner_v2/control_views_after.json) records all eleven isolated arm states and matched-reader comparisons; [`recovery_baseline.json`](../../reports/spark_persistent_learner_v2/recovery_baseline.json) preserves the recovery reproduction snapshot.

| Check | Observed result | Interpretation |
|---|---|---|
| Focused protocol/readout/control/recovery tests | 62 passed | Software contracts and adversarial regressions |
| Full repository suite | 3,395 passed; 9 skipped; 21 subtests passed | 862.73 s; one unrelated sparse-tensor warning in [`full_suite_integrated06.log`](../../reports/spark_persistent_learner_v2/full_suite_integrated06.log) |
| `integrated06` | Complete; 11 arms × 12 cases = 132 | Protocol-matched engineering acceptance |
| Full-context baseline | All 12 traces contain all 11 records; `status=ok`, `truncated=false` | Actual all-eligible input, not top-two selection |
| Generated answers | 0/12 in every arm | Learned capability inconclusive; no useful-readout claim |
| Independent verification | 147 calls | Host/scorer checks, not model-written success claims |
| Logical retrieval / actual selection attempts | 148 / 109 | Separate request and physical selection accounting |
| Revalidations / failures | 317 / 0 | Live post-acquisition checks |
| Readout cache | 301 hits; 16 misses | In-process copy behavior, not physical residency |
| Exact readout I/O | 2,668 file reads; 1,876,983 bytes; 432,027,585 ns | Includes candidate keys and rejected reads |
| Integrated cold path | 0 cold tickets/resumes | Fixture was warm; cold evidence is the separate probe |
| Pressure probe | 2 hits; 1 eviction; cold wait; 5 rejected cold reads/4,114 bytes; corrected cold read with current provenance | Real cold/I/O and correction evidence in `readout_pressure04.json` |
| Fresh-process continuation | 1 restart; 1 reload; exact token parity and tiered-P reload passed | Persistence/engineering evidence |
| Artifact storage | 96,058,182 peak live bytes; 139 final files/81,614,771 bytes | Includes temporary overlap; no base weights duplicated |

The pressure probe's warm selections are labeled `degraded/selection_truncated` because the two-record budget truncates four candidates; selected records themselves are `ok`. Its corrected cold record is `cold-rule-corrected`, with 19 payload bytes and one cold file read. Logical hot/warm/cold labels do not establish GPU or OS residency.

## Failure analysis and preserved negatives

1. `integrated01` refused the old 888-read cap. That cap predated candidate-key and rejected-read accounting and is not reused. The final conservative ceilings are 296 selection attempts, 66,304 total file reads, 23,680 payload/cold reads, 2,368 cold tickets/resumes and 7,104 revalidations. They include 64 candidate key reads per selection, up to 16 full-context records and at most two selections per logical query; they are not measurements.
2. `integrated02` used a two-record baseline and a pre-binding protocol hash. It is preserved but is not accepted as full-context or final-lineage evidence. `integrated03` exposed a shuffled-control mirror mismatch: payload rotation dropped `supersedes`. The control now preserves correction authority and reconciles an absent parent explicitly. `integrated04` and `integrated05` completed before the final dependency-list correction and remain prior protocol generations.
3. The first pressure probe used a 1 KiB cache. Records did not fit, so zero hits and evictions were correct but not useful pressure evidence. `readout_pressure02` and `readout_pressure03` preserve earlier accounting/provenance iterations; only `readout_pressure04` is final.
4. Native official-FP8 execution remains a precise backend blocker. The recorded MPS path lacks qualified FP8 allocation/compute and exact forward/backward/single-update evidence. Silent decompression, official BF16 and FP8-derived BF16 are different execution identities and are not substituted.

## Claim/evidence boundary

| Claim | Status | Evidence or remaining gate |
|---|---|---|
| Scoped readout wiring and current-record restoration | Implemented; CPU exercised | `integrated06`, focused tests |
| Full-eligible baseline construction | Implemented; all 12 baseline traces complete | 11/11 records per case, no truncation |
| Recovery, correction invalidation and restart persistence | Implemented; targeted and integrated checks pass | `tests/test_tiered_recovery.py`, `integrated06/restart`, pressure receipt |
| Cache pressure and cold recall cost | Measured on a small real-file fixture | `readout_pressure04.json`; not a large-bank or GPU-residency measurement |
| Full-context competence and useful P transfer | Inconclusive | All integrated arms 0/12; native baseline unmeasured |
| Calibrated task-success prediction | Not trained or calibrated by this increment | No new outcome-head labels or calibration claim |
| Native FP8 training and native cost | Not run | Exact backend receipt and external authorization required |
| P4/P16/P32/P64 capacity, learned writer, multi-clock learning and RSI | Not promoted or qualified | Useful readout, retention, correction and matched-cost gates remain closed |

## Next discriminating action

Keep the official-FP8 launch closed. If the backend is qualified, use the bound P1 packet at sites 3/7 and 23/31 with the same eleven arms, charged teaching/encoding/retrieval/replay/generation/verification costs, fresh-process continuation and independent outcome checks. A derived-BF16 arm requires a new versioned protocol and explicit authorization; it cannot satisfy the official-FP8 gate. Do not expand banks or promote the D-only writer from these engineering results.
