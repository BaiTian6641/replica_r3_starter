# Spark text-memory milestone — 28 October 2026

## Scope and identity

Starting reviewed HEAD: `b350e87e`. Implementation commit: `9b1a02d` on
`codex/d2-capacity-runtime-pilot`. The protocol is
[`configs/spark_text_memory_protocol_v1.json`](../../configs/spark_text_memory_protocol_v1.json),
bound to that implementation commit and the exact implementation-file hashes.
The additive final-product bundle is reviewed separately; historical protocols
and the 117-input manifest are unchanged. The protected runtime pilot was not
read, imported, modified or staged.

**Native status:** preparation only. No MPS job, model download, native optimizer
update, native checkpoint write, BF16/dequantized substitution or learned
capability claim occurred. Official FP8 is the active source baseline. Native FP8
requires a separate forward/backward/single-update backend receipt; it is absent,
and the active adapter publishes no native-FP8 admission contract.

## Implemented

- Reconciled the active roadmap to Spark-first text learning with the internal
  spine. MiniCPM and vision/DC milestones remain historical controls, not
  prerequisites. Nested outer learning, self-distillation, RSI, personality,
  capacity expansion and multimodal training remain deferred.
- Reconciled readout, durable model identity, publication recovery and retired
  runner behavior. Active MiniCPM-only launches are refused; explicit historical
  replay remains authorized and guarded.
- Added query-driven P reads that use current question/native hidden state after
  E removal, query-dependent bounded selection with full selected/omitted
  provenance, separate storage/read budgets, ownership-safe packets, strict
  representation/codec checks and complete context invalidation.
- Added versioned text production, canonical/durable source references,
  procedures with applicability/prerequisites/exceptions/verified outcomes,
  task workspaces with observed completion, and durable text/procedure/task
  checkpoints. Generated explanations remain candidates.
- Added true combined E+P reads: E causal availability and P query slots remain
  separate, one memory port returns their summed delta, and E/P work is charged
  separately. No extra decoder forward is introduced by composition.
- Added response and P learning through the frozen Spark decoder and retained P
  expert. Each fit event performs one response update and one P update with
  NativeFusedAdamW. P targets are derived from verified answer representations.
  Actual changed tensors, P clock/revision, retries and no-op updates are
  distinguished from train fit, transfer, retention and useful learning.
- Added a versioned fit/development/restart runner, fail-closed preflight and
  capacity-watchdog registration. It preserves raw trajectories, failed cases,
  extension/dynamic-state checkpoints and sealed-source references.

## Verification

The complete affected suite passed **177 tests**:
spine/text/Spark compatibility, protocol/preflight, native-runner orchestration,
retirement, readout, publication, recovery and capacity guards. The additive
final-product verifier passed **16 tests** with 101 inputs. A final CPU run also
completed its default CLI path and fresh-process restart.

### Preserved CPU behavioral result

Raw evidence is under `reports/spark_text_memory_v1/cpu_fixture_attempt01/` and
`reports/spark_text_memory_v1/cpu_fixture_attempt01_report.json`.

| Behavior or work | Observed |
|---|---:|
| Fit events / response updates / P updates | 4 / 4 / 4 |
| Development cases | 108 = 12 per each of 9 arms |
| Parity + restart-parity requests | 2 + 1 |
| Fresh-process restart | Completed; exact-token parity matched |
| Source evidence removed before restart read | Yes; derived caches absent |
| P clock events / P revision / changed P matrices | 4 / 4 / `gate`, `up`, `down` |
| Protocol mismatches | 0 |
| New-instance accuracy, every arm | **0/12** |
| Correction / older-rule retention | **0.0 / 0.0** |
| Useful-learning verdict | **Failed** |
| Train-fit generated answers | 0/4; teacher-forced token accuracy 0/1 each |
| Artifact bytes / retained checkpoints | 49,326,224 / 4 |
| Reconciled work violations | 0 |

The response and P parameter changes are real, but they did not produce useful
new-instance behavior. Full eligible context also scored 0/12, so the failure is
not attributable to P alone. This is a negative behavioral pilot, not a learned
memory, reasoning or speedup result.

### Actual CPU work accounting

| Counter | Actual |
|---|---:|
| Training forwards / P forward calls / P backward calls | 4 / 4 / 4 |
| Backward calls / backward processed positions | 8 / 2,048 |
| P feature vectors / P backward feature vectors | 2,048 / 2,048 |
| Optimizer steps | 8 |
| Query-sensing forwards | 119 |
| Forward calls | 571 |
| Generation requests / autoregressive forwards / generated tokens | 111 / 444 / 444 |
| Retrieval calls / verification calls | 115 / 112 |
| Checkpoint writes / retained artifacts / reloads | 5 / 4 / 1 |
| Model / tokenizer loads | 2 / 2 |
| Cache hits / misses | 0 / 115 |
| Teacher / replay / consolidation / native retries | 0 / 0 / 0 / 0 |
| Processed text positions | 290,970 |
| Wall time | 9.06 seconds |

Cold-start, steady-state and amortized costs are separately recorded in the raw
report. The native schedule is a ceiling, not a measurement: 8 optimizer steps,
3,560 query-sensing forwards, 7,120 total forwards, 14,683,680 processed text
positions, 8,192 decoder backward positions and 8,192 P feature vectors.

## Interpretation correction and next experiment — later review, 28 October

The raw all-arm 0/12 result and rejection are preserved. The original
interpretation as a frozen Spark full-context failure was too broad: this run
uses a random tiny model whose blocks are pointwise and contain no attention.
An independent prefix perturbation confirms identical final-position logits
for different prefixes ending in the same token. This fixture supports
engineering and restart checks; it cannot diagnose Spark's native reasoning
or memory-learning capacity.

The next native experiment must establish full-context competence and compare
exact context against contextual memory reads on the real, precisely identified
source/backend. Audit masks, alignment, output grammar and actual optimizer
ownership, including that response CE currently does not update P. The
[revised design](../architecture/28_Spark_Persistent_Learner_and_Tiered_Memory.md)
allows bounded tiered-storage engineering and capacity profiles now while
requiring useful readout before trained expansion.

Native execution remains blocked independently by missing external authorization
and missing native-FP8 backend evidence. A future run must bind the exact current
implementation, source identity, execution format/backend, fresh output and
capacity guards. No native result, resource measurement, learned retention,
reasoning improvement, break-even speedup or RSI is established by this package.
