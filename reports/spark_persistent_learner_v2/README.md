# Spark persistent learner v2 — implementation and evidence

**Status: integrated CPU engineering is complete; native execution is not admitted and useful learning is inconclusive.** This is a separately versioned runtime, not a rewrite of sealed v1 evidence.

## 0. Current increment — scoped readout, recovery and controls

Implementation commit: `83c7214bcf12877e4c7b9d0c367825905a16cf55`. Protocol
SHA-256: `91c45fbf75fed46ad383a1c385b77775b1f69e6f16f8aca0079e1eefe6183ec5`.
The launch pointer and hash map bind 38 executable dependency files, including
`ports.py`, `model_identity.py`, `training_state.py` and the script preflight.
This increment is CPU engineering evidence only; no native model was loaded.

### Reproduce the integrated demonstration

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:scripts PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv/bin/python scripts/run_spark_persistent_learner_v2.py \
  --stage all --backend cpu_contextual_fixture --device cpu \
  --fixture-context-tokens 256 --fixture-max-new-tokens 2 \
  --output runs/spark_persistent_learner_v2/integrated09 \
  --config configs/spark_persistent_learner_protocol_v2.json \
  --prepared data/spark_persistent_learner_v2
```

Raw evidence: [`run_report.json`](../../runs/spark_persistent_learner_v2/integrated09/run_report.json),
[`fit/trajectories.jsonl`](../../runs/spark_persistent_learner_v2/integrated09/fit/trajectories.jsonl),
[`development/trajectories.jsonl`](../../runs/spark_persistent_learner_v2/integrated09/development/trajectories.jsonl),
[`development/parity.jsonl`](../../runs/spark_persistent_learner_v2/integrated09/development/parity.jsonl),
[`restart/restart_report.json`](../../runs/spark_persistent_learner_v2/integrated09/restart/restart_report.json),
[`storage_artifacts.json`](../../runs/spark_persistent_learner_v2/integrated09/storage_artifacts.json).
Two additional fresh-process continuation cycles are preserved as
[`integrated08/restart/cycle01_report.json`](../../runs/spark_persistent_learner_v2/integrated08/restart/cycle01_report.json)
and [`cycle02_report.json`](../../runs/spark_persistent_learner_v2/integrated08/restart/cycle02_report.json).
The integrated report has the same protocol hash shown above and `status=complete`.
Engineering acceptance passes: initial null parity, later nonzero influence,
source-free P-only audit, correction reset, fresh-process restart and tiered-P
reload. Every arm scored 0/12; learned benefit remains inconclusive.

The untouched baseline uses `all_eligible`, not semantic top-two selection. All
12 baseline traces contain all 11 eligible records with `status=ok` and
`truncated=false`. Semantic arms retain two-record selection. The baseline
refuses any incomplete or truncated full-context read rather than silently
calling it full context. Failed-check and quarantined records are filtered
before retrieval-key reads.

| Measurement | Result |
|---|---:|
| Fit events / development cases / arms | 4 / 132 / 11 |
| Direct-CE objective forwards / backward calls | 8 / 8 |
| Selected-P optimizer steps / port-output steps | 4 / 4 |
| Correction quarantine + authenticated reset | 1; exact unlearning not claimed |
| Development parity requests / fresh restarts / reloads | 2 / 1 / 1 |
| Independent verification calls | 147 |
| Total forward calls / query-sensing calls | 434 / 148 |
| Autoregressive forwards / generated tokens | 270 / 270 |
| Logical retrieval calls / actual selection attempts | 148 / 109 |
| Revalidations / revalidation failures | 317 / 0 |
| Readout cache hits / misses | 301 / 16 |
| Exact readout file reads / bytes / wall time | 1,163 / 621,251 / 265,315,668 ns |
| Payload read calls / payload file reads / payload bytes | 16 / 80 / 66,755 |
| Cold tickets in this warm fixture | 0 |
| Wall time | 136.946262 s |
| Peak live artifact bytes (including temporary overlap) | 93,306,970 |
| Final artifact receipts | 139 files, 81,630,342 bytes |
| Tiered E records / readable value slots | 13 / 52 |
| Tiered clusters / verified procedures / resumable workspaces | 5 / 4 / 1 |

Exact readout counters include candidate retrieval keys, admitted text/values,
direct/cold payload reads and reads later rejected by validation. Logical tier
labels are not physical residency claims, and the in-process cache does not
eliminate internal store-selection reads. The integrated fixture is warm; cold
evidence comes from the independent probe below. Artifact receipts use magic
bytes, include extensionless states and do not include base-model weights.

### Pressure, cold recall and correction during I/O

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python scripts/measure_tiered_readout_pressure.py \
  --output reports/spark_persistent_learner_v2/readout_pressure05.json
```

[`readout_pressure05.json`](readout_pressure05.json) uses real store files and
no artificial delay. It observes two warm-cache hits, one eviction, one cold
wait ticket, one invalidated resume after correction and a successful corrected
cold read with current provenance (`cold-rule-corrected`, 19 payload bytes, one
cold file read). The invalidated cold resume reports five actual payload file
reads and 4,114 bytes; earlier receipts omitted those file counts. Warm
selection remains `degraded/selection_truncated` because the two-record budget
truncates four candidates; selected records themselves are `ok`.

Focused protocol/readout/control/recovery tests: **62 passed**; derivative
contracts pass **12** and Sparkx lineage/bundle tests pass **57**. The final
full suite passes **3,424 tests with 9 skips and 21 subtests** in 869.59 s,
with one unrelated sparse-tensor warning; see
[`full_suite_integrated09.log`](full_suite_integrated09.log). The earlier
completed run's eight failures and the first 360 s timeout are preserved in
[`full_suite_integrated09_attempt01.log`](full_suite_integrated09_attempt01.log)
and `execution_state.md`. Native preflight [`review_final03.json`](review_final03.json)
remains preparation-only with no model load and exactly the two official-FP8
bindings missing: exact backend evidence and external authorization. The
separate derived-BF16 packet is also preparation-only and has only its external
derivative authorization binding missing.

### Preserved failures and protocol generations

- `integrated01` refused the old 888-read cap. That cap was derived before
  candidate-key and rejected-read accounting existed; it is not reused.
- `integrated02` used a two-record baseline and a pre-binding protocol hash. It
  is preserved but is not accepted as full-context or final-lineage evidence.
- `integrated03` exposed a shuffled-control mirror mismatch: payload rotation
  dropped `supersedes`. The control now preserves correction authority and uses
  explicit reconciliation when the parent is absent. `integrated04` and
  `integrated05` completed before the final dependency-list correction and are
  also preserved as prior protocol generations.
- The first pressure probe used a 1 KiB cache and correctly recorded zero
  hits/evictions because records did not fit. `readout_pressure02` and
  `readout_pressure03` remain earlier accounting/provenance iterations; only
  `readout_pressure05` is the final pressure receipt. No failed output was
  overwritten.

The corrected conservative ceilings are 296 selection attempts, 66,304 total
file reads, 23,680 payload/cold reads, 2,368 cold tickets/resumes and 7,104
revalidations. They include 64 candidate key reads per selection, up to 16
full-context records and at most two selections per logical query. They are
upper bounds, not measurements or targets.

## 1. Implemented components


- `src/replica_lab/persistent_learner_runner.py`: four chronological fit events, direct response CE, eleven information/control arms, all-eligible baseline construction, canonical tiered-record restoration and correction reconciliation, extension-only checkpoints, separate-process development and fresh-process restart.
- `src/replica_lab/p1_response_learning.py`: direct response-CE gradients through selected P, a dedicated port-output-only NativeFusedAdamW owner, actual parameter/moment/clock evidence, P-only leakage audits and early/late site accounting.
- `src/replica_lab/tiered_memory.py`: scoped immutable clusters, separate retrieval keys and four ordered readable values per record, canonical text/exact-span fallback, cold recall tickets, correction quarantine, authenticated reset, recovery, tier accounting and resumable task journals.
- `src/replica_lab/tiered_readout.py`: scope/time/verification-first selection, current-record provenance restoration, exact cache/key/payload I/O accounting, cold resume and post-I/O invalidation.
- `scripts/measure_tiered_readout_pressure.py`: real-file cache-pressure and correction-during-cold-I/O evidence probe.
- `src/replica_lab/contextual_spark_fixture.py`: a CPU-only causal-attention fixture with GELU MLP and tied embeddings. It is not the native Spark model and does not diagnose native competence.
- `src/replica_lab/learned_local_writer.py`: an offline D-only functional SGD/delta study with exact derivative declarations. Imperative NativeFusedAdamW is not treated as a differentiable inner optimizer.

The v2 protocol is [`configs/spark_persistent_learner_protocol_v2.json`](../../configs/spark_persistent_learner_protocol_v2.json). The independent fixture is [`data/spark_persistent_learner_v2`](../../data/spark_persistent_learner_v2); fit/development labels are separated and the sealed final split is unmaterialized and unopened.

## 2. Previous v2 verification and demonstration (historical)

Historical focused Spark/spine/persistent-learner regression: **138 passed**. No native model was loaded by these tests.

Historical command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:scripts PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv/bin/python scripts/run_spark_persistent_learner_v2.py \
  --stage all --backend cpu_contextual_fixture --device cpu \
  --fixture-context-tokens 256 --fixture-max-new-tokens 2 \
  --output runs/spark_persistent_learner_v2/cpu_final01 \
  --config configs/spark_persistent_learner_protocol_v2.json \
  --prepared data/spark_persistent_learner_v2
```

Evidence: [`run_report.json`](../../runs/spark_persistent_learner_v2/cpu_final01/run_report.json), [`fit/tiered_receipts.json`](../../runs/spark_persistent_learner_v2/cpu_final01/fit/tiered_receipts.json), [`development/tiered_receipts.json`](../../runs/spark_persistent_learner_v2/cpu_final01/development/tiered_receipts.json), and [`storage_artifacts.json`](../../runs/spark_persistent_learner_v2/cpu_final01/storage_artifacts.json).

| Measurement | Result |
|---|---:|
| Fit events / development cases / arms | 4 / 120 / 10 |
| Direct-CE objective forwards / backward calls | 8 / 8 |
| Selected-P optimizer steps / port-output steps | 4 / 4 |
| Non-port spine / local-MSE optimizer steps | 0 / 0 |
| Initial null parity / post-fit nonzero influence | pass / pass |
| Correction quarantine + authenticated reset | 1; exact unlearning not claimed |
| Final P clock / final P lineage records | 2 / 2 |
| Development parity requests / fresh restart | 2 / exact token parity |
| Independent verification calls | 135 |
| Total forward calls / query-sensing calls | 398 / 136 |
| Autoregressive forwards / generated tokens | 246 / 246 |
| Wall time | 35.66 s |
| Peak live artifact bytes (including temporary overlap) | 73,976,967 |
| Final artifact receipts | 123 files, 71,208,829 bytes |
| Tiered E records / readable value slots | 13 / 52 |
| Tiered clusters / verified procedures / resumable workspaces | 5 / 4 / 1 |

The artifact receipts use magic bytes, not suffixes, and include extensionless checkpoint states. `extensionless_archives_safe_to_delete` is false. Base model weights are not duplicated in the run output.

## 2.1 Previous acceptance at its committed HEAD

- Standalone `fit` → `development` → `restart` completed in three separate processes at [`cpu_final_resume01`](../../runs/spark_persistent_learner_v2/cpu_final_resume01/restart/restart_report.json). Dynamic sidecars restored development state; correction reset, source-free P-only exact-token parity and tiered P digest reload passed.
- The final full repository suite at `5bdd64eac2fdcba28221e1161f6d167b3489b36a` passed **3363 tests with 9 skipped and 21 subtests in 847.99 seconds**; one unrelated sparse-tensor warning is preserved in [`full_suite.log`](full_suite.log).
- Final preflight at that HEAD is preparation-only with only `external_authorization_receipt` and `native_fp8_backend_evidence` missing. The implementation ancestry and all 13 protocol tests pass.

## 3. Learning and persistence verdict

Engineering acceptance passed: initial null parity, later nonzero influence, correction reset, exact fresh-process restart, source-free P-only audit and tiered P digest reload. The restart restores the final checkpoint, removes E/S/C/source-bearing query state and preserves P and port state.

Learning is **inconclusive**: the previous ten-arm run and the current eleven-arm `integrated09` run each scored 0/12 on this randomly initialized engineering fixture. Full-context competence and useful P transfer are not established. Parameter changes, optimizer moments and nonzero influence are not presented as useful learning. No calibration, native task-success claim, lifelong-learning claim, reasoning improvement claim or RSI claim is made.

The D-only writer has numerical and software evidence only. It remains an offline study behind the useful-readout gate; it was not promoted or used to claim improvement over time.

## 4. Preserved failure analysis

Failed attempts remain in their original output directories:
- Earlier full suite: 3358 passed and five documentation-lineage failures; see [`full_suite_initial_failure.log`](full_suite_initial_failure.log). The repaired migration/deliverable/bundle records pass 57 focused checks, and the subsequent full suite passed 3363 tests.

- `cpu_smoke01`: tokenizer IDs exceeded the initial 32-token fixture vocabulary; the runner now binds an explicit 8192-token fixture vocabulary.
- `cpu_full01`: dynamic sidecar omitted the required `plastic` key; the serializer now persists plastic, evidence and text state.
- `cpu_full02`: parent restart merge omitted `restart_parity_requests`; the merge now includes it.
- `cpu_tiered01`: pickled P state was rejected by exact FP32 layout checks; the sidecar now stores ordered raw FP32 weights and two moment arrays.
- Earlier protocol arithmetic and tiered API failures are preserved in the corresponding test/output history. No failed directory was overwritten.
- `integrated01` refused when the derived 888-read ceiling was exceeded; the failure and exact accounting correction are preserved in section 0.
- The first pressure probe used an undersized 1 KiB cache and is preserved separately from the valid `readout_pressure05` receipt.

## 5. Native admission and next experiment

The active source is `XHToken/Spark-X2.6-4B-FP8` at `0318d6f6ab070c7b5b43052db571489b2508006c`, with 36 layers. Sites 3/7 and 23/31 are valid candidates, not measured native results. Official BF16 and FP8-derived BF16 are separate identities and cannot substitute for official native FP8.

The integrated implementation is committed at `83c7214bcf12877e4c7b9d0c367825905a16cf55`; the protocol launch pointer and complete 38-file dependency hash map bind it. [`review_final03.json`](review_final03.json) is preparation-only and leaves only exact official-FP8 forward/backward/single-update backend evidence on MPS and an external authorization receipt. [`native_backend_plan.md`](native_backend_plan.md) and [`native_experiment_packet.json`](native_experiment_packet.json) record the blocker and bounded P1 packet. The separate [`derivative_bf16_packet.json`](derivative_bf16_packet.json) is prepared at the same commit and awaits external derivative authorization. `review03.json` and `review02.json` preserve earlier reviews. The capacity wrapper registers `run_spark_persistent_learner_v2.py`; native execution requires its watchdog environment, offline flags, `PYTORCH_ENABLE_MPS_FALLBACK=0` and the exclusive MPS worker. The parent releases its model before the supervised restart child starts.

Review command (no model load):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:scripts \
  .venv/bin/python scripts/preflight_spark_persistent_learner_v2.py \
  --config configs/spark_persistent_learner_protocol_v2.json \
  --receipt reports/spark_persistent_learner_v2/review_final03.json
```

Do not launch native training until the exact backend receipt and external authorization exist. Do not expand to P4/P16/P32/P64 or promote the learned writer merely because the CPU engineering path passes. The next discriminating experiment is one bounded native P1 full-context/direct-CE readout with charged 3/7 versus 23/31 sites and the same controls, after exact backend qualification.

The protected `src/replica_lab/runtime_pilot.py` remains untouched and untracked.
