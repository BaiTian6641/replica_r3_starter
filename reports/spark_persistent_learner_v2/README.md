# Spark persistent learner v2 — implementation and evidence

**Status: integrated CPU engineering is complete; native execution is not admitted and useful learning is inconclusive.** This is a separately versioned runtime, not a rewrite of sealed v1 evidence.

## 1. Implemented components

- `src/replica_lab/persistent_learner_runner.py`: four chronological fit events, direct response CE, ten information/control arms, extension-only checkpoints, separate-process development and fresh-process restart.
- `src/replica_lab/p1_response_learning.py`: direct response-CE gradients through selected P, a dedicated port-output-only NativeFusedAdamW owner, actual parameter/moment/clock evidence, P-only leakage audits and early/late site accounting.
- `src/replica_lab/tiered_memory.py`: scoped immutable clusters, separate retrieval keys and four ordered readable values per record, canonical text/exact-span fallback, cold recall tickets, correction quarantine, authenticated reset, recovery, tier accounting and resumable task journals.
- `src/replica_lab/contextual_spark_fixture.py`: a CPU-only causal-attention fixture with GELU MLP and tied embeddings. It is not the native Spark model and does not diagnose native competence.
- `src/replica_lab/learned_local_writer.py`: an offline D-only functional SGD/delta study with exact derivative declarations. Imperative NativeFusedAdamW is not treated as a differentiable inner optimizer.

The v2 protocol is [`configs/spark_persistent_learner_protocol_v2.json`](../../configs/spark_persistent_learner_protocol_v2.json). The independent fixture is [`data/spark_persistent_learner_v2`](../../data/spark_persistent_learner_v2); fit/development labels are separated and the sealed final split is unmaterialized and unopened.

## 2. Verification and final demonstration

Focused Spark/spine/persistent-learner regression: **138 passed**. This includes six original protocol tests, seven v2 protocol/supervision tests, 23 tiered tests, the authenticated-reset regression and the P1/writer tests. No native model was loaded by these tests.

Final command:

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

## 2.1 Final acceptance at the committed HEAD

- Standalone `fit` → `development` → `restart` completed in three separate processes at [`cpu_final_resume01`](../../runs/spark_persistent_learner_v2/cpu_final_resume01/restart/restart_report.json). Dynamic sidecars restored development state; correction reset, source-free P-only exact-token parity and tiered P digest reload passed.
- The final full repository suite at `5bdd64eac2fdcba28221e1161f6d167b3489b36a` passed **3363 tests with 9 skipped and 21 subtests in 847.99 seconds**; one unrelated sparse-tensor warning is preserved in [`full_suite.log`](full_suite.log).
- Final preflight at that HEAD is preparation-only with only `external_authorization_receipt` and `native_fp8_backend_evidence` missing. The implementation ancestry and all 13 protocol tests pass.

## 3. Learning and persistence verdict

Engineering acceptance passed: initial null parity, later nonzero influence, correction reset, exact fresh-process restart, source-free P-only audit and tiered P digest reload. The restart restores the final checkpoint, removes E/S/C/source-bearing query state and preserves P and port state.

Learning is **inconclusive**: all ten arms scored 0/12 on this randomly initialized engineering fixture. Full-context competence and useful P transfer are not established. Parameter changes, optimizer moments and nonzero influence are not presented as useful learning. No calibration, native task-success claim, lifelong-learning claim, reasoning improvement claim or RSI claim is made.

The D-only writer has numerical and software evidence only. It remains an offline study behind the useful-readout gate; it was not promoted or used to claim improvement over time.

## 4. Preserved failure analysis

Failed attempts remain in their original output directories:
- Earlier full suite: 3358 passed and five documentation-lineage failures; see [`full_suite_initial_failure.log`](full_suite_initial_failure.log). The repaired migration/deliverable/bundle records pass 57 focused checks, and the subsequent full suite passed 3363 tests.

- `cpu_smoke01`: tokenizer IDs exceeded the initial 32-token fixture vocabulary; the runner now binds an explicit 8192-token fixture vocabulary.
- `cpu_full01`: dynamic sidecar omitted the required `plastic` key; the serializer now persists plastic, evidence and text state.
- `cpu_full02`: parent restart merge omitted `restart_parity_requests`; the merge now includes it.
- `cpu_tiered01`: pickled P state was rejected by exact FP32 layout checks; the sidecar now stores ordered raw FP32 weights and two moment arrays.
- Earlier protocol arithmetic and tiered API failures are preserved in the corresponding test/output history. No failed directory was overwritten.

## 5. Native admission and next experiment

The active source is `XHToken/Spark-X2.6-4B-FP8` at `0318d6f6ab070c7b5b43052db571489b2508006c`, with 36 layers. Sites 3/7 and 23/31 are valid candidates, not measured native results. Official BF16 and FP8-derived BF16 are separate identities and cannot substitute for official native FP8.

The implementation and protocol are committed at `1d8626957c84aa37cf4858ea4a74cd3797a0bff5`. [`review03.json`](review03.json) is preparation-only and leaves only exact official-FP8 forward/backward/single-update backend evidence on MPS and an external authorization receipt. `review02.json` preserves the earlier pre-binding review. The capacity wrapper registers `run_spark_persistent_learner_v2.py`; native execution requires its watchdog environment, offline flags, `PYTORCH_ENABLE_MPS_FALLBACK=0` and the exclusive MPS worker. The parent releases its model before the supervised restart child starts.

Review command (no model load):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:scripts \
  .venv/bin/python scripts/preflight_spark_persistent_learner_v2.py \
  --config configs/spark_persistent_learner_protocol_v2.json \
  --receipt reports/spark_persistent_learner_v2/review03.json
```

Do not launch native training until the exact backend receipt and external authorization exist. Do not expand to P4/P16/P32/P64 or promote the learned writer merely because the CPU engineering path passes. The next discriminating experiment is one bounded native P1 full-context/direct-CE readout with charged 3/7 versus 23/31 sites and the same controls, after exact backend qualification.

The protected `src/replica_lab/runtime_pilot.py` remains untouched and untracked.
