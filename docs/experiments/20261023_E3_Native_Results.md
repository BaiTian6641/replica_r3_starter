# E3 shared core, exact task journal and model-driven action loop

23 October 2026 UTC / 22 October America/Chicago. Branch
`codex/d2-capacity-runtime-pilot`. Implementation commits `e16a3f3`, `59ef1bb`,
`8e95e2b`; protocol `configs/e3_qualification_v1.json` committed at `8e95e2b`
before native execution. Attempt 3 is the accepted evidence.

**Engineering result: pass. Learned task completion: negative on the exposed
fixture family.** The shared core, independent workspace readers, E2 parent
migration, exact journal, typed actions, real tools, independent success check,
controls and fresh-process restart are connected. MiniCPM5-2B learns the
single-object action format, selects real actions and produces real receipts,
but does not complete the configuration-repair task: the full-core arm repeats
`read`, and the arm that reaches the repair writes an invalid file. No success
was declared by the host.

## 1. Protocol and initialization

| Item | Declared and executed |
|---|---|
| Source | sealed `checkpoints/minicpm5_source`, revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52` |
| Parent | `runs/e1_e2_native_attempt02_20261023/selected_extension` used as initialization **and** regression reference |
| Migration | 43 E1/E2 tensors copied exactly; 51 E3 tensors kept at fresh initialization; parent manifest/inventory validated |
| Core | width 256, 8 slots, 2 tied iterations, 1024-wide FFN, 4 heads |
| Branches | memory readers and workspace readers at blocks 13/27; independent zero-initialized outputs |
| Training | modules-only, Muon 8-bit plus FP32 AdamW, 16 updates × 4 examples **per arm**, seed 20261023, lr 0.01 Muon / 0.005 AdamW, warmup 2, clip 1.0 |
| Arms | recurrent selected; matched feed-forward control trained from the same parent and swapped only for its control episode |
| Task fixtures | 2 train, 1 dev, 2 held-out, 1 interrupted; exposed synthetic fixtures |
| Budgets | 7 decisions, 5 tool calls, 640 prompt tokens/decision, 128 generated tokens/decision |
| Decoding | greedy, no thinking, uncached re-prefill, CPU fallback disabled |
| Storage | one extension-only v2 export; no base copy, no optimizer checkpoint, zero downloads |

The parent extension is intentionally nonzero, so initial null parity was
measured on the freshly initialized core adapter **before** loading it. All new
output projections start at zero and the initialized augmented model reproduces
native logits exactly.

## 2. Engineering checks (all pass)

| Check | Result |
|---|---|
| Initial native parity, fresh core adapter | max absolute logit difference 0 |
| E2 parent migration | accepted, exact E1/E2 inventory, parent bytes preserved |
| E2 parent regression | all 43 loaded parent tensors equal the artifact |
| Independent workspace-reader influence | nonzero logit change, separate from memory |
| Independent memory-reader influence | nonzero logit change, separate from workspace |
| Recurrent core gradient paths | sensors, core and workspace readers receive finite gradients |
| Feed-forward core gradient paths | same, in the matched control arm |
| Stale workspace after host correction | rejected as `stale_revision`; workspace cleared/rebound |
| Extension-only export | `replica.minicpm5.extensions.v2`, base not saved |
| Original parameters unchanged | sealed original stamp equal before/after |
| No base checkpoint copy | only workspace/latent tensors and the extension exist |
| Success head scope | prediction-only; no calibration claim |

## 3. Alignment

Both arms reduce teacher-forced action loss from **11.67** to about **0.2**
across 16 updates. The recurrent arm's update-2 audit contains finite nonzero
gradients for `core.*`, `sense_ports.*` and `workspace_ports.*`; the
feed-forward arm does the same. This establishes the required gradient path,
not capability.

A pre-native short alignment probe at 8 updates generated a valid
`{"kind":"stop"}` on a train prompt, so the format was learnable before the
bounded run. The full run's episodes confirm format learning but expose wrong
action selection/arguments.

## 4. Model-driven episodes

All outcomes below come from MiniCPM generation, host validation and actual
fixture tools. Nothing was substituted by the host.

| Case / arm | Host-verified success | Observed trajectory |
|---|---:|---|
| dev `full` | no | `read` ×7; no test executed |
| held-out timeout `full` | no | `read` ×7 |
| held-out retry `full` | no | `read` ×7 |
| dev `workspace_zero` | no | read → test(fail) → write → test(fail) → malformed → stop(fail) → stop(fail) |
| dev `workspace_shuffled` | no | read → test(fail) → malformed → test(fail) → malformed → stop(fail) ×2 |
| dev `ff_core` | no | read → test(fail) ×4 → denied tool ×2 |
| dev `core_disabled` | no | `stop` ×7 |
| dev `extra_inference` | no | stop/read alternation; stop claims fail the independent check |

Totals: **56 decisions, 32 tool calls, 3 malformed actions, 13 false success
claims, 0 host-verified completions**, 26,978 prompt tokens, 1,033 generated
tokens and 241.5 s of episode wall time. Every false `stop` was denied or failed
by the independent evaluator; no model-written success statement authorized
completion.

The `workspace_zero` arm did change its plan after the observed failing test and
wrote a repair. The written file was not the reference repair: it replaced the
docstring and constants with a malformed block (`BOUND_VALUE`, a stray
`"""`, missing `REQUEST_TIMEOUT_MS`/`BACKOFF_SECONDS`), so the evaluator still
failed. The full arm's repeated `read` is consistent with a task-owned
previous-workspace attractor: zeroing that previous workspace is the only
control that progressed to a write. The feed-forward core instead repeated the
test. These are negative behavioral findings, not a core-superiority result.

### Success head

Eight actual episode outcomes were available, but all were negative
(`labels=8, positives=0, negatives=8`). The declared rule therefore skipped
training (`single_outcome_class`). No success prediction, confidence or
calibration is claimed. A future arm with both classes must use a disjoint
calibration role.

## 5. Fresh-process interruption

The interrupted fixture was exercised through the guarded CLI, not in the main
driver. A workspace-zero arm observed the failing test in process 1
(`read`, `run_test` exit 1), then a CPU host correction changed revision 10→11.
The stale workspace was detected and rejected (`stale_revision`,
`stored_workspace_digest_matches=false`). Process 2 loaded the journal and
resumed at revision 11, recorded `workspace_discarded: stale_revision`, did not
re-execute the completed read/test, and spent the remaining five decisions.
The model then emitted denied actions and false `stop` claims. The independent
evaluator still fails; the repair never landed.

Acceptance checks split cleanly: **10/14 pass** (observed failure, incomplete
process 1, correction revision/goal preservation, stale workspace, incompatible
workspace rebuild, unmodified evaluator) and four learned-completion checks
fail (process 2 complete, independent pass, repair landed once, workspace equals
reference). The report is
[acceptance.json](../../reports/e3_interrupted_20261023/acceptance.json). The
state reconstruction and idempotency engineering pass; the task does not.

## 6. Cost and storage

| Attempt 3 measurement | Value |
|---|---:|
| Guard/profile wall time | 424.18 s profiled / 426.78 s guard |
| Sampled peak MPS driver | 14.72 GiB |
| Sampled peak process RSS | 3.36 GiB |
| Maximum observed swap growth | 0 MiB |
| Minimum observed free memory | 65% |
| Optimizer updates | 32 (16 recurrent + 16 feed-forward) |
| Final extension | 37,588,831 bytes / 35.85 MiB |
| Run artifacts | 41,036,240 bytes / 39.14 MiB |
| New original/base checkpoint copies | 0 |
| Downloads | 0 |

MPS driver and RSS overlap in unified memory and must not be added. Peaks are
sampled and can miss transients. The extension contains E2, core, feed-forward
control, workspace readers and the untrained success head; it references the
sealed source and is not a resumable optimizer checkpoint. The source was reused
unchanged; no Neural Engine execution is claimed.

## 7. Preserved attempts and failure analysis

| Attempt | Outcome | Cause |
|---|---|---|
| 1 | failed before training | null parity was measured after loading the trained E2 parent, which is intentionally nonzero; fixed by measuring on the fresh adapter |
| 2 | completed training; 56/56 actions denied; 0 tool calls | valid JSON followed by the tokenizer's `</s>` was parsed as a second object; fixed with a bounded decoder-stop-token rule |
| 3 | engineering pass; 0 task completions | model repeats `read` in the full arm; `workspace_zero` writes an invalid repair; controls fail differently |
| interrupted pre-check | failed immediately | CLI initial-state binding used a nested `component_versions` dict; fixed to scalar fields and preserved |

Attempt 2 consumed 32 updates; attempt 3 consumed another 32. Both extensions
and all raw reports are retained. The parser failure is a real interface defect,
not a model-capability result; attempt 3 is the first run whose action text
reached the host validator intact.

## 8. What is established and what is not

**Established (engineering):** the reference core and two independent native
reader branches are connected; the E2 parent migrates by validated inventory;
the exact journal reconstructs an interrupted task and rejects stale/revoked
workspace state; typed actions are parsed/validated; real tools produce
receipts; only the independent evaluator can satisfy the goal; controls and
fresh-process restart work; costs and storage are measured.

**Trained:** modules-only action-format/core/reader alignment for two
sequential arms (32 optimizer updates this attempt; 64 across attempts 2 and 3).
The success head is untrained because the label set had one class.

**Measured negative:** held-out configuration repair, full-core action
selection and plan continuation after restart. The model learns JSON action
syntax but does not reliably select the next action or reproduce a correct
file. No task success was declared.

**Not established:** generalization, calibrated confidence, memory/core
benefit, retrieval quality, full-bank native capacity, Replica fidelity,
self-improvement, VLM or ANE execution. The fixtures are exposed development
cases; the controls are bounded engineering comparisons. A future narrower
repair tool and fresh held-out cases would be a new preregistered arm, not a
reinterpretation of this result.

## Evidence

- [attempt 3 run report](../../reports/e3_native_attempt03_20261023/run_report.json)
  and [guard](../../reports/e3_native_attempt03_20261023/guard.json)
- [attempt 2 run report](../../reports/e3_native_attempt02_20261023/run_report.json)
  (EOS-parser negative)
- [attempt 1 failure](../../reports/e3_native_attempt01_20261023/failure_report.json)
- [interrupted process 1](../../reports/e3_interrupted_20261023/run1_session_report.json),
  [process 2](../../reports/e3_interrupted_20261023/run2_session_report.json),
  [correction](../../reports/e3_interrupted_20261023/correction.json),
  [acceptance](../../reports/e3_interrupted_20261023/acceptance.json)
- [protocol](../../configs/e3_qualification_v1.json),
  [implementation spec](../architecture/08_E3_Implementation.md),
  [fixture review](../../reports/e3_fixture_review_20261023.json)
