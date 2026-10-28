# Resume repair and staged learning — 27 October 2026

## Scope and authority

Prior implementation base: `f0fff3e4`. Starting HEAD for this repair: `4c5a8485`.
Resume/staging implementation commit: `71955206`. MiniCPM5 remains
baseline and NativeFusedAdamW/`adamw_fused` default. The 50 GiB driver, 512 MiB
sustained swap-growth, 15% host-free, exclusive-worker and disabled-fallback
policies are unchanged. The protected runtime pilot was not read, imported,
modified or staged. No new native model run, download, promotion, neural RSI or
personality adaptation occurred.

## 1. Experiment-level resume and resulting state

`src/replica_lab/value_readout_abstention.py` now persists the global arm cursor,
completed-arm content-addressed snapshots, active optimizer state, clocks,
memory, input/config identities and physical work by process. Completed arms are
never retrained. Resume restores the active arm only, then starts the next arm at
`next_arm_index`.

`RunInterrupted` is also handled by the `all` CLI: update 29 returns exit 3 with
`completed_updates=29` and a resume path rather than an unhandled traceback.
`all --resume` continues through fit, development, held-out and controls. The
unhandled-CLI defect is preserved in
`reports/verification_20261027/resume_all_interrupt_error.json`.

Actual update-29 evidence:

- Process split `[29, 43]`; cumulative **72 optimizer steps, 72 backward calls
  and 72 training forwards**.
- `cursor.next_arm_index=3`, all three arms complete; completed arms are never
  retrained.
- Checkpoint, fit predictions, fit report, fit receipt, both control checkpoints
  and both control prediction streams are byte-identical to an uninterrupted run.
- `resume_evidence.json` is included in the complete artifact tree. Peak resume
  bytes are 28,407,590, within the unchanged 33,554,432-byte cap. Heavy resume
  tensors are removed only after measurement, leaving the evidence receipt.

Boundary tests cover updates 24/25/29/48/49 and repeated fresh-process restarts.
They assert physical counts, completed-arm behavior, exact cursor contents,
process histories, identity checks, final update/state hashes and deterministic
replay. **47 memory tests passed.**

Resulting-state validation now rejects valid-looking final update hashes that do
not match checkpoint state, contradictory event/case/world/epoch/clock records,
rehashed controls and truncated update histories. Treatment and trained controls
use the same checks. Strict tokenizer decoding and fit-before-development-before-
held-out access are unchanged.

## 2. Integration success is not learning improvement

The 22-class CPU fixture now includes explicit baselines:

- untouched zero-update parent: **27/27**, already passing all fit gates;
- reset-memory baseline: **12/27** (all UNKNOWN), not a learned improvement.

Its passing result remains software integration evidence only. It is not learned
retrieval, persistent learning or an answer-index shortcut.

DC-P1 and DC-R1 are separate, role-sealed protocols using native representations
and generated answers/readout rather than the integration fixture:

- **DC-P1** changes only `loss.answer_span_ce_weight` 0.0 → 0.5. It requires
  fresh paraphrase/new-instance gains over identical no-update and reset-P
  controls with source context and source KV removed, plus correction,
  abstention and unaffected-skill retention. Budget02 is explicitly not revived.
- **DC-R1** compares exact context, selected literal, compressed record,
  wrong value and no memory across layers 13/27/41. It separates accepted-layer
  accuracy, complete-group recall, exact/compressed reconstruction and
  wrong-value follow rate. Learned retrieval/storage expansion remains blocked
  until a useful readout gate passes.

Fixtures include 12 development source groups plus 4 sealed-final groups for
P1, and 12 development plus 4 sealed records for R1. The 22-class fixture is not
a parent or scoring oracle in either protocol. Validators reject source/target
fields, answer indexes, answer-bearing routing metadata, source-retained/KV rows,
tokenizer/decode mismatches and prompt mismatches. **33 DC tests passed.**

Review commands:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/validate_dc_protocols_20261027.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/preflight_dc_p1_native_response_20261027.py \
  --receipt reports/verification_20261027/dc_p1_preflight.json --review
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/preflight_dc_r1_native_value_20261027.py \
  --receipt reports/verification_20261027/dc_r1_preflight.json --review
```

No native runner or authorization exists in this preparation package. A future
native run requires a separate authorization receipt bound to exact config and
fixture hashes, one attempt, fresh output and the unchanged guard.

## 3. Bounded vision readiness

`reports/vision_smaller_step_launch_readiness_v1_20261027/readiness.json` confirms
**no valid authorization exists** for `vision_smaller_step_lr3e5_v1`. The current
launch packet is verified at canonical digest
`9be3a8d2edc7d0b4d819197e22cd316a82affd7b83345cc95c2b2ef7d591a770`.
**161 vision readiness/contract tests passed**, with `readiness_current` and
`coadaptation_preparation_current`.

The exact remaining launch requirements are:

1. a separate authorized receipt with all 14 declared fields and config SHA-256
   `931851ecc954cea91d97aa7a087c1a2bef528172379b06d453e6bec93ef4abad`;
2. a full 40-character protocol commit whose config blob has that hash;
3. a passing `review-launch` receipt/config/commit/fresh-output check before any
   native import;
4. an exclusive MPS worker with the 50 GiB / 512 MiB / 15% guard and disabled
   fallback;
5. one optimizer step at lr 3e-5, four backward calls, within 4,824 decoder
   forwards, 168 visual observations, 1,200 seconds and 32 MiB including the
   final report, with raw generation evidence preserved.

The optimization and functional-output gates remain independent. There is no
automatic retry, held-out reopening or promotion. The conditional co-adaptation
control is prepared but ineligible: it requires a preserved smaller-step failure
and a diagnosis excluding step size, labels/masking and accounting. It is not an
encoder swap, new loss or larger-budget expansion.

## 4. Spark v5.1 migration and staged capacity

The v5.1 documents, design config and review bundle now have an explicit reviewed
migration (`sparkx-2.6-v5.1-review-migration-20261027`) with old/new identities
and reasons. Historical source snapshots and receipts are unchanged. The
migration has 694 checks; the final-product verifier has 508 checks and a
75-input, 4,511,256-byte bundle with digest
`3299e1e6f6c38a3d4007f96490aa754ef7cec30532f202274ff120c4d8f587ab`.
**29 migration/final-product tests passed.**

The residency inventory separates retained vision tower, base/extended weights,
optimizer state, activations, restart transients, P16/P32/P64 and Mkv-4k/64k.
Official BF16, official FP8 storage, dequantized derivatives and native FP8
execution remain distinct. Arithmetic is labeled analytical, not measured fit or
training qualification. Small-extension estimates are not complete-evolution
residency.

P32/P64 and Mkv remain deferred until a smaller useful learner exists. Sparse
experiments first require inactive-row and optimizer-state preservation,
duplicate-index gradient handling, per-row clocks and actual allocation/restart
measurements. Source-group splits and derived-example split inheritance are
declared; exposed audit data is not recycled. Teacher competence, gradient
approximation and all teacher/rollout/replay/evaluation work must be declared
before outer loops, multiple clocks, self-distillation or RSI. Personality is a
separately specified and evaluated component.

The native-gate register remains unresolved for response learning, vision,
retrieval/readout, P capacity, Mkv execution, teacher/distillation, outer
gradients/clocks, Spark full training, native FP8 and personality. Design checks
do not satisfy these gates.

## 5. Verification and next bounded experiment

Focused results:

| Check | Result |
|---|---|
| Token/admission | 272 passed |
| Source provenance | 174 passed |
| Memory resume/admission | 47 passed |
| DC protocols | 33 passed |
| Vision readiness/contract | 161 passed |
| Data/Spark | 54 passed |
| Spark v5.1 migration/final product | 29 passed |

Historical continuation replay is `exact_match`; accounting replay is 7,695/7,755;
corrected step-1 preserves original engineering failure and derived supported
verdict. Response-CE fit remains 0/28, memory v4 remains 21/384, and the vision
curve remains `curve_no_optimization_progress`.

The next bounded experiment is DC-P1 only after a separate protocol commit and
authorization: test whether answer-span CE produces useful fresh paraphrase/new-
instance generated answers over identical no-update and reset-P controls, with
source context/KV removed. If it fails, report the result without retry. The
smaller-step vision diagnostic remains blocked only by its separate authorization
receipt; if it fails, the prepared co-adaptation hypothesis is the only allowed
next control and requires its own diagnosis/commit/receipt. Spark native FP8 and
full-training qualification remain separate unexecuted gates.
