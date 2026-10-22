# S0 — reproducible full-text training readiness

Reviewed **22 October 2026, America/Chicago**. See the [master roadmap](00_Master_Roadmap.md), [current execution state](../../execution_state.md) and [dated primary-source review](07_Research_Evidence_20261022.md). Gates and budgets are proposed unless a linked existing report establishes execution. Planned artifacts below do not claim files already exist.

The historical constant-zero negative is preserved. The new native full-text run from the sealed original now meets its **training-generation subgate**: **8/8 correct and eight distinct answer classes `0`–`7` at both updates 4 and 8**, with EOS endings. This is deliberately an eight-class memorization/overfit diagnostic, not a binary-label test. Final checkpoint/export/reload acceptance, memory review, fresh-process native resume and independent development/control results remain separate gates.

The declared successful intervention combines answer-balanced accumulation of eight examples per update, matched Muon/auxiliary AdamW base rates `1e-4`, two-update warmup and zero weight decay. The run stopped at update 8 after two perfect checks within its declared 64-update cap. Because several variables changed together, the observation does not identify their individual causal contributions. All examples were still **55 actual tokens**. [Update-4 generations](../../runs/minicpm5_balanced_mask1_B8_lr1e4_chunk01_20261022/train_generation_step_0004.json), [update-8 generations](../../runs/minicpm5_balanced_mask1_B8_lr1e4_chunk01_20261022/train_generation_step_0008.json).

Existing reproducible entry points remain:

```bash
uv sync --locked --extra neural --extra test
.venv/bin/python scripts/check_core.py --output reports/core_checks_local.json
.venv/bin/python scripts/validate_data.py data/fixtures/train.jsonl data/fixtures/dev.jsonl --require-verified
```

Use new output paths for experiments; exact current balanced-run commands/results belong in the main runbook and execution state. Muon rates and auxiliary AdamW rates are different controls. Audit actual target coverage per effective update, response/EOS supervision and native template prefix before tuning either. A drop dominated by response formatting/EOS can coexist with wrong task answers.

Capacity qualification must retain actual encoded length, source/support/supervised token counts, maximum new tokens, memory sampling cadence, process lifetime and phase events. On unified memory, driver and RSS counters overlap. Keep host memory safeguards and declare stop thresholds before loading. Tiny real CPU checkpoint/next-update equivalence and MPS mixed-optimizer/RNG restoration tests now pass; native pretrained fresh-process resume is not yet proven. Checkpoints are supported only between completed optimizer updates with gradients cleared. Optimizer-state serialization alone does not establish the full RNG/schedule/cursor and native model continuation contract; mid-accumulation resume is unsupported.

## S0.1 — Preserve the reproducible native source and environment

**Status:** `measured_short_scope`. **Dependencies:** none; can start independently.

Reuse the existing Git history, Python lock, sealed checkpoint and source template. Capture code revision and hardware/framework inventory before each new run; never substitute a sibling checkpoint.

- **Artifact:** uv.lock; configs/minicpm5_sealed_lock.local.json; reports/hardware_macos_m2_ultra.json; reports/minicpm5_mps_experiment_20261022.json.
- **Interface:** Source lock -> downloader/sealer -> LlamaMemoryAdapter and native evaluator; one exact source revision shared by all arms.
- **Test gate:** Existing 55-token native baseline and full D0 evidence is preserved. A clean locked setup can run offline checks, validate fixtures and reproduce native load with the same model/template hashes.
- **Resources:** M2 Ultra 64 GiB; Python 3.12.14; torch 2.14.1/transformers 5.20.2; one model resident. No reacquisition or environment upgrade needed for this task.
- **Failure policy:** Stop on seal or version mismatch; identify the changed file/environment before proceeding. Preserve original reports and user changes.
- **Next smallest task:** Record the current commit and sealed revision in the new balanced-overfit run manifest.

## S0.2 — Diagnose the constant-zero failure

**Status:** `diagnosed_bundled_remediation_observed`. **Dependencies:** S0.1.

Audit the historical two selected rows, target/EOS masks and native generation boundary, and preserve the short constant-zero negative. The balanced eight-class run now removes constant-zero behavior on its training cases; report its sampler, accumulation, rate, warmup and weight-decay changes as a declared bundled intervention rather than isolated causal factors.

- **Artifact:** reports/minicpm5_mps_preflight_report.json; runs/minicpm5_balanced_mask1_B8_lr1e4_chunk01_20261022/run_report.json; data/diagnostics/tiny_overfit_mask1/selection_manifest.json.
- **Interface:** Raw train export + encoded examples + strict generation report -> diagnosis; evaluator must preserve raw completions and EOS/cap reasons.
- **Test gate:** Historical four supervised tokens and constant-zero control are accounted for, and mask/template/target-coverage checks are inspected. Eight-class training generation now succeeds under the bundle. Exact causal attribution to balancing, rate, accumulation, warmup or decay remains unisolated.
- **Resources:** CPU file/token audits first; at most one short source/D0 MPS evaluation if an unresolved prefix or mask hypothesis requires it.
- **Failure policy:** Preserve the historical failure and do not infer that two updates demonstrate an intrinsic Muon defect. Any future causal-factor claim requires a separate controlled ablation; the successful bundled remediation does not supply it.
- **Next smallest task:** Complete checkpoint/export and development/control review of the successful bundled run before assigning broader scope.

## S0.3 — Run balanced full-text tiny overfit and verify generated answers

**Status:** `training_generation_subgate_passed_acceptance_pending`. **Dependencies:** S0.2.

The native full-text run from the sealed original generated all eight tiny-overfit training answers correctly, with eight distinct classes 0–7, at updates 4 and 8. Each accumulated update covers every answer class once. The declared bundle uses accumulation 8, matched Muon/auxiliary AdamW base rates 1e-4, two-update warmup and zero decay; the 64-update cap stopped at 8 after two perfect checks. Complete artifact acceptance and evaluate development/control cases separately.

- **Artifact:** data/diagnostics/tiny_overfit_mask1/train.jsonl; data/diagnostics/tiny_overfit_mask1/selection_manifest.json; runs/minicpm5_balanced_mask1_B8_lr1e4_chunk01_20261022/train_generation_step_0004.json; runs/minicpm5_balanced_mask1_B8_lr1e4_chunk01_20261022/train_generation_step_0008.json; runs/minicpm5_balanced_mask1_B8_lr1e4_chunk01_20261022/run_report.json; pending: final checkpoint/export/reload acceptance and memory review.
- **Interface:** Existing eight-field trainer export -> full regime -> native saved export -> unchanged strict generation parser. Record per-update target coverage, effective batch and all optimizer group rates.
- **Test gate:** Generation subgate observed: 8/8 strict greedy matches with all eight answer classes 0–7 at both updates 4 and 8, each ending in EOS. Overall acceptance still requires final full-scope/checkpoint/export/reload and memory review. Fresh-process native resume is S0.4; development transfer and controls are independent and not established by overfit exactness.
- **Resources:** Completed one bounded MPS full-text run: microbatch 1, accumulation 8, actual length 55, activation checkpointing, eight completed updates under a declared 64-update cap. Final memory acceptance remains under review; this is not a 256-token workload.
- **Failure policy:** Retain the historical collapse and all control/development failures. A later artifact or transfer failure narrows the conclusion without erasing the 8/8 training observation. Do not credit any single variable in the bundled intervention without ablation.
- **Next smallest task:** Finish the saved-checkpoint/export/reload and memory audit, then verify generated answers from the export and a fresh-process native resume.

## S0.4 — Make full-text training resumable

**Status:** `tiny_resume_proven_native_pending`. **Dependencies:** S0.3.

Checkpoint model, mixed Muon/AdamW optimizer tensors, schedule/warmup, RNGs, data order/cursor, completed update/token counts and exact parent/config hashes only between completed optimizer updates with gradients cleared. Tiny real CPU resume equivalence and MPS optimizer/RNG tests pass; qualify a fresh-process native pretrained continuation next. Mid-accumulation checkpoints/resume are unsupported.

- **Artifact:** src/replica_lab/training_state.py; src/replica_lab/checkpointing.py; tests/test_checkpointing.py; tests/test_training_state.py; reports/tests_balanced_training_code.xml; pending: fresh-process native pretrained resume report.
- **Interface:** Completed-update trainer boundary -> atomic model/runtime checkpoint with byte receipt -> contract-bound optimizer/model/RNG/data/schedule restore. Native inference export is separate; an interrupted partial accumulation restarts from the last completed checkpoint.
- **Test gate:** Tiny CPU continuous versus interrupted/resumed next-update equivalence and MPS mixed-optimizer/RNG restoration tests have passed. Native pretrained resume still must consume identical next example IDs and schedule, retain INT8/FP32 states, produce finite full-scope updates and meet declared loss/weight tolerance in a fresh process. Reject source/group/data/config mismatches and incomplete checkpoints.
- **Resources:** Existing CPU/tiny MPS evidence precedes a bounded native MPS continuation. Measure save/restore host/MPS peaks and disk bytes. Do not run another full-size model concurrently.
- **Failure policy:** A weights-only reload is not resume. Resume only at completed-update checkpoints; no mid-accumulation gradient buffer recovery is claimed. Quantify any MPS numerical tolerance while keeping cursor/schedule compatibility exact.
- **Next smallest task:** Load the completed-update native checkpoint in a fresh process, validate its byte receipt/contract and execute the next bounded full update with recorded cursor/schedule.

## S0.5 — Measure genuine 256-token and 512-token full training capacity

**Status:** `pending`. **Dependencies:** S0.3, S0.4.

Generate varied episodes whose native encoded sequence lengths actually reach 256 and then 512 without padding-only inflation or answer truncation. Profile D0 forward/backward/update, accumulation, evaluation, state save and reload, then repeat for the selected D2 configuration.

- **Artifact:** planned: capacity workload manifest with length and supervised-token histograms; planned: per-regime 256/512 memory samples and throughput reports.
- **Interface:** Tokenizer/template length audit -> bounded trainer -> profiler phase events. Compare allocated/driver/RSS counters with host pressure/swap; identify overlap and sampling cadence.
- **Test gate:** Each claimed length is reached by real encoded content. Full scope, finite updates, qualified restart and export pass with declared headroom, no fallback and no sustained memory-pressure/swap-growth breach. D0 success alone does not qualify D2.
- **Resources:** Mac 64 GiB; run 256 before 512, one model resident, microbatch 1/checkpointing first. Declare a conservative pressure/driver-memory stop threshold before starting; retain OS safeguards.
- **Failure policy:** Stop and preserve the failing profile. Reduce length/accumulation or add tested residency changes without freezing hidden layers. Do not extrapolate from the historical 55-token result.
- **Next smallest task:** Create the 256-token workload and verify actual lengths with the pinned tokenizer before allocating the full model.
