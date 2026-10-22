# S0 — reproducible full-text training readiness

Reviewed **22 October 2026, America/Chicago**. See the [master roadmap](00_Master_Roadmap.md), [current execution state](../../execution_state.md) and [dated primary-source review](07_Research_Evidence_20261022.md). Gates and budgets are proposed unless a linked existing report establishes execution. Planned artifacts below do not claim files already exist.

The historical constant-zero negative is preserved. The native full-text **eight-class tiny-overfit gate now passes**: **8/8 correct and eight distinct answer classes `0`–`7` at updates 4 and 8**, with EOS endings. Checkpoint/export/reload and the **55-token** memory profile are accepted. A fresh process then resumed **update 8→12**, again generated **8/8 correct with eight distinct answers**, retained nonzero gradients in all **381 original tensors**, and exported/reloaded with **0.0 maximum compared-logit difference**. [Final result record](../../reports/minicpm5_balanced_overfit_experiment_20261022.json).

The declared successful intervention combines answer-balanced accumulation of eight examples per update, matched Muon/auxiliary AdamW base rates `1e-4`, two-update warmup and zero weight decay. The initial run stopped at update 8 after two perfect checks within its declared 64-update cap. Several variables changed together, so their individual causal contributions are unisolated. Sampled peak MPS allocation was **24.88 GiB**, driver allocation **29.18 GiB**, at 0.25-second cadence. These counters overlap RSS and must not be added. [Update-4 generations](../../reports/minicpm5_balanced_mask1_chunk01_generation_step0004.json), [update-8 generations](../../reports/minicpm5_balanced_mask1_chunk01_generation_step0008.json), [memory samples](../../reports/minicpm5_balanced_mask1_chunk01_memory_samples.jsonl).

This is training-case memorization evidence. Independent exported-checkpoint generation review and broader development transfer/control interpretation remain separate; no generalization claim follows. Native continuation is qualified, while **full-size uninterrupted-versus-split numerical equivalence is unmeasured**. Exact equivalence is proven only by the tiny CPU experiment. [Native continuation report](../../reports/minicpm5_balanced_mask1_chunk02_report.json), [update-12 generations](../../reports/minicpm5_balanced_mask1_chunk02_generation_step0012.json).

The next D0 capacity milestone is now measured at **actual 256 and 512 encoded tokens**. Each length starts from the sealed original and completes two full-text updates as **1→fresh-process resume 2**, accumulation 8 with all eight answer classes per update. All 381 original tensors receive nonzero gradients; independent saved-checkpoint audits show all 381 changed including embeddings/head at both lengths. Export/reload compared logits match exactly. [256 audit](../../reports/minicpm5_capacity256_checkpoint_audit_20261022.json), [512 audit](../../reports/minicpm5_capacity512_checkpoint_audit_20261022.json), [capacity summary](../../reports/capacity_readiness_experiment_20261022.json).

| Actual length | Sampled peak MPS allocation | Sampled peak driver allocation |
|---|---:|---:|
| 256 tokens | 24.956 GiB | 29.094 GiB |
| 512 tokens | 25.295 GiB | 29.063 GiB |

All four whole-run guard traces passed the declared 40 GiB driver / 512 MiB swap-growth / 15% minimum-free policy, with a three-host-sample sustained criterion. No sampled system swap growth was observed from 1,562.69 MiB. Driver/RSS overlap; sampling can miss transients. These are short **D0-only** capacity results, not sustained training, task quality or D2 fit. Native D2 capacity remains the unfinished part of S0.5.

Existing reproducible entry points remain:

```bash
uv sync --locked --extra neural --extra test
.venv/bin/python scripts/check_core.py --output reports/core_checks_local.json
.venv/bin/python scripts/validate_data.py data/fixtures/train.jsonl data/fixtures/dev.jsonl --require-verified
```

Use new output paths for experiments; exact current balanced-run commands/results belong in the main runbook and execution state. Muon rates and auxiliary AdamW rates are different controls. Audit actual target coverage per effective update, response/EOS supervision and native template prefix before tuning either. A drop dominated by response formatting/EOS can coexist with wrong task answers.

Capacity qualification must retain actual encoded length, source/support/supervised token counts, maximum new tokens, memory sampling cadence, process lifetime and phase events. On unified memory, driver and RSS counters overlap. Keep host memory safeguards and declare stop thresholds before loading. Tiny real CPU checkpoint/next-update equivalence and MPS mixed-optimizer/RNG restoration tests pass; native pretrained fresh-process continuation 8→12 is now qualified, while full-size uninterrupted-versus-split numerical equivalence remains unmeasured. Checkpoints are supported only between completed optimizer updates with gradients cleared. Optimizer-state serialization alone does not establish the full RNG/schedule/cursor and native model continuation contract; mid-accumulation resume is unsupported.

## S0.1 — Preserve the reproducible native source and environment

**Status:** `measured_short_scope`. **Dependencies:** none; can start independently.

Reuse the existing Git history, Python lock, sealed checkpoint and source template. Capture code revision and hardware/framework inventory before each new run; never substitute a sibling checkpoint.

- **Artifact:** uv.lock; configs/minicpm5_sealed_lock.local.json; reports/hardware_macos_m2_ultra.json; reports/minicpm5_mps_experiment_20261022.json.
- **Interface:** Source lock -> downloader/sealer -> LlamaMemoryAdapter and native evaluator; one exact source revision shared by all arms.
- **Test gate:** Existing 55-token native baseline and full D0 evidence is preserved. A clean locked setup can run offline checks, validate fixtures and reproduce native load with the same model/template hashes.
- **Resources:** M2 Ultra 64 GiB; Python 3.12.14; torch 2.14.1/transformers 5.20.2; one model resident. No reacquisition or environment upgrade needed for this task.
- **Failure policy:** Stop on seal or version mismatch; identify the changed file/environment before proceeding. Preserve original reports and user changes.
- **Next smallest task:** Reuse the current sealed source, template and locked environment; bind their hashes in the next augmented-generation/learning report rather than repeating acquisition.

## S0.2 — Diagnose the constant-zero failure

**Status:** `complete`. **Dependencies:** S0.1.

Completed the historical sampling/target-mask/template diagnosis and preserved the short constant-zero negative. A declared bundled intervention now learns the eight training answers; balancing, accumulation, rates, warmup and decay changed together, so no individual causal factor is isolated.

- **Artifact:** reports/minicpm5_mps_preflight_report.json; reports/minicpm5_balanced_mask1_chunk01_report.json; data/diagnostics/tiny_overfit_mask1/selection_manifest.json; reports/minicpm5_balanced_overfit_experiment_20261022.json.
- **Interface:** Raw train export + encoded examples + strict generation report -> diagnosis; evaluator must preserve raw completions and EOS/cap reasons.
- **Test gate:** Historical four supervised tokens and constant-zero control are accounted for, and mask/template/target-coverage checks are inspected. Eight-class training generation now succeeds under the bundle. Exact causal attribution to balancing, rate, accumulation, warmup or decay remains unisolated.
- **Resources:** CPU file/token audits first; at most one short source/D0 MPS evaluation if an unresolved prefix or mask hypothesis requires it.
- **Failure policy:** Preserve the historical failure and do not infer that two updates demonstrate an intrinsic Muon defect. Any future causal-factor claim requires a separate controlled ablation; the successful bundled remediation does not supply it.
- **Next smallest task:** Retain the bundled-fix caveat and per-update answer coverage in the upcoming D1 tiny-overfit diagnostic.

## S0.3 — Run balanced full-text tiny overfit and verify generated answers

**Status:** `tiny_overfit_passed`. **Dependencies:** S0.2.

Accepted the native full-text tiny-overfit run from the sealed original: 8/8 strict training answers and eight classes 0–7 at updates 4 and 8. Every accumulated update covers each answer once. The declared bundle uses accumulation 8, matched Muon/auxiliary AdamW base rates 1e-4, warmup 2 and zero decay; the 64-update cap stopped at 8. Native checkpoint/export/reload and measured 55-token memory fit pass; broader transfer remains unqualified.

- **Artifact:** data/diagnostics/tiny_overfit_mask1/train.jsonl; data/diagnostics/tiny_overfit_mask1/selection_manifest.json; reports/minicpm5_balanced_mask1_chunk01_generation_step0004.json; reports/minicpm5_balanced_mask1_chunk01_generation_step0008.json; reports/minicpm5_balanced_mask1_chunk01_report.json; reports/minicpm5_balanced_mask1_chunk01_memory_samples.jsonl; reports/minicpm5_balanced_overfit_experiment_20261022.json.
- **Interface:** Existing eight-field trainer export -> full regime -> native saved export -> unchanged strict generation parser. Record per-update target coverage, effective batch and all optimizer group rates.
- **Test gate:** Passed the scoped tiny-overfit gate: all eight training answer classes 0–7 correct at both updates 4 and 8, full-text scope, finite gradients, accepted checkpoint/export/reload with zero compared-logit difference, and sampled MPS memory fit. Native continuation is separately qualified under S0.4. Independent export-generation checks and broader development transfer/control conclusions belong to the final result record, not this memorization gate.
- **Resources:** Measured MPS full-text run: microbatch 1, accumulation 8, actual length 55, checkpointing and eight completed updates under declared 64 cap. Sampled peak MPS allocation 24.88 GiB and driver 29.18 GiB at 0.25-second cadence; counters overlap RSS and are not additive. This is not 256-token capacity.
- **Failure policy:** Retain the historical collapse and all control/development failures. A later artifact or transfer failure narrows the conclusion without erasing the 8/8 training observation. Do not credit any single variable in the bundled intervention without ablation.
- **Next smallest task:** Use this eight-class diagnostic for D1 after prompt-fixed augmented generation parity; retain its memorization-only interpretation.

## S0.4 — Make full-text training resumable

**Status:** `native_continuation_qualified`. **Dependencies:** S0.3.

Qualified full-size pretrained fresh-process continuation from completed update 8 through 12 with strict training-contract and byte-receipt acceptance, restored mixed optimizer/RNG/cursor/schedule state, nonzero gradients in all 381 original tensors, 8/8 training generations with eight distinct answers at 12, and checkpoint/export/reload maximum logit difference 0. Checkpoints remain completed-update boundaries with gradients cleared; mid-accumulation resume is unsupported.

- **Artifact:** src/replica_lab/training_state.py; src/replica_lab/checkpointing.py; tests/test_checkpointing.py; tests/test_training_state.py; reports/tests_balanced_training_code.xml; reports/minicpm5_balanced_mask1_chunk02_report.json; reports/minicpm5_balanced_mask1_chunk02_generation_step0012.json; reports/minicpm5_balanced_overfit_experiment_20261022.json; reports/capacity_readiness_experiment_20261022.json.
- **Interface:** Completed-update trainer boundary -> atomic model/runtime checkpoint with byte receipt -> contract-bound optimizer/model/RNG/data/schedule restore. Native inference export is separate; an interrupted partial accumulation restarts from the last completed checkpoint.
- **Test gate:** Native short continuation 8→12 is qualified with contract/receipt validation, full-scope finite updates, correct next cursor/schedule and retained eight-class training answers; export/reload compared logits match exactly. Tiny CPU uninterrupted-versus-resumed exact equivalence and MPS optimizer/RNG tests pass. No full-size uninterrupted-versus-split numerical equivalence experiment was run, so no such claim is made.
- **Resources:** Measured one fresh-process native MPS continuation of four completed updates after checkpoint 8. Save/restore/runtime memory is recorded separately in the continuation report; use one full-size process at a time.
- **Failure policy:** A weights-only reload is not resume. Resume only at completed-update checkpoints; no mid-accumulation gradient buffer recovery is claimed. Quantify any MPS numerical tolerance while keeping cursor/schedule compatibility exact.
- **Next smallest task:** Reuse the completed-update checkpoint path for D1 and later D2. Full-size uninterrupted-versus-split numerical equivalence remains a separate unmeasured experiment.

## S0.5 — Measure genuine 256-token and 512-token full training capacity

**Status:** `D0_actual_256_512_qualified_D2_capacity_pending`. **Dependencies:** S0.3, S0.4.

Qualified actual 256-token and 512-token D0 full-text workloads, each from the sealed original with two updates split 1→fresh-process resume 2. Accumulation 8 covers all eight answers in every update. All 381 original tensors have nonzero gradients and independent saved-checkpoint changes, and export/reload maximum compared-logit difference is 0. Guard traces pass. The corresponding D2 optimizer/update/capacity path is still pending, so the complete stage is not finished.

- **Artifact:** reports/capacity_readiness_experiment_20261022.json; reports/minicpm5_capacity256_chunk01_report.json; reports/minicpm5_capacity256_chunk02_report.json; reports/minicpm5_capacity512_chunk01_report.json; reports/minicpm5_capacity512_chunk02_report.json; reports/minicpm5_capacity256_checkpoint_audit_20261022.json; reports/minicpm5_capacity512_checkpoint_audit_20261022.json; planned: matching D2 actual 256/512 capacity reports.
- **Interface:** Tokenizer/template length audit -> bounded trainer -> profiler phase events. Compare allocated/driver/RSS counters with host pressure/swap; identify overlap and sampling cadence.
- **Test gate:** D0 portion passes at both real encoded lengths: full 381-tensor gradients/changes including embeddings/head, resumed second update, exact compared-logit reload and no observed guard breach. Two short updates do not establish sustained training or task quality. Keep the overall gate partial until native D2 performs optimizer updates and the same declared length/guard/restart checks.
- **Resources:** Measured D0 live/driver sampled peaks: 256 tokens 24.956/29.094 GiB; 512 tokens 25.295/29.063 GiB. Driver cap 40 GiB, swap-growth cap 512 MiB, minimum host-free 15%, sustained-host criterion 3 samples. All four whole traces passed with no observed growth from 1,562.69 MiB system swap. Samples can miss transients; driver and RSS overlap.
- **Failure policy:** Stop and preserve the failing profile. Reduce length/accumulation or add tested residency changes without freezing hidden layers. Do not extrapolate from the historical 55-token result.
- **Next smallest task:** Qualify prompt-fixed augmented generation, run D1 tiny overfit, then execute a bounded full-scope D2 capacity/update preflight to finish the remaining S0.5 portion.
