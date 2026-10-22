# S2 — native D0/D1/D2 and matched controls

Reviewed **22 October 2026, America/Chicago**. See the [master roadmap](00_Master_Roadmap.md), [current execution state](../../execution_state.md) and [dated primary-source review](07_Research_Evidence_20261022.md). Gates and budgets are proposed unless a linked existing report establishes execution. Planned artifacts below do not claim files already exist.

Native **actual 256-token forward/backward/export and fixed-context uncached generation qualification** now pass for the reader/workspace wrapper. D1 has no trainable originals or original gradients; D2 backward reaches all 381 originals. The sidecars add **895,490 parameters in 48 tensors**. Both null branches match native generated tokens and all eight steps of next-token logits exactly; nonzero sidecars affect logits and reload exactly. [Forward/backward report](../../reports/minicpm5_native_adapter_qualification256_report.json), [generation result](../experiments/20261022_Augmented_Generation.md).

The bounded **D1 eight-class memorization gate passes** under the [declared protocol](../experiments/20261022_D1_Fixed_Context_Protocol.md): a fresh 1e-2 arm reaches 8/8 at updates 24 and 28, after the preserved 1e-3 arm fails through 64. Independent saved-export evaluation confirms train 8/8 but dev only 2/16. All 381 original tensors remain unchanged. This does not establish useful memory or generalization. [Consolidated D1 results](../../reports/native_augmented_D1_experiment_20261022.json).

**D2 full-augmented capacity passes at actual 256/512 tokens** with the same 64-wide, four-slot, two-step configuration: two updates per length, fresh-source 1→fresh-process resume 2, full 381-original/48-new optimizer ownership and changed-tensor audits, exact export/reload and clean guard replay. The fresh D2 tiny run now produces 8/8 strict answers/eight distinct/all EOS at updates 4 and 8, then stops; independent saved-export scores are train 8/8 and reused dev 14/16, all EOS/no caps, and final scope/runtime audits pass. Bounded S2.3 is qualified. **S2.4 useful architecture gain remains pending**. [D2 results](../experiments/20261022_D2_Results.md).

B0 is the frozen original. B1 is the original with strong source-text memory and the same tools. D0 trains all original text parameters; D1 trains only new modules with originals frozen; D2 trains both. D3/D4 LoRA follow only as explicitly labelled extensions after these minimum comparisons. Every trained arm receives the same eligible source information and target data; duplicated source text remains present in the first equal-information comparison. Compression is a later separate experiment.

The feed-forward parameter/compute control and inference budget distinguish useful recurrence from simply adding capacity or steps. Rate search is fair per arm; warmup and failed/search runs count. A selected plain D0 checkpoint is a valid simpler developmental parent if added components fail their useful-gain gate. Preserve that negative and continue the complete learned-loop objective.

## S2.1 — Qualify native wrapper bypass and generation

**Status:** `qualified_native256_fixed_context_generation`. **Dependencies:** S0.3, S1.4.

Qualified native MiniCPM5 null-logit parity, branch sensitivity, checkpoint context lifetime, causal target separation, source-feature refresh and export/reload at 256 tokens. Fixed-context generation now matches native uncached tokens and 1,044,480 next-token logits across eight decode steps in each D1/D2 arm, with zero maximum difference. A separate three-token EOS-disabled probe checks context reuse. Writer/workspace each build once; input IDs remain unchanged, modes restore and context releases. Independent nonzero memory/workspace projections change logits; both nonzero sidecars preserve tokens/logits after reload. These are compatibility probes with no optimizer update or quality claim.

- **Artifact:** reports/minicpm5_native_adapter_qualification256_report.json; reports/minicpm5_augmented_generation_qualification256_report.json; paired memory samples and [generation record](../experiments/20261022_Augmented_Generation.md).
- **Interface:** LlamaMemoryAdapter -> original complete decoder block + per-batch support/workspace context -> strict evaluator; native names and checkpoint compatibility preserved.
- **Test gate:** Native forward/backward/export and original-prompt-fixed uncached generation pass at the bounded tested configuration. KV caching, concurrency, useful memory benefit and optimizer learning are separate gates.
- **Resources:** This original compatibility probe measured one native MPS batch with 64-wide memory/workspace, 4 slots, 2 workspace steps and 895,490 new parameters / 48 tensors. Later optimizer/capacity evidence belongs to S2.2/S2.3.
- **Failure policy:** Keep base path intact and repair parity/context lifetime before training new modules. Do not optimize KV caching until correctness holds.
- **Next smallest task:** Retain this uncached fixed-context route as the saved-export D2 generation reference; KV caching is a separate unqualified optimization.

## S2.2 — Overfit native D1 new modules with frozen original tensors

**Status:** `qualified_native_D1_tiny_memorization_scope_resume`. **Dependencies:** S2.1, S0.4.

Trained all 48 new tensors / 895,490 parameters with originals frozen on the balanced eight-class diagnostic. The first 1e-3 arm completed 64 updates with 0/8 at every training check. The conditionally predeclared fresh 1e-2 arm achieved 8/8 and eight distinct EOS-terminated answers at updates 24 and 28, then stopped. Every new tensor has a nonzero gradient by update 2; all 48 change during continuation. All 381 originals exactly equal the sealed source promoted to FP32. Both arms resume from update 8 in a new process and reload with zero compared-logit difference. External GPU evaluation of the successful export yields train 8/8 and reused synthetic dev 2/16, all EOS, at cap 256. This is memorization, not a useful-gain result.

- **Artifact:** reports/native_augmented_D1_experiment_20261022.json; reports/minicpm5_D1_checkpoint_audit_20261022.json; reports/validation_D1_runtime_20261022.json; [D1 experiment](../experiments/20261022_D1_Diagnostic.md); raw trainer/generation/memory/guard records.
- **Interface:** modules_only parameter registry -> mixed optimizer owns each new tensor once -> augmented evaluator. Frozen original tensor hashes remain stable.
- **Test gate:** Bounded scope, later upstream gradients, nonconstant correct training generations, restart/reload and unchanged original weights pass. The failed rate and poor dev transfer remain explicit; architecture and retention gates remain open.
- **Resources:** Four serial MPS FP32 processes, 55 actual tokens, B8; 64 + 28 updates across two rates. Successful-arm sampled live/driver peaks 9.467/11.113 GiB; optimizer state 1,029,824 bytes. Total training-update time across both arms is 151.002 s; complete trainer processes total 445.953 s, including both rate searches and checkpoint/generation overhead.
- **Failure policy:** Diagnose gradients, null initialization and frozen-autograd paths. Do not unfreeze layers while retaining the D1 label.
- **Next smallest task:** Use completed D2 evidence in the preregistered S2.4 control study; retain D1's poor transfer and all search costs.

## S2.3 — Overfit and profile native D2 full augmented training

The S0.5 dependency originally admitted this work through its completed D0 subgate. The D2 256/512 capacity output now also passes, so S0.5 is complete. This note preserves the original execution order without changing stage IDs or dependencies.

**Status:** `qualified_bounded_scope_memorization_capacity_resume`. **Dependencies:** S2.1, S0.4, S0.5.

Native D2 completes two full-augmented updates at both actual 256 and 512 tokens, with a new-process 1→2 continuation per length. All 381 original tensors receive nonzero gradients and show saved changes including embeddings/head. Added gradients open 2→48 by update 2, all 48 sidecars change in continuation, and native/sidecar reload matches exactly. Independent capacity runtime/optimizer/history/guard and saved-value audits pass. The historical 256 code contract stays distinct from the storage-aware 512 contract. The fresh-source balanced tiny run then completes 0→4→8 at declared original/new rates 1e-4/1e-3. Checks 4 and 8 each give 8/8 strict correct, eight distinct, all EOS/no caps; it stops at 8 under the declared rule. Independent tiny original/sidecar and runtime/guard audits pass; saved-export train/dev generation gives 8/8 and 14/16, all EOS/no caps at cap256. All 381 originals change at updates 4 and 8; all 48 sidecars change over 4→8. This closes bounded scope/memorization/capacity/restart only; selected D0 and D1 artifacts remain preserved.

- **Artifact:** [D2 protocol](../experiments/20261022_D2_Protocol.md); configs/d2_resume_plan_20261022.json; reports/validation_D2_capacity256_20261022.json; reports/validation_D2_capacity512_20261022.json; native/sidecar saved audits and archived raw evidence linked in [D2 results](../experiments/20261022_D2_Results.md); reports/validation_D2_tiny_20261022.json; reports/validation_D2_saved_answers_20261022.json; reports/native_augmented_D2_experiment_20261022.json; archived tiny trainer/generation/scope evidence.
- **Interface:** full_augmented registry -> original/new named groups -> native wrapper and augmented export/reload; full-text scope includes embeddings/head.
- **Test gate:** Capacity scope, two-update continuation and bounded memory pass at both lengths. Tiny memorization also passes: two consecutive 8/8 generation checks/eight distinct/all EOS at 4 and 8, independent saved train 8/8 and dev 14/16 at cap256, complete saved scope and fresh-process continuation. No rate escalation or budget extension occurred. Useful gain remains S2.4, not a conclusion from NLL or tiny memorization.
- **Resources:** Serial MPS FP32, microbatch 1, accumulation 8 and checkpointing. Capacity live/driver peaks are 26.751/29.072 GiB at 256 and 26.876/30.080 GiB at 512; optimizer state 6,266,291,904 bytes. Tiny actual55/ceiling256 stops at8: 160.7403 unique update seconds and 299.5567 whole-profile seconds; live/driver peaks 26.6532/29.1951 GiB. Generation every4 and checkpoints every16 plus forced boundaries; clean sampled guards. Keep 110 GiB prelaunch tiny disk headroom.
- **Failure policy:** Stop on pressure/scope/parity failure. Use a smaller declared module configuration or tested residency change without freezing original layers.
- **Next smallest task:** Prepare the S2.4 equal-information/parameter/compute/retention matrix after S1.5; freeze its grouped data, margins and per-arm rate/search budgets before training. Do not treat D2 dev 14/16 as an architecture advantage over D0.

## S2.4 — Run one-seed D0/D1/D2 and matched capacity/compute screen

**Status:** `pending`. **Dependencies:** S2.2, S2.3, S1.5.

Run B0/B1 capability/evidence references, D0 ordinary full tuning, D1 modules-only and D2 full augmented from declared parents. Add memory-disabled, workspace-disabled, shuffled-evidence and parameter/compute-matched feed-forward controls.

- **Artifact:** planned: preregistered arm and hyperparameter manifest; planned: one-seed task/retention/calibration/cost matrix.
- **Interface:** Same eligible release/evidence/tools/targets -> per-arm fair rate screen -> fixed generation evaluator. Warmup tokens and search costs included; data-only effects distinguished from architecture effects.
- **Test gate:** Report equal-token and equal-total-compute views, actual scopes, confidence/retention and wall-clock/memory overhead. No claim from NLL alone, unequal information or unmatched recurrent inference budget.
- **Resources:** One shortlisted initial architecture, one seed, bounded steps per arm; serialize full-model MPS jobs. All-AdamW is an explicit separately tuned comparison only if fit/budget allows.
- **Failure policy:** If D2 ties D0, report ordinary training benefit and remove/redesign expensive modules. Do not stack optional memory systems to conceal a negative.
- **Next smallest task:** Freeze arm matrix, trainable names, parents and useful gain/regression margins before running the screen.

## S2.5 — Repeat shortlisted mechanisms and select developmental checkpoint

**Status:** `pending`. **Dependencies:** S2.4.

Repeat promising arms with three seeds and independent held-out families. Analyze errors and compute/parameter controls; select or reject the interface based on fixed task and retention criteria.

- **Artifact:** planned: multi-seed clustered comparison report; planned: selected developmental manifest or explicit negative decision.
- **Interface:** One-seed screen -> preregistered shortlist -> independent evaluator -> versioned base/extensions/source-encoder compatibility manifest.
- **Test gate:** Useful gain beyond controls with acceptable uncertainty/retention and feasible measured costs; hard causality/identity invariants all pass. Otherwise report inconclusive or negative. No final audit informs selection.
- **Resources:** Three seeds for shortlisted trained arms; use measured token/time budgets. Replenish independent promotion families when exposure limits are reached.
- **Failure policy:** Do not mark architecture complete because budget ends. Freeze a well-characterized simpler baseline if appropriate and preserve the research negative.
- **Next smallest task:** Choose one promising mechanism or a simpler D0 parent and record why its held-out evidence earns freezing.
