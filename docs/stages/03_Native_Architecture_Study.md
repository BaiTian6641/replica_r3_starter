# S2 — native D0/D1/D2 and matched controls

Reviewed **22 October 2026, America/Chicago**. See the [master roadmap](00_Master_Roadmap.md), [current execution state](../../execution_state.md) and [dated primary-source review](07_Research_Evidence_20261022.md). Gates and budgets are proposed unless a linked existing report establishes execution. Planned artifacts below do not claim files already exist.

Native **actual 256-token forward/backward/export qualification** now passes for the reader/workspace wrapper: null-logit parity, branch sensitivity, causal target separation, source-feature refresh and checkpoint context lifetime. D1 has no trainable originals or original gradients; D2 has nonzero gradients in all 381 originals. The sidecars add **895,490 parameters in 48 tensors**. Initial gradients in the two zero-initialized output projections are expected; no optimizer update, upstream post-update learning or augmented generation has been measured. **S2.1 remains partial.** [Native report](../../reports/minicpm5_native_adapter_qualification256_report.json).

The next concrete task is **prompt-fixed augmented generation parity**, followed by **D1 eight-class tiny overfit**. Keep observed source and original prompt context fixed while decoding answers; re-prefill is acceptable before KV-cache optimization. D2 optimizer/capacity and the controlled architecture study remain later gates. Use the measured 64-wide, four-slot, two-step configuration before architecture search.

B0 is the frozen original. B1 is the original with strong source-text memory and the same tools. D0 trains all original text parameters; D1 trains only new modules with originals frozen; D2 trains both. D3/D4 LoRA follow only as explicitly labelled extensions after these minimum comparisons. Every trained arm receives the same eligible source information and target data; duplicated source text remains present in the first equal-information comparison. Compression is a later separate experiment.

The feed-forward parameter/compute control and inference budget distinguish useful recurrence from simply adding capacity or steps. Rate search is fair per arm; warmup and failed/search runs count. A selected plain D0 checkpoint is a valid simpler developmental parent if added components fail their useful-gain gate. Preserve that negative and continue the complete learned-loop objective.

## S2.1 — Qualify native wrapper bypass and generation

**Status:** `partial_native256_forward_backward_qualified_generation_pending`. **Dependencies:** S0.3, S1.4.

Qualified native MiniCPM5 reader/workspace null-logit parity, nonzero branch sensitivity, checkpoint backward/context lifetime, target/workspace causal separation, source-feature refresh and export/reload on an actual 256-token batch. D1 has zero trainable original parameters/no original gradients; D2 has nonzero gradients in all 381 originals. The 895,490 new parameters span 48 tensors; only the two zero-initialized output projections have nonzero first-backward gradients as expected. No optimizer update or augmented generation ran.

- **Artifact:** reports/minicpm5_native_adapter_qualification256_report.json; reports/minicpm5_native_adapter_qualification256_memory_samples.jsonl; planned: prompt-fixed augmented generation parity report.
- **Interface:** LlamaMemoryAdapter -> original complete decoder block + per-batch support/workspace context -> strict evaluator; native names and checkpoint compatibility preserved.
- **Test gate:** Forward/backward/export portion passes at native dimensions and actual 256 tokens, with scoped D1/D2 gradients and causal/context checks. Entire S2.1 remains partial until augmented generation holds prompt-derived workspace/support fixed across answer decoding and matches native generation on the null path. Initial projection gradients alone do not establish upstream learning or successful optimizer updates.
- **Resources:** Measured one native MPS batch with 64-wide memory/workspace, 4 slots, 2 workspace steps and 895,490 new parameters / 48 tensors. No training campaign or native D1/D2 optimizer capacity qualification is claimed.
- **Failure policy:** Keep base path intact and repair parity/context lifetime before training new modules. Do not optimize KV caching until correctness holds.
- **Next smallest task:** Implement and verify prompt-fixed augmented generation parity against the native null path before D1 tiny overfit.

## S2.2 — Overfit native D1 new modules with frozen original tensors

**Status:** `pending_native_D1_optimizer_overfit`. **Dependencies:** S2.1, S0.4.

Train only the 48 new reader/writer/workspace tensors on the balanced eight-class diagnostic after prompt-fixed generation parity. Native initial D1 backward has qualified frozen-original ownership and two nonzero output-projection gradients; later upstream gradients, optimizer updates, generated overfit answers and checkpoint continuation remain unrun.

- **Artifact:** reports/minicpm5_native_adapter_qualification256_report.json; planned: D1 scope, gradient and generation reports; planned: D1 native checkpoint/export/reload audit.
- **Interface:** modules_only parameter registry -> mixed optimizer owns each new tensor once -> augmented evaluator. Frozen original tensor hashes remain stable.
- **Test gate:** New output projections receive initial gradients and upstream branch receives later gradients; originals do not change. Nonconstant correct diagnostic generations and restart/reload work or a clearly recorded D1 learning failure.
- **Resources:** One MPS run with matched diagnostic examples and explicit new-module rates; new-module training still stores frozen original weights and backward activations.
- **Failure policy:** Diagnose gradients, null initialization and frozen-autograd paths. Do not unfreeze layers while retaining the D1 label.
- **Next smallest task:** After S2.1 prompt-fixed generation parity, run a bounded D1 tiny-overfit update sequence with original hashes fixed, first/later new-gradient audits and generated answer checks.

## S2.3 — Overfit and profile native D2 full augmented training

The S0.5 dependency here means its completed D0 256/512-token subgate. D2 capacity is an output of this stage and then closes S0.5's remaining D2 portion; it is not a prerequisite that would block itself.

**Status:** `pending_native_D2_optimizer_and_capacity`. **Dependencies:** S2.1, S0.4, S0.5.

Native D2 initial forward/backward is qualified with all 381 original gradient tensors and the same 48 new tensors, but no optimizer update has run. After D1 diagnostic/generation readiness, perform full-augmented update/capacity qualification and then balanced overfit, with separate original/new Muon/AdamW rates and declared parent.

- **Artifact:** reports/minicpm5_native_adapter_qualification256_report.json; planned: D2 changed-tensor and optimizer scope reports; planned: D2 capacity/generation/restart reports.
- **Interface:** full_augmented registry -> original/new named groups -> native wrapper and augmented export/reload; full-text scope includes embeddings/head.
- **Test gate:** All intended groups receive finite gradients and verified changes; none omitted or double-owned. Diagnostic generation, causal inputs, memory bounds and restart pass. Full D0 fit never substitutes for this D2 result.
- **Resources:** One native MPS job; microbatch 1/checkpointing initially. Count added modules, support encodings, workspace steps, logits and optimizer temporaries.
- **Failure policy:** Stop on pressure/scope/parity failure. Use a smaller declared module configuration or tested residency change without freezing original layers.
- **Next smallest task:** Use the measured 895,490-parameter/48-tensor additions for a bounded D2 optimizer/update capacity preflight; do not substitute the successful D0 length profile.

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
