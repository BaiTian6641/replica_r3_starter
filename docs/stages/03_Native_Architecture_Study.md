# S2 — native D0/D1/D2 and matched controls

Reviewed **22 October 2026, America/Chicago**. See the [master roadmap](00_Master_Roadmap.md), [current execution state](../../execution_state.md) and [dated primary-source review](07_Research_Evidence_20261022.md). Gates and budgets are proposed unless a linked existing report establishes execution. Planned artifacts below do not claim files already exist.

Do not infer pretrained D1/D2 success from random tiny-Llama contracts. Native null-path and generation parity come first, then actual new-module gradients and native D2 full scope. Start with the existing small 64-wide memory/workspace, four slots and two steps before architecture search.

B0 is the frozen original. B1 is the original with strong source-text memory and the same tools. D0 trains all original text parameters; D1 trains only new modules with originals frozen; D2 trains both. D3/D4 LoRA follow only as explicitly labelled extensions after these minimum comparisons. Every trained arm receives the same eligible source information and target data; duplicated source text remains present in the first equal-information comparison. Compression is a later separate experiment.

The feed-forward parameter/compute control and inference budget distinguish useful recurrence from simply adding capacity or steps. Rate search is fair per arm; warmup and failed/search runs count. A selected plain D0 checkpoint is a valid simpler developmental parent if added components fail their useful-gain gate. Preserve that negative and continue the complete learned-loop objective.

## S2.1 — Qualify native wrapper bypass and generation

**Status:** `pending`. **Dependencies:** S0.3, S1.4.

Test the reader/workspace wrapper at real MiniCPM5 dimensions with null branches against native logits and generation. Implement augmented inference with correctly reconstructed support/prompt context; re-prefill is acceptable before cache optimization.

- **Artifact:** planned: native bypass/parity report; planned: augmented generation path and context-lifetime tests.
- **Interface:** LlamaMemoryAdapter -> original complete decoder block + per-batch support/workspace context -> strict evaluator; native names and checkpoint compatibility preserved.
- **Test gate:** Null path matches native within declared tolerance; no target text in workspace, no source feature reuse after writer updates, and checkpoint recomputation has no side effects. Context survives backward and is released afterward.
- **Resources:** CPU/tiny tests followed by one bounded native MPS sample. Use initial 64-wide, 4-slot, 2-step configuration before dimension search.
- **Failure policy:** Keep base path intact and repair parity/context lifetime before training new modules. Do not optimize KV caching until correctness holds.
- **Next smallest task:** Run a real-dimension null-branch forward and compare native logits under identical masks/template.

## S2.2 — Overfit native D1 new modules with frozen original tensors

**Status:** `pending`. **Dependencies:** S2.1, S0.4.

Train only reader/writer/workspace modules on the balanced diagnostic with original parameters frozen. Initialize branch output projections to zero and upstream factors normally; keep frozen downstream layers in the autograd graph.

- **Artifact:** planned: D1 scope, gradient and generation reports; planned: D1 native checkpoint/export/reload audit.
- **Interface:** modules_only parameter registry -> mixed optimizer owns each new tensor once -> augmented evaluator. Frozen original tensor hashes remain stable.
- **Test gate:** New output projections receive initial gradients and upstream branch receives later gradients; originals do not change. Nonconstant correct diagnostic generations and restart/reload work or a clearly recorded D1 learning failure.
- **Resources:** One MPS run with matched diagnostic examples and explicit new-module rates; new-module training still stores frozen original weights and backward activations.
- **Failure policy:** Diagnose gradients, null initialization and frozen-autograd paths. Do not unfreeze layers while retaining the D1 label.
- **Next smallest task:** Audit optimizer membership and a first/second-step gradient trace through the native frozen decoder.

## S2.3 — Overfit and profile native D2 full augmented training

**Status:** `pending`. **Dependencies:** S2.1, S0.4, S0.5.

Train all original parameters plus the same D1 modules from a declared parent, with separate original/new Muon/AdamW rates. Repeat the balanced diagnostic and actual 256/512-token capacity checks for the chosen module size.

- **Artifact:** planned: D2 changed-tensor and optimizer scope reports; planned: D2 capacity/generation/restart reports.
- **Interface:** full_augmented registry -> original/new named groups -> native wrapper and augmented export/reload; full-text scope includes embeddings/head.
- **Test gate:** All intended groups receive finite gradients and verified changes; none omitted or double-owned. Diagnostic generation, causal inputs, memory bounds and restart pass. Full D0 fit never substitutes for this D2 result.
- **Resources:** One native MPS job; microbatch 1/checkpointing initially. Count added modules, support encodings, workspace steps, logits and optimizer temporaries.
- **Failure policy:** Stop on pressure/scope/parity failure. Use a smaller declared module configuration or tested residency change without freezing original layers.
- **Next smallest task:** Generate the exact native D2 parameter inventory and compare it against D0 originals plus D1 additions.

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
