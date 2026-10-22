# Coding-agent instructions — TAIS Obsidian Full Chain R2

## Mission and scope precedence

Implement a full-chain semi-cognitive learning system and personal-replica research platform. **Do not reduce this project to RAG, a memory plugin, persona imitation, or a collection of disconnected demonstrations.** Memory is a subsystem. The measurable chain is goal → reasoning → action/inquiry → verification → experience → learning proposal → adaptation → independent evaluation → promotion/rollback → improved later behavior.

R2 supersedes R1’s scope and milestone order. Read `00_README.md`, documents 01–05, and this instruction. Retain R1’s numerical, cache, provenance and permission safeguards. Treat any prior code and reported tests as unverified until inspected in the actual repository.

## First work session: C0–C2, then register the full-tuning study

1. Inspect Git status, branch, remotes, uncommitted work, source tree, tests, raw experiment reports and current execution notes. Preserve user changes. Use an existing repository and an isolated branch/worktree. Do not invent a remote or start a replacement codebase merely because the old code is unfamiliar.
2. Inventory both workstations read-only. Identify GPUs by UUID; record per-device VRAM, supported dtypes/kernels, CPU RAM, disk, topology, and Optane mode where applicable. Do not assume the new small workstation equals the July Blackwell system. Do not change drivers automatically.
3. Resolve the exact four requested model IDs, pin model/tokenizer/processor/framework revisions, and count actual unique parameters. Null revision values in this package mean **unresolved design configuration**, not permission for an untracked run. Do not substitute a Base/SFT sibling.
4. Establish the unmodified MiniCPM5 baseline and explicit support for text-only versus full-VLM scopes. Record generation/cache behavior and a fixed capability set before adding modules.
5. Build a small deterministic task family with an independent oracle, hidden tests, support/query separation and chronological event IDs. A fixed-bitwidth expression evaluator or finite-state task is an acceptable first domain. The model cannot edit the oracle, held-out tests, permissions or budget.
6. Implement the smallest complete loop using simple modules: explicit goal/workspace, action policy, evidence access, verification, typed experience record, ordinary candidate-adapter training, independent evaluation and atomic promotion. Real components must replace mocks in any integration claim. This early full-loop implementation is a baseline, not a substitute for the neural cognitive experiments.
7. Add one identity-initialized internal reader and a small bounded workspace/action core with explicit trainable parameters. Preserve native backbone cache indices and multimodal preparation. Test independent gates, null paths, causality, gradient flow and save/load before training.
8. Register D0 original full fine-tuning, D1 augmented modules-only, and D2 augmented full fine-tuning as the minimum development comparison. D3 augmented LoRA and D4 original LoRA follow as budget allows. Full fine-tuning is a central scientific comparison, not a forbidden operation in the laboratory.
9. Begin the replica data/split/schema work in parallel, using only consented authentic records. Do not synthesize a biography or treat prior assistant-written answers as the human’s authentic preferences.
10. Update `execution_state.md` after each work package with commit, commands, raw outputs, unresolved failures, tested modes, and next smallest task.

## Developmental versus deployed weights

During D0/D2, explicitly selected original weights are trainable. After developmental training, freeze a versioned foundation and run separate post-freeze learning cycles. Deployable updates initially target memory, fast state, controller, skill adapters and other authorized components. A small-model full-tuning success does not prove a frozen 35B model can use the same mechanism without adaptation.

Never claim `full_vlm` when the vision path was inactive. Never silently freeze layers to make a full-tuning run fit. Change and document the arm, or use a validated offload/sharding implementation. QLoRA is not full fine-tuning.

## Scientific acceptance

Compare against ordinary full fine-tuning, strong text memory, fixed-policy learning and compute-matched reasoning. Measure new-instance transfer, prior-task retention, calibration, inquiry efficiency, learning cost and replica fidelity separately. A larger archive, lower inner loss, rising confidence or unchanged base hash is not proof of self-improvement.

Separate development data, meta-adaptation support/query, candidate-selection validation and sealed audit tests. Do not feed sealed audit questions or answers to the learner, writer, curriculum generator or compiler. Repeated release gates need a rotating audit strategy; do not silently train on the acceptance suite.

Use at least the first 10 chronological learning cycles before describing a longitudinal improvement result. This is a proposed initial protocol, not an assurance that ten cycles suffice. Include rejection/rollback events and fixed-update controls. Report negative and inconclusive results.

## Gradient, state, and runtime contracts

- Frozen downstream layers may still need input gradients. Do not wrap the whole model in no-grad when intermediate modules are trainable.
- Jointly trained writer outputs remain differentiable through later task loss. Cached features require an immutable producer fingerprint.
- Fast-state inner updates have explicit functional state and ownership; no cross-example/session reuse. Preserve outer gradients where the specified meta-learning method requires them.
- Checkpoint-recomputed forwards cannot commit records, update persistent counters, or mutate shared state. Commit idempotently outside recomputation.
- Shared-depth fast memory reads a chunk-start snapshot; current-chunk writes become visible only after that chunk’s outputs are fixed.
- Model/interface/adapter revisions invalidate dependent representations, caches and KAL calibration unless a tested migration says otherwise.
- Memory records and model text cannot grant tool permissions or change the evaluation policy.

## Replica contracts

Preserve `source_snapshot`, `source_current`, `parallel_replica`, and `assistant_for_source` modes. The human source’s life events, the replica’s events, and hypotheses about the source are different record types. An authentic correction may update a source profile; the replica’s own new experience must not silently rewrite human history.

Evaluate conditional decisions and tradeoffs, not just voice. Keep the source person’s test–retest variability and legitimate preference changes visible. Do not force perpetual agreement or optimize affection/approval as a proxy for fidelity.

## Work allocation and continuation

One integration owner controls interfaces, state ownership and milestone acceptance. Separate workers can handle model integration/training, verifier/evaluation, memory/cognition, and replica provenance. Give each an isolated file/worktree scope. A reviewer independently checks leakage, gradient ownership, state causality and claims before merge. Do not let workers redefine the same schema independently.

Continuation: read current execution state, recent commits, failing tests and the relevant R2 gate. Resume the next incomplete work package; do not restart completed work, broaden into all research papers, or drop the full-chain objective for a simpler demo. Commit focused changes with reproducible reports; push only within existing user authorization.

## Deliverable format

Every report distinguishes **implemented**, **unit-tested**, **model-integrated**, **trained**, **evaluated**, and **unrun**. Include exact model/source locks, dataset split hashes, changed parameter groups, optimizer state layout, hardware peaks, task results, uncertainty, costs, identity mode and rollback version.

The only executable tool provided by this document package is `python tools/validate_bundle.py`. Trainer/runner paths in the construction plan are implementation targets until created and tested. Do not claim they ran because their names appear in a plan.
