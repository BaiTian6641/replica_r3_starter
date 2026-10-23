# Coding-agent start instructions - TAIS Obsidian / Replica R3

**Latest package:** S1.2 now has a source-bound 2,000-target/four-mechanism
infrastructure release. The original-model MPS dev16 probe passes resource
guards but scores 0/16 strict/format: all 16 hit cap 32 on prose, no EOS. This is a
preserved development output-contract negative. Predeclare a versioned explicit
one-line output contract before another probe; do not loosen the parser or
rewrite this result. The suite passes 613 with one restricted-context MPS skip.
Read [the release/result record](docs/experiments/20261022_Infrastructure_Slice_V2.md)
and execution_state.md. No downloads, updates or checkpoints were created.
S1.3 source admission, remaining S1.4 controls and S1.5 still precede S2.4.

**Current storage state:** the bounded D2 GPU package is complete. The user
subsequently authorized [post-D2 checkpoint retirement](docs/experiments/20261022_Checkpoint_Retirement_After_D2.md):
19 superseded payload directories and 10 orphan pool files are removed.
Preserve the one sealed original, selected D0 update 8, D1 update 28, D2 tiny
update 8 and useful final negatives. Retained selected exports/runtime states
remain valid; older capacity/intermediate payload paths in archived reports are
historical and cannot be resumed. Native training/generation remain idle.

Copy this complete instruction to the coding agent working in the actual project repository.

**Existing workspace storage rule:** reuse `checkpoints/minicpm5_source` and its immutable seal; a fresh experiment loads local bytes and does not download a fresh model. Source checkpoints, experiment weights, optimizer state and caches stay outside Git; follow [the retention inventory and Git policy](docs/experiments/20261022_Git_and_Checkpoint_Hygiene.md). Acquisition examples below are initial-setup history; only an intentional missing-source acquisition/repair may use `--allow-network`. Preserve all selected and undocumented artifacts. The [applied cleanup](docs/experiments/20261022_Checkpoint_Cleanup_Applied.md) supersedes the old marked-only storage snapshot. The user subsequently resumed the bounded D2 package.

## Mission

Continue toward a persistent semi-cognitive model that learns from experience and supports a personal parallel Replica. The complete target is goal → evidence → reasoning/action → independent outcome → experience → learning proposal → candidate update → independent evaluation → promotion/rejection → improved later behavior.

Preserve two stages: full-scope small-model architectural development first; normally frozen large-model deployment with adaptable memory/skills/approved components afterward. Memory is central, but retrieval alone, a persona prompt, a toy symbolic demo or disconnected passing modules are not project completion.

## Read and establish ground truth

Current setup decision (22 October 2026, America/Chicago): work first on the confirmed Apple M2 Ultra Mac Studio (24 CPU cores, 60 GPU cores, 64 GiB unified memory), using the locked Python 3.12.14 environment and the project's experimental `muon8bit` optimizer on MPS. The same optimizer is planned for the later NVIDIA RTX PRO 4000 Blackwell SFF Edition (24 GB). Keep auxiliary FP32 AdamW for embeddings/head/slot queries and other ineligible parameters; preserve exact D0/D1/D2 scope and separate Muon/base/new rates. Use all-AdamW only as an explicitly selected comparison. Read execution_state.md for completed setup evidence and existing source/sealed locks; do not reacquire a different `main` revision. The FP32 full-text persistent budget alone exceeds 24 GiB, so the later CUDA full-scope path requires P2 offload/sharding or a separately qualified residency change.

1. Read README.md, docs/01 through docs/06, docs/07_Architecture_Diagrams.md, reports/validation.json, and docs/03_Implementation_and_Runbook.md's status table.
2. Inspect the actual repository status, remotes, branch, uncommitted changes, current execution notes and latest test failures. Preserve user work. Read applicable repository instructions.
3. If there is an existing project, use a focused branch/worktree and integrate useful starter components through a migration plan. If this is a genuinely new project with no repository, initialize Git inside this project root and make a local initial commit. Do not invent a remote or push without established authorization.
4. Distinguish original delivered CPU/tiny-Llama evidence from subsequent native execution. D0 actual 256/512 capacity, fixed-context augmented generation and D1/D2 tiny memorization are already qualified. Read execution_state.md and the dated results for completed optimizer/capacity, restart and generation gates. The current suite passes 613 tests with one restricted-context MPS skip. Broad quality, useful architecture benefit and Replica fidelity remain unmeasured.

## First session: do concrete work

For continuation, follow execution_state.md and the latest accepted raw evidence. Do not repeat acquisition, collapse diagnosis or completed D0/D1/D2 gates. The D0 eight-update diagnostic remains selected; D2 infrastructure results do not promote a developmental model. The reviewed episode contract is bounded infrastructure; release-level source/purpose/verification admission is still required before a corpus campaign.

Run the dependency-free core checks and fixture validator. If numerical dependencies are available, run the existing suite once. Record the exact commands/results; fix a concrete failing contract before expanding scope. Do not rewrite passing components just to match your preferred style.

Inventory the small workstation and Mac separately, without changing drivers. Identify GPU UUIDs, per-device memory, dtype support, host/unified-memory availability, OS and framework versions. Do not assume the small workstation is a previous machine from older plans.

The original acquisition/sealing step is complete: reuse `checkpoints/minicpm5_source`, revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52`. Keep its tokenizer/template provenance. Offline verification is the default; a missing or corrupt source must fail without downloading. Do not swap to a Base/SFT sibling or another model silently. The production runner requires native Llama metadata and the actual 2,516,756,480 original-parameter inventory.

Establish native generation/task and response-NLL baselines. Then run the smallest real forward/backward/update and export/reload check. Use the actual hardware constraints:

- If all state fits on one device, use the reference trainer for a short preflight.
- If conventional full tuning does not fit, implement a tested optimizer-offload or training-sharding adapter under P2. Ordinary DDP and inference device_map do not pool training memory.
- Never freeze original layers to make a run fit while retaining the full-tuning label.
- Do not call a 4-bit adapter run full fine-tuning.

Register D0 full, D1 modules_only and D2 full_augmented from the same declared parent and eligible data. Include embeddings/head in full scope, independent new-module rates, response-only masks, exact optimizer membership and actual changed-tensor reporting.

Build the generation-based task evaluator and a verifiable task family before increasing dataset size. A declining loss is not the project success criterion.

## Parallel work allocation

Use subagents when available, with separate files/worktrees and explicit deliverables:

- Model/training worker: native integration, optimizer residency, gradients, checkpoints.
- Runtime/memory worker: scoped events, causality, real tool observations, restart and publication.
- Dataset/Replica worker: source intake, chronological episodes, independent validation, held-out decisions.
- Evaluation reviewer: leakage, parameter ownership, controls, statistical resolution and unsupported claims.

One integration owner controls shared schemas and acceptance. Workers must not independently redefine record/proposal/manifest formats. Review interface changes before merging.

## Required architecture and data contracts

Build workspace from known prompt tokens and memory from already observed source content. The teacher-forced response is never a workspace input. Support evidence may naturally contain a fact needed by the task; oracle answer annotations may not be injected as support.

Keep frozen downstream layers in the autograd graph when new modules require input gradients. Initialize branch output projections to zero, not every factor. Keep source writer outputs differentiable during training. Recompute features after changing their producer.

During checkpoint recomputation, no storage writes, counter mutations or context switches occur. Release the adapter's batch context only after backward. Fast state is functional and session-owned; all depth ports read one chunk-start snapshot and publish writes after the chunk.

Split by source/persona/decision/task family before teacher expansion. All descendants inherit the split. Preserve explicit unknowns, temporal corrections, permission scope and source IDs. Do not expose audit data to teachers, candidate training or curriculum selection. Public data does not automatically have one permissive license.

Replica source history, approved snapshots and agent branch experience remain separate. Do not fabricate personal records or promote previous assistant prose into authentic user testimony. Style, factual memory and conditional decisions have separate evaluation.

## Complete the learning loop

After the first fair D0/D1/D2 study, freeze the chosen developmental base. Connect the actual model's bounded action/learning proposals to the runtime. Train isolated candidate extensions/skills, compare against the current parent on unseen tasks and retention, then publish only a compatible accepted manifest. Rejection and inconclusive outcomes must work.

Replace the fixed symbolic compiler/director in the demo with real learned policy and candidate neural/skill updates before making any model self-improvement claim. Run T0/T1/T2/T3 controls for ten initial chronological cycles. Count selected practice, replay, verifier and compute costs. A procedure library gain and a neural learning gain are separate results.

MiniCPM-V-4.7-1B needs a native hybrid VLM port and actual image gradients. Preserve processor expansion, mRoPE and recurrent caches; do not insert the Llama wrapper blindly. Larger model ports need freshly trained interfaces and re-encoded source memories.

## Progress discipline

After each coherent work package, update execution_state.md with:

- Current commit/branch and exact tested scope.
- Commands, raw report paths and result.
- Implemented versus trained/evaluated versus unrun.
- Current blocker and evidence.
- The next smallest task that resolves it.

Commit focused working changes and maintain a short decision log. Use a new output directory for each experiment. Preserve negative results and failed candidates. Do not restart completed work, keep planning indefinitely, activate every research mechanism simultaneously or declare success from unit tests alone.

Stop optional testing when the concrete risk is resolved. Continue through the authorized construction work; ask the user only for genuinely missing information that blocks a meaningful next step.
