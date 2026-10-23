# Coding-agent start instructions - TAIS Obsidian / Replica R3

**Current work — E3 implemented and native-qualified on 23 October 2026 UTC.**
The shared width-256/eight-slot/two-iteration core, independent memory and
workspace readers at blocks 13/27, exact task journal, typed actions, real
fixture tools, independent success check and fresh-process restart are
connected. Read the [E3 implementation](docs/architecture/08_E3_Implementation.md)
and [native results](docs/experiments/20261023_E3_Native_Results.md).

**Engineering acceptance passes; learned task completion is negative.** All
eight model-driven episodes failed independent verification: the full-core arm
repeated `read`, the workspace-zero control reached a failing test and wrote an
invalid repair, and the other controls failed differently. Totals: 56 decisions,
32 tool calls, 3 malformed actions, 13 false success claims and 0 completions.
The success head was not trained because all eight actual outcomes were
negative. No model-written success statement was accepted.

The E2 parent extension was used as validated initialization and regression
reference: 43 tensors copied exactly, 51 E3 tensors freshly initialized, 43/43
parent tensors equal to the artifact. Peak sampled driver memory is **14.72 GiB**
with zero observed swap growth; the final v2 extension is **35.85 MiB**; no base
copy, optimizer checkpoint, download or Neural Engine claim.

Fresh-process interruption: process 1 observed the failing test, a CPU host
correction bumped revision 10→11, and process 2 rejected the stale workspace,
did not re-execute completed actions, spent the remaining decisions and still
failed to repair. The split acceptance is **10/14**; restart/idempotency
engineering passes, learned completion does not.

**E3 narrower repair (23 October):** fresh `set_constant` family passes
all engineering checks but scores **0/7**; the model repeats `read` despite
teacher loss 0.015. This is a second negative action-policy family, not a tool
defect. Do not rerun a near-identical action arm; target action selection or
advance E4. See [results](docs/experiments/20261023_E3_Set_Constant_Results.md).

**E2 reader candidate (23 October):** D1 answer-loss training on 24 cases
left dev+held-out exact accuracy **0/16** (parent 0/16); values were not copied.
Deferred, no default change. See
[reader candidate results](docs/experiments/20261023_E2_Reader_Candidate_Results.md).

**E3 mixed-outcome policy head (23 October):** 4/4 head-conditioned and
2/2 no-head held-out episodes completed with independent evaluator passes; head
and controls had identical action kinds, so head benefit is inconclusive. See
[policy-head results](docs/experiments/20261023_E3_Policy_Head_Results.md).

**E3 mixed-outcome probe (23 October):** 1.00 kind accuracy vs 0.875
step-only, 0.375 majority on 16 held-out states; representation signal only,
no policy head or action generation. See
[mixed probe results](docs/experiments/20261023_E3_Mixed_Policy_Probe_Results.md).

**E3 policy probe (23 October):** prompt-only core states classified action
kinds at 100%, but all teacher trajectories share the same 5-step order; result
is step-confounded and inconclusive. No policy head promoted. See
[probe results](docs/experiments/20261023_E3_Policy_Probe_Results.md).

**E4 delayed-use candidate (23 October):** learned write projection + native
fast readers; 0/8 held-out, 7/24 train, correct/shuffled differ but no
association gain; compositional 1/8 vs 1/8 shuffled, 0/8 controls. Rejected,
no default change. See
[E4 delayed-use results](docs/experiments/20261023_E4_Delayed_Use_Results.md).

**E5 neural candidate (23 October):** existing E2 reranker evaluated against
its parent on held-out sources: top1 0.25 → 0.50, but lexical 0.75. Accepted
opt-in only; lexical default unchanged. See
[neural candidate results](docs/experiments/20261023_E5_Neural_Candidate_Results.md).

**E5 procedure candidate (23 October):** a compiled repair procedure passed
3/3 held-out + dev with independent receipts, published with harmful-candidate
rejection and rollback. No neural candidate or learned action policy; model
action remains 0/7. See
[procedure results](docs/experiments/20261023_E5_Procedure_Results.md).

**E4 typed artifacts (23 October):** kind-specific ContextPack/RecordLatent/
BranchMemoryImage/ProcedureArtifact/NeuralCandidate metadata, lifecycle and
dependency invalidation are implemented/tested. No execution, candidate
training or publication yet. See
[typed artifacts](docs/architecture/11_Runtime_Artifacts.md).

**E4 fast-memory reference (23 October):** staged observation writes, one
chunk-start commit, bounds/lineage/reset and strict image persistence are
implemented and tested on the host. Opt-in native read ports pass null parity/independent influence; an untrained observation bridge now commits actual read/test observations into the next decision. Learned writes and delayed-use evaluation remain pending. See
[fast-memory reference](docs/architecture/10_Fast_Memory_Reference.md).

**C16 identity contracts (23 October):** source episodes, approved snapshots,
conditional principles, mode-bound contexts, branch isolation, correction/
revocation, a null prompt path and an optional zero-init reader are implemented
with fictional tests. No authentic source, trained identity conditioning or
fidelity evaluation exists. See
[identity interface](docs/architecture/09_Replica_Identity_Interface.md).

**E2 follow-up (23 October):** source-separated retrieval gives held-out
recall@1 **0.75** lexical/dense and **0.50** learned rerank; each slot arm
answers **0/6**, oracle slots **0/4**, and exact text **3/3**. The novel-value
writer/reader path is the bottleneck; eligibility and long-chunk inclusion pass.
The largest actually exercised capacity is **16 units / 2,043 source tokens /
256 prompt tokens / 8 generated tokens** plus a modules-only 16-unit backward,
at **12.51 GiB** sampled peak driver and zero swap growth. Reranking remains
opt-in; lexical remains the default. D2 capacity was not run. See
[E2 results](docs/experiments/20261023_E2_Retrieval_and_Capacity_Results.md).
**Open:** held-out configuration repair, correct action/argument selection,
full-core plan continuation, success-head training/calibration, novel-value E2
writer/reader transfer, ranking quality and capacity beyond the qualified
16-unit/256-token workload. These are
exposed synthetic fixtures, not a generalization result. Do not retune this
held-out set; a narrower repair tool or fresh cases requires a new preregistered
arm. Next: either that bounded tool-interface study or E2 retrieval/capacity
qualification, not a broad corpus campaign or model promotion.

Validation: the integrated full suite passed **1081 tests / 21 subtests**; the
E3-focused suites pass **157 tests**. Attempts 1 (post-migration parity order),
2 (EOS-as-second-JSON parser) and the interrupted pre-check (nested binding) are
preserved; attempt 3 is the accepted evidence. The source seal, historical
checkpoints and the protected untracked runtime draft remain unchanged.

**Storage rule:** reuse `checkpoints/minicpm5_source` revision
`f97400052a43d642bbc6e9975e2397e3ae6a6b52`. Preserve the selected earlier D0/D1/D2
checkpoints, all E3 attempt evidence, and the E1/E2 and E3 extensions.
No routine downloads, duplicate native saves, or unrelated cleanup. New native
jobs use MPS with fallback disabled and the existing exclusive worker/guard.
`runtime_pilot.py` remains protected, untracked and outside the implementation.

## Mission

Continue toward a persistent semi-cognitive model that learns from experience and supports a personal parallel Replica. The complete target is goal → evidence → reasoning/action → independent outcome → experience → learning proposal → candidate update → independent evaluation → promotion/rejection → improved later behavior.

Preserve two stages: full-scope small-model architectural development first; normally frozen large-model deployment with adaptable memory/skills/approved components afterward. Memory is central, but retrieval alone, a persona prompt, a toy symbolic demo or disconnected passing modules are not project completion.

## Read and establish ground truth

Current setup decision (22 October 2026, America/Chicago): work first on the confirmed Apple M2 Ultra Mac Studio (24 CPU cores, 60 GPU cores, 64 GiB unified memory), using the locked Python 3.12.14 environment and the project's experimental `muon8bit` optimizer on MPS. The same optimizer is planned for the later NVIDIA RTX PRO 4000 Blackwell SFF Edition (24 GB). Keep auxiliary FP32 AdamW for embeddings/head/slot queries and other ineligible parameters; preserve exact D0/D1/D2 scope and separate Muon/base/new rates. Use all-AdamW only as an explicitly selected comparison. Read execution_state.md for completed setup evidence and existing source/sealed locks; do not reacquire a different `main` revision. The FP32 full-text persistent budget alone exceeds 24 GiB, so the later CUDA full-scope path requires P2 offload/sharding or a separately qualified residency change.

1. Read README.md, docs/01 through docs/06, docs/07_Architecture_Diagrams.md, reports/validation.json, and docs/03_Implementation_and_Runbook.md's status table.
2. Inspect the actual repository status, remotes, branch, uncommitted changes, current execution notes and latest test failures. Preserve user work. Read applicable repository instructions.
3. If there is an existing project, use a focused branch/worktree and integrate useful starter components through a migration plan. If this is a genuinely new project with no repository, initialize Git inside this project root and make a local initial commit. Do not invent a remote or push without established authorization.
4. Distinguish original delivered CPU/tiny-Llama evidence from subsequent native execution. D0 actual 256/512 capacity, fixed-context augmented generation and D1/D2 tiny memorization are already qualified. Read execution_state.md and the dated results for completed optimizer/capacity, restart and generation gates. The final integration suite passes 819 tests with one restricted-context MPS skip; the pre-native suite remains 798/one skip. Broad quality, useful architecture benefit and Replica fidelity remain unmeasured.

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
