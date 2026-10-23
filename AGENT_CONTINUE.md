# Coding-agent continuation instruction

**Current work — E3 implemented and native-qualified on 23 October 2026 UTC.**
The shared core, independent memory/workspace readers, exact task journal,
typed actions, real fixture tools, independent success check and fresh-process
restart are connected. Engineering acceptance passes; **learned task completion
is negative**: 0/8 model-driven episodes completed, the full-core arm repeated
`read`, and the workspace-zero arm wrote an invalid repair. The success head was
not trained because all eight actual outcomes were negative. See
[E3 implementation](docs/architecture/08_E3_Implementation.md) and
[native results](docs/experiments/20261023_E3_Native_Results.md).

Fresh-process restart is split: 10/14 acceptance checks pass (state
reconstruction, revision correction, stale-workspace rejection, no re-execution
of completed actions); learned completion checks fail. Peak sampled driver
memory is 14.72 GiB, swap growth zero, final v2 extension 35.85 MiB, no base copy
or download. The E2 parent extension was copied exactly (43 tensors) as both
initialization and regression reference.

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
recall@1 **0.75** lexical/dense and **0.50** learned rerank; each slot arm answers
**0/6**, oracle slots **0/4**, and exact text **3/3**. The novel-value
writer/reader path is the bottleneck; eligibility and long-chunk inclusion pass.
Largest actually exercised capacity: **16 units / 2,043 source tokens / 256
prompt tokens / 8 generated tokens** plus modules-only 16-unit backward, at
**12.51 GiB** peak driver, zero swap. Reranking stays opt-in; lexical is the
default. D2 capacity was not run. See
[E2 results](docs/experiments/20261023_E2_Retrieval_and_Capacity_Results.md).

**Open:** configuration repair, correct action/argument selection, full-core
plan continuation, success-head training/calibration, novel-value E2
writer/reader transfer, ranking quality and capacity beyond the qualified
16-unit/256-token workload. Do not retune the held-out set. Next is either a
separately preregistered narrower repair-tool study with fresh cases or a
writer/reader improvement arm. No broad corpus campaign or model promotion is
admitted.

Validation: integrated full suite **1081 passed / 21 subtests**; E3-focused
suite **157 passed**. Attempt 1 (parity order), attempt 2 (EOS-as-second-JSON
parser) and the interrupted pre-check (nested binding) are preserved; attempt 3
is accepted. Source seal and the protected untracked runtime draft remain
unchanged.

**Storage rule:** reuse `checkpoints/minicpm5_source` revision
`f97400052a43d642bbc6e9975e2397e3ae6a6b52`. Preserve the selected earlier D0/D1/D2
checkpoints, attempt 1's negative evidence, and the current E1/E2 extension.
No routine downloads, duplicate native saves, or unrelated cleanup. New native
jobs use MPS with fallback disabled and the existing exclusive worker/guard.
`runtime_pilot.py` remains protected, untracked and outside the implementation.

Resume the existing TAIS Obsidian / Replica work. Preserve the objective and all established constraints. Do not restart the project.

**Storage rule:** original model bytes at `checkpoints/minicpm5_source` must be reused, never reacquired routinely. They and `runs/` stay outside Git. Follow [checkpoint retention and Git hygiene](docs/experiments/20261022_Git_and_Checkpoint_Hygiene.md); the 16 marked old runtime directories were retired by the [authorized cleanup](docs/experiments/20261022_Checkpoint_Cleanup_Applied.md). Preserve selected D0/D1 artifacts and D2 evidence. New model downloads require an explicit intentional `--allow-network` invocation; normal source checks remain offline. The earlier maintenance pause was superseded by the user's D2 resume instruction.

Read execution_state.md, recent commits, git status, the last raw run report, AGENT_START.md and the relevant work package in docs/03_Implementation_and_Runbook.md. Distinguish tested code from proposed work and pretrained-model experiments from fixture tests.

Future checkpoint saves under `runs/` now use verified hash-based APFS weight
sharing. Read [applied storage and disk measurements](docs/experiments/20261022_Checkpoint_Cleanup_Applied.md).
The pool was seeded from retained native snapshots; sidecars and runtime state
remain independent. Preserve the BF16 original and strict historical runtime
code contracts. Pool garbage collection is separate explicit work. Storage
maintenance alone does not authorize unrelated training campaigns.

Preserve the setup choice: experimental Muon 8-bit on the M2 Ultra Mac Studio (64 GiB), then the later RTX PRO 4000 Blackwell SFF (24 GB). Auxiliary FP32 AdamW keeps embeddings/head/slot queries eligible. Reuse the locked environment, sealed original and completed checkpoints; do not reacquire another revision or silently substitute AdamW. Preserve the historical constant-zero failure and the successful eight-update tiny-overfit result (export train 8/8, dev 14/16); its diagnostic checkpoint remains selected. No architecture or post-freeze learning candidate was promoted. CUDA full-text residency still needs the separate P2 path.

Current branch is `codex/d2-capacity-runtime-pilot`. S0.5 bounded D0/D2 actual 256/512 capacity and S2.3 D2 tiny memorization/scope/restart pass. D2 stops at update 8 after 8/8 at updates 4 and 8; saved-export train 8/8 and dev 14/16, all EOS at cap 256. All 381 originals and 48 added tensors change; all recorded guards pass. D1 memorization and its failed arm remain preserved. The historical D2 suite passed 479 tests with one restricted-context MPS skip; the final v2 integration package passes 819/one skip; preregistration remains 798/one skip. Source and acquisition receipt remain unchanged. Read execution_state.md and reports/native_augmented_D2_experiment_20261022.json. Continue the integrated E1/E2/E3 architecture build and independent S1 admission/evaluation; preserve the S2.4 fair-comparison proposal without launching the 20k campaign. Useful gain, retention and full learning-loop integration remain pending.

**Hardware requirement:** native model training and generation on this Mac must use `--device mps`, with CPU fallback disabled. CPU is for host I/O, data/weight verification and small contract tests. Neural Engine execution requires a separately qualified backend; do not describe MPS as ANE or silently substitute CPU model execution. See docs/experiments/20261022_Apple_Accelerators.md.

State briefly: current accepted checkpoint/manifest; last verified gate; current failing or incomplete gate; next concrete action. Then perform that action. Reuse prior successful checks unless a change invalidates them.

Use parallel workers with disjoint ownership when useful. Keep one owner for schemas and integration. Preserve uncommitted user changes and use focused commits. Do not invent remotes, silently substitute models or broaden into unrelated architecture rewrites.

The mandatory scientific path remains D0 plain full tuning, D1 frozen base/new modules and D2 full augmented tuning, followed by a frozen-base longitudinal comparison against memory-only and fixed-learning controls. The first complete loop includes cognition/control, experience, candidate training, independent evaluation and the Replica branch.

If a run is blocked by VRAM, implement a valid offload/sharding path or report the failed scope; do not silently change full tuning into LoRA or partial training. If the native VLM path is not implemented, keep it explicitly pending.

When generating data, use the supplied prompt packet, maintain source/split lineage, keep teacher output unverified until independent checks, and never fabricate autobiography or expose sealed tests.

Finish the next work package with raw evidence, update execution_state.md and the status matrix, and identify the next smallest action. Report implemented, tested, trained, evaluated and unrun separately.
