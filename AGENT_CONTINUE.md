# Coding-agent continuation instruction

**Current priority — architecture research and rebuild design, 22 October 2026.**
Read the [integrated design](docs/architecture/README.md), [backbone ports](docs/architecture/02_Backbone_Integration.md),
[memory and compiler](docs/architecture/03_Memory_and_Runtime_Compilation.md),
[cognition and Replica](docs/architecture/04_Semi_Cognition_Learning_and_Replica.md), and
[six-package build sequence](docs/architecture/05_Implementation_Sequence.md).
The actual TAIS-Obisian source and primary memory research are now reviewed at
pinned revisions. This is a design deliverable; the 18 proposed components have
not been implemented or trained by this review. No native job was launched.

Next implement **E1/E2**: explicit request/manifest ownership, contextual
source-only record encoding and complete-block ports at 13/27. Follow with E3's
shared cognitive core, exact task graph and real action/outcome loop. Qualify
source grounding and generation inside these connected interfaces. Preserve
the three failed short-answer probes; repeating their wording is no longer the
main workstream. Data admission and fair D0/D1/D2 evaluation proceed alongside
construction; no broad training campaign is admitted by this design.

The latest native evidence remains the preserved query-policy v2 negative at
`f707325`: strict and independent semantic judgments both 4/16, equal to the
constant-UNKNOWN control. Last software suite: 819 passed, one restricted-context
MPS skip. Reuse that evidence while code remains unchanged. Preserve the sealed
local model, selected checkpoints and untracked runtime draft. See
[execution state](execution_state.md) and the [design review record](reports/architecture_research_20261023/review.json).

**Current storage state:** the bounded D2 GPU package is complete. The user
subsequently authorized [post-D2 checkpoint retirement](docs/experiments/20261022_Checkpoint_Retirement_After_D2.md):
19 superseded payload directories and 10 orphan pool files are removed.
Preserve the one sealed original, selected D0 update 8, D1 update 28, D2 tiny
update 8 and useful final negatives. Retained selected exports/runtime states
remain valid; older capacity/intermediate payload paths in archived reports are
historical and cannot be resumed. Native training/generation remain idle.

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
