# Coding-agent continuation instruction

**Current work — E1/E2 implemented and GPU-qualified on 22 October 2026
America/Chicago (23 October UTC).** The explicit MiniCPM5 adapter, two native
memory ports, contextual writer, temporal ledger, eligible retrieval, source
cache and persistent runtime CLI are connected. Read the
[E1/E2 implementation](docs/architecture/07_E1_E2_Implementation.md) and
[native results](docs/experiments/20261023_E1_E2_Native_Results.md).

The bounded diagnostic generates **4/4** correct answers from record slots;
ports disabled gives **0/4**. Exact fallback, source-swap controls, correction,
serialization and a fresh-process runtime restart pass. D2 backward reaches all
381 original tensors; no D2 optimizer update was performed in this package.
Peak sampled driver memory is **21.11 GiB**, with zero observed swap growth.
One **18.62 MiB extension** is selected at
`runs/e1_e2_native_attempt02_20261023/selected_extension`; original model bytes
remain only in the reused sealed source. No downloads or duplicate base saves.

**Open:** reranker top-1 remains 1/2 after its four-update probe; novel-source
quality, full-bank native capacity and E3 cognition remain unqualified. These
are exposed synthetic integration cases, not a generalization result. Keep
reranking opt-in. Next build E3's shared core and exact task/action journal using
these interfaces, while planning separate retrieval and capacity qualification.
No broad corpus campaign or learning promotion is admitted here.

Validation: one full software run **922 passed / one restricted-context MPS skip /
21 subtests**, then **60 affected tests** after an MPS backward-thread fix,
followed by successful native qualification and runtime restart. Attempt 1's
failure and all old results remain preserved. Source seal, selected historical
checkpoints and the protected untracked runtime draft remain unchanged. See the
[final validation receipt](reports/validation_e1_e2_final_20261023.json).

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
