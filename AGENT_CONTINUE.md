# Coding-agent continuation instruction

Resume the existing TAIS Obsidian / Replica work. Preserve the objective and all established constraints. Do not restart the project.

**Storage maintenance / pause:** model construction is paused at the user's request. Git/storage maintenance does not resume training. Original model bytes at `checkpoints/minicpm5_source` must be reused, never reacquired routinely. They and `runs/` stay outside Git. Follow [checkpoint retention and Git hygiene](docs/experiments/20261022_Git_and_Checkpoint_Hygiene.md); deletion candidates are marked, not deleted. Preserve selected D0/D1 artifacts and recent undocumented D2 runs. New model downloads require an explicit intentional `--allow-network` invocation; normal source checks remain offline.

Read execution_state.md, recent commits, git status, the last raw run report, AGENT_START.md and the relevant work package in docs/03_Implementation_and_Runbook.md. Distinguish tested code from proposed work and pretrained-model experiments from fixture tests.

Preserve the setup choice: experimental Muon 8-bit on the M2 Ultra Mac Studio (64 GiB), then the later RTX PRO 4000 Blackwell SFF (24 GB). Auxiliary FP32 AdamW keeps embeddings/head/slot queries eligible. Reuse the locked environment, sealed original and completed checkpoints; do not reacquire another revision or silently substitute AdamW. Preserve the historical constant-zero failure and the successful eight-update tiny-overfit result (export train 8/8, dev 14/16); its diagnostic checkpoint remains selected. No architecture or post-freeze learning candidate was promoted. CUDA full-text residency still needs the separate P2 path.

Current branch is `codex/augmented-generation-d1`. Native 256/512-token D0 capacity already passes. Fixed-context native D1/D2 generation has exact token/logit parity and nonzero-sidecar reload. D1 tiny memorization passes at 1e-2: train 8/8 at updates 24 and 28, saved train 8/8 and dev 2/16, all 381 originals unchanged, fresh-process resume 8→28. Preserve the failed 1e-3/64-update arm. Successful D1 driver peak is 11.113 GiB at 55 actual tokens; D2 capacity and useful memory benefit remain unqualified. The source-bound S1.2 slice contains 20 episodes from ten worlds, four train/four dev exports, and exposed audit fixtures; the 2k release and sealed audit remain pending. The final suite passes 290 tests. Read execution_state.md and reports/native_augmented_D1_experiment_20261022.json. Next run bounded native D2 optimizer/capacity at actual 256 tokens under S2.3; extend independent data families and early host-loop preparation in parallel. Reuse completed D0/D1 gates.

**Hardware requirement:** native model training and generation on this Mac must use `--device mps`, with CPU fallback disabled. CPU is for host I/O, data/weight verification and small contract tests. Neural Engine execution requires a separately qualified backend; do not describe MPS as ANE or silently substitute CPU model execution. See docs/experiments/20261022_Apple_Accelerators.md.

State briefly: current accepted checkpoint/manifest; last verified gate; current failing or incomplete gate; next concrete action. Then perform that action. Reuse prior successful checks unless a change invalidates them.

Use parallel workers with disjoint ownership when useful. Keep one owner for schemas and integration. Preserve uncommitted user changes and use focused commits. Do not invent remotes, silently substitute models or broaden into unrelated architecture rewrites.

The mandatory scientific path remains D0 plain full tuning, D1 frozen base/new modules and D2 full augmented tuning, followed by a frozen-base longitudinal comparison against memory-only and fixed-learning controls. The first complete loop includes cognition/control, experience, candidate training, independent evaluation and the Replica branch.

If a run is blocked by VRAM, implement a valid offload/sharding path or report the failed scope; do not silently change full tuning into LoRA or partial training. If the native VLM path is not implemented, keep it explicitly pending.

When generating data, use the supplied prompt packet, maintain source/split lineage, keep teacher output unverified until independent checks, and never fabricate autobiography or expose sealed tests.

Finish the next work package with raw evidence, update execution_state.md and the status matrix, and identify the next smallest action. Report implemented, tested, trained, evaluated and unrun separately.
