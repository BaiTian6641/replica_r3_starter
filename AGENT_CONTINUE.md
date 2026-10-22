# Coding-agent continuation instruction

Resume the existing TAIS Obsidian / Replica work. Preserve the objective and all established constraints. Do not restart the project.

Read execution_state.md, recent commits, git status, the last raw run report, AGENT_START.md and the relevant work package in docs/03_Implementation_and_Runbook.md. Distinguish tested code from proposed work and pretrained-model experiments from fixture tests.

Preserve the setup choice: experimental Muon 8-bit on this M2 Ultra Mac Studio (64 GiB), then the later RTX PRO 4000 Blackwell SFF (24 GB); FP32 AdamW auxiliary groups keep embeddings/head/slot queries eligible. The locked environment, source seal and historical constant-zero failure remain preserved. Balanced full D0 now passes: eight training answers correct at updates 4 and 8; saved-export train 8/8 and dev 14/16. Native completed-update resume from 8 to 12 retains train 8/8 and scores dev 13/16. Both exports changed all 381 tensors and reloaded exactly. The predeclared update-8 checkpoint remains the diagnostic result; no architecture or learning candidate was promoted. All 169 tests pass. Actual training length is still 55, not the configured 256 ceiling. Next execute S0.5 actual-length capacity, followed by native D1/D2; source-grouped evaluation/data work can proceed in parallel. Reuse scheduler/optimizer/RNG/cursor checkpoints at completed updates; full-size continuous-versus-resumed equivalence and mid-accumulation recovery are not established. Do not rerun setup or silently replace Muon with AdamW. CUDA full-text residency needs the separate P2 path. Read the 33-substage roadmap in docs/stages/00_Master_Roadmap.md and its dated evidence register.

State briefly: current accepted checkpoint/manifest; last verified gate; current failing or incomplete gate; next concrete action. Then perform that action. Reuse prior successful checks unless a change invalidates them.

Use parallel workers with disjoint ownership when useful. Keep one owner for schemas and integration. Preserve uncommitted user changes and use focused commits. Do not invent remotes, silently substitute models or broaden into unrelated architecture rewrites.

The mandatory scientific path remains D0 plain full tuning, D1 frozen base/new modules and D2 full augmented tuning, followed by a frozen-base longitudinal comparison against memory-only and fixed-learning controls. The first complete loop includes cognition/control, experience, candidate training, independent evaluation and the Replica branch.

If a run is blocked by VRAM, implement a valid offload/sharding path or report the failed scope; do not silently change full tuning into LoRA or partial training. If the native VLM path is not implemented, keep it explicitly pending.

When generating data, use the supplied prompt packet, maintain source/split lineage, keep teacher output unverified until independent checks, and never fabricate autobiography or expose sealed tests.

Finish the next work package with raw evidence, update execution_state.md and the status matrix, and identify the next smallest action. Report implemented, tested, trained, evaluated and unrun separately.
