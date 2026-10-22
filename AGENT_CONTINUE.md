# Coding-agent continuation instruction

Resume the existing TAIS Obsidian / Replica work. Preserve the objective and all established constraints. Do not restart the project.

Read execution_state.md, recent commits, git status, the last raw run report, AGENT_START.md and the relevant work package in docs/03_Implementation_and_Runbook.md. Distinguish tested code from proposed work and pretrained-model experiments from fixture tests.

Preserve the setup choice: experimental Muon 8-bit on the M2 Ultra Mac Studio (64 GiB), then the later RTX PRO 4000 Blackwell SFF (24 GB). Auxiliary FP32 AdamW keeps embeddings/head/slot queries eligible. Reuse the locked environment, sealed original and completed checkpoints; do not reacquire another revision or silently substitute AdamW. Preserve the historical constant-zero failure and the successful eight-update tiny-overfit result (export train 8/8, dev 14/16); its diagnostic checkpoint remains selected. No architecture or post-freeze learning candidate was promoted. CUDA full-text residency still needs the separate P2 path.

Subsequent readiness work is complete on branch `codex/native-capacity-readiness`: actual 256/512-token D0 capacity, two balanced full-scope updates and fresh-process resume per length, all 381 saved tensors changed, exact reload and sampled driver peaks about 29.1 GiB. Native 256-token D1/D2 null-parity/causality/backward/export checks pass, with no optimizer updates or augmented generation. The causal episode contract is implemented and reviewed; the current suite has 246 passing tests. The earlier update-8 learning checkpoint remains selected. Read current execution_state.md and the new capacity/native qualification reports; do not repeat S0.5 D0 capacity. Next finish prompt-fixed augmented generation parity, then native D1 tiny overfit; source-bound S1.2 data work can proceed in parallel. S0.5 D2 optimizer capacity remains pending and is performed under S2.3; its D0 subgate is sufficient to enter that work once the other dependencies pass.

State briefly: current accepted checkpoint/manifest; last verified gate; current failing or incomplete gate; next concrete action. Then perform that action. Reuse prior successful checks unless a change invalidates them.

Use parallel workers with disjoint ownership when useful. Keep one owner for schemas and integration. Preserve uncommitted user changes and use focused commits. Do not invent remotes, silently substitute models or broaden into unrelated architecture rewrites.

The mandatory scientific path remains D0 plain full tuning, D1 frozen base/new modules and D2 full augmented tuning, followed by a frozen-base longitudinal comparison against memory-only and fixed-learning controls. The first complete loop includes cognition/control, experience, candidate training, independent evaluation and the Replica branch.

If a run is blocked by VRAM, implement a valid offload/sharding path or report the failed scope; do not silently change full tuning into LoRA or partial training. If the native VLM path is not implemented, keep it explicitly pending.

When generating data, use the supplied prompt packet, maintain source/split lineage, keep teacher output unverified until independent checks, and never fabricate autobiography or expose sealed tests.

Finish the next work package with raw evidence, update execution_state.md and the status matrix, and identify the next smallest action. Report implemented, tested, trained, evaluated and unrun separately.
