# Execution state — actual-length D0 capacity and native adapter readiness

22 October 2026, America/Chicago. Branch `codex/native-capacity-readiness`; no remote or push configured. Milestones: `a4958d3` protocol/guard, `62390ba` exact-length data, `8ce2444` final-telemetry guard fix, `a7bd06f` native qualification tooling and `8500455` causal episode contract. The final evidence/documentation commit follows. [Previous balanced-overfit state](reports/execution_state_balanced_overfit_20261022.txt) is preserved.

**The selected learning diagnostic remains the eight-update checkpoint** at `runs/minicpm5_balanced_mask1_B8_lr1e4_chunk01_20261022/checkpoint/base`: train 8/8 and reused synthetic dev 14/16. No new architecture or post-freeze candidate was promoted. New capacity models start from the sealed original and are infrastructure probes.

The latest verified gates are **bounded actual 256/512-token D0 training and fresh-process restart**, **native D1/D2 null-branch forward/backward/export checks at 256**, and **the bounded chronological episode contract**. Augmented generation, D1/D2 learning and full D2 optimizer capacity remain incomplete. See [consolidated evidence](reports/capacity_readiness_experiment_20261022.json), [capacity record](docs/experiments/20261022_Capacity_Qualification.md), [native adapter record](docs/experiments/20261022_Native_Adapter_Qualification.md) and [roadmap](docs/stages/00_Master_Roadmap.md).

## Environment and unchanged source

Mac Studio M2 Ultra, 24 CPU / 60 GPU cores, 64 GiB unified memory, macOS 26.6.2. Python 3.12.14, torch 2.14.1, Transformers 5.20.2 and safetensors 0.8.0 remain locked; `.venv` is ready. The source remains `openbmb/MiniCPM5-2B` at `f97400052a43d642bbc6e9975e2397e3ae6a6b52`; source seal and core training implementation are unchanged. Historical completed-update checkpoints retain their original contract compatibility.

All four capacity processes ran MPS FP32 without CPU fallback, microbatch 1, accumulation 8 and activation checkpointing. Each effective update saw answers `0..7` exactly once. All 2,516,756,480 original parameters were trainable. The portable experimental Muon path uses INT8 hidden-matrix momentum in 2048-element blocks with FP32 scales/temporary arithmetic, momentum 0.95, five Newton–Schulz iterations and no Nesterov; embeddings/head and excluded tensors retain auxiliary FP32 AdamW. Both peak rates were `1e-4`, warmup two updates, zero decay. These are bounded diagnostic settings, not production SFT hyperparameters.

## Measured capacity

| Measure | Actual 256 tokens | Actual 512 tokens |
|---|---:|---:|
| Full / prompt / supervised tokens per row | 256 / 254 / 2 | 512 / 510 / 2 |
| Capacity train/dev rows | 8 / 8 | 8 / 8 |
| Completed updates | 2 | 2 |
| Fresh-process continuation | 1 → 2 | 1 → 2 |
| Cumulative processed / supervised train tokens | 4,096 / 32 | 8,192 / 32 |
| Training-update time across both chunks | 44.988 s | 50.945 s |
| Profiled process time, chunk 1 / chunk 2 | 53.279 / 64.238 s | 61.850 / 76.586 s |
| Sampled live MPS peak across chunks | 24.96 GiB | 25.29 GiB |
| Sampled driver MPS peak across chunks | 29.09 GiB | 29.06 GiB |
| Saved original tensors changed | 381 / 381 | 381 / 381 |
| Reload logit difference | 0.0 | 0.0 |

Complete chronological register writes, two MASK observations and truthful status fields fill these sequences; there is no padding-only inflation, evidence truncation or copied gold annotation. The [data manifests](data/diagnostics/capacity_256/manifest.json) and [512-token manifest](data/diagnostics/capacity_512/manifest.json) bind source tokenizer/template bytes, generator and file hashes, exact token counts and independent per-bit solver results. These deterministic capacity worlds are not a representative learning corpus or independent long-context benchmark.

Both source/export CPU audits confirm all 381 original tensors changed, including embeddings/head, with native names/shapes intact. Every process recorded nonzero gradients in all 381 originals. Reload comparisons covered 33,423,360 logits at 256 and 66,846,720 at 512. [Runtime validation](reports/validation_capacity_runtime_20261022.json) checks all four complete checkpoint receipts, INT8/FP32 storage, RNG devices, cursor, schedule and history; optimizer tensors occupy 6,265,262,080 bytes. Exact full-size uninterrupted-versus-split equivalence remains unmeasured.

The declared guard stops at sampled driver memory of 40 GiB, sustained system swap growth of 512 MiB, or system free-percentage at/below 15% for three host samples. Nominal host/profile polls are 2 s / 0.25 s. All complete traces pass; system swap occupancy stayed at 1,562.69 MiB in recorded samples. An independently reproduced exit race in the initial watchdog could miss final profile writes; commit `8ce2444` drains them before acceptance. Both earlier 256 traces were independently rechecked in full and contain no breach. Sampled guards cannot guarantee no OOM or detect sub-sample peaks. Host counters are system-wide; MPS and RSS overlap and must not be added.

These two-update runs qualify bounded D0 execution and restart. Their teacher-forced losses do not establish generated task accuracy, retention or sustained-training capacity; no capacity-model generation score is claimed. The resumed report histories are cumulative, whereas profile duration/memory describe only the current process.

## Native D1/D2 and episode foundation

The native 256-token sidecar qualification uses 895,490 new parameters in 48 tensors. Both D1 and D2 match the original model exactly across 33,423,360 logits when branch outputs are zero. Target-suffix changes leave observed memory/workspace fixed, target-contaminated prompts are rejected, and source features rebuild after writer changes. Checkpointed backward recomputes the native decoder while constructing writer/workspace once and preserving context until explicit release. D1 has no original gradients; D2 has nonzero gradients in all 381 originals. All 48 new tensors have gradient buffers, initially only the two zero-initialized branch output projections are nonzero, as expected.

A nonzero diagnostic sidecar probe exports/reloads exactly. This performs **zero optimizer updates** and no augmented generation. The probe is not a trained checkpoint; its 18.90 GiB live / 21.26 GiB driver peak does not qualify full D2 optimizer residency.

The [episode contract](docs/experiments/episode_contract_design.md) implements immutable chronological snapshots, source/agent ownership, current permissions, late correction/tombstone handling, explicit oracle exclusion, audit-view restrictions and exact eight-field export with byte-bound verification receipts. Forty-seven targeted tests cover its boundaries. Raw-source identity/hash authenticity, semantic gold laundering, complete-release admission, global deletion and permission retention outside flattened SFT rows remain host/intake responsibilities. No public corpus, authentic biography or 2,000-target release was admitted.

## Validation and next concrete work

**246 tests pass**, zero failures/errors/skips, on the actual Mac host: [output](reports/test_output_capacity_readiness_20261022.txt), [JUnit](reports/tests_capacity_readiness_20261022.xml), [current validation](reports/validation_capacity_readiness_20261022.json). Three existing top-level schemas plus the nested episode schema validate; all 32 new capacity rows and 48 original fixtures pass solver/schema/split checks in [data validation](reports/validation_capacity_data_20261022.json). Large weights/runtime remain ignored; compact evidence and receipts are tracked.

Next finish **S2.1 prompt-fixed augmented generation**: first qualify an uncached correctness path against native generation under null branches, preserve observed-memory/original-prompt state throughout generated answers, then qualify any cache. Follow with **S2.2 native D1 balanced tiny overfit** and generated-answer verification; D2 optimizer capacity and learning remain separate. In parallel, **S1.2** builds one raw-source-bound solver family and its release manifest before scaling to the proposed 2,000 targets. Preserve source roles, permission/purpose receipts and independent evidence controls.

The 20k pilot, retention study, ten-cycle learned T0–T3 loop, authentic Replica fidelity, VLM training, Blackwell offload and large-model ports remain gated. This Mac result does not qualify the later 24 GB card: current full-text FP32 persistent state alone is 24.59 GiB before activations/temporaries.
