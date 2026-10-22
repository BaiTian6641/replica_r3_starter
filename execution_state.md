# Execution state — fixed-context generation and native D1 memorization

22 October 2026, America/Chicago. Branch `codex/augmented-generation-d1`; no remote or push configured. Code/protocol: `8df6a0b`; cognitive contracts: `0f36b3f`; conditional rate protocol: `8271520`; source-bound episode slice: `5766768`. The final evidence commit follows. [Previous capacity state](reports/execution_state_capacity_readiness_20261022.txt) is preserved.

**Native S2.1 fixed-context uncached generation and S2.2 bounded D1 memorization now pass.** The successful D1 sidecars produce 8/8 training answers, but only 2/16 reused synthetic development answers. This is a training-infrastructure result, not useful memory benefit, broad transfer or self-improvement. The existing plain D0 update-8 checkpoint remains the selected learning diagnostic; no developmental model or post-freeze candidate is promoted.

## GPU execution and environment

Mac Studio M2 Ultra, 24 CPU / 60 GPU cores, 64 GiB unified memory, macOS 26.6.2. All native model training and answer generation use **MPS GPU, FP32, CPU fallback disabled**. Host tokenization, datasets, file checks and small contract tests remain CPU work. The Neural Engine is not part of the current PyTorch/Muon backend; the [primary-source feasibility review](docs/experiments/20261022_Apple_Accelerators.md) distinguishes public Core ML prediction from experimental ANE training. No ANE execution is claimed.

The locked Python 3.12.14 / torch 2.14.1 / Transformers 5.20.2 environment and sealed `openbmb/MiniCPM5-2B` revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52` remain unchanged. Bound training code stayed identical across each fresh-process continuation. Earlier checkpoints retain their historical code contracts; use the corresponding Git revision when resuming them.

## Measured generation and D1 results

At actual sequence length 256 (prompt 254), D1/D2 null sidecars exactly match native uncached generation: eight generated tokens and 1,044,480 next-token logits per arm, zero difference. A three-token EOS-disabled probe forces context reuse. Nonzero memory/workspace probes affect logits, and nonzero sidecars export/reload exactly. Source slots and workspace build once from the original prompt/source, remain fixed during answer decoding and release afterward. These capped native prose continuations establish interface parity only. [Generation record](docs/experiments/20261022_Augmented_Generation.md).

D1 trains 895,490 new parameters in 48 tensors while freezing all 2,516,756,480 originals in 381 tensors. Both arms use the original balanced eight-case mask-1 diagnostic, 55 actual tokens, microbatch one, accumulation eight, checkpointing, two-step warmup, zero decay and experimental Muon 8-bit plus FP32 AdamW exclusions. All eight classes occur once per update.

| Measure | First rate 1e-3 | Conditional fresh rate 1e-2 |
|---|---:|---:|
| Updates / fresh-process continuation | 64 / 8→64 | 28 / 8→28 |
| Final periodic train generation | 0/8 | 8/8 at 24 and 28 |
| Saved-export train generation | 0/8, cap 8 | 8/8, cap 256, all EOS |
| Saved-export dev generation | 0/16, cap 8 | 2/16, cap 256, all EOS |
| Processed / supervised training tokens | 28,160 / 1,024 | 12,320 / 448 |
| Training-update time | 105.050 s | 45.952 s |
| Profiled trainer processes | 296.080 s | 149.872 s |
| Sampled live / driver MPS peak | 9.496 / 11.006 GiB | 9.467 / 11.113 GiB |

The first negative is preserved. The higher rate was declared before the first arm finished, starts fresh from the same parent/seed, and changes only new-module rates. Caps differ for failed versus successful external evaluation; do not use those scores as a matched quality comparison. All costs count: **92 updates, 40,480 processed tokens, 1,472 supervised tokens, 151.002 s update time and 445.953 s profiled trainer time**. An initial argument-only launch failure performed zero updates and is retained separately.

New nonzero gradients open from two output projections at update 1 to all 48 tensors at update 2. The independent saved-weight audit finds zero changed original tensors/elements and all four exports byte-identical in native weights. All 48 sidecars change during each continuation. Optimizer ownership is 16 new Muon matrices plus 32 new AdamW tensors; persistent state is 1,029,824 bytes. Every reload compares 7,180,800 logits with maximum difference zero. Independent runtime validation passes 1,665 assertions over checkpoint bytes, cursor/history/rates/scopes and all four complete guard traces. No sampled swap growth or guard breach is observed. Driver and RSS overlap; sampled peaks can miss transients.

[Consolidated evidence](reports/native_augmented_D1_experiment_20261022.json), [D1 experiment](docs/experiments/20261022_D1_Diagnostic.md), [weight audit](reports/minicpm5_D1_checkpoint_audit_20261022.json), [runtime validation](reports/validation_D1_runtime_20261022.json).

## Parallel documents and data

The [proposed cognitive contracts](docs/experiments/20261022_Cognitive_Contracts.md) now spell out the starter-to-typed-core bridge, slot/cache lifetimes, evidence pins/live revocation, task restart, update authorities, IdentityChangeProposal, mode-aware belief versus correctness and director ownership/reward/cadence. These remain specifications. An early fixed-baseline host-loop pilot can proceed after S1.1 alongside the neural study; full selected-system binding retains S2.5.

S1.2 now has a [source-bound 20-episode slice](docs/experiments/20261022_Episode_Slice.md): ten synthetic worlds, four cases per role, exact archived source/span hashes, independent chronology/solver checks, and only four train/four dev flattened exports. All 20 checks and 12 targeted tests pass. Audit/promotion fixtures are exposed contract material, not sealed scientific evaluation; the 2,000-target release remains incomplete.

The final suite has **290 passing tests**, zero failures/errors/skips. [Test evidence](reports/test_output_augmented_d1_20261022.txt), [validation](reports/validation_augmented_D1_20261022.json). Archived R2 sources remain unchanged.

## Next concrete gate

Run a bounded **S2.3 full-augmented D2 optimizer/capacity preflight at actual 256 tokens**, preserving full original/new parameter ownership and guarded MPS execution, before D2 tiny overfit. Its memory is not established by D0 capacity or D1's frozen-base result. In parallel, extend S1.2 beyond one rule family and build the early thin connected host-loop pilot from a fixed baseline manifest.

Matched architecture benefit, retention, ten-cycle learning, authentic Replica fidelity, VLM training, augmented KV caching, Blackwell residency and ANE acceleration remain unqualified. The 20k campaign and optional mechanisms retain their data, capacity and independent evaluation gates.
