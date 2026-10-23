# Execution state — native D2 bounded qualification

22 October 2026, America/Chicago. Branch `codex/d2-capacity-runtime-pilot`. The user resumed this bounded D2 GPU package after the completed storage-maintenance pause. The package is complete; no native job remains active. No remote or push is configured.

**S0.5 actual 256/512-token D0/D2 capacity and S2.3 D2 tiny memorization/scope/restart now pass.** D2 generates all eight balanced training answers at updates 4 and 8, then stops by its declared rule. Independent saved-export evaluation confirms **train 8/8 and dev 14/16**, all EOS at the same cap 256. These are small reused synthetic fixtures. Useful architectural gain, broad transfer, retention and autonomous learning remain unqualified.

The original plain D0 update-8 learning diagnostic remains selected. D1 update-28 and its failed rate arm remain preserved comparison artifacts. No developmental or post-freeze model is promoted. The unfinished `src/replica_lab/runtime_pilot.py` draft is preserved separately and is not part of this package.

## Single original model and GPU execution

Reuse **`checkpoints/minicpm5_source`**, immutable MiniCPM5-2B revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52`, with its existing sealed lock and tokenizer/template. There is one original source template for ongoing experiments. All 36 original files retain identical sizes, inodes and timestamps; the acquisition receipt is unchanged; all nine sealed inventory files reverify by hash. **Zero downloads** occurred. Missing/corrupt source fails offline instead of fetching automatically. [Final source check](reports/D2_source_reuse_final_20261022.json).

Mac Studio M2 Ultra: 24 CPU/60 GPU cores, 64 GiB unified memory, macOS 26.6.2. Native training and generation use **MPS GPU FP32, CPU fallback disabled**, one native process at a time. Tokenization, file-value audits and small contract tests use the host CPU. Python 3.12.14, torch 2.14.1 and Transformers 5.20.2 remain locked. The Neural Engine is not used by this backend. [Accelerator review](docs/experiments/20261022_Apple_Accelerators.md).

Every fresh capacity/tiny arm loads the same local source; within-pair continuation restores its own retained completed-update checkpoint. Selected D0/D1 states are comparison artifacts rather than substituted D2 parents. The completed historical 256 pair binds to actual Git objects at `9bc719b`; its receipts were preserved. New 512/tiny pairs use the unchanged storage-aware training hashes pinned in [the resumption plan](configs/d2_resume_plan_20261022.json).

## Measured D2 results

D2 trains **all 2,516,756,480 originals / 381 tensors**, including embeddings/head, plus **895,490 new parameters / 48 tensors**. No original is frozen. Added memory/workspace is 64-wide, four slots, two workspace steps and four heads at the final block. Experimental Muon 8-bit and FP32 auxiliary AdamW own every tensor once: 294/87 original and 16/32 new tensors. Original/new peak rates are 1e-4/1e-3, warmup two, constant schedule, zero decay, seed 17, microbatch one and accumulation eight.

| Gate | Updates and fresh-process restart | Sampled peak MPS live / driver GiB | Answer result |
| --- | --- | ---: | --- |
| Actual 256 tokens | 0→1→2, reused qualified pair | 26.751 / 29.072 | Capacity only |
| Actual 512 tokens | 0→1→2 | 26.876 / 30.080 | Capacity only |
| Tiny, actual 55 tokens | 0→4→8, declared maximum 64 | 26.653 / 29.195 | 8/8 at 4 and 8; saved train 8/8, dev 14/16 |

All 381 originals receive finite nonzero gradients and independently verified saved changes at each audited boundary. Added gradients open from two null output projections at update 1 to all 48 by update 2. All 48 sidecars change during each continuation. Runtime/export native and sidecar bytes match; all compared reload logits have maximum difference zero. Optimizer tensor state is 6,266,291,904 bytes. Both capacity and tiny chains pass strict metadata/optimizer/RNG/cursor/history/generation/guard validation with byte-bound native audits. Metadata mode does not independently rehash every large runtime payload.

All six training guards and both independent evaluation guards pass, with no observed swap growth. Declared limits are 40 GiB driver, sustained 512 MiB swap growth, host free minimum 15% over three samples; driver cadence .25 s, host cadence 2 s. Samples can miss transients; RSS/driver overlap. Native restart is verified, while full-size uninterrupted-versus-split numerical equivalence remains unmeasured.

Unique cost across all three evidence chains is 12 updates, 15,808 sequence tokens, 192 supervised tokens, 258.749 s update time and 634.273 s profiled trainer lifetimes. This resumption reuses the old 256 pair: **10 new updates**, 11,712 sequence tokens, 213.068 s update time and 500.531 s profiled trainer lifetimes. Independent saved-export evaluation adds 23.981 s. Audits/preflight/tests are additional work. [Source/workspace token accounting](reports/D2_token_accounting_20261022.json) reports added input views separately from native sequence tokens.

[Protocol](docs/experiments/20261022_D2_Protocol.md), [D2 results](docs/experiments/20261022_D2_Results.md), [consolidated evidence](reports/native_augmented_D2_experiment_20261022.json), [256 validation](reports/validation_D2_capacity256_20261022.json), [512 validation](reports/validation_D2_capacity512_20261022.json), [tiny validation](reports/validation_D2_tiny_20261022.json), [independent answer audit](reports/validation_D2_saved_answers_20261022.json).

## Storage and code checks

The [latest user-authorized retirement](docs/experiments/20261022_Checkpoint_Retirement_After_D2.md) removes **19 superseded payload directories and 10 orphan pool weights**, after the D2 experiment evidence was archived. Completed capacity probes, intermediate D1/qualification exports, D0 update 12 and D2 update 4 are retired. Their raw reports, answers, profiles and 100 byte-verified small metadata archives remain. Archived runtime audits describe the pre-retirement files and cannot be rerun against deleted payloads.

The original sealed source and acquisition receipt are unchanged. Selected D0 update-8, D1 update-28 and latest D2 tiny update-8 **exports and resumable checkpoints** remain, as do both useful final negative exports. All 35 protected byte claims reverify, and all protected/source/raw file identities remain unchanged. No model was loaded or downloaded for cleanup. [Plan](reports/checkpoint_retirement_plan_after_D2_20261022.json), [completion journal](reports/checkpoint_retirement_after_D2_20261022.json).

Observed free space increased **123.24 GiB**, from **465.76 to 589.00 GiB**. These are system-wide filesystem observations, not unique APFS extent accounting. The initial guard stopped on ctime-only drift in one unused pool file; its fresh full hash matched its content address, and a narrow recovery finished the remaining orphan removals without changing the original guards. The initial journal and recovery record are preserved.

Four pool versions remain, all referenced by retained native files. APFS sharing continues for identical future saves. End each completed experiment by archiving evidence, retaining chosen results/restart points and useful negatives, then explicitly retiring superseded payloads and orphan cache objects. Unique learned weights and optimizer states consume real space; deduplication alone cannot bound their growth. The [earlier cleanup](docs/experiments/20261022_Checkpoint_Cleanup_Applied.md) remains a historical record.

The D2 package's full suite passed **479 tests**, with one restricted-context MPS skip. The subsequent retirement/recovery and storage helpers pass **75 focused tests**, including real tiny-file APFS checks, with no failures/skips. [Retirement validation](reports/validation_checkpoint_retirement_after_D2_20261022.json), [D2 test evidence](reports/validation_D2_code_20261022.json). Training-bound implementation files stayed unchanged. Source, weights, runs and caches remain outside Git, with the artifact guard enabled.

## Next concrete gate

Predeclare **S2.4**: the same eligible information, parents, target data, fair rate opportunities, and parameter/compute controls for D0/D1/D2, with equal-token and equal-total-compute views. Expand S1.2 beyond the current 20-episode/ten-world fixture family before claiming useful transfer. Existing D0/D1/D2 diagnostic passes do not justify the 20k campaign or model promotion. Early host-loop preparation may proceed separately; selected-system integration still depends on S2.5.

Cognitive contracts, identity/lineage, source permissions and the proposed semi-cognitive loop remain as specified in [the design contracts](docs/experiments/20261022_Cognitive_Contracts.md). Useful memory benefit, ten-cycle learning, authentic Replica fidelity, VLM, augmented KV caching, ANE acceleration and Blackwell residency remain separate open gates.
