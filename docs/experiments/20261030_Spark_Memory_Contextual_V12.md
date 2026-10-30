# Spark contextual source features and intrinsic memory v12

Status: **native attempt01 is running under one exclusive worker.** CPU engineering is independently qualified. No fitting, development, useful-recall, persistent-learning, promotion, density, or RSI result is inferred from that qualification.

## Mechanism and fixed boundaries

V12 replaces random byte-context values with frozen canonical-source decoder hidden states followed by live trainable compression into four ordered width-256 slots per admitted span, with at most two spans/eight slots. The source pass sees canonical text only, without questions, answers, templates, external spine or intrinsic-memory extensions. Original Spark parameters remain frozen; only the compressor, order and readers train through NativeFusedAdamW.

Reader sites remain 3/7 with block-input queries and post-block residuals. The frozen source forward is the only intended source-side detach. Durable admission/read/checkpoint serialization detaches separately; live response loss reaches the compressor, order and every active reader after zero-initialized reader outputs begin updating. No base weights, task state or source-feature cache is serialized.

The experiment protocol and exact packet are `configs/spark_intrinsic_coverage_v12.json` and `reports/intrinsic_memory_native_v12/native_packet_two_v12.json` (SHA-256 `6dc96d507bee85d807657cf81187146ab226d7d0146c6abf3d4f56f9377bb6e1`). The one-attempt receipt is `authorization_two_v12.json`; attempt marker and output are unique. No v11 output or receipt is reused.

## Exposure and evaluation roles

Calibration and fitting reuse the already exposed v11 rows for the matched mechanism comparison. The v11 development split was read by a delegated reviewer and is retired, not treated as fresh. The new development set contains 16 cases in two independent groups of eight and uses two held-out compound rule families:

- `partner_colour`: relation + partner attribute;
- `same_colour`: equality relation + anchor attribute.

The generator uses an independent seed to derive assignments, presentation order, unknown sides and palette rotation. Unknown cases contain no answer-bearing target colour. Public metadata exposes identities, roots, seed and plan digest, not labels. This is compound-rule-family holdout evidence, not broad compositional generalization or independent-seed replication.

Historical v11 results remain unchanged: fitting failed at 26/32 with groups 5/8, 6/8, 7/8, 8/8, and development stayed closed. V12 has no utility result yet.

## Native protocol and gates

1. Frozen complete-context calibration must pass 8/8 with zero false recall and no truncation.
2. The complete frozen fitting teacher must pass 32/32 before detached top-64 targets are admitted.
3. Exactly 512 fixed fitting updates run in the declared order. Only the final prefix may open development; every eight-world fitting group must independently pass.
4. Fitting freezes and teacher targets evict before development. Both groups and every compound rule family must independently achieve contextual ≥6/8, zero false recall, at most one answer behind exact text, and strictly better than wrong, shuffled, zero and absent controls.
5. Correction and revoked-read behavior are evaluated separately from cache invalidation.
6. Only extension plus optimizer state is saved. A fresh process must restore exact logits and perform one actual continuation update with original parameter hashes unchanged.

The current envelope has 23 charged stages, 6,955 seconds inside the 7,200-second outer ceiling, 9,778 forwards, 949 frozen source-feature forwards, 513 backward/updates, 7,256 generated tokens, 3,618 writes, 55,439,767,209 reserved read bytes, and 205,537,280 artifact bytes. Final probes reserve 580 writes (579 for earlier probes); restart reserves 16 writes. One durable read reserves 13 descriptor writes including failure fallbacks. Native uses one exclusive MPS worker, 50 GiB sampled driver, 512 MiB sustained swap growth and 15% minimum host free, with CPU fallback disabled.

## Independent CPU qualification

`reports/intrinsic_memory_review_v12/final_launch_gate/` records the positive verdict: **CPU engineering is qualified for the reviewed source-free contracts, and the native packet is ready for authorization review**. It is not a native capability claim.

Evidence includes:

- 1,024 exact clean-HEAD tests with zero skips and no overlays;
- origin/compound/budget checks 5/5 and the failed-persistence reproduction 1/1;
- integrated tiny-BF16 2/9/36 gradients, disabled parity, two-span live/release and cold value/logit parity, and exact uninterrupted fresh-process continuation;
- migration 1482/1482, final product 552/552, final-product bundle 293 inputs, and artifact guard 17,835 files;
- native static environment/closure review without model import/load.

The failed ticket fallback uses charged persistence with quota and observed receipt bytes. Dual rebuild has one owner: outer rebuild reserves zero duplicate writes and the actual per-world total is 18.

Earlier negative reviews are preserved and were not rewritten. Their defects—release authority, cache identity, checkpoint ownership, transactional formation, P revocation, receipt durability, split exposure, work accounting, and native stage lifecycle—were repaired and independently re-tested before this attempt.

## Research cross-check

`docs/research/spark_memory_20261029/Contextual_Compression_V12_Crosscheck.md` distinguishes the candidate from ICAE, AutoCompressors and gist tokens: those systems train different encoders/objectives at different scales and do not predict four-slot Spark readout.

`reports/intrinsic_memory_takeover_20261029/audit_20261030.md` records additional primary sources:

- Lee et al., ACL 2022: name-swapped templated examples are near-duplicates, so new names alone do not establish independent evaluation.
- Søgaard et al., EACL 2021: random splits overestimate performance.
- Keysers et al., ICLR 2020: generalization requires compound/dependency holdout, not atom substitution.
- Moyer (LWN): file fsync → rename → parent-directory fsync.
- SQLite atomic commit: multi-file transactions require an explicit commit record, not a sequence of renames.
- Apple `fsync(2)`: plain fsync is not a drive-cache durability guarantee; Darwin publication uses `F_FULLFSYNC` where supported.
- PyTorch MPS synchronization: waits for MPS kernels; `PYTORCH_ENABLE_MPS_FALLBACK=0` is also required for device-locality claims.

These sources motivate controls, not expected accuracy or MPS cost.

## Running evidence locations

- Native protocol, packet, authorization and preparation lineage: `reports/intrinsic_memory_native_v12/`.
- Independent positive verdict: `reports/intrinsic_memory_review_v12/final_launch_gate/`.
- Producer implementation and CPU workflows: `reports/intrinsic_memory_contextual_v12/`.
- Density controls: `reports/intrinsic_memory_density_v12/coordinator13/`.
- Replica formation/chunk/tier contracts: `reports/replica_successors_v12/`.
- Repair and research register: `reports/intrinsic_memory_takeover_20261029/`.

Native raw outputs, answers, controls, resource samples, stage receipts and failure traces will be appended after attempt01 completes. Until then, native competence, fitting, useful recall, density and persistent learning remain unmeasured.
