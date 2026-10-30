# Spark fitting-only counterfactual diagnosis v13

Status: **native attempt03 completed; fusion placement is the selected single next mechanism.** No development labels were read and no parameters were updated. Useful recall, generalization, persistent learning and promotion remain unestablished.

## Execution and lineage

- Source: sealed official BF16 Spark, with original parameter hash unchanged.
- Parent: attempt10 extension checkpoint, loaded through an explicit validated parent-execution migration; the new diagnostic identity was restored immediately afterward.
- Implementation/authorization commit: `c05e2e08`.
- Attempt01 failed on parent checkpoint execution identity. Attempt02 passed migration, then failed because the artifact watchdog compared growth with partial rather than total planned bytes. Both outputs are preserved.
- Attempt03 completed with packet SHA-256 `1f8fa0100c4243ff3bcf11ce05ed5e9cc95c9d9f4be74dfd2cb05a1a85a78b07` and checkpoint manifest SHA-256 `379e0402448771de2002cbca4e0c10d552ca216600eed5e6386dd30734308d15`.
- Independent decision receipt: `reports/intrinsic_memory_counterfactual_v13/attempt03_mechanism_decision.json`.
- Research boundaries: `docs/research/spark_memory_20261029/Counterfactual_Diagnosis_V13_Crosscheck.md`.

## Measured controls

All 32 responses in every control terminated; truncation was zero in every control.

| Control | Correct | Interpretation |
|---|---:|---|
| Contextual | 24/32 | Same final answers as sealed v12 attempt10. |
| Complete-row permutation | 24/32 | Slot content and learned address move together; near-invariance is expected up to reduction order. |
| Raw slot/address shuffle | 24/32 | Content is reassigned to original addresses; sealed v12 shuffle interpretation is unchanged. |
| Wrong source | 6/32 | Semantic corruption of the answer-bearing fact. |
| Zero | 5/32 | Memory values zeroed after validation. |
| Absent | 8/32 | Memory disabled. |
| Exact text | 32/32 | Charged exact-evidence reference, not learned retention. |

Eight of sixteen same-question counterfactual pairs had both sides correct. Per-pair evidence below uses mean absolute slot/residual distance and maximum absolute first-token logit changes. These are diagnostics, not calibrated confidence or proof that a representation is sufficient.

| Pair | Both correct | Slot mean Δ | Residual mean Δ | Logit max Δ | Row-perm max Δ | Slot-shuffle max Δ |
|---|---:|---:|---:|---:|---:|---:|
| 00 | no | 0.08641 | 0.00347 | 1.875 | 0.375 | 0.375 |
| 01 | no | 0.11296 | 0.00407 | 2.250 | 0.250 | 0.250 |
| 02 | yes | 0.38675 | 0.01541 | 12.953 | 0.250 | 0.375 |
| 03 | yes | 0.43279 | 0.01634 | 13.828 | 0.375 | 0.375 |
| 04 | no | 0.10780 | 0.00390 | 1.875 | 0.375 | 0.375 |
| 05 | no | 0.12424 | 0.00426 | 2.688 | 0.250 | 0.312 |
| 06 | no | 0.44510 | 0.01847 | 11.750 | 0.375 | 0.500 |
| 07 | yes | 0.44280 | 0.01607 | 14.250 | 0.375 | 0.312 |
| 08 | no | 0.07053 | 0.00229 | 1.125 | 0.375 | 0.375 |
| 09 | no | 0.06737 | 0.00300 | 1.625 | 0.250 | 0.250 |
| 10 | yes | 0.32095 | 0.01512 | 13.844 | 0.375 | 0.500 |
| 11 | yes | 0.41785 | 0.01863 | 11.438 | 0.375 | 0.438 |
| 12 | yes | 0.10260 | 0.00307 | 2.625 | 0.375 | 0.375 |
| 13 | no | 0.09277 | 0.00309 | 2.375 | 0.375 | 0.375 |
| 14 | yes | 0.38634 | 0.01751 | 12.355 | 0.500 | 0.500 |
| 15 | yes | 0.38592 | 0.01504 | 15.375 | 0.375 | 0.375 |

Every pair has nonzero compressed-slot and reader-output distinctions; the minimum maximum-absolute residual distinction is 0.0019123. Mean reader/projection/residual RMS values are 0.997205, 0.123976 and 0.061907. Full per-pair source features, slots, reader outputs, residuals and full-vocabulary first-token distributions are preserved under `native_bf16_pair_diagnostic_attempt03/pairs/`. The artifact guard rejected the binary tensors for Git tracking, so they remain outside Git with exact hashes and byte counts in `reports/intrinsic_memory_counterfactual_v13/external_tensor_manifest.json`; JSON receipts and the decision are tracked.

## Decision

Select **pre-MLP fusion placement at the existing 3/7 sites** as the single next mechanism experiment. Keep frozen source features, four-slot compressor, slot count, objective, optimizer, fitting rows and development boundary fixed. Do not add LoRA, autoencoding, more slots or a second placement change.

Less-compressed source features are not the first choice because distinctions survive compression and reader output. Reader/query alignment remains the fallback if the placement arm fails. This diagnosis does not establish that pre-MLP fusion will help; that causal benefit is unmeasured.

## Cost and watchdog evidence

Attempt03 used 480 forwards, 448 generated tokens, 448 prompt tokenizations, 32 source encodes/source-feature forwards, one model load, zero updates and 35 writes. Raw artifacts are 136,894,288 bytes. Supervision wall was 171.032 seconds; peak sampled driver was 9,351,856,128 bytes, swap growth zero and minimum host free 74%. Driver/RSS overlap and sampling limits apply. The process group exited successfully with no watchdog stop.
