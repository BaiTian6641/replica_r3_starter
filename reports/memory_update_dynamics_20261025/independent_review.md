# Independent update-dynamics outcome review

**Reviewer:** `UpdateDynamicsReview`, read-only raw-artifact and mechanism audit.

## Verdict

The evidence supports **write-position/recency interference**, with write
position as the first isolated mechanism. It does not support shared-depth
reduction as the first cause. It justifies order-aware/refresh-only diagnostic
work with the writer kept frozen, not writer training or per-depth promotion.

## Recomputed exact-key variants

All exact-key variants contain 240 fact rankings: 120 queries across layers 13
and 27.

| Variant | Top1 | Mean rank | Mean margin | Final-write top rate |
|---|---:|---:|---:|---:|
| `shared_mean/full` | 0.15 | 3.30417 | −0.0250612 | 0.966667 |
| `per_depth/full` | 0.15 | 3.27083 | −0.0452305 | 1.0 |
| `shared_mean/reversed` | 0.35 | 2.58750 | +0.0133272 | 0.975 |
| `per_depth/reversed` | 0.35 | 2.57083 | +0.00368282 | 1.0 |
| `shared_mean/target_and_later` | 0.15 | 3.24583 | −0.0208738 | 0.9875 |
| `per_depth/target_and_later` | 0.15 | 3.24583 | −0.0421763 | 1.0 |
| `shared_mean/prefix_through_target` | **1.0** | **1.0** | +0.123672 | 1.0 |
| `per_depth/prefix_through_target` | **1.0** | **1.0** | +0.140749 | 1.0 |
| `shared_mean/target_only` | **1.0** | **1.0** | +0.171714 | 1.0 |
| `per_depth/target_only` | **1.0** | **1.0** | +0.172207 | 1.0 |
| `shared_mean/target_refresh_last` | **0.995833** | **1.004167** | +0.103245 | 0.995833 |
| `per_depth/target_refresh_last` | **1.0** | **1.0** | +0.128073 | 1.0 |

Both full baselines recover 36/36 final-write targets and 0/204 non-final
targets. Per layer this is 18/18 versus 0/102. Removing later writes restores
exact-key recovery in both reducers; merely removing earlier writes does not.
Refreshing the target last also passes in both reducers.

No forced-full gate variant approaches the exact recovery gate; the best top1
is 0.225 for `shared_mean/forced_full_decay`. Query-derived addressing is
invariant across all 20 variants: fact top1 232/240 = 0.966667, mean rank
1.03333, mean margin +0.0877312. Layer 13 exactly reproduces matched native03's
baseline metrics.

## Integrity and precedence

All 22,224 readout/ranking identities are unique and consistent. All 1,464
histories re-derived from raw baseline/target state rows match. All 372
codebook entries and 120 repeated vector groups are invariant. Every 324
query-address group has identical raw readout hashes across variants.

Mechanism precedence is correct: mechanical validity passes; per-depth does not
pass or improve, ruling out shared reduction as first cause; prefix and
target-only pass and improve, selecting write position before refresh or gate
controls. `per_depth/prefix_through_target` wins a deterministic tie-break, not
a claim of per-depth superiority: shared prefix and both target-only variants
also reach top1 1.0 with rank 1.0. Refresh also passes, but `isolated_refresh_variant`
is correctly null because write position was isolated first.

## Execution

Zero optimizer updates and unchanged parameter versions. Mechanical gates pass:
42/42 production parity, 1,464 ordered-history parity rows, 1,212 candidate-
parity groups and all 22,224 inventory rows. Guard elapsed is 86.923 s, below
900 s. Peak sampled driver is 11,815,763,968 bytes, swap growth zero and minimum
host free 78%. These are sampled limits, not transient-OOM guarantees.

Next action is only a frozen-writer, order-aware/refresh diagnostic with source,
candidates, vectors and scoring fixed. Target-dependent refresh/prefix arms are
evaluator-side interventions; any production order or re-observation policy
requires separate validation. Useful learned memory remains unproven.
