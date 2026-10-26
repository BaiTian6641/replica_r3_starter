# Memory update dynamics: write-position interference isolated

**25 October 2026.** The matched update-dynamics comparison isolates
**write-position/recency interference** as the first mechanism behind exact-key
retention failure. Shared-depth reduction is not the first cause. The next
justified action is a frozen-writer, order-aware/refresh diagnostic with the
same sources, candidates, vectors and scoring—not writer training or per-depth
promotion.

Final evidence: [independent review](../../reports/memory_update_dynamics_20261025/independent_review.md),
[native run](../../runs/memory_update_dynamics_20261025/native01/run_report.json),
and [guard](../../reports/memory_update_dynamics_20261025/native01.guard.json).

## Exact-key variants

| Variant | Top1 | Mean rank | Mean margin |
|---|---:|---:|---:|
| `shared_mean/full` | 0.15 | 3.30417 | −0.0250612 |
| `per_depth/full` | 0.15 | 3.27083 | −0.0452305 |
| `shared_mean/reversed` | 0.35 | 2.58750 | +0.0133272 |
| `per_depth/reversed` | 0.35 | 2.57083 | +0.00368282 |
| `shared_mean/target_and_later` | 0.15 | 3.24583 | −0.0208738 |
| `per_depth/target_and_later` | 0.15 | 3.24583 | −0.0421763 |
| `shared_mean/prefix_through_target` | **1.0** | **1.0** | +0.123672 |
| `per_depth/prefix_through_target` | **1.0** | **1.0** | +0.140749 |
| `shared_mean/target_only` | **1.0** | **1.0** | +0.171714 |
| `per_depth/target_only` | **1.0** | **1.0** | +0.172207 |
| `shared_mean/target_refresh_last` | **0.995833** | **1.004167** | +0.103245 |
| `per_depth/target_refresh_last` | **1.0** | **1.0** | +0.128073 |

Both full baselines recover 36/36 final-write targets and 0/204 non-final
targets. Removing later writes restores exact-key recovery in both reducers;
merely removing earlier writes does not. Refreshing the target last also passes
in both reducers. No forced-full gate variant approaches the exact recovery gate.

Query-derived addressing is invariant across all 20 variants: fact top1
232/240 = 0.966667, mean rank 1.03333, mean margin +0.0877312. Layer 13 exactly
reproduces the matched native03 baseline.

## Interpretation and next boundary

Mechanism precedence is correct: mechanical validity passes; per-depth does not
pass or improve, ruling out shared reduction as the first cause; prefix and
target-only pass and improve, selecting write position before refresh or gate
controls. `per_depth/prefix_through_target` wins a deterministic tie-break, not
a claim of per-depth superiority: shared prefix and both target-only variants
also reach top1 1.0 with rank 1.0. Refresh also passes, but it is not selected
because write position was isolated first. Target-dependent refresh/prefix arms
are evaluator-side interventions, not evidence that a production order or
re-observation policy is valid.

All 22,224 readout/ranking identities are unique. All 1,464 histories and 372
codebook entries match. Mechanical gates pass with zero updates and 42/42
production parity. Guard elapsed is 86.923 s, below 900 s. Peak sampled driver
is 11,815,763,968 bytes, swap growth zero and minimum host free 78%.

Next validate only an order-aware/refresh policy with the writer frozen and
identical sources, candidates, vectors and scoring. Do not enlarge the bank,
start another blind writer arm or declare useful learned memory.
