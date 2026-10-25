# Independent balanced image-dependence outcome review

**Reviewer:** `BalancedDiagnosticReview`, read-only re-audit of completed artifacts.

## Verdict

No material discrepancies. The evidence supports the narrow predeclared verdict:
frozen features show incremental image dependence on this procedural corpus on
both dev and the untouched final holdout. It does not establish uniform five-task
competence, real-image perception, or a training/projector claim.

| Split | Primary | Question-only | Shuffled primary | Gain over question-only [95% world CI] | Gain over shuffled [95% world CI] |
|---|---:|---:|---:|---:|---:|
| Dev | 71/120 = 0.591667 | 36/120 = 0.300000 | 50/120 = 0.416667 | 0.291667 [0.216667, 0.366667] | 0.175000 [0.083333, 0.258333] |
| Final | 75/120 = 0.625000 | 36/120 = 0.300000 | 46/120 = 0.383333 | 0.325000 [0.258333, 0.383542] | 0.241667 [0.141667, 0.350000] |

The reviewer independently recomputed every point and interval from prediction
rows: 60 worlds, 2,000 resamples, 95% confidence.

## Integrity evidence

- Both splits have 120/120 same-question donors, 120 unique donors, zero
  cross-task donors and zero fixed points. Permutation hashes match.
- Every expected label, task and world matches verifier truth. Feature arrays are
  finite and match indexed shapes/dtypes/value hashes: pooled `[360,768]`,
  regions `[360,4,768]`, questions `[360,2048]`.
- All 15 implementation and five data pins match. Workload is 360 observations,
  encoder passes and question encodes, with zero updates. Encoder parameter
  stamps are unchanged.
- The guard completed without a stop reason or swap growth. Peak sampled driver
  was 11,815,763,968 bytes and minimum host free was 76%.
- Dev contains no final scores or predictions. Final scoring required the exact
  completed dev/run reports and hash. The sole global receipt completed and pins
  the actual final-report hash; repeat finalization left no second report.

## Interpretation limits

- Spatial leads the aggregate at 24/24 on both splits. Only spatial has strictly
  positive per-task lower bounds for both gains in final; presence and shape have
  exactly zero shuffled gain. The supported verdict is aggregate, not per-task.
- Question-only is a weak global comparator (0.30, including 0.0 on shape), so
  the same-primary shuffled comparison is the cleaner incremental-image evidence.
- Same-question shuffling includes 4 dev and 12 final pair-twin donors. This is
  allowed by the declared control but is not strictly cross-world.
- Totals 3→2 and 8→5 still uniquely determine labels on 12/72 count rows,
  although the original complete total-to-label mapping is gone.
- Eight questions per split (24 total) limit lexical diversity; 12/24 exact
  strings also occur in V1. Scenes are small synthetic shape layouts.
