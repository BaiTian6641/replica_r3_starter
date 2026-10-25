# Balanced image-dependence diagnostic: supported on procedural holdouts

**25 October 2026.** The fresh paired diagnostic passes its preregistered
image-dependence gate on development and the untouched final holdout. The claim
is narrow: frozen SigLIP2 features provide incremental visual information on this
owned procedural corpus. It is not uniform five-task competence, real-image
perception, or evidence that a projector has learned to use that information.

Final evidence: [qualification](../../reports/vision_dependence_20261025/qualification.json),
[independent review](../../reports/vision_dependence_20261025/independent_review.md),
[dev report](../../runs/vision_dependence_20261025/dev01/dev_report.json), and
[final report](../../runs/vision_dependence_20261025/final01/final_report.json).

## Data and controls

The corpus contains 180 worlds and 360 images: 12 worlds per task per split and
120 rows per split. Each pair has the same question, one decisive visual change
and different verified answers. Count varies from 2–5, presence and spatial
labels are balanced, and train covers every evaluation label. Target and
 distractor object descriptors and exact distractor compositions are disjoint
across train/dev/final. The old V1 cases are excluded.

The primary readout is `question_conditioned_region`: 2×2 region means routed to
per-task ridge models using question text only. Comparators include question-only,
pooled image, pooled image+question, region image and region image+question.
Shuffling uses same-question donors with no cross-task donors or fixed points.
Uncertainty uses 2,000 world-clustered bootstrap resamples at 95% confidence.

| Split | Primary | Question-only | Shuffled primary | Gain over question-only [95% CI] | Gain over shuffled [95% CI] |
|---|---:|---:|---:|---:|---:|
| Dev | 71/120 = 59.17% | 36/120 = 30.00% | 50/120 = 41.67% | 29.17 pp [21.67, 36.67] | 17.50 pp [8.33, 25.83] |
| Final | 75/120 = 62.50% | 36/120 = 30.00% | 46/120 = 38.33% | 32.50 pp [25.83, 38.35] | 24.17 pp [14.17, 35.00] |

Both splits exceed the declared 5-point mean-gain threshold with strictly
positive lower bounds. Finalization required the exact completed dev hash and a
completed native run report. A second finalization was refused.

## Interpretation

Spatial accuracy is 24/24 on both splits and leads the aggregate. It is the only
final task with strictly positive per-task lower bounds for both gains; presence
and shape have zero shuffled gain. The supported verdict is therefore aggregate,
not independently established for every task. Same-primary shuffling is the
cleanest comparison because the global question-only model is weak.

Same-question shuffling includes 4 dev and 12 final pair-twin donors, so it is
not strictly cross-world. Totals 3 and 8 still uniquely determine labels on
12/72 count rows, although the original complete total-to-label mapping is gone.
Eight questions per split limit lexical diversity; 12/24 exact strings also
occur in V1. These limitations constrain the claim to this diagnostic family.

## Engineering and cost

The native stage completed 360 observations, encoder passes and question encodes
with zero updates. Native elapsed was 28.620 s and guard elapsed 31.657 s. Peak
sampled driver was 11,815,763,968 bytes, swap growth zero and minimum host free
76%. Feature storage is 8,574,224 bytes. Dev artifacts occupy 8.5 MiB and final
artifacts 112 KiB. The full suite passes **1,706 tests +21 subtests in 207.54 s**.

## Next boundary

The diagnostic gate for image-dependent information passes. It does not by itself
justify another large V1 arm. Next run the matched-episode memory localization,
then design the staged projector curriculum from this evidence and the admitted
30-root external caption corpus. Any training must still show fresh-world visual
gains, wrong/no-image sensitivity, adequate abstention, text retention and
fresh-process reproduction. Muon-8bit remains the measured optimizer route; an
optimizer comparison, if retained as a hypothesis, must use equal data/compute
and a tuned AdamW control rather than the rejected SGD shortcut.
