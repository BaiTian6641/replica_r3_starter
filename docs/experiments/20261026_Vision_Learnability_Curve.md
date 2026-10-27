# Vision learning curve protocol — training-only 0/1/2/4/7

**Status: executed at `2f0d0d9`; final decision `curve_no_optimization_progress`.
Development, held-out and V0 remain closed.** This is a bounded first-order
training diagnostic, not a source-disjoint fine-tuning pilot, generalization
test, V0 run or capability claim. Development, held-out and semantic-audit data
remain sealed.

## Why a new measurement is required

The earlier seven-update gradient arm completed 30 backward calls and changed
all six projector tensors, but failed train-fit: 0/14 correct answers, zero
answer-prefill gold accuracy and zero caption semantic F1. Its artifacts contain
no multi-prefix trajectory: the first and last losses are different rows, and
only two caption examples repeat within the prefix. Offline re-scoring cannot
recover a learning curve. A new in-process measurement is therefore required;
the failed arm is not repeated as a blind endpoint rerun.

## Frozen lineage

- Protocol: `configs/vision_learnability_curve_v1.json` at commit `f67b4ed`.
- Prepared data: `data/vision_balanced_alignment_v2_20261026/prepared04`.
- Protocol SHA-256:
  `3bf636f00dce4e73168eb5f4b2c2454560276b32dfb1122b91d58ffcd44c2d77`.
- Manifest SHA-256:
  `12f43b081e147a0c678e2137a321b709eb528099411b46b913735198563e0227`.
- Training alignment SHA-256:
  `be6d62e32e688b7e4fc14ff327ec41df7062747e13d4c4d041cd7d05caa00bd8`.
- Batches SHA-256:
  `08484a4dbc000488c94c54510fad655c06251598a87294837e24d80cad09433a`.
- Caption labels SHA-256:
  `84788b08bd6c3a66f9c241e18db73ca8828c8dd15b62aad83e8d3bdf235b1954`.
- Precondition: `spatial_contribution_observed` in
  `runs/vision_mean_only_ablation_v1_20261026_attempt09/report.json`.
- Sealed MiniCPM5 and SigLIP2 sources, revision
  `f97400052a43d642bbc6e9975e2397e3ae6a6b52`; no downloads.

The curve imports the historical learnability runner and module without editing
them. Three migration targets had drifted concurrently:
`architecture/contracts.py`, `architecture/minicpm5.py` and
`memory_system/session.py`. Their current pins are refreshed only in this new
curve; the two runner migrations remain unchanged. Historical protocols and
their existing stale-pin failures are not rewritten.

## Bounded schedule

One native process and one model load execute:

1. Verify prepared, source, migration and new-file pins and the ablation
   precondition.
2. Initialize a fresh seed-29 projector, freeze the backbone/encoder and create
   one `NativeFusedAdamW` optimizer at lr `3e-4`, weight decay `0.01`.
3. Run two backward-only probes (answer and caption), with zero optimizer steps
   and no weight change.
4. Measure prefix 0 before any optimizer step; its projector digest must equal
   the initialization digest and optimizer state must be empty.
5. Complete exactly seven updates, 28 update-backward calls and 30 total
   backward calls, then measure prefixes 1, 2, 4 and 7.

Every readout runs the identical 26-row × four-control matrix: 104 records, 98
available and six unavailable sibling placeholders. It uses one greedy
hypothesis per cell, prefill analysis, and the same teacher-forced loss path for
answers and captions. Answer loss targets `row.response`; caption loss targets
the sealed caption answer. Per-prefix answer/caption scores use the unchanged
`score_train_fit` implementation. Prefix-7 endpoint gates are reported only as endpoint context and never as the curve decision.


No intermediate checkpoint is needed or allowed. Scoring is forward-only on the
live in-memory projector. Each prefix digest must change across prefixes and
remain unchanged during its readout; optimizer steps/state and the sealed base
stamp must not change during scoring. `adapter.train()` is restored after every
readout. All per-prefix artifacts are written before the fixed-point report.
## Execution amendments before attempt 04

Three unsuccessful attempts are preserved. Attempt 01 used the baseline
interpreter and failed before updates because Pillow was absent. Attempt 02
used the existing `.venv-vision` overlay and reached prefix 0 but failed on a
missing `curriculum` import alias. Attempt 03 completed one update and one
readout, then stopped after 443.7 seconds when 13 early generated cells decoded
to empty text under the historical nonempty-text hypothesis rule. All three
had zero observed swap growth and no resource stop; none produced a complete
curve.

The `26b353c` change imports the projector digest helper directly and adds a
behavioral regression test. The `2f0d0d9` amendment adds the explicit
`empty_text_policy=score_as_wrong_and_record`: empty decoded responses remain
in the raw trajectory and count as incorrect, while empty IDs, stop errors and
prefill mismatches remain engineering failures. This changes no model weights,
data, threshold, endpoint gate or T1/T2/T3 decision rule. The curve suite now
passes **37 tests**.

Attempt 04 uses the overlay interpreter and a fresh output directory. The
original predeclaration and all failed artifacts remain preserved.


## Decision contract

The curve decision uses only T1, T2 and T3, in that order after engineering
acceptance:

- **T1 — descent:** answer and caption correct-image teacher-forced NLL each
  decrease by at least 25% from prefix 0 to prefix 7.
- **T2 — monotone:** both role NLL sequences are non-increasing at every step
  0→1→2→4→7 within absolute tolerance `1e-3`.
- **T3 — conversion:** at least one of:
  - answer prefill gold accuracy reaches at least `1/14` from zero;
  - answer correct accuracy reaches at least `1/14`;
  - caption token F1 improves by at least `0.10`;
  - mean answer-prefill gold rank falls to at most half its prefix-0 value.

The statuses are:

1. `stopped_engineering_failure` for any pin, access, trace, generation,
   resource, immutability or accounting failure.
2. `curve_readout_conversion_observed` for T1 ∧ T2 ∧ T3. A longer budget would
   require a separate predeclared hypothesis, not an automatic retry or V0.
3. `curve_optimization_only_no_conversion` for T1 ∧ T2 ∧ ¬T3. Extending the
   budget is not justified by this status.
4. `curve_no_optimization_progress` otherwise. The vision sequence stops here.

The trailing 4→7 slope is descriptive only and never a decision input. No status
authorizes V0, development, held-out evaluation, promotion or automatic retry.

## Limits and execution

- 5 readouts, 490 greedy generations, 490 teacher-forced cells and 1,470
  forward-only passes.
- 7 optimizer steps, 28 update-backward calls, 30 total backward calls and zero
  additional backward calls for readouts.
- 1 model load, 0 extension loads/writes, 0 checkpoint writes, 0 downloads and
  0 readout weight writes.
- At most 64 MiB of artifacts and 900 seconds. The measured prior round cost is
  about 65.6 seconds, giving an expected roughly 351-second total.
- One exclusive MPS worker, CPU fallback disabled, the unchanged 50 GiB driver /
  512 MiB swap-growth / 15% minimum-free guard and fresh output.

The declared native command is:

```bash
.venv-vision/bin/python scripts/run_capacity_preflight.py \
  --entry-point run_vision_learnability_curve.py \
  --report reports/vision_learnability_curve_v1_20261026_attempt04/guard.json -- \
  --config configs/vision_learnability_curve_v1.json \
  --output runs/vision_learnability_curve_v1_20261026_attempt04 \
  --device mps --model-path checkpoints/minicpm5_source \
  --model-lock configs/minicpm5_sealed_lock.local.json
```

The CPU contract suite passes **37 tests**. It covers config and pin drift,
readout invariants, all T3 clauses, every decision status and the unchanged
historical gradient failure behavior.

## Native result

Attempt 04 completed all seven updates and all five readouts. Engineering and
access checks pass: 30 backward calls, seven optimizer steps, one model load,
490 generations, 490 teacher-forced cells, 1,470 scoring forwards, zero
checkpoint/extension writes and zero downloads. The guard completed in 862.51
seconds with peak sampled driver 12,029,247,488 bytes, zero swap growth and 74%
minimum observed host free memory. The stop report's data-plus-report total is
4,061,064 bytes, below the 64 MiB cap.

| Prefix | Answer correct NLL | Caption correct NLL | Mean answer-prefill gold rank | Empty decoded responses |
|---:|---:|---:|---:|---:|
| 0 | 10.704499 | 9.773421 | 21,413.57 | 0 |
| 1 | 11.741904 | 13.448152 | 27,081.50 | 13 |
| 2 | 8.396319 | 7.109048 | 5,845.86 | 14 |
| 4 | 5.263936 | 4.999400 | 524.57 | 20 |
| 7 | 3.704158 | 2.963564 | 103.43 | 0 |

T1 passes: answer and caption NLL fall by 65.40% and 69.68%, respectively. T3
passes because the mean gold rank falls to 0.00483 of its prefix-0 value. The
other conversion criteria do not pass: correct accuracy and prefill gold
accuracy remain 0/14, and caption token-F1 gain is 0.02781 (<0.10).

T2 fails. Both role NLLs increase at prefix 1: answer 10.704499→11.741904 and
caption 9.773421→13.448152. Therefore the terminal decision is
`curve_no_optimization_progress`, not `curve_optimization_only_no_conversion`.
The prefix-1 increase is the reason for the disposition; the later descent and
gold-rank improvement do not erase it. No longer budget, development, held-out,
V0 or retry is authorized by this result.

The prefix-7 endpoint context is also negative: answer correct accuracy and
prefill gold accuracy are 0/14, caption semantic F1 is 0, and only two unique
answer texts remain. Those endpoint gates were not used as the curve decision.

Independent CPU replay of `validate_curve_readouts`,
`curve_trajectory_report` and `curve_decision` reproduces every T1/T2/T3 value
and the terminal status with no violations. The 37-test curve suite and the
unchanged historical failure behavior remain separate evidence.

## Preserved attempts and artifacts

| Attempt | Outcome |
|---|---|
| 01 | Baseline interpreter lacks Pillow; zero updates/readouts |
| 02 | Overlay reaches prefix 0; missing digest-import alias, zero updates/readouts |
| 03 | One update/readout; 13 empty decoded cells trigger the historical nonempty-text rule; 443.7 s |
| 04 | Complete five-prefix curve; `curve_no_optimization_progress`; 862.5 s |

Raw reports, prefix readouts, training trace, backward probes, guard logs and
profile samples are under `runs/vision_learnability_curve_v1_20261026*` and
`reports/vision_learnability_curve_v1_20261026*`. The completed receipt is
`runs/vision_learnability_curve_v1_20261026_attempt04/report.json`; its summary
is `reports/vision_learnability_curve_v1_20261026/summary.json`. No historical
failure was rewritten.
