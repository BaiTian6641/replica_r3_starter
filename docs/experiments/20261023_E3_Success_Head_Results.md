# E3 pooled success head: trained diagnostic, not calibrated

23 October 2026 UTC. **Result: trained on actual outcomes; case-held-out
accuracy 0.8125 vs 0.7778 majority, Brier 0.342; no calibration or authority.**
This closes the “success head remains untrained” gap only at the diagnostic
level. It is not published and cannot declare task success.

## 1. What was fixed and trained

The older success-head driver assumed `feature_vector` contained full workspace
slots (24 + 8×256). Runtime summaries actually carry **24 task features + one
pooled 256-wide workspace**. The old reshape could not train; it was replaced
with `PooledTaskOutcomeHead`, whose input exactly matches the runtime summary.

- 18 actual episode outcomes from the mixed, prompt-evidence, fresh and restart
  reports: **14 positive / 4 negative**, 12 case IDs.
- Inputs: 24 exact task features + pooled workspace[256].
- Model: 64-hidden MLP, 300 AdamW steps, seed 20261023.
- Evaluation: leave-one-case-out; no disjoint calibration role.
- Artifact: `runs/e3_success_head_attempt02/success_head.safetensors` (small
  diagnostic; not tracked, no default publication).

## 2. Metrics

| Metric | Value |
|---|---:|
| Case-held-out mean accuracy | **0.8125** |
| Majority-class baseline | 0.7778 |
| Brier score | **0.342** |
| Calibration | none |

The aggregate gain is small. The important failure is visible in the
`prompt_heldout_repair_01` fold: that case has four actual episodes (full,
core-disabled, feed-forward, extra-inference), three failures and one success,
and the head scores only **0.25** accuracy with probabilities near 0.95–0.999.
It cannot distinguish failures within the same case from its pooled features.
This is an **unqualified monitor**, not a calibrated task-success estimator.

## 3. Authority and limits

- The head never authorizes completion, changes verification, writes memory or
  overrides the goal.
- No disjoint calibration set exists; the Brier/accuracy numbers are diagnostic
  only.
- The runtime remains correct when the head is absent/untrained; the null path
  is unchanged.
- The result partly reflects mode/feature correlation and small sample size.
  More data and a case/topic-held-out calibration design are required before any
  capability estimate.

## 4. Costs

CPU-only training: no native model load, no optimizer checkpoint, no base copy.
Full suite after the change: **1129 passed / 21 subtests**.

Evidence: [run report](../../reports/e3_success_head_attempt02/run_report.json),
[manifest](../../reports/e3_success_head_attempt02/manifest.json).
