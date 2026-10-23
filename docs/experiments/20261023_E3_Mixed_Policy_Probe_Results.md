# E3 mixed-outcome probe: action-kind information survives, policy not built

23 October 2026 UTC. Protocol and fixtures were committed before execution.
**Result: representation signal under the declared margin; not a policy or
task-completion result.** The step-confounded probe was superseded by a family
where pass cases stop after verification and repair cases continue after a
failing test.

## 1. Design and receipts

- 10 fresh synthetic cases: 4 train, 2 dev, 4 held-out; disjoint source groups.
- Half initially pass the independent evaluator; half require one bounded
  `set_constant` repair. Every initial and repaired state was checked by the
  actual evaluator.
- Teacher labels are host-authored but come from actual tool receipts:
  `read -> run_test -> stop` for pass cases;
  `read -> run_test -> set_constant -> run_test -> stop` for repair cases.
- 40 prompt-only states: 16 train, 8 dev, 16 held-out; max prompt 572 tokens.
- 64-hidden MLP, 240 AdamW steps, trained on train states only; no model
  optimizer update, generation or checkpoint.

## 2. Results

| Feature set | Train | Dev | Held-out |
|---|---:|---:|---:|
| Workspace + task | 1.00 | 1.00 | 1.00 |
| Workspace only | 1.00 | 1.00 | 1.00 |
| Task features only | 1.00 | 1.00 | 1.00 |
| Step-only baseline | 0.875 | 0.875 | 0.875 |
| Held-out class majority | — | — | 0.375 |

The joint held-out score exceeds the step-only baseline by 0.125, just above the
declared 0.10 margin. The mixed labels occur at the post-test step: `stop` for
passing cases and `set_constant` for failing cases. This establishes that the
prompt-only state carries information distinguishing those outcomes; it does not
show that the model can generate the correct action arguments or complete the
task.

## 3. Scope and caution

- The held-out set has only 16 examples and the nonlinear probe can memorize
  small training data; all feature subsets reach 1.00.
- The probe predicts **kind**, not tool choice/arguments, execution, feedback
  incorporation or task success.
- No policy head, constrained decoder or extension was added. The accepted E3
  extension and defaults remain unchanged.
- The earlier step-confounded probe is preserved separately.

## 4. Costs

**15.2 s** profiled, **11.0 GiB** sampled peak driver, no optimizer updates, no
generated tokens, no checkpoint. Raw teacher receipts, probes and report:
[evidence](../../reports/e3_mixed_probe_attempt01_20261023/).

## 5. Next

The next E3 package must train a separately declared policy head and evaluate
actual generated arguments and full task episodes on fresh mixed cases, with a
step-only baseline and no host choosing the repair. This probe is not sufficient
for deployment.
