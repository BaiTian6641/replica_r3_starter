# E3 candidate-ranking restart: exact state restored, 1/2 completed

23 October 2026 UTC. Protocol `configs/e3_candidate_restart_v1.json` committed
before execution. Two fresh held-out cases; accepted E3 extension; candidate
reranking; no training or optimizer update.

Process 1 used two decisions per case, persisted the exact journal and
workspace, and left each case after `read → run_test` with a failed test and no
success. Process 2 copied that host state into a fresh guarded output directory,
reopened the journal/workspace, and continued.

| Process | Cases | Completions | Notes |
|---|---:|---:|---|
| 1 | 2 | 0/2 | `read`, failed `run_test`; no false success |
| 2 resume | 2 | **1/2** | one evaluator pass; one budget-exhausted failure |

The completed case selected `set_constant`, reran the test twice, stopped, and
passed the independent evaluator with no false success claim. The failed case
selected the unchanged-value candidate, then repeated repair/test actions and
exhausted the five-decision continuation budget; it produced one false stop
claim and did not pass the evaluator.

Costs: process 1 **11.5 s**; process 2 **26.6 s**, peak **15.14 GiB**. No base
save, no checkpoint, no default promotion.

## Interpretation

This verifies fresh-process state reconstruction and continuation for the
candidate-ranking interface in one case. It also shows a ranking failure on a
second case: the model preferred the current value over the required bound.
Restart correctness is distinct from completion success. The family is small and
now exposed; no generalization claim follows.
Evidence: [process 1](../../reports/e3_candidate_restart_process1/partial_report.json),
[process 2](../../reports/e3_candidate_restart_process2/partial_report.json).
