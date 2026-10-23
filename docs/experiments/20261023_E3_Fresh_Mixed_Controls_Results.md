# E3 fresh mixed family: 3/4 completions, argument typo on the miss

23 October 2026 UTC. Protocol and fixtures were committed before execution.
**Result: 3/4 no-head held-out completions** on constant names and values absent
from accepted training. The miss is an argument-generation spelling error, not
a tool/parser or eligibility failure. No head or training was used.

## 1. Fresh family

Four held-out cases with new constants (`COOLDOWN_SECONDS`, `QUEUE_DEPTH`) and
new values; two require one repair, two already pass. Source groups are distinct.
Initial states were checked by the independent evaluator before the run.
Budget: 7 decisions, 4 tool calls, 640 prompt / 128 generated tokens, greedy MPS,
`workspace_zero`, accepted frozen E3 extension.

## 2. Results

| Case | Required | Outcome | Actions |
|---|---|---|---|
| `fresh_heldout_repair_01` | repair | **complete** | read → fail → `COOLDOWN_SECONDS=7.5` → pass → stop |
| `fresh_heldout_repair_02` | repair | **failed** | read → fail → `QUUEUE_DEPTH=12` (invalid) → invalid → stop → `QUUE_DEPTH=12` (invalid) → stop |
| `fresh_heldout_pass_03` | none | **complete** | read → pass → stop |
| `fresh_heldout_pass_04` | none | **complete** | read → pass → stop |

Totals: **3/4 completed**, **1 malformed action**, **2 false stop claims**;
the independent evaluator rejected both stop claims on the miss. No model-written
success statement was accepted. The repeated extra `run_test` on successes was
denied by the cumulative tool-call bound after the success condition was met.

## 3. Diagnosis

The correct value (`12`) was generated, but the constant name was misspelled
twice. This is an **argument-generation/generalization defect** for a new
identifier, not:

- retrieval/eligibility (the file and test were read/executed);
- tool schema/validation (the host rejected the unknown name correctly);
- task-state or restart (no restart in this arm);
- false authority (the host rejected the stop claims).

It also shows the earlier mixed 4/4 success was partly distribution-specific.
The next bounded arm should target exact-name copying/verification from the
source file, e.g. a constrained name-selection interface or a read-back check,
without letting the host choose the repair value.

## 4. Status and costs

- **Implemented/qualified:** fresh fixtures, initial-status validation, actual
  tools, no-head control path, independent success authority.
- **Trained:** none in this arm.
- **Evaluated:** 3/4 held-out.
- **Not established:** general capability, new-identifier copy, head benefit.
- Cost: **89.1 s**, **11.0 GiB** sampled peak driver, no optimizer updates, no
  checkpoint, no download.

Evidence: [run report](../../reports/e3_mixed_fresh_controls_attempt01_20261023/run_report.json).
