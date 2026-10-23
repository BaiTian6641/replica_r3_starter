# E3 mixed-case fresh-process continuation: complete

23 October 2026 UTC. Protocol committed before execution. **Result: 12/12
acceptance checks pass.** Process 1 observed the failing evaluator and stopped
after two decisions; a new process resumed the same exact journal, applied the
single repair, passed the independent evaluator and did not replay completed
actions.

## 1. Setup

| Item | Value |
|---|---|
| Case | copied `mixed_heldout_repair_08` (`POLL_ATTEMPTS` 2 → 7) |
| Source fixture | unchanged `fixtures/e3_mixed/cases/mixed_heldout_repair_08` |
| Extension | accepted E3 `set_constant` extension, frozen |
| Modes | both processes `workspace_zero`; previous workspace not used |
| Process 1 | max 2 decisions of 7 cumulative |
| Process 2 | same journal/ledger/workspace; max 5 more decisions |
| Budget | 7 decisions, 4 tool calls, 640 prompt / 128 generated tokens per decision |
| Storage | no extension save, no base copy, no download |

Attempt 1 used `mode=full`; the model repeated `read` instead of reaching the
test and the run is preserved at
`runs/e3_mixed_restart_attempt01`. The protocol was corrected to
`workspace_zero` before attempt 2; the final run is attempt 2.

## 2. Observed trajectory

| Process | Actions | Outcome |
|---|---|---|
| 1 | `read`; `run_test` | evaluator exit **1**; success false; no repair |
| 2 | `set_constant(POLL_ATTEMPTS=7)`; `run_test`; extra `run_test`; `stop` | exit **0**; success true |

Process 2 saw no replayed action IDs from process 1. The extra `run_test` was
denied by the cumulative tool-call budget; the success condition was already
satisfied, and the stop was accepted by the independent verifier. There were no
malformed actions and no false success claims.

## 3. Acceptance checks

All **12/12** pass:

- process 1 observed a nonzero evaluator exit;
- process 1 used exactly two decisions, was incomplete and did not repair;
- process 2 completed using the same journal/ledger/workspace;
- process 2 did not re-execute completed process-1 actions;
- exactly one `set_constant` repair landed;
- the copied workspace equals the reference repair;
- the evaluator was unmodified;
- the independent evaluator passed;
- no malformed action or false success claim was accepted;
- the final journal has a satisfied condition.

Raw evidence: [acceptance](../../reports/e3_mixed_restart_attempt02/acceptance.json),
[process 1](../../reports/e3_mixed_restart_attempt02/process1_session_report.json),
[process 2](../../reports/e3_mixed_restart_attempt02/process2_session_report.json),
[process 1 guard](../../reports/e3_mixed_restart_attempt02/process1_guard.json),
[process 2 guard](../../reports/e3_mixed_restart_attempt02/process2_guard.json).

## 4. Costs and limits

| Measure | Process 1 | Process 2 |
|---|---:|---:|
| Decision wall time | 6.32 s | 15.63 s |
| Guard wall time | 13.5 s | 22.6 s |
| Sampled peak MPS driver | 11.0 GiB | 10.02 GiB |
| Optimizer updates | 0 | 0 |
| Checkpoint saves | 0 | 0 |

This is one copied synthetic mixed case, not a generalization or personal-
fidelity result. It demonstrates exact task-state reconstruction, no completed-
action replay, one real repair and independent verification after a fresh
process. The success head remains untrained and uncalibrated.

## 5. Resulting E3 status

- Mixed held-out task completion: **4/4 no-head controls**, **4/4 head-conditioned**.
- Fresh-process continuation: **12/12 acceptance checks** on the interrupted
  mixed repair.
- Action-kind head: rejected as an improvement candidate (identical action
  sequences with no-head controls).
- Remaining: broader fresh task families, success-head training with disjoint
  calibration, and general-capability claims.
