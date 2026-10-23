# E3 mixed core controls: full path repairs; controls fail the repair

23 October 2026 UTC. Protocol and fixtures were committed before execution.
**Result: full-core/workspace completes 2/2 fresh prompt-evidence cases;
core-disabled, feed-forward and extra-inference controls complete 1/2 each.**
All three controls complete the already-passing case but fail the repair, with
rejected false stop claims. This is a bounded causal signal for the enabled
core/workspace path, not evidence that the recurrent core beats feed-forward.

## 1. Arm

- Fresh prompt-evidence cases with exact host-read constants:
  `prompt_heldout_repair_01` and `prompt_heldout_pass_03`.
- Accepted E3 `set_constant` extension frozen; no head, no training.
- Modes: `full`, `core_disabled`, `ff_core`, `extra_inference`; workspace
  history disabled; independent evaluator decides success.
- Budget: 7 decisions, 4 tool calls, **768 prompt** / 128 generated tokens.
  The first attempt used 640 and failed after the full episodes because the
  core-disabled prompt exceeded the cap; that attempt is preserved.

## 2. Results

| Mode | Pass case | Repair case | Malformed | False claims |
|---|---|---|---:|---:|
| full | complete | **complete** | 0 | 0 |
| core_disabled | complete | failed | 0 | 2 |
| ff_core | complete | failed | 0 | 1 |
| extra_inference | complete | failed | 0 | 2 |

The full repair action was the exact `set_constant(MAX_CONNECTIONS=40)` followed
by a passing evaluator. Controls never applied the repair; their stop claims were
rejected. The pass case succeeds in every mode, so the difference is specifically
the failure→repair transition, not tool access or basic JSON generation.

## 3. Interpretation

- **Task success vs added mechanism:** full succeeds 2/2; every control fails the
  repair. The enabled core/workspace path matters for this pair.
- **Recurrent-core-specific benefit:** not established. `ff_core` also fails the
  repair, so this arm cannot distinguish recurrent computation from the presence
  of the workspace/state path or other enabled components.
- n=2 cases; synthetic and near the accepted extension's training distribution.
  No general capability, calibration or personal-fidelity claim.
- No host repair choice: controls and full used model actions; the evaluator
  alone accepted success.

## 4. Costs

**178.5 s**, **11.0 GiB** sampled peak driver, no optimizer updates, no
checkpoint or download. Raw report:
[evidence](../../reports/e3_mixed_core_controls_attempt02/run_report.json).
