# E3 action-policy probe: inconclusive by step-deterministic labels

23 October 2026 UTC. Protocol and probe were committed before execution.
**Result: evaluation ambiguity, not a policy success.** The diagnostic reports
100% action-kind accuracy for joint, workspace-only and task-only features on
train, dev and held-out. Every teacher trajectory has the same five-step action
order (`read`, `run_test`, `set_constant`, `run_test`, `stop`), so step position
alone predicts the label; the probe cannot separate representation from
chronology. No policy head was deployed or promoted.

## 1. Implementation path

- Accepted E3 `set_constant` extension loaded frozen; base sealed source reused.
- Eight teacher trajectories generated from actual fixture receipts: 4 train,
  1 dev, 3 held-out; 40 examples total; max prompt 565 tokens.
- Prompt-only core state extracted through `MemorySession.prepare` with no
  previous workspace and no teacher suffix; a 64-hidden MLP was trained only on
  train states (240 AdamW steps, seed 20261023).
- No model optimizer update, generation, checkpoint or extension save.

Attempt 1 failed before extraction because teacher `TaskGraph` objects were not
bound into examples. Attempt 2 failed because selections were retrieved before
all policy admissions, then invalidated by the ledger epoch advance. Both are
implementation defects preserved as `failure_report.json`; attempt 3 completed.

## 2. Raw result

| Feature set | Train | Dev | Held-out |
|---|---:|---:|---:|
| Workspace + task | 1.00 | 1.00 | 1.00 |
| Workspace only | 1.00 | 1.00 | 1.00 |
| Task features only | 1.00 | 1.00 | 1.00 |

Train loss reached **4.17e-7**; held-out class majority is 0.40. Labels:
8 `read`, 16 `run_test`, 8 `set_constant`, 8 `stop`.

A workspace-only or task-only classifier also reaches 100% because both encode
the step/action history. Held-out cases repeat the same action sequence. This
does not establish that the state can choose a different action when the
observed outcome changes.

## 3. Decision and next test

The preregistered threshold mechanically fired (`representation_signal`), but
the test design is invalid for the intended claim. Treat the result as
**inconclusive/evaluation ambiguous**; do not build a policy head from it. The
next probe must use a **mixed-outcome family** where the same step has different
correct next actions (for example, `stop` after an already-passing test versus
`set_constant` after a failing test), and must report a step-only baseline.

The accepted E3 engineering extension remains unchanged. No completion,
confidence or policy claim is added.

## 4. Costs

**15.2 s** profiled, **10.77 GiB** sampled peak driver, zero optimizer updates,
zero generated tokens, no checkpoint. Raw report and both failures are in
[the evidence directory](../../reports/e3_policy_probe_attempt03_20261023/).
