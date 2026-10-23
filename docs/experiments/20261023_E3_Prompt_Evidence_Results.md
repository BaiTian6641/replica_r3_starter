# E3 exact constant evidence: 4/4 fresh completions

23 October 2026 UTC. Protocol and prompt change committed before execution.
**Result: 4/4 fresh held-out completions**, with the model still selecting the
constant/value and the independent evaluator deciding success. This is a
narrower, easier interface, not a new general capability.

## 1. Change

The runtime prompt now includes one host-read evidence line when `set_constant`
is allowed:

`EXACT WORKSPACE CONSTANTS (host-read current values; no expected bound supplied):
- NAME=current, ...`

The list is parsed from the current task-owned workspace entry. It contains all
names and current values, never the evaluator's expected bound, a proposed
repair, or a host choice. The model still emits the complete action JSON and the
host still validates/executes it. No training or head was used.

The previous fresh family miss was a new-constant name typo
(`QUUEUE_DEPTH`/`QUUE_DEPTH`); this arm tests whether exact context fixes that
boundary.

## 2. Fresh family and result

Four new held-out cases with unseen constants `MAX_CONNECTIONS`, `BATCH_SIZE`,
`IDLE_TIMEOUT_SECONDS`; two repair, two pass. Initial evaluator states were
checked; the accepted E3 extension stayed frozen; `workspace_zero`, greedy MPS,
7 decisions, 4 tool calls, 640/128 token caps.

| Case | Required | Result |
|---|---|---|
| `prompt_heldout_repair_01` | `MAX_CONNECTIONS` 10→40 | **complete** |
| `prompt_heldout_repair_02` | `BATCH_SIZE` 8→64 | **complete** |
| `prompt_heldout_pass_03` | none | **complete** |
| `prompt_heldout_pass_04` | none | **complete** |

Totals: **4/4 complete, 0 malformed, 0 false success claims**. Repair actions
used the exact names and correct values; all evaluator passes were independent.
The fifth run_test on successes was denied by the tool-call budget after the
success condition was met; stop was then accepted.

## 3. Interpretation and limits

- The prompt context changes the difficulty: exact identifiers are now supplied
  by the host from the workspace. This is a **narrower evidence interface**, not
  free-form name generation. Report it as such.
- The host does not choose the name or value; all candidates are listed and the
  model selected the repair.
- One fresh family with four cases; no broad generalization, core-benefit,
  confidence or personal-fidelity claim.
- The prior 3/4 fresh result remains valid evidence for the free-form-argument
  boundary without this context.

## 4. Costs

**84.0 s**, **11.0 GiB** sampled peak driver, no optimizer updates, no
checkpoint, no download. Raw report:
[evidence](../../reports/e3_prompt_evidence_attempt01_20261023/run_report.json).
