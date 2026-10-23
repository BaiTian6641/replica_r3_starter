# E3 action-kind head: bounded completions, head contribution inconclusive

23 October 2026 UTC. Protocol and mixed fixtures were committed before
execution. **Result: 4/4 held-out mixed episodes completed under head-
conditioned generation; 2/2 no-head control episodes also completed.** The
connected loop now demonstrates real repairs and independent evaluator passes,
but this run does **not** establish that the head improves action selection.

## 1. Arm

- Accepted E3 `set_constant` extension loaded frozen; sealed source reused.
- Mixed-outcome family: 4 train, 2 dev, 4 held-out; pass cases stop after a
  passing evaluator, repair cases change one constant after a failing test.
- 40 teacher prompt-only states; actual receipts; 16 train states only for head
  training.
- `ActionKindHead`: pooled workspace + exact task features → four kinds
  (`read`, `run_test`, `set_constant`, `stop`), 64 hidden, 240 AdamW steps,
  final loss **1.96e-5**. Head artifact **72 KiB**, no extension/base save.
- At evaluation the head selects a kind; the frozen model generates the JSON
  arguments under a short kind hint. The host validates and executes; it does
  not choose the repair.
- Budget: 7 decisions, 4 tool calls, 640 prompt / 128 generated tokens per
  episode; greedy, MPS, CPU fallback disabled.

Attempts 1 and 2 failed before episodes (batched-head validation; CPU previous
workspace on MPS). Both are preserved as implementation defects.

## 2. Head classification

| Role | States | Kind accuracy |
|---|---:|---:|
| Train | 16 | 1.00 |
| Dev | 8 | 1.00 |
| Held-out | 16 | 1.00 |

The earlier mixed probe's step-only baseline was 0.875 and class majority 0.375.
This table is small-sample kind accuracy, not action execution or calibration.

## 3. Actual episodes

| Case | Requires repair | Head arm | No-head control |
|---|---|---:|---:|
| `mixed_heldout_repair_07` | yes | complete | complete |
| `mixed_heldout_repair_08` | yes | complete | complete |
| `mixed_heldout_pass_09` | no | complete | complete |
| `mixed_heldout_pass_10` | no | complete | not run |

Head-conditioned totals: **4/4 success, 16 decisions, 20 tool calls, 0 malformed,
0 false success claims**. Controls: **2/2 success** with the same action-kind
sequence as their head-conditioned counterparts.

Repair traces:
- `read` → failing `run_test` → `set_constant` with the correct name/value →
  passing `run_test` → stop.
- Pass traces: `read` → passing `run_test` (repeated once) → stop.
- The fifth run_test is denied by the declared tool-call budget; the success
  condition is already satisfied, and stop is accepted.

All success decisions came from the independent evaluator, not model text. Raw
receipts and prompts are in
[the report](../../reports/e3_policy_head_attempt03_20261023/run_report.json).

## 4. Interpretation and limits

- This is a **bounded mixed-outcome task success**, not a general E3 capability
  or a core/head benefit result. Control cases used values/settings close to the
  accepted extension's training family; the family is synthetic and exposed.
- The head-conditioned and no-head control action sequences are identical. The
  head's kind predictions are accurate on 16 held-out states, but the hint did
  not cause a visible behavior change. Do not claim the head improves completion.
- The 4/4 result also shows the earlier set_constant 0/7 was not a universal
  tool/parser failure; task mixture and state matter.
- Success head remains untrained (this run has positive labels but no disjoint
  calibration protocol); no confidence calibration is claimed.
- No default changed. The head artifact is diagnostic, not published.

## 5. Next

1. Run a **head-disabled full** set on all four held-out cases (two controls are
   currently missing) and audit whether the head hint changes any decision.
2. Include **fresh task values** not close to accepted training families, plus a
   **step-only/kind-shuffled control**.
3. Only then decide whether to publish the head or keep exact text/lexical
   defaults. Any head publication needs a manifest, rollback and a new held-out
   family.

## 6. Costs

**138.4 s** profiled, **11.0 GiB** sampled peak driver, 240 head optimizer
steps, 72 KiB head artifact, no extension/base save, no download.
