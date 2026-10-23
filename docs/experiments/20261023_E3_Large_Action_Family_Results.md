# E3 large supervised action family: still 0 completions

23 October 2026 UTC. Protocol `configs/e3_set_large_v1.json` committed before
execution. Fresh 12-train/3-dev/6-held-out/1-interrupted `set_constant` family
with disjoint source groups; modules-only training from the E2 parent; no base
save or optimizer checkpoint.

**Result: 0/6 held-out and 0/3 development completions.** Every full-model
episode emitted `read` seven times. Core-disabled, feed-forward and
extra-inference controls also failed. The teacher-forced loss fell to **0.0276**
(recurrent) and **0.1377** (feed-forward), so this is not insufficient training
loss or a tiny data set. The model still does not select `set_constant` after
observing the failed test.

| Arm | Cases | Completions | Actions |
|---|---:|---:|---|
| Full model | 6 held-out + 3 dev | **0/9** | `read` ×7 |
| core_disabled control | 1 dev | 0/1 | `invoke_tool` proposals, no success |
| ff_core control | 1 dev | 0/1 | `invoke_tool` proposals, no success |
| extra_inference | 1 dev | 0/1 | `invoke_tool` proposals, no success |

No false success was accepted, no repair was selected and no base checkpoint was
saved. Training took **32 optimizer updates** across the two arms; peak sampled
driver memory **16.78 GiB**, profiled wall **470 s**, final extension
**37.6 MB**. This closes the larger-supervision hypothesis for this interface.
The consistent failure across 4 train cases, 12 train cases, guard/control and
constrained decoding points to action representation/policy, not prompt syntax,
data scale or loss optimization. A new action mechanism or full-backbone arm is
required.
Evidence: [report](../../reports/e3_set_large_attempt01/run_report.json).
