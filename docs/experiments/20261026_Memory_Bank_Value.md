# Selected-record literal value and MiniCPM readout

**26 October 2026 · native03 selected · model development endpoint rejected.**
The predeclared protocol and implementation are at `25ce096`, `6dc5252` and
`dfcf309`. The machine-readable receipt is
[summary.json](../../reports/memory_bank_value_20261026/summary.json); raw
outputs and guard telemetry remain under `runs/memory_bank_value_20261026/`.

## Result

The exact bank supplies a selected runtime value without reading evaluation
labels. Literal development scoring passes. Sealed MiniCPM then receives only
the query, that authorized value and runtime status in a bounded greedy prompt.
Its development readout fails every declared semantic endpoint, so held-out and
train evidence were not produced.

| Development check | Result | Required | Disposition |
|---|---:|---:|---|
| Literal fact value exact | 124/128 (96.875%) | ≥85% | Pass |
| Unknown selections | 0/16 | 0 | Pass |
| Revoked selections | 0/8 | 0 | Pass |
| MiniCPM exact answers | 19/152 (12.5%) | ≥85% | **Fail** |
| Correction answers | 5/16 (31.25%) | ≥60% | **Fail** |
| Unaffected answers | 5/40 (12.5%) | ≥70% | **Fail** |
| Unknown answers | 0/16 | ≥90% | **Fail** |
| Revoked answers | 0/8 | ≥90% | **Fail** |

This separates exact host-side value recovery from model readout. It is not a
learned-selector, calibrated-confidence, persistent-learning or useful-memory
result. The separate compressed-vector receipt is referenced only and is not
combined with this arm.

## Execution and independent check

`native03` used one sealed MiniCPM load, MPS with CPU fallback disabled, offline
mode, zero gradients/optimizer updates and no checkpoint or extension writes.
It generated 1,204 tokens across 152 development rows in 93.93 s. The guard
completed in 96.69 s with 10,763,255,808-byte sampled driver peak, zero swap
growth and 74% minimum observed host free memory. Artifacts total 505,747
bytes. Sampled telemetry can miss transients; MPS and RSS overlap.

An independent fresh replay validated every generation identity, recomputed
the exact report metrics and gate, and verified both raw-output hashes.
`native02` and `native03` decisions and generations are byte-identical. No
held-out or train filenames exist in the selected output.

## Preserved failures and corrections

- `native01` stopped before generation because an empty enabled-branch tuple
  violated `DecisionFrame`; its failure receipt is preserved.
- The caller was changed to the established empty-memory frame convention:
  `enabled_ports=[]`, `enabled_branches=["memory"]`, empty ledger, no record
  memory and no workspace. A mock test that pinned the invalid call was
  removed rather than re-pinned.
- `native02` completed but its report still declared empty frame branches.
  The protocol was corrected to distinguish the required frame label from
  actual memory use. `native03` regenerated consistent evidence without
  changing thresholds, splits or generation budgets.
- The synthetic label fixture originally superseded the initial fact before
  its correction was causally live. It now follows the production cutoff rule.

The focused suite passes **14 tests**. The next gate is to inspect the preserved
development readout failures and predeclare one evidence-backed hypothesis;
held-out remains sealed and no threshold tuning is authorized.
