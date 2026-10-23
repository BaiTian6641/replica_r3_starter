# E3 action-space guard: no completion improvement

23 October 2026 UTC. Protocol `configs/e3_guard_v1.json` was committed before
execution. Fresh four-case `set_constant` family (three held-out, one dev), the
accepted E3 extension, no training, no optimizer update.

**Result: guard 0/4 and control 0/4 held-out/dev completions.** The model emitted
`read` repeatedly in both arms. With the guard, the first read completed and the
next six reads were denied (`action_guard: repeated_read_without_workspace_change`);
with the control, four reads completed and later reads were denied by the tool
budget. No episode selected `set_constant`, no episode passed the independent
evaluator, and no false success was accepted. The guard changed the action
outcome to denial but did **not** change model action selection.

| Arm | Cases | Completions | Model action sequence |
|---|---:|---:|---|
| Guard | 4 | **0** | read, then six denied reads |
| Control | 4 | **0** | four reads, then denied reads |

Guard and control consumed 7 decisions per episode. Guard used one tool call;
control used four. No tools modified the workspace under the guard. This is an
**action-selection failure**, not a parser, schema, evaluator or host-selection
defect. The host never supplied the expected repair or selected `set_constant`.

No training, checkpoint, base copy or download. Peak sampled driver memory for
the batch was recorded in the guard/profile files; total episode wall time was
about 187 s across eight episodes.
Evidence: [guard/control reports](../../reports/e3_guard_attempt01).
