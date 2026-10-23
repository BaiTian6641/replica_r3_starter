# E3 constrained action decoding: aborted on interface damage

23 October 2026 UTC. Protocol `configs/e3_constrained_action_v1.json` was
committed before execution. The arm added token-level masking of the redundant
action strings `read`/`run_test` to the runtime guard, on a fresh
`set_constant` family. No training, no optimizer update.

**Result: stopped at the smoke episode (0/1 completion).** After the first read,
banning the `read` token did not produce a different valid action. The model
emitted `{"kind":".read",...}` and then malformed/unauthorized JSON:

| Decision | Raw kind | Outcome |
|---:|---|---|
| 1 | `read` | completed |
| 2 | `.read` | parse denied: unknown_kind |
| 3 | `invoke_tool` without tool | parse denied: tool_not_authorized |
| 4 | `invoke_tool` without tool | parse denied: tool_not_authorized |
| 5 | `refactor` | parse denied: malformed_json |
| 6 | `run_test` with wrong args | parse denied: malformed_json |
| 7 | `run_test` with wrong args | parse denied: malformed_json |

The mask damages the model's JSON continuation rather than selecting
`set_constant`. This is an **interface/decoding defect**, not evidence that the
runtime guard or repair selection works. The batch was aborted after the
predeclared smoke; only one tool call occurred and no evaluator passed. No false
success was accepted, no workspace mutation occurred, and no base copy or
checkpoint was saved. Do not run the remaining cases against this mechanism.

Peak sampled driver memory was recorded in the guard/profile files; the episode
took about 36 s wall time. Evidence:
[smoke report](../../reports/e3_constrained_attempt01/smoke_session_report.json).
