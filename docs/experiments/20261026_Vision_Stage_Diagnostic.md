# Stage-wise fused-projector diagnostic: engineering recovered, learning rejected

26 October 2026. The isolated projector experiment used native
`torch.optim.AdamW(fused=True)` through `NativeFusedAdamW`, with 16 caption
updates followed by 16 procedural updates. All six projector tensors changed;
base and encoder remained frozen. No larger connector, backbone unfreezing or
promotion is justified.

## Outcome

| Evidence | Correct | Paired wrong | Shuffled | No image | Caption token F1 | Abstention |
|---|---:|---:|---:|---:|---:|---:|
| Stage A development | 0/24 = 0% | 0.00 pp | 0.00 pp | −33.33 pp | 0.00 | 0/4 |
| Stage B development | 9/24 = 37.50% | 0.00 pp | +8.33 pp | +4.17 pp | 0.00 | 0/4 |
| Stage B audit | 6/24 = 25.00% | 0.00 pp | 0.00 pp | 0.00 pp | 0.00 | 0/4 |

Every declared learned-quality gate fails. The audit's correct-control answers
are all `yes` (24/24), demonstrating a constant-output prior rather than useful
image dependence. Caption alignment never became useful; later procedural gains
do not establish perception. Caption response loss worsened by 0.3797, while
procedural accuracy rose 37.5 points. No caption F1 decrease occurred because it
was already zero. Text retention stayed 22/32, but that baseline is itself
ineligible.

## Engineering, recovery and reload

The original native process completed all 32 updates and audit generation but
failed while assembling the final report because the custom forgetting keys
were absent. The completed traces, extensions and scores were preserved and the
report was recovered offline with **zero updates rerun**. The recovered report
records that optimizer-state audit was not recovered. The separate sustained
probe records actual fused optimizer state and should not be confused with this
recovered run.

Stage-B extension reload in a fresh process performs zero updates and reproduces
audit generation, caption generation and retention files byte-for-byte. See
`runs/vision_stage_diagnostic_20261026/native05/run_report.json` and
`reload06/reload_report.json`.

Training guard: 265.856 s, peak sampled driver 13,600,210,944 bytes, zero swap
growth and 71% minimum host free. Reload guard: 63.681 s, peak driver
10,834,657,280 bytes, zero swap growth and 76% minimum host free. The original
guard's child exit is 1 because of the finalization error; recovery is explicit,
not a claim that the original process returned success.

## Disposition and next boundary

**Engineering evidence recovered; learned quality rejected.** Do not enlarge
the projector, unfreeze the backbone or start a larger arm. Analyze constant
output priors and caption alignment on fresh sealed data. Native visual memory
and RSI remain gated. Full evidence:
`reports/vision_stage_diagnostic_20261026/final_disposition.json`.
