# Bounded response-CE behavioral reference

**26 October 2026 · native response-CE transaction implemented and measured;
teaching completed, fit gate rejected; dependent evaluation remains closed.**
The response-CE learner primitive was introduced at `5da43a76`; this increment's
runner, protocol and device validation are committed at `57a4f69`. The protocol
file was present before native execution; the exact executed configuration is
pinned by SHA-256 below. This is a bounded diagnostic, not a learned-utility or
promotion result.

## Scope and lineage

The new `configs/nested_text_behavioral_reference_v1.json` admits one objective:
verified response cross-entropy through the frozen MiniCPM5 decoder. The
response callback creates a fresh differentiable plastic read, masks non-response
labels with `-100`, obtains gradients through the native action pass, and then
uses the existing atomic selected-expert transaction. Detached prompt features
and response targets remain observation evidence; they are not the training
graph. The MPS callback explicitly places all token and mask tensors on the
adapter device.

The arm uses one fixed-routed shared expert, width 256, hidden width 1024, one
update per eligible event, no replay, `NativeFusedAdamW`, learning rate `1e-4`,
and the declared primary/wrong-world parent checkpoints. The backbone, fixed
projection, read decoder, sensors and core readers remain frozen. The sealed
source is `checkpoints/minicpm5_source`; no model download or duplicate base
checkpoint was used.

| Binding | Value |
|---|---|
| Implementation/protocol commit | `57a4f69` |
| Configuration SHA-256 | `053ed39a08bfa55e09b36314cc1bb4ae9c55308d774581bd6d3886ab94e4c14e` |
| Primary parent | `runs/nested_text_nl1_v2_20261026/init_primary/full_checkpoint.pt` |
| Primary protocol SHA-256 | `2b83b3501fd4a60c3510583a1e8abbdc249afc58df9e67a530d8b1b1ce9b6b84` |
| Primary P-only state | `runs/nested_text_behavioral_reference_attempt03_20261026/teach_primary/p_only/state.pt` |
| P-only SHA-256 | `abf2bf5df2c4b5f4a6e2b9cd3e4031b3febafd1d78dbc5993d9954d14b739efb` |

The P-only manifest contains only the declared weights and binding keys. Its
`forbidden_absent=true` check covers optimizer, moments, clock, dedup, revision,
coordinator, observation and source state.

## Native execution

Every native invocation used one exclusive MPS worker through
`scripts/run_capacity_preflight.py`, CPU fallback disabled, the sealed local
source, and the sampled 50 GiB driver / 512 MiB swap-growth / 15% minimum-free
policy. The watchdog's default driver limit is 53,687,091,200 bytes; sampled
peaks are not a guarantee against transient OOM.

| Attempt | Scope | Result | Watchdog peak driver | Swap growth |
|---|---|---|---:|---:|
| `attempt01/teach_primary` | Direct teaching launch | Rejected before model load: missing capacity-preflight wrapper | Not applicable | Not applicable |
| `attempt02/teach_primary` | First watchdog teaching launch | Failed in the first response-CE callback: CPU/MPS input-device mismatch | 11,815,763,968 B | 0 B |
| `attempt03/teach_primary` | 28 primary eligible events, including four corrections | 28 committed updates and 28 response-CE backward calls; exports written | 11,815,763,968 B | 0 B |
| `attempt03/fit_primary` | First fit launch | Rejected by an overly broad CLI P-state guard; no scoring requests | 0 B | 0 B |
| `attempt04/fit_primary` | 28 teaching-fit generations from the attempt03 P-only state | 0/28 correct; `min_fit_accuracy` failed; dependent stages stopped | 10,765,516,800 B | 0 B |

The device fix is in `src/replica_lab/nested_text.py`; the fit CLI fix and
behavioral validation are in `scripts/run_nested_text.py`. Failed attempt
folders, child logs, host samples and failure metadata are retained. The
attempt03 teaching run has no failed transaction: all 28 transaction records
are `committed`, revisions run from 1 through 28, and the native parameter stamp
is unchanged. Its measured transaction losses are finite (first `12.992959`,
last `7.665615`; range `4.450089`–`12.992959`). These are optimization traces,
not evidence of correct generation.

### Teaching cost and artifact audit

The completed primary run measured 13.077574 s end-to-end, 8.919060 s in the
learning-update region, 2,670 processed positions, 122 supervised response
tokens, 140 attributed forward calls and 28 backward calls. It exported a
9,981,643-byte full checkpoint and a 3,148,851-byte P-only state. The audit
records exactly three trainable expert tensors (`expert_0.gate`, `expert_0.up`,
`expert_0.down`), zero trainable read-path parameters and 28 expert steps.
Watchdog host samples reached a minimum of 75% free memory with zero observed
swap growth.

### Fit result and raw outputs

The fit command loaded only the P-only weights and bindings, constructed no
controller or coordinator, performed no updates, and audited all 28 requests as
source-free. It generated 335 tokens in 29.049763 s. The measured work report
contains 28 prefill calls, 307 decode calls, 28 plastic-read calls, 28 session
sense calls and 391 total forward calls. Two attempts ended at the token budget;
20 ended with sentence completion and six with a stop token. Total-request
latency was p50 1,016.03 ms and p95 1,713.39 ms. The fit watchdog recorded zero
swap growth and a minimum of 76% free memory.

All 28 raw generations are preserved in
`runs/nested_text_behavioral_reference_attempt04_20261026/fit_primary/gates/fit.json`.
Every row has `correct=false`; there were no scoring/request failures. Outputs
include generic or refusal-like text and incomplete synthetic-name answers,
rather than the taught targets. The result is therefore a rejected behavioral
fit, not a decoding-success claim and not evidence of useful memory.

## Gate disposition

The declared fit gate requires accuracy at least `0.50`; measured accuracy is
`0.0` (`0/28`), so `run.json` records `status=stopped_fit_gate` and
`gate_failures=["min_fit_accuracy"]`. `heldout_opened=false`. Development,
wrong-world behavioral teaching/evaluation and sealed-final scoring were not
run after this failure. Earlier one-clock wrong-world and development artifacts
are separate experiments and are not reused as evidence for this arm.

No additional objective, optimizer, clock, backbone update or held-out retry is
authorized by this result. A future diagnostic must remain development-only,
predeclare its own bounds, and first explain the response readout/optimization
failure without tuning against sealed-final rows.

## Verification

The focused CPU regression suite passes **197 tests** across nested text,
nested learning, scoring, architecture adapter, memory session and matched
memory tests. Native teaching exercised actual response-CE gradients and
exports; native fit evaluation exercised P-only loading, generation, scoring,
source-free audits and the fit barrier. The direct-launch and first-callback
failures are retained as engineering evidence, not silently discarded.

The full repository run passed **2,362 tests plus 21 subtests**, with 8 skips
and 7 failures. The failures are Spark first-tranche/residency receipt checks
and vision caption/collapse/learnability implementation-pin checks; they are
outside the nested-text behavioral paths and are preserved rather than
rewritten. The focused 197-test result above is the passing verification scope
for this increment.

**Implemented:** response-CE transaction callback, MPS tensor placement,
parent-lineage checks, bounded teaching and fit CLI modes, strict P-only export
and workload/accounting reports. **Trained:** 28 primary selected-expert updates
only. **Measured:** optimization traces, fit generations, latency, memory,
artifact sizes and gate outcome above. **Rejected:** teaching-fit behavioral
utility at the declared gate. **Unrun:** dependent development, wrong-world
behavioral controls, held-out evaluation, multi-seed promotion and any broader
capability/calibration claim.
