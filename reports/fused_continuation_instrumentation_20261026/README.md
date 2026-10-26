# Fused continuation instrumentation, 26 October 2026

This directory adds instrumentation and a predeclared analysis policy around the
existing two-update fused-AdamW continuation comparison. It does not rewrite any
historical receipt, run a native job, export a model or claim learning quality.
The exact historical failures remain failures.

## What is new

- `src/replica_lab/training_state.py` now fingerprints a checkpoint boundary:
  parameter value bytes, dtype, device, shape, storage extent/offset/stride and
  contiguity; L1/L2/max-abs norms; optimizer moment tensors, counters and group
  hyperparameters; Python/CPU/MPS/CUDA RNG; schedule and batch cursor; train/eval
  mode per module; and every named buffer, including non-persistent rotary
  buffers. Comparisons localize the earliest differing section/name and report
  numeric deltas, missing/extra paths and truncated difference details.
- `capture_training_fingerprint`/`compare_training_fingerprints` are pure CPU
  comparisons over JSON-safe summaries. `summarize_training_norms` emits
  per-update parameter, pre-clipping gradient and optimizer-moment norms.
- `materialize_resume_model` optionally records model loading, whether a full CPU
  model copy existed, temporary checkpoint tensors, staged CPU optimizer bytes,
  per-tensor migration counts/bytes, cache releases, sampled driver memory and
  host swap growth.
- The runtime checkpoint metadata now also stores a deterministic RNG
  fingerprint, so resume metadata is inspectable without relying only on a hash
  of the whole file.
- `scripts/probe_fused_adamw_sustained.py` captures first-process and restored
  boundary fingerprints, compares them through the localization function, and
  records per-update norms. Reported training throughput excludes fingerprint
  instrumentation time.
- `scripts/compare_fused_continuation.py` now consumes repeated uninterrupted
  arms. It reports explicit A/A variation first, then A/B variation against the
  matched uninterrupted arm, the earliest A/A and A/B divergence, boundary
  fingerprint equality, resource accounting and all guard outcomes. Exact checks
  are unchanged and remain authoritative.

No tolerances are applied by default. The predeclared policy in
`tolerance_policy.json` is opt-in and cannot change `exact_status`.

## Historical A/A evidence

Three uninterrupted arms share the protocol and all non-resume implementation
paths (`comparison02`, `comparison05`, `comparison09`). Their recorded
`src/replica_lab/training_state.py` hashes differ because that file contains the
resume-materialization path; the continuous arms do not execute it. The new
comparison reports this as `implementation_identical: false` and
`execution_implementation_identical: true`.

| Metric | A/A pairwise max | A/B (comparison09 resume vs matched control) |
|---|---:|---:|
| Update 1 loss | 0 | 0 |
| Update 1 pre-clip gradient norm | 0 | 0 |
| Update 2 loss | `1.905485987663269e-06` | `4.842877388000488e-07` |
| Update 2 pre-clip gradient norm | `0.0011749267578125` | `0.0004425048828125` |
| Final fixed-output loss | `4.100799560546875e-05` | `1.1920928955078125e-05` |
| Final parameter hashes | 381/381 differ | 381/381 differ |
| Final logits SHA256 | all differ | differs |

Therefore the update-2 and output differences are not evidence of a restore
defect by themselves. Update-2 loss is recorded before that update's optimizer
step, and the resumed process already matched all first-process final parameter
hashes at start. Exact parameter-hash and logits-byte equality is not achievable
across the measured uninterrupted arms. The historical reports also never
fingerprinted optimizer moments; the new boundary records close that gap. This
conclusion does not retroactively pass comparison09: its exact checks remain
failed, and the receipt is preserved.

## Historical resource failures

The resource mechanism is separate from the numerical variation:

- comparison10 stopped in `native_resume_staged_optimizer_migration` with
  `2,823,815,168`-byte resume-guard swap growth, `31,147,294,720`-byte peak
  driver and 38% minimum free.
- comparison11, with deterministic algorithms and MPS fast math disabled,
  stopped in the same stage with `1,019,677,245`-byte swap growth,
  `25,124,945,920`-byte peak driver and 36% minimum free.
- comparison09 resume stayed inside the guard (`143,791,227`-byte swap growth)
  but exact comparison failed.

The checkpoint contains roughly 20.1 GB of FP32 state while the model is already
device resident; the current path stages optimizer state on CPU and migrates
tensor by tensor. The new accounting records this directly in future runs. The
historical reports predate that accounting, so their guard receipts are
preserved unchanged.

`historical_failures.json` indexes every comparison receipt and failure stage
with its content hash. Further identical retries are stopped as uninformative.

## Bounded next step

`hypothesis.json` declares one resource hypothesis: direct-to-device optimizer
state restoration that releases each CPU reference as its tensor is adopted,
while leaving model residency and all numerical policy fixed. It requires one
uninterrupted control and the existing two-process interrupted pair, with one
stop on resource or boundary failure. `tolerance_policy.json` separately
predeclares the A/A-envelope interpretation. Neither the hypothesis nor the
policy authorizes a native run in this ticket.

## Evidence files

- `comparison_aa_ab.json` — new A/A-then-A/B receipt over historical reports.
  It is descriptive and leaves `policy.applied` false: its A/A envelope is
  in-sample, so it cannot itself qualify the historical comparison.
- `historical_failures.json` — exact preserved failure index.
- `tolerance_policy.json` — predeclared, opt-in `aa-envelope-v1` for a future
  experiment, not a retroactive pass for these reports.
- `hypothesis.json` — single bounded replacement hypothesis.

Integration verification passed 33 tests in `tests/test_training_state.py
tests/test_fused_adamw_sustained.py` and 25 in `tests/test_optimizers.py`.
These are software contracts; no native exact-equivalence or quality pass is
claimed.
