# Native fused AdamW integration and sustained performance

26 October 2026. New training now selects `adamw_fused`, implemented by
`NativeFusedAdamW`, a strict wrapper around **`torch.optim.AdamW(fused=True)`**.
The older `adamw` option remains the custom eager `ReplicaOptimizer`; historical
Muon8bit experiments and reports are unchanged.

## Software integration

The shared factory and production `train_text.py` route every declared trainable
parameter exactly once. D0/D2 retains embeddings, output head and final norm;
D1 retains only its declared new modules. Checkpoints record optimizer class,
implementation, fused flag, groups, hyperparameters, state dtypes and PyTorch
version. Legacy custom/Muon and unversioned optimizer states are rejected.
Checkpoint serialization streams optimizer state instead of cloning it, and
resume migrates model/optimizer state with explicit device ownership.

Focused results: **52 optimizer/resume tests** in the final targeted set, plus
**74 earlier optimizer/probe tests**. Continuous versus interrupted CPU tiny
runs match weights, optimizer moments, scheduler/RNG, generation and reload for
full, modules-only and full-augmented scopes. Nonfinite loss/gradient, missing
and duplicate ownership, sparse gradients and incompatible states are rejected.
These are software contracts, not native learning-quality evidence. The final
resume-repair regression set passes **88 tests**, including exact tiny
interrupted-versus-uninterrupted weights, moments and schedule state.

## Sustained native probe

Protocol: `configs/fused_adamw_sustained_probe_v1.json`. Sealed MiniCPM5,
FP32 parameters/compute, eager attention, activation checkpointing, physical
microbatch 1, effective batch 8, 55 tokens/example, response-only labels, causal
shift and supervised-token weighting. Twelve updates used actual native fused
AdamW at `1e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, zero decay and two warmup
updates. All **381 tensors / 2,516,756,480 parameters** changed. No model was
exported or promoted.

| Measurement | Warmup (2) | Steady (10) | Total (12) |
|---|---:|---:|---:|
| Seconds | 17.6462 | 84.2163 | 101.8625 |
| Updates/s | 0.11334 | 0.11874 | 0.11781 |
| Processed tokens/s | 49.8690 | 52.2464 | 51.8346 |
| Supervised tokens/s | 1.81342 | 1.89987 | 1.88489 |
| Forward/backward s/update | 2.24578 | 2.08431 | 2.11122 |
| Gradient audit s/update | 0.10285 | ~0.00000 | 0.01714 |
| Clip s/update | 5.42053 | 5.42499 | 5.42425 |
| Optimizer s/update | 1.05396 | 0.91233 | 0.93593 |

The probe brackets forward/backward, gradient audit, clipping and optimizer phases
with explicit `torch.mps.synchronize()` calls. Clipping remains the global
`torch.nn.utils.clip_grad_norm_(..., 1.0, error_if_nonfinite=True)` operation;
there is no per-parameter norm substitute, nonfinite suppression or backend
fallback. The sustained native run measured **5.42425 s/update** in clipping and
**0.93593 s/update** in the optimizer over all 12 updates. These timings profile
synchronization-inclusive phases; they do not change clipping semantics or
establish a learning-quality result.

The steady period therefore has 8.4216 s/update. Gradient audit runs only in
the first two updates per process; clipping remains every update. Final native
state is **20,134,053,364 bytes (18.75 GiB), all FP32**. Sampled driver peak is
**43,090,886,656 bytes (40.13 GiB)** with **zero swap growth** and 29% minimum
host free. The 50 GiB guard passed. Evidence:
`runs/fused_adamw_20261025/sustained01/run_report.json` and
`reports/fused_adamw_20261025/sustained01.guard.json`.

The historical three-update fused/Muon comparison remains **2.57× faster**,
but this sustained run does not establish universal speedup, convergence
equivalence or retention. Its loss trace is not a quality evaluation.

## Full-size native resume: bounded smoke passes, matched equivalence open

The earlier loader repeatedly failed the sustained 512 MiB swap-growth guard
during optimizer-state restoration or the first resumed update. Those failed
guards and child logs remain preserved under `reports/fused_adamw_20261025/`
and `runs/fused_adamw_20261025/`; they are not rewritten as passes.

The repaired path validates the architecture and declared base rates on the meta
device, materializes eager checkpoint tensors without a second full CPU model,
then restores optimizer/RNG/scheduler/cursor state on CPU and migrates each
tensor to MPS with periodic cache release. Train mode is restored explicitly
before updates. A separate predeclared comparison uses deterministic algorithms
and disables MPS fast math. No backend fallback is permitted.

The original two-update smoke passed in two fresh native processes on 26 October
2026, using the sealed MiniCPM5 source and its protocol. The first process
completed update 1 and retained one **30,211,387,008-byte** checkpoint. The
second started at step 1, matched all first-process final parameter hashes,
completed update 2 and changed all **381 tensors / 2,516,756,480 parameters**.
Optimizer state steps were `[2.0]`; checkpoint and resume-start hash validation
passed. No final model was exported or promoted.

| Evidence | Guard wall time | Peak sampled driver | Swap growth | Result |
|---|---:|---:|---:|---|
| `resume01/smoke_first03` | 71.51 s | 40,886,763,520 B | 0 B | checkpoint passed |
| `resume01/smoke_resume03` | 70.53 s | 43,243,962,368 B | 0 B | bounded resume passed |
| `comparison09/interrupted_resume` | 206.07 s | 43,473,207,296 B | 143,791,227 B | resume passed; exact comparison failed |
| `comparison11/interrupted_resume` | 58.48 s | 25,124,945,920 B | 1,019,677,245 B | stopped by swap guard |

The independent `scripts/compare_fused_continuation.py` receipt is required in
addition to guard success. Comparison01 exposed a missing train-mode restore;
that defect was fixed. Comparison05 exposed the direct constructor's missing
eager-attention setting, also fixed. Comparison09 then passed resume resource
checks but still differed from the uninterrupted control in all 381 final
parameter hashes and the final logits: update-two loss differed by
`4.842877388e-07` and fixed-output loss by `-1.1920928955e-05`. Comparison11,
with deterministic algorithms and fast math disabled, stopped during optimizer
migration before a continuous control could run. No exact matched native
continuation receipt passes.

Evidence is retained under `runs/fused_adamw_20261026_resume01/`,
`runs/fused_adamw_20261026_comparison{01..11}/` and the corresponding tracked
`reports/` directories, including failed comparisons and resource stops. The
bounded checkpoint smoke is implemented evidence; **robust repeated residency
and exact uninterrupted-versus-resumed execution remain unqualified**. Longer
interrupted schedules, other scopes, convergence, retention and learning-quality
equivalence also remain open. Further fused retries were stopped rather than
turning the final comparison into a tuning loop.

## Interruption-boundary instrumentation and A/A evidence

26 October 2026 addendum. The exact failures above remain failures; no historical
receipt, run report, guard log or output fingerprint was rewritten. No further
native job was run for this instrumentation.

The checkpoint boundary is now fingerprinted in `src/replica_lab/training_state.py`
rather than represented only by final parameter hashes. A boundary record
contains, for every parameter and named buffer (including non-persistent rotary
buffers), the value SHA256, dtype, device, shape, storage bytes, storage offset,
stride and contiguity, plus L1/L2/max-abs norms. It also contains optimizer moment
tensors and counters, group hyperparameters, Python/CPU/MPS/CUDA RNG
fingerprints, scheduler state, batch cursor and per-module train/eval mode.
`compare_training_fingerprints` reports missing/extra tensors, metadata fields,
value-hash mismatches, norm deltas and the earliest differing section and name.
Each runtime checkpoint also stores a deterministic RNG fingerprint in metadata.

Per-update records now include parameter norms, direct pre-clipping gradient
norms and optimizer-moment norms. Reported throughput subtracts fingerprint
instrumentation time explicitly. Resume materialization can record model-copy,
temporary checkpoint-tensor, staged CPU optimizer, per-tensor migration, driver
memory and host swap accounting. The historical comparison reports predate these
fields, so their key evidence remains the parameter hashes, losses, logits and
guard receipts already cited above.

### Uninterrupted A/A variation versus uninterrupted-versus-resumed A/B

Four uninterrupted arms exist, but only `comparison02`, `comparison05` and
`comparison09` share the same protocol and all non-resume implementation paths.
Their `src/replica_lab/training_state.py` hashes differ because that file also
contains the resume-materialization path, which continuous arms do not execute;
the comparison receipt reports both `implementation_identical: false` and
`execution_implementation_identical: true`.

| Metric | A/A pairwise max | A/B, comparison09 resume vs matched control |
|---|---:|---:|
| Update 1 loss | 0 | 0 |
| Update 1 pre-clip gradient norm | 0 | 0 |
| Update 2 loss | `1.905485987663269e-06` | `4.842877388000488e-07` |
| Update 2 pre-clip gradient norm | `0.0011749267578125` | `0.0004425048828125` |
| Final fixed-output loss | `4.100799560546875e-05` | `1.1920928955078125e-05` |
| Final parameter hashes | 381/381 differ in every pair | 381/381 differ |
| Final logits SHA256 | all differ | differs |

The earliest A/A divergence is update 2 loss; update 1 is exactly identical
across all arms. Update-2 loss is recorded before that update's optimizer step,
and the resumed process starts from first-process final parameter hashes. The
resumed arm's update-2 and output differences are smaller than the measured
uninterrupted spread. Exact parameter-hash and logits-byte equality therefore
cannot be required as a restore criterion. Historical reports never
fingerprinted optimizer moments; the new boundary records close that gap. This is
a variation finding, not a pass: comparison09's exact receipt stays failed, and
comparison01/05 remain failed for the same exact fields.

The independent receipt under
`reports/fused_continuation_instrumentation_20261026/` records A/A first, then
A/B against the matched arm, the earliest A/A and A/B divergence, boundary
comparison status (unavailable for historical reports), guards and accounting.
Further identical retries are stopped as uninformative. Optimizer moments were
never fingerprinted in the historical reports; that gap is what the new boundary
records close.

### Preserved resource failure and one bounded hypothesis

The numerical-variation conclusion does not explain the swap stops. Comparison10
and comparison11 both stopped in `native_resume_staged_optimizer_migration`
before a continuous control could run: comparison10 at `2,823,815,168`-byte swap
growth with `31,147,294,720`-byte peak driver and 38% minimum free; comparison11
at `1,019,677,245`-byte swap growth with `25,124,945,920`-byte peak driver and
36% minimum free. The current path stages roughly 20.1 GB of FP32 optimizer state
on CPU while the model is already device resident, then migrates tensor by tensor.
Those failure reports and guard logs remain the resource evidence.

`reports/fused_continuation_instrumentation_20261026/hypothesis.json` declares
one bounded replacement hypothesis: direct-to-device optimizer-state restoration
that drops each CPU reference as its tensor is adopted, with model residency,
`NativeFusedAdamW`, clipping, schedule, RNG and the two-update plan unchanged. It
requires one uninterrupted control plus the same two-process interrupted pair,
one resource or boundary stop and no tuning loop. It is not scheduled by this
change.

A tolerance policy is separately predeclared in `tolerance_policy.json` as
`aa-envelope-v1`. It is opt-in, uses only the measured A/A pairwise maximum with
no extra margin, requires exact update-1 and boundary equality, and excludes
exact hashes/logits and all resource limits. The historical receipt leaves it
unapplied because its envelope is in-sample; the policy is for a future bounded
experiment. It cannot change `exact_status`; the permitted label for a future
qualifying run is `failed_exact_within_declared_variation`, never `passed`.

Deterministic CPU tests cover tensor metadata/norm fingerprints, exact error
norms, boundary round trips, mismatch localization for parameters, non-persistent
buffers, train mode, cursor, RNG and moments, per-update norm summaries, A/A then
A/B verdicts, explicit policy application, boundary mismatch and unqualified
single-arm comparison. Integration verification passed 33 tests in
`tests/test_training_state.py tests/test_fused_adamw_sustained.py` and 25 in
`tests/test_optimizers.py`. These are deterministic software contracts, not a
native exact-equivalence or learning-quality pass.
