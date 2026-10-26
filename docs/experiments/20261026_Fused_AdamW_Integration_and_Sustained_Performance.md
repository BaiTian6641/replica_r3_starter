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
