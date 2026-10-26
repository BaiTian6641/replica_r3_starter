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
These are software contracts, not native learning-quality evidence.

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

## Full-size native resume remains unqualified

First-process full checkpoints complete within limits. Fresh-process full
resume repeatedly fails the sustained 512 MiB swap-growth guard during large
optimizer-state restoration or the first resumed update. One earlier device
mismatch was fixed, but the resource failure persists. No further duplicate
full-resume attempts were made after isolating the failure. This does not
invalidate the successful fresh sustained run, and it does not qualify full
native interrupted resume.

Failed guards and child logs are preserved under
`reports/fused_adamw_20261025/` and `runs/fused_adamw_20261025/`.
