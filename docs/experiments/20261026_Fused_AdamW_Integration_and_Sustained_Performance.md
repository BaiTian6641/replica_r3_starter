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

## Full-size native resume: bounded repair and qualification

The earlier loader repeatedly failed the sustained 512 MiB swap-growth guard
during optimizer-state restoration or the first resumed update. Those failed
guards and child logs remain preserved under `reports/fused_adamw_20261025/`
and `runs/fused_adamw_20261025/`; they are not rewritten as passes.

The repaired path constructs the architecture and optimizer on the meta device,
restores the validated optimizer/RNG/scheduler/cursor state with explicit target
device ownership, then loads the real checkpoint model and attaches the restored
state by the validated parameter-group order. It does not migrate a second full
CPU model copy. Declared base learning rates are captured before optimizer-state
restoration. The same-device state handoff retains one moment allocation; no
backend fallback is permitted.

The predeclared two-update smoke passed in two fresh native processes on 26
October 2026, using the sealed MiniCPM5 source and the unchanged protocol. The
first process completed update 1 and retained one **30,211,387,008-byte**
checkpoint. The second process started at step 1, matched all first-process
final parameter hashes, completed update 2, and changed all **381 tensors /
2,516,756,480 parameters**. Native optimizer state step values were `[2.0]`;
checkpoint reload validation and resume-start hash validation passed. No final
model was exported or promoted.

| Native process | Guard wall time | Peak sampled driver | Swap growth | Minimum host free |
|---|---:|---:|---:|---:|
| First (`smoke_first03`) | 71.51 s | 40,886,763,520 B | 0 B | 57% |
| Fresh resume (`smoke_resume03`) | 70.53 s | 43,243,962,368 B | 0 B | 50% |

Evidence: `runs/fused_adamw_20261026_resume01/smoke_first03/run_report.json`,
`runs/fused_adamw_20261026_resume01/smoke_resume03/run_report.json`, and the
corresponding guard JSON/child/host logs under
`reports/fused_adamw_20261026_resume01/`. Two earlier attempts in this new
directory are retained as setup/code failures (`HF_DATASETS_OFFLINE` missing and
a probe `base_path` regression); neither performed a model update.

This qualifies only the bounded full-scope two-update interrupted smoke. It does
not establish longer interrupted schedules, other model/optimizer scopes,
convergence, retention, or learning-quality equivalence.
