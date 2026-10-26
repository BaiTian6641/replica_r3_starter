# M2 Ultra full-parameter training performance

25 October 2026, America/Chicago. Requested scope: explain the slow full
fine-tuning path, identify improvements that keep every original parameter
trainable, and compare the result responsibly with market GPUs.

The main bottleneck in the cached runs is the combined gradient-check,
clipping and optimizer phase. It consumes 69–89% of update time. The custom
Muon implementation alone schedules about 97.65 TFLOP of FP32 matrix
orthogonalization per update. Faster attention or larger physical batches
cannot remove that fixed optimizer cost.

In a fresh matched three-update probe, native fused AdamW reached **0.11175
updates/s versus 0.04344 updates/s for Muon8bit: 2.57× faster**. Every original
tensor remained trainable and changed in both arms. This establishes a
short-sequence throughput improvement, not equivalent convergence; fused AdamW
used more memory and both arms passed the existing resource guard.

This is a performance investigation. The existing MiniCPM5 source, selected
models, experiment outcomes and training defaults retain their status.

## 1. Measured cached phase costs

Hardware is the recorded M2 Ultra with 60 GPU cores and 64 GiB unified memory.
The runs use all 2,516,756,480 original parameters, FP32 resident weights,
FP32 compute, MPS, eager attention, activation checkpointing, physical batch
one and eight accumulated examples per optimizer update.

| Workload | Forward/backward per update | Checks + clipping + optimizer | Total | Updates/s | Post-backward share |
|---|---:|---:|---:|---:|---:|
| Full, 55 actual tokens | 2.198 s | 17.508 s | 19.705 s | 0.0507 | 88.8% |
| Full, 256 actual tokens | 4.706 s | 17.788 s | 22.494 s | 0.0445 | 79.1% |
| Full, 512 actual tokens | 7.681 s | 17.792 s | 25.472 s | 0.0393 | 69.8% |
| Full augmented, 256 tokens | 4.910 s | 17.893 s | 22.803 s | 0.0439 | 78.5% |
| Full augmented, 512 tokens | 8.215 s | 17.949 s | 26.164 s | 0.0382 | 68.6% |

The 55-token record contains eight updates; each capacity record contains two
updates across fresh processes. These are short diagnostics, not long-duration
benchmarks. Values are recomputed from synchronized phase timestamps, so minor
rounding/marker differences from the original update timer are expected.
The post-backward column is a combined phase: the original logs do not time
the optimizer's internal kernels independently.

The [machine-readable analysis](../../reports/m2_ultra_training_performance_20261025/cached_analysis.json)
records each contributing phase and source hash. Reproduce it offline with:

```bash
.venv/bin/python scripts/analyze_training_throughput.py \
  --output reports/local/full-training-analysis.json
```

Original evidence: [55-token run](../../reports/minicpm5_balanced_mask1_chunk01_report.json),
[256-token run](../../reports/minicpm5_capacity256_chunk02_report.json),
[512-token run](../../reports/minicpm5_capacity512_chunk02_report.json),
[augmented 256](../../reports/minicpm5_D2_capacity256_chunk02_report.json),
[augmented 512](../../reports/minicpm5_D2_capacity512_chunk02_report.json).

## 2. What the code spends time doing

### Five FP32 Newton–Schulz rounds over 294 matrices

`src/replica_lab/optimizers.py:orthogonalize_fp32` performs three matrix
multiplications per iteration, repeated five times for every eligible matrix.
It explicitly disables autocast. Muon8bit quantizes persistent momentum; it
does not make these matrix multiplications INT8 or BF16.

| Parameter shape | Count |
|---|---:|
| 2048 × 2048 | 84 |
| 256 × 2048 | 84 |
| 6144 × 2048 | 84 |
| 2048 × 6144 | 42 |

For a matrix oriented to `r=min(rows, columns)` and `c=max(rows, columns)`,
the matrix-multiplication estimate is `5 × (4r²c + 2r³)` FLOPs. Across all
294 matrices that is **4,410 GEMMs and 97.649 TFLOP per optimizer update**.
Elementwise operations, momentum conversion, validation, clipping and AdamW
are additional costs. This is an arithmetic count, not measured GPU utilization.

For context, the rough `6 × parameters × tokens` language-model estimate is
6.64 TFLOP for eight 55-token examples, before checkpoint recomputation and
detailed attention/output geometry. The optimizer can therefore require more
matrix arithmetic than the short-sequence model pass. A low it/s here is not
an isolated measurement of the M2 Ultra's transformer throughput.

### Eager per-tensor validation and temporary buffers

`ReplicaOptimizer.step` checks finiteness separately for 381 gradients.
`quantize_blockwise` separately checks 294 momentum tensors. These produce
at least 675 scalar GPU-to-host decisions per update in this full arm, plus
the trainer's loss/metric checks. Some waits also collect queued GPU work;
the cached phase markers cannot assign an independent time to each check.

`dequantize_blockwise` expands each scale with `repeat_interleave` across all
elements. Quantization allocates a padded FP32 copy and clones the INT8 result,
including the common case where matrix sizes are exact multiples of the block
size. These allocations and memory passes are candidates for an exact FP32
implementation improvement.

The auxiliary AdamW group includes the input embedding and output head:
534,947,840 parameters use FP32 moments. The momentum compression applies to
the other 1,981,808,640 parameters; it does not quantize the model itself.

### Secondary costs in the trainer

`scripts/train_text.py` loads FP32 and forces `attn_implementation="eager"`.
It rejects MPS BF16 compute and hardcodes a one-example forward for each
accumulated example. `RunProfiler.mark` synchronizes at each backward marker.
Loss/response diagnostics also read scalars to the host. Activation
checkpointing recomputes model work during backward.

`model.gradient_audit` scans every original gradient with `count_nonzero`.
The production trainer runs that audit on the first two updates of each
process. A longer run amortizes those initial audits; restarting in tiny
chunks repeatedly pays them again. Removing independent validation or all
nonfinite checks is not an accepted optimization.

The historical/current source hashes differ, but their differences were
inspected. The dense Muon arithmetic is unchanged; later changes add sparse
AdamW handling, generation/storage paths and a second early gradient audit.
See [source provenance](../../reports/m2_ultra_training_performance_20261025/historical_source_provenance.json).

## 3. Improvements that preserve full training scope

| Priority | Change | Scope preserved | Qualification needed |
|---|---|---|---|
| 1 | Native fused AdamW: measured 2.57× faster in the matched short probe | Every original weight, embedding and head remains trainable | Different optimizer; matched learning/retention and memory tests before changing defaults |
| 1 | Optimize Muon: shape-batched FP32 orthogonalization, fewer temporary copies, block broadcasting, aggregate finite validation | Same 381 trainable tensors and declared optimizer routing | Independent update/quantization parity, nonfinite rejection, continuation and full-size timing |
| 2 | Physical microbatch 2/4 with fewer accumulation passes, holding effective batch at 8 | Same examples, tokens, labels and trainable parameters | Token-weighted loss/gradient parity and measured memory |
| 2 | SDPA attention for compatible native text execution | Same parameters and attention semantics | Mask/logit/gradient parity; export/reload and backend support |
| 3 | Fewer host synchronizations; sampled detailed statistics; keep required safety checks | Same training objective and parameter ownership | Accurate timing boundaries, finite-loss/gradient rejection, guard telemetry retained |
| 3 | Compute output logits only at supervised prediction positions, or use a qualified chunked loss | Same causal prompt, labels, embeddings and fully trainable output matrix | Exact causal shift, EOS, loss and gradient tests; retain full-logit diagnostic mode |
| 4 | Qualify BF16 compute or BF16 Muon temporaries with FP32 master weights/state where required | Precision changes; parameter scope stays full | New versioned numerical contract, finite gradients, update/reload, task/retention and MPS support |
| 4 | Consider an MLX implementation only after the immediate optimizer work | Requires explicit ownership of embeddings/head/final norm and all layers | Native model/tokenizer/loss/gradient/checkpoint equivalence and exact scope inventory |

Fused AdamW is a concrete comparison because the official
[PyTorch AdamW documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)
describes fused/foreach implementations and prototype MPS FP32 support. It is
not a claim that AdamW converges identically to Muon. The reference defaults
remain unchanged by this investigation.

The measured fused route currently exists in the profiling entry point only.
The production trainer's `--optimizer adamw` still uses the custom eager
`ReplicaOptimizer`; it does not select this native fused implementation.
Adoption needs an explicit optimizer option, versioned checkpoint/state
compatibility and matched learning-quality qualification.

Do not raise it/s by reducing gradient accumulation, examples per update,
context length or the trainable layer set and then compare it with the old
workload. Use matched updates/s **and** processed tokens/s. AMP alone does not
reduce this trainer's FP32 parameter/gradient/optimizer residency.

Stock MLX-LM's `full` mode first freezes the model and then unfreezes selected
decoder layers in the [current official implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/lora.py).
Its mode name alone is therefore insufficient for this repository's stronger
full-scope requirement, which includes embeddings, output head and final norm.
No MLX port or benchmark was performed here.

## 4. What a speedup could mean

These are conditional calculations, not measured improvements. For the cached
55-token case, keep the measured 2.198 s of forward/backward and accelerate
only the measured 17.508 s post-backward phase:

| Post-backward acceleration | Calculated seconds/update | Calculated updates/s |
|---|---:|---:|
| 2× | 10.95 | 0.0913 |
| 4× | 6.57 | 0.1522 |
| 8× | 4.39 | 0.2280 |

Conversely, even infinitely fast forward/backward would improve this cached
short-sequence run by only 1.13× while its post-backward phase stayed fixed.
That is why attention/batching alone should not be presented as the main fix.
For 512-token examples, forward/backward matters more and its infinite-speed
ceiling rises to approximately 1.43×.

## 5. Fresh native comparison

The [precommitted protocol](../../configs/full_training_throughput_probe_v1.json)
and `scripts/profile_full_training.py` declare three updates per arm from the
same sealed parent and eight exposed 55-token examples. All arms retain FP32,
all 381 original tensors, effective batch 8, response-only labels, checkpointing,
the same LR/warmup/clip settings and exclusive MPS execution. Every arm verifies
nonzero gradients and before/after SHA256 changes for every original tensor.
Weights are discarded; no checkpoint is promoted or exported.

The initial launches refused the already-owned worker lock before model load.
Both fresh retry arms completed within the unchanged 50 GiB driver /512 MiB
swap-growth /15% minimum-free guards:

| Measured three-update arm | Mean seconds/update | Updates/s | Forward/backward | Checks + clipping | Optimizer | Sampled driver peak |
|---|---:|---:|---:|---:|---:|---:|
| Muon8bit reference | 23.021 | 0.04344 | 4.220 s | 5.527 s | 13.275 s | 30.08 GiB |
| Native fused AdamW | 8.949 | 0.11175 | 2.229 s | 5.518 s | 1.202 s | 40.03 GiB |

The measured mean-rate speedup is **2.57×**. Comparing only the third update
gives **2.35×** (19.614 versus 8.352 s), with optimizer time 11.992 versus
0.721 s. The reference's second forward/backward has an allocation/warmup
outlier; three updates cannot establish sustained performance.
One iteration here means one optimizer update after eight examples: 440
processed tokens, or 19.11 versus 49.17 processed tokens/s respectively.

Both arms have identical initial parameter hashes, tokenized data and first
loss. All **381 original tensors** have nonzero gradients on all three updates
and changed before/after hashes. Minimum sampled host free memory is 48% for
Muon and 25% for AdamW; swap growth is 4.38 MiB and zero respectively. No
checkpoint is promoted, and the short loss traces do not establish equivalent
learning or retention.

The roughly 5.5 s phase combines the exhaustive scope audit and gradient norm
clipping; their individual costs were not measured in these two runs. The
harness audits every update, while the production trainer audits only its first
two updates per process and clips every update. A timing-only refinement now
separates those costs, but it has not executed natively. The predeclared batch-4
and batch-4/SDPA arms were deferred because the shared worker remained occupied;
no batching or attention speedup is claimed.

The comparison outcome and per-arm metrics are recorded in the
[native summary](../../reports/m2_ultra_training_performance_20261025/native_summary.json).
Raw guard logs remain beside it; ignored run directories retain full parameter
hashes, phase traces and any failures. CUDA-equivalent throughput has not been
measured; see the [GPU comparison](../research/m2_ultra_training_performance_20261025/GPU_Comparison.md).

## 6. Validation

The [validation record](../../reports/m2_ultra_training_performance_20261025/validation.json)
recomputes cached timings and arithmetic, verifies raw report and historical
implementation hashes, checks both native guards and compares initial weights,
data, environment, first loss and all changed tensors. Local report links resolve.
The committed harness/optimizer/guard tests passed **40 tests in 2.76 s**;
the timing-only refinement's scope/collation checks passed **2 tests in 1.18 s**.
These existing checks are reused for unchanged contracts. No CUDA or additional
native benchmark was run during final report validation.
