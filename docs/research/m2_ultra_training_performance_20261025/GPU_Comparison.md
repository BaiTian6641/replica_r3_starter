# M2 Ultra full-training speed and market GPU comparison

25 October 2026, America/Chicago. The local workload is full-parameter
MiniCPM5-2B, actually 2,516,756,480 parameters, FP32 weights/gradients,
effective batch 8, and a custom five-step FP32 Muon8bit optimizer. The
[performance report](../../experiments/20261025_M2_Ultra_Full_Training_Performance.md)
separates its model and optimizer costs.

**There is no defensible exact “this Mac equals GPU X” training result in the
available evidence.** No matched CUDA run exists for this model, optimizer,
precision, data, accumulation and validation policy. Calling 0.04 it/s an
RTX 3060/4060/4090 equivalent would combine software overhead with hardware
capability. Gaming FPS, inference tokens/s, LoRA benchmarks and advertised
sparse FP4 AI TOPS do not answer this full-training question.

## 1. Hardware facts and current configuration fit

| Device | Memory | Published FP32 shader peak | Fit of the current full FP32 Muon configuration |
|---|---:|---:|---|
| This M2 Ultra, 60 GPU cores | 64 GiB shared with the system | Not specified by the cited Apple product page | Measured short runs fit; sampled driver peaks around 29–30 GiB |
| GeForce RTX 4060 Ti | Up to 16 GB | 22 TFLOPS | Cannot hold the existing persistent state without a memory strategy change |
| RTX PRO 4000 Blackwell SFF Edition | 24 GB | 24 TFLOPS | Also short of the current persistent-state lower bound |
| GeForce RTX 4090 | 24 GB | 83 TFLOPS | Same capacity issue despite much higher arithmetic peak |
| GeForce RTX 5090 | 32 GB | Approximately 104.9 TFLOPS, calculated | Persistent state can fit in principle; activations/backend workspace still need qualification |
| RTX 6000 Ada Generation | 48 GB | 91.1 TFLOPS | More memory headroom; actual CUDA fit and throughput remain unmeasured |

Sources: [Apple Mac Studio specifications](https://support.apple.com/en-us/111835),
[NVIDIA RTX 4060 Ti](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-4060ti/),
[RTX PRO 4000 SFF](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-4000-sff/),
[RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/),
[RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/),
[RTX 6000 Ada](https://www.nvidia.com/en-us/products/workstations/rtx-6000/).
The 5090 calculation is explicitly `21,760 CUDA cores × 2.41 GHz × 2 operations`;
it is a theoretical arithmetic ceiling derived from rounded published values,
not an observed training rate. Other NVIDIA peaks above are published rounded
shader/single-precision figures. Memory product labels and actual allocator
capacity must be reconciled with the device inventory before running.

Apple specifies 800 GB/s memory bandwidth for M2 Ultra; the SFF Blackwell card
specifies 432 GB/s. This does not imply the Mac trains faster: different kernels,
compute pipelines, synchronization and precision can dominate. The SFF card's
770 AI TOPS is sparse FP4, not the FP32 throughput used in the local Muon path.
[Apple](https://support.apple.com/en-us/111835),
[NVIDIA SFF specifications and precision footnotes](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-4000-sff/).

## 2. Why 24 GB is insufficient for the unchanged configuration

The local inventory gives this persistent-state lower bound:

| Component | GiB |
|---|---:|
| All FP32 parameters | 9.376 |
| All FP32 gradients | 9.376 |
| Current Muon8bit plus auxiliary FP32 AdamW state | 5.835 |
| Total before activations, temporaries and allocator | **24.586** |
| All-FP32 AdamW parameters + gradients + two moments | **37.503** |

Thus a 24 GB RTX PRO 4000 SFF or 4090 cannot simply take the existing fully
resident FP32 trainer and run it unchanged. A 32 GB 5090 does not solve the
37.5 GiB persistent requirement of the all-FP32 AdamW alternative either.

Full scope can be preserved through a separately qualified lower-precision
residency contract, optimizer offload or sharding. Every original parameter
must still receive updates. AMP compute alone leaves this repository's
resident FP32 weights/gradients/state unchanged. Ordinary inference device maps
and LoRA/freezing are not substitutes for full-parameter training.

These capacity conclusions come from actual parameter/state counts, not a
generic “parameters times model size” guess. Raw bytes and formulas are in
[cached_analysis.json](../../../reports/m2_ultra_training_performance_20261025/cached_analysis.json).

## 3. A useful arithmetic comparison, with explicit limits

The current optimizer's Newton–Schulz products require approximately
97.649 TFLOP per update. Dividing only that arithmetic by the NVIDIA published
FP32 peak produces these **ideal lower bounds**, assuming 100% peak utilization:

| GPU | Ideal seconds for only those NS matrix products |
|---|---:|
| RTX 4060 Ti, 22 TFLOPS | At least 4.44 s |
| RTX PRO 4000 Blackwell SFF, 24 TFLOPS | At least 4.07 s |
| RTX 4090, 83 TFLOPS | At least 1.18 s |
| RTX 5090, calculated 104.9 TFLOPS | At least 0.93 s |
| RTX 6000 Ada, 91.1 TFLOPS | At least 1.07 s |

These are not GPU iteration-time predictions or a ranking of measured training
systems. They exclude forward/backward, validation, quantization, memory traffic,
launch overhead and any offload needed to fit. Real kernel utilization is below
the ideal and depends on shape, precision and implementation. TF32/BF16 tensor
math would require a different numerical comparison; sparse FP4 TOPS cannot be
inserted into this formula. No published peak is used to claim a measured
speedup over this Mac.

## 4. Decision for this project

Optimize and measure the Mac's optimizer path before buying hardware on the
basis of 0.04 it/s. The cached 55-token transformer forward/backward takes only
about 2.2 s/update; most time is in checks, clipping and a costly custom
optimizer. There is concrete software headroom while preserving all parameters.

The fresh matched probe demonstrates that headroom: native fused AdamW reaches
0.11175 updates/s versus 0.04344 for Muon8bit, a **2.57× mean-rate improvement**
over three updates. Both arms train and change all 381 original tensors.
Sampled driver peak rises from 30.08 to 40.03 GiB. The optimizer algorithm
changes; learning equivalence and sustained throughput remain unqualified.

If the later 24 GB SFF workstation is used, qualify full-scope memory residency
or offload first, then benchmark the same sealed parent and examples. A faster
CUDA peak does not overcome an unchanged configuration that cannot fit.
For an unchanged resident FP32 setup, larger-memory GPUs are the relevant
capacity comparison; actual sustained performance still needs measurement.

The apples-to-apples benchmark must hold the 381-tensor trainable inventory,
parameter/compute precision, optimizer math, effective batch, sequence lengths,
masking, checkpointing and audit policy fixed. Record synchronized warm/steady
timings, processed tokens/s, peak memory and nonzero/changed-tensor evidence.
Then compare the separately optimized full-scope implementations as a second
experiment, clearly identifying algorithm and precision differences.

The [source record](../../../reports/m2_ultra_training_performance_20261025/external_sources.json)
distinguishes official specifications, derived arithmetic, local measurements
and unmeasured CUDA results. No shopping prices or unrelated benchmark scores
were used.
