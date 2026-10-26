# MiniCPM5 and SparkX2.6: BF16, Muon and parallel GPU budget review

26 October 2026, America/Chicago. Follow-up to the
[initial Spark estimate](20261026_SparkX_Formal_Full_Finetuning_Compute_Estimate.md).
This review changes the hardware/optimizer assumptions, preserves the full
architecture, and includes the user's 100M, 1B and 5B processed-position scenarios.

**The 3090 Ti unlock is real and was missing from the previous estimate.** With
a qualified BF16/Muon implementation, the revised **1B-position** planning range
is **353–1,325 GPU-hours for the current MiniCPM backbone** and **627–2,349
GPU-hours for the full Spark design** on two unlocked 3090 Ti cards. Those are
**177–663 and 313–1,174 wall hours**, respectively. A complete multimodal MiniCPM
projection is a separate, larger workload: **424–1,590 GPU-hours** on that pair.

These are conditional engineering ranges, not measured CUDA performance. The
logs directly establish only MiniCPM/M2 FP32 performance. The largest remaining
uncertainties are useful training volume, CUDA kernel efficiency, live media
cost, whole-matrix distributed Muon, memory peaks and learning-quality parity.
No optimizer implementation, training default, hardware unlock or resource guard
is changed by this review.

## Corrections to the hardware comparison

The [unlock project's measured table](https://github.com/Zagitta/gpu-unlock)
reports the following **one-card dense 8192³ GEMM** results at 450 W with **FP32
accumulation**. These are neither sparse-operation figures nor FP16 accumulation:

| RTX 3090 Ti input | Before | After | Measured GEMM ratio |
|---|---:|---:|---:|
| BF16 | 87.50 TFLOP/s | 157.41 TFLOP/s | 1.80× |
| FP16 | 87.46 TFLOP/s | 118.39 TFLOP/s | 1.35× |
| TF32 | 42.58 TFLOP/s | 82.61 TFLOP/s | 1.94× |

The [original hardware report](https://github.com/Zagitta/gpu-unlock/issues/1)
and subsequent independent confirmation support considering an unlocked 3090 Ti
scenario. They do not benchmark this model, two-card training or sustained
workstation cooling. The forecast applies the 1.80× ratio only to eligible
compute. If 60–85% of a phase benefits, its speedup is approximately
`1 / (1-f + f/1.80) = 1.36–1.61×`. Other work retains its cost. The calculated
whole-update gain here is about **1.36–1.58×**, not 2×.

**TITAN RTX is different:** NVIDIA already specifies **130.5 TFLOP/s per card
for FP16 with FP32 accumulation**, or 261 TFLOP/s for the pair before software
inefficiency. Its BF16 entry is unavailable. Therefore this review grants
**no additional 2× unlock** to TITAN RTX and uses FP16/loss scaling there.
[NVIDIA Ampere whitepaper, Table 9, printed p.44](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf).
That corrects any implication that TITAN should be treated like a half-rate
GeForce Turing card.

Kernel compatibility still matters. MiniCPM has head dimension 128, causal
GQA and no dropout; the [Turing attention implementation](https://github.com/ssiu/flash-attention-turing)
supports those features. Spark uses head dimension 256 and local masks, outside
that implementation's documented coverage. The TITAN Spark estimate therefore
charges dense attention arithmetic **while retaining the original local masks**
and allows lower delivered performance. Replacing local attention with unrestricted
attention would change the model. A qualified exact implementation is required.
Ampere/Hopper attention paths also need their own
[backend qualification](https://github.com/Dao-AILab/flash-attention).

## Local evidence and what it can calibrate

All 22 input files are pinned by SHA256 in the
[evidence manifest](../../reports/mixed_precision_budget_20261026/input_manifest.json).
The architecture, state contracts, dataset plans and sustained-run inputs still
match the earlier frozen review. Current source snapshots preserve the actual
trainer and optimizer implementations used in this assessment.

| Measured M2 Ultra workload | Positions/update | Seconds/update | Updates/s | Processed positions/s |
|---|---:|---:|---:|---:|
| Cached full MiniCPM, FP32 Muon, 256/example | 2,048 | 22.494 | 0.0445 | 91.0 |
| Cached full MiniCPM, FP32 Muon, 512/example | 4,096 | 25.472 | 0.0393 | 160.8 |
| Cached augmented MiniCPM, FP32 Muon, 512/example | 4,096 | 26.164 | 0.0382 | 156.6 |
| Newer full MiniCPM, fused AdamW, 55/example, steady | 440 | 8.422 | 0.1187 | 52.25 |

The [12-update fused run](20261026_Fused_AdamW_Integration_and_Sustained_Performance.md)
changed all **381 original tensors / 2,516,756,480 parameters**. Its ten steady
updates averaged **2.084 s forward/backward, 5.425 s global clipping and 0.912 s
optimizer work**. The sampled driver peak was 40.13 GiB with zero swap growth.
It did not validate checkpoint reload. Later continuation work does not establish
a robust long-running schedule merely by completing this short probe.

The current cached augmented arm adds only **895,490 parameters / 48 tensors**.
It is not evidence that the complete vision/audio/memory/core architecture has
already run. Its roughly 7% forward/backward increase at 512 positions despite
that small addition also cautions against parameter-only throughput scaling.

Using the explicit checkpointed `8P` work convention **plus attention arithmetic**,
the measured 55/256/512-position forward/backward phases imply approximately
**4.05–11.11 TFLOP/s of equivalent model work** on this Mac. The reference Muon
optimizer implies **7.36–8.14 TFLOP/s of equivalent optimizer work** from its
mean and last-update times. These are workload-accounting rates, not silicon
benchmarks or transferable CUDA utilization measurements.

The M2 has no measured BF16 training result here. The current trainer rejects
MPS BF16, uses FP32 resident parameters and eager attention. CUDA BF16 autocast
is available, but it leaves FP32 resident weights/state intact. The estimates
for the NVIDIA devices require explicit, unmeasured delivery assumptions; local
logs supply the work, scope and phase costs rather than a fabricated Mac-to-CUDA
speed conversion.

## What optimized Muon changes

The current `orthogonalize_fp32` explicitly disables autocast. `muon8bit`
compresses stored momentum, while all five Newton–Schulz rounds remain FP32.
The [upstream Muon implementation](https://raw.githubusercontent.com/KellerJordan/Muon/master/muon.py)
uses BF16 for this matrix computation; its persistent momentum follows parameter
storage, so that cast alone does not halve optimizer residency. Its update rule
also differs from this repo's no-Nesterov quantized-momentum variant.

For this forecast, route backbone hidden matrices to five-round low-precision
Muon and retain fused AdamW for embeddings, the output head, norms, gates and
**every additional module**, including sensory encoders and bounded plastic
inner updates. This keeps all parameters trained and avoids silently extending
Muon to unqualified new branches. It follows the upstream
[hidden-matrix/auxiliary-AdamW distinction](https://github.com/KellerJordan/Muon).
The existing architecture contract still selects `NativeFusedAdamW`; the Muon
route is an explicit candidate, not a changed default or quality-equivalence claim.

| Exact backbone inventory | MiniCPM5 | SparkX2.6 |
|---|---:|---:|
| Unique original parameters | 2,516,756,480 | 4,112,079,360 |
| Hidden matrices routed to Muon | 294 | 180 |
| Hidden parameters | 1,981,808,640 | 3,774,873,600 |
| Original auxiliary parameters | 534,947,840 | 337,205,760 |
| Five-round Muon GEMM work/update | 97.65 TFLOP | 223.47 TFLOP |
| Muon work/position at 32,768 positions/update | 2.98 GFLOP | 6.82 GFLOP |

Spark's fused QKV is treated as its complete stored matrix; its attention gates
use AdamW. Splitting Q/K/V or orthogonalizing flattened FSDP shards independently
would change the update. Sharding requires a whole-matrix-aware implementation,
bounded gathers and correct ownership, not just adding a distributed launcher.

Muon saves persistent moments compared with AdamW, but adds matrix work. On an
H200, where full optimizer residency is easier, this model predicts the following
**1B-position** sensitivity using the same forward/backward and other-phase bands:

| Model | BF16 + fused AdamW | BF16 + Muon/auxiliary AdamW |
|---|---:|---:|
| Current MiniCPM backbone | 68–246 GPU-h | 76–277 GPU-h |
| MiniCPM full-module projection | 82–299 GPU-h | 91–330 GPU-h |
| Full Spark | 118–429 GPU-h | 138–501 GPU-h |

The AdamW sensitivity removes Muon GEMMs; its actual optimizer overhead and
residency need measurement. No discount for faster convergence is assumed for
either optimizer. At 8,192 positions/update Spark's Muon work becomes 27.28
GFLOP/position; at 131,072 it is 1.70. A larger effective batch amortizes updates
but changes optimization dynamics and must be qualified, not presented as a free
quality-preserving acceleration.

TITAN's Muon estimate assumes FP16 GEMMs with FP32 normalization, suitable
momentum and loss scaling. Upstream BF16 code is not a direct native-BF16 path
there. If its orthogonalization must stay FP32, the modeled 1B totals become
**707–2,360 GPU-h for current MiniCPM** and **2,107–7,610 GPU-h for Spark**.

## Full-architecture memory account

The Spark planning model retains the earlier **4.4B** envelope: the complete
4.112B backbone, 186.17M vision/audio encoder parameters, bridges, nine-reader
memory candidate, F/A/P machinery, sixteen stored plastic experts with two active,
core/appraisal, vocabulary and bounded unified interfaces. The partial reconciled
inventory is 4.363B; 4.4B remains a placeholder, not a cap or complete final tensor
inventory. Four-observation recurrence and teaching/replay passes remain charged.

For comparable full-module MiniCPM planning, use **2.8B**, comprising the measured
2.5168B backbone plus about 283M of proposed additions. This is a declared envelope,
not a newly audited implementation. Head widths, reader placement and actual media
rates still need a MiniCPM-specific inventory. The current 2.5168B backbone and
that future complete model remain separate rows throughout.

All values below are **GiB of persistent state before activations and scratch**.
Muon hidden groups have one momentum; every auxiliary/extension group retains
two FP32 AdamW moments. Tied Spark embedding/head storage is counted once.

| State layout | Current MiniCPM | MiniCPM full projection | Full Spark |
|---|---:|---:|---:|
| Current all-FP32 AdamW: weight + gradient + two moments | 37.50 | 41.72 | 65.57 |
| BF16 weight/gradient + FP32 master + two FP32 AdamW moments | 37.50 | 41.72 | 65.57 |
| BF16 weight/gradient + FP32 master + FP32 Muon momentum | 30.12 | 34.34 | 51.50 |
| Same, with BF16 hidden Muon momentum | 26.43 | 30.65 | 44.47 |
| BF16 weight/gradient + FP32 Muon momentum, no master | 20.74 | 23.91 | 35.11 |
| FP32 weight/gradient + current INT8 hidden momentum hypothesis | 24.59 | 28.81 | 40.96 |

The last two rows are sensitivities, not promoted numerical recipes. Persistent
FP32 gradients add **2 bytes/parameter** to the BF16-gradient rows: another
**4.69 GiB for current MiniCPM and 8.20 GiB for Spark**. Autocast does not create
a new residency policy. BF16 Muon compute, BF16 momentum and BF16 model ownership
are three different implementation choices.

For full Spark, the 44.47-GiB candidate is **22.24 GiB per GPU under ideal two-way
sharding**, leaving little of a nominal 24 GiB for activations, gathered matrices,
logits and optimizer scratch. FP32 momentum raises it to 25.75 GiB/card before
those costs. Selective offload or different placement remains necessary unless
measured peaks demonstrate fit. DDP replication alone does not provide this
sharding. Neither an unlock nor NVLink adds VRAM.

On this 64-GiB M2, the **current Spark FP32 AdamW state cannot fit** under the
50-GiB driver guard. BF16 momentum with masters could reduce persistent state to
44.47 GiB and leave only 5.53 GiB below that guard. That is a candidate to test,
not a pass. The no-master row gives more headroom but needs independent numerical
qualification. CPU offload on unified memory creates no additional physical RAM.
The unchanged swap/free-memory limits also apply during load and resume.

## Forecast assumptions and GPU-hours

The reference workload uses **2,048-position sequences and 32,768 processed
positions per optimizer update**. Preserve all parameters, causal masks, actual
media inputs, checkpointed backward paths and repeated learning passes. Count
every text/media/latent/padding position actually computed; count repeated epochs,
replay and inner/outer executions again. One billion is the total per schedule,
not one billion unique positions plus free repeats. Frozen-backbone inner passes
are conservatively charged as full training work here.

Model-phase work is approximately **22.95 GFLOP/position for current MiniCPM**,
**27.95 for its full-module projection**, and **40.01 for full Spark** with an
efficient local-attention path. Spark dense attention costs **42.73**. The 5-GFLOP
extension reserve includes live encoders and native branches; it is unprofiled.
Video frame count, audio duration, inner loops and observation histories can
exceed it. Compressed media positions must preserve the task and pass quality
checks; they are not a license to discard inputs to meet a budget.

| Entire job | Assumed model-phase TFLOP/s | Assumed Muon TFLOP/s | Other seconds/update at 4.4B |
|---|---:|---:|---:|
| H200 | 40–120 | 40–120 | 0.2–1 |
| H100 | 32–100 | 32–100 | 0.2–1 |
| L40S, selective offload | 12–32 | 16–48 | 1–5 |
| Two stock 3090 Ti | 12–32 combined | 20–60 combined | 2–6 |
| Two unlocked 3090 Ti | Same base, partial-phase unlock applied | Same base, eligible GEMMs /1.80 time | 2–6 |
| Two TITAN RTX, MiniCPM | 12–32 combined | 16–48 combined | 3–10 |
| Two TITAN RTX, Spark | 6–18 combined, dense attention | 16–48 combined | 3–10 |

These are explicit delivery assumptions. Model-phase rates allow kernel,
launch and parameter-sharding inefficiency; the other term allows clipping,
auxiliary AdamW, remaining optimizer work, gradient synchronization and selective
offload. It scales by stored parameters. They are not obtained by multiplying
the M2's measured speed by a vendor peak ratio. The 3090 base band is a new
optimization target for the mixed-state path; it is a separate assumption from
the independently reported unlock gain.

**If the old conservative 6–18 TFLOP/s model-phase delivery persists**, the
unlocked pair instead needs **595–2,495 GPU-h for current MiniCPM** and
**1,047–4,387 GPU-h for full Spark** at 1B positions. Keep this fallback until a
matched CUDA pilot replaces the assumed phase rates. Host RAM, OS, PCIe lanes,
bridges and simultaneous-card power limits have not been inventoried; the
targets assume a suitable Linux/CUDA host and adequate host-memory headroom.

```text
step seconds = model work / achieved model rate
             + Muon matrix work / achieved Muon rate + other update seconds
wall hours = processed positions / positions per update × step seconds / 3600
             × 1.25–1.50 elapsed allowance
GPU-hours = wall hours × occupied GPUs
development campaign = 3 schedule equivalents + fixed qualification allowance
```

The 25–50% elapsed allowance covers warmup, evaluation/retention, checkpoint IO,
data stalls and bounded recovery. It does not repeat the optimizer/offload term
or the separately charged initial qualification. Ranges below include this
allowance and exclude initial qualification. They are not confidence intervals.

**Current MiniCPM5, 2.5168B original parameters, Muon candidate**

| Hardware | 100M positions | 1B positions | 5B positions |
|---|---:|---:|---:|
| H200 | 8–28 GPU-h | 76–277 GPU-h | 381–1,387 GPU-h |
| H100 | 9–34 | 91–345 | 456–1,725 |
| L40S | 28–91 | 277–911 | 1,383–4,555 |
| Two stock 3090 Ti | 56–181 | 557–1,805 | 2,784–9,027 |
| Two unlocked 3090 Ti | **35–133** | **353–1,325** | **1,767–6,627** |
| Two TITAN RTX, FP16 | 58–189 | 578–1,895 | 2,888–9,473 |

**Full SparkX2.6 architecture, 4.4B planning parameters, Muon candidate**

| Hardware | 100M positions | 1B positions | 5B positions |
|---|---:|---:|---:|
| H200 | **14–50 GPU-h** | **138–501 GPU-h** | **688–2,503 GPU-h** |
| H100 | 16–62 | 165–622 | 824–3,112 |
| L40S | 49–163 | 494–1,630 | 2,470–8,152 |
| Two stock 3090 Ti | 99–322 | 990–3,215 | 4,948–16,076 |
| Two unlocked 3090 Ti | **63–235** | **627–2,349** | **3,134–11,743** |
| Two TITAN RTX, FP16 | 181–654 | 1,811–6,544 | 9,054–32,720 |

**MiniCPM with the complete proposed modules, 2.8B envelope:** at 1B positions,
H200 **91–330 GPU-h**, H100 **109–411**, L40S **332–1,089**, stock 3090 Ti pair
**668–2,162**, unlocked pair **424–1,590**, TITAN pair **690–2,258**. Multiply
these schedule totals by 0.1 or 5 for the other volumes; fixed qualification
does not scale. Full three-volume numbers are in the calculator output.

At the declared 32,768 positions/update, the unlocked pair's modeled rates are
**629–1,965 positions/s, or 0.0192–0.0600 updates/s**, for current MiniCPM;
**355–1,108 positions/s, or 0.0108–0.0338 updates/s**, for full Spark. Its smaller
updates/s than the M2 short probe reflects a roughly 74× larger position count
per update. Compare processed work, not the bare `it/s` field.

## M2 Ultra: conditional BF16 projection

Retain measured clipping, use the measured FP32 model/optimizer equivalent-work
rates, and apply a **1.0–1.5× GEMM sensitivity** for a future BF16 path. The upper
factor is an explicit what-if, not evidence that M2 BF16 is 50% faster. Dense
attention is charged. The state plan, implementation, peak memory and numerical
behavior are all unqualified; the machine-readable **executable budget is null**.

| Conditional M2 BF16/Muon model | 100M | 1B | 5B |
|---|---:|---:|---:|
| Current MiniCPM | 62–260 device-h | 620–2,601 device-h | 3,102–13,006 device-h |
| MiniCPM full-module projection | 73–312 | 731–3,124 | 3,655–15,620 |
| Full Spark | 118–491 | 1,184–4,907 | 5,922–24,534 |

For Spark, that is about **49–204 continuous days per 1B schedule**, before
shared-machine availability. At 50% availability, double calendar time. It is
not a currently runnable Spark plan. By comparison, directly extrapolating the
measured 512-position FP32 MiniCPM/Muon rate alone gives **1,727 training hours
per 1B**, or **2,159–2,591 with the elapsed allowance**, with that exact short
workload. The 55-position fused run gives a much slower token-rate extrapolation;
its 0.1187 updates/s is not the right denominator for a long-context budget.

## Using both GPU pairs in parallel

There are two distinct useful schedules:

1. **Independent preparation:** the 3090 pair works on backbone/text retention
   preparation while TITAN works on modality bridge warmups, encoder experiments,
   data validation or evaluation. F1 uses the document's frozen parent where
   required. Their elapsed preparation time is approximately `max(Tbackbone,
   Tmodality)` rather than their sum. Changed backbone representations invalidate
   stale feature caches, and separately trained components still need joint
   integration. GPU-hours add even when calendar time overlaps.
2. **One synchronous joint model:** place live vision/audio encoders and bridges
   on the TITAN pair; place the backbone, intrinsic readers, F/P/core and vocabulary
   on the 3090 pair. Transfer features forward and their gradients backward, retain
   autograd, synchronize optimizer boundaries, and use consistent loss scaling
   across the FP16/BF16 boundary. Pipeline independent microbatches so encoder work
   can overlap decoder work. This preserves full joint scope but needs a new
   distributed implementation. Detached cached features do not train live encoders.

The second layout moves about **192.73M parameters** and roughly **2.87 GiB of
persistent auxiliary AdamW state** off the 3090 pair. Its Spark BF16-momentum
account falls from 44.47 to **41.60 GiB across the pair**, or **20.80 GiB/card**
before activations. Actual placement need not move all intrinsic branches;
keeping per-layer memory readers with the backbone avoids repeated host transfers.

For an explicit pipeline sensitivity, assign 1 GFLOP of the existing 5-GFLOP
extension reserve per processed position to TITAN encoders, retaining 4 GFLOP
with the backbone. Use 16–48 delivered TFLOP/s for the encoder pair and a
10–30% bubble/transfer penalty. These are placement assumptions; no media work
is removed or counted twice. The resulting **1B-position** totals are:

| Full joint model on all four GPUs | Wall hours | Aggregate GPU-hours |
|---|---:|---:|
| MiniCPM full-module projection | 222–972 | 890–3,889 |
| Full Spark | 332–1,444 | 1,328–5,774 |

This layout helps residency and division of work; it does **not** automatically
halve training time. The backbone remains the bottleneck in the declared model,
and communication can make it slower than two 3090s alone. If moving encoders
eliminates costly offload or enables a larger physical batch, a matched pilot
may show a gain not credited here. NVLink within a pair does not provide an
automatic four-card memory pool; cross-pair topology needs measurement.

For formal scope, use `Ttotal = max(Tbackbone_prep, Tmodality_prep) + Tjoint +
Tevaluation`. Independent training of two disconnected halves cannot replace
F2/F3 full joint training. The four-GPU table budgets the joint stage; the
25–50% allowance covers preparation/evaluation. Do not add the same preparation
work again without removing it from that allowance.

## Development reserve and cloud cost

Use **three complete schedule equivalents plus initial qualification** for an
initial development reserve. This includes retries/ablations only up to that
finite work budget; it is not a claim that three attempts converge. Qualification
is 8–16 GPU-h for H200, 24–56 for the unlocked pair, 16–40 for TITAN, and 40–96
for the four-GPU pipeline, per modeled campaign.

| 1B positions per schedule | Current MiniCPM campaign | Full-module MiniCPM campaign | Full Spark campaign |
|---|---:|---:|---:|
| H200, Muon candidate | 237–848 GPU-h | 281–1,007 GPU-h | **421–1,518 GPU-h** |
| Two unlocked 3090 Ti | 1,084–4,032 | 1,295–4,826 | **1,904–7,102** |
| Two TITAN RTX | 1,749–5,724 | 2,087–6,815 | 5,448–19,672 |
| Four-GPU joint pipeline | Not applicable to text-only scope | 2,709–11,762 | 4,025–17,419 |

[DigitalOcean's current on-demand prices](https://www.digitalocean.com/pricing/gpu-droplets)
are H200 **$4.47/GPU-h**, H100 **$4.41**, and L40S **$1.57**. The H200 Muon
campaign therefore costs about **$1,060–$3,790 for current MiniCPM**,
**$1,250–$4,500 for full-module MiniCPM**, or **$1,880–$6,780 for full Spark**.
These are GPU-compute costs; account quota, availability, storage, taxes and
idle time are not established here.

If both current MiniCPM and Spark receive a three-equivalent 1B campaign, reserve
approximately **658–2,366 H200 GPU-hours**, about **$2,940–$10,580**. This retains
both qualification allowances. Choosing a smaller MiniCPM mechanism-validation
volume rather than repeating the full Spark-scale curriculum can lower the
combined project budget, but is a separate experimental-plan decision.

For cloud budgeting, retain fused AdamW as a strong control because H200 capacity
reduces Muon's memory advantage. For owned hardware, prioritize the qualified
unlocked 3090 pair for backbone/joint work and TITAN for useful concurrent
multimodal preparation. Use the four-GPU pipeline when a measured memory or
throughput benefit justifies its added communication and implementation work.

Before replacing these assumptions, run one matched native qualification per
candidate: same examples, positions, labels, effective batch, all-parameter
ownership and five Muon rounds; measure forward/backward, clipping, optimizer,
communication/offload, peak memory and restart. Verify finite gradients, output
and gradient tolerances, task learning and retention. Sample several windows
after warmup. For the unlock, compare the same two-card host before/after with
unchanged power policy. Fewer trainable modules, detached encoders, altered
attention masks or dropped replay are not throughput improvements for this task.

## Reproducibility and limits

The [calculator and evidence package](../../reports/mixed_precision_budget_20261026/README.md)
contain the explicit assumptions, exact phase arithmetic, all three volumes,
memory profiles, long-context sensitivity and qualification/campaign accounting.
Nine offline checks passed, including 22 evidence hashes, independent Decimal
arithmetic, raw steady-run reconstruction, GPU-hour units, volume scaling,
partial-phase acceleration, four-GPU accounting, deterministic CLI replay and
corrupt-input/overwrite rejection. These are accounting checks, not CUDA or BF16
training tests.

Long-context arithmetic alone raises Spark's modeled 8K step cost about 8% with
window-aware attention and 31% on the dense TITAN path at fixed positions/update.
Memory pressure, actual media inputs and cross-observation graphs may add much
more. Unspecified large teachers, device-specific adapters, data acquisition and
unbounded recursive improvement remain separate budgets. None of the measured
short probes establishes final-model quality, dataset sufficiency or convergence.
