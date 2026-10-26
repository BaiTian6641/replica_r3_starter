# SparkX 2.6 4B: formal full fine-tuning compute estimate

26 October 2026, America/Chicago. Review of the current revision 4.5 design,
corrected residency ledger and recorded MiniCPM training runs. The user selected
100M, 1B and 5B total processed-position scenarios. This is a planning report,
not a training protocol, convergence prediction or permission to rent hardware.

For the **1B-position scenario**, plan approximately **116–417 H200 GPU-hours
for one integrated schedule**, or **355–1,266 H200 GPU-hours for a three-schedule
development campaign including initial qualification**. At the current
DigitalOcean rate, the latter is about **$1,590–$5,660 in GPU compute**. A rounded
procurement reserve of **400–1,400 H200 GPU-hours, about $1,800–$6,300**, is
reasonable until a matched Spark pilot replaces the assumed throughput.

The current Mac cannot execute the complete FP32 training state within its
unchanged guard. Each dual-24-GB pair needs a qualified sharding/offload path;
the two pairs are separate jobs, not one automatically pooled 96-GB device.
All Spark/CUDA hours below are conditional estimates. Only MiniCPM timing is
measured. The ranges are engineering scenarios, not statistical confidence bounds.

## 1. Scope reviewed and what remains unknown

The [current architecture](../architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md)
requires full offline joint training before audiovisual online learning. The
[state contract](../architecture/22_State_Mechanism_and_RSI_Contracts.md),
[intrinsic design](../research/sparkx_intrinsic_learning_spine_20261026/Architecture_and_Spine.md)
and [nested-learning contract](../research/sparkx_nested_learning_20261026/Nested_Learning_Contract.md)
define additional state, passes and learning clocks. This estimate retains:

| Component | Reviewed scope | Effect on the estimate |
|---|---|---|
| Spark backbone | 4,112,079,360 unique parameters, 36 layers, tied embedding/head counted once | Every original parameter remains trainable in the joint full-FT stages |
| Vision and audio | SigLIP2 92,884,224 parameters; SPEAR 93,289,476, or separately qualified audio control | Live encoder forward/backward and optimizer state; cached frozen features cannot replace full encoder training |
| Spatial/temporal bridges and heads | Native points/regions, tracks/events, acoustic and AV evidence | Additional parameters, encoder work and actual conditioning positions |
| Intrinsic memory | Nine-reader candidate; bounded records; shared `F`; optional `A` | Reader computation, one write per observation boundary, up to four-transition gradient history |
| Plastic knowledge | Sixteen full-rank experts, two active per site; `Pf/Ps` clocks | All experts remain stored; eligible inner passes and replay are charged; resident moments retained |
| Core/appraisal and vocabulary | Eight recurrent slots, initially two iterations; 256 active phrase rows | Native controller/lexical work and causal state; two workspace iterations are not automatically two complete decoder passes |
| Unified spine | Bounded intrinsic interface and declared readouts | Additional device/model-specific adapters and attached large models require their own inventory and budget |

The versioned residency inventory totals **4.3208B** fixed parameters, but is a
two-reader lower inventory. Replacing its reader matrices with the nine-reader
candidate and adding the separately listed plastic bank/projections yields a
**partial 4.3630B account**. Use **4.4B as a rounded planning placeholder**, with
about 37M parameters of reserve, rather than claiming a final tensor inventory.
Norms, gates, controller details and potential overlapping projection definitions
still need reconciliation. Added parameters have no fixed product ceiling.

The [dataset plan](../datasets/20261026_SparkX_Dataset_Plan.md) and
[MiniCPM curriculum](../datasets/20261026_MiniCPM_Dataset_Validation.md) provide
candidate pools and small pilots, not a final Spark dataset size or epoch count.
The 10k visual/15-hour audio pool and earlier 2M-token cap are pilot limits,
not an adequate basis for claiming final-product training cost. Real-media
admission and useful visual/memory quality remain open; no training is launched
by this review.

All reviewed inputs are preserved with SHA256 hashes in the
[input manifest](../../reports/sparkx_gpu_hour_estimate_20261026/input_manifest.json).
It explicitly includes concurrent working-tree design changes and the newer
ignored sustained run, without modifying those sources.

## 2. What one billion positions means

Count **every position actually processed by the backbone**: text, image/audio
conditioning, clock/event markers, latent reasoning positions, computed padding,
each epoch, replay and additional inner/outer or sensing/prefill executions.
Supervised answer tokens alone substantially undercount compute. A repeated
forward visits those positions again and must be charged again.

The forecast conservatively prices frozen-backbone inner executions like full
checkpointed training executions. This can overestimate first-order adaptation
work, but avoids pretending that a frozen downstream decoder is free. Do not
multiply by the same inner-step count a second time after counting those passes.

For example, 1B unique positions trained for three passes with another 20%
replay already gives **3.6B processed positions**, before any uncounted sensing
or teaching passes. Conversely, our 1B scenario is a **total across the declared
schedule**, not 1B for every stage and every epoch.

Use the architecture's provisional mixture by positions: **30% text retention,
20% image/region, 20% audio, 15% temporal/AV and 15% memory/learning episodes**.
An illustrative schedule allocates 60% to F2 integration, 25% to F3 grounded
reasoning, 10% to F4 learning behavior and 5% to F5 consolidation. These stage
shares are new planning assumptions; the documents do not approve this recipe.
F4's share must include its charged teaching/replay/future-query executions.

One schedule includes a **25–50% elapsed allowance** for F1 warmup, held-out
generation/interventions/retention, normal data stalls, checkpoint I/O and bounded
recovery. It does not cover unlimited failed experiments or unspecified teacher
generation. A development campaign is **three full-schedule equivalents** plus
one initial hardware-qualification allowance. This is a finite research reserve,
not a promise that three attempts produce a qualified product.
F0 hardware qualification is the separate fixed allowance, not charged again
inside the elapsed multiplier.

The eight-arm memory/vocabulary/plastic-learning factorial reuses a qualified
multimodal parent. It is not automatically eight fresh full-backbone training
runs. Count its real forward/backward/update work instead.

## 3. Calibration from measured MiniCPM logs

The historical three-update result was 0.04344 updates/s for Muon8bit and
0.11175 for fused AdamW at 55 positions/example. The newer
[production fused integration report](20261026_Fused_AdamW_Integration_and_Sustained_Performance.md)
supersedes the old profiling-only implementation status:

| Actual MiniCPM measurement | Result |
|---|---:|
| Full trainable scope | 2,516,756,480 parameters; all 381 original tensors changed |
| Physical/effective batch | 1 / 8 examples; 440 processed positions/update |
| Steady period | 10 updates after 2 warmup updates |
| Steady update time / rate | 8.4216 s / 0.11874 updates/s |
| Steady processed rate | 52.246 positions/s |
| Forward/backward | 2.0843 s/update |
| Global clipping | 5.4250 s/update |
| Native fused AdamW | 0.9123 s/update |
| Sampled peak driver | 40.13 GiB; guard passed |

Thus most of the remaining short-sequence delay is clipping, not the scope
audit. Audit is about 0.103 s on each of the first two updates and absent in the
steady period. The sustained run does not establish Spark fit, convergence or
exact native continuation; its checkpoint-reload flag is false. Separate
continuation experiments still have unresolved qualification requirements.

The older full MiniCPM forward/backward timings are 2.198 s at 55 positions,
4.706 s at 256 and 7.681 s at 512, with eight accumulated examples. Using the
explicit checkpointed-work convention `8 × parameters × positions`, these
correspond to **4.03–10.74 TFLOP/s of equivalent work**. This is not a silicon
utilization measurement. It supplies a broad Mac compute calibration only.

The **2.57× optimizer comparison is not multiplied into the Spark forecast**.
The current architecture calls for native fused AdamW; its full state residency
and the longer multimodal workload must be accounted for separately.
The cached 512-position full-augmented MiniCPM arm spends about 7% more time
in forward/backward than its plain full control despite a small parameter
addition. This reinforces the need to measure repeated branch calls and launch
overhead rather than scale throughput solely by stored parameter count.

## 4. Forecast method and assumed hardware delivery

At a 2048-position sequence, the reference arithmetic budget is approximately
**40.01 GFLOP per charged position**:

- `8 × 4.112B = 32.90 GFLOP` for checkpointed backbone training, a rough dense
  language-model convention including an extra forward recomputation.
- `16 × query_heads × head_dim × (9 × length + 27 × min(length,512))`, or
  2.114 GFLOP, for checkpointed full/sliding attention products.
- An assumed 5 GFLOP reserve for live sensory encoders, native branches,
  non-GEMM work and unimplemented details. This is a planning reserve, not
  a component profile; actual frame/sample rates and encoder activations matter.

The mixed-precision CUDA rows assume a qualified efficient local-attention path,
checkpointing, bounded media windows, bucketing/packing without cross-example
leakage, and accumulation sufficient to amortize optimizer/offload work. They
retain all original parameters and full joint-stage encoder scope. Numerical
precision changes are explicit, not implied by the full-scope label.

| Entire job | Precision/execution assumption | Assumed delivered equivalent work | Implied processed positions/s |
|---|---|---:|---:|
| 1 H200 | BF16 compute; FP32-owned optimizer state | 40–120 TFLOP/s | 1,000–3,000 |
| 1 H100 | Same, with complete residency qualified | 32–100 TFLOP/s | 800–2,500 |
| 1 L40S | BF16 plus qualified optimizer offload | 12–32 TFLOP/s | 300–800 |
| 2 RTX 3090 Ti | BF16; sharding and optimizer offload | 6–18 TFLOP/s combined | 150–450 |
| 2 TITAN RTX | FP16/loss scaling; sharding/offload; dense attention allowance | 3.2–9.6 TFLOP/s combined | 75–225 |
| 1 H200, strict FP32 sensitivity | TF32 off; dense attention allowance | 12–30 TFLOP/s | 281–702 |

**These delivery bands are assumptions, not CUDA benchmarks derived from the
Mac.** They represent low effective delivery for a small, custom multimodal
graph with routing and, where needed, communication/offload. They can be wrong
in either direction. H200's band is roughly 4–12% of its dense BF16 peak; the
3090 Ti pair's is roughly 4–11% of its applicable dense BF16/FP32-accumulation
peak. These percentages are plausibility checks, not predicted utilization.
[NVIDIA H200 datasheet](https://resources.nvidia.com/en-us-gpu-resources/hpc-datasheet-sc23),
[NVIDIA Ada whitepaper, Appendix A, pp.29–31](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf).

The TITAN row does not assume modern BF16/FlashAttention parity. The official
[FlashAttention implementation](https://github.com/Dao-AILab/flash-attention)
distinguishes Ampere/Ada/Hopper from Turing. Its linked
[Turing implementation](https://github.com/ssiu/flash-attention-turing) supports
head dimensions 64/96/128 and no local masks, while Spark requires head dimension
256 and local attention. An exact alternative must be qualified; this report
allows dense attention cost and does not change Spark's attention geometry.

For each row:

```text
rate = assumed delivered FLOP/s / modeled FLOP per charged position
wall hours = total charged positions / rate / 3600 × elapsed allowance
GPU-hours = wall hours × number of GPUs occupied by that job
campaign GPU-hours = 3 × one-schedule GPU-hours + initial qualification
```

## 5. GPU-hours for one integrated schedule

Values include the 25–50% elapsed allowance, exclude the one-time qualification
allowance, and are rounded. Every row is conditional on implementation, resource
and numerical qualification. A larger context, richer media or extra passes
must be charged rather than silently reducing training scope.

| Hardware / configuration | 100M positions | 1B positions | 5B positions |
|---|---:|---:|---:|
| DigitalOcean 1 H200, BF16 compute | **12–42 GPU-h** | **116–417 GPU-h** | **579–2,084 GPU-h** |
| DigitalOcean 1 H100, memory-qualified | 14–52 GPU-h | 139–521 GPU-h | 695–2,605 GPU-h |
| DigitalOcean 1 L40S, offload-qualified | 43–139 GPU-h | 434–1,389 GPU-h | 2,171–6,946 GPU-h |
| 2 RTX 3090 Ti, sharding/offload | 154–556 GPU-h | 1,544–5,557 GPU-h | 7,718–27,785 GPU-h |
| 2 TITAN RTX, FP16/sharding/offload | 309–1,113 GPU-h | 3,091–11,127 GPU-h | 15,454–55,636 GPU-h |
| 1 H200, strict FP32 sensitivity | 49–148 GPU-h | 495–1,484 GPU-h | 2,473–7,418 GPU-h |
| This M2 Ultra, current full FP32 scope | **Not feasible under current residency/guard** | **Not feasible** | **Not feasible** |

For the 1B case, the dual-3090 Ti job takes **772–2,779 wall hours
(32–116 days continuously)**; the dual-TITAN job takes **1,545–5,564 wall hours
(64–232 days)**. Their GPU-hour totals are twice their wall-hour totals. The
single-H200 schedule takes approximately **5–17 days** of continuous occupancy.
Shared-machine availability and scheduling gaps extend calendar time.

The strict FP32 row makes the precision tradeoff visible. Full scope does not
require preserving the Mac's slow execution strategy, but the faster BF16/FP16
rows do require independent loss/gradient/retention qualification.

## 6. Development-campaign reserve and DigitalOcean cost

Current on-demand rates are **H200 $4.47/GPU-hour, H100 $4.41, L40S $1.57**.
The listed capacities are 141, 80 and 48 GB respectively. Rates were checked on
26 October 2026; instance availability and eligibility need confirmation before
provisioning. No reservation, spot discount, paid resource or account change was
made. [DigitalOcean pricing](https://www.digitalocean.com/pricing/gpu-droplets),
[regional availability](https://docs.digitalocean.com/products/droplets/details/gpu-availability/).

| Total positions per main schedule | H200 three-equivalent campaign, including 8–16 qualification GPU-h | H200 compute cost |
|---|---:|---:|
| 100M | 43–141 GPU-h | $190–$630 |
| 1B | **355–1,266 GPU-h** | **$1,590–$5,660** |
| 5B | 1,745–6,268 GPU-h | $7,800–$28,020 |

For 1B positions, alternative three-equivalent campaigns are approximately
425–1,579 H100 GPU-hours ($1,870–$6,960), or 1,314–4,192 L40S GPU-hours
($2,060–$6,580). Lower hourly price is not automatically lower total cost.
The expected H200 advantage here is capacity and avoidance of mandatory offload,
not a claimed universal speed ratio.

DigitalOcean also lists a **192-GB MI300X at $2.59/GPU-hour**. It is a serious
capacity/cost candidate, but Spark's complete ROCm, custom attention, optimizer
and extension path has not been qualified. No invented MI300X training rate is
used to rank it against the CUDA estimate. RTX 6000 Ada's 48 GB has the same
resident-state obstacle as L40S before an offload/residency change.
[DigitalOcean product/pricing](https://www.digitalocean.com/pricing/gpu-droplets).

GPU costs exclude storage, egress, taxes, idle time and unspecified large-teacher
generation. Local cost excludes electricity and hardware depreciation. CPU data
preparation, source/asset admission and engineering time remain separate.

## 7. Memory feasibility comes before the hour forecast

At the rounded 4.4B planning count, persistent training state alone is:

| Explicit storage plan | GiB before activations and scratch |
|---|---:|
| FP32 weights/gradients/two FP32 moments, 16 bytes/parameter | **65.57** |
| BF16 stored weight + FP32 master/gradient/two moments, 18 bytes/parameter | **73.76** |
| BF16 weight + FP32 gradient/moments, no master, 14 bytes/parameter | 57.37 |
| BF16 weight/gradient + FP32 moments, no master, 12 bytes/parameter | 49.17 |

The 12-/14-byte plans are numerical/storage alternatives, not the repository's
current strict FP32 `NativeFusedAdamW` ownership. AMP compute alone does not
remove FP32 weights, gradients or moments. FP8 artifact bytes do not establish
FP8 full training, and no 8-bit/Muon optimizer is substituted in this review.

The pinned corrected ledger projects **89.46 GiB resident** and up to
**121.27 GiB with its declared BF16 restore staging**, even at its smaller
4.3208B inventory. These are analytical sums of declared buffers, not measured
CUDA requirements; CUDA clipping/streaming restore may use different temporaries.
Do not assume an MPS reduction buffer is a mandatory CUDA allocation, or that
an 80-GB card fits because the checkpoint is only 7.66 GiB.

**M2 Ultra:** 65.57 GiB of FP32 persistent state already exceeds both its physical
64-GiB unified memory and the unchanged 50-GiB process guard. Moving state to
the CPU does not create extra physical memory on this machine. Even the
49.17-GiB no-master alternative has almost no driver headroom before activations.
Consequently there is no finite executable formal-training budget for the
current contract. Preserve the Mac for MiniCPM mechanisms, diagnostics and
smaller declared scopes; do not relabel them as complete Spark FT.

**Each two-card pair:** both [RTX 3090 Ti](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
and [TITAN RTX](https://www.nvidia.com/content/dam/en-zz/Solutions/titan/documents/titan-rtx-for-creators-us-nvidia-1011126-r6-web.pdf)
have 24 GB per card. Even ideal partitioning of the current 16-byte state is
about 32.78 GiB/card, so sharding alone is insufficient. Qualify optimizer/parameter
offload with sufficient CPU memory, block-level gathering, checkpointing and
tied-weight ownership. Plan **128 GiB host RAM as a starting target; 256 GiB
provides more staging/data headroom**, subject to the actual live ledger. CPU,
PCIe topology, usable peer access and an installed NVLink bridge must be
inventoried. NVLink support is not automatic application-level memory pooling.

[PyTorch FSDP](https://docs.pytorch.org/docs/2.14/fsdp.html) supplies parameter,
gradient and optimizer-state sharding/offload mechanisms; gradient accumulation,
mixed precision, tied parameters and save/restore still need tests for this
custom model. [ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/)
is another implementation family. A CPU optimizer substitution changes the
execution implementation and checkpoint contract; it cannot silently replace
the current native fused path. The hour bands are contingent on that work.

**Cloud choice:** H200 is the first capacity candidate to qualify. H100 needs an
efficient phase-residency implementation; L40S needs offload. A cloud CUDA guard
would need a separate hardware-specific admission policy. This report leaves
the Mac's 50 GiB /512 MiB swap-growth /15% host-free policy unchanged.

## 8. Conditional Mac arithmetic and separate local jobs

For hardware comparison only, extrapolating measured Mac forward/backward
equivalent work, the measured clip/optimizer costs scaled by parameter count,
and dense attention at 2048 positions gives approximately **89–215 processed
positions/s** if a full-scope memory implementation could be qualified. This
is neither a measured Spark speed nor a runnable configuration:

| Charged positions | Conditional compute-only M2 device-hours, including elapsed allowance |
|---|---:|
| 100M | 162–470 |
| 1B | 1,617–4,698 |
| 5B | 8,083–23,491 |

For 1B, this is 67–196 days of continuous occupancy, or 135–392 calendar days
if the Mac is available half the time. These numbers omit the unknown cost
and feasibility of the required memory implementation. They must not be used
as a current executable commitment or a promise that offload fits.

For the two local pairs working independently, an illustrative three-candidate
campaign runs **two whole schedules on the 3090 Ti pair and one on the TITAN
pair**. At 1B positions per schedule:

- Wall time is `max(2 × 3090-pair schedule, TITAN-pair schedule)`, approximately
  **1,545–5,564 hours, or 64–232 days**, before initial qualification.
- Total occupied GPU time is `2 × (2 × 3090-pair hours + TITAN-pair hours)`,
  approximately **6,178–22,241 GPU-hours**, plus **32–80 qualification GPU-hours**.

This parallelizes independent candidates and controls. It does not accelerate
one candidate by summing the two pairs' token rates, merging independent
weights, or treating four heterogeneous cards as a single synchronized job.

## 9. Sensitivity and the next measurement

The main unresolved inputs are data volume/epochs, source/target sequence
distribution, retained media resolution/rate, executed reader density, effective
batch, exact inner/outer schedule and complete tensor inventory.

| Change | What to charge |
|---|---|
| 1B to 5B total positions | Approximately 5× variable hours; qualification remains a fixed allowance |
| 2048 to 8192, qualified sliding kernels | Arithmetic budget rises from 40.01 to 43.63 GFLOP/position; memory/efficiency effects can add more |
| 8192 with dense masked attention | About 57.22 GFLOP/position, 1.43× the main arithmetic budget, before kernel/memory penalties |
| Richer media before compression is qualified | Count the extra conditioning positions and encoder samples/frames, not just the final text length |
| Extra complete sensing/teaching pass | Count its positions again; do not hide it in a small-module parameter count |
| More/all-layer readers or a larger expert bank | Recompute active calls and resident state separately; stored but inactive experts are not free memory |
| Another 1B dense parameters active on every position | Roughly another 8 GFLOP/position and 14.90 GiB of FP32 state before scratch; sparse storage has different active work |
| Exact differentiable inner AdamW instead of the specified first-order reference | Separate algorithm, retained graph and repeated-backward budget; no exact-meta-learning cost claim is included |

For the planned 16-second window, compression proposes 640 new media positions.
Using six full 196-patch views and 50-Hz audio instead yields **2,072 positions**
before ordinary text, about 3.24× that media-position payload. The full-stage
quality controls must determine which rate is acceptable; the estimate does
not reduce sensory scope to force a favorable number.

The next useful paid step is a **bounded matched Spark qualification pilot**,
not committing the entire estimated budget: establish the complete trainable
inventory and precision contract; profile 2048 and 8192 total positions with
representative media; separately exercise the four-transition `F` graph and
bounded `P` inner/future-query path; test checkpoint/restore and sampled memory;
then replace the assumed rate with synchronized steady measurements. Record
both processed and supervised positions, unique media, encoder time, optimizer
steps and all required retention/grounding controls. No paid job was started here.

The estimate is reproducible offline:

```bash
.venv/bin/python scripts/estimate_sparkx_training_budget.py \
  --output reports/local/sparkx-budget-recomputed.json
```

See [assumptions](../../configs/sparkx_formal_training_budget_estimate_v1.json),
[machine-readable result](../../reports/sparkx_gpu_hour_estimate_20261026/estimate.json),
[official source record](../../reports/sparkx_gpu_hour_estimate_20261026/external_sources.json)
and [validation](../../reports/sparkx_gpu_hour_estimate_20261026/validation.json).
