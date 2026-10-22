# 02 - Two stages, explicit controls, measured hardware

**Current native preflight review: 22 October 2026 (America/Chicago). Original bundle planning label: 23 October 2026. All proposed budgets and acceptance margins require preregistration; the native measurements below establish short-run execution only.**

Use the [detailed substage roadmap](stages/00_Master_Roadmap.md) for dependencies, artifacts, resource limits and exit gates. The [constant-zero diagnosis](experiments/20261022_Constant_Zero_Diagnosis.md) preserves the failed preflight and the successful eight-class training diagnostic separately.

## 1. The two-stage program

Stage I is a small-model laboratory with three concurrent tracks: architecture/training, verified experience learning, and authentic Replica data. Stage II ports only useful mechanisms to the larger models. The small models remain regression and rapid-experiment platforms afterward.

| Step | Work | Deliverable and exit condition |
|---|---|---|
| I-0 | Inventory machines; pin exact model, framework, tokenizer and dataset revisions | Native load/update profile; unique parameter counts; reproducible environment |
| I-1 | Run original-model and source-text-memory baselines | Task, retention and calibration scores under fixed inference budgets |
| I-2 | Tiny overfit; ordinary full tuning; modules-only; full augmented tuning | Correct masks, actual gradients/updates, export/reload and no causal leakage |
| I-3 | Compare architectural benefit on held-out tasks | Fair D0/D1/D2 matrix, controls and uncertainty |
| I-4 | Freeze chosen developmental base; run ten initial experience cycles | Acquisition, strategy transfer, learning-selection and retention measured separately |
| I-5 | Port to MiniCPM-V-4.7-1B | Native VLM parity; actual image gradients; hybrid-cache correctness |
| I-P | Collect authentic decisions in parallel | Source archive, identity snapshot and sealed person-specific test set |
| II-0 | Select a strong small-model mechanism and repeat its result | Multiple seeds; generalization beyond trivial fixture templates |
| II-1 | Build fresh Gemma/MiniCPM-V large interfaces | Model-specific readers and calibrators, new baseline and memory profile |
| II-2 | Frozen-base longitudinal large-model trials | Gains beyond records-only and fixed-replay controls at stated budgets |
| II-3 | Longer operation and personal parallel branch | Versioned learning history, drift tests, rollback and deletion workflow |

Do not wait for every optional memory algorithm before running I-4. Do not skip the small VLM test by assuming the dense text wrapper applies to its hybrid architecture.

## 2. Developmental comparisons

Retain R2 names in reports so work remains comparable:

| Arm | Base architecture | Original weights | New modules | Question |
|---|---|---|---|---|
| B0 | Original | Frozen | None | What does the checkpoint already do? |
| B1 | Original + source-text memory/tools | Frozen | None | How much does exposing useful evidence help? |
| D0 | Original | Full text fine-tuning | None | Does ordinary training explain the gain? |
| D1 | Original + reader/workspace | Frozen | Trained | Can the base use a new interface without co-adaptation? |
| D2 | Original + reader/workspace | Full text fine-tuning | Trained | Does joint architectural adaptation help beyond D0? |
| D3 | Original + reader/workspace | LoRA | Trained | Is parameter-efficient co-adaptation sufficient? |
| D4 | Original | LoRA | None | Does LoRA alone explain D3? |

The starter runner implements baseline, full, modules_only and full_augmented. D3/D4 are specified next work; no LoRA trainer is claimed in this package. B0/B1 need broader task/generation evaluation beyond the short fixture checks.

Begin with D0/D1/D2. Give all arms the same eligible examples, evidence, tools and response targets. The first implementation leaves source text in every prompt; augmented models also receive a trainable representation of that same information. Later compare an internal-memory-only condition and an equal-token-budget text retrieval condition.

Record both equal-data and equal-compute results. More modules may cost more compute even at the same token count. A full-tuned model is a strong comparator, not a theoretical upper bound.

### Mechanism controls after the first stable matrix

Compare the recurrent workspace with parameter-matched feed-forward capacity; extra workspace steps with additional ordinary inference tokens or tool calls; learned memory selection with a fixed heuristic; internal memory with the same evidence expressed as text; private fast state with shared state at equal state bytes; and a multi-rate extension with a capacity-matched single-rate extension.

If an architectural change only increases confidence, or uses more evidence, or performs more verification calls, identify that cause. A latent-space plot is diagnostic rather than a reasoning result.

## 3. Small-model data and optimization

Use the [20,000-target pilot](04_Datasets_and_Generation.md): 8,000 general replay, 3,000 verified reasoning, 4,000 temporal-memory episodes, 2,000 generic personalization examples, 1,000 tool-use/recovery examples, and 2,000 inquiry/control examples.

Before that, use the bundled 48 deterministic examples and then 64-256 carefully inspected episodes. These verify infrastructure. They are too small and simple to establish cognition, architectural superiority or broad transfer.

Proposed initial training budget is **8-32M processed tokens per trained arm**, increasing only when learning curves justify it. Count prompt, target, evidence-encoding and replay tokens, including repeated epochs. Report teacher-generation tokens and full-history ingestion separately. A 20,000-event dataset with long histories will exceed this budget unless training windows and history processing are designed explicitly.

| Setting | Proposed initial search | Rationale |
|---|---|---|
| Sequence length | Actual 256 and 512 now qualified for short D0/D2 runs; 1024 remains proposed | Two updates per length and arm with native restart passed; sustained workloads remain separate |
| Microbatch | 1 | Clear per-episode causal/state ownership |
| Effective batch | 8-32 episodes via accumulation | Increase only after profiling and loss normalization checks |
| Muon original-matrix LR | Inherits corresponding original auxiliary LR unless explicitly overridden; diagnostic `1e-4` | RMS-aligned update scale supports a matched-rate starting point; tune before campaigns |
| Muon new-matrix LR | Inherits corresponding new auxiliary LR unless explicitly overridden; diagnostic `1e-4` | New interfaces still need their own search |
| Auxiliary AdamW LR, original/new | 1e-5 / 1e-4 provisional defaults | Embeddings, output head, 1D and other ineligible tensors remain trainable |
| AdamW comparison LR, original/new | 5e-6, 1e-5, 2e-5 / 1e-4, 3e-4 | Retained as an explicit comparison arm |
| LoRA LR, later arm | 1e-5, 3e-5, 1e-4 | Not assumed identical to full-tuning optimum |
| Compute | FP32 for initial MPS validation; BF16 only after backend checks | Keep parameter, gradient, state and temporary dtypes explicit |
| Selected optimizer experiment | Muon 8-bit + auxiliary FP32-state AdamW | Same project implementation on macOS MPS, later CUDA |
| Muon state quantization | Linear signed INT8 momentum, 2048-element blocks, FP32 scales | Quantizes persistent momentum only |
| Activation checkpointing | Nonreentrant; pure forwards | Reduce memory without duplicate external writes |

These are proposed starting points. The selected optimizer is the project's experimental **Muon 8-bit** path. It uses eligible 2D hidden matrices for Muon, with auxiliary AdamW groups for embeddings, output head, learned slot queries, 1D tensors and other ineligible parameters. Parameter routing changes the update rule, not the D0/D1/D2 trainable scope: full-text arms must still update every intended original parameter, including embedding/head tensors. Deduplicate tied tensors and audit every trainable tensor's assignment.

The portable eager PyTorch implementation stores Muon momentum in blockwise INT8, then dequantizes and orthogonalizes in FP32 using Newton-Schulz steps. Model parameters and gradients remain FP32 in the initial path, and auxiliary AdamW moments remain FP32. It is a project experiment inspired by [Effective Quantization of Muon Optimizer States](https://arxiv.org/abs/2509.23106), not a reproduced paper result or a claim of native fused-kernel speed. PyTorch's [Muon documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html) provides the matrix-eligibility reference; stock `torch.optim.Muon` is not itself this 8-bit optimizer. Installing bitsandbytes or torchao does not turn their AdamW implementations into Muon 8-bit.

Retain `adamw` as an explicitly selected numerical comparison. This implementation uses the RMS-aligned Muon adjustment `0.2 * sqrt(max(rows, columns))`; [PyTorch's Muon documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html) explains that this adjustment supports reusing AdamW-tuned rates. Muon rates now inherit the corresponding auxiliary rates when omitted, replacing the historical implicit `1e-3` setting. The matched `1e-4` diagnostic below is a declared memorization recipe, not a campaign optimum.

The trainer now implements configurable warmup, constant/cosine schedules and contract-bound completed-update checkpoints containing model, optimizer, RNG, schedule and data-cursor state. Tiny CPU interrupted/resumed training matches exactly; MPS mixed-optimizer/RNG restoration tests also pass. A fresh-process native checkpoint continuation from updates 8 to 12 completed, retaining 8/8 correct training generation. This establishes the measured native continuation path; full-size numerical equivalence against an uninterrupted control remains untested. Mid-accumulation checkpoints, offload and distributed training are unsupported.

For every run, record eligible parameter names, unique trainable count, nonzero gradients, changed tensors, optimizer dtypes and effective loss normalization. During full augmented tuning, a writer must use current candidate embeddings; stale cached features are invalid unless deliberately labelled frozen-teacher targets.

## 4. What to test first

### A. Verifiable technical episodes

Use small finite-state controllers, bounded bit-vector transforms, simple protocol rules and deterministic Python microtasks. A useful episode omits one decisive constraint: the model must ask or inspect an allowed trace, propose a solution, observe a verifier failure, repair it and later solve a structurally changed task.

The oracle must be independent of the answer generator. Start with enumerated finite domains and a second implementation; then add Verilator or a formal-property tool with explicit reset/clock assumptions. Merely passing a test generated by the same LLM is insufficient.

Hold out topology, operation composition, reset semantics or rule family, as well as numeric values. Changing names and integers tests only limited instance transfer. The bundled XOR fixture deliberately claims only input-level transfer within a known grammar.

### B. Temporal memory and active inquiry

Include new facts, corrections, expired facts, contradictory reports, missing answers, deletion, distractors and unauthorized records. Evaluate a meaningful action choice: answer, ask, retrieve again, verify or stop. An empty record store should not lead the model to invent a human memory.

### C. Replica decisions

Hold out authentic decisions by topic/decision family. Evaluate choice, priority ranking, needed information, decisive counterfactuals, principled disagreement and legitimate preference change. Measure factual correctness separately from fidelity to the person's documented decision.

### D. Multimodal extension

Use genuine images and temporal sequences. Evaluate whether an answer changes when the relevant frame changes, whether shuffled images damage performance, and whether the vision path receives gradients. Caption-only improvement is a text result. The small VLM needs its own processor, collator, modality loss masks and hybrid-cache tests.

## 5. Post-freeze longitudinal study

Start from one selected developmental checkpoint and a common permitted initial archive.

| Arm | Persistent change | Interpretation |
|---|---|---|
| T0 | None | Static system |
| T1 | Evidence records only | Acquisition |
| T2 | Fixed-schedule verified replay / candidate updates | Ordinary continuing adaptation |
| T3 | Selected practice + bounded candidate updates | Value of autonomous learning selection |
| T4 | T3 plus the chosen fast-state or multi-rate mechanism | Incremental value of online state |

Run ten initial chronological cycles; extend toward 30-50 only after instrumentation and error bars are useful. Each cycle contains new experience, a permitted update opportunity, independent comparison, and later unseen tasks. Include a no-improvement period, a distribution change, a deliberately damaging candidate that must be rejected, restart and rollback.

When T3 chooses different data, include a replay-matched T2 to separate better data selection from a better optimizer. Match evaluation compute; disclose additional learning compute. The learned director can consume development feedback, not sealed audit examples.

Let S(i,j) be performance after cycle i on held-out family j. Report the full matrix. Average forgetting over old families can be computed as the best previous score minus the current score, with per-family scores and uncertainty beside the mean. Report forward transfer, delayed recall, inquiry cost, retained skills, fabricated-autobiography rate, learning cost and update rejection.

## 6. Acceptance and statistical resolution

Predeclare the smallest useful task gain and acceptable regression. R2's **+2 percentage points target improvement, at most 1 point average regression and 2 points on critical tasks** are reasonable starting proposals, not universal standards. An inconclusive result is a valid outcome.

Use three training seeds for shortlisted arms; one-seed screens are exploratory. Resample independent conversations, decision episodes or task families, not thousands of correlated paraphrases as if they were independent. Public benchmark contamination from base pretraining remains a possible confound, so keep a fresh private audit set too.

The starter gate uses paired bootstrap intervals and a conservative constant-sample guard. With no observed differences in 200 independent pairs, a 95% two-sided zero-count bound still permits about 1.83% unseen adverse probability:

$$u=1-0.025^{1/200}\approx0.0183.$$

Thus a 1% retention tolerance should not automatically pass at that sample size. The code returns inconclusive in this case. Its interval routine is a screening utility, not a solution to every clustered, adaptive or repeated-testing design.

Separate train, development, calibration, learning-policy reward, rotating promotion windows and final audit. Access to promotion scores also creates selection pressure; cap proposals and log queries. The final audit does not influence checkpoint selection. No average gain compensates for a failed hard causality/identity/scope invariant.

## 7. Model inventory

| Model | Verified role and architecture | R3 scope |
|---|---|---|
| openbmb/MiniCPM5-2B | Native Llama; 42 layers; 2048 hidden width; **2,516,756,480 total parameters** | First text full-tuning laboratory |
| openbmb/MiniCPM-V-4.7-1B | Hybrid Qwen3.5 text core plus vision; native V4.7 processor | Second, separate VLM port |
| google/gemma-4-26B-A4B-it | Official spelling is A4B | Later frozen-base / adapter-and-components port |
| openbmb/MiniCPM-V-4.7-35B-A3B | Hybrid multimodal MoE | Later large VLM port |

Sources: [MiniCPM5 card](https://huggingface.co/openbmb/MiniCPM5-2B), [small V4.7 config](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B/blob/main/config.json), [Gemma card](https://ai.google.dev/gemma/docs/core/model_card_4), [large V4.7 config](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B/blob/main/config.json). The requested checkpoints are retained; no Base/SFT sibling is silently substituted.

Count actual unique tensors at the pinned revision. Active MoE parameters determine only part of per-token compute; they are not the storage requirement. For a VLM report decoder, embeddings/head, vision, projector and new modules separately. Full_text and full_vlm are different claims.

## 8. Hardware: current host and later CUDA target

The current host is the **Apple M2 Ultra Mac Studio**, arm64, with **24 CPU cores and 64 GiB unified memory**. The user's `M@Ultra` description is resolved by the actual host inventory. The initial experiment runs here using PyTorch MPS. The later target is the **NVIDIA RTX PRO 4000 Blackwell SFF Edition**, which NVIDIA specifies with **24 GB GDDR7 ECC**. Its CUDA experiment will use the same project Muon 8-bit algorithm and audited parameter routing. [NVIDIA product specifications](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-4000-sff/).

Earlier dual-3090 Ti, TITAN RTX, DRAM and Optane discussions are historical alternatives, not the installed-machine inventory or prerequisites for this setup. Reinventory the actual CUDA host before its run; do not assume multiple GPUs or pooled VRAM.

### Persistent memory arithmetic

For full training with FP32 parameters and gradients, each Muon parameter requires approximately **9 bytes**: 4 for its weight, 4 for its gradient and 1 for quantized momentum, plus a shared FP32 scale per 2048-element block. Each auxiliary AdamW parameter requires approximately **16 bytes**: weight, gradient and two FP32 moments. Thus a fully trainable mixture needs approximately `9 * P_muon + 16 * P_adamw + scale_bytes` before activations and temporaries. For frozen-base/module-only runs, include loaded frozen weights separately; optimizer state follows only trainable parameters.

| Trainable parameters | Illustrative all-Muon lower bound, 9 bytes/parameter before scales | All-AdamW reference, 16 bytes/parameter |
|---|---:|---:|
| 1.0B | 9.00 GB / 8.38 GiB | 16.00 GB / 14.90 GiB |
| 2.0B | 18.00 GB / 16.76 GiB | 32.00 GB / 29.80 GiB |
| Actual MiniCPM5-2B | 22.65 GB / 21.10 GiB | 40.27 GB / 37.50 GiB |

The all-Muon column is an arithmetic lower bound, not this model's actual optimizer mixture: embeddings/head and other ineligible tensors use AdamW. Scales add roughly 4 bytes per block, including each tensor's partial final block. FP32 dequantization, momentum/update intermediates, Newton-Schulz matrices, activations, vocabulary logits, loss buffers, temporary casts and allocator headroom are additional. MiniCPM5's 512-token batch alone has 66.8 million vocabulary-logit entries before loss temporaries.

The [sealed snapshot tensor inventory](../reports/minicpm5_snapshot_inventory.json) now confirms **1,981,808,640 Muon parameters and 534,947,840 auxiliary AdamW parameters**, totaling 2,516,756,480. With this FP32-resident layout, weights, gradients and persistent optimizer tensors alone require **26.40 GB / 24.59 GiB**, before new modules, activations and temporary buffers. This is shape-based arithmetic from the actual checkpoint, not a measured training peak.

**Mac Studio:** unified memory is shared by the OS, applications and training; the full 64 GiB is not an isolated GPU allocation. The historical native D0 preflight below fit at the measured 55-token workload with a sampled MPS driver peak of **29.15 GiB**. The balanced eight-update run, accumulating eight 55-token examples per update, reached **29.18 GiB**. Longer sequences, larger effective batches and sustained runs need separate profiles.

**Later 24 GB Blackwell SFF:** the current FP32-resident full-text state budget already exceeds this card's capacity. Full D0/D2 requires a tested offload/sharding adapter or a separately validated lower-residency layout, while retaining Muon 8-bit and the full trainable scope. Begin CUDA qualification with the tiny model; module-only remains a separately labelled control. An inference device map does not provide training-aware offload. [DeepSpeed ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/) is a possible later adapter, not an installed or validated capability of this runner.

## 9. Mac Studio first-run protocol

Use the current Mac as the first training and development machine. CPU numerical checks and tiny MPS D0/D1/D2 updates were followed by the native pretrained baseline and D0 preflight on 22 October 2026. Confirm gradients and updates in both the Muon matrices and trainable auxiliary AdamW tensors for each new workload. Record block size, momentum/scales dtypes, eligible parameter names/counts and optimizer bytes with the run.

The historical native preflight used microbatch 1, accumulation 1 and a configured maximum length of 256; every fixture was actually **55 tokens**. The later S0.5 screen below measured genuine 256-token episodes before progressing to 512, with accumulation eight and native restart. Keep another teacher model out of memory during profiling. Record available memory, MPS allocation, memory pressure, swap, throughput and save/reload behavior. Preserve OS memory safeguards. Higher-order fast-state training and custom kernels need separate backend checks. See [PyTorch MPS backend](https://docs.pytorch.org/docs/2.14/notes/mps.html).

### Native baseline and full D0 preflight: measured 22 October 2026

The [experiment record](../reports/minicpm5_mps_experiment_20261022.json), [baseline report](../reports/minicpm5_mps_baseline_report.json) and [D0 report](../reports/minicpm5_mps_preflight_report.json) use the sealed `openbmb/MiniCPM5-2B` revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52`, all **2,516,756,480** original parameters, FP32 parameters/compute and MPS without CPU fallback. The environment was PyTorch 2.14.1, Transformers 5.20.2 and safetensors 0.8.0. Both runs used seed 17 and the same 16 deterministic development fixtures with 32 supervised response tokens.

| Measurement | Native frozen baseline | Native full D0 preflight |
|---|---:|---:|
| Optimizer updates | 0 | 2 |
| Original trainable parameters | 0 | 2,516,756,480 |
| Actual sequence length | 55 | 55 |
| Development response-only NLL | 15.011792898 | 1.572720953 |
| Strict generation matches, same 16 fixtures / 256 new-token cap | 0/16 | 2/16, all answers `0` |
| Sampled peak MPS allocated memory | 9.41 GiB | 24.74 GiB |
| Sampled peak MPS driver memory | 10.07 GiB | 29.15 GiB |
| Process-lifetime peak RSS | 14.49 GiB | 14.52 GiB |

D0 used nonreentrant activation checkpointing, Muon LR `1e-3`, auxiliary AdamW LR `1e-5`, momentum `0.95`, five Newton-Schulz steps, 2048-element INT8 momentum blocks and weight decay `0.01`. All 294 eligible matrices (1,981,808,640 parameters) used Muon; the other 87 tensors (534,947,840 parameters) used auxiliary AdamW. Recorded persistent optimizer tensors occupied 6,265,262,080 bytes. Each update processed one 55-token example with two supervised target tokens: **110 processed tokens and four supervised tokens total**, taking **47.764 seconds**, or **2.303 processed tokens/second**. The profiled process, including loading, evaluation, export and reload, took **64.228 seconds**.

All **381 original tensors had nonzero gradients** in the first update. The [independent saved-checkpoint comparison](../reports/minicpm5_mps_preflight_checkpoint_audit.json) found changes in all 381 tensors, including input embeddings and output head, with the native names/shapes preserved. Export/reload compared 7,180,800 logits from one development example and observed a maximum absolute difference of **0.0**. This establishes the full native D0 trainable scope and that export/reload path; it does not establish resumable optimizer or scheduler state.

Memory samples were taken every 0.25 seconds; sampled peaks can miss shorter transients. Process RSS and MPS driver memory overlap in unified memory and **must not be added**. The RSS high-water mark includes the process lifetime, including CPU loading. Host swap usage was 1,562.69 MiB at both the start and end of each run; unchanged endpoints do not measure intervening swap activity. The host memory-pressure command reported 93% free at preflight start and 72% at its end. These are endpoint diagnostics, not a continuous pressure trace.

The matched [baseline generation report](../reports/minicpm5_mps_generation_baseline_256_report.json) and [D0 generation report](../reports/minicpm5_mps_generation_D0_256_report.json) used the same 16 development examples, 53-token prompts, native source chat template with thinking disabled, greedy cached decoding and a maximum of 256 new tokens. The baseline had 0/16 strict matches; eight responses reached the cap and eight ended by EOS. D0 had 2/16 strict matches, but **every completion was `0` followed by EOS**, two generated tokens each. This is constant-answer collapse, not learned XOR or generalization. The lower response NLL is consistent with learning response formatting/EOS and the constant answer; it does not establish independent task improvement.

These historical results are infrastructure observations on a tiny, templated fixture set after only four supervised tokens. They are not evidence of generalization, architecture benefit, retention or autonomous learning. Native D1/D2 optimizer training, sustained representative workloads, MLX parity and later CUDA/offload qualification remain separate work. Scheduler and completed-update state support were added for the next diagnostic below.

### Balanced full D0 tiny overfit: measured 22 October 2026

The [preregistered design](../configs/tiny_overfit_design.json) restarts from the same sealed original rather than continuing the collapsed candidate. It selects eight solver-rechecked width-4/mask-1 training rows, one for each answer `0..7`, and accumulates all eight in every update. Muon and auxiliary AdamW both use peak LR `1e-4`, two-update warmup (`5e-5`, then `1e-4`), constant scheduling and zero weight decay. Momentum, block size, five FP32 Newton–Schulz iterations, full original parameter scope and nonreentrant activation checkpointing are retained.

The [completed chunk-01 report](../reports/minicpm5_balanced_mask1_chunk01_report.json) records **8/8 strict correct training generations with eight distinct answers at both updates 4 and 8**. Every completion contains one answer token followed by EOS, within the eight-new-token budget. The 64-update cap stopped at eight after the second perfect check. All 2,516,756,480 original parameters were trainable in FP32 on MPS, and all 381 original tensors had nonzero first-update gradients. [Detailed diagnosis and answer table](experiments/20261022_Constant_Zero_Diagnosis.md), [consolidated experiment record](../reports/minicpm5_balanced_overfit_experiment_20261022.json).

| Measurement | Balanced eight-update D0 diagnostic |
|---|---:|
| Actual prompt + target length | 55 tokens |
| Microbatch / effective examples per update | 1 / 8 |
| Processed / supervised training tokens | 3,520 / 128 |
| Update time / processed training tokens per second | 157.701 s / 22.321 |
| Full profiled process, including checkpoint/export/reload | 192.287 s |
| Sampled peak live MPS allocations / driver memory | 24.88 / 29.18 GiB |
| Process-lifetime / sampled peak RSS | 14.49 / 8.63 GiB |
| Host swap used, start → end | 1,562.69 → 1,562.69 MiB |
| Dev answer NLL / EOS NLL | 1.093525 / approximately `7.45e-9` |
| Dev combined response NLL | 0.546763 |
| Native export/reload maximum absolute logit difference | 0.0 across 7,180,800 logits |

The completed-update checkpoint at step 8 includes a byte receipt and optimizer/runtime state; native export/reload passed independently. The [completed chunk-02 continuation](../reports/minicpm5_balanced_mask1_chunk02_report.json) restored it in a fresh process and completed updates 9–12, retaining 8/8 strict correct training answers and eight distinct responses at update 12. Export/reload again matched exactly. The resumed process took 121.582 seconds including restore/checkpoint/export/reload, with 80.355 additional update seconds; cumulative training totals are 238.056 update seconds, 5,280 processed tokens and 192 supervised tokens. Sampled live MPS/driver peaks were 24.83/29.18 GiB, process-lifetime RSS 13.87 GiB, and swap endpoints remained 1,562.69 MiB. No full-size uninterrupted-versus-resumed equivalence control was run.

Separate fresh-process native generation from both saved exports completed:

| Saved export | Training strict correct, 8-new-token cap | Development strict correct, 256-new-token cap |
|---|---:|---:|
| Selected original, update 8 | [8/8](../reports/minicpm5_balanced_chunk01_export_train_report.json) | [14/16 (87.5%)](../reports/minicpm5_balanced_chunk01_export_dev_report.json) |
| Resume probe, update 12 | [8/8](../reports/minicpm5_balanced_chunk02_export_train_report.json) | [13/16 (81.25%)](../reports/minicpm5_balanced_chunk02_export_dev_report.json) |

All 48 completions across these four evaluations contained one answer token plus EOS, with no token-budget caps. Development uses the same 16 width-5/mask-2-or-6 rows and the same 256-new-token cap as the historical parent/collapsed comparisons. Its source families are distinct from training, but this tiny synthetic development set has been reused during investigation; it is not a sealed audit or evidence of broad generalization. Independent CPU checkpoint audits found changes in all 381 original tensors, including embedding/head, with native names/shapes preserved for [update 8](../reports/minicpm5_balanced_chunk01_checkpoint_audit.json) and [update 12](../reports/minicpm5_balanced_chunk02_checkpoint_audit.json).

The declared stopping rule selects the original eight-update artifact before these development scores; update 12 is a resume probe, not a new selected winner. Its dev response NLL worsened to 1.232098 (answer 2.464197, EOS 0.0), consistent with the need to measure development behavior separately from training memorization. The combined intervention removes constant-zero output and improves this fixture score; balancing, accumulation, rates, warmup, weight decay and exposure changed together, so their individual causal contributions remain unisolated. Broader transfer and retention require new independent evidence.

The same sampling limits apply: 0.25-second memory samples can miss transients; process RSS and driver allocations overlap; unchanged swap endpoints do not measure intervening traffic. This is a 55-token training result, not a genuine 256-token capacity measurement. See [S0 readiness and restart gates](stages/01_Environment_and_D0_Readiness.md) and the [runbook](03_Implementation_and_Runbook.md) before expanding the campaign.

### Actual 256-token and 512-token D0 capacity: measured 22 October 2026

The [preregistered capacity design](../configs/capacity_design.json) now passes its short D0 scope at both lengths. Each length starts from the fresh sealed original and runs update 1, then restores its completed-update native model/runtime checkpoint in another process for update 2. The [256](../data/diagnostics/capacity_256/manifest.json) and [512](../data/diagnostics/capacity_512/manifest.json) workloads contain complete chronological register events, not padding-only inflation: every sequence is exactly its declared native full length, with two supervised answer/EOS tokens. Both splits contain all eight answer classes and separate source/world groups. These are capacity fixtures, not quality-training data.

| Actual full length | Completed updates / accumulation | Processed / supervised tokens | Total update time | Maximum sampled live / driver memory |
|---:|---:|---:|---:|---:|
| 256 | 2 / 8 | 4,096 / 32 | 44.988 s | 24.956 / 29.094 GiB |
| 512 | 2 / 8 | 8,192 / 32 | 50.945 s | 25.295 / 29.063 GiB |

All 381 original tensors had nonzero finite gradients, both mixed optimizer states retained int8 Muon momentum, all four export/reload checks matched with maximum absolute logit difference `0.0`, and independent [256](../reports/minicpm5_capacity256_checkpoint_audit_20261022.json)/[512](../reports/minicpm5_capacity512_checkpoint_audit_20261022.json) saved-weight audits found changes in every original tensor with native names/shapes retained. No CPU fallback or allocator-policy change occurred. The 40 GiB driver, 512 MiB sustained swap-growth and 15% sustained host-free stops were not breached; sampled host swap stayed at 1,562.69 MiB. The 256 guard exit-race correction and retrospective complete-sample review are recorded in the [capacity qualification report](experiments/20261022_Capacity_Qualification.md), together with per-process timings, memory samples and reproduction commands in the [runbook](03_Implementation_and_Runbook.md).

These are two-update D0 capacity and continuation results. They leave the original eight-update learning winner unchanged and do not prove uninterrupted-versus-resumed full-size numerical equivalence. A separate [native adapter check](../reports/minicpm5_native_adapter_qualification256_report.json) passed 256-token D1/D2 null-branch/causality and checkpointed forward/backward checks without optimizer updates or augmented generation. Subsequent fixed-context uncached generation and D2 full training capacity now pass. The [D2 result](experiments/20261022_D2_Results.md) records actual256/512 capacity, tiny8/8 training memorization, saved train8/8/dev14/16 and strict scope/restart audits. These remain bounded infrastructure and reused-fixture observations; useful architecture benefit is the next controlled study.

The common initial implementation is PyTorch, making the eventual CUDA comparison an explicit backend experiment with the same optimizer equations. MLX remains an optional later port. Apple's unified-memory behavior is described in the [MLX documentation](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html), but that does not establish parity for this project's custom optimizer or modules.

For a future MLX port, audit the whole parameter tree: [MLX-LM lora.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/lora.py) freezes the model and, for its full mode, unfreezes selected transformer blocks. Choosing all blocks does not necessarily unfreeze embeddings, final norm and output head. Call that full_blocks until the inventory proves full_text. Use nonquantized source/converted weights for full-weight training; a 4-bit inference conversion is not automatically equivalent. [MLX-VLM V4.7 support](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/minicpmv4_7/README.md) does not establish this project's custom-module or full-multimodal gradient support.

Consult the [runbook](03_Implementation_and_Runbook.md) for commands and observed checks. The native baseline, historical two-update D0 path, balanced eight-update overfit and actual 256/512 D0 screen establish the stated short execution, memorization and capacity scopes. Broader MiniCPM5 quality, sustained representative workloads, augmented training capacity, later Blackwell execution and V4.7 multimodal training require their own measurements.

## 10. The scale-up decision

Advance only when the mechanism earns its cost: D2 beats D0 on a relevant task/control, post-freeze T3 improves beyond T1, no critical retention/identity invariant fails, and the result survives restart and multiple seeds. If T3 matches T2, report useful continued adaptation without claiming better autonomous selection.

Carry source data, task generators, state contracts and evaluation protocols to larger models. Relearn readers, writers, workspace alignment and calibration per backbone. Reset incompatible fast state and caches. The larger backbone may solve tasks that were impossible for the small one, so remeasure the floor/ceiling and task difficulty.

If the small model's failure is capacity rather than a broken interface, that is useful evidence for scaling. If the small model's gain disappears against plain fine-tuning or stronger text retrieval, scaling has not yet been justified by the architecture.
