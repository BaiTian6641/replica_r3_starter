# 02 - Two stages, explicit controls, measured hardware

**23 October 2026. All proposed budgets and acceptance margins require preregistration; none are model-quality results from this bundle.**

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

The starter runner implements baseline, full, modules_only and full_augmented. D3/D4 are specified next work; no LoRA trainer is claimed in this package. B0/B1 need task/generation evaluation beyond the provided NLL runner.

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
| Sequence length | 256 for preflight; 512 then 1024 | Find a feasible representative update before scaling activations |
| Microbatch | 1 | Clear per-episode causal/state ownership |
| Effective batch | 8-32 episodes via accumulation | Increase only after profiling and loss normalization checks |
| Original-parameter LR | 5e-6, 1e-5, 2e-5 | Short developmental adaptation |
| New-component LR | 1e-4, 3e-4 | New interfaces need their own search |
| LoRA LR, later arm | 1e-5, 3e-5, 1e-4 | Not assumed identical to full-tuning optimum |
| Compute | BF16 where native support is established | Keep optimizer-state dtype explicit |
| Reference optimizer | FP32-state AdamW | A clear numerical baseline before optimizer quantization |
| Activation checkpointing | Nonreentrant; pure forwards | Reduce memory without duplicate external writes |

These are proposed starting points. The included trainer is intentionally a simple FP32-resident AdamW reference with optional BF16 autocast, no warmup scheduler and no resume optimizer state. Add the scheduler and complete resumable state before long campaigns. Do not describe its 100-step default as a training recipe that has converged.

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

## 8. Hardware: feasible routes and honest limits

The small workstation's current GPU is unspecified. The M2 Ultra Mac Studio with 64 GB unified memory is confirmed. Earlier context described a dual RTX 3090 Ti plan, 128 GB DRAM and 256 GB Optane, with additional TITAN RTX cards considered. R3 does not assume all four GPUs are installed, nor that the new small workstation is that machine.

For P trainable parameters, FP32 resident weights + FP32 gradients + two FP32 Adam moments cost 16P bytes. Other mixed-precision layouts can reach 18-20P with extra copies. Activations, vocabulary logits, workspaces, temporary casts, communication and allocator headroom are additional.

| Trainable parameters | 16 bytes/parameter | 20 bytes/parameter |
|---|---:|---:|
| 1.0B | 16.00 GB / 14.90 GiB | 20.00 GB / 18.63 GiB |
| 2.0B | 32.00 GB / 29.80 GiB | 40.00 GB / 37.25 GiB |
| Actual MiniCPM5-2B | 40.27 GB / 37.50 GiB | 50.34 GB / 46.88 GiB |

These are calculations, not measured peaks. MiniCPM5's large vocabulary also makes logits and loss buffers material. A 512-token batch has 66.8 million vocabulary-logit entries before loss temporaries.

**One 24 GB GPU:** conventional resident full AdamW training of this 2.5168B model cannot fit. Use training-aware optimizer offload/sharding or a separately named lower-memory optimizer experiment. The shipped single-device trainer intentionally does not pretend to provide offload.

**Two 24 GB GPUs:** ordinary DDP replicates state; memory does not pool automatically. A tested FSDP/ZeRO setup is necessary. Begin with the homogeneous 3090 Ti pair if it is available. Heterogeneous TITAN RTX integration is a later placement/kernel experiment; native BF16 assumptions from Ampere cannot be copied to Turing. See [PyTorch DDP](https://docs.pytorch.org/docs/2.14/notes/ddp.html), [Ampere](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html), [Turing](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html).

**One 48 GB GPU:** short-context full training is plausible at the lower state budget but not guaranteed after activations and temporary copies. Measure the representative update first.

**CPU offload:** [DeepSpeed ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/) is the initial engineering route for preserving full-parameter scope with constrained VRAM. Implement it in a dedicated trainer adapter and test identical loss/update behavior before adding differentiable inner-memory updates. An inference device map is not a substitute.

**Optane:** inspect its configured mode and OS-visible capacity. Do not add installed persistent-memory and DRAM capacities as if they were equally fast allocatable RAM.

## 9. Mac Studio role

The M2 Ultra is useful for local inference, dataset review, teacher assistance with a suitable local model, independent evaluation, and a separate full-text tuning feasibility trial. Its unified memory is shared with the OS, applications, model buffers and inputs; 64 GB is not an isolated training allocation. [MLX unified memory documentation](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html).

Start the feasibility trial at microbatch 1 and 256-512 tokens, without a second teacher model resident. Record available memory, peak allocation, memory pressure, swap, throughput and save/reload behavior. Do not change OS memory safeguards to force a fit claim. Higher-order fast-state training and custom kernels need separate backend checks.

There is a current MLX scope trap: [MLX-LM lora.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/lora.py) freezes the model and, for its full mode, unfreezes selected transformer blocks. Choosing all blocks does not necessarily unfreeze embeddings, final norm and output head. Call that full_blocks until the parameter inventory proves full_text; a programmatic all-parameter trainer should unfreeze and audit the whole intended parameter tree.

Use nonquantized source/converted weights for genuine full-weight training. A 4-bit inference conversion is not automatically an equivalent full-tuning starting point. MLX-VLM documents V4.7 support, but that is not evidence that this project's custom modules and full multimodal gradients already work there. [MLX-VLM model documentation](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/minicpmv4_7/README.md).

The shipped PyTorch runner accepts MPS FP32 as an explicit feasibility path, but no MPS run was performed here. The recommended first numerical reference remains PyTorch; use the Mac for a separate backend comparison, not a hidden pooling of memory with the workstation.

## 10. The scale-up decision

Advance only when the mechanism earns its cost: D2 beats D0 on a relevant task/control, post-freeze T3 improves beyond T1, no critical retention/identity invariant fails, and the result survives restart and multiple seeds. If T3 matches T2, report useful continued adaptation without claiming better autonomous selection.

Carry source data, task generators, state contracts and evaluation protocols to larger models. Relearn readers, writers, workspace alignment and calibration per backbone. Reset incompatible fast state and caches. The larger backbone may solve tasks that were impossible for the small one, so remeasure the floor/ceiling and task difficulty.

If the small model's failure is capacity rather than a broken interface, that is useful evidence for scaling. If the small model's gain disappears against plain fine-tuning or stronger text retrieval, scaling has not yet been justified by the architecture.
