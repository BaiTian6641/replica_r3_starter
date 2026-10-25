# Research and repository findings

The conclusion is conditional but positive: there is enough evidence to justify a small native-memory research program. There is no evidence that attaching all the named mechanisms at once will produce a reliable evolving model. Their costs, learning signals and state lifetimes differ substantially.

## What the MiniCPM work establishes

This review inspected current code, the reliability repairs and preserved outcomes. It ran existing focused CPU/tiny tests, not native training or a full integration suite. Final reconciliation is against commit `757ace1565c92f23bc418cc83286411832790e3f`, which landed during this review. Current executable hashes are recorded in the [review manifest](../../../reports/dynamic_engram_research_20261025/review.json). Earlier reports remain historical evidence.

| Evidence | Observed result | Consequence for this design |
|---|---|---|
| [M2 shared-value memory](../../experiments/20261024_M2_Dynamic_Memory_Results.md) | 4/4 development and 4/4 held-out bindings with six reused values; parent retention 2/2 became 0/2 | A usable adaptation pathway exists on a narrow family. Novel-value generalization and retention need separate supervision |
| M2 unique-value arm, same record | 8/8 training, 0/4 development and 0/4 held-out | Low training loss can reflect memorized answer vocabulary rather than learning to store new values |
| [M3 depth ownership](../../experiments/20261024_M3_Depth_Ownership_Results.md) | Shared/private states both 8/8 across the paired small splits; later port alone sufficient; retention regressed | Existing state ownership and gradient interfaces are useful. Shared-state superiority and a need for more ports were not demonstrated |
| [M4 static optimization](../../experiments/20261024_M4_Pattern_Results.md) | Direct pattern 0/4 development, 0/8 held-out, 0/8 regression; all 20 delayed router cases used fast memory | Existing routing results provide no attribution to static Engram. Increasing its table is not a justified remedy |
| M4 collision audit, same record | No distinct-ngram collisions in the audited prompts; complete query-hash coverage absent | Poor quality cannot simply be blamed on collisions. Whole-query hash coverage is a poor relevance criterion |
| [Latest v5 writer-binding result](../../experiments/20261024_Memory_Reliability_Writer_Binding_Results.md) | Fast and disabled both 56/192; initial facts 0/64, changed facts 0/16, unaffected facts 0/48 | Influence on tokens is not useful memory. Diagnose encoding, addressing and decoding independently |
| v5 exact-context control | Selective context 184/192, full context 176/192; complete reference gate still rejected | Exact evidence is a valuable diagnostic, while reference non-regression remains a distinct requirement |
| v5 restart | Seven recorded answers reload with state/token parity | Persistence mechanics work for those cases, including wrong answers; this is not a capability result |

The previous Spark audit described three remaining integrity defects. The new commit repairs caller-supplied scope/query/policy/limit with canonical context recomputation, an explicitly reconstructed ordered training schedule, and representation-aware observation reconstruction. The **99 passing focused tests** include the affected exact-memory, reliability and new persistent-representation suites. The later [repair qualification report](../../../reports/memory_contract_repairs_20261025/qualification.json) records a clean source export, **1,372 tests plus 21 subtests**, and 140 integrated repair tests. Those larger results were inspected, not rerun here. This advances software/source admission; it does not qualify native memory or vision. Some older execution-document paragraphs still describe the completed repairs as planned.

The representation repair binds tokenizer, parser, normalization, writer and engine identity. It distinguishes raw observations from fact-binding representations, ordered-history reconstruction from a caller-selected current view, and explicit legacy migration from silently reinterpreting old tensors. Its [fresh-process smoke report](../../../reports/memory_contract_repairs_20261025/smoke.json) confirms deterministic reload/rebuild behavior using a tiny decoder fixture and explicitly sets `native_execution: false`. This is the correct foundation for a future dynamic bank, but it does not change the rejected v5 quality result.

After this review's focused test run, concurrent V0 work added draft `VisionSpec` / `VisualInput` contracts for pinned spatial patches, crop geometry and ownership. That diff was inspected as work in progress; it is outside the 99-test snapshot and is not a measured visual capability. The implementation roadmap should extend those contracts when accepted rather than introduce a competing visual-state format.

The semi-cognitive core is already a tied recurrent workspace with typed sense/memory/task inputs. Its current contract is batch-one, width 256 with eight slots in the reference configuration. Existing monitoring/policy heads detach workspace features. Consequently, head loss alone does not teach the core to acquire better information. `SensePort` pooling also discards spatial distinctions needed by the proposed visual system. The learning director is a deterministic baseline, not a trained learning policy. [Core](../../../src/replica_lab/architecture/core.py), [heads](../../../src/replica_lab/architecture/heads.py), [ports](../../../src/replica_lab/architecture/ports.py), [director](../../../src/replica_lab/runtime/director.py).

The static `PatternAccessPort` implements causal hashed 2/3-gram tables and a contextual residual gate. It does not reproduce every component of the published Engram architecture. Sparse embedding gradients are densified in the current optimizer path; sparse reads therefore do not imply sparse training memory. The current SGDN-2 primitive uses a Python token loop. Its arithmetic is small, but launch and interpreter overhead can dominate. These are concrete reasons to measure end-to-end costs. [Pattern and memory ports](../../../src/replica_lab/architecture/ports.py), [SGDN-2 implementation](../../../src/replica_lab/neural.py).

## Primary-source reading notes

Public full texts were retrieved and the relevant methods, evaluation sections and appendices inspected. The [manifest](../../../reports/dynamic_engram_research_20261025/review.json) records versions, URLs, byte hashes and reading scope. Full copyrighted texts stay in temporary research storage; this package contains concise summaries and our original design. Paper results below are author-reported, not reproduced here.

### MemT-1.0

[Paper v2](https://arxiv.org/html/2610.26825v2), sections 2–4, 6–8 and appendices B/C/E/F. It adds memory attention at selected backbone layers, with fixed learned memory-processing weights and evolving per-trajectory matrices. SGDN-2 supplies a symmetric erase/write recurrence. MoS uses **dense** expert participation; the main evaluated configurations use one expert per layer. Nano has roughly 36B total/3B active parameters and 40 MiB of FP32 memory state. Training uses 34B tokens, segment language modeling and full-context-teacher distillation. Experience-to-permanent-weight consolidation remains future work. Its agent study also uses retrieved procedural experience. Thus it supports native state adaptation, not a demonstrated autonomous lifelong weight-improvement loop or a direct fit for our hardware.

The [pinned official inference package](https://github.com/MemTensor/MemT-1.0/tree/e75c13d090a6a9d34b29146d901bd84323185f24/inference) was also inspected. It separates fixed-snapshot reads from commits, uses CUDA/Triton implementations and declares PolyForm Noncommercial licensing for its original package code. It is an implementation reference, not a selected MPS dependency.

### Engram

[Paper v2](https://arxiv.org/html/2601.07372v2), architecture, scaling and efficiency sections. Engram hashes normalized token n-grams into trained static embeddings, then applies context-dependent gating and a causal convolution. Its principal benefit is extra sparsely accessed knowledge capacity. It is not an online episodic writer. The main models were pretrained on 262B tokens. The reported low host-offload overhead used an H800 and 512 sequences; it does not establish single-user Mac latency. The [official demonstration](https://github.com/deepseek-ai/Engram/blob/fb7f84a21f91223715394a33a1dc24bbfb7f788e/engram_demo_v1.py) confirms tokenizer compression, hashing and convolution machinery. We should retain the inexpensive pattern pathway as a comparator and train direct utility before enlarging it.

### Metis

[Paper](https://arxiv.org/html/2607.26760v1), native-memory formulation and implementation. Metis distinguishes a dynamic local state from trainable hyper-memory procedures. Inference updates state with a forward pass while learned weights remain frozen. This is an especially useful distinction for defining our API and avoiding confusion between remembering an event and learning a generally better update rule. Its particular storage procedure and backbone assumptions need independent qualification before reuse.

### Memory Layers at Scale

[Paper](https://arxiv.org/html/2412.09764v1), section 3 and scaling limitations. Trainable product-key lookup provides large sparse parametric capacity, with shared memory across layers and a gated Memory+ variant. Search, embedding aggregation, optimizer storage and bandwidth still require engineering. These are learned global parameters rather than per-session episodic records. Product-key lookup is an alternative if our dynamic bank outgrows a small dense search; it is not needed for the first 4,096-slot pilot.

### Titans

[Paper](https://arxiv.org/html/2501.00663v1), sections 3–4. Titans learns an associative memory with online gradients, momentum and forgetting, and explores several ways to combine it with attention. The inner objective updates the memory; the outer objective teaches representations that make later reads useful. The relevant surprise signal is a memory-loss gradient, not simply uncertainty or a mood score. We should test a small nonlinear memory against the existing linear recurrence at matched state and update cost, after the binding problem is diagnosed.

### HOPE / Nested Learning

[Paper](https://arxiv.org/html/2512.24695v1), CMS, self-modifying modules and continual-learning experiments. HOPE combines multiple optimization levels and context flows. CMS defines actual parameter updates at different frequencies, with a task objective and knowledge transfer between levels. Its Llama retrofit experiments include 15B continued-pretraining tokens. A periodic adapter-training job alone is not faithful HOPE. Our initial proposal is explicitly HOPE-inspired; a faithful branch must implement and ablate the update laws, learned initializations and cross-level credit assignment.

### Universal Test-Time Training

[Paper](https://arxiv.org/html/2610.05484v1), shared weights, routing and update schedule. uTTT lets multiple depths access a shared fast-weight pool. Its aggregated schedule evaluates writes against the same chunk-start state, then applies one update. Routed and private controls distinguish sharing from simply increasing active state. Current M3 averages SGDN-2 state proposals instead; it is not this optimizer. Shared state is promising for visual continuity, but our experiment must match total state, active compute and update semantics.

### Doc-to-LoRA

[Paper](https://arxiv.org/html/2602.15902v1), sections 3/5 and appendix C. A meta-trained hypernetwork converts context activations to low-rank adapters. This can eliminate per-document gradient optimization after substantial training. The main generator has 309M parameters; concatenating chunk adapters grows total rank. Its unrelated-context tests show interference. Visual transfer is a classification demonstration, not evidence of spatial or temporal reasoning. We propose a much smaller, explicitly experimental compiler and a null route. The [official code](https://github.com/SakanaAI/doc-to-lora/tree/baa85db4d5df9b29d618af432d6ebf28b3ad5a29) was inspected for adapter generation and application, not executed.

### Doc-to-Atom

[Paper](https://arxiv.org/html/2606.12400v1), section 3. It compiles typed atoms into micro-adapters and retrieves relevant atoms for a query. Irrelevant-query suppression is directly relevant to the MiniCPM failure. Its pipeline uses LLM atomization and optionally a separate reranker, so it does not by itself satisfy our fully integrated deployment requirement. We would train a native writer/router and retain bounded latent atoms first. Its factor-composition choice is discussed mathematically in the architecture report; we do not assume arbitrary adapter averaging preserves individual memories.

### Text-to-LoRA

[Paper](https://arxiv.org/html/2506.06105v1), adaptation formulation. A hypernetwork maps task descriptions to adapters after learning across task/adaptation examples. This supports an instructed-adaptation branch. A description can select or compose capabilities learned during meta-training; it is not evidence that an arbitrary unseen skill appears without demonstrations, feedback or a suitable learning objective.

### SEAL

[Paper v2](https://arxiv.org/html/2506.10943v2), section 3. SEAL learns to propose self-edits/training material, evaluates their effect after adaptation, and reinforces useful proposals. The nested training and evaluation cost matters. We can distill successful choices into a compact learning-policy head and restrict initial updates to small modules. An edit proposal, self-critique or synthetic answer is not independent evidence of improvement.

### Backpropamine

[Paper](https://arxiv.org/html/2002.10585v1), section 3. It trains fixed weights and plasticity coefficients while network outputs modulate episodic Hebbian traces; eligibility traces support delayed modulation. This provides a concrete antecedent for a native controller influencing what is learned. Its supervised/RL and language-model experiments do not establish useful emotional simulation in Spark. Borrow the testable plasticity mechanism rather than assigning biological names to arbitrary scalars.

### Cognitive appraisal and reinforcement learning

[Paper v2](https://arxiv.org/html/2309.06367v2), model and experiment sections. It connects appraisal variables such as goal relevance, conduciveness, suddenness and control to learning and value estimates. Evaluation concerns agreement with human judgments of vignettes. This supports interpretable simulation variables, not a demonstrated gain in LLM learning or a claim of subjective feelings. Our appraisal module must earn its place through decisions, calibration and retention at equal cost.

### AdaMLearn / biological engram-inspired lifelong learning

[Paper](https://arxiv.org/html/2610.14940v1), methods and evaluation setup. This separate use of “engram” builds and prunes gradient subspaces to protect earlier learning. Experiments are task-incremental image classification, including CIFAR and Mini-ImageNet. It is a candidate consolidation regularizer, not DeepSeek Engram or an LLM episodic memory system. Test a low-rank gradient-subspace control only on small trainable modules, with its decomposition cost included.

## Combination potential and limits

| Combination | Assessment | Experiment that could justify it |
|---|---|---|
| Episodic slots + small fast memory | Strong engineering hypothesis | Novel bindings and corrections improve over each isolated path at fixed state bytes |
| Static Engram + dynamic memory | Complementary responsibilities, unproven locally | Direct static utility on covered unseen patterns, dynamic utility on recent changing facts, and joint non-regression |
| Core + visual/event memory | Necessary representation upgrade for the requested goals | Cross-view identity, delayed spatial relation and temporal-order tasks beat pooled-caption controls |
| Native hyper-adapters + verified replay | Promising but greater training risk | Skill transfer survives memory reset and unrelated-task checks |
| Appraisal + plasticity | Plausible experimental controller | Better acquisition or inspection decisions than equally sized neutral controls |
| HOPE + all of the above | Research branch with substantial optimization complexity | Multi-rate learning beats single-rate learning under identical data and total compute |
| Open-ended recursive improvement | Not established by this evidence | Improved future learning efficiency across unseen task families and several independently assessed cycles |

The central uncertainty is **whether the model learns useful representations and update rules**, not whether more memory tensors can be attached. The plan therefore treats memory capacity, inference computation, update computation and general capability as four separate variables.
