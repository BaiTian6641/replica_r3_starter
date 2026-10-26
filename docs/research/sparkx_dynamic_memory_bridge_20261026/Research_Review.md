# Dynamic memory attention and attachable neural modules: research review

26 October 2026. The [source manifest](../../../reports/sparkx_dynamic_memory_bridge_20261026/evidence.json) records versions, URLs, hashes, failed retrievals and the sections actually read. This is a selected-methods review of 15 primary papers, with static inspection of the official CALM implementation. Reported model results were not reproduced locally.

## 1. Conclusions supported by the evidence

**The proposed direction is feasible as an architecture research program.** Learned memory access, changing neural memory weights and composition through intermediate representations each have experimental precedents. Their joint integration into Spark, with audiovisual reasoning, vocabulary turnover and persistent learning, remains an untested system hypothesis.

The strongest design is a combination of complementary mechanisms: explicit records retain identities and corrections; fast weights compress associations; learned queries and routing choose what to use; trained bridges make different modules consumable by the backbone. None alone establishes continual acquisition of general skills. A common tensor shape establishes mechanical compatibility; useful interchangeability additionally requires a common learned representation and causal contract.

## 2. Memory and dynamic parameters

### R01 — MemT-1.0: unusually close architectural precedent

[MemT-1.0 v2](https://arxiv.org/html/2610.26825v2), §§2.2–2.3, distinguishes mutable local memory from trainable hyper-memory projections and gates. Its Mixture-of-Storage has separate storage and utilization projections sharing learned expert addresses. This directly supports distinguishing “where to write” from “where to read.”

Crucially, MoS evaluates **all** experts: it is dense routing. Both principal backbones in Appendix B use one expert per selected layer; the four-expert workload is separate. Frozen-backbone memory training does not demonstrate Spark full fine-tuning, and dynamic memory state does not mean its addressing projections are continually optimized during deployment. The agentic experiment also retrieves experiences with BGE-M3, so that experiment is not evidence for entirely native experience selection.

**Decision:** use separate read/write routing over shared addresses as a control. Treat sparse routing and online address adaptation as new hypotheses.

### R02 — HOPE / Nested Learning: timescales and adaptive projections

[Nested Learning / HOPE](https://arxiv.org/html/2512.24695v1), §§7.3–8.1, describes pretrained initialization of memory levels, different update frequencies and self-referential memory modules. Equations 79–82 include adaptive projection components. The later self-modifying formulation explicitly retains a fixed input query projection; it should not be summarized as making every final projection adaptive.

Meta-learned initialization and training across context flows matter. Merely attaching fast matrices, or assigning different update periods, does not reproduce HOPE. Its self-generated update targets also do not provide independent evidence that a remembered proposition is correct.

**Decision:** separate fast association, feedback-driven addressing and slower consolidation. Compare fixed and adaptive addressing explicitly instead of assuming the latter is beneficial.

### R03 — ATLAS: learn a local context, not only its last token

[ATLAS v1](https://arxiv.org/html/2505.23735v1), §§3.2–3.3 and 5, optimizes associative memory against a recent window, with weighted retention; the Omega rule is broader than a single-token delta update. Its richer variants introduce deeper memory, feature maps and Muon-style memory optimization.

The paper expressly distinguishes within-context memorization from skills surviving a cleared global context. Its parallel training uses chunk-based state approximations; recurrent and parallel implementations need parity tests. Additional optimizer iterations and retained key/value windows have real costs.

**Decision:** compare single-event updates against a short, bounded recent-event window. Muon is an optional inner-memory experiment, not a conclusion about the repository's full-backbone optimizer.

### R04 — MIRAS: specify the objective, not just the memory shape

[MIRAS v1](https://arxiv.org/html/2504.13173v1), §§3–4, separates four choices: memory architecture, associative objective, retention regularizer and update algorithm. Its “attentional bias” is an internal optimization objective, not simply an emotional label or an ordinary attention mask. The [ICLR 2026 paper](https://openreview.net/pdf/57a92b845c8787f52fb7d83c3f9a5ae8a0eac9b2.pdf) metadata was cross-checked; detailed reading used the accessible arXiv version after the final-PDF endpoint rejected download.

**Decision:** every proposed fast-memory module must state what it predicts, how targets are obtained, what preserves prior associations and which gradients or recurrence execute. “Self-learning memory” alone is an incomplete specification.

### R05 — Self-referential weight matrices: a concrete antecedent

[Irie et al., ICML 2022](https://proceedings.mlr.press/v162/irie22b.html), §3, gives a matrix that produces its own keys, queries and learning rates and changes through an outer-product delta rule. Its initial matrix is meta-trained. Section 4.2 tests adaptation to a domain switch with delayed labels; this is stronger grounding for learned update dynamics than an analogy with human cognition.

The evaluated adaptation is not evidence for indefinite autonomous improvement of an LLM. Implementation uses custom CUDA kernels, and the authors discuss efficiency and interpretability limits.

**Decision:** consider a self-referential module only after a simpler fast matrix works. Evaluate it against an equal-size fixed updater using unseen episodes and actual feedback.

### R06 — Memorizing Transformers: native query access to explicit history

[Memorizing Transformers](https://arxiv.org/abs/2203.08913v1), §3, retrieves stored keys/values using the model's own queries, then attends over the retrieved subset. Gradients do not traverse the historical memory. Its gate is a learned per-head scalar, not a token-dependent addressing controller.

The paper explicitly discusses stale keys when the producer changes during training. Normalizing queries and keys mitigates magnitude drift but does not eliminate representation drift. Document-level clearing also differs from a persistent personal experience store.

**Decision:** retain a transparent indexed-bank control, train query quality separately from hard selection, and version the producer of stored representations.

### R07 — Extended Mind Transformers: dense access can help, at a cost

[Extended Mind Transformers](https://arxiv.org/html/2406.02332v1), §2, uses native queries for per-token retrieval from cached model representations. It finds broad layer access useful in its tested models, with architecture-specific positional handling and pruning. This complements the previous density review's mixed results: there is no universal best layer count.

Section 3.3 amortizes document preprocessing across later queries. That is not zero-cost ingestion, fixed-size lifelong storage or learning of new durable parameters. Its unrotated retrieved keys for Llama2 are not automatically correct for Spark's hybrid attention.

**Decision:** expose all-layer access, but benchmark retrieval frequency separately from reader density; qualify any direct native-KV path independently.

### R08 — Product Key Memory: many parameters, few active values

[Large Memory Layers with Product Keys](https://papers.neurips.cc/paper/9061-large-memory-layers-with-product-keys.pdf), §§3.1–3.2, provides trainable sparse value lookup with exact top-k search under a product-key structure. Only selected value slots update for each input. This grounds the distinction between total capacity and active computation.

Its keys are constrained combinations of subkeys. The exact-search result does not apply to arbitrary audiovisual event embeddings. Nor does an offline-trained value table automatically become an appendable experience store.

**Decision:** use PKM-like learned capacity as a separate module candidate; use ordinary exact retrieval first for explicit episodes. Report actual indexing, gather and optimizer-state costs.

## 3. A unified access bridge

### R09 — CALM: compose capabilities through hidden states

[CALM](https://arxiv.org/html/2401.02412v1), §3, learns projections and cross-attention between selected layers of two frozen language models. Composition training teaches how to use their complementary capabilities. The authors leave thorough multi-augmenter evaluation for future work. Section 4.4 finds that using only the augmenter's initial prefix representations loses performance relative to updating it during decoding.

**Decision:** CALM is a strong precedent for a trained neural bridge, not a universal zero-training connector. A compact event module can be cheap; attaching another autoregressive model can require that model's repeated execution.

### R10 — LegoNN: interchangeability can be trained and tested

[LegoNN v2](https://arxiv.org/abs/2206.03318v2), §III, grounds interfaces in distributions over a predefined vocabulary, with an output-length controller and an ingestor. Its differentiable weighted-embedding and discrete alternatives expose the gradient-flow tradeoff. Section IV-B tests swaps across seeds, architectures and independently trained encoders, including reuse across ASR and translation.

**Decision:** adopt those swap tests. A grounded categorical side channel can help compatibility, but converting all audio and vision into text would discard information needed by this product. Continuous sensory/event payloads remain necessary.

### R11 — Perceiver IO: query a shared latent interface

[Perceiver IO](https://arxiv.org/abs/2107.14795v3), §3, separates input encoding, latent processing and output queries. Queries can carry modality, position and task information. With fixed latent size, input/output attention scales linearly in their lengths; latent processing still costs computation.

**Decision:** use a bounded latent bridge with structured query fields and modality-specific ingestion. This supports an interface design; it does not establish compatibility between independently trained arbitrary encoders.

### R12 — ImageBind: alignment requires an actual learning signal

[ImageBind](https://openaccess.thecvf.com/content/CVPR2023/papers/Girdhar_ImageBind_One_Embedding_Space_To_Bind_Them_All_CVPR_2023_paper.pdf), §§3–4, trains modality-specific projections with paired contrastive supervision anchored through images. It demonstrates useful cross-modal embedding relationships without observing every modality pair.

The inspected setup uses large pretrained image/text encoders and represents video with two frames. Global embedding alignment is not a temporal reasoning mechanism or a guarantee of event correspondence.

**Decision:** align semantic addresses, while retaining separate acoustic, spatial and temporal features in values and event packets.

### R13 — Relative representations: a useful compatibility control

[Relative Representations](https://arxiv.org/abs/2209.15430v2), §3, describes samples by similarities to corresponding anchors. The invariance argument assumes preservation of relevant angles; anchor choice and coverage affect expressivity. Its limitations discuss unresolved anchor-cost and multilayer-stitching issues.

**Decision:** compare anchor-relative outputs with a learned bridge when swapping encoders. Do not call it a solution for arbitrary latent spaces or treat anchors as free, permanently valid assets.

### R14 — Rosetta: modular expansion with joint training

[Rosetta v1](https://arxiv.org/html/2607.00293v1), §3, combines shared attention and a global shared expert with modality-specific routed experts. Momentum-based gradient projection targets interference. Its experiments upcycle Qwen3-0.6B into a 3.77B-total/approximately-0.97B-active model with substantial staged pretraining, including approximately 300B text tokens.

The abstract's non-destructive wording needs qualification: Table 1 reports MMLU 51.6→49.2 and ARC-c 67.3→62.9 after the generation stage, alongside gains elsewhere. A projection against optimizer momentum is not a general functional no-forgetting guarantee.

**Decision:** borrow the expansion and retention-comparison idea. Treat gradient protection as an ablation, not a replacement for replay, retention measurement or Spark full-fine-tuning qualification.

### R15 — UniR: a different interface, useful as a comparison

[Universal Reasoner v2](https://arxiv.org/html/2505.19075v2) combines a trained reasoning model's token logits with a frozen backbone's logits. Its evaluated pairs include within-family Qwen2.5 and Llama3.2 models.

**Decision:** distinguish output-level guidance from intermediate neural access. Logit addition needs corresponding token meanings; swappable vocabulary makes that correspondence an explicit requirement. It does not provide native region/event memory or a general tensor port, and its attached model adds inference work.

## 4. Paper/code cross-check: CALM is inspiration, not code to transplant

The official [repository](https://github.com/google-deepmind/calm) was inspected at commit `c369bea5e249b541c4fb16c5ef1d33ffd943817b`; no downloaded code was executed. Its README limits the implementation to Gemma-style models. The [forward path](https://github.com/google-deepmind/calm/blob/c369bea5e249b541c4fb16c5ef1d33ffd943817b/model/calm.py#L312) runs the augmenter and anchor with the same input IDs: it does not establish cross-tokenizer interoperability.

Two implementation details materially affect this plan:

- The [cross-attention hook](https://github.com/google-deepmind/calm/blob/c369bea5e249b541c4fb16c5ef1d33ffd943817b/model/layers.py#L106) stores/converts an augmentation mask but does not pass it into the attention call. If full teacher-forced sequences are supplied, later augmenter states are consequently accessible at earlier queries in that call. This is a static masking concern, not a reproduction or verdict on the original experiments. Spark needs explicit per-query availability masks and causal parity tests.
- Appendix C's illustrative overhead counts three attention matrices and multiplies an embedding contribution by layer count. The inspected implementation additionally has an output projection. Recompute tensor counts from the actual chosen design; do not reuse its illustrative “1.5%” as a Spark cost estimate.

The current repository's explicit native calls and owned decision frames are better starting points than importing global hook state.

## 5. What this evidence does not yet establish

No reviewed paper qualifies the complete combination on MiniCPM5 or Spark. Fast memory does not guarantee exact correction, a shared embedding does not prove temporal reasoning, and dynamic experts do not automatically reduce latency. A trainable interface can co-adapt until modules cease to be independently replaceable. Full backbone updates can invalidate old stored representations even when an interface's tensor dimensions remain unchanged.

The architecture therefore makes these failure modes observable. The [design](Unified_Bridge_and_Memory_Design.md) defines separate state classes and compatibility levels. The [validation plan](MiniCPM_Validation.md) requires memory-only, fast-parameter-only, combined and cleared-state comparisons before any self-learning claim.
