# SparkX 2.6: more backbone ports, including every layer

**26 October 2026 · Research proposal; no model code changed or training launched.**

**Yes: Spark can have native memory readers at every one of its 36 layers. Two ports were a conservative experimental baseline, not an architectural limit.** The most promising next comparison is two versus six versus nine readers, followed by an all-layer arm. Nine readers have a particularly natural placement: Spark's nine full-attention blocks. Keep a single experience bank and sparse, explicit write boundaries while testing read density.

This recommendation separates four choices: how many layers read memory, how often the bank is searched, how many independent memory states exist, and how often those states are updated. Increasing one does not require increasing all four. A model can consult memory repeatedly while writing each observation only once.

## 1. What the research establishes

### Dense access can help: Flamingo

[Flamingo, §3.3 Table 3 and Appendix B](https://arxiv.org/html/2204.14198v2) directly tests cross-attention frequency. Its small-model comparison reports:

| Gated cross-attention frequency | Overall score | Training step time | Total parameters |
|---|---:|---:|---:|
| Every layer | 70.7 | 1.74 s | 3.2B |
| Every second layer | 68.2 | 1.24 s | 2.6B |
| Every fourth layer | 68.8 | 1.02 s | 2.3B |
| One middle layer | 59.8 | 0.87 s | 2.0B |

Every-layer conditioning is strongest here, while every-fourth offers a useful cost tradeoff. The second/fourth ordering also warns against assuming monotonic improvement. These are **training times**, and parameter counts change with frequency. This is a frozen-language-model visual conditioning study, not a controlled Spark experiment or evidence of persistent learning. It supports including a dense arm, not dismissing it because it has more modules.

### MemT offers a close hybrid-backbone precedent

[MemT-1.0 v2, Appendix B](https://arxiv.org/html/2610.26825v2) places memory at **10 of 40 layers** in Nano: zero-indexed `3,7,…,39`, all full-attention layers. Flash uses 20 of 43 selected compressed-attention layers. Thus “memory-native” does not mean every decoder layer has a separate memory.

Nano has about 450.9M trainable memory parameters and 40 MiB of FP32 persistent matrix state per trajectory; Flash is substantially larger. These sizes are not comparable to our proposed tiny bank without qualification. The branch is integrated with attention, while generation reads an incoming snapshot and storage updates it after a completed step. The backbone is frozen during memory training. This motivates a Spark full-attention-layer candidate, but does not validate our full-FT recipe, audiovisual memory or state budget.

### Early sensory injection is a different interface

[DeepStack, §4.3](https://arxiv.org/html/2406.04334v1) adds spatially aligned visual features at several decoder depths without lengthening the language sequence. Its Phi-3 study favors early integration and reports its best tested stacking result with four layers. This is richer sensory conditioning at existing visual positions, not arbitrary persistent-memory retrieval.

[Qwen3-VL, §§2.2 and 5.12.2](https://arxiv.org/html/2511.21631v2) uses features from three vision-encoder levels in the first three language layers. The [2B configuration](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/89644892e4d85e24eaac8bacfd4f463576704203/config.json) selects **vision** blocks `[5,11,17]`; those are not language-layer insertion indices. The [Transformers implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py) adds them after language blocks 0, 1 and 2, only at visual positions. It does not run cross-attention to a memory bank at every text position.

**Our implication:** separately test three early visual-feature injections, and later an analogous audio variant. Do not count these additions as memory retrieval ports or infer that the audio variant is already demonstrated.

### More memory modules are not automatically better

| Evidence | Actual comparison and limitation |
|---|---|
| [MeMViT, CVPR 2022, §5.2/Table 1c](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_MeMViT_Memory-Augmented_Multiscale_Vision_Transformer_for_Efficient_Long-Term_Video_Recognition_CVPR_2022_paper.pdf) | Alternating memory augmentation across half the vision layers performs best in its temporal action-localization study; all layers are unnecessary. This is a video backbone, not an autoregressive language model or continual-learning result. |
| [Memory Layers at Scale, §5.4](https://proceedings.mlr.press/v267/berges25a.html) | Shared-memory placements improve from one to roughly three layers, then degrade in the tested six-layer case. Crucially, these layers **replace FFNs**, removing dense capacity. The result cannot prove that six additive Spark readers are worse than three. The final proceedings paper agrees with the inspected preprint on this point. |
| [Engram v2, §6.2](https://arxiv.org/html/2601.07372v2) | At fixed memory capacity, early insertion and a two-location split work well in its 12-layer MoE ablation. It studies static n-gram lookup and context-aware fusion, not our writable audiovisual experience bank. Early lexical memory and later task-dependent episodic reads may need different placements. |
| [video-SALMONN S v2, Appendix E](https://arxiv.org/html/2510.11129v2) | Applying fast-memory training to all four input/deepstack streams gives numerically lower Video-MME/LVBench scores than input-only, while multiplying additional overhead by at least four. This concerns repeated **fast-memory processing**, not inexpensive readers sharing one state. No statistical significance is inferred from the small reported score difference. |

[Titans, §§4 and 5.5](https://arxiv.org/html/2501.00663v1) distinguishes memory-as-context, gating and layer variants. Its “deep memory” experiments also vary the neural network **inside** a memory module. That is different from distributing readers through more backbone blocks.

[Nested Learning/HOPE, §§7.3 and 9.1](https://arxiv.org/html/2512.24695v1) describes initializing adaptive CMS components from pretrained MLPs, with different optimization levels/update frequencies and continued training. A larger number of read ports does not by itself create those learning levels. Keep spatial depth, inner-memory depth and update timescales as independent experimental variables.

## 2. What the repository currently allows

The inspected [PortSpec](../../../src/replica_lab/architecture/contracts.py) validates a tuple of distinct in-range injection layers; it is not intrinsically limited to two. [MiniCPM5Adapter](../../../src/replica_lab/architecture/minicpm5.py) constructs readers for that tuple and adds their residuals after native blocks. However, changing the tuple also expands sensing ports and, in some configurations, workspace readers and depth-specific fast-memory writers. A single configuration edit would therefore change several mechanisms at once.

The current [MemoryAccessPort](../../../src/replica_lab/architecture/ports.py) is masked cross-attention with independent Q/K/V/output projections, norms and a gate. Workspace readers use another independent instance. They are not the smaller fused reader sketched in revision 4. The reference generation path re-prefills without KV reuse; its latency cannot qualify a future cached Spark deployment.

The pinned [Spark source](https://huggingface.co/XHToken/Spark-X2.6-4B/blob/66e3ae7bee54e0a523d692cb8ef841f35409807b/modeling_spark.py) confirms pre-normalized attention followed by a gated GELU MLP, residuals, and type-specific positional/mask handling. Start additive ports **after the complete native block**, matching the current MiniCPM interface. Comparing a MemT-like pre-output-projection branch is a separate placement experiment; it changes fusion geometry and must preserve Spark's attention gates and head dimensions.

Consequently, the next implementation needs separate declarations for `read_layers`, `sense_layers`, `retrieval_layers`, `write_sources` and `online_adapter_targets`. Do not expand all of them when increasing reader count. No such runtime refactor was performed in this research pass.

## 3. Recommended architecture candidates

### Three complementary paths

1. **Early sensory detail:** a small number of feature additions at existing visual positions, with matched crop/feature budgets. Audio additions require separate alignment and timing tests.
2. **Distributed memory readers:** native gated residual modules consulting the same bounded sensory/semantic records and workspace, at selected or all language layers.
3. **Sparse writes and learning:** one causal commit per observation boundary; optional depth-specific proposals are aggregated before that commit. Fast-state learning and durable adapters keep separately declared schedules and quotas.

Early, middle and late readers could specialize in grounding, temporal relations and answer construction. That is a hypothesis to test using interventions; it is not an assumed universal division of labor between transformer layers. A late reader can re-access a visual/audio pattern while an early reader helps the backbone build the initial interpretation.

### Concrete layer sets

All indices below are zero-based and identify post-block injection. MiniCPM's sets approximate relative depth and include its existing baseline; it does not inherit Spark's sliding/full-attention interpretation.

| Arm | Spark, 36 layers | MiniCPM, 42 layers | Role |
|---|---|---|---|
| Historical two | `[11,23]` | `[13,27]` | Preserve existing comparison |
| Early two | `[3,23]` | `[4,27]` | Isolate moving one reader earlier |
| Six | `[3,11,15,23,27,35]` | `[4,13,18,27,32,41]` | Two readers in each broad depth band |
| Nine | `[3,7,11,15,19,23,27,31,35]` | `[4,8,13,18,22,27,32,36,41]` | Spark's full-attention blocks; leading denser candidate |
| Alternating | `[1,3,…,35]` (18) | `[1,3,…,41]` (21) | Later diagnostic if nine is promising |
| All | `[0,…,35]` (36) | `[0,…,41]` (42) | Direct per-layer hypothesis test |

The two-port model remains a control. Six and nine are the first denser candidates; all-layer access is a legitimate study arm, not prohibited or presumed optimal. Sensory-injection placement is held fixed during this comparison.

### Shared bank, layer-specific interpretation

Use one persistent bank, one canonical set of identities and at most 16 selected records per query. Additional layers can read that same set through different queries/gates. Keep fixed limits on full-bank searches; increasing read count must not multiply retrieval to `16 × number_of_layers` records. The bank's 7 MiB tensor estimate remains independent of reader count when it is genuinely shared.

First compare independent readers. Then test three depth-group shared projection sets with layer-specific normalization and gates, or a rank-64 factorization of the output projection. Sharing weights saves parameter/optimizer storage but does not eliminate repeated matrix multiplications. A factored projection reduces arithmetic but restricts the residual's representational rank; quality must be revalidated.

Cache projected keys/values for the selected records within a generation when their producers and reader weights are unchanged; invalidate them after relevant updates. Use a zero-effect initialization with a live gradient path, not two zero factors in a factored output. Monitor residual-to-hidden norms and gate saturation across depth to detect repeated over-conditioning or inactive branches.

Soft gates near zero do not save runtime if the branch still executes. Conditional execution requires actual branch skipping and a measured routing/launch cost. Static placement is the first control; learned routing comes later with training that matches its deployment decisions.

### Causal retrieval is a real design constraint

A reader at layer 0 cannot use a query-dependent selection that will only be computed at layer 23 of the same forward pass. Likewise, selecting records from the final query token and broadcasting them into earlier prompt positions can violate causal training assumptions. More ports expose this issue more sharply than the original late-port design.

For the **first controlled MiniCPM density study**, use the existing separate, known-query sensing pass with fixed sensing layers `[13,27,41]`, regardless of reader count. Two fixed query summaries propose up to eight records each; union/cap at 16 and freeze the selection for the following action pass. This is a proposed native selector, never gold-label selection. Count the entire sensing/selection pass in every arm. Its schedule is an explicit study alternative to revision 4's no-extra-planning-pass candidate, not a claim that the extra pass is free or already implemented.

A deployment optimization can instead use a previously committed selection throughout the current pass, or update availability only for later layers at explicit retrieval anchors. No retroactive access to deeper-layer results is allowed. Current-query early access without an extra pass needs a separately trained early selector. Cached prefill/decode behavior, teacher-forced training and recomputation must agree before accepting any cheaper schedule.

Never perform the same global memory decay/write once per reader. Repeating a decay factor `lambda` at P readers changes it to `lambda^P` per observation. That silently changes the time scale and may overwrite evidence. Independent per-layer fast states are possible, but constitute a separate state-capacity/update experiment and require individual ownership, reset and versioning.

## 4. Analytical cost: feasible in size, latency still unqualified

For revision 4's **proposed fused reader**, the query/output matrices contain `D×128 + 512×D` parameters per port. At Spark width 2560 this is 1,638,400. The following excludes gates, norms, other read/fusion projections, writers, encoders, activations and optimizer workspaces:

| Spark read ports | Query/output parameters | BF16 weights | FP32 weight+gradient+Adam moments |
|---:|---:|---:|---:|
| 2 | 3.277M | 6.25 MiB | 50 MiB |
| 6 | 9.830M | 18.75 MiB | 150 MiB |
| 9 | 14.746M | 28.125 MiB | 225 MiB |
| 18 | 29.491M | 56.25 MiB | 450 MiB |
| 36 | 58.982M | 112.5 MiB | 900 MiB |

The 36-port projection count is about 1.43% of Spark's original parameter count. This makes the experiment credible; it does **not** establish a 1.43% latency overhead. Per-layer small kernels, repeated norms/attention, memory traffic and device synchronization can dominate nominal arithmetic.

Factoring only the output as `512→64→2560`, retaining a full width-128 query, reduces these 36-port matrices to **18.874M parameters / 36 MiB BF16**. Three shared full-rank projection sets require **4.915M / 9.375 MiB**, but still execute at all active layers. These are different capacity controls, not guaranteed equivalent models.

For the **actual current memory-plus-workspace reader pair**, the exact count at width D and memory width m is `2 × (2Dm + 2m² + 2D + 2m + 1)`. With D=2560, m=256, 36 pairs require **104,214,600 parameters / about 198.77 MiB BF16**, before sensing/writer/fast branches. Do not use the smaller proposed formula to describe an unchanged implementation.

One BF16 `[1,8192,2560]` activation is 40 MiB. Retaining one such tensor for every port would be 1440 MiB at 36 ports; this is an illustration of activation sensitivity, not an autograd-memory prediction. Profile actual saved tensors and checkpointing. Full Spark FT already has a substantial optimizer-state floor.

State sharing matters more than the reader count alone: 36 separate 7 MiB banks would use **252 MiB**, violating the 32 MiB resident-state cap before metadata. One shared bank remains 7 MiB. The proposed visual/audio fast-matrix pair is 0.25 MiB if shared, versus 9 MiB for 36 independent pairs, with proportionally different write work. All buffers and transient copies still count in live device limits.

## 5. Validation and full-fine-tuning plan

Start only after existing vision, memory and restart prerequisites admit the experiment. Preserve failed historical results. The study changes readers first, not storage correctness or labels.

1. **Reference contracts:** independent layer masks, immutable state during reads, exact once-only writes, zero-effect initialization, checkpoint/restart parity, and all-layer disabled equivalence. Separate sensing/selection from read placement.
2. **Fixed mechanism:** hold bank contents, writer, native selection, selected-record cap, sensory input, core recurrence and two online-adapter targets fixed. Compare historical two, early two, six and nine readers with the same data and update budget. Then admit all-layer and alternating arms if resources permit. Include no-memory and matched-extra-parameter controls.
3. **Capacity and cost controls:** compare independent full-rank, rank-64-output and group-shared readers. Fixed-width comparisons answer whether added resources help; matched-parameter and matched-wall-time comparisons answer whether depth uses those resources better. Weight tying alone is not a compute-matched control.
4. **Causal utility:** test delayed facts/corrections, unknown queries, spatial relations, object continuity, temporal reversal, acoustic pattern transfer and conflicting modalities. Ablate early/middle/late reader groups; replace relevant versus irrelevant memory. Gate magnitude alone cannot establish utility.
5. **Training:** warm new branches, then repeat the controlled comparison with full-backbone MiniCPM co-adaptation and text/old-modality retention. After transfer, Spark still needs full original-backbone FT. Frozen-backbone paper results cannot substitute for that experiment.
6. **Learning and long-run gates:** repeat the memory × vocabulary × learning factorial only for the selected reader design. More readers do not expand the online LoRA target set or learning step cap automatically. Preserve durable-learning reset tests, correction, representation migration and soak requirements.

Use three seeds for confirmation and paired world/video/speaker evaluation with corrected intervals. Select the **least expensive non-dominated** design on the quality/latency/memory frontier. Do not force a five-point gain for every extra density step: the existing five-point capability gate applies to the capability comparison against its matched lawful control. To replace a qualified cheaper reader design, require an independently confirmed improvement on a preregistered difficult-task slice, acceptable retention, and a resource profile within the product limits.

Measure capture/encoding, selection pass, prefill, decode P95, memory updates, checkpointing and training separately. Keep the ≤10% warm text-overhead target, ≤5% learning duty-cycle target and existing device/state caps. A later cache optimization needs its own parity and end-to-end measurements; re-prefill timings are not cached-generation timings.

## 6. Plan decision

**Expand the architecture search; do not freeze the product at two ports.** Prioritize six and nine shared-bank readers, with nine aligned to Spark's full-attention blocks. Include a compact all-layer reader candidate. Keep raw sensory injection, memory search, persistent writes and durable optimization as distinct native mechanisms with independent budgets.

This provides a credible route to deeper audiovisual reasoning and repeated use of learned experience without requiring 36 independent memory systems. Whether the extra access produces useful reasoning or self-learning remains a MiniCPM-first empirical question.

Artifacts: [study config](../../../configs/sparkx_2_6_4b_port_density_study_v1.json), [cost ledger](../../../reports/sparkx_port_density_20261026/analytical_costs.json), [source manifest](../../../reports/sparkx_port_density_20261026/evidence.json), [repository observation](../../../reports/sparkx_port_density_20261026/repository_snapshot.json), and [preserved revision 4](../../../reports/sparkx_port_density_20261026/architecture_v4_snapshot.txt).
