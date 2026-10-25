# Repository audit for the Spark-X2.6 next phase

Review dates: 24–25 October 2026. Scope: current working tree at HEAD `7c7018e`, selected implementation files, current experiment verdicts, architecture documents, public Spark source and targeted offline tests. This is a capability and integration audit, not a complete security or mathematical-correctness audit of every module.

## Evidence reconciliation

Use [execution_state.md](../../../execution_state.md), the [v5 writer experiment](../../experiments/20261024_Memory_Reliability_Writer_Binding_Results.md), and the [canonical eval02 verdict](../../../reports/memory_reliability_v5_scorer_fix_eval02/verdict_final.json) for current results. Final reconciliation includes the new 25 October [integrated design](../../architecture/19_Integrated_Memory_Learning_and_RSI_Design.md), [recovery plan](../../architecture/17_Memory_Reliability_Recovery.md), [Spark FP8 preparation](../../architecture/20_Spark_X2_6_FP8_Preparation_Plan.md) and [dataset curriculum](../../08_Dataset_Acquisition_and_Curriculum.md), which arrived during this research. Historical statements about missing imports, direct changed-span verification and original route counters are superseded by their repairs; narrower defects remain.

The [source-export report](../../../reports/reproducibility_v5_source_export_20261024.json) records repaired reproducibility and the corrected scorer. This review inspected that report and reran a focused working-tree subset. It did not independently repeat the complete native training, all 1,920 answer decodes, or the full clean-export suite.

| Current evidence | Interpretation |
|---|---|
| Writer-only D1 experiment: 48 updates, 192 cases, 14 trained tensors / 5,251,072 parameters | A bounded training path exists; changed parameters alone do not demonstrate learning |
| Fast memory and disabled memory both 56/192 | No aggregate benefit from the current learned memory |
| Initial fact 0/64; changed fact 0/16; unaffected fact 0/48 | Association storage, addressing and/or answer readout remain unresolved |
| Tokens differ on 128 factual routes | The branch influences computation; the failure is not simply an entirely disconnected branch |
| Selective exact context 184/192; full context 176/192 | Evidence can support the task, but total reference acceptance still fails; do not call either universally reliable |
| 32 authenticated memory images; seven-answer fresh-process parity | Serialization/restart works mechanically. “Images” here means memory snapshots, not visual inputs |
| Eight independent dev/held-out families combined | Useful diagnostic evidence, insufficient for broad generalization claims |
| General-question audit: 16 unique overall, eight in each split | Broader than the earlier four-question fixture, still inadequate as general language-retention coverage |
| Historical final clean-export suite: 1,311 tests + 21 subtests | Recorded software reproducibility; not a cognitive or multimodal benchmark |
| This review: 58 tests passed | Current tiny/offline core, port, fast-memory, learning-cycle and reliability contracts pass within the selected coverage |

Existing dirty files and untracked experiment directories belong to the ongoing work. This deliverable adds a separate report package and index links; it does not rewrite their protocols or canonical results. The untouched `src/replica_lab/runtime_pilot.py` remains a protected draft.

## Remaining integrity prerequisites from the latest review

The newer recovery document reports adversarial reproductions. This pass inspected the affected code and found the described structures; it did not rerun those separate reproductions. The 58 passing tests below do not cover away these gaps.

| Defect | Current code evidence | Required change before dependent native work |
|---|---|---|
| Caller/scope and canonical-route binding | `ExactMemoryReference.verify_context` passes `context.selection.scope` as its own expected scope, accepts only an optional independent query, and accepts empty non-relevant contexts without independently deriving status | Bind caller scope/query/policy/limit; verify canonical eligible evidence and disabled/conflict status at the consumer |
| Full training schedule audit | `validate_training_routes` checks case inventory/routes and per-update totals but does not require the complete unique ordered update sequence | Reconstruct expected update/case sequence from protocol and reject duplicates, omissions, reordering and boolean indices |
| Representation-preserving rebuild | `PersistentFastMemory.append_fact` uses a fact-binding encoder, while `rebuild` replays all records through raw-text `append` | Version writer representation and replay through its matching encoder; distinguish historical replay from rebuilding the current corrected view |

Sources: [exact context](../../../src/replica_lab/runtime/exact_memory.py), [route scorer](../../../src/replica_lab/reliability_scoring.py), [persistent state](../../../src/replica_lab/runtime/persistent_fast_memory.py). These are software prerequisites separate from the observed failure to learn useful associations. They do not show that the recorded native results were corrupted.

## What the implementation already provides

| Component and inspected source | Actual capability | Spark/vision consequence |
|---|---|---|
| [Backbone protocol](../../../src/replica_lab/architecture/backbone.py) | `encode_observations`, `sense`, `forward_with_frame`, `generate_action` operations | Reuse the conceptual boundary; introduce explicit multimodal batches and model-specific implementation |
| [MiniCPM loader](../../../src/replica_lab/architecture/loading.py) | Sealed offline MiniCPM revision, FP32 eager loading, inventory check, exclusive guarded MPS execution | It is deliberately MiniCPM-specific; a generic model-name substitution is insufficient |
| [Native adapter](../../../src/replica_lab/architecture/minicpm5.py) | Complete-block residual ports, immutable decisions, memory/workspace/fast/pattern branches, extension saving | Reuse contracts and tests; preserve Spark's distinct mask/RoPE/projection/cache behavior |
| [Port contracts](../../../src/replica_lab/architecture/contracts.py) | Default MiniCPM insertion 13/27, observer 41; width-256 banks; scope/producer binding | Spark has only 36 layers. Old layer indices are not transferable |
| [Memory and sense ports](../../../src/replica_lab/architecture/ports.py) | Independent residual readers; zero output initialization; masked known-token pooling; learned write projections | Useful neutral baseline. Pooled sense loses region, relation and time structure needed by vision |
| [Cognitive core](../../../src/replica_lab/architecture/core.py) | Tied recurrent attention workspace over sense/memory/task inputs; explicit previous state; matched feed-forward control | A real neural core exists. It needs typed visual/event banks and trained decision objectives |
| [Prediction heads](../../../src/replica_lab/architecture/heads.py) | Task-success and action-kind monitors with detached core inputs | Head-only supervision does not teach the core. Add an explicitly differentiable policy path and keep the independent monitor separate |
| [Fast-state service](../../../src/replica_lab/runtime/fast_memory.py) | Scoped state, staged writes, one commit, linear/SGDN-2 options and snapshot compatibility | Extend state schemas for nonlinear memory and momentum; do not globally remove runtime detaches |
| [Learning director](../../../src/replica_lab/runtime/director.py) | Deterministic bounded proposal selector over supplied verified outcome summaries | No learned practice-selection or self-improvement policy is established |
| [Cycle bridge](../../../src/replica_lab/runtime/learning_cycle.py) | Converts runtime counters into outcomes and requests director review | A proposal is not an optimizer update; visual failure diagnosis needs richer evidence than tool-call counters |
| [Identity interface](../../architecture/09_Replica_Identity_Interface.md) | Source/snapshot/mode/branch contracts; optional native workspace reader | Good place for the future personality file; no authentic conditioning or personal fidelity demonstrated |
| [Artifact and learning design](../../architecture/11_Runtime_Artifacts.md) | Versioned artifacts, source lineage, publication/rollback machinery plus bounded candidate experiments | Reuse this machinery for model bundles and improvement candidates; complete coordination still needs qualification |

The core defaults to width 256, eight slots and two tied iterations, with at most eight. Its analytic parameter count is **1,066,243**, excluding backbone access ports and prediction heads. It accepts batch-one source tensors, uses three source-type embeddings, and initializes each call from learned queries plus the previous workspace. There is no detach between its internal recurrent iterations. These are implementation facts, not evidence of metacognition, consciousness, learned visual tracking or recursive improvement.

The sense port averages valid native token features into a bounded summary. A picture of “red left of blue” and a picture of “blue left of red” must remain distinguishable by explicit features downstream; relying on a global average to preserve every binding is an avoidable bottleneck. Similarly, an eight-slot working space should coordinate access to visual banks, not be required to contain every patch or a complete video.

## Spark source facts and adapter risks

Source pin: `XHToken/Spark-X2.6-4B@66e3ae7bee54e0a523d692cb8ef841f35409807b`. File hashes are saved in the [source manifest](../../../reports/sparkx_2_6_4b_research_20261024/spark_source_manifest.json). The official configuration, implementation, template and weight index were read; weights were not acquired and custom code was not executed.

| Property | Verified value |
|---|---|
| Unique parameters / weight bytes | 4,112,079,360 / 8,224,158,720 in index metadata; 290 indexed tensors |
| Decoder | 36 layers; hidden 2,560; MLP intermediate 10,240 |
| Attention | 16 query heads, four KV heads, head dimension **256** |
| Projection | Fused `q_k_v_proj`; query width 4,096 and KV width 1,024 each; headwise sigmoid output gate |
| Attention schedule | Full attention at zero-based 3, 7, 11, 15, 19, 23, 27, 31, 35; other layers use a 512-token window |
| Rotary encoding | Full: partial factor 0.25, theta 5,000,000; sliding: factor 1, theta 10,000 |
| Feed-forward | GELU-gated MLP; preserve this exact behavior |
| Embeddings | Vocabulary 131,072; tied input/output weights; index omits duplicate `lm_head.weight` |
| Source dtype / context config | BF16 / maximum position 1,048,576; neither specifies local feasible training length |
| Dependency | Custom `spark2_5` code; config identifies Transformers 4.57.1, versus repository lock 5.20.2 |

[Pinned configuration](https://huggingface.co/XHToken/Spark-X2.6-4B/blob/66e3ae7bee54e0a523d692cb8ef841f35409807b/config.json), [implementation](https://huggingface.co/XHToken/Spark-X2.6-4B/blob/66e3ae7bee54e0a523d692cb8ef841f35409807b/modeling_spark.py), [weight index](https://huggingface.co/XHToken/Spark-X2.6-4B/blob/66e3ae7bee54e0a523d692cb8ef841f35409807b/model.safetensors.index.json).

Adapter review must specifically cover fused-QKV LoRA targeting, tied storage counted once, partial rotary dimensions, per-layer masks, left padding, cache positions, full/sliding cache ownership and floating-point casts. The source starts the residual stream in FP32 and casts into sublayers; loading BF16 weights does not make all activations BF16. The inspected attention path explicitly materializes attention scores. A sliding mask by itself does not establish efficient windowed attention.

Native parity is required before replacing eager attention with an optimized kernel. Test prefill and decode, padded/unpadded batches, positions around the 512-token boundary, and visuals before/after that boundary. A text prefix can become unavailable to later sliding layers; full-attention layers provide a route, but long-range visual utility still needs measurement. Do not copy MiniCPM's uncached loop and assume Spark's generation behavior remains identical.

### Official FP8 preparation added by the current repository plan

The existing plan selects `XHToken/Spark-X2.6-4B-FP8@0318d6f6ab070c7b5b43052db571489b2508006c` as an adaptation-base candidate and keeps the separately released BF16 model as a control. This pass independently fetched its [pinned config](https://huggingface.co/XHToken/Spark-X2.6-4B-FP8/blob/0318d6f6ab070c7b5b43052db571489b2508006c/config.json), matching the existing recorded hash. It uses `compressed-tensors` float quantization, static channel-wise 8-bit floating weights and dynamic per-token 8-bit floating input activations for selected projections; attention gate modules are ignored and KV quantization is null. See the [configuration review](../../../reports/sparkx_2_6_4b_research_20261024/spark_fp8_config_review.json).

Qualify original FP8 inference, input gradients for bridge/LoRA training, optimizer support and export semantics separately. A dequantized derivative of FP8 is not the independent original BF16 release and cannot recover lost precision. Keep an explicit identifier for each path. The BF16 memory arithmetic in this report is a reference/control calculation, not a decision to replace the repository's FP8 preparation. No weight formats were loaded by this review.

## Highest-value next diagnosis

Before paying for nonlinear memory or combined video training, separate three failures on new source families:

1. **Storage:** train a simple diagnostic decoder to recover observed keys/values from the final state, using source-family-disjoint training and evaluation. Success is only a recoverability diagnostic; report its supervision and capacity.
2. **Addressing:** give the reader a correct key but no value; compare learned query formation against this privileged-key diagnostic.
3. **Output binding:** if values are recoverable, compare explicit learned copy/pointer decoding against residual generation. The pointer's candidate set must come from permitted observations, not the answer key.

Use exact source access, raw observed features, uncompressed learned slots and fixed-size state as distinct information budgets. Include repeated values, swapped subject/value pairs, unseen values, temporal corrections and multiple queries after a single write. If raw features succeed but compressed state fails, changing the language backbone is an expensive first response. If state probes succeed but language answers fail, focus on readout training.

## Review limits and immediate engineering implications

This review does not establish that the current recurrent core outperforms a feed-forward controller, that the current memory updates generalize, or that native Spark fits either target device. The proposed integration should preserve the existing no-core, disabled-port, exact-memory and frozen-backbone baselines. Tests must distinguish a truthful saved/reloaded state from a useful state.

The most valuable reusable assets are the source/owner contracts, parity discipline, native complete-block ports, actual outcome recording and rejection/rollback machinery. The largest gaps are model-native multimodal representation, useful associative memory, differentiable core policy training, continual-learning evaluation across diverse tasks, and an actual learned improvement policy. The [architecture proposal](02_Integrated_Architecture.md) addresses these gaps directly.
