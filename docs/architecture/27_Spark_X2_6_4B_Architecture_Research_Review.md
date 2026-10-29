# Spark-X2.6-4B: architecture investigation and document cross-check

**28 October 2026 · America/Chicago · Research and review only.**

Spark-X2.6-4B is a **4,112,079,360-parameter dense text decoder**. Its published architecture is unchanged from Spark-X2.5-4B. The project's vision/audio encoders, intrinsic experience stores, plastic experts, vocabulary overlay and neural spine are additional research components. Their presence in the product blueprint does not make them capabilities of the released Spark checkpoint.

The important near-term issue is the execution path: the active text-memory protocol requests official native FP8 on MPS, but the current adapter explicitly rejects native FP8 construction. The verified local runs are the separate official BF16 and dequantized-FP8 controls. Source research resolves the topology; it does not resolve that backend gap.

This review combines fresh public-source retrieval, byte comparisons, local tensor-header inspection, the existing run receipts, and small CPU checks of cache and template behavior. The [evidence package](../../reports/spark_architecture_review_20261028/README.md) contains 62 public text/metadata snapshots including license/attribution files, their hashes, reproducible analysis, and the exact workspace input hashes. No model weights were downloaded or loaded, and no native inference, training, held-out evaluation or promotion was performed.

## 1. Findings that affect decisions

| Finding | Evidence and implication |
|---|---|
| X2.6 really retains X2.5 topology | Freshly retrieved `config.json`, `configuration_spark.py` and `modeling_spark.py` are byte-identical across the inspected X2.5/X2.6 revisions. Weight identities and chat templates differ. Reuse architecture understanding, but preserve checkpoint and prompt identity. |
| Most native geometry in Doc23 is correct | 36 layers, 2560 residual width, 10240 MLP width, 16 query heads, 4 KV heads, 256 head dimension, GELU gating, headwise sigmoid attention gates and tied embeddings all agree with code and headers. |
| The active FP8 path is not implemented by the current adapter | `Spark2_5Adapter.__init__` rejects `official_native_fp8` even after general backend admission. MPS also has a recorded float8 allocation failure. A different device string or loader flag is insufficient. |
| Windowed attention does not guarantee cheap execution | The reference forward explicitly forms dense attention scores. The 512-token mask restricts information flow; it does not supply a sparse prefill kernel. |
| Visible cache length can understate physical allocation | In Transformers 4.57.1, the sliding cache retains a slice of the full prefill tensor. The CPU probe observes 511 visible tokens backed by 2048 tokens of storage until the next decode update. |
| Local Spark smoke evidence exists | All eight referenced raw receipt hashes for `bf16_06` and `derivative_02` match. Both recorded one extension update and exact fresh-process reload; neither trained the original backbone or demonstrated useful memory learning. |
| Document authority has drifted | Older “current” MiniCPM-first and no-adapter statements conflict with Doc26 and later receipts. Treat them as dated history. Doc26 owns the active text-memory direction; Doc23/24 describe the broader product candidate. |
| The 1M claim is a capacity claim with substantial limits | The configuration permits 1,048,576 positions. The vendor's own results weaken at the longest tested ranges, and the local eager implementation cannot be budgeted from KV payload alone. |

These findings do not require another architecture expansion. They call for a precise backend contract, native parity/cost qualification, and evidence that the smallest text-memory path changes generated answers usefully.

## 2. Source identity and what changed

The official repository dates the 4B launch to **21 October 2026** and attributes X2.6 improvements to post-training. The retained `Spark2_5ForCausalLM` class name is intentional, not evidence that the wrong model was loaded. A separately published `Spark-X2.6-4B-Base` also exists; “base for adaptation” in the project means the selected post-trained checkpoint, not that Base variant. [Official release](https://github.com/XHToken/Spark-X2.6/blob/27ef075943cd9af43c1d1138e2ee267f1a4f872c/README.md).

| Source | Workspace pin / inspected public revision |
|---|---|
| Official X2.6 BF16, workspace pin | `66e3ae7bee54e0a523d692cb8ef841f35409807b` |
| Official X2.6 BF16, public head observed in this review | `d1b5f26071101c27fcd07728e7a7770e428281f4` |
| Official X2.6 FP8, workspace pin | `0318d6f6ab070c7b5b43052db571489b2508006c` |
| Official X2.6 FP8, public head observed in this review | `15b16e794bf251505f68501294fce730cceaa4e9` |
| Official X2.5-4B comparison | `0bcb35678590218655dff3765b9e61c83b35e9c4` |
| Official X2.6-4B-Base comparison | `88439533d38a0e6a81125c72ec9fad50d6f98692` |

The fresh pinned BF16 and FP8 configuration, model implementation, configuration class and chat template all match the respective local files byte-for-byte: **8/8 comparisons pass**. Within each X2.6 precision variant, the newer public revision preserves those four files and the published weight LFS identities. The inspected BF16 README change updates its citation title to include 1.7B. There is no architecture-driven reason to repin the existing source. [Comparison receipt](../../reports/spark_architecture_review_20261028/architecture_audit.json).

X2.5 and X2.6 share the three architecture files but have different weight identities. X2.6 adds reasoning-effort instructions and changes tool serialization whitespace in its template. The Base and post-trained X2.6 variants likewise share topology while carrying different weights. Weight equality here is assessed from Hub LFS metadata; the existing multi-gigabyte local payloads were **not rehashed** by this review.

## 3. Reconstructed native architecture

### 3.1 Topology

| Property | Verified value |
|---|---|
| Model family / type | Dense decoder-only causal language model; `spark2_5` |
| Decoder layers | 36 |
| Residual / MLP widths | 2560 / 10240 |
| Attention geometry | 16 Q heads; 4 KV heads; 256 channels/head |
| Fused QKV projection | 2560 → 6144; split 4096 Q + 1024 K + 1024 V |
| Attention output projection | 4096 → 2560 |
| Attention gate | 2560 → 16; one sigmoid scalar per query head/token |
| Local/global schedule | `[sliding, sliding, sliding, full] × 9` |
| Full-attention sites, zero-based | 3, 7, 11, 15, 19, 23, 27, 31, 35 |
| Local window | 512 |
| Position encoding | Local: 256 rotary dimensions, theta 10,000; full: 64 rotary dimensions, theta 5,000,000 |
| MLP | Three bias-free matrices; GELU-gated multiplicative branch |
| Normalization | Two pre-RMSNorms per block, final RMSNorm; epsilon 1e-6 |
| Vocabulary | 131072; tied input/output table |
| Configured position limit | 1,048,576 |
| Bias/dropout | Attention and MLP biases disabled; attention dropout 0 |

Source: [pinned configuration](https://huggingface.co/XHToken/Spark-X2.6-4B/blob/66e3ae7bee54e0a523d692cb8ef841f35409807b/config.json), independently checked against local tensor headers in the [audit](../../reports/spark_architecture_review_20261028/architecture_audit.json).

```mermaid
flowchart LR
  T[Text tokens] --> E[Tied embedding: 131072 x 2560]
  E --> S[Three sliding-attention decoder blocks]
  S --> F[One full-attention decoder block]
  F --> R[Repeat group nine times total]
  R --> N[Final RMSNorm]
  N --> H[Tied embedding transpose]
  H --> O[Next-token logits]
```

Each decoder block contains both attention and a dense MLP. “Hybrid” describes **local versus global softmax attention**. It does not mean a linear-attention/Mamba hybrid, MoE feedforward, or a trained online-memory module.

### 3.2 Native computation and integration consequences

For a normalized token representation `u`, the attention branch constructs Q/K/V, applies layer-specific RoPE to Q/K, computes causal grouped-query attention, multiplies each head output by `sigmoid(Wg u)`, concatenates 16×256 channels, then projects to width 2560. Each KV head serves four Q heads. The next residual branch is:

```text
u = RMSNorm(h)
h_attn = h + Wout(concat(head_attention(u) * sigmoid(Wg u)))
v = RMSNorm(h_attn)
h_next = h_attn + Wdown(GELU(Wgate v) * Wup v)
```

This is a reconstruction of the inspected operators, not a proposal to replace them. There is no Q/K normalization module in that implementation. [Pinned model code](https://huggingface.co/XHToken/Spark-X2.6-4B/blob/66e3ae7bee54e0a523d692cb8ef841f35409807b/modeling_spark.py).

The practical consequences follow from those shapes:

- `hidden_size / num_heads = 160` is **not** the head dimension. Generic Llama assumptions would produce incompatible tensors.
- A residual memory port works in 2560-space; a port before the native attention output projection works in 4096-space. They need different learned interfaces.
- The attention gate is a native pretrained component, distinct from the project's memory or plastic gates. Dropping it changes the model.
- Replacing the GELU branch with SwiGLU changes the model even if tensor shapes remain compatible.
- Modifying a nominal `lm_head` module may miss the actual tied `F.linear(hidden, embedding.weight)` readout. Vocabulary work must target the executed path and maintain tying.

GQA explains reduced KV-head storage; the original [GQA paper](https://arxiv.org/abs/2305.13245) supports that mechanism. The headwise sigmoid placement resembles the mechanism studied in [Gated Attention](https://arxiv.org/abs/2505.06708). That is an architectural comparison, **not evidence of Spark's unpublished design lineage or an ablation showing this gate caused X2.6's gains**.

### 3.3 Independent parameter reconstruction

The following is arithmetic over the inspected tensor shapes, cross-checked against the local BF16 safetensors headers:

| Component, all layers where applicable | Unique parameters |
|---|---:|
| Embedding and tied output table | 335,544,320 |
| Fused QKV matrices | 566,231,040 |
| Attention output matrices | 377,487,360 |
| Attention head gates | 1,474,560 |
| GELU-gated MLP matrices | 2,831,155,200 |
| Block RMSNorms | 184,320 |
| Final RMSNorm | 2,560 |
| **Total** | **4,112,079,360** |

The header inventory has **290 parameter tensors**. It agrees exactly with the calculation `V×H + 36×[H×6144 + 4096×H + H×16 + 3×H×10240 + 2×H] + H`. The tied output table is counted once. The MLP contains about 68.85% of the parameters, which matters when estimating full tuning or deciding which projections an adapter affects. [Reproducible arithmetic and headers](../../reports/spark_architecture_review_20261028/architecture_audit.json).

## 4. Long context: separate information flow, storage and compute

A local layer can directly read only its permitted recent window. Full-attention layers can integrate earlier tokens, after which subsequent local layers can use those contextual representations. Therefore a 512-token local window does **not** imply that the whole model forgets everything older than 512 tokens. It also does not establish persistent memory after the prompt/cache is removed.

The official X2.5 description reports approximately 20T pretraining tokens and a dedicated long-context stage with hundreds of billions of tokens, reaching 1M sequences. Those are vendor statements about training. They do not provide a local 1M execution measurement. [X2.5 training description](https://huggingface.co/XHToken/Spark-X2.5-4B/blob/0bcb35678590218655dff3765b9e61c83b35e9c4/README.md).

### 4.1 Analytical cache payload

For batch one, BF16 K/V and no compression, a cached token costs `2 × 4 × 256 × 2 = 4096` bytes per layer. A backend that physically bounds local caches has approximate payload:

`KV(T) = 4096 × [9T + 27 min(T,512)]`.

| Context | Bounded local-cache payload | All 36 caches retained at full length |
|---:|---:|---:|
| 2,048 | 126 MiB | 288 MiB |
| 8,192 | 342 MiB | 1.125 GiB |
| 32,768 | 1.178 GiB | 4.5 GiB |
| 131,072 | 4.553 GiB | 18 GiB |
| 1,048,576 | 36.053 GiB | 144 GiB |

These are derived payload amounts, excluding weights, batching, padding, workspaces, prefill peaks, copies and allocator overhead. Full-attention layers retain a context-dependent cost even when local caches are perfectly bounded. FP8 **weight** storage does not halve BF16 KV storage.

### 4.2 The reference eager implementation is a separate constraint

Static inspection shows the reference code calls its own `eager_attention_forward`, repeats KV heads, materializes QK scores, applies a mask, and performs FP32 softmax. It does not dispatch an efficient attention backend based merely on an `attn_implementation` name. The local adapter also requires eager attention and disables cache for its training/reference path. [Source lines 68–93 and 189–194](../../checkpoints/sparkx2_5_bf16_source/modeling_spark.py), [local adapter](../../src/replica_lab/architecture/spark_adapter.py).

For an uncached batch-one prefill, one score tensor has `16T²` elements. At 8,192 tokens, that is **2 GiB in BF16**, while the FP32 softmax tensor is **4 GiB**. At 32,768, those amounts become 32/64 GiB. At 1,048,576, they become 32/64 **TiB**. These are individual tensor amounts, not total peaks or timing measurements. They explain why the configured context limit cannot be treated as an affordable eager workload.

An optimized sliding-window kernel can reduce local attention work toward `O(TW)`, but nine full layers still require global attention. Memory-efficient exact attention can avoid retaining a full score matrix without making global prefill arithmetic linear. Switching execution requires mask, gate, RoPE, precision and output-parity checks.

### 4.3 Newly reproduced cache allocation detail

The [Transformers 4.57.1 cache documentation](https://huggingface.co/docs/transformers/v4.57.1/kv_cache) describes bounded sliding caches. Its installed implementation stores `full_key_states[..., -window+1:, :]`, which is a tensor view. In the CPU probe with one head/channel:

| Point | Visible stored tokens | Underlying K allocation | Returned attention length |
|---|---:|---:|---:|
| After 2,048-token prefill | 511 | 4,096 bytes, equivalent to 2,048 BF16 tokens | 2,048 |
| After one decode-token update | 511 | 1,024 bytes, equivalent to 512 BF16 tokens | 512 |

The cache is logically windowed, but immediately after prefill it still owns the larger backing allocation. This does **not** show indefinitely growing decode storage: the next update shrinks it. Measure prefill and steady decode separately, and inspect backing storage rather than `numel()` alone. [CPU probe and versions](../../reports/spark_architecture_review_20261028/cache_template_probe.json).

### 4.4 Position handling needs an explicit test contract

The reference forward accepts `position_ids`, but computes RoPE from the one-dimensional `cache_position`; `position_ids` also goes to mask construction. Arbitrary per-example position resets, packed discontinuities or new multimodal position rules are therefore not established by the signature. Ordinary contiguous positions remain the reference. This observation is not a blanket assertion that left padding is wrong: a uniform position offset can preserve rotary relative relationships. Add specific packed/reset-position and cache-offset checks before depending on those behaviors. [Pinned code, RoPE construction](../../checkpoints/sparkx2_5_bf16_source/modeling_spark.py).

## 5. Precision and backend investigation

The official FP8 config declares compressed-tensors 0.16.0, float-quantized weights, static channel-wise weight scaling and dynamic per-token input activation quantization. Attention gates and the language head are excluded; `kv_cache_scheme` is null. Its top-level BF16 dtype does not describe every stored tensor. [Pinned FP8 configuration](https://huggingface.co/XHToken/Spark-X2.6-4B-FP8/blob/0318d6f6ab070c7b5b43052db571489b2508006c/config.json).

The reference BF16 implementation also promotes the residual stream, normalization arithmetic and softmax work to FP32 while casting branch inputs and final outputs as specified in code. BF16 checkpoint storage therefore does not mean every intermediate is BF16. [Inspected implementation](../../checkpoints/sparkx2_5_bf16_source/modeling_spark.py).

Header inspection refines the earlier parameter-only accounting:

| Artifact | Local header result | Tensor payload, before header/container overhead |
|---|---|---:|
| Official BF16 | 290 tensors; 4,112,079,360 BF16 parameters | 8,224,158,720 bytes = 7.659 GiB |
| Official FP8 | 3,774,873,600 F8_E4M3 parameters; 337,205,760 BF16 parameters; 180 BF16 scale tensors containing 1,142,784 elements | 4,451,570,688 bytes = 4.146 GiB |

FP8 has **470 stored tensors including scales**. Its 1,142,784 scale elements are metadata for quantized operations, not additional learned backbone parameters. The earlier 4,449,285,120-byte parameter-only estimate omits 2,285,568 bytes of scales; that earlier figure is correct when labeled as excluding scales. [Local header audit](../../reports/spark_architecture_review_20261028/architecture_audit.json).

| Execution identity | What is established in this workspace |
|---|---|
| Official BF16 | A bounded frozen-base extension smoke and fresh-process reload are recorded. |
| Official native FP8 | No successful local execution receipt; MPS failure recorded; current adapter construction expressly refused. |
| FP8-derived BF16 | A separate dequantized artifact passed its own bounded extension smoke/reload; it lacks the official FP8 activation semantics. |
| Full FP32 Spark AdamW training | Analytical base weights + gradients + two moments = **61.275 GiB**, before activations; exceeds the 50 GiB driver guard. |

The BF16 and derivative runs each changed four extension tensors with the base unchanged and reloaded with zero within-arm logit difference. Cross-arm maximum logit difference was 10.875, so those successful reloads do not establish interchangeability. Eight-token smoke generations are not the declared larger-budget quality evaluation. [Existing selected-run receipt](../../reports/spark_qualification_20261026/selected_runs.json); its eight referenced raw files were rehashed successfully in this review.

There are useful upstream implementation references, but they have different scopes:

- The pinned [Spark MLX model](https://github.com/XHToken/Spark-MLX-LLM/blob/079a6a328ef878ac3cc4884c07e830add0540007/spark_mlx_llm/model.py) independently represents the same Q/KV geometry and head gates, and creates rotating local caches plus ordinary full-attention caches. Its conversion and serving paths are alternative inference implementations; they do not qualify this PyTorch learner or official FP8 training.
- The pinned [official vLLM plugin](https://github.com/XHToken/Spark-plugin/blob/3e1040e63a5907e4c748a485d6795d739dcd5bd6/src/vllm_spark2_5_plugin/spark2_5.py) uses `GeluAndMul`, per-layer sliding windows, explicit head dimensions and the head gate. That supplies a second implementation cross-check, not numerical parity evidence from this machine.
- The [release README](https://github.com/XHToken/Spark-X2.6/blob/27ef075943cd9af43c1d1138e2ee267f1a4f872c/README.md) lists SGLang, llama.cpp and other deployment routes. A serving implementation's support statement does not imply input-gradient, optimizer or checkpoint compatibility for the proposed learner.

The active protocol's `official_native_fp8 + device=mps + Spark2_5Adapter` combination remains a **concrete blocker**. Preserve the three precision identities. Backend qualification must establish both official activation semantics and gradients through frozen quantized layers into the extensions; this is more than ordinary inference support.

## 6. Training, thinking controls and published performance

X2.5's card describes pretraining, long-context training, supervised fine-tuning, domain-specialized RL teachers and MOPD consolidation. X2.6's release says gains come from improved post-training, but the reviewed primary sources do not disclose a complete X2.6 recipe: exact token counts, mixtures, teacher checkpoints, rewards, schedules, compute, seed variance and contamination controls remain unresolved.

The [MOPD paper](https://arxiv.org/html/2606.30406v1) explains distillation from domain teachers on student-generated rollouts. Its named experiments concern Qwen3-30B-A3B and MiMo-V2-Flash; the inspected text contains no Spark evaluation. It is useful method background, not a Spark training report. I did not locate a standalone Spark-X2.6-4B architecture/training paper in the searched official repositories and arXiv results. That is a search result, not proof that none exists.

The template specifies the actual thinking interface:

| Requested behavior | Template arguments | Observed prompt behavior |
|---|---|---|
| None | `enable_thinking=False` | Ends generation prefix with `</think>` |
| Low | `enable_thinking=True, reasoning_effort="low"` | Low instruction plus `<think>` |
| High | `enable_thinking=True, reasoning_effort="high"` | High instruction plus `<think>` |
| Omitted | Defaults | Same as high |

With thinking enabled, `reasoning_effort="none"` raises an error; it is not the no-thinking selector. These are prompt controls, not numerical token-budget guarantees. Enforce generation caps separately and report actual tokens. The template supports text/tool messages and rejects non-text content blocks; that is further evidence that the released interface is text-only. [Pinned template](https://huggingface.co/XHToken/Spark-X2.6-4B/blob/66e3ae7bee54e0a523d692cb8ef841f35409807b/chat_template.jinja), [CPU rendering checks](../../reports/spark_architecture_review_20261028/cache_template_probe.json).

Selected numbers transcribed directly from the official SVG, all **vendor-reported, not reproduced here**:

| Benchmark | X2.6-4B | X2.5-4B |
|---|---:|---:|
| BFCL-V4 | 71.8 | 65.1 |
| SWE Verified | 62.8 | 41.6 |
| SWE Pro | 54.3 | 44.4 |
| AA-LCR v1.1 | 67.3 | 64.7 |
| Graphwalks BFS 256k F1 | 92.2 | 47.1 |
| Graphwalks BFS 1M F1 | 34.2 | 2.5 |
| MRCR v2 256k–512k | 65.6 | 9.7 |
| MRCR v2 512k–1M | 26.0 | 2.9 |

The source uses **high thinking**, temperature 1.0 and top-p 0.95. Its in-house SWebBench/PAgentBench were described as awaiting open release. The table supports a reported improvement, especially on coding and some long-context tasks; it also exposes substantial remaining difficulty near 1M. It cannot establish that our 32-token, no-thinking pilot inherits the same scores. Avoid comparing the older card's AA-LCR number directly with the new **v1.1** row. [Pinned official benchmark chart](https://huggingface.co/XHToken/Spark-X2.6-4B/blob/d1b5f26071101c27fcd07728e7a7770e428281f4/images/all_benchmark.svg), [30 parsed rows](../../reports/spark_architecture_review_20261028/architecture_audit.json).

## 7. Cross-check of the project's documents and implementation

This review adds a dated interpretation without rewriting frozen historical reports or source pins. The architecture index is itself included in earlier bundle manifests, so silently rewriting its old “current priority” paragraph would also create provenance drift.

| Document or implementation | Review result | Correct reading / needed follow-up |
|---|---|---|
| [Doc20, opening and S0–S1 status](20_Spark_X2_6_FP8_Preparation_Plan.md) | **Stale as current state** | Its no-weights/no-adapter and unrun-control statements describe an earlier preparation phase. Later BF16/derivative receipts supersede those status readings. Native FP8 remains blocked. |
| [Architecture index](README.md), [Doc02](02_Backbone_Integration.md), [Doc23 §20](23_SparkX_2_6_4B_Final_Product_Blueprint.md#20-minicpm5-first-validation-program) | **Priority conflict if read as current instructions** | MiniCPM-first language belongs to that design lineage. [Doc26](26_Spark_Text_Memory_Milestone.md) and current `AGENT_START.md` make Spark text memory active, with MiniCPM as a historical control. |
| [Doc23 §5.1](23_SparkX_2_6_4B_Final_Product_Blueprint.md#51-preserve-the-original-spark-computation) | **Verified** | Native dimensions, GELU-gated MLP, head gate, full/sliding schedule and tied output path match public code and local headers. |
| [Doc23 §5.2–5.3](23_SparkX_2_6_4B_Final_Product_Blueprint.md#52-reference-intrinsic-block) | **Proposal, not implemented equivalence** | Nine internal residual sites are a design. The current adapter applies post-block hooks; its ordinary default is sites 11/23. The active protocol separately declares 3/7. Keep these profiles explicit. |
| [Doc23 §22.3](23_SparkX_2_6_4B_Final_Product_Blueprint.md#223-kv-and-context) | **Arithmetic verified; runtime qualification incomplete** | The 342 MiB / 1.125 GiB figures are correct under their stated assumptions. Add the dense eager prefill and slice-backing allocation findings before using them for admission. |
| [Doc24 capacity inventory](24_SparkX_2_6_v5_1_Refinement_Plan.md#44-residency-inventory-measured-analytical-and-unsealed) | **Appropriately provisional** | The base 61.27 GiB FP32 training arithmetic agrees. P64/Mkv/vision estimates remain product candidates, not evidence that expanding them repairs current learning. |
| [Doc26 and executable protocol](../../configs/spark_text_memory_protocol_v1.json) | **Clear target, blocked native path** | Text-memory/spine-first, source-free controls, correction/retention and cost accounting are appropriate. The fixed MPS/native-FP8 adapter combination cannot currently fulfill the pilot. |
| [Current Spark adapter](../../src/replica_lab/architecture/spark_adapter.py) | **Prototype implementation exists** | Frozen backbone, eager uncached execution, post-block injections and separate spine branches are visible in code. Do not describe it as absent, or as proof of the planned nine-site internal surgery. |
| [Selected-run receipts](../../reports/spark_qualification_20261026/selected_runs.json) | **Integrity reverified; scope narrow** | Preserve extension-update/reload results and native-FP8 refusal separately. A single changed-tensor check is not knowledge acquisition. |

The source/code review also confirms a distinction already made correctly in the blueprint: the native checkpoint has **no vision or audio encoder, persistent E/P bank, Mkv table, dynamic vocabulary overlay, spine, or online optimizer loop**. Each proposed component changes the graph or saved state and requires its own trained interface and causal evaluation.

The literature does not close that gap:

- [MemT-1.0 v2](https://arxiv.org/html/2610.26825v2) introduces Gated DeltaNet-2 memory, Mixture-of-Storage, backbone preparation and a 34B-token training program. It supports studying trained intrinsic memory; it does not validate grafting the project's sparse plastic experts onto Spark. No Spark result appears in the inspected text.
- [Nested Learning / HOPE](https://arxiv.org/abs/2512.24695) treats learning levels as optimization problems with their own context flows. Several memories or update clocks alone do not establish that learning algorithm or its benefits.
- [Engram](https://arxiv.org/abs/2601.07372) studies scalable conditional n-gram lookup. A writable episodic store with correction, provenance and continual parameter updates is a different mechanism.

These are useful research directions. The combined Spark product remains a hypothesis to test, and the active milestone appropriately defers the expanded mechanisms until a small learner is useful.

## 8. Recommended order after this review

1. **Close the execution identity first.** Keep the active official-FP8 source pin. Specify an executable backend/adapter combination that preserves channel-scaled weights and dynamic per-token activations, plus gradients into the selected extensions. Official BF16 and the derivative remain separately labeled controls. The existing MPS path is not admitted by this review.
2. **Prove native equivalence before mechanism breadth.** Check disabled-extension parity, full/sliding boundary behavior at 511/512/513 and longer prompts, cache versus no-cache outputs, tying, gate/MLP preservation, and exact positional assumptions. The existing short reload result does not cover all of these.
3. **Measure ingestion, prefill and decode separately.** Charge query/sensing passes, dense eager workspaces, physical cache storage, backward, updates and reload. Maintain the 50 GiB driver / 512 MiB swap / 15% host-free guards. A model's parameter count is not a workload budget.
4. **Run the bounded text-memory question only after those prerequisites.** Compare no-memory/full-eligible-context, E-only, source-free P-only and E+P with reset/wrong/shuffled controls. Evaluate generated answers, corrections, unknowns, delayed recall, restart, new instances and prior-rule retention. Strong baseline performance should not be mislabeled as learning.
5. **Keep the later product work conditional.** Nine-site surgery, P64/Mkv expansion, multimodal training, nested outer optimization and RSI require their own positive causal evidence. No architecture paper or benchmark average replaces that experiment.

The architecture is now well specified enough to build against. The unresolved decisions concern **backend execution, trained memory use, and measured utility/cost**, not uncertainty about the original decoder dimensions.

## 9. Scope, reproducibility and remaining uncertainty

Reviewed workspace HEAD: `b350e87e972e00786404e827f202539962726fe9`, branch `codex/d2-capacity-runtime-pilot`, including identified uncommitted text-memory work. Inputs and exact hashes are recorded in the [audit](../../reports/spark_architecture_review_20261028/architecture_audit.json). The later commit containing this report is a documentation/evidence change, not a new experimental preregistration.

Fresh checks completed:

- 62 downloaded public text/metadata files, including licenses/attribution, match their recorded SHA-256 values.
- Eight local architecture/template files match newly fetched immutable upstream revisions.
- Both local checkpoint headers agree with the independently calculated parameter count; FP8 scales are separately inventoried.
- All eight existing raw BF16/derivative receipt hashes match their recorded identities.
- The small CPU cache-allocation and four-mode template checks pass under torch 2.14.1 / Transformers 4.57.1.

Still unknown or unproven: complete X2.6 post-training recipe; independent reproduction of vendor benchmarks; actual 1M throughput/peak memory on an admitted backend; official native-FP8 extension training; cached equivalence of the current memory path; useful persistent learning, multimodal capability and long-run retention of the proposed product. This report does not claim exhaustive access to unpublished training information or internal evaluation traces.

Public model cards and source files were read as evidence. Search-result snippets and community conversion claims were not used as authority for the architecture. Some Hugging Face file URLs failed in the web reader; bounded HTTPS retrieval of the exact public revisions succeeded and is captured in the source manifest. Historical documents, failed results, checkpoint pins and the user's existing implementation changes remain intact.
