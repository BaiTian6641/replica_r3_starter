# Cost and implementation plan for vocabulary and LoRA

Use separate budgets for static weights, mutable state, optimizer state, saved activations and temporary tensors. The calculations here use the inspected Spark dimensions. They do not predict device peak memory or throughput.

## Spark dimensions and target selection

The prepared official source has 36 layers, hidden width 2,560, MLP width 10,240, 16 query heads, four key/value heads and head dimension 256. Consequently the attention output width before projection is 4,096, rather than 2,560. Its five large projections have the following shapes. [Prepared configuration](../../../reports/spark_preparation_20261025/run_01/assets/official_bf16_control/config.json), [model source](../../../reports/spark_preparation_20261025/run_01/assets/official_bf16_control/modeling_spark.py).

| Projection | Input width | Output width | Rank-8 LoRA parameters per layer |
|---|---:|---:|---:|
| `self_attn.q_k_v_proj` | 2,560 | 6,144 | 69,632 |
| `self_attn.out_proj` | 4,096 | 2,560 | 53,248 |
| `mlp.gate_proj` | 2,560 | 10,240 | 102,400 |
| `mlp.up_proj` | 2,560 | 10,240 | 102,400 |
| `mlp.down_proj` | 10,240 | 2,560 | 102,400 |

Target names such as `q_proj` and `v_proj` will not select this fused QKV module. Adapting fused QKV with one shared A is also different from separate logical Q/V adapters; equal ranks do not ensure equal parameter counts or update spaces. Inspect and record every matched module and its trainable parameters. Leave the small head gate, norms and base embedding outside the first backbone-LoRA comparison.

A narrow first backbone experiment can use `down_proj` in zero-based layers 11 and 23, aligned with the earlier proposed memory-port depths. Compare it with direct learning in the new residual ports. An eight-block variant uses indices 3, 7, 11, 15, 19, 23, 27 and 31 as an explicitly declared broader control. All 36 blocks are a capacity comparison, not the starting default.

| Scope | Rank | Parameters | BF16 adapter weights | FP32 weights + gradients + two Adam moments |
|---|---:|---:|---:|---:|
| Two square residual ports | 8 | 81,920 | 0.15625 MiB | 1.25 MiB |
| Two backbone down projections | 8 | 204,800 | 0.390625 MiB | 3.125 MiB |
| Eight blocks, all five projections | 8 | 3,440,640 | 6.5625 MiB | 52.5 MiB |
| All 36 blocks, all five projections | 8 | 15,482,880 | 29.53125 MiB | 236.25 MiB |
| All 36 blocks, all five projections | 16 | 30,965,760 | 59.0625 MiB | 472.5 MiB |

The two-square-port row describes hypothetical rank-r updates, not the full parameter count of the existing native modules. The optimizer column assumes 16 bytes per trainable scalar and is one explicit storage scenario, not a universal PyTorch allocation rule. Mixed precision, master copies, optimizer choices and temporary buffers change it. LoRA-FA trains only B but retains A and its correction data; both are included separately in the [calculator output](../../../reports/vocabulary_lora_research_20261025/analytical_costs.json).

The unmerged LoRA branch performs approximately `r(in + out)` additional multiply-accumulates per position per target. That arithmetic can be small relative to the frozen matrix while launch overhead remains material. Merging a stable adapter can remove these extra matrix multiplies, but requires a new effective weight artifact and cache invalidation. Repeatedly merging into a quantized base can accumulate rounding changes and should not be the routine online update path.

## Vocabulary and lexical memory budgets

| Additional tied rows | Parameters | BF16 weights | FP32 training-state scenario | Wider output-head arithmetic |
|---|---:|---:|---:|---:|
| 256 | 655,360 | 1.25 MiB | 10 MiB | +0.1953125% |
| 1,024 | 2,621,440 | 5 MiB | 40 MiB | +0.78125% |
| 2,048 | 5,242,880 | 10 MiB | 80 MiB | +1.5625% |

The percentages refer only to the vocabulary projection, not whole-model latency. Input lookup is inexpensive, but the new rows still participate in output scoring. Full sequence logits and cross-entropy can dominate activation storage even when only a few rows are trainable. Evaluate a compatible chunked loss rather than assuming selected-row training removes this cost.

Reuse the existing episodic slot budget for lexical bindings. Materializing just 16 selected full-width BF16 embeddings requires 80 KiB. Materializing full-width embeddings for every memory slot is a different, larger design. Account separately for source spelling, token payloads, concept links, temporal metadata, indices and replay. The prior Dynamic Engram state estimate must not be added twice when a lexical record is simply another use of the same slot.

## Activation compression and update cadence

For a row-oriented adapter, let `X[N, d]`, `A[d, r]`, `B[r, out]` and `Z = XA`. CARE-LoRA reconstructs the input through a regularized small solve:

```text
M = solve(Z^T Z + lambda I, Z^T X)
X_hat = ZM
```

Backward computation can use the factors without materializing `X_hat`. The inspected implementation uses FP32 reconstruction data in relevant paths. A useful conservative payload comparison is BF16 `X` plus `Z` for vanilla LoRA, BF16 `Z` for LoRA-FA, and FP32 `Z` plus `M` for CARE. This is a branch-level illustration; aliasing, additional saved tensors, solve workspaces and the rest of the network require device measurement. [CARE-LoRA method](https://arxiv.org/html/2607.11940v1), [inspected source](https://github.com/fishandyu/CARE-LoRA/blob/f60d5f3f70f80451e4a4851dcbb945e5b9e4d574/peft/src/peft/tuners/lora/layer.py).

| Flattened positions N, with d=2,560 and r=16 | Vanilla payload | FA payload | CARE payload |
|---|---:|---:|---:|
| 1 | 5,152 B | 32 B | 163,904 B |
| 32 | 164,864 B | 1,024 B | 165,888 B |
| 256 | 1,318,912 B | 8,192 B | 180,224 B |
| 2,048 | 10,551,296 B | 65,536 B | 294,912 B |

Under these assumptions CARE's payload becomes smaller when `N > 2rd/(d-r)`, or approximately 32.2 positions. The 256-position example saves branch payload, whereas a one-position update increases it roughly 31.8 times. This is one reason to aggregate verified learning examples instead of taking a gradient step for every new word. The real break-even point also includes solve time and workspaces.

Exact first-order input gradients do not establish higher-order/meta-gradient compatibility. Custom autograd, checkpoint recomputation, mixed precision and recurrent state detachment each need qualification. CARE's saved-context buffers may not all appear in simple saved-tensor hooks; measure actual allocator/device peaks. Approximate methods must report gradient error and downstream learning, not only loss from one step.

Offline initializers also have peaks: one FP32 input covariance occupies 25 MiB at width 2,560, 64 MiB at 4,096, or 400 MiB at 10,240. SVD workspaces and statistics for multiple layers can be larger. Stream calibration, process one module at a time and charge all preparation to the comparison.

## Compatibility work in this repository

The initial review snapshot was `7418b99040a921f1b43fae54c77c69e70ae68903`; ongoing repository work advanced during the research. The [manifest](../../../reports/vocabulary_lora_research_20261025/review.json) records both the inspected source hashes and a later snapshot. This is a source/design review, not an admission result for concurrent runtime or vision changes.

| Existing area | Required implementation work |
|---|---|
| [Architecture ports](../../../src/replica_lab/architecture/ports.py) | Add native lexical selection and bounded readout contracts. Existing `RecordCopyHead` and `RecordPointerHead` receive caller-selected candidates |
| [Contracts](../../../src/replica_lab/architecture/contracts.py) | Bind vocabulary version, active rows, alias/concept identity, adapter scale and update-state ownership |
| [MiniCPM adapter](../../../src/replica_lab/architecture/minicpm5.py) | Use as an interface/experiment prototype; update explicit ID-range and copy-bias bounds when testing a vocabulary extension |
| [Loading](../../../src/replica_lab/architecture/loading.py) | Validate base/tokenizer/adapter/state identities together; reject incompatible shapes and histories |
| [Memory session](../../../src/replica_lab/memory_system/session.py) | Preserve raw observations, representation digests and explicit migration/rebuild behavior |
| Prepared Spark modeling source | Implement input and tied-output row integration through the actually executed path; preserve custom attention and loss behavior |
| Future adapter implementation | Target fused modules explicitly; save scaling, initialization, residual-base identity and trainable scope |
| Evaluation and reports | Add lexical, grounding, retention, gradient and measured-cost protocols before declaring any method admitted |

The tied Spark forward path calls `F.linear(hidden_states, self.model.embedding.weight)`. Its `get_output_embeddings()` API returns a head that this path bypasses. Therefore a successful wrapper installation is not proof that either trainable rows or output adapters influence the model. Check actual input gradients, generated logits and save/reload behavior. PEFT selected-token support is a useful reference, but its inspected implementation constructs merged weights for some paths; inspect memory behavior rather than assuming a sparse optimizer implies sparse forward allocation. [Pinned selected-token implementation](https://github.com/huggingface/peft/blob/7d6315ae910bf62b6044a7ca917bccf5881f5e43/src/peft/tuners/trainable_tokens/layer.py).

The reviewed environment has Torch 2.14.1, Transformers 5.20.2 and tokenizers 0.23.3, with no installed PEFT. Existing native ports are not an already implemented PEFT integration. The official repositories were inspected as source, not installed or executed. Pin a tested dependency set only when the implementation experiment begins.

## Hardware and qualification

Retain the earlier 64 GiB M2 Ultra limits: one model-owning MPS worker, no silent CPU fallback, a 50 GiB driver-memory stop, a 512 MiB sustained swap-growth stop, and at least 15% free physical memory. These inherited limits concern full experiments, not only adapter weights. A future 24 GB RTX PRO 4000 Blackwell SFF run needs independent CUDA, quantization and peak-memory qualification.

The prepared Spark FP8 and BF16 releases are separate controls. The source-preparation report does not establish FP8 input-gradient support, a working native training path or equivalence to the BF16 release. A qualified backward path is required for trainable rows, upstream vision modules and LoRA. If unsupported, use a separately qualified control; do not silently change precision or device.

Qualification should establish, in order: no-op parity; intended module targeting and zero initial updates; forward/backward correctness; new-row input/output gradients; immutable base rows; restart and artifact identity; bounded state allocation; then task learning, retention and measured cost. For approximate methods, compare gradients with the exact baseline and set tolerances from observed task stability before accepting results. Custom solves and decomposition operators require explicit MPS/CUDA checks.

## Deliverables for the implementation phase

The first implementation unit should provide the lexical-state schema, native query/write/readout path, controller-disabled parity, and a benchmark that distinguishes spelling from meaning. The second should provide a narrow adapter interface with complete target/scale metadata and matched vanilla/FA runs. The third should add the split tied-row mechanism and migration checks. Only after these pass should joint vision training, advanced compression and learned update policies be integrated.

The plan deliberately leaves base-checkpoint rewriting, byteification and permanent merging as separate campaigns. They may eventually be valuable, but their preparation cost and changed artifact identity belong in their own evaluation. The immediate objective is useful persistent learning with bounded memory, stable old behavior and a measured compute budget.
