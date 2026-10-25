# Compute budget and implementation roadmap

All new architecture numbers are analytical proposals. They do not establish device fit, latency, learned capacity or a successful training recipe. [Calculation source](../../../reports/dynamic_engram_research_20261025/estimate_capacity.py) and [generated values](../../../reports/dynamic_engram_research_20261025/analytical_costs.json) make the assumptions inspectable.

## What is cheap, and what still costs

Fixed-capacity memory can reduce repeated long-context work and store more information than the current tiny fast state. Its main potential advantage is amortization: train a useful writer once, then apply inexpensive native updates to many observations. Neither sparse lookup nor generated weights removes the cost of learning that writer. More stored facts also do not automatically improve reasoning.

Keep four ledgers: shared parameters; live state per owner; inference work; and learning work. Report optimizer buffers, saved snapshots, replay archives and training activations separately. A low active-parameter count is not a low total storage count, and a tiny matrix update does not imply low end-to-end latency.

| Quantity | Compact pilot | Expanded candidate | Accounting boundary |
|---|---:|---:|---|
| Static pattern table | 4,194,304 parameters | 33,554,432 parameters | Embeddings only; trainable shared weights |
| Static table BF16 storage | 8 MiB | 64 MiB | Excludes projections and runtime overhead |
| Static table FP32 weights, gradients and two Adam moments | 64 MiB | 512 MiB | Four FP32 arrays; not an estimate of every mixed-precision optimizer |
| Episodic keys and values | 3 MiB | 12 MiB | 4,096 / 16,384 slots, 128+256 BF16 dimensions |
| Optional observed-token payload | 0.25 MiB | 1 MiB | 16 int32 tokens per slot |
| Metadata allowance | 0.25 MiB | 1 MiB | 64 bytes per slot; final schema must be sized |
| Shared fast state | 64 KiB | 256 KiB | One / four experts, four 64×64 FP32 heads each |
| One rank-8 adaptation at two ports | 320 KiB | 320 KiB | Generated FP32 factors, not one adapter per slot |
| Workspace | 16 KiB | 16 KiB | 16×256 FP32 activations |
| Temporal cache | 32 KiB | 32 KiB | 64×256 BF16 activations |
| Appraisal | 32 bytes | 32 bytes | Eight FP32 values, excluding controller weights |
| **Sum of listed live mutable payloads** | **3.92 MiB** | **14.61 MiB** | Excludes the static table and all shared weights |

The state sum excludes indices, full provenance manifests, padding, allocator overhead, temporary reads, retained graphs, replay and duplicate snapshots. Exact scope/identity strings may be represented by indices into a separately accounted manifest. Saving multiple generations multiplies storage; batching owners multiplies their live state. FP32 rather than BF16 keys/values doubles that part of the bank. The optional generated adapter is included so it is not hidden behind a “few kilobytes of fast memory” claim.

### Shared projection and generator costs

One proposed fast-memory port uses a 2,560→256 stem, six 256→256 matrices for query/key/value/erase/write/decay, and a 256→2,560 output. This is **1,703,936 matrix parameters per port**, or **3,407,872 for two private ports reading shared state**. It excludes biases, normalization, output gates and routers. This factorization is a new design choice; it differs from larger direct backbone-width projections in existing implementations.

The compact static table, these two ports and a proposed 2–8M-parameter adapter generator subtotal approximately **9.60–15.60M shared parameters**. This is not the total extension: episodic readers/writers, core, visual bridge, vision tower, appraisal and any skill adapters need their own counted manifests. The generator allowance is an experiment, not evidence that a generator much smaller than published Doc-to-LoRA will work.

Preserve the prior **4.2–4.6B total shared-parameter planning envelope** only if actual selected modules fit it. The user's 4B requirement applies to the language backbone, so this envelope is not a hard user cap. Adding a much larger generator or perception tower requires a revised total, with an explicit quality/cost reason.

### Arithmetic does not predict latency

Under the compact one-expert factorization, two fast-memory reads require about **5.57M FLOPs per generated token**. Spark's linear transformations, including its tied output projection, require approximately **8.224B FLOPs per token** under the source geometry. The read-only fraction is approximately **0.068%**. This comparison excludes attention's sequence-dependent work and includes neither the other new branches nor writes, routing, memory traffic or launch overhead. It cannot be presented as the total system overhead.

A dense query-boundary search costs 524,288 key-dot-product MACs at 4,096 slots, or 2,097,152 at 16,384. Top-k selection, filters, normalization and value reads add work. Keep the shortlist fixed during ordinary decoding; charge any extra inspection/search explicitly. The static table reads 512 payload bytes per token in the compact configuration before caching and projection costs. Irregular gathers can be bandwidth-limited despite low arithmetic.

The existing SGDN-2 Python token loop is a likely throughput bottleneck. Establish a pure functional reference, then compare vectorized, chunked or fused implementations for identical outputs and gradients. Avoid assuming CUDA/Triton performance transfers to MPS. For only four experts, dispatch overhead may outweigh arithmetic saved by sparse selection.

## Amortization and fair baselines

Let `P(H)` encode a history, `W(H)` write it, `P(q)` encode a later question, and `D(a)` decode an answer. A simplified history-reuse comparison over `n` questions is:

```text
native memory: P(H) + W(H) + n * [P(q) + D_memory(a) + read/inspection cost]
full replay:   n * [P(H + q) + D_full(a)]
```

Also compare **full context with reusable KV**, which can pay its history prefill only once. Native memory must then justify its compression, attention cost or cross-session reuse against the reusable-cache baseline. Do not claim a prefill saving that comes only from making the control unnecessarily replay its history.

The useful cost equation includes source encoding, writer computation, selected visual crops/frames, any second decoder pass, generated adapters, consolidation and validation. Re-encoding every observation through an extra full decoder can overwhelm the nominal matrix savings. Prefer writers consuming activations already computed by the native model. When an additional pass is necessary, report it and test whether it meets the total-interaction gate.

Record median and p95 observation latency, time to first token, decode latency, peak allocated/reserved/driver memory, swap, bytes read/written and total campaign tokens. Measure cold start separately from steady state. Use matched output lengths and quality, three or more independent trials and synchronized device timings. Include dense and indexed search only when both are fully implemented; a theoretical index is not a benchmark result.

For the incremental memory/controller overhead gate, hold the backbone, perception pipeline, frame/crop budget and precision fixed. Report the vision encoder's absolute cost separately. A comparison that charges vision only to the memory candidate cannot isolate memory overhead. Conversely, excluding vision from the complete application's resource ledger would hide a major cost.

## Hardware qualification

The pinned Spark model has 4,112,079,360 unique parameters: approximately **7.66 GiB in BF16** or **15.32 GiB in FP32** for weights. FP32 parameters, gradients and two Adam moments alone require about **61.27 GiB**. Thus full-base Adam training is outside the intended 64 GiB Mac envelope before activations. Begin with frozen-backbone/module training, then qualify small adapters if necessary.

Frozen parameters still require activation storage and backward computation when training an upstream writer or vision bridge. Short sequences, bounded observation-to-query credit, gradient checkpointing and small trainable scopes are consequential; placing the decoder in `no_grad` is not an equivalent optimization. Checkpoint recomputation must not cause a second state commit.

Retain the repository's single MPS worker, no CPU fallback, 50 GiB driver stop, sustained 512 MiB swap-growth stop and 15% minimum host-free guard. The prospective 24 GB RTX Pro 4000 Blackwell SFF requires its own measured fit and throughput qualification. Do not assign completion dates before a representative forward/backward pilot establishes throughput.

The [new source-preparation manifest](../../../reports/spark_preparation_20261025/run_01/preparation_manifest.json) is explicitly **preparation-only and unqualified**. It distinguishes plain BF16 weights from the FP8 release's compressed-tensors storage, per-channel weight scaling and dynamic activation quantization. Neither native forward/backward nor reload is established. A capability-matrix BF16 entry currently mentions compressed-tensors despite the manifest's plain-BF16 storage description; resolve that wording against the actual source before using it as an admission decision. FP8 storage does not prove MPS FP8 kernels or input-gradient support. A dequantized derivative must retain its own identity.

The inspected Spark code uses a sliding mask without demonstrated sliding KV eviction. Do not budget cache as constant merely because most layers have a 512-token window. Under BF16 KV with all 36 layers retaining history, raw keys and values use 144 KiB per token, or **4.5 GiB at 32,768 tokens**, before overhead. Memory can replace discarded history only after the learned-memory quality gates pass. Position counters, recurrent state and KV invalidation still need correct semantics.

## Concrete repository implementation map

These changes are proposed for a later implementation task. Current code and preserved experiments were not modified by this review.

| Work area | Existing foundation | Proposed change and completion evidence |
|---|---|---|
| Typed state | `src/replica_lab/architecture/contracts.py` | Explicit owner-scoped episodic/fast/workspace/appraisal states; schemas, batch isolation, reset and serialization equivalence |
| Dynamic bank | New `src/replica_lab/architecture/dynamic_engram.py` | Native writer, query encoder, candidate selection, null route, correction links and exact observed-span payload; storage/address/readout probes |
| Readers and fusion | `src/replica_lab/architecture/ports.py` | Independent episodic and fast readers; bounded output scales; gradients and branch attribution |
| Fast update | `src/replica_lab/neural.py` | Preserve SGDN-2 reference; explicit observation-boundary reduction, optional MLP inner optimizer, then measured vectorization |
| Cognitive core | `src/replica_lab/architecture/core.py`, `heads.py` | Typed region/event sources, trainable policy heads and bounded event-level deliberation; compare detached baseline |
| Appraisal | New `src/replica_lab/architecture/appraisal.py` | Eight bounded controls, event decay, uncertainty separation; neutral/shuffle/latent controls |
| Backbone adapter | `src/replica_lab/architecture/minicpm5.py`; new Spark-specific adapter | Shared contracts with distinct native geometry, positions, dtype and hook semantics; forward/gradient/reload parity |
| Persistent representation | `src/replica_lab/runtime/memory_representation.py`, `persistent_fast_memory.py` | Extend repaired producer identity to new tensors and visual encoders; snapshots store model-produced state without semantic host routing |
| Learning operations | `src/replica_lab/runtime/director.py`, `learning_cycle.py` | Native trainable selection policy and model-owned loss; runtime executes numerical updates and records candidate identity |
| Training/evaluation | New DE experiment protocols and scripts | Unexposed worlds, explicit trainable tensors, canonical schedules, controlled source access, direct-model and cost reports |

Names of new files are suggestions; preserve existing interfaces where possible. The exact-memory path remains a diagnostic/reference control, not an implicit fallback counted as learned memory. Keep `src/replica_lab/runtime_pilot.py` untouched.

The new [vision preparation report](../../../reports/vision_data_preparation_20261025/report.json) contains 2,688 candidate model rows across V0/V1 synthetic controls but **zero admitted training rows**. Its audit does not establish broad visual competence or permission to skip the named training protocol. Reuse its generator/audit separation for fresh memory episodes; scene graphs and expected answers remain verifier-only.

## First implementation slice and stop decisions

Implement DE0/DE1 first: export-qualified source, fresh worlds and independent storage/addressing/readout probes. Then build the smallest DE2 bank on MiniCPM, with a null route, corrections and unchanged-fact tests. In parallel only where dependencies permit, compare DE3 with the existing fast-memory baseline. A passing isolated V0 can prepare vision without waiting for ambitious self-improvement experiments.

Do not enlarge the static table because the current branch failed; first establish its direct usefulness. Do not add a generator to repair an unexplained novel-value failure. Do not count appraisal as learning progress unless it changes useful decisions. Advance to Spark, visual integration, adapters and multi-rate learning using the [stage gates](03_Learning_and_Evaluation.md), not the attractiveness of the architecture diagram.

The first deliverable should be modest and decisive: a native model that remembers a new observed binding, corrects it, uses it across views or later events, and leaves unrelated answers intact at a measured cost. Later skill consolidation and recursive improvement build on that result.
