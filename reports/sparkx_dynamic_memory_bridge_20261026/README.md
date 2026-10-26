# Dynamic memory and unified bridge research receipt

26 October 2026. Companion to [canonical revision 4.2](../../docs/architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md).

## Deliverables

- [Research review](../../docs/research/sparkx_dynamic_memory_bridge_20261026/Research_Review.md): 15 primary papers, selected-section reading and paper/code checks.
- [Architecture](../../docs/research/sparkx_dynamic_memory_bridge_20261026/Unified_Bridge_and_Memory_Design.md): learned memory addressing, fast parameters and a native attachable-module interface.
- [MiniCPM study](../../docs/research/sparkx_dynamic_memory_bridge_20261026/MiniCPM_Validation.md): causal controls, swap tests and cleared-state learning.
- [Study config](../../configs/sparkx_2_6_4b_dynamic_memory_bridge_study_v1.json): proposal only; runtime/training flags false.
- [Evidence manifest](evidence.json): 23 source retrieval records, including failed endpoints and two previously downloaded papers reread for this update. Full papers remain outside the repository.
- [Repository snapshot](repository_snapshot.json) and [prior canonical snapshot](architecture_v4_1_snapshot.txt): dated source state before this revision.

## Analytical capacity examples

The [calculator](estimate_costs.py) produces [this ledger](analytical_costs.json) using only the standard library. These are matrix/state counts, not measured runtime or complete model totals.

| Spark candidate | Accounted capacity | Stored size |
|---|---:|---:|
| Nine width-512 Q/K/V/O readers | 28,311,552 matrix weights | 54 MiB BF16 |
| All 36 width-512 readers | 113,246,208 matrix weights | 216 MiB BF16 |
| Three groups × eight fast experts × four 128×128 heads | 1,572,864 mutable elements | 6 MiB FP32 |
| Same fast geometry with 24 experts per group | 4,718,592 mutable elements | 18 MiB FP32 |
| Three full-rank 128×128 address transforms | 49,152 mutable elements | 0.1875 MiB FP32 |
| Nine sites × 16 fast heads × 256×256 | 9,437,184 mutable elements | 36 MiB FP32 |
| Separate full Q/K/V/O projections for that larger option | 377,487,360 matrix weights | 720 MiB BF16 |
| 64 conditional width-512/FFN-2048 gated MLP experts | 201,326,592 matrix weights | 384 MiB BF16 |

The last expert pool activates 6,291,456 matrix MACs per processed vector if exactly two experts execute, excluding routing and other operations. Applying it to every token, every layer or 64 event slots has different cost; there is no claim of constant whole-system latency.

At 4,096 records, two flat width-128 key scans require 1,048,576 dot-product MACs. A balanced 64-shard/top-two-shard illustration requires 49,152. This reduction excludes routing-network computation, indexing and gathers and may reduce retrieval quality; it is not a measured 21-fold speedup. At 65,536 paired records, tensors plus the metadata cap alone use **144 MiB**, exceeding the proposed 128 MiB state profile. That larger bank needs a separate larger profile or measured paging/compression, not an implied fit.

Old-plus-candidate fast-state snapshots can double the state figures before optimizer buffers. Meta-learned initial state weights are a separate resident quantity. Port estimates exclude norms/gates and every other module. Sharing a bank avoids duplicated histories; it does not eliminate repeated reader execution.

## Cross-checks that changed the plan

1. MemT's expanded expert memory uses dense routing. Sparse top-two routing is explicitly an untested extension.
2. HOPE's adaptive-projection variant and final self-modifying formulation are distinguished; the latter retains a fixed input query projection.
3. CALM needs trained composition and repeated augmenter execution in its strongest tested form. Its pinned public hook omits its stored mask from the attention call; import/reuse is not qualified.
4. Rosetta's retention table includes real regressions despite broad non-destructive language. Retention tests remain mandatory.
5. Local restart04 repaired memory replay parity. Learned memory and visual perception remain rejected; no failed run was rewritten.

## Verification scope

[Document verification](verify_research.py) checks arithmetic/configuration agreement, source hashes, snapshot integrity, important design distinctions and local Markdown links. It writes [the validation receipt](documentation_validation.json). These are documentation checks; they do not test a model, establish module interoperability or reproduce research results.

No model weights were downloaded, no external source code was executed, no native job was launched, and the protected runtime file was neither read nor imported. Existing experimental protocols, baseline configs and other tasks' files were preserved.
