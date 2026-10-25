# Vocabulary adaptation and efficient LoRA for Spark

**Canonical plan, 25 October:** use the [consolidated SparkX 2.6 architecture and MiniCPM5 validation plan](../../architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md) for current design decisions and joint memory/vocabulary/learning experiments. This package and its validation remain a historical research snapshot.

Research date: **25 October 2026**. This extends the [Dynamic Engram plan](../dynamic_engram_20261025/README.md) and the [Spark-X2.6 architecture plan](../sparkx_2_6_4b_20261024/README.md). The target remains a 4B-class language backbone with additional native memory, learning and vision modules.

**Recommendation:** let Spark learn new names and concepts through its native memory first, then test a bounded swappable lexical vocabulary backed by that memory. Compare a small, append-only extension for repeatedly useful permanent terms. Use compact LoRA updates for reusable skills and slower consolidation. Investigate dynamic input segmentation and byte-level conversion as separate architecture experiments after the simpler path demonstrates useful learning and retention.

The strongest near-term LoRA baseline is carefully tuned standard LoRA at ranks 8 and 16, with explicit rank scaling and a bounded learning-rate search. Compare LoRA-FA when activation memory is the limiting factor. Evaluate CARE-LoRA for longer learning batches and MatryoshkaLoRA when variable adapter rank has a measurable benefit. Recent research does not justify choosing a complex variant solely because it is newer; a 2026 controlled study found tuned vanilla LoRA competitive with the variants it examined on math and code tasks. [Learning Rate Matters](https://arxiv.org/html/2602.04998v2).

| Document | Purpose |
|---|---|
| [Dynamic vocabulary research](01_Dynamic_Vocabulary_Research.md) | Six research directions, Spark tokenizer findings, and the distinction between new spellings, new concepts and new skills |
| [Efficient LoRA research](02_Efficient_LoRA_Research.md) | Eighteen adaptation papers, including 2026 methods, their costs, evidence and suitability for persistent learning |
| [Integrated design and experiments](03_Integrated_Design_and_Experiments.md) | Native lexical memory, bounded vocabulary growth, visual grounding, learning policy, staged experiments and acceptance criteria |
| [Cost and implementation plan](04_Cost_and_Implementation.md) | Spark-specific parameter counts, activation tradeoffs, compatibility changes and hardware qualification |
| [Swappable dynamic vocabulary](05_Swappable_Dynamic_Vocabulary.md) | Four additional papers, reusable lexical slots, atomic phrase generation, identity/cache rules and capacity estimates |
| [Machine-readable proposal](../../../configs/vocabulary_lora_research_plan_v1.json) | Component limits, experiment dependencies and proposed gates; not an executable runtime configuration |
| [Evidence and reading manifest](../../../reports/vocabulary_lora_research_20261025/review.json) | Paper versions, selected reading scope, official source-code revisions and repository snapshots |

## Findings that affect the design

The pinned Spark tokenizer assigns **all 131,072 IDs**, including reserved and control entries. Its shared input/output embedding matrix contains 335,544,320 parameters, or **640 MiB at BF16**. Adding 1,024 tied rows costs **5 MiB of weights**; a deliberately conservative FP32 weight, gradient and two-Adam-moment budget is **40 MiB**, excluding activations. The output projection becomes 0.78125% wider. These are arithmetic estimates, not device measurements. [Calculated budgets](../../../reports/vocabulary_lora_research_20261025/analytical_costs.json).

A small tokenizer-only probe found four tokens for the invented name “veltraxon” and five for `q_k_v_proj`. A temporary token reduced one repeated-name example from 20 to 14 tokens, but boundary settings also changed which strings matched. This demonstrates a possible compression opportunity and a compatibility hazard; it says nothing about model understanding or speed. [Probe results](../../../reports/vocabulary_lora_research_20261025/tokenizer_probe.json).

Spark's tied output path directly reads `model.embedding.weight` and calls `F.linear`. Wrapping `lm_head` alone therefore misses that path. New vocabulary and selected-row training need explicit input/output integration. The current native copy and pointer heads also receive candidate records or spans from their caller; they are useful diagnostics, but do not yet implement autonomous lexical selection.

Learning fewer parameters does not remove the backbone forward pass, activation costs, retention evaluation or visual credit assignment. The most economical proposal uses fast memory updates frequently, and runs gradient-based skill consolidation on small verified batches at a slower cadence. New token IDs are stable artifacts; they are not recycled like cache slots.

## Recommended experiment order

1. Inventory compatibility and measure fragmentation on representative workloads.
2. Establish native lexical recall, correction and grounded naming with the original tokenizer.
3. Compare tuned LoRA with a small, fixed set of adaptation targets and LoRA-FA.
4. Test 256 swappable lexical slots and compare them with a 256-row permanent extension; study larger active sets only if quality and amortized cost justify them.
5. Combine memory, vision and skill adaptation; verify retention across repeated learning cycles.
6. Explore dynamic input compression, CARE-LoRA and variable-rank adapters independently before combining winners.

The underlying MiniCPM5 memory quality gate remains a prerequisite from the earlier plan. This follow-up adds research, a tokenizer probe and reproducible arithmetic. It does not implement vocabulary growth, install PEFT, train adapters, execute Spark or establish recursive self-improvement. [Validation scope and results](../../../reports/vocabulary_lora_research_20261025/validation.json).
