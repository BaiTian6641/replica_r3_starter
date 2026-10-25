# Dynamic Engram for Spark: native memory and learning

**Canonical plan, 25 October:** use the [consolidated SparkX 2.6 architecture and MiniCPM5 validation plan](../../architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md) for current design decisions and joint memory/vocabulary/learning experiments. This package and its validation remain a historical research snapshot.

Research and repository review: **25 October 2026**. This extends the [Spark-X2.6 plan](../sparkx_2_6_4b_20261024/README.md), using the current MiniCPM5 implementation and preserved experiment results. **Status: architecture proposal and experiment design. No architecture implementation, weight download or native training was performed in this review.**

**Recommendation:** develop a model-native memory subsystem with three complementary paths: a stable pattern table, an editable episodic tensor bank, and compact adaptive fast weights. Couple these to the existing recurrent core through learned readers, writers and relevance gates. Add bounded adapter generation and slower consolidation only after useful recall and retention are demonstrated. Vision supplies object, region and event representations to the same subsystem. A small appraisal controller can modulate learning and attention, subject to a measured advantage over a neutral controller.

This is a plausible way to improve a small model's persistent knowledge, adaptation and experience-dependent decisions. It does not establish unlimited memory or autonomous general improvement. The most valuable first result would be **learning a genuinely novel binding, correcting it, and using it later while unrelated answers remain correct**. The repository has not yet established that result on its broader memory protocol.

“Dynamic Engram” below is our proposed system name, not a claim that a published architecture already implements this entire combination. The confirmed scope remains a **4B-class Spark language backbone plus additional modules**, with **HOPE / Nested Learning** as the multi-timescale research direction. The personality description remains pending.

| Document | What it provides |
|---|---|
| [Research and repository findings](01_Research_and_Repository_Findings.md) | Paper-by-paper reading notes, evidence limits, current MiniCPM results and implementation gaps |
| [Native architecture](02_Native_Architecture.md) | Backbone integration, dynamic memory equations, state ownership, visual memory and functional emotion simulation |
| [Learning and evaluation plan](03_Learning_and_Evaluation.md) | Training objectives, staged experiments, proposed gates, retention and recursive-improvement criteria |
| [Compute budget and implementation map](04_Compute_and_Implementation.md) | Calculated state sizes, hidden costs, hardware qualification and concrete repository changes |
| [Machine-readable proposal](../../../configs/dynamic_engram_research_plan_v1.json) | Component contracts, dependencies, trainable scopes and proposed acceptance thresholds |
| [Evidence manifest](../../../reports/dynamic_engram_research_20261025/review.json) | Current source hashes, public-paper snapshots, official code revisions and validation scope |

The compact numerical design has a **4.19M-parameter static table**, approximately **3.92 MiB of live mutable state payload**, and about **3.41M matrix parameters in two proposed factorized fast-memory ports**. These are separate accounting categories. The state estimate includes an optional generated adapter, but excludes model weights, indices, temporary tensors, replay, optimizer buffers and allocator overhead. It is not measured device usage. [Reproducible calculations](../../../reports/dynamic_engram_research_20261025/analytical_costs.json).

The recommended order is diagnostic separation → native episodic memory → compact fast memory → grounded visual memory → persistent skill learning → learned learning policy. Static Engram optimization and appraisal are controlled branches, not prerequisites that delay reliable dynamic recall. The design deliberately keeps the architecture small enough to test each contribution independently.

The follow-up [vocabulary adaptation and efficient LoRA research](../vocabulary_lora_20261025/README.md) develops native lexical learning, bounded token growth, adapter methods and Spark-specific cost estimates. It extends this proposal without changing its capability admission gates.

The reconciled focused software check passed **99 tests in 25.60 seconds**, with one existing sparse-tensor warning. It covers SGDN-2, pattern, core, reliability and persistent-representation interfaces at repository commit `757ace1565c92f23bc418cc83286411832790e3f`. It does not test Spark execution or validate the proposed system. [JUnit](../../../reports/dynamic_engram_research_20261025/focused_tests.xml).

[Package validation](../../../reports/dynamic_engram_research_20261025/validation.json) checks local links, code fences, the ten-stage dependency graph, eleven proposed gates, source pins and recomputed cost arithmetic. Concurrent V0 edits after the test snapshot are recorded separately and are not covered by that result. The newer repair qualification report's full-suite result is attributed in the findings rather than presented as a rerun by this review.
