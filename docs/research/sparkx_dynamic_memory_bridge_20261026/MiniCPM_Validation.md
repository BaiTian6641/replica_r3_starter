# MiniCPM-first validation of dynamic memory and attachable modules

26 October 2026. Research design, not a launch protocol. Preserve the [existing release gates](../sparkx_2_6_product_20261025/Validation_and_Release.md), [audiovisual tests](../sparkx_av_experience_20261026/Validation_Addendum.md) and [density controls](../sparkx_port_density_20261026/Port_Density_Research_and_Plan.md). No model training or inference was executed for this report.

## 1. Starting evidence and the immediate boundary

The current repository observation is recorded by hash in the [snapshot](../../../reports/sparkx_dynamic_memory_bridge_20261026/repository_snapshot.json). It updates, rather than erases, the earlier revision 4.1 observation:

- [Causal refresh native04](../../../runs/memory_causal_refresh_20261026/native04/run_report.json) remains rejected with zero optimizer updates: 110/240 recall, 8/24 corrected and 20/60 unaffected. [Restart04](../../../runs/memory_causal_refresh_20261026/restart04/run_report.json) now reports `fresh_process_parity=true`, zero re-encoding and mechanical success. The old restart01 failure remains historical.
- The [vision final disposition](../../../reports/vision_stage_diagnostic_20261026/final_disposition.json) rejects learned quality: 6/24 audit accuracy, constant `yes` on all 24 audit answers and caption F1 zero. The completed 32 projector updates were recovered into a report without repeating training; [reload06](../../../runs/vision_stage_diagnostic_20261026/reload06/reload_report.json) reproduces outputs in a fresh process.
- Existing `MemoryAccessPort` is a read-only masked neural reader. `DecisionFrame` supplies immutable, owned evidence. Neither establishes a general module registry, online learned addressing or useful native audiovisual memory. Spark has preparation/source contracts, not the proposed native extension executor.

The immediate learned-quality work remains calibrated addressing/abstention and visual alignment diagnosis under the existing protocols. More parameters are allowed in the product design; this does not turn failed perception into a reason to launch an uncontrolled larger run.

## 2. BR0 — Interface and causality qualification

Before learned-quality comparisons, implement the smallest explicit native interface in a separate implementation task. Exercise compact CPU fixtures first, then a short admitted native qualification.

| Test | Required observation |
|---|---|
| Disabled/empty port | Matches the unmodified backbone within the declared precision tolerance; no stale state contribution |
| Gradient path | Exercised connectors, readers and intended outer-loop parameters receive gradients; frozen and inner-only state do not accidentally enter the global optimizer |
| Exactly-once writes | Repeated reads, gradient checkpointing, retries and extra port sites do not multiply commits, decay or update counters |
| Causal availability | Changing future media, future labels or later teacher-forced tokens cannot change an earlier prediction |
| Token/chunk parity | Sequential and batched implementations agree under the same declared read-before-write or boundary-update semantics; chunk approximation is named and trained consistently |
| Scope isolation | Another stream/user's memory and fast parameters cannot affect the output; deliberate wrong-scope packets are rejected |
| Swap boundary | Attach/detach invalidates dependent caches; old live graphs retain the original module/lexical mapping |
| Restart | Fresh process restores weights, `E/F/A/C`, clocks, module versions and mappings; zero hidden source re-encoding |
| Null/error handling | Missing modality, unsupported capability, bad shape/version and all-masked packets return a qualified null or explicit error, without silent substitution |

The current MiniCPM executor re-prefills and does not qualify native KV reuse. Do not claim cache efficiency from a cache field in a proposed schema. Add cached-generation parity before benchmarking that optimization.

## 3. BR1 — Establish learned addressing before expanded memory

Hold the writer, records, reader density and model checkpoint fixed. Use fresh split worlds and queries whose answers cannot be inferred from recency, name position or a repeated answer prior.

Compare: disabled memory; an exact eligible-bank reference with learned fixed queries; the current lawful causal-refresh control; fixed-query soft versus hard selection; and the same bank with feedback-adapted `A`. Gold record selection is a separately labeled upper bound, never the deployed selector.

Measure record recall@k, correction retrieval, false recall on unknown queries, final answer accuracy and unaffected-query retention. Evaluate identities unseen during training, delayed queries, paraphrases, distractors sharing surface form, conflicting dates and task switches. Separate source selection errors from failure to use a correctly retrieved value.

For the adaptive-address condition, feedback is available only after its associated prediction, or as an explicitly supplied teaching example. Log each update's sources, objective and resulting state hash. Then freeze updates and test new queries. Answer-aware updates during the held-out query are leakage, not learning.

At 1,024 records, exact search is the efficiency reference. Hierarchical addressing is admitted only after useful exact addressing. At larger banks, report candidate recall against exact top-k and end-to-end utility/latency, including shard maintenance and gathers. Routing accuracy must not be inferred from balanced expert loads.

After the full sensing-pass reference works, compare a compact native query encoder and causally available single-pass selection. Measure the entire saved sensing cost alongside any retrieval/answer loss. A production candidate cannot inherit reference quality if it receives different query representations or makes evidence available at different layers.

## 4. BR2 — Test what fast parameters add

Use the same observation stream, training budget and later queries for four core arms:

| Arm | Episodic records | Fast associative state |
|---|---|---|
| N | Disabled | Disabled |
| E | Available | Disabled |
| F | Cleared/unavailable after teaching | Available |
| EF | Available | Available |

Add EF with frozen versus adaptive addressing as a separate comparison. Test value-shuffled, wrong-expert, no-update and time-reversed controls. This identifies whether an apparent gain comes from record access, pattern compression, learned addressing or extra compute.

Compare simple delta updates, a bounded recent-event-window updater and, only if justified, a self-referential updater. Use the same state bytes and an additional equal-wall-time control. A larger algorithm may win at its natural compute; that result must be distinguished from better learning efficiency.

Train on support episodes and score later queries, including visual motifs on new objects, acoustic patterns from held-out speakers/devices and temporal rules with changed absolute timing. Retained `F` can support persistent fast-state adaptation after restart. It cannot support a durable-backbone-learning claim after `F` is cleared.

Dense versus sparse expert routing is a separate factor. Compare one, eight and expanded expert pools while holding the top-two active budget fixed in sparse arms. Include dense pools at the same capacity. Report expert collapse, missed rare patterns, load skew, active operations, state bytes and task utility. Treat MemT's dense routing as the source-grounded control; top-two dispatch is our experimental variant.

## 5. BR3 — Demonstrate actual module interchangeability

Borrow the seed/architecture/modular-plug tests from [LegoNN](https://arxiv.org/abs/2206.03318v2). Use independently trained module checkpoints rather than renaming copies of the same weights.

| Swap | What it establishes if successful |
|---|---|
| Same architecture, different seed | The interface is not tied to one latent coordinate realization |
| Different encoder or module width, qualified connector | A trained connector can preserve useful interface semantics |
| Independently trained module using the fixed interface objective | Modules can be attached without retraining the backbone for that particular swap |
| New capability with a learned connector | Capability expansion is possible with declared adaptation cost; it is not zero-training compatibility |
| Module omission or replacement with null | The model knows the capability/evidence is unavailable and preserves unaffected behavior |
| Two compatible modules together | The combination provides complementary benefit without unacceptable interference |

Compare direct shape-only attachment, a learned projection, a query-resampler bridge and an anchor-relative bridge. Include random/frozen module controls with matched connector parameters. Pretrain/align on development examples, then test unseen content and held-out module combinations. Leave one module seed and one composition out of compatibility training.

Evaluate a visual encoder swap, an audio encoder swap and a memory-module swap separately. Repeat after full-backbone fine-tuning and after `A/F` updates. A swap that succeeds before integration may fail after co-adaptation. Record whether the connector, backbone, module weights or memory producer had to be retrained; all are different compatibility claims.

Tokenizer IDs, frame coordinates and audio clocks receive explicit mismatch tests. A common output width does not authorize their reinterpretation. For an attached language model, do not reuse token IDs across vocabularies; test the alignment mechanism itself. An unavailable compatible mapping must prevent attachment.

## 6. BR4 — Joint dynamic Engram, vocabulary and learning

Retain the existing **2×2×2 memory × swappable-vocabulary × verified-learning factorial**. Freeze the selected memory implementation and bridge layout before this stage. The memory factor contains its fixed write/read/fast-state rules; the learning factor is an additional, separately specified verified durable update. This prevents counting a fast-memory write twice as two factors.

For a UAP-native alternative to the earlier LoRA control, let the learning factor update a small full-rank skill module whose initial weights, connector and allowed update budget are identical across all eight arms. The model/module remains present with updates disabled in L-off arms. The old rank-8 online-adapter variant is an optional separate comparison; its results are not silently pooled with the full-rank module.

Use fresh artificial worlds with a taught audiovisual rule, a new canonical name and a delayed query about a new instance. Examples should vary object appearance, sound source, event timing and distractors independently. Swap active vocabulary slots between teaching and testing while retaining stable lexeme identities. Evaluate equivalent base-token/byte outputs in V-off arms so token compression is not mistaken for better knowledge.

Include cases that require **both** modalities, cases solvable from only one, and cases where they conflict. Reverse event order, shift audio, swap object identity and retain captions/transcripts while altering nonverbal evidence. The answer should follow the actual evidence or calibrated uncertainty. Transcript-only and caption-only controls reveal whether native sensory representations matter.

Measure interaction effects with confidence intervals. A successful EF+V+L model alone does not show that its components reinforce one another. Add an A-adaptation on/off contrast for the selected combination after the eight-arm design; do not explode the first study into every capacity, placement and router variant simultaneously.

## 7. BR5 — Separate recall, transfer and durable learning

Use a declared reset matrix for every capability claim:

| Condition | Keep | Clear | Permitted interpretation |
|---|---|---|---|
| Experience recall | Eligible `E`, declared `F/A` | Raw source and source KV | Delayed access to a previous event |
| New-instance transfer with memory | Declared records/prototypes and `F/A` | Exact test instance; teaching prompt | Using prior experience on a novel instance |
| Fast-parameter transfer | `F`, declared `A` | `E`, teaching prototypes, raw sources, KV and teaching-derived lexical shortcuts | Useful pattern/address state stored in fast parameters |
| Durable module learning | Only explicitly updated durable module weights | All teaching-derived `E/F/A/C`, caches, replay access and lexical candidates | Consolidated module skill |
| Durable backbone learning | Explicitly updated backbone plus declared durable shared weights | Same complete teaching-state reset | Skill acquired in the durable integrated model |

Every condition freezes updates during evaluation. No teaching rule is restated in the test prompt. Persist/reload the allowed state in a fresh process. Clearing an explicit record but leaving its derived prototype, routing state or lexical definition is not a cleared-state test. Retaining `A` or a learned module must be reported as retaining parameters.

For instructed learning, distinguish recalling a supplied instruction from discovering a rule and from applying a supplied rule to unseen instances. For self-supervised learning, independently score prediction/correspondence targets; reduced entropy or increased confidence is not correctness evidence.

Recursive improvement requires at least two successive learning-policy revisions outperforming a fixed policy on fresh families at equal cumulative ingestion, adaptation, replay and verification cost. A single successful fast-state update or a better final answer does not meet that criterion.

## 8. Statistical and resource decisions

Confirm selected designs with at least three independent training seeds. Begin with 128 independent worlds per seed; fix the sample-size decision before unsealing and use the established maximum of 1,024 worlds if development uncertainty warrants it. Split natural media by source/video/speaker and resample confidence intervals at that level. Multiple questions about one clip are not independent samples.

Keep the proposed product gates: a new primary capability needs at least five percentage points over the strongest matched lawful control and an adjusted superiority interval above zero; old-task retention uses a noninferiority margin no larger than two points. These are design thresholds, not results. Density or capacity increments instead need a confirmed preregistered benefit on their target slice and an acceptable quality/cost tradeoff; they do not each need an additional five-point gain.

Measure separate ledgers for total/stored parameters, resident weight bytes, active module operations, mutable state, optimizer/snapshots, ingestion, query/prefill, decoding, adaptation and checkpoint/reload. Test cold attachment and warm repeated use. Include the native sensing pass and attached-module execution. A zero gate or top-two score is not measured sparsity unless the other branches are skipped.

The old 32 MiB mutable-state cap is the baseline profile, not a universal product restriction. Compare explicitly declared 32/64/128 MiB research profiles with complete accounting; none is admitted to a local model run by this document. Retain the existing 50 GiB driver, single MPS worker, swap and host-free limits for execution. The ≤10% warm-text overhead, ≤5% learning duty and real-time ingestion objectives remain targets; larger variants must report any misses rather than redefine the workload.

Run bounded 30-minute, 8-hour and 72-hour soaks only after useful component behavior. Track both new learning and retention while varying event order, corrections, vocabulary turnover and module availability. Growth of checkpoint history, raw media, routing indexes or optimizer buffers must appear in the ledger.

## 9. Transfer to the final Spark product

MiniCPM success admits a Spark qualification experiment, not a product pass. Recheck Spark's source/precision, hybrid attention, all-original-parameter optimizer ownership and native extension sites. Repeat the no-module baseline, causal controls, module swaps, cleared-state learning, retention and restart tests after full offline fine-tuning. Larger optional modules can then be selected from the measured quality/cost frontier; total added parameters are not the selection criterion by themselves.
