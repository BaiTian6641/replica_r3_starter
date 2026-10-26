# MiniCPM validation: reusable audiovisual experience

**26 October 2026 · Proposed follow-on validation.** Existing execution protocols and rejection gates remain binding. This addendum refines the [v3 release specification](../sparkx_2_6_product_20261025/Validation_and_Release.md), not an authorization to start native runs. No experiments described below were executed by this review.

## 1. Respect the current failed controls

At the [repository observation cut](../../../reports/sparkx_av_experience_20261026/repository_snapshot.json), frozen-writer causal refresh achieved **110/240 exact answers**, compared with 36/240 for full-history writes, 44/240 disabled and 33/240 shuffled values. It remained rejected: unaffected-query accuracy was 33.3%, unknown false recall 16.7%, and the separate fresh-process attempt reported differing outputs. There were zero optimizer updates.

This shows a useful diagnostic change, not validated native learning. The refresh arm is classified as cached-observation retrieval plus neural refresh, and the near-perfect target-last oracle knows the target. Keep both classifications visible. Address deterministic state/restart behavior before claiming persistent experience; compare an explicit bank under a preregistered protocol rather than continuing to infer a repair from the oracle.

Earlier vision audit failures also remain. A new memory mechanism must not conceal an inability to perceive the source correctly.

## 2. Separate four claims with different tests

| Claim | Required test | Essential exclusion |
|---|---|---|
| Delayed episodic recall | Teach/observe, remove original sensory tokens and KV, delay/restart, then ask about that episode | No source hidden in prompt, cache, file-selection policy or answer-aware retriever |
| Experience-assisted transfer | Teach a visual/audio pattern, then recognize new instances with eligible bank evidence available | Split instances, scenes, speakers and surfaces; reject exact-example memorization |
| Persistent parameter learning | Retain allowed updated parameters, reset fast/recurrent/event state and read caches, remove teaching records/prototypes, source access, old KV and teaching-derived lexical candidates; test fresh instances with updates disabled | Improvement must survive removal of all transient and externalized teaching state |
| Better learning policy | Compare future-task gain against a fixed policy at equal total adaptation/verification/replay cost | No test-answer access or reward based only on confidence |

Replaying the beginning of an old demonstration and asking what follows is a useful recall condition. It is not the new-instance or new-rule-family test. Report these scores separately.

## 3. MiniCPM experiment sequence

### E0: state and native integration contracts

Before quality runs, inspect module wiring and verify extension-disabled parity, gradients into exercised ports, tied/untied embedding behavior, native writer/reader execution, and exact once-only state commits under checkpoint recomputation. Restart checks must use a fresh process and compare outputs, selected record identities, state versions and subsequent writes under declared deterministic conditions.

Test vocabulary generation pinning across turns, memory links to stable lexeme IDs, producer-version mismatch, missing/revoked sources, record correction, eviction and partially completed updates. The protected runtime path remains untouched. Research reports do not alter its restrictions.

### E1: grounded and temporal reasoning before long-term learning

Compare the existing perceptual-prefix control, a compact recurrent bridge, backbone ports alone, and bridge plus ports. Include matched additional-parameter and matched-compute controls; using both may legitimately cost more and must earn that cost. Qualify uncompressed perception before testing 64-token visual resampling or pooled audio.

Use fresh synthetic geometry with independent renderings plus independently split natural clips. Test object identity across occlusion, viewpoint changes, interval order, short state changes, relational chains and evidence from separated frames. Where 3D relations are evaluated, provide valid geometry/reference data; do not score guessed depth as measured geometry.

Intervene on the relevant object/event state, unrelated state and matched random state. Replacing a required visual state should alter the answer appropriately; irrelevant replacements should have smaller effects. Also use content swaps with geometry held fixed. This tests causal evidence use beyond attractive point/box output.

### E2: paired sensory bank versus summaries

Compare semantic-only records, paired visual/semantic records, paired audio/semantic records and paired audiovisual records, with the same record count and also a matched-byte comparison. Include caption-only, transcript-only, separate unaligned banks and a richer raw-evidence ceiling. Report current-window-only performance separately.

Questions are unknown during storage. Use delayed queries after context reset, distractor episodes, corrected facts, identity swaps and unknown items. Include silent video, audio-only, time shifts, reversed order, conflicting modalities and missing evidence. Stress high-frequency events outside the sampling capacity; abstention may be the only justified answer.

Memory-only scoring must clear the original window. A second condition can restore a short query cue, clearly labeled. Every retained representation and auxiliary pool counts toward storage, including prototypes and relation edges.

### E3: learn a pattern, then use it as experience

Construct procedurally generated task families with fresh rules: shape/color relations, action sequences, acoustic motifs, or audiovisual event conjunctions. Examples include “two flashes followed by a descending tone” versus the same elements in reversed order. Include visual variation, new speakers/instruments, pitch shifts where the rule should be invariant, and pitch-sensitive rules where it should not.

Run no-update, memory-only, fixed verified-update and sham-update controls on identical teaching streams. For parameter-learning claims apply the complete reset contract above; the test prompt must not restate the taught rule. For experience-assisted transfer retain the permitted records and report that distinction. Test corrections and contradictory instruction; ensure successful transfer is not answer-label or new-name leakage.

Begin with fixed replay and bounded vanilla LoRA. Test learned replay/update selection only after this succeeds. Bridge self-supervision can use newly observed targets but needs the same held-out transfer and forgetting checks. An entropy reduction is not a pass condition.

### E4: combine memory, vocabulary and learning

Retain the existing eight-arm **M × V × L factorial**, shared fresh worlds and slot-turnover schedule. Here V means swappable vocabulary, not vision. All worlds receive the same eligible multimodal inputs. Cross-modal transfer is an outcome within those worlds; do not create an unmanageable Cartesian product of every architecture choice.

Run targeted follow-up comparisons only for mechanisms that passed E1–E3: paired bank versus semantic-only, bank versus optional fast state, and fixed versus learned adaptation. Test whether a newly named audiovisual concept survives lexical-slot reuse, correction, restart and later parameter consolidation. Do not require a statistically positive three-way interaction if the integrated system provides independently useful components without interference.

## 4. Statistics and proposed acceptance criteria

Use at least three independent training seeds for confirmation, initially 128 independent worlds per seed, and a frozen sample-size decision before unsealing. Increase up to the existing 1024-world cap if development uncertainty warrants it. Resample by world/video/speaker, not correlated question rows, and report per-seed results. Correct the declared family of comparisons with Holm or simultaneous intervals.

The thresholds below are **proposed product gates**, not published measurements:

- Delayed fact and correction accuracy at least 90%, with existing unknown-recall and unaffected-query criteria preserved.
- Each new primary reasoning/AV/transfer claim: at least five percentage points over the strongest matched lawful control, with adjusted superiority interval lower bound above zero.
- Old-task retention: noninferiority margin no larger than two percentage points; report text, vision, speech and nonverbal slices independently.
- Zero state-contract violations; all required fresh-process parity checks pass before a persistence claim.
- Existing vocabulary quality/latency gates, grounding gates and vision dependence gates remain in force.

All metrics need a preregistered direction and denominator. A weighted average cannot hide failed audio or failed correction. No new threshold retroactively changes the rejection of a historical run.

## 5. Resource and long-session qualification

Compare the complete capture→encoder→bridge→write→retrieve→backbone→output path on the same workload. Measure ingest real-time factor, queue growth, P50/P95 response delay including chunk lookahead, encoder duty, update/replay time, checkpoint I/O and peak live/device/driver memory. Real-time factor ≤1 does not mean low response latency if the model waits several seconds for a chunk.

Retain the 8192-position context candidate, 640 media-position budget per 16-second window, 32 MiB resident state, 64 MiB replay and optional 256 MiB source cache. Account for newly proposed paired records and metadata within those totals. Two ports share one bank and optional fast state; checkpoint transients and copies are additional live allocations.

Proposed gates remain ≤10% warm text P95 overhead, ≤5% learning duty cycle, no sustained backlog, and bounded state through 30-minute, 8-hour and 72-hour soaks. Counterbalancing event order and novelty is necessary; a long idle run is not evidence of useful lifetime learning. Use one MPS worker, no silent CPU fallback, the 50 GiB driver cap, swap-growth limit and host-free-memory guard. Stop native qualification when those protections require it.

MiniCPM success admits a Spark engineering trial. Spark must repeat source/backend qualification, all-original-parameter full-FT checks, causal utility, retention, restart and cost validation. It does not inherit a pass from a smaller model or from published GPU throughput.
