# Memory reliability recovery and bounded fine-tuning

24 October 2026. This sequence supersedes older next-package instructions after
the review of `280d4dd`. Preserve the M0–M4 records and rejected candidates.
The user authorized documentation, benchmark/scorer repair, persistent-memory
work and subsequent bounded fine-tuning. No RSI promotion is implied.

**Executed outcome:** [the first recovery pilot](../experiments/20261024_Memory_Reliability_Recovery_Results.md)
completed 24 MiniCPM5 D1 updates and was rejected. Benchmark/scorer checks,
authenticated persistent images and fresh-process parity pass. Recall is 0/16,
corrected facts 0/4, unaffected facts 0/12 and unrelated answers 0/8 versus the
disabled base's 8/8. Full-context control scores 48/48. The next unresolved gate
is selective useful memory with noninterference; do not repeat this recipe or
advance RSI because the mechanical checks passed.

**Final reader-only outcome:** [the experiment record](../experiments/20261024_Memory_Reliability_Reader_Only_Results.md)
and [canonical verdict](../../reports/memory_reliability_v2_attempt03/verdict_final_v2.json)
complete the comparison. Selective exact context passes 192/192, but full context
is 190/192 because two absent-fact answers are orange/green rather than unknown;
the declared parity gate therefore rejects the reference bundle. The 48-update
reader-only D1 candidate scores 64/192 (32 unknown plus 32 general), with all
fact/correction/unaffected floors and delayed-above-controls failing. All 32
images authenticate and fresh-process restart passes 7/7. Protocol v4 is at
`9a43bad`; the finalizer separates native and smoke-fixture provenance and records
the gate-off counter erratum (64 independently derived versus 0 reported).
The next discriminating action is per-fact writer/state binding diagnosis, not
more updates, KL increases, model changes or neural RSI promotion.

## Current evidence

M4 v7 completed 20 updates but direct pattern scored 0/4 development and 0/8
held-out/regression. All 20 delayed routes used fast memory. Fast+pattern
adversarial accuracy was 0/4 against disabled 2/4. Same-process extension/image
reload and the collision audit passed. The rejection remains unchanged.

The review independently decoded all 427 answers and reconciled their metrics.
However, the scorer trusted cached aggregates: four changed aggregate values
could reverse its gates without changing answers. A last-line extractor scores
32/32 on M2 shared and 44/44 on M3 delayed data. Those exposed fixtures measure
information transport, not established query-selective memory. Existing
correction checks rebuild a fresh state and do not establish incremental edits.

## Immediate execution order

1. **Benchmark and independent scorer.** Construct new counterfactual episode
   pairs with identical queries and different facts. Split complete families
   before expansion. Randomize fact order; distractor values must be independent
   of answers. Query several attributes from the same state, include absent
   facts and unseen values, and separate training/dev/audit families. Audit
   first/last-value and query-only shortcuts. Decode raw output token IDs with
   the sealed tokenizer, derive gold from the dataset, enforce exact case/arm
   membership and reject duplicate, missing, changed-target or inconsistent
   metric reports. Cached `correct` flags cannot authorize acceptance.
2. **Persistent-memory mechanics.** Reuse SGDN-2 and the existing image format.
   Append observations to one functional state; read it for multiple queries.
   Persist and reload that state without replaying observations. Append a
   correction to the loaded state and test both the corrected and unaffected
   facts. Bind scope, source lineage, counters and producer versions; reject
   wrong-scope/stale images. Test mechanics offline before native work.
3. **One bounded MiniCPM5 fine-tuning pilot.** Freeze the sealed base and train
   declared fast-memory writer/reader components (D1, not full tuning). Use a
   fresh hashed protocol, fixed steps, fixed train-only practice and retention
   supervision. Record parent and disabled controls. Evaluate delayed recall,
   incremental correction, unaffected-fact retention, unknowns and unrelated
   questions. Include a base-preservation objective as a separately declared
   hypothesis; do not repeat the exposed M3 replay recipe. Save an isolated
   candidate regardless of quality, then independently accept/reject/defer.
4. **Fresh-process continuation.** Reload the exact candidate and persisted
   corrected image in a new guarded process. Reproduce multiple answers and
   state/lineage before claiming durable useful memory. A quality rejection
   may still support a separately labelled mechanical restart diagnostic.
5. **Bounded improvement only after reliability.** One verified practice →
   isolated update → fresh-task/retention comparison → publish/reject → restart
   cycle precedes longitudinal RSI. Then compare T0/T1/T2/T3 with matched
   exposure and compute. C13's 10/10 collection threshold gates a learned
   selector, not deterministic plumbing or fixed verified practice.

Static Engram is an optional comparison after these gates, not a prerequisite
for every fixed-baseline loop. Test stable acquired knowledge/rules with unseen
query forms separately from new episode facts. Hash coverage is not evidence
that a fact is known. Routing may consume eligible runtime evidence and query
features, never split/role/gold labels; swapping audit metadata must not change
the route. Do not lower coverage thresholds to manufacture a successful arm.

## Acceptance and experiment discipline

Benchmark/scorer integrity is a prerequisite for training admission. Mechanical
persistence checks do not imply memory quality. Quality promotion requires
delayed gain over disabled/shuffled controls, correct incremental updates with
unaffected-fact preservation, and no regression on unrelated development AND
audit questions against the disabled baseline and parent. Report exact-answer
format failures separately from semantic diagnostics without changing gates
after observing results. Any failed gate keeps the candidate diagnostic only.

Keep protocol/data/code hashes and timestamped pre-run evidence; record whether
Git committed them. A hashed pre-run file is not a preregistered Git commit.
Preserve prior attempts. Do not repeatedly tune on exposed audits. New native
work uses one exclusive MPS worker, fallback disabled, 50 GiB sampled driver,
512 MiB sustained swap growth and 15% minimum host-free limits. Preserve the
protected `src/replica_lab/runtime_pilot.py` and all unrelated user changes.

MiniCPM5 remains baseline; Spark BF16 is optional and separately qualified.
MiniCPM-V needs its native processor/adapter before visual memory training.
Authentic personality fidelity requires permitted source records and human
review; fictional fixtures remain engineering evidence. Learned multi-rate
memory and architecture-changing RSI follow simpler reliable adaptation.

## Required handoff

Record changed files, exact data/protocol/parent hashes, software checks, actual
updates and trainable groups, decoded results by split and control, persistent
correction/restart evidence, retention, resources, candidate disposition and
the next discriminating hypothesis. Never describe a completed rejected run
as a successful memory or RSI milestone.

## Implemented recovery tools and declared pilot

- `scripts/make_memory_reliability_data.py` generates 12 counterfactual episodes
  across six disjoint families. The fixed v1 diagnostic contains 96 training
  and 48 evaluation cases; each evaluation split has only one paired family.
  First-value and last-value shortcuts score 11/48 and 9/48; the best query-only
  answer over each counterfactual pair is capped at 24/48. Crimson and indigo
  are absent from training. This is a small exposed diagnostic, not broad
  generalization evidence.
- `src/replica_lab/answer_audit.py` checks dataset-derived gold, decoded tokens,
  exact memberships, correctness and cached aggregates. The historical M4
  replay verified 434 nested answer rows including duplicate restart references
  and six external controls; this does not change its 427-generation count.
  Aggregate tampering and deleting an arm together with its metric are rejected.
  The existing historical v7 protocol still pins its old implementation; do not
  edit it to match today's scorer or overwrite the old verdict.
- `src/replica_lab/runtime/persistent_fast_memory.py` reuses the production
  per-depth writers, convex shared SGDN-2 reducer and authenticated depth-image
  format. Reload preserves scope, producer, exact lineage and counters without
  encoding old pages. Appending a correction is a latest-value experiment,
  **not deletion, revocation or neural unlearning**; those still require the
  existing permitted-source invalidation/rebuild policy.
- [The v1 protocol](../../configs/memory_reliability_v1.json) fixes 24 updates,
  four examples/update, a seeded single pass over training cases, learning rate
  0.001 and base-preservation KL weight 1.0 on train-only general questions.
  Only fast-memory writers/readers train; the sealed MiniCPM5 base stays frozen.
  One training job and one mechanical fresh-process check are declared.
  Parent/disabled/shuffled/last-chunk/full-context controls and independent
  image/answer scoring govern rejection. No selected default is replaced.

The [pre-run seal](../../reports/memory_reliability_prerun_20261024.json) records
code/data/protocol hashes before launch. The protocol is not committed; its
Git base is `280d4dd`. Use the resulting experiment report for actual outcome,
not this pre-run declaration.
