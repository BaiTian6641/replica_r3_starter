# Intrinsic memory v3 — separate from the external spine

The user's 29 October architecture feedback supersedes the combined internal-spine
wording in the prior increment. Memory is intrinsic model computation; the spine
provides typed communication with external models, sensors and actors. This restores
[the original intrinsic architecture](../../docs/research/sparkx_intrinsic_learning_spine_20261026/Architecture_and_Spine.md)
and [the state contract](../../docs/architecture/22_State_Mechanism_and_RSI_Contracts.md).
The implementation is `f471f54a34e99f8143164e00b3247827c6ba1f1c`, following provenance,
recovery and calibration implementation `b2489ba` and documentation `6900469`.
Memory-layer work is the active focus; external spine features remain preserved.

The [v4 configurable-reader candidate](../intrinsic_memory_layout_v4/README.md)
now makes all-layer access an explicit target and supports direct 2/9/36 CPU
layouts. It preserves this sealed two-site protocol and packet; native density
comparison needs separate bindings, budgets and qualification.

## Implemented separation

`extension.intrinsic_memory` now owns the decoder memory readers. Its independent
`IntrinsicMemorySpec` fixes width 256, eight slots and sites 3/7. Memory uses its own
`intrinsic_memory` decoder argument and an immutable revalidated read context.
It constructs no Packet, invokes no codec or SpineCore and keeps `use_spine=False`.
The residual attention primitive is reused; memory queries still come from current
decoder hidden states. The existing post-block residual placement is preserved.
The original Spark attention, positional behavior, gating, GELU and tied output
are unchanged. Empty external spine operation is tested through forward/backward,
optimizer updates and fresh-process continuation. External site settings can differ
from memory sites. Sites 23/31 remain a separately measured follow-up.

Installing the new path transfers memory port Parameter objects out of the legacy
SpineCore; no second trainable memory owner remains there. The unused packet-to-memory
projection is removed. Remaining external typed endpoints and compatibility APIs are
preserved. Legacy composite models can load their old full-extension checkpoint
before this explicit transfer. Intrinsic checkpoints use a v3 manifest and
`intrinsic_memory.reader.*` tensor namespace; producer representation/codec identities
remain unchanged because the contextual values did not change. A v2 load is refused
unless explicit migration is requested. `migrate_legacy(source, destination)` verifies
the source, remaps the reader namespace, preserves optimizer moments and writes a
new artifact plus migration receipt without editing the old artifact. This is
compatibility engineering, not a new native qualification.

The four contextual slots, canonical spans, producer provenance, durable validation,
exact fallback, zero-output initialization, gradient horizon and one NativeFusedAdamW
owner remain as documented in the [v2 engineering report](../contextual_memory_v2/README.md).
E/S/P/phi/C/I/K ownership stays separate. The durable hot-state authority repair and
executable-source manifest remain active and unchanged. Official FP8, existing
FP8-derived BF16 and official BF16 stay distinct; native multimodal work is deferred.

The active [Spark-only S0–S1 protocol](../../docs/experiments/20261029_Spark_Memory_S0_S1.md)
separates foundation, recall, formation, persistent learning, functional control,
nested adaptation and learner improvement. Current status is recorded once in the
[29 October evidence ledger](evidence_ledger_20261029.json). Dimensions, sites,
32 updates and acceptance margins are project hypotheses, not measured optima.

## Evidence and limits

The final clean regression exposed a stale binding: documentation commit 9645547
had also removed the intrinsic site-range check. Restoring the f471f54 check
restores the sealed executable identity and now has a test proving out-of-range
sites fail before ownership transfer. The S0–S1 packet additionally binds the
local static import closure, including shared runner and tiered-store helpers.
The earlier packet is retained under its original filename. No historical
qualification or conversion receipt was changed.


- **Clean f2c772a3: 3,371 passed, 104 skipped, 21 subtests, one warning in
  866.01 s.** No ignored checkpoints were present. v5.1 checks pass 1,165/1,165,
  final-product checks 516/516, the 135-input bundle and tracked-artifact guard
  pass. See [`verification_summary.json`](verification_summary.json).
- **59 focused tests passed in 13.27 s**, including the separation, independent site
  settings, disabled parity, gradients, invalidation, migration, readout workflow,
  provenance and baseline gates. [`focused02.log`](focused02.log). The earlier test
  failure was a fixture that installed memory twice; it is preserved in
  `focused_attempt01.log` and fixed by constructing an unconfigured fixture adapter.
- Migrating the actual v2 CPU checkpoint preserves **exact logits and exact optimizer
  continuation**. The old artifact was not edited. [`actual_v2_migration.json`](actual_v2_migration.json).
- The intrinsic CPU diagnostic removes `extension.spine` entirely. It confirms
  parity, nonzero producer/reader gradients after output projections open, unchanged
  base and exact fresh-process continuation. [`cpu_diagnostics.json`](cpu_diagnostics.json)
  records CPU RSS, time and storage costs; these are not native quality evidence.
- Integrated calibration on the **random CPU fixture** remains **0/8**, so fit, development and learning stay
  closed. Independent engineering diagnostics are retained, not used as a bypass.
  See `cpu_competence.json`, `cpu_gate_failure.json` and `cpu_integrated.log`.
- Full-hash provisioned source validation and the current execution manifest are
  in `native_readout_packet_s0_s1.json`; no model load or native run is performed by it.
- `verification_summary.json` records completed clean-checkout regressions and
  established checks. Historical tests/packets remain under contextual_memory_v2;
  they are not relabelled as v3 evidence.

The final clean empty-spine CPU diagnostic took **3.627924 s**. Parent peak RSS was
**402,849,792 bytes**, child peak RSS **401,440,768 bytes**, and artifacts measured
before the final report were **12,266,007 bytes**. Native driver allocation and
swap growth were not measured in this CPU fixture. These costs describe engineering
execution, not useful native learning or guaranteed physical residency.

The current episodic values are fixed during generation, while hidden-state queries
vary by token/site. The P wrapper remains a **fixed question-slot control**: one
available slot makes its attention weight one, so it is not persistent knowledge
queried afresh from each token's reasoning state. Its receipt now says this explicitly.
Useful readout, persistent knowledge transfer, retention, reasoning cost reduction,
learned writing and self-improvement are still unmeasured on native Spark.

## Active S0–S1 experiment and subsequent gates

Review [`native_experiment_s0_s1.json`](native_experiment_s0_s1.json) and
[`intrinsic_memory_readout_protocol_v3.json`](../../configs/intrinsic_memory_readout_protocol_v3.json).
The command uses the existing derivative under its own execution identity, the exact
code/protocol/execution-manifest bindings and fresh `runs/intrinsic_memory_v3/native_readout01`.
No external receipt exists to reuse; the packet is not authorization. The old packet
remains historical and does not authorize this changed implementation.

First pass the separate eight-case frozen-original competence stage (8/8, zero false
recall, no truncation). Then perform 32 fixed producer/reader updates, open eight
independent development worlds and compare contextual, matched selected exact text,
wrong, shuffled and absent values. Require >=6/8, zero false recall, within one of
exact text and better than each negative control. Reload in a fresh process, check
exact logits and one actual optimizer continuation. Keep phi frozen only after useful
qualification. This stage performs no P update and makes no promotion decision.

The active stage ceiling remains 1,571 forwards, 33 backwards/updates, 48 generation
requests, 1,536 generated tokens and 3,241,216 processed positions, including calibration
and continuation. The larger conservative whole-protocol reserve retains all eleven
historical controls. One exclusive native worker, 50 GiB sampled driver, 512 MiB
sustained swap growth, 15% host free, 7,200 seconds, 256 MiB live artifacts and no retries
remain enforced. No bulk download, duplicate large checkpoint or full-base export.
Directory tiers do not establish physical GPU residency; measure driver, RSS, swap,
latency and I/O independently. Prior results, raw failures and unrelated work remain.

After useful intrinsic recall, the next memory work is a bounded **provisional working
state** under C/F, separate from authoritative E/S and durable P. Generated hypotheses,
intermediate bindings and subgoals may enter provisional state without being certified
as facts. First test cross-turn formation: read existing state -> generate -> process
completed interaction -> commit -> expose next interaction. Test within-reasoning
formation separately: read chunk-start state -> compute -> accumulate -> commit
once -> expose later chunks. These schedules are proposals, not implemented writers.
Writes need sequence IDs, absolute positions, revisions, retry/checkpoint-recompute
semantics, idempotence and interrupted/restart tests. The current uncached generation loop replays the growing prefix; a naive writer
would duplicate old writes. Implement exactly-once transitions or explicit replay from
a snapshot before enabling writes. Future KV caching must bind cache entries to the
memory revision that produced them; cold recall pauses at a declared boundary.

Then compare fixed question-slot P with selected plastic experts queried by current
hidden states or genuinely query-dependent associative memory. Preserve the repaired
11-arm controls and compare learned/reset P under the same frozen reader and evidence.
Source-free generated tests remove E/S/C and answer-bearing I/auxiliary state. Distinct
teaching/future queries, independent worlds/rule families, chronological correction,
revocation, interference and delayed retention are required across fresh processes.
Score new reasoning problems, retained knowledge and total work, including teaching,
retrieval, updates/replay and generation. A recalled answer or lower local loss alone
is not that evidence. Verified experience must reduce future cost without reducing accuracy.

Keep P1 until measured capacity pressure. Evaluate the learned writer only after useful
readout, with its differentiable inner update separately declared from imperative
NativeFusedAdamW and a direct-CE reference. Repeated retention/correction/rollback and
independent candidate evaluation precede new learning levels or RSI. Preserve the prior
working policy. Approved identity is versioned outside task reward, with its bounded
active core and task state prioritized; archival history stays on slower tiers with
explicit recall. Never read/import/edit/stage the protected runtime pilot.
