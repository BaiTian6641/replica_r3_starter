# Spark persistent learner and tiered-memory review

**28 October 2026 · reviewed HEAD 07f64ab2.** This is a repository review,
research cross-check and design revision. No native model was loaded, no model
training or dataset acquisition was launched, and no runtime memory limit was
changed. The protected runtime pilot was not read or modified.

## Assessment

Spark is the active backbone candidate. The user's positive long-context
inference experience is compatible with preserving its pretrained competence
and adding a small useful learner. The repository now has substantially better
memory lifecycle, text-path and restart engineering. It does not yet establish
native Spark memory-learning benefit, improved reasoning, longitudinal knowledge
growth or RSI.

The current direction is in
[chapter 28](../../docs/architecture/28_Spark_Persistent_Learner_and_Tiered_Memory.md),
the [expanded design configuration](../../configs/spark_persistent_learner_tiered_design_v1.json)
and [revised worker prompt](../../docs/architecture/16_Loop_Agent_Instructions.md).
The supplied feedback is identified by digest in [baseline.json](baseline.json);
its private attachment path is not needed for repository replay.

## What changed since the prior review

The four commits after b350e87e are:

| Commit | Change |
|---|---|
| 5e65f062 | Spark source/architecture research and evidence reconciliation |
| 25a5e57f | Bounded Spark text-memory milestone |
| 9b1a02d6 | Protocol binding to implementation ancestry |
| 07f64ab2 | Bound protocol and preserved negative CPU pilot |

The new memory path supports query-driven P-only access after evidence removal,
separate E/P composition, bounded selections/caches, owner-controlled snapshots,
correction invalidation, checkpoint continuation and actual-update counters.
The runner exercises fit, generated answers, chronological development, source
removal and fresh-process parity. These repairs should not be reported as
still missing merely because earlier reviews found them.

## Findings that change the next experiment

**High priority — the CPU negative result is not a Spark capability test.**
The tiny backend is random and its blocks perform no attention/token mixing.
An independent [prefix probe](prefix_probe.json) found identical final-position
logits for two different prefixes sharing their last token. Preserve 0/12 and
the useful-learning rejection, but retract the implication that real Spark
failed full eligible context. Native context/readout evaluation is still needed.
Relevant code: [tiny decoder](../../src/replica_lab/spark_text_memory_runner.py)
and its explicitly non-native factory.

**High priority — P is not optimized by response CE in the current path.**
[Response groups and response_step](../../src/replica_lab/architecture/spine_text.py)
update only extension.spine parameters. P gradients are inspected but its owner
later applies a separate projected-answer-hash target update. This is an
explicit implementation choice, not proof of task-trained local writes.
Create an actual selected-P response-CE reference, then compare a learned local
target trained through future-task loss.

**High priority — protocol tests rely on ignored model assets.**
The tracked-source snapshot yielded 138 passed, six failed and one skipped.
Every failure originates from the absent checkpoints/sparkx2_5_fp8_source
directory. The same metadata-only review case passes in the provisioned
checkout. Unit contracts should use tracked metadata fixtures/mocks; genuine
native admission must still authenticate real source files.

**High priority — resolve the native execution identity.**
The current native-FP8/MPS training path lacks admitted backend support.
The [architecture audit](../../docs/architecture/27_Spark_X2_6_4B_Architecture_Research_Review.md)
distinguishes source FP8, actual execution, official BF16 and dequantized BF16.
Serving successfully through a different backend does not qualify training
gradients, optimizer state or reload on this adapter. Prepare the real feasible
backend/control protocol explicitly before a native pilot.

**Design priority — durable tiered clusters are not implemented by the current
in-process bounded stores.** Capacity is small and the runner hardcodes two
hidden-64 experts. Add configurable profiles and shard ownership without
loosening existing lifecycle contracts. Ordered readable content, exact-source
fallback and correction-aware paging matter more than simply adding matrices.

**Storage follow-up — extensionless checkpoint binaries escape simple suffix
rules.** Four tracked fixture “state” archives total about 35.4 MB. Keep their
historical evidence; future large checkpoints should use bounded ignored artifact
storage and digest receipts. Include file size/type, temporary publish copies,
journals and all retained generations in guards. No artifact was deleted here.

## Independent verification

See [review_checks.json](review_checks.json), [raw JUnit](focused_snapshot.xml)
and [raw output](focused_snapshot_output.txt).

- Snapshot: git archive of the reviewed HEAD, with Git metadata supplied only
  for source/ancestry checks; ignored model assets were absent.
- Focused source/spine/text/protocol/retired-runner coverage: 138 passed,
  six failed, one skipped in 93.67 seconds. Failed evidence is preserved.
- Provisioned metadata-only review comparison: one pass in 0.05 seconds,
  with no model loading or native launch.
- Prefix probe: random tiny base is prefix-blind at the final position when the
  final token is fixed; current P parameter count independently equals 98,304.
- Before document edits, no-write v5.1 verification passed 690/690 checks and
  final-product verification passed 503/503. These are documentation/arithmetic
  checks, not capability or resource measurements.
- Capacity arithmetic is independently recomputable with
  python3 reports/spark_persistent_learner_review_20261028/calculate_capacity.py.
  The script allocates no model or large bank.
- Post-edit verification and publication status are recorded in
  verification_after.json. An uncommitted development manifest is not a strict
  clean-checkout publication. Historical receipts remain preserved.

The focused suite selected:
test_spine, test_spine_text, test_spark_text_memory_protocol,
test_spark_text_memory_runner, test_spark2_5_adapter, test_spark2_5,
test_dc_p1_native_runner_v2, test_dc_r1_native_runner_v2, test_dc_r1_v3_backend,
test_dc_r1_v3_runner and test_retired_runners. Use their files under tests/ with
PYTHONDONTWRITEBYTECODE=1, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1,
PYTHONPATH=src:tests and pytest -p no:cacheprovider.

## Expanded capacity and identity

The new design target P64 has 50,331,648 gated-expert parameters, 512 times
the current 98,304. P128 is a 100,663,296-parameter stretch. Start useful-learning
experiments at P1, then P4; compare larger stored banks at fixed one/two active
experts. Full P64 FP32 weights occupy 192 MiB; all-bank weights/gradients/Adam
moments would occupy 768 MiB, excluding the backbone and other modules.
A D-only local-write family has separate parameter/update arithmetic and cannot
inherit full-expert quality evidence. See [capacity_arithmetic.json](capacity_arithmetic.json).

Proposed subsystem ceilings are 2 GiB hot, 8 GiB warm and 64 GiB disk. Reserve
8 MiB inside hot for the approved identity core. Retain frequent useful memories
and rare critical records, with versioned cluster manifests and scoped recall.
Keep large history on disk and allow explicitly budgeted slow recall. These
are lazy-allocation design ceilings subordinate to the existing global guards.

Apple silicon uses shared physical CPU/GPU memory; hot/warm are logical
residency roles. This review does not claim a measured total RAM capacity or
that every proposed profile fits this computer. See
[Apple's memory architecture](https://developer.apple.com/videos/play/wwdc2021/10153/).

Replica identity follows approved, attributable source history; agent experiences,
learned methods, task hypotheses and measured capabilities remain distinct.
Identity popularity or task reward cannot silently amend approved personal
identity. Memory tier pressure must not evict its required compact core during
an admitted operation.

## Research disposition and next order

The feedback's strongest proposal is a persistent learner whose read/write
machinery is trained for later task success. Titans/uTTT/HOPE motivate different
parts of that mechanism, but the current runner has not reproduced them.
MemGPT/DiskANN inform tiering; Engram informs conditional lookup and prefetch.
None makes a larger untrained bank useful by itself. Primary references and
adaptation limits are linked individually in chapter 28.

Proceed with portable verification and explicit backend binding; durable tiers
and contextual readout in parallel; competent native context plus direct-CE
P1; a future-task-trained local writer; measured expansion and long-term
correction/retention. Add multi-clock learning and bounded RSI only with
independently measured policy improvement. Keep native multimodal training
deferred and retain the internal spine from the start.

## Documentation migration

Current direction notes were added to the v5.1 blueprint, refinement and
MiniCPM correction chapters without rewriting their historic results.
The architecture index, active handoffs, text milestone and worker prompt now
point at chapter 28. The new JSON is design-only; sealed runtime v1 is unchanged.

Prior manifests are retained as before_*.json, and the reviewed baseline file
hashes are in baseline.json. Subsequent migration records preserve original
snapshot identities while updating explicitly reviewed current-document pins.
The additive final-product bundle includes this design and review. Its strict
publication status remains pending while these edits are unstaged/untracked;
development verification must not be reported as a native or clean publication.
