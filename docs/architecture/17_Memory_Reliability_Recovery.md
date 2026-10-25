# Verification and memory reliability recovery

**Implementation update, 25 October 2026.** R1–R4 passed at `757ace1`: 140
integrated tests, a fresh-process representation replay smoke and a tracked export
of 1,372 tests +21 subtests in 166.68 s. Isolated V0 is now the next eligible
package. The v5 writer remains rejected; no native memory retraining was needed.
Spark/data preparation stays host-only and independent.

Use the [loop prompt](16_Loop_Agent_Instructions.md),
[integrated design](19_Integrated_Memory_Learning_and_RSI_Design.md),
[vision contract](18_MiniCPM5_Vision_Training_and_Continual_Learning.md) and
[data plan](../08_Dataset_Acquisition_and_Curriculum.md). Historical reproductions
below are retained; they describe the pre-repair code, not newly accepted behavior.

### Implemented repair contracts

- `verify_context` now requires independent caller scope/query/policy/limit and
  reconstructs the entire canonical selection, route, ordering and exact payload.
  `request_args` requires the authoritative reference for any supplied evidence.
- `training_schedule` pins `round_robin_seeded_shuffle.v1` with `repeat_policy=once`.
  The audit requires every ordered update, case and group; booleans cannot be
  indices/counters. Recorded v1–v5 protocols have an explicit legacy schedule
  interpretation. The untouched v5 report still audits as 48 updates /192 cases,
  128 fast /64 disabled, with 192 backward calls and no per-example gradient claim.
- Persistent encodings are `raw_observation.v1` and `fact_binding_key_value.v1`.
  Producer/cache/snapshot identities bind representation, tokenizer, parser,
  normalization, writer and engine configuration. `rebuild(..., semantics='history')`
  requires the original ordered source history; `semantics='current_view'` fetches
  an authoritative eligible view using independent scope. Both invalidate source
  publication bindings. Refresh with `bind_source(reference, scope=caller_scope)`.
- New snapshots carry `persistent_manifest.json` alongside the unchanged depth
  tensor format. Legacy snapshots require explicit `legacy_representation` and
  load read-only; `migrate_legacy` re-encodes observations and records parent hashes.
  No historical snapshot is silently relabelled or rewritten.

## 1. Current evidence and completed fixes

The [v5 writer experiment](../experiments/20261024_Memory_Reliability_Writer_Binding_Results.md)
ran one 48-update / 192-case writer-only D1 arm. Exactly 14 tensors / 5,251,072
parameters changed; 53 extension tensors were unchanged. Original backbone,
readers and gates stayed frozen. The new representation writes an entity/attribute
key and value-only content through the existing SGDN-2 path.

The [canonical eval02 verdict](../../reports/memory_reliability_v5_scorer_fix_eval02/verdict_final.json)
independently decodes 1,920 saved answers and rejects 11 of 31 gates. Fast factual
tokens differ from disabled on 128 cases, but their correct-answer sets are
identical. Influence is not useful association.

| Current v5 arm | Initial facts /64 | Corrected /16 | Unaffected /48 | Unknown /32 | General /32 | Total /192 |
|---|---:|---:|---:|---:|---:|---:|
| Fast | 0 | 0 | 0 | 32 | 24 | 56 |
| Disabled, shuffled, last-observation, adversarial | 0 | 0 | 0 | 32 | 24 | 56 |
| Ungated | 0 | 0 | 0 | 0 | 0 | 0 |
| Selective exact context | 64 | 16 | 48 | 32 | 24 | 184 |
| Full context | 64 | 16 | 48 | 31 | 17 | 176 |

Selective exact retrieval answers all 128 factual cases. Its reference bundle
still rejects: general accuracy fails the declared floor and two full-context
correct RAM answers become incorrect selective answers. Preserve the prospective
correctness/nonregression decision; do not change the scorer after inspecting
those failures. No candidate is selected or promoted.

What changed since the `988a46b` review:

| Previous issue | Current verified status |
|---|---|
| Missing scorer/builder/test dependencies | Required files are tracked; clean source export imports and focused tests pass |
| Changed consumed span accepted | Direct changed-span payload is now rejected; scope and canonical-routing defects remain below |
| Incorrect zero gate-off count | Explicit per-example routes now report 128 fast / 64 disabled; all 192 invoke backward, including 64 disabled non-fast graphs |
| Unclear gradient attribution | Audit labels accumulated-update attribution; it does not claim per-example writer gradients |
| Equality with imperfect full-context reference | v5 uses prospective correctness and correct-to-wrong nonregression checks; reference genuinely fails those checks |
| Four repeated arithmetic retention questions | v4 generator broadens to 16 distinct evaluation questions; still insufficient for broad retention claims |

Scorer corrections are `eff2079` and `5aa7e75`. Failed eval01 and the original v5
outputs remain preserved; eval02 is a no-training replay with its own pins.
The review verified all 27 eval02 implementation hashes, regenerated v4 data and
manifest byte-for-byte, ran **49 focused tests in 9.39 seconds** in an isolated
tracked export and reproduced the canonical verdict with the CPU scorer.
It inspected the candidate delta and 32 authenticated memory snapshots/source
bindings, and checked seven recorded restart answers. It did not run another
native experiment or the full suite. The experiment's separately recorded full
clean-export result is **1,311 tests +21 subtests in 148.69 seconds**.

The 32 snapshots are serialized memory states, not visual images. Seven-answer
restart parity is mechanical evidence, not answer-quality, crash durability or
unlearning evidence. Training peak driver memory was 11,815,763,968 bytes, with
zero swap growth. The saved extension is 48,948,272 bytes without a base copy.
These measurements do not establish a future visual or Spark training fit.

## 2. R1 — Bind exact context to the caller and canonical route

Affected paths: `runtime/exact_memory.py:ExactMemoryReference.verify_context`
and `scripts/run_memory_reliability.py:request_args`.

The pre-repair verifier checked a selection against its own scope; the consumer
supplied the query but not an independently expected scope. It also accepted a
syntactically valid disabled status without deriving the canonical route.

Reproductions from the review:

- A branch-A request consumes a genuine branch-B context for the same query.
- Replacing a relevant context with `status='no_evidence'`, empty records/spans
  and a nonempty invented reason passes verification.
- Changing the selection query to a general arithmetic query and supplying that
  same query to verification can retain factual spans, despite a fresh canonical
  selection for the query disabling evidence.

These are API/consumer defects, not proof of corruption in saved native results.
The previously reproduced changed-span case is fixed and must remain a regression.

Repair contract:

1. Require the consumer's expected scope and query; derive them from the actual
   request/case, never from the supplied context itself.
2. Bind the selection policy/version and limit. Recompute or equivalently verify
   canonical eligibility, coverage, conflict/abstention status and consumed span
   inventory under that caller context. Do not trust a reason string as evidence.
3. Authenticate IDs, source hashes, offsets, text, epoch, supersession/revocation,
   duplicate handling and canonical ordering at the actual consumption boundary.
4. Reject forged empty/disabled contexts as well as extra/substituted evidence.
   Keep valid no-evidence and conflict cases working under the documented policy.
5. Add focused adversarial tests through `request_args` or the production consumer,
   not only a helper test that manually passes a matching scope.

Exit: branch crossing, changed query/route/status, stale epoch, changed payload,
wrong policy/limit and substituted spans fail; authentic relevant, conflict and
unknown routes pass. New reports pin the repaired implementation. Do not rewrite
historical source pins to make old runs appear to use this contract.

## 3. R2 — Verify the complete training schedule

Affected path: `reliability_scoring.py:validate_training_routes`.
The pre-repair audit checked per-example coverage and counters but did not require
the exact unique ordered update indices. Replacing update 48 with a duplicate
of update 3 was accepted; the repaired audit rejects this mutation.

Repair contract:

- Reconstruct the declared deterministic schedule from protocol, dataset and seed.
  Require the exact sequence of update IDs and scheduled case IDs, including the
  declared repeat policy; reject missing, duplicated, substituted or reordered
  entries rather than accepting matching totals alone.
- Require integer indices excluding booleans, exact per-update example counts and
  consistent route, state-use, backward and accumulated-gradient audit records.
- Reconcile the independent schedule with both per-example rows and update
  summaries. A substituted update cannot hide behind another update's counters.
- Preserve the distinction between backward execution and nonzero gradients in
  a particular module. Gate-off may still have a non-fast differentiable graph.

Exit: mutation tests reject the duplicate-last-update reproduction, omissions,
wrong grouping/seed, boolean/out-of-range indices and inconsistent summaries;
untampered native report fixtures remain auditable under their historical policy.
A retrospective fixture or compatibility audit is not a new native execution.

## 4. R3 — Preserve the writer representation through rebuild

Affected path: `runtime/persistent_fast_memory.py:PersistentFastMemory.rebuild`.
The pre-repair v5 path constructed state with `append_fact` but rebuilt through
raw `append(record_id, text)`, changing its digest with the same producer/lineage.
The versioned implementation now preserves encoding and declares rebuild semantics.

Repair contract:

1. Give the representation an explicit name/version, such as raw observation
   encoding versus fact-binding key/value encoding. Include it in config,
   producer identity, cache keys and snapshot metadata. Bind parser/normalization
   versions and tokenizer/writer dependencies as needed.
2. Rebuild through the representation-specific encoder from permitted source
   records. Preserve event ordering, IDs, source hashes and declared correction
   semantics. Reject unknown or unsupported mixed representations.
3. Distinguish replaying the same observation history from rebuilding the current
   eligible view after correction/revocation. Those have different semantics and
   need not produce the same tensor. Declare which operation a consumer requests.
4. Preserve exact parity for an unchanged ordered input view, producer and
   representation. For a changed eligible view, verify correct source membership,
   unaffected facts and downstream behavior rather than asserting equality with
   the old incremental state.
5. Invalidate the old current-view binding after a successful rebuild and require
   a fresh authenticated binding before source-bound publication. Never relabel
   existing snapshots as if they used the new producer/schema.
6. Define explicit legacy load/migration behavior. Old artifacts either retain
   their original raw mode or require a declared re-encode; missing metadata must
   not silently select the new mode. Keep migration evidence and parent hashes.

Exit: raw and fact-binding unchanged rebuilds, save/load/rebuild, corrections,
revocations, stale binding, producer mismatch and legacy behavior are covered.
Use tiny deterministic fixtures first. A native smoke is justified only for
unresolved native semantics; no 48-update training rerun is required to prove
this software contract.

## 5. R4 — Reproducible handoff and admission to V0

Keep runtime, builders, scorers and tests in one complete source package. Verify
new code/data/protocol hashes from a clean tracked export when Git operations are
permitted, or an explicit hashed source bundle with the limitation reported.
Do not label an uncommitted bundle a committed pre-run protocol. Preserve old
runs, original scorer failures and rejected candidates.

The review's focused command is a starting point; add only tests justified by
changed boundaries and run broader integration checks when shared contracts change:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_exact_memory.py tests/test_memory_reliability_v2.py \
  tests/test_memory_reliability.py
```

**V0 admission:** R1–R3 are repaired and verified, the source/handoff package is
reproducible, and the v5 rejection remains explicit. The completed v5 experiment
satisfies the earlier requirement to finish a diagnosed memory-repair package;
it did not satisfy learned-memory quality. Proceed to isolated MiniCPM5 V0 with
memory/core/identity injections disabled. V1 training then requires V0, admitted
paired data and a prospective quality/resource protocol.

Spark metadata/backend preparation and the expanded dataset intake plan may
advance alongside these fixes without a native worker. Native Spark qualification
follows reusable interfaces and its own backend gates, not a requirement to wait
for RSI success. Neither optional path blocks MiniCPM5 V0.

## 6. Independent memory diagnosis and later integration

Continue fresh storage/addressing/readout probes on observed facts. Use explicit
copy/pointer controls only when values are recoverable. Keep whole counterfactual
families together, reserve unseen values, ask multiple questions from one state
and include independent distractors, unknowns and broader retention. Existing
v1/v3/v4 results are exposed regressions, not new audit data.

Choose one targeted hypothesis from the probe. Do not simultaneously change the
backbone, writer, reader, optimizer and curriculum. For each bounded native arm,
predeclare source/code/data/parent pins, trainable tensors, objective/masks,
optimizer, replay, controls, independent-family counts, numeric quality and
regression gates, updates/tokens/time and resource limits. Preserve D1 versus
co-adaptation scope accurately.

The learned-memory gate requires nonzero useful factual recall above disabled,
shuffled and last-observation controls, correct dated corrections, unaffected
retention, unknown handling and persistence across a fresh process without old
observation re-encoding. Exact retrieval and full context remain distinct
comparisons; neither is an infallible teacher. Set actual numeric thresholds
prospectively for each run rather than treating “better than zero” as acceptance.

Visual memory additionally requires qualified perception and frozen producer
lineage initially. Integrated neural RSI requires the relevant useful memory,
correction, retention and verifier gates. Deterministic practice/rejection and
rollback plumbing may be prepared earlier. No new accepted memory or RSI claim
follows merely from completing R1–R4 or V0.

## 7. Historical evidence and operating limits

The earlier [24-update recovery](../experiments/20261024_Memory_Reliability_Recovery_Results.md)
and [reader-only experiment](../experiments/20261024_Memory_Reliability_Reader_Only_Results.md)
remain rejected. Reader-only selective context was 192/192 versus full context
190/192 under an older equality gate. That historical decision is unchanged;
it is not the current v5 result. Earlier contextual E4 recall was 0/8, and M3/M4
retention/lookup failures remain preserved. M1's narrow procedure lifecycle is
reusable engineering evidence.

Keep one exclusive MPS worker, fallback disabled, 50 GiB driver stop and sustained
512 MiB swap-growth / 15% host-free checks over three samples. Do not alter the
allocator, kill another session's worker or infer visual/Spark capacity from a
text run. Preserve sealed source/selected weights and the protected untracked
`src/replica_lab/runtime_pilot.py`. Documentation changes do not authorize a
bulk corpus campaign, checkpoint download or native experiment by themselves.
