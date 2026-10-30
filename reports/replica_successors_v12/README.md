# Replica successors v1 — isolated CPU formation, chunks, tiers and adapters

**Status: isolated CPU engineering contracts implemented and measured. No native
model, no native memory run, no persistent-P update, no capacity expansion, no
promotion, no nested adaptation, no RSI and no personal-fidelity claim.**

The current Spark readout contract is untouched: block-input queries and
post-block residuals, sites 3/7 first, four ordered width-256 slots per span and
two spans/eight slots maximum. Frozen source features remain the only intended
neural detach besides the declared durable/inference boundaries. No optimizer,
`NativeFusedAdamW` owner or original Spark parameter is used or modified here.

This directory holds the separately versioned successor work requested by
`docs/handoffs/20261029_spark_intrinsic_memory/05_REPLICA_SUCCESSORS_PROMPT.md`.
It does not edit `CognitiveCore`, `TaskRuntime`, `WorkspaceStore`,
`IdentityArchive/IdentityStore`, `IdentityReader`, the sealed v3–v11 runtime
bytes or any contextual-memory v12 module. `src/replica_lab/runtime_pilot.py`
was never read, imported, edited or staged.

## 1. Delivered code

| Version string | Module | Responsibility |
|---|---|---|
| `replica.successors.s2a.v1` | `src/replica_lab/successors/formation.py` | completed-interaction formation: immutable read pins, observation/outcome separation, staged durable authority, exactly-once receipts, correction/revocation, crash/restart recovery |
| `replica.successors.s2b.v1` | `src/replica_lab/successors/chunks.py` | chunk-start/chunk-barrier formation: logical identity, growing-prefix watermark/digest, retry/recompute purity, branch isolation, correction; explicit live-unroll detach boundary |
| `replica.successors.tiers.v1` | `src/replica_lab/successors/tiers.py` | hot/warm/cold validity, revalidation after I/O, pending/resume, missing/partial/corrupt detection, P publication before eviction, projection/replay invalidation, measured cost |
| `replica.successors.*_adapter.v1` | `src/replica_lab/successors/adapters.py` | thin adapters to the existing core, exact task runtime, workspace store and identity interfaces; task/identity reset interventions; equal-information masks |
| `replica.successors.study.v1` | `src/replica_lab/successors/study.py` | actual CPU write-enabled/disabled later-answer comparison on fictional fixtures |
| `replica.successors.protocols.v1` | `src/replica_lab/successors/protocols.py` | prepared later protocols, interface map and unsatisfied gates |

New tests (not run by this worker; the main agent owns verification):

- `tests/test_successor_formation_v12.py`
- `tests/test_successor_chunks_v12.py`
- `tests/test_successor_tiers_v12.py`
- `tests/test_successor_adapters_v12.py`

- `tests/test_replica_successor_repairs_v1.py`

Exact test command:

```bash
.venv/bin/python -m pytest -q \
  tests/test_successor_formation_v12.py \
  tests/test_successor_chunks_v12.py \
  tests/test_successor_tiers_v12.py \
  tests/test_successor_adapters_v12.py
```

Initial coordinator verification passed 38 tests after two repair cycles. After
independent review and two further repair cycles, **73 tests passed**, including
all 16 independent reproductions. Failed attempts 01–03 remain preserved.
Review repairs covered branch lineage, immutable pins, cold authority, P clocks,
chunk pre-state, tier read contracts, workspace/epoch interventions, observed
attribution and token bounds. The final measurement workflow completed with
`formation_used_new=true`, `task_improvement_requires_write=true`,
`chunk_retry_same=true` and cold `ready`; see `review_repair_verification.json`.

Final-cycle recovery, generic-P revocation, correction/privacy policy,
authority-manifest and chunk-edge repairs added six more passing checks plus the
16 original reproductions. One reviewer-owned X6 density test retains an
intentionally obsolete premise: altered frozen tolerance is now refused. The
reviewer will invert that premise for final review; see `final_verification.json`.
The final workflow again passed all four measured success fields.

## 2. Contract behavior that is actually implemented

### S2a completed-interaction formation

- `FormationStore.pin(...)` captures generation, authority sequence, head
  digest, scope/branch/purpose/mode, source epochs, producer/representation/codec
  versions, task/decision revision, permitted uses and the cutoff **before** any
  interaction reads memory.
- Observations require a host receipt. Model text is `ProvisionalNote` only and
  is stored in a separate branch-note table; it never appears in `visible()`.
  Outcome receipts are stored separately and can be read with `outcomes(pin)`.
- `prepare()` writes a content-addressed canonical payload with `fsync` but
  publishes no authority. `commit()` performs one SQLite transaction that binds
  the predecessor receipt digest, payload digest, record digests, pin digest and
  advances generation/sequence exactly once.
- Duplicate delivery with identical bytes returns the existing receipt.
  Identical logical identity with different bytes raises `FormationConflict`,
  preserves the conflicting payload under `conflicts/`, and fails closed.
- Recovery verifies the receipt hash chain, generation/sequence progress and
  every payload digest; the record index is derived and can only be rebuilt from
  committed events. A corrupt payload is quarantined by report while unrelated
  healthy records remain readable.
- Correction creates a successor with verification rank and permitted uses at
  least equal to its parent, tombstones the parent and known in-scope
  descendants, and refuses unverified rewrites of verified evidence; host
  revocation is the explicit erasure path. Outcome receipts belonging to a
  revoked event are excluded even for an older pin. Parents/ancestors must match
  subject/branch/purpose/mode. Revocation is idempotent; cross-scope revocation needs the
  explicit `scope="host_wide", authorized_by="host"` contract.
- Correction tombstones are generation/observed-at gated, so a pin is immutable
  under a later correction. Privacy revocation is an immediate barrier and
  invalidates even an older pin. Evidence invalidated by revocation may not be
  served as ready.
- New records become visible only to pins taken after the commit. Records inside
  one event retain proposal ordinal; outcome receipts are cutoff/source-epoch
  filtered and ordered by event sequence then ordinal.
- Recovery recomputes/authenticates every derived tombstone, note, lineage,
  revocation and record-index row (including scope columns) from committed
  payload/receipt authority even when unrelated events are quarantined. Rows
  owned by a quarantined event are preserved; valid-event deletions are
  repaired. A quarantined revocation event is reconstructed fail-closed from its
  committed receipt (record/affected IDs and receipt digest), so deleting its
  derived tombstone/revocation rows cannot resurrect evidence. `derived_rebuilt`/degraded is reported rather than claiming health,
  and unrelated healthy state remains readable.

### S2b chunk formation

- The logical key binds exactly the fields in the specification; attempt IDs,
  transport retries and recomputation IDs are audit-only.
- `open_sequence` enforces the global token bound and integer token types.
- Recomputation replays the pre-chunk committed state and recorded operations,
  and reproduces the actual committed receipt digest. Recompute after `begin`
  but before the first proposal uses that chunk's pre-state, never the previous
  commit. A second `begin_chunk` while pending is refused; `invalidate` restores
  the last committed state. Exactly 8,192 committed tokens creates an explicitly
  sealed sequence that cannot start another chunk.
- A chunk starts at the committed absolute-position watermark; the prefix digest
  must match. Overlap, gaps and changed prefixes are refused.
- `functional_inner_update` is a pure tensor delta rule. `recompute()` has no
  I/O, counter, commit or session mutation.
- Retrying a committed logical key returns the prior receipt and does not
  advance the revision. A different proposal for the same key conflicts.
- A correction bumps the revision, so an outstanding chunk fails with
  `StaleChunk` and its uncommitted proposal can be discarded.
- Live gradients persist across the declared unroll; durable commits use a
  detached digest. No optimizer is used in this module.

### Tiers

- Explicit `ready`, `pending`, `unavailable`, `corrupt`, `rejected`,
  `ineligible`, `mismatch`, `not_yet_valid` and `expired` results; failures never
  return payload bytes.
- Reads require a `ReadContract` (producer, representation, scope, required
  cutoff and optional generation) or a validating authority callback. A minimal
  `{"eligible": true}` authority cannot return `ready`; a `ready` result
  requires all of producer, representation, scope_digest and a non-`None`
  cutoff (expiry is refused); `ReadContract(cutoff=None)` also returns
  `contract_required`. Attempted payload bytes are accounted separately from
  successful bytes for rejected, mismatched and corrupt reads. The contract is enforced on cache hits and after I/O;
  authority is rechecked before and after cold I/O and for fallback. Rejected,
  mismatched, expired and corrupt attempts charge `attempted_bytes` while only
  `ready`/successful bytes count in `bytes_read`; a fallback read also adds the
  failed primary attempt and its bytes. Cold
  tickets retain the callback and contract.
- Payloads and manifests are digest-bound with central schema validation;
  malformed records are quarantined by `recover()` while healthy records remain.
- Movement into warm/cold charges actual write bytes and temporary overlap.
  Dirty hot state is skipped when a clean eviction candidate exists.
- P state is clock-versioned (`p_<owner>_c<clock>`) with an atomic current
  pointer consumed and validated on read. Pointer target/digest mismatch fails
  closed; older pointers are repaired to the latest valid record; pointer
  publication failure restores the prior pointer or removes pointer/record
  coherently, preserving the original error and never leaving a live pointer to
  a deleted record. `revoke_p`
  requires explicit host authority and its durable marker is re-read by every
  instance and every publication/read/movement/eviction/recovery operation,
  invalidating all clocked owner hot/projection/replay entries. `restore_p` is
  host-only, removes the durable marker with parent-directory fsync, and clears
  owner caches so other instances can read later valid P state again. Revoke and
  restore remain durable and non-sticky under marker fsync failure tests.
- Projection cache keys include generation, producer, representation, reader
  release, site and eligibility revision. Replay eligibility is invalidated
  separately.
- Budgets are proposals (2/8/64 GiB, 8 MiB identity inside hot, 256 MiB replay
  inside warm), not allocations. The measurement run uses tiny local limits.
  Hot, warm and cold are separate accounts; no summed physical-memory total is
  reported, so a hot copy is not added to its warm file as one measurement.

### Adapters

- `CoreWorkspaceAdapter` builds one `CoreState` through the existing
  `CognitiveCore`; inference detaches, training retains gradients; the previous
  workspace is detached; `WorkspaceStore` save/load/reset is bound to task
  digest and revision; `IdentityReader` attachment is optional and null/disabled
  parity is exact.
- `TaskRuntimeFormationAdapter` pins before decisions and commits at the exact
  host boundary. Only host outcome receipts become observations; generated raw
  text becomes provisional notes. The runtime's admitted `source_epoch` is
  propagated to every observation/outcome and validated at commit. Task success
  authority remains with the runtime verifier. `run`/`finish` clear pin state in
  `finally`; `identity_then_task` and `task_then_identity` record their true
  operation order. Task reset always clears `runtime.previous_workspace` even
  without a `WorkspaceStore`.
- Interventions implement task-reset, identity-null, both orders and both-reset
  on fresh worlds. Approved identity cannot satisfy a verifier or change scope.
- `formation_authority(...)` validates formation eligibility and returns expected
  producer/representation/scope/cutoff/generation, which tier reads enforce on
  cache hits and after I/O.

## 3. Actual measurements (this machine, CPU only)

Produced by `run_cpu_successors_v12.py` in about 0.2 s on this machine.
Sample counts are small and timings vary between runs; the numbers are
engineering measurements, not benchmarks or native costs.

| Measurement | Result |
|---|---|
| Later answer, write-enabled | uses the new observation `vault code 4127` |
| Later answer, write-disabled | `unknown`; base evidence only |
| Delayed visibility | stale pin cannot see the new record |
| Removing the write | removes the improvement |
| Wrong-branch write | invisible to the other branch |
| Duplicate delivery | returned existing receipt |
| Conflicting retry | rejected and preserved |
| Crash before publication | 0 events, 1 staged orphan, nothing visible |
| Commit after restart | 1 event, duplicate returns the same receipt |
| Corrupt payload | recovery `degraded`, 1 quarantined; unrelated record readable |
| S2a commit latency / read latency | one run: p50 ≈0.35 ms / 0.26 ms; p95 ≈0.67 ms / 0.40 ms |
| S2a bytes | 17,665 written, 173,282 read across the fixture lifecycle |
| TaskRuntime enabled vs disabled | 2 visible observations vs 0; 2 provisional notes vs 0; both success `false`; 4,096-byte task journal each |
| Chunk growing prefix | watermark advanced 3 → 5 → 6; retry `duplicate` with same receipt |
| Chunk recomputation | identical digest, zero new commits |
| Chunk conflict / overlap / changed prefix / stale correction | all rejected |
| Chunk barrier latency | one run: p50 ≈0.04 ms, p95 ≈0.30 ms |
| Hot / warm / cold read | `ready` at every tier with identical digest |
| Cold pending → correction → resume | `pending` → `rejected`; no payload returned |
| Missing / corrupt | `unavailable` / `corrupt` |
| P publish/evict/reload | generation 3 restored from warm |
| Recovery | interrupted staging quarantined; 1 healthy record |
| Projection / replay invalidation | 1 cache entry cleared; replay no longer eligible |
| Canonical fallback | `fallback`, charged 12 tokens; ineligible fallback refused |
| Tiny core | 14,499 parameters (57,996 FP32 bytes); live gradient present; inference detached |
| Workspace / identity slots | 1,128 bytes / 1,024 bytes; proposed identity reservation 8 MiB |
| Identity null/disabled parity | exact workspace equality; untrained reader zero-initialized |
| Task/identity interventions | both orders clear C, null I and preserve the exact goal; no identity/verifier authority |

A tiny 14k-parameter `CognitiveCore` and fictional strings were used. These are
not Spark measurements, not memory utility and not identity fidelity evidence.

## 4. Interface map (also in `interface_map.json`)

| Existing interface | Successor adapter | Boundary preserved |
|---|---|---|
| `architecture/core.py::CognitiveCore` | `CoreWorkspaceAdapter` | one `CoreState` per decision; detached previous workspace |
| `architecture/contracts.py::CoreState` | same | identity digest binds host-supplied slots; no encoder invented |
| `runtime/loop.py::TaskRuntime` | `TaskRuntimeFormationAdapter` | pin before, host commit after; verifier authority unchanged |
| `runtime/workspace_store.py::WorkspaceStore` | `CoreWorkspaceAdapter` | workspace is an optimization; exact journal remains authority |
| `replica/identity.py::IdentityArchive/IdentityStore` | `IdentityAdapter` | approved snapshots only; never a task-success authority |
| `replica/reader.py::IdentityReader` | `CoreWorkspaceAdapter` / `IdentityAdapter` | optional, gated, zero-initialized; untrained |
| `tiered_memory.py::TieredMemoryStore` | `TieredBlobStore` | isolated generic tier validity; existing store untouched |

## 5. Prepared later protocols and unsatisfied gates

`protocols.json` contains:

- S2a write-enabled/disabled behavioral protocol with cases, metrics and exact
  command.
- S2b chunk protocol with frozen chunk-length requirement and purity cases.
- Tier validity protocol with transition, recovery, P and fallback cases.
- S3 persistent-learning protocol: 11 repaired controls plus
  hidden-state-conditioned vs fixed-question arms, independent world families,
  source-free removals, chronological acquisition/transfer/restart/interference/
  retention/correction/rollback, and separate generated-quality, false-recall,
  cost and memory metrics.
- Identity/task dissociation and equal-information protocol.
- Gate list with current status.

Unsatisfied gates (unchanged by this work):

1. Useful native intrinsic readout — v11 fitting gate failed; development was
   never opened.
2. Native 2/9/36 access, native formation utility and native chunk memory — unrun.
3. Persistent P learning — zero new updates; P1 remains the only active expert.
4. Authentic identity source intake and trained source-to-slot producer — absent;
   identity remains untrained.
5. Capacity expansion, promotion, nested clocks and RSI — closed.

## 6. Exact commands

Measurements (writes `measurements.json`, `protocols.json`, `interface_map.json`):

```bash
.venv/bin/python reports/replica_successors_v12/run_cpu_successors_v12.py \
  --output reports/replica_successors_v12
```

Component subsets:

```bash
.venv/bin/python reports/replica_successors_v12/run_cpu_successors_v12.py \
  --output reports/replica_successors_v12 --only formation
.venv/bin/python reports/replica_successors_v12/run_cpu_successors_v12.py \
  --output reports/replica_successors_v12 --only chunks
.venv/bin/python reports/replica_successors_v12/run_cpu_successors_v12.py \
  --output reports/replica_successors_v12 --only tiers
.venv/bin/python reports/replica_successors_v12/run_cpu_successors_v12.py \
  --output reports/replica_successors_v12 --only adapters
```

All commands are CPU-only, use no checkpoint, perform no download, start no
native worker and do not import `runtime_pilot`.

## 7. Preservation notes

- No sealed v3–v11 file was modified.
- No existing cognitive/task/identity module was edited.
- No native model was loaded; no P state was updated; no capability or
  personal-fidelity claim is made.
- The unrelated dirty `reports/vision_overfit_20261025/helper_tests.xml` and all
  other pre-existing untracked work were left untouched.
