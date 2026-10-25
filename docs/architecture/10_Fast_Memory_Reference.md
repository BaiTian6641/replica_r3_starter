# E4 reference fast memory: staged observation writes

**Recovery qualification, 24 October 2026:** historical M2/M3 fixtures admit a
last-line shortcut; their accepted mechanics do not establish selective recall.
[Document 17](17_Memory_Reliability_Recovery.md) adds counterfactual multi-query
episodes, authenticated reload without source replay, and a correction appended
to the existing state. `PersistentFastMemory` integrates the native per-depth
writers and existing depth-image format in this diagnostic; it does not replace
the production ledger/revocation service. Latest-value correction quality and
unaffected-fact retention require measured acceptance. A correction append does
not erase the old fact and is not a revocation/unlearning implementation. The
default invalidation/rebuild policy below remains authoritative for revocation.

24 October 2026 UTC. Status: **M2 learned delayed-use and M3 shared/private
depth ownership accepted on exposed synthetic episodes. Fresh-process image
restart passes. M3 retention replay fails 0/2 and remains an active blocker;
M4 routing and production private dispatch are not promoted.**
This increment follows the functional linear reference in `neural.py`; it is not
a reproduction of Titans, TTT, uTTT or MemT-1.0.

## 1. Ownership and schedule

```mermaid
sequenceDiagram
    participant O as Observed event
    participant H as Host fast-memory session
    participant P as Native read ports
    participant Q as Later query
    O->>H: stage_observation(event_id, source_hash, keys, values)
    Note over H: writes invisible; one chunk-start state
    H->>H: commit() applies all staged writes once
    H-->>P: committed snapshot (one owner/scope)
    P->>Q: read from the same snapshot
    H->>H: reset/invalidate on correction or revocation
```

`FastMemorySession` is owned by subject/branch/purpose/mode/epoch and a producer
identity. `stage_observation` accepts only host-observed keys/values and a source
hash; it never accepts generated text as source testimony. Writes remain
invisible until `commit`, which evaluates every staged write against **one**
chunk-start state and advances the chunk index once. Duplicate observation IDs
with identical bytes are idempotent; different bytes are rejected. Empty commits
are no-ops. This is the host contract a future shared-depth port layer must use.

## 2. State, bounds and persistence

State is one tensor `[1, heads, key_size, value_size]` plus chunk index, update
count, cursor, lineage and generation. The reference config uses four heads of
64×64 FP32 (64 KiB), a 256-token chunk write budget, 4096 chunks, and a 1 MiB
image quota. Bounds are enforced before mutation; no write can grow state
silently. `read` normalizes queries and concatenates the per-head associative
reads.

`save` writes a safetensors state plus a canonical metadata sidecar binding
format/schema, owner scope, producer, config, weights hash, state digest,
chunk/cursor/lineage and generation. `load` accepts only the exact owner, scope,
config and producer and verifies bytes/digest; a mismatched branch, producer or
tampered file is rejected. `reset` and `invalidate` discard state and pending
writes and increment a generation. A source correction/revocation must rebuild
from permitted observations; this is not a claim of neural unlearning.

## 3. Native read-port qualification

`FastMemoryAccessPort` is now attached at blocks 13/27 when the manifest
declares `FastMemorySpec`; `DecisionFrame.fast` carries one immutable committed
snapshot and the branch has an independent gate. A bounded native MPS check
(`configs/e4_fast_read_v1.json`) passed: zero-state fast branch **exactly equals**
native, branch-off exactly equals native, and a nonzero random state changes
logits by 0.269. Peak sampled driver was 11.0 GiB; no update or checkpoint was
saved. This qualifies read wiring only.

An optional host bridge now connects actual `read_file`/`run_test` observations
to the session: it stages one observation, commits once, and passes the committed
state to the next `ProposalRequest`; `SessionProposer` converts it to a
`FastMemoryRead` only when the adapter declares fast ports. Verification-only
receipts are excluded. This path is tested with a deterministic, **untrained**
token encoder; it proves scheduling and state ownership, not semantic usefulness.

Still not connected: a learned semantic encoder and a successful delayed-use
task result in the operating bridge. Separate differentiable learned-writer
experiments have run and failed; this does not mean no writer was trained.
Cross-session host/runtime image reload is now connected: a bridge loads an
exact owner/scope/producer image into a fresh session, reconstructs its
observation count, rejects cross-branch/tampered images, and can seed the next
runtime decision from the restored snapshot. `TaskRuntime.invalidate_source`
records a host-authority correction note, clears dependent workspace, and resets
live fast state when the source hash is in the committed lineage; unmatching
sources leave fast state intact. Persisted images remain historical evidence:
after a correction, rebuild/replay permitted observations instead of treating an
old image as current. `FastMemoryBridge.replay` and
`TaskRuntime.replay_fast_observations` are the explicit bounded replay path:
they reset state, accept only still-permitted `(id, text, source_hash)` items,
clear dependent workspace, and journal a host correction note. The deterministic
encoder remains an evidence/scaffolding path, not a learned capability.

The [single-page experiment](../experiments/20261023_E4_Delayed_Use_Results.md)
already differentiates later answer loss through `LinearFastMemory.update_chunk`
and a learned write projection. Its script repeats one written matrix into four
heads. The host reference likewise starts heads identically and applies the
same keys/values/update rule to each. This geometry does not provide independent
head-specific written capacity. The
[contextual-writer follow-up](../experiments/20261023_E4_Contextual_Writer_Results.md)
also failed, so contextual features alone are not the next hypothesis.

`stage_observation` and runtime snapshots deliberately detach tensors. Preserve
that deployment lifetime while adding a pure functional, bounded multi-chunk
training unroll with head-specific projections. Do not remove runtime detaches
globally or publish from inside checkpointed forwards. The exact proposed
SGDN-2 update, chronology, controls and separately specified shared-depth reducer
are in [M2/M3](15_Research_Crosscheck_and_Backbone_Plan.md#5-minicpm5-construction-details).

## 4. M2/M3 learned depth evidence

M2 established useful shared-value delayed memory after source removal: 4/4 dev
and 4/4 held-out, above linear and all null controls. Candidate retention is
0/2 (`amber`) versus parent 2/2, an ungated training-recipe failure. The
unique-value candidate memorized training values and was rejected. See
[M2 results](../experiments/20261024_M2_Dynamic_Memory_Results.md).

M3 adds independent SGDN-2 write encoders at blocks 13/27. The shared arm uses
`SGDN2Memory.reduce_shared_chunk`: each depth proposes from the same chunk-start
state and one equal-weight convex mean publishes after the chunk. Private keeps
one state per depth. Both received identical source pages and had matched
writer/reader parameters; private stores 131072 bytes versus shared 65536, so
the comparison is not capacity matched and is not uTTT.

`FastMemorySpec` omits default shared-depth fields to preserve legacy manifests.
`FastMemoryRead.private_weights` carries an immutable layer map and the native
frame rejects private/shared payload mismatches. `save_fast_depth_image` and
`load_fast_depth_image` bind every layer tensor, owner/scope/config, stable
component hash, chunk/cursor/lineage and generation. The stable
`fast_version_key` permits fresh-process validation while process-local
`fast_producer_key` continues to bind decision frames.

Both M3 candidates scored 8/8 dev+held-out, passed update-two historical gradient
checks and reloaded correctly in fresh processes. Both also scored **0/2** on
retention replay versus parent 2/2. The interface result is accepted; retention
is not. See [M3 results](../experiments/20261024_M3_Depth_Ownership_Results.md).
Private runtime dispatch remains diagnostic only.

## 5. Tests

`tests/test_fast_memory.py` covers staged-write invisibility, single-commit
idempotency, reset/shuffled-state controls, chunk token bounds, invalidation,
persistence round-trip and rejection of cross-branch, cross-producer and
tampered images. `tests/test_fast_memory_bridge.py` covers duplicate
observation idempotency and runtime propagation of committed read/test
observations into the next decision. These are software contracts, not learned
capability.
