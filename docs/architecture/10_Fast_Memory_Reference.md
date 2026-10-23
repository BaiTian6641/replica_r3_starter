# E4 reference fast memory: staged observation writes

23 October 2026 UTC. Status: **host reference service and opt-in native read ports implemented;
no native write/commit path, training, delayed-use or task evaluation yet.**
This increment follows the functional linear reference in `neural.py`; it is not
a reproduction of Titans, TTT, uTTT or MemT-1.0.

## 1. Ownership and schedule

```mermaid
sequenceDiagram
    participant O as Observed event
    participant H as Host fast-memory session
    participant P as Future depth ports
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
task result. Cross-session host/runtime image reload is now connected: a bridge loads an
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

## 4. Tests

`tests/test_fast_memory.py` covers staged-write invisibility, single-commit
idempotency, reset/shuffled-state controls, chunk token bounds, invalidation,
persistence round-trip and rejection of cross-branch, cross-producer and
tampered images. `tests/test_fast_memory_bridge.py` covers duplicate
observation idempotency and runtime propagation of committed read/test
observations into the next decision. These are software contracts, not learned
capability.
