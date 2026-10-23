# E4 reference fast memory: staged observation writes

23 October 2026 UTC. Status: **host reference service implemented and tested;
not attached to native depth ports, not trained, not evaluated on tasks.**
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

## 3. What is not connected

The native action pass still has no fast-memory reader. No `DecisionFrame` fast
snapshot, no depth-port read, no end-to-end write from observed tool output and
no fast-state outer-loop training is implemented. Therefore this increment does
not establish delayed native use, interference, saturation, task benefit or
cross-session behavior. Those require a separately preregistered native arm
with one chunk-start snapshot shared across ports and one deferred commit.

## 4. Tests

`tests/test_fast_memory.py` covers staged-write invisibility, single-commit
idempotency, reset/shuffled-state controls, chunk token bounds, invalidation,
persistence round-trip and rejection of cross-branch, cross-producer and
tampered images. These are software contracts, not learned capability.
