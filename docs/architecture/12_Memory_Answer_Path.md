# Memory answer path: opt-in pointer vs exact-source fallback

23 October 2026 UTC. `MemoryAnswerService` is a host-facing facade over the
request-owned memory frame. It does not generate text and does not authorize
success.

## Behavior

- **Pointer enabled** (explicit opt-in, copy branch present): returns the
  model-selected `PointerAnswer`, including record/span/source provenance and
  exact copied text.
- **Pointer disabled/unavailable**: returns exact source span text with record
  IDs, span hashes and source hashes. This is evidence/citation material, not a
  generated answer.
- **No eligible spans**: returns an explicit `unavailable` result.
- The service never selects a value on the model's behalf and never calls the
  LM to replace missing evidence.

The exact-text default remains unchanged. The pointer candidate was published in
a separate opt-in manifest; this facade is not an active default.

## Tests

`tests/test_memory_answer.py` covers opt-in pointer provenance and exact-source
fallback. The pointer behavioral result remains dev **3/6**, held-out **2/10**,
shuffled **0/16**.
