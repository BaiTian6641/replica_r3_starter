# E2 pointer answer consumer: service path passes

23 October 2026 UTC. Bounded consumer check of the published pointer through
`MemoryAnswerService`; **no training or default change**.

## 1. Result

| Path | Dev | Held-out |
|---|---:|---:|
| Pointer copy | 3/6 | 2/10 |
| Exact-source fallback | provenance present for all 16 requests | same |

Checks:

- all pointer results used `pointer_copy`;
- all had nonempty record IDs, span hashes and source hashes;
- all exact fallback results used `exact_source`;
- pointer accuracy matches the published candidate (3/6, 2/10).

The service returns exact source text when pointer is disabled; it never
fabricates a short answer. The pointer remains opt-in and exact text remains the
default.

## 2. Fixes found by the consumer

Attempt 1 failed because `exact_text` returns an `ExactSpan` object, not a
string; the facade now extracts `.text` and binds the returned record/span/source
hashes. A second provenance inconsistency was corrected: record IDs and source
hashes are now separate fields in `PointerAnswer` and `MemoryAnswer`.

## 3. Costs

**8.0 s** profiled, **11.0 GiB** sampled peak driver, one model load, no
optimizer updates, no checkpoint save, no base copy. Raw:
[report](../../reports/e2_pointer_service_attempt02/run_report.json).
