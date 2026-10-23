# E2 integrated context head: 12/12 and opt-in publication

23 October 2026 UTC. The saved 6-parameter context head was integrated into
`MemoryAnswerService`, validated on a new untouched 12-case family, and
published in a separate opt-in manifest with harmful rejection and rollback.

## Result

| Arm | Fresh exact |
|---|---:|
| Parent pointer, top record | **9/12** |
| Integrated pointer + context head | **12/12** |

The head fixed the remaining entity-name misses (`Onyx`, `Pearl`, `Zephyr`).
All candidate results carried record/span/source provenance. Loading verified
the pointer parent manifest, context weights hash and a strict tensor inventory;
the first diagnostic artifact's bare `weight`/`bias` names were explicitly
mapped to `linear.*` rather than silently accepted.

No training. One model load, **7.8 s**, **11.0 GiB** sampled driver peak.

## Publication

- Candidate artifact `e2-pointer-context-1`: accepted/passed; changed parameters
  `linear.weight`, `linear.bias`; parent pointer manifest checked.
- Harmful candidate: rejected.
- Separate opt-in manifest `ec173220…` published and rolled back to exact-text
  baseline `97319970…`.
- `default=False`; exact text remains the production default.

## Limits

The feature set is template-specific; the family is synthetic. The candidate is
accepted for opt-in use only, not promoted as default. Further fresh-family
evaluation and rollback/retention checks are required before any broader claim.

Evidence: [service report](../../reports/e2_pointer_context_service_attempt02/run_report.json),
[publication report](../../reports/e2_pointer_context_publication_attempt02/report.json).
