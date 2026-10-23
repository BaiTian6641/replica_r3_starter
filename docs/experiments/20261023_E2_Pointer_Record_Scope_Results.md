# E2 pointer record scope: top lexical record raises fresh accuracy to 10/12

23 October 2026 UTC. No training. The published pointer was evaluated on a
third untouched 12-case family with whole-word candidates under two source
scopes.

| Arm | Exact held-out | Selected record |
|---|---:|---|
| All selected records | **3/12** | often distractors |
| Top lexical record only | **10/12** | 12/12 relevant record present |

The host still does not choose the span or value: lexical retrieval selects the
top record (its normal ranking), and the pointer scores whole-word spans within
that record. Pointer weights, base model and defaults are unchanged.

The two misses (`Glen`, `Jewel`) selected entity-name spans inside the correct
record, so the remaining error is within-record span/query scoring. Provenance
and exact-source fallback pass for every request.

This confirms the previous diagnosis: **source selection was the dominant fresh
accuracy bottleneck**, while whole-word candidates fixed truncation. It is a
narrower, retrieval-conditioned interface, not free selection over all records.

Costs: **8.6 s**, **11.0 GiB** sampled peak driver, no optimizer or checkpoint.
Attempt/output:
[report](../../reports/e2_pointer_scope_attempt01/run_report.json).

Decision: retain opt-in only; exact text remains the default. A future arm may
train/score within-record spans on fresh development cases, but the exposed 12
cases must not be retuned.
