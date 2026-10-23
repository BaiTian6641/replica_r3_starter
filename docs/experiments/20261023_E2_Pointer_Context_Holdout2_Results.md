# E2 context candidate repeats 12/12 on a second untouched family

23 October 2026 UTC. No training. The published opt-in context head was loaded
through the integrated answer service and evaluated on another new 12-case
family.

| Arm | Fresh exact |
|---|---:|
| Parent pointer, top record | **9/12** |
| Pointer + context head | **12/12** |

Parent misses were entity-name spans (`Frost`, `Gale`, `Knot`); the context head
selected the complete values. Parent/context manifests, weights hashes and
provenance checks pass for all requests.

This is a **repeat across two independent 12-case families**, but the feature
set remains template-specific and the candidate remains **opt-in only**. Exact
text is still the default; no checkpoint/default change. Cost: **7.5 s**,
**11.0 GiB** peak, no optimizer updates.

Evidence: [run report](../../reports/e2_pointer_context_holdout2_attempt01/run_report.json).
