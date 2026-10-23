# E2 context candidate on cross-template sources: retrieval confound

23 October 2026 UTC. No training. A fresh 12-case family varied source phrasing
while keeping the query semantics. **The top-record scope is not clean here:**
lexical top-1 is **7/12** (recall@4 **12/12**).

| Arm | Exact |
|---|---:|
| Parent pointer, top lexical record | **3/12** |
| Context candidate, top lexical record | **6/12** |

The context head improved within the accessible top records, but five cases had
the relevant record at lexical rank 2–3, so the top-record candidate could not
copy them. This is an **evaluation/retrieval confound**, not a clean
cross-template context result. It demonstrates that the context candidate is
template/retrieval-conditioned; it must not be used for broader template claims
or promoted by default.

The test was corrected to require the relevant record within the candidate bank
(rank<4), not lexical top-1. No model, candidate or publication state changed.
Costs: **8.1 s**, **11.0 GiB** peak, no optimizer updates.

Evidence: [run report](../../reports/e2_pointer_cross_template_attempt01/run_report.json).
