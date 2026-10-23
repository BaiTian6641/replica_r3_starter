# E2 pointer candidate units: truncation fixed, selection unchanged

23 October 2026 UTC. No training. On a second untouched 12-case family, token
and whole-word candidate enumeration were compared with the published pointer.

| Candidate unit | Exact held-out | Truncated copies |
|---|---:|---:|
| Token spans | **2/12** | `saff` for `saffron` |
| Whole-word spans | **2/12** | none |

Whole-word candidates remove subword truncation, but the pointer still selects
complete values from the wrong source record. Provenance and exact-source
fallback pass for all requests. Costs: **8.5 s**, **11.0 GiB** peak, no optimizer
or checkpoint.

Attempt 1 failed before evaluation because word spans crossed punctuation gaps
and exceeded `max_span`; the builder now requires contiguous token indices. The
first attempt is preserved.

**Conclusion:** candidate boundary truncation was real but not the remaining
accuracy bottleneck. The next pointer arm must change source/query scoring or
training, not just enumerate longer spans. Pointer remains opt-in; exact text is
the default.

Evidence: [run report](../../reports/e2_pointer_unit_attempt02/run_report.json).
