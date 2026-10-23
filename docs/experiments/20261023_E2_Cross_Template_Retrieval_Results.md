# E2 cross-template retrieval: lexical remains best, reranker fails

23 October 2026 UTC. No training. Fresh cross-template family, 12 cases.

| Arm | Top-1 | Top-4 |
|---|---:|---:|
| Lexical | **7/12** | 12/12 |
| Native dense | 5/12 | 12/12 |
| Existing opt-in reranker | **1/12** | 12/12 |

The existing reranker was trained on the original template and does not transfer
to this phrasing. It remains opt-in and is not a default. Lexical remains the
best default retrieval arm. All arms retain the relevant record within top-4,
so the cross-template context failure is primarily source ordering, not missing
evidence.

No checkpoint, optimizer update or default change. Cost: **6.6 s**, **11.0 GiB**
peak. Evidence:
[report](../../reports/e2_cross_template_retrieval_attempt01/run_report.json).
