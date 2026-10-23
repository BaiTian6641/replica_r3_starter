# E2 cross-template reranker: two candidates rejected

23 October 2026 UTC. Fresh varied-template retrieval family: 8 train, 4 dev, 8
held-out cases; source groups are disjoint. Ranking only, no answer generation.

| Candidate | Updates | LR | Dev top1 | Held-out top1 | Held-out top4 |
|---|---:|---:|---:|---:|---:|
| Lexical (default) | — | — | 3/4 | **5/8** | 8/8 |
| Attempt 01 | 8 | 1e-3 | 1/4 | 4/8 | 8/8 |
| Attempt 02 | 32 | 5e-3 | 0/4 | **1/8** | 8/8 |

Attempt 01 loss fell from 11.00 to 10.65 and stayed below lexical. Attempt 02
reduced loss to 5.42 but **overfit**, driving held-out top1 to 1/8. This is a
supervision/representation failure, not under-optimization. Both candidates
were rejected, not published and not made defaults.

Lexical remains the supported default. The existing reranker remains opt-in and
still does not transfer to cross-template phrasing. All arms preserve recall@4.

No base save, no downloads, no default change. Attempt 02 cost **51.6 s** and
**11.0 GiB** peak. Evidence:
[attempt 01](../../reports/e2_cross_rerank_attempt01/run_report.json),
[attempt 02](../../reports/e2_cross_rerank_attempt02/run_report.json).
