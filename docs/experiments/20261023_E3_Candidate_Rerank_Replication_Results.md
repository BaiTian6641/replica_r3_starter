# E3 candidate reranking replication: 6/6 model vs 0/6 random

23 October 2026 UTC. Fresh 1-dev/5-held-out family, protocol
`configs/e3_candidate_rerank_replication_v1.json`, accepted E3 extension, no
training or optimizer update.

| Arm | Completions | Held-out | False success claims |
|---|---:|---:|---:|
| Model likelihood | **6/6** | **5/5** | 0 |
| Random same menu | **0/6** | 0/5 | 2 per episode |

The host enumerated the same candidate types: read, run_test, stop, and repair
values derived from the observed failure and current constants. The model scored
each candidate and the host executed the model-selected argmin.

Across the first candidate family (5/5) and this fresh replication (6/6), the
model is **11/11 episodes and 9/9 held-out versus 0/11 random**, with zero false
success claims. Costs: model **14.14 GiB** peak and 66.6 s; random **11.0 GiB**
and 7.4 s for this family.

## Limits

This remains a **narrow opt-in ranking interface**, not free-form action
selection. The menu includes the disclosed repair value, so the task is easier
than generating a repair. Both families are small and now exposed; no broader
generalization, default promotion or free-form policy claim follows. The random
control establishes that the menu alone is insufficient, but does not isolate
which model features drive ranking.
