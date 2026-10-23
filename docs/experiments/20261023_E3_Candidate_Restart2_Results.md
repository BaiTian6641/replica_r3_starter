# E3 filtered candidate restart: process 1 0/3, process 2 3/3

23 October 2026 UTC. Fresh three-case held-out family; protocol
`configs/e3_candidate_restart2_v1.json`; accepted E3 extension; no training.

The only menu change was declared before execution: values already contradicted
by the observed failure (below the disclosed bound) are excluded. At least two
viable repair values remain, so the host still does not choose the repair.

| Process | Cases | Completions | Notes |
|---|---:|---:|---|
| 1 | 3 | 0/3 | `read → failed run_test`; exact journal/workspace persisted |
| 2 resume | 3 | **3/3** | exact state reloaded; independent evaluator passed in all |

All resumed episodes used `set_constant → run_test → run_test → stop`; no false
success claims. Process 1: **10.7 s**, 12.12 GiB; process 2: **25.7 s**,
15.11 GiB peak. No base save, checkpoint or optimizer update.

Across the candidate-ranking experiments, the model now has **14/14 completed
episodes (12/12 held-out)** across three fresh families; same-menu random was
0/11 on the two families where it was run. This remains a narrow opt-in ranking
interface with a host-built menu, not free-form action selection. The filtered
restart is a new fresh family and was not used for tuning.
