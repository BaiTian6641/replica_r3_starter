# Causal frozen-writer refresh: bounded improvement, quality rejected

26 October 2026. The causal memory experiment kept the rejected v5 writer
frozen and used only the current query and cached observed records to choose a
refresh. Gold-target refresh remained a separately labelled oracle bound.
Protocol: `configs/memory_causal_refresh_v1.json`; implementation:
`src/replica_lab/memory_causal_refresh.py`.

## Outcome

| Arm | Query-value top1 | Exact-key top1 | Corrected | Unaffected | Unknown false recall |
|---|---:|---:|---:|---:|---:|
| Full history, no refresh | 36/240 = 15.00% | 36/240 = 15.00% | 24/24 = 100% | 0/60 = 0% | 38.10% |
| Causal query refresh | 110/240 = 45.83% | 110/240 = 45.83% | 8/24 = 33.33% | 20/60 = 33.33% | 16.67% |
| Shuffled values | 33/240 = 13.75% | 33/240 = 13.75% | 2/24 = 8.33% | 5/60 = 8.33% | 16.67% |
| Disabled | 44/240 = 18.33% | 44/240 = 18.33% | 4/24 = 16.67% | 6/60 = 10.00% | 0% |
| Gold-target oracle | 240/240 = 100% | 239/240 = 99.58% | 24/24 = 100% | 60/60 = 100% | 38.10% |
The causal policy therefore improves learned value recovery by 74 layer-readouts but
fails every required learned-quality gate: top1 <0.85, correction <0.60,
unaffected <0.70 and unknown false recall >0.10. The oracle bound passes,
confirming the recency intervention is available only when target identity is
known. Causal refresh is **hybrid query-addressed cached-observation retrieval
plus neural refresh**, not pure learned-memory success. Exact-source and
full-context controls are also hybrid retrieval/context evidence, reported
separately.

## Restart and resource evidence

Fresh-process continuation passes: decisions, readouts, rankings, codebooks and
all metrics are byte-identical, with **zero old-observation encodes**. The
restart bug was device drift: reloaded vectors initially moved computation to
CPU. Both paths now restore cached vectors to MPS before evaluation.
Checkpoint: `runs/memory_causal_refresh_20261026/native04/policy_checkpoint.pt`.

Native guard: peak sampled driver **11,040,129,024 bytes**, zero swap growth,
77% minimum host free and no stop reason. See
`reports/memory_causal_refresh_20261026/native04.guard.json` and
`restart04.guard.json`. Exact-source/full-context generation controls were
reused from the authenticated matched native03 report, not rerun: exact source
162/162, full context 160/162, memory 0/162 and context-removed 42/162.

## Disposition

**Rejected for learned-memory quality; engineering continuation accepted.**
The smallest justified next diagnostic is abstention/address calibration on a
fresh sealed set, not writer training or V3 promotion. Perception and useful
memory remain prerequisites for native visual memory and RSI.
