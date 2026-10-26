# Fresh causal selector, restart and reliability diagnostic

26 October 2026. Protocol: `configs/memory_causal_fresh_v1.json` and the refreshed
`data/memory_causal_fresh_20261026/prepared02/protocol.json` (SHA-256
`636429c89a473a65748b02ef9d055750d85bd60beaedb8b29e4116ca40dc07bf`). The
fresh fixture has 12 episodes, 24 evaluation states, 108 first-process queries,
108 genuinely new restart queries and 12 revocation controls. Five selector arms
are compared with zero optimizer updates. No generation was run.

This is a **hybrid query-addressed cached-observation retrieval plus neural
refresh** diagnostic, not a learned selector or pure learned-memory result. The
frozen rejected writer remains rejected.

## Mechanical and reliability results

Both native processes completed under the 50 GiB driver / 512 MiB swap-growth /
15% host-free guard with zero swap growth.

| Check | First process | Fresh restart |
|---|---:|---:|
| Observation encodes | 144 | 0 |
| New query encodes | 216 | 216 |
| Old observation/query re-encodes | 0 / 0 | 0 / 0 |
| Cached-history state rebuilds | 48 | 48 |
| Maximum commits per observed record | 1 | 1 |
| Accepted duplicate refresh applied | 0 / 556 attempts | 0 / 1,112 cumulative attempts |
| Same-query byte parity | n/a | passed |
| Revoked/stale accepted rows | 0 / 0 | 0 / 0 |
| Checkpoint loads | 0 | 1 |

The restart checkpoint binds protocol, fixture, source/model/tokenizer, writer,
representation, implementation and engine identity. The first checkpoint is
2,104,799 bytes; the restart checkpoint is 2,493,127 bytes. Independent review
recomputed the first checkpoint SHA-256, first-phase raw parity, ledger limits,
revocation blocking and cost counts.

## Cached-history cost

| Measurement | First process | Fresh restart |
|---|---:|---:|
| Evaluation wall time | 15.3573 s | 28.0534 s |
| Cache-load time | n/a | 0.093507 s |
| Query-encoding time | 0.371725 s | 0.347110 s |
| State-rebuild time | 1.151642 s | 1.112373 s |
| Readout time | 1.231546 s | 2.208834 s |
| Readout rows | 8,400 | 17,520 |
| Peak sampled driver | 10,947,805,184 B | 10,830,364,672 B |
| Guard wall time | 22.5982 s | 34.8277 s |

These are full cached-history rebuild costs, not an incremental-state cutover.
No incremental implementation is admitted by this measurement.

## Held-out quality: rejected

On the decisive restart-held-out cohort, the causal arm has 24/136 raw top-one
query groups (17.65%), **2/128 accepted-correct fact queries (1.56%)**, 35
accepted retrievals with 5.71% accepted precision, and 2/8 unknown false recalls
(25%). Every declared quality gate fails, including the oracle raw bound. The
oracle and causal accepted views coincide in this protocol because accepted
refresh of an already observed record is rejected as duplicate replay.

The process-level decision is therefore `inconclusive`: mechanical/restart gates
pass, but learned quality is rejected. There is no useful-memory, learned-selector,
calibration or promotion claim.

Evidence: `runs/memory_causal_fresh_20261026/native01/run_report.json`,
`runs/memory_causal_fresh_20261026/restart01/run_report.json`, their raw
`decisions.jsonl`, `readouts.jsonl`, `rankings.jsonl`, `codebooks.json` and
`ledger.json`, and `reports/memory_causal_fresh_20261026/*.guard.json`.
Preparation and focused CPU contracts are covered by
`tests/test_memory_causal_fresh.py`; the combined fresh causal/visual/stage set
passed 22 tests before native execution.
