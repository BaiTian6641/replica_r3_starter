# Fresh causal selectors: durable ingestion, query-local refresh and v3 scoring

The v1 diagnostic below remains historical evidence. The v2 protocol in
`configs/memory_causal_fresh_v2.json` changes the write/refresh contract and
uses `memory.acceptance.v3`. The sealed native run used
`data/memory_causal_fresh_20261026/prepared04/protocol.json` (SHA-256
`0c301903e0a512e48a882ad2f980ead5044282d253db03e4fa33a222d7a3570b`).

## v2 protocol and native execution

The fixture has seed `20261102`, 12 episodes and 24 evaluation states, with
disjoint paraphrase, distractor, correction, unknown and post-restart queries.
The 228 declared query rows expand to 456 episode/state/phase/query cases and
17,520 arm-expanded ranking rows. The first process writes 8,400 rows; the
restart process evaluates all 17,520 rows across first-process, restart and
revocation phases.

Durable ingestion is keyed by `episode_id/record_id/record_version`. Identical
version replay is rejected, a same-attempt retry returns its original outcome,
and only observation writes enter the durable ledger. Accepted refresh is a
separate query-local operation: it requires a live record with the exact
committed version, rebuilds only the current query history, and cannot write
durable state, optimizer state or decay. The shuffled arm deranges value donors
only when its own refresh is admitted.

Two guarded MPS processes completed with CPU fallback disabled and no downloads:

| Process | Evidence | Guard wall time | Peak sampled driver | Swap growth | Result |
|---|---|---:|---:|---:|---|
| First | `runs/memory_causal_fresh_20261026/native02/` | 38.18 s | 10,956,193,792 B | 0 B | completed |
| Restart | `runs/memory_causal_fresh_20261026/restart02/` | 65.64 s | 10,830,364,672 B | 0 B | completed; parity true |

The restart encoded 216 genuinely new queries, reused 24 cached query vectors,
and re-encoded zero old observations or old queries. Same-query rows were
byte-identical across the fresh process boundary. There were 72 durable
observation commits, one version per record, zero durable refresh commits and
1,056 applied query-local refreshes. The refresh controls changed all 1,056
state hashes and perturbed all 144 paired shuffled refreshes. Durable state
remained unchanged. These are mechanical and intervention results, not a
learned-selector claim.

## v3 scoring and quality outcome

Candidate quality uses only the actual `query_key_value` readout. The
`exact_key_value` arm is a separate mixture diagnostic, never an upper bound or
a component of candidate quality. Oracle target-refresh rows are a separate
gold-target diagnostic population, not a deployable policy. Runtime selection,
policy blocking, abstention and correctness remain separate fields. Missing,
malformed, duplicate or extra rows fail closed.

The acceptance config pins the prepared03 inventory at SHA-256
`5994de0da6e7f863f4a697a40a25041e760d0d9e8e5c77f565713e1bdf763a93`.
Prepared04 contains byte-identical inventory bytes; the native run's protocol
and fixture bindings are independently checked.

The immutable restart02 rankings were replayed into
`reports/memory_acceptance_20261026/causal_fresh_restart02_acceptance_v3.json`
and `receipt_manifest_v3.native02.json`. The receipt validates all 17,520 rows
and reports `decision: rejected` for quality. The native process decision is
`inconclusive` because mechanical gates pass while quality is rejected and the
full-cohort oracle raw bound is below its declared 0.85 threshold.

| Population | Raw top-1 | Accepted | Accepted-correct | Accepted precision | Coverage |
|---|---:|---:|---:|---:|---:|
| Candidate, all 456 primary groups | 176/456 | 138 | 21/384 fact groups | 15.22% | 30.26% |
| Candidate, restart-held-out | 39/80 | 13 | 2/64 fact groups | 15.38% | 16.25% |
| Oracle diagnostic, all groups | 384/456 | 217 | 132/384 fact groups | 60.83% | 47.59% |

Candidate false recall includes 10 unknown-address and three
revoked-target-other-live cases. Revocation policy acceptance is zero; the
three other-live selections are scored as wrong retrieval rather than
evaluator-forced abstention. Correction and unaffected gates fail, and unknown
false recall is 20.83%. No useful-memory or learned-selector promotion follows.

## Tests and remaining limits

The focused suites pass 33 tests: 12 fresh causal, 9 v3 acceptance and 12
unchanged v2 acceptance tests. They cover fixture leakage, durable replay and
retry identity, query-local refresh non-mutation, applied and shuffled
interventions, checkpoint binding, fresh-process rebuild, revocation/stale
enforcement, unknown handling, strict inventory and receipt replay. A
deterministic crash-injection test also simulates failed atomic writes,
checks temporary-file cleanup and preservation of the prior checkpoint, and
reloads it successfully. No SIGKILL or native crash test is claimed; native
evidence is planned fresh-process checkpoint continuation.

Raw decisions, readouts, rankings, codebooks, ledgers, refresh audits and both
checkpoints remain under the two run directories. Guard and preparation
receipts are under `reports/memory_causal_fresh_20261026/`; v3 receipts are
under `reports/memory_acceptance_20261026/`. The frozen writer remains a
rejected diagnostic artifact with zero optimizer updates.


## Historical v1 result (unchanged)

The v1 protocol used `configs/memory_causal_fresh_v1.json` and
`data/memory_causal_fresh_20261026/prepared02`. Two native processes completed
under the resource guard. Every accepted refresh was rejected as duplicate
replay, so causal and oracle equaled the full-history state. On the decisive
restart-held-out cohort the causal arm had 24/136 raw top-one query groups,
2/128 accepted-correct fact queries, 35 accepted retrievals at 5.71% precision
and 2/8 unknown false recalls; every quality gate failed. The process decision
was `inconclusive` because mechanical gates passed while quality was rejected.

Evidence remains `runs/memory_causal_fresh_20261026/native01/` and
`runs/memory_causal_fresh_20261026/restart01/` with their guard receipts under
`reports/memory_causal_fresh_20261026/`. Those v1 receipts and configs were not
modified.
