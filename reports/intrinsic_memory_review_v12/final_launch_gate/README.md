# Exact clean-HEAD final launch-gate review at f29d34e1

## Verdict

**CPU engineering is qualified for the reviewed source-free contracts, and the native packet is ready for authorization review.** This is not a native capability or useful-recall claim. The origin/compound/budget first checks pass **5/5**, the failed-`_persist` reproduction passes with exact charged receipt bytes and quota, and exact committed source passes **1024 tests with zero overlays and zero skips**, including every original, updated, residual, acceptance, budget, compound and Darwin-specific reproduction.

The X3 blocker is closed: after charged persistence fails, `_persist_charged_fallback` now reserves and charges the fallback descriptor before writing it, records the observed receipt bytes and retains the typed `ContextualReadFailure`. In the independent reproduction, the failed ticket has `status=failed`, `ticket_write_calls=3`, `ticket_write_bytes_reserved=18289`, `durable_write_bytes=18289`, and the `tier_usage` component is exactly **18289 bytes**. The terminal JSON file is exactly 6190 bytes and contains the failed reason. No write bypasses quota or the native work ledger.

Reviewed integration: `f29d34e1`; components `d37f2919`, `418bb066`, `3ea5e4d5`, `f29d34e1`, density `d5aa9b43`, and Replica `adf89a52`. Packet SHA-256 `6dc96d507bee85d807657cf81187146ab226d7d0146c6abf3d4f56f9377bb6e1`; protocol `0b9b32a313af22ed3fe2d264e67a15f620ef6a23ca9e30fa362312ca8495e10c`. No production file was edited, no authorization/attempt marker was created, and no native launch/model import/load/download occurred. Neither development data file was read or parsed; only public generator code, identities, roots, hashes and exposure receipts were used.

## Disposition of prior findings

### Fixed in exact committed source

- **N1-N3:** restart budget, truncation and loaded-tokenizer identity.
- **V1-V5:** source/buffer authority, text-bound cache, selective eviction, release identity and empty active sites.
- **D1-D5:** pilot gradient reset, exact numeric exposure/types, profiler atomicity, permitted source recomputation and producer boundary.
- **R1-R4:** branch isolation, cold authority, P clock/current pointer and chunk state.
- **F1-F2, T1-T2, A1-A3, S1, C1:** privacy/outcome semantics, full manifest identity+validity and rejected-read accounting, movement/recovery, task reset, epochs, interventions, attribution and sealed prefix.
- **X0-X2:** compound renderer/oracle/structured facts, quarantine receipt-kind recovery and torn-ledger handling. `same_colour` anchors use the answer colour and unknown sides are seed-selected.
- **X3 failure-fallback accounting:** failed-ticket persistence now uses quota and observed receipt bytes; the original typed error is preserved.
- **X4-X5, X7, X9-X10, X13-X14:** cross-instance revocation/restoration and clocked caches, cold failure/process control, verified corrections, attempt/environment closure, density empty-list parity, tracked runtime module and fitting-close lifecycle.
- **O1-O10 named cases:** origin reservation/rollback, cleanup/original error handling, corrupt-backup navigation, final-receipt retry, unknown-accelerator refusal, quarantined correction/revocation kinds, P pointer restore/recovery, fallback attempted bytes and in-flight revocation reads.
- **Dual rebuild and artifacts:** outer rebuild reserves zero duplicate writes; per-world actual is 18, giving 580/579 and total 3617. Restart 16 is correct. Origin/manifest/pointer artifact allowances cover 524,288 bytes per probe and 16,384 at restart.
- **Metadata/teacher/preflight reads:** configuration/template/tokenizer, restart identity, teacher-cache hash+read, prepared splits and preflight bound-JSON reads have reservations and observed actuals. Unsupported timing inputs and receipt false flags are rejected.
- **Publication/recovery/durability:** failed P publication rolls back, generic P writes recheck revocation, P-current recovery reconciles pointers, index ownership collisions are authenticated, and full-sync primitives are used on Darwin.
- **Replica evidence:** content-addressed measurement/protocol/interface receipts and matching hashes replace stale aliases. Failed-publication records are not left readable.
- **Darwin durability:** `F_FULLFSYNC` injection leaves no final/temp file and retry succeeds; MPS fallback timing is refused.

### Remaining limitations and coverage boundaries (not launch blockers)

- No native run, native competence, useful recall, persistent learning, promotion or RSI was performed or established. Two compound families and one seed remain the declared development scope.
- The environment marker is locally deletable, and wheel coverage is `*.dist-info/RECORD` rather than every installed payload file. These are explicit operational limitations, not hidden evidence.
- Pointer/marker rollback still contains best-effort cleanup paths; a filesystem failure can leave authority uncertain and must be surfaced rather than treated as successful recovery. Current code does not convert such a failure into a healthy receipt.
- Preparation and navigation are not all backed by one transactional journal. Some README/report tables are historical, and only the canonical packet/preflight values are used for this verdict.
- No native uninterrupted-update equivalence was measured. CPU fresh-process continuation matches uninterrupted weights and optimizer moments on the tiny BF16 fixture only.

The fitting group length check remains correct because every eight-row group is evaluated separately and all groups are required.

## Exact clean evidence

### Checkout and tests

The worktree `/private/tmp/replica-independent-v12-f29d34e1-final-launch-gate` used sparse exclusions before checkout; protected module and checkpoints were absent. The density runtime module is tracked and collection succeeded without overlays. Owned tests were supplied externally: X tests `dab31abd…`, final residuals `1bdca80b…`, policy tests `5b48d926…`, budget tests `c019534a…`, origin test `1bc39e1a…`, compound test `9261c64b…`, and fallback test `d9b6bcc8…`.

Initial origin/compound/budget checks: **5 passed, 0 failed** in 2.45 s; outer 2.94 s, max RSS **375,341,056 bytes**. Final exact suite: **1024 passed, 0 failed, 0 skipped** in 293.19 s; outer 293.74 s, max RSS **924,712,960 bytes**. Fallback reproduction: **1 passed** in 2.37 s; outer 2.85 s, max RSS **391,036,928 bytes**. Receipts: `origin_compound_budget_f29d34e1.*`, `full_regression_f29d34e1.*`, and `fallback_persistence_f29d34e1.*`.

### Integrated and track workflows

The exact integrated tiny-BF16 2/9/36 workflow passed gradients, disabled/masked behavior, two-span live/durable and live/release parity, two-record cold value/logit parity, and exact fresh-process uninterrupted weights/moments with one actual update per arm. Outer 18.92 s, max RSS **796,246,016 bytes**, internal wall 16.280282042 s, artifacts **102,309,097 bytes**. Synthetic CPU engineering only.

Density profile/pilot/receipt/navigation completed with a canonical manifest-bound set. Pilot target **61,621,618.384615384 ns**, selected updates 2/1/1, all arms matched under executable tolerance. Profile 24.93 s/RSS **836,141,056 bytes**; pilot 11.78 s/RSS **943,685,632 bytes**. No density temporary-store directory remained. Metric/flag/navigation validation and unsupported timing rejection passed; no native cost or capability is established.

Replica workflow completed in 2.75 s/RSS **359,972,864 bytes**, internal 0.295904583 s, producing content-addressed measurement, protocol and interface receipts. `formation_used_new=true`, `task_improvement_requires_write=true`, `chunk_retry_same=true`. Publication/index recovery and versioned evidence passed their checks.

### Established checks

Migration strict tracked **1482/1482**, 11.44 s/RSS 31,244,288; final product **552/552**, 8.52 s/RSS 37,027,840; bundle **293 inputs, 7,657,081 bytes**, digest `5be29a21022408aca555ba77284940ff42a734d3a4e9bd64909a34e09f1970d3`, 8.37 s/RSS 27,721,728; tracked artifact guard **17,835 files**, no violations, 0.21 s/RSS 41,598,976. Earlier verdicts and historical receipts were not overwritten.

### Native public closure

Independent `--review` validated the exact packet, 62 implementation files, two source-execution files, **8,238,996,247 source bytes**, and environment closure **67 files / 450,061,586 bytes**, with code before model loading. It reports compound identity and no admission label parsing; 4.71 s/RSS **75,104,256 bytes**. Environment-only returned `environment_verified`, 12 critical files, `model_imported=false`, `native_execution=false`. This establishes static launch eligibility for authorization review, not a native attempt or learned result.

No native competence, useful recall, persistent learning, promotion or RSI is established. Exact commands and machine-readable dispositions are in `commands.md` and `verification_summary.json`.
