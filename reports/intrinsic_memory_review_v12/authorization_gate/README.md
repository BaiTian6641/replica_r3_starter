# Exact clean-HEAD authorization-gate review at d5aa9b43

## Verdict

**CPU engineering remains unqualified and the native packet is not ready.** The origin/compound/budget first checks pass **5/5**, and exact committed source passes **1023 tests with zero overlays and zero skips**, including every original, updated, residual, acceptance, budget, compound and Darwin-specific reproduction. Dual rebuild charging now has a single owner: the outer probe event reserves zero duplicate writes, so 580/579 and restart 16 are correct. Compound renderer, structured facts and unknown sides are consistent in public code.

The remaining authorization blocker is accounting, not test coverage: a charged persistence failure still falls back to an uncharged physical `atomic_json` ticket write. Other metadata/preflight reads are now reserved and observed, but this fallback bypasses store quota and the native work ledger. Full readiness requires every executed read/write to be charged, so the packet cannot be authorized.

Reviewed integration: `d5aa9b432ef8b457650be88d0c309accf2e8f60f`; components `d37f2919`, `3ea5e4d5`, `58e558ac`, density `f42ee79b`, and Replica `adf89a52`. Packet SHA-256 `824d1171d58b31e9fc530f3cd663c1c12e1de2877e7b2ca6864a39120df94ecd`; protocol `2564fb5b795101c494845cff37f79b03c7cc833cd32915759484eb299dde03e4`; implementation `3779b497117c90e0fcfa26e4d77888b04d8ecc8d79a723653e00c06f9eac39e7`. No production file was edited, no authorization/attempt marker was created, and no native launch/model import/load/download occurred. Neither development data file was read or parsed; only public generator code, identities, roots, hashes and exposure receipts were used.

## Disposition of prior findings

### Fixed in exact committed source

- **N1-N3:** restart budget, truncation and loaded-tokenizer identity.
- **V1-V5:** source/buffer authority, text-bound cache, selective eviction, release identity and empty active sites.
- **D1-D5:** pilot gradient reset, exact numeric exposure/types, profiler atomicity, permitted source recomputation and producer boundary.
- **R1-R4:** branch isolation, cold authority, P clock/current pointer and chunk state.
- **F1-F2, T1-T2, A1-A3, S1, C1:** privacy/outcome semantics, full manifest identity+validity and rejected-read accounting, movement/recovery, task reset, epochs, interventions, attribution and sealed prefix.
- **X0-X2:** compound renderer/oracle/structured facts, quarantine receipt-kind recovery and torn-ledger handling. `same_colour` anchors use the answer colour and unknown sides are seed-selected rather than hardcoded.
- **X4-X5, X7, X9-X10, X13-X14:** cross-instance revocation/restoration and clocked caches, cold failure/process control, verified corrections, attempt/environment closure, density empty-list parity, tracked runtime module and fitting-close lifecycle.
- **O1-O10 named cases:** origin reservation/rollback, cleanup/original error handling, corrupt-backup navigation, final-receipt retry, unknown-accelerator refusal, quarantined correction/revocation kinds, P pointer restore/recovery, fallback attempted bytes and in-flight revocation reads.
- **Dual rebuild and artifacts:** outer rebuild reserves zero duplicate writes; per-world actual is 18, giving 580/579 and total 3617. Restart 16 is correct. Origin/manifest/pointer artifact allowances cover 524,288 bytes per probe and 16,384 at restart.
- **Metadata/teacher/preflight reads:** configuration/template/tokenizer, restart identity, teacher-cache hash+read, prepared splits and preflight bound-JSON reads now have reservations and observed actuals. Unsupported timing inputs and receipt false flags are rejected.
- **Publication/recovery/durability:** failed P publication rolls back, generic P writes recheck revocation, P-current recovery reconciles pointers, index ownership collisions are authenticated, and full-sync primitives are used on Darwin.
- **Replica evidence:** content-addressed measurement/protocol/interface receipts and matching hashes replace stale aliases. Failed-publication records are not left readable.
- **Darwin durability:** `F_FULLFSYNC` injection leaves no final/temp file and retry succeeds; MPS fallback timing is refused.

### Open or partial

| Finding | Current defect |
|---|---|
| **X3 failure-fallback charging** | `contextual_memory_v12.py:528-534` still calls `atomic_json(self._ticket_path(...), doc)` after charged `_persist` fails, with no `store._charge`, no provider/ledger reservation and no observed actual. The reserved descriptor count exists, but bytes/quota and the physical fallback write are not charged. |
| **Pointer rollback durability** | Replica pointer/marker restoration still contains best-effort `except BaseException: pass` paths. A rollback or directory-sync failure can leave authority state uncertain while the original operation raises. This is narrower than the fixed publication rollback path but remains a durability limitation. |
| **Evidence durability/identity** | Preparation output is not uniformly committed through temp+full-sync+link semantics, and cleanup notes can conflate an unlinked file with a failed parent sync. Navigation replacement is now authority-validated, but the durability label must describe the executed primitive, not merely the platform. |
| **Timing/evidence scope** | Device objects without `.type` are now rejected, and receipt flags are validated. Remaining limitations are evidence scope only: two compound families/one seed, no native run, and no native cost or useful-recall claim. |

No open defect was found in the corrected 580/579/16 budget arithmetic itself. The fitting group length check remains correct because every eight-row group is evaluated separately and all groups are required.

## Exact clean evidence

### Checkout and tests

The worktree `/private/tmp/replica-independent-v12-d5aa9b43-authorization-gate` used sparse exclusions before checkout; protected module and checkpoints were absent. The density runtime module is tracked and collection succeeded without overlays. Owned tests were supplied externally: X tests `dab31abd…`, final residuals `1bdca80b…`, policy tests `5b48d926…`, budget tests `c019534a…`, origin test `1bc39e1a…`, and compound test `9261c64b…`.

Initial origin/compound/budget checks: **5 passed, 0 failed** in 2.45 s; outer 2.94 s, max RSS **375,341,056 bytes**. Final exact suite: **1023 passed, 0 failed, 0 skipped** in 295.48 s; outer 296.02 s, max RSS **990,216,192 bytes**. Receipts: `origin_compound_budget_d5aa9b43.*` and `full_regression_d5aa9b43.*`.

### Integrated and track workflows

The exact integrated tiny-BF16 2/9/36 workflow passed gradients, disabled/masked behavior, two-span live/durable and live/release parity, two-record cold value/logit parity, and exact fresh-process uninterrupted weights/moments with one actual update per arm. Outer 18.98 s, max RSS **790,642,688 bytes**, internal wall 16.366681208 s, artifacts **102,309,109 bytes**. Synthetic CPU engineering only.

Density profile/pilot/receipt/navigation completed with a canonical manifest-bound set. Pilot target **61,519,578.615384616 ns**, selected updates 2/1/1, all arms matched under executable tolerance. Profile 24.80 s/RSS **850,493,440 bytes**; pilot 11.69 s/RSS **976,617,472 bytes**. No density temporary-store directory remained. Metric/flag/navigation validation and unsupported timing rejection passed; no native cost or capability is established.

Replica workflow completed in 2.81 s/RSS **358,580,224 bytes**, internal 0.295042209 s, producing content-addressed measurement, protocol and interface receipts. `formation_used_new=true`, `task_improvement_requires_write=true`, `chunk_retry_same=true`. Publication/index recovery and versioned evidence passed their checks.

### Established checks

Migration strict tracked **1482/1482**, 11.08 s/RSS 31,129,600; final product **552/552**, 8.63 s/RSS 36,913,152; bundle **293 inputs, 7,657,081 bytes**, digest `5be29a21022408aca555ba77284940ff42a734d3a4e9bd64909a34e09f1970d3`, 8.22 s/RSS 27,377,664; tracked artifact guard **17,543 files**, no violations, 0.23 s/RSS 41,418,752. Earlier verdicts and historical receipts were not overwritten.

### Native public closure

Independent `--review` validated the exact packet, 62 implementation files, two source-execution files, **8,238,996,247 source bytes**, and environment closure **67 files / 450,061,586 bytes**, with code before model loading. It reports compound identity and no admission label parsing; 4.57 s/RSS **76,431,360 bytes**. Environment-only returned `environment_verified`, 12 critical files, `model_imported=false`, `native_execution=false`. Static preflight does not execute the failure-fallback ticket path and cannot establish complete work charging or authorization readiness.

No native competence, useful recall, persistent learning, promotion or RSI is established. Exact commands and machine-readable dispositions are in `commands.md` and `verification_summary.json`.
