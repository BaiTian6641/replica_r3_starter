# Exact clean-HEAD release review at 258679fa

## Verdict

**CPU engineering remains unqualified and the native packet is not ready.** The origin-budget and compound-consistency checks now pass first, and exact committed source passes **993 tests with zero overlays and zero skips**. Full-probe accounting is still defective: the plan omits the second per-world corpus construction, restart omits its corpus construction, and the probe artifact-byte allowance covers only the value payloads with no origin/manifest/pointer bytes. These remain deterministic late-launch failures.

Reviewed integration: `258679fa`; components `98251be6`, `76c737e8`, `207388de`, `258679fa`, and Replica `0f2ff332`. Packet SHA-256 `135f7b1ee9d6034181247343d5068bb0566f9474603493ae7fe2869d5a71e9a0`; protocol `6269a6f753fd8ff4129db7383910952a2cab39a06c6c9cd924bdc63df3c2e283`; implementation `2a8402ef0f72ba837b1453b8f61bb21052f0b062e8f2720832ebd010f29cbd81`. No production file was edited, no authorization/attempt marker was created, and no native launch/model import/load/download occurred. Neither development data file was read or parsed; only public generator code, identities, roots, hashes and exposure receipts were used.

## Disposition of prior findings

### Fixed in exact committed source

- **N1-N3:** restart budget, truncation and loaded-tokenizer identity.
- **V1-V5:** source/buffer authority, text-bound cache, selective eviction, release identity and empty active sites.
- **D1-D5:** pilot gradient reset, exact numeric exposure/types, profiler atomicity, permitted source recomputation and producer boundary.
- **R1-R4:** branch isolation, cold authority, P clock/current pointer and chunk state.
- **F1-F2, T1-T2, A1-A3, S1, C1:** privacy/outcome semantics, full manifest identity+validity and rejected-read accounting, movement/recovery, task reset, epochs, interventions, attribution and sealed prefix.
- **X0-X2:** compound renderer/oracle consistency, quarantine receipt-kind recovery and torn-ledger handling.
- **X4-X5, X7, X9-X10, X13-X14:** cross-instance revocation/restoration and clocked caches, cold failure/process control, verified corrections, attempt/environment closure, density empty-list parity, tracked runtime module and fitting-close lifecycle.
- **O3-O10 named cases:** orphan-refund handling, corrupt-backup navigation abort/restore, final-receipt retry, unknown-accelerator refusal, quarantined correction/revocation kinds, P pointer restore/recovery, fallback attempted bytes and in-flight revocation reads.
- **Darwin durability:** `F_FULLFSYNC` injection leaves no final/temp file and retry succeeds; MPS fallback timing is refused.

### Open or partial

| Finding | Current defect |
|---|---|
| **Probe write envelope** | `fitting_probes` constructs `ProducerReleaseCorpus` twice per world, and every construction charges one write. Required per world is origin 1 + rebuild 3 + second origin 1 + read 13 = **18**, plus four stage receipts. Final probe therefore requires **580**, not 548; other probes require 579. Independent reproduction fails `548 >= 580`. Declared total is short by 128. |
| **Restart write envelope** | Restart constructs one corpus (1), reads (13), writes restart result (1) and source costs (1): **16 required versus 15 reserved**. The last source-cost write fails after continuation evidence. Independent reproduction fails `15 >= 16`. |
| **O1 probe artifact bytes** | `fit_probes_128.artifact_bytes = 32*4096 = 131072` exactly equals the 32 value payloads alone. Origins add about 9.7 KiB before manifest/pointer writes. Independent reproduction requires more than `32*4096` and fails. |
| **X3 metadata/fallback accounting** | Prepared-data reads, packet/protocol/implementation/source-execution reads and failure-fallback ticket writes remain uncharged. Teacher-cache hash read precedes its reservation. Some actual byte fields are size sums rather than observed reads. |
| **O2/X12 cleanup and rollback** | Non-ENOENT pending-unlink failure can replace the primary exception and skip charge rollback. A successful pointer replace followed by cleanup failure can leave a live pointer with refunded charge. Origin completion failures can mask the original write error. |
| **X6 metric/receipt binding** | Pilot uses the frozen synchronized metric, but receipt generation still accepts/copies arbitrary cost metrics; the checked-in fixture with `measured_wall` succeeds instead of being refused. |
| **X8 evidence identity/navigation** | Canonical receipts still contain stale or inconsistent path/test identities, including a test path absent in the clean checkout. Navigation and canonical manifest identities are not uniformly repository-relative or crash-resolved. |
| **X11 crash durability** | Removal after a failed parent-directory fsync is never made durable, so a crash can leave a final receipt while the caller saw an error. Durability labels are platform-derived rather than bound to the executed primitive. |
| **X1/O7 recovery residuals** | Quarantined non-revocation correction rows are not independently authenticated, and quarantined-event-owned derived rows are preserved without full authentication. This is weaker than the named receipt-kind fix. |
| **R3/O8 pointer recovery residual** | `recover()` now reconciles P-current, but publish/cleanup paths can still replace the original error, and no full orphan-pointer recovery test exists. |
| **In-flight writes/outcomes** | P publication/movement checks revocation only at entry; racing revocation can leave a new durable P record and live pointer. `outcomes()` computes revoked events before payload I/O and does not revalidate after reading. |
| **Replica evidence** | `final_verification.json` names one measurement file while recording another run's hash. `measurements.json` is a stale never-updated alias, and protocol/interface JSON is unversioned, unhashed and written in place. |

The fitting group length check itself is not a defect: `TeacherEngineV12` calls the eight-row base probe once per predeclared group and requires every group, so `len(result)==8` is correct.

## Exact clean evidence

### Checkout and tests

The new worktree `/private/tmp/replica-independent-v12-258679fa-release` used sparse exclusions before checkout; protected module and checkpoints were absent. The density runtime module is tracked and collection succeeded without overlays. Owned tests were supplied externally: X tests `dab31abd…`, final residuals `1bdca80b…`, policy tests `5b48d926…`, budget tests `c019534a…`, origin test `1bc39e1a…`, and public compound test `9261c64b…`.

The requested first check passed **2/2** for origin and compound in 2.33 s; outer 2.81 s, max RSS **375,586,816 bytes**. Final exact suite: **993 passed, 0 failed, 0 skipped** in 292.36 s; outer 292.92 s, max RSS **1,013,350,400 bytes**. Receipts: `origin_and_compound_258679fa.*` and `full_regression_258679fa.*`.

Additional budget probes: **2 failed, 1 passed** in 2.37 s, confirming the 580/16 write-cap defects; outer 2.86 s, max RSS **374,325,248 bytes**. Artifact-byte probe: **1 failed** in 2.26 s, confirming value-only cap exhaustion. Receipts: `budget_edges_258679fa.*` and `probe_artifact_budget_258679fa.*`.

### Integrated and track workflows

The exact integrated tiny-BF16 2/9/36 workflow passed gradients, disabled/masked behavior, two-span live/durable and live/release parity, two-record cold value/logit parity, and exact fresh-process uninterrupted weights/moments with one actual update per arm. Outer 18.96 s, max RSS **792,641,536 bytes**, internal wall 16.273418583 s, artifacts **102,308,777 bytes**. Synthetic CPU engineering only.

Density profile/pilot/receipt/navigation completed with a canonical manifest-bound set. Pilot target **62,373,568.615384616 ns**, selected updates 2/1/1, all arms matched under executable tolerance. Profile 25.28 s/RSS **841,695,232 bytes**; pilot 11.72 s/RSS **1,037,746,176 bytes**. No density temporary-store directory remained. X6/X8/X11/X12 evidence residuals prevent density qualification; no native cost or capability is established.

Replica workflow completed in 2.53 s/RSS **356,564,992 bytes**, internal 0.182535333 s, producing versioned `measurements_2bc1649fdb9c06f4.json` with matching reported hash `2bc1649fdb9c06f4bcfde96a7f28836caac4bd9341f14386ae0ae0e5c2dfdd51`. `formation_used_new=true`, `task_improvement_requires_write=true`, `chunk_retry_same=true`. These do not close in-flight write/outcome or publication-evidence defects.

### Established checks

Migration strict tracked **1482/1482**, 11.30 s/RSS 31,277,056; final product **552/552**, 8.38 s/RSS 37,666,816; bundle **293 inputs, 7,657,081 bytes**, digest `5be29a21022408aca555ba77284940ff42a734d3a4e9bd64909a34e09f1970d3`, 8.31 s/RSS 28,246,016; tracked artifact guard **16,673 files**, no violations, 0.20 s/RSS 40,665,088. Earlier verdicts and historical receipts were not overwritten.

### Native public closure

Independent `--review` validated the exact packet, 62 implementation files, two source-execution files, **8,238,996,247 source bytes**, and environment closure **67 files / 450,061,586 bytes**, with code before model loading. It reports compound identity and no admission label parsing; 4.52 s/RSS **75,005,952 bytes**. Environment-only returned `environment_verified`, 12 critical files, `model_imported=false`, `native_execution=false`. Static preflight does not exercise dual-corpus/restart accounting, artifact-byte exhaustion, or in-flight writes and cannot establish readiness.

No native competence, useful recall, persistent learning, promotion or RSI is established. Exact commands and machine-readable dispositions are in `commands.md` and `verification_summary.json`.
