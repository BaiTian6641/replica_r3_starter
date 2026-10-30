# Exact clean-HEAD qualification review at bb269e54

## Verdict

**CPU engineering remains unqualified and the native packet is not ready.** Exact committed source collects and passes **948 tests with zero overlays**, including all original, updated X6/X11, residual, acceptance and fitting-close lifecycle reproductions. However, two new native budget reproductions fail: the final probe stage is one write short even under its documented single-resume envelope, and one read consumes 11 writes against its five-write reservation when four caller resumes occur. These are deterministic late-launch budget failures, not coverage preferences.

Reviewed integration: `bb269e54`; components `fb8c1425`, `7516bfeb`, `2d1da476`, `bb269e54`, and Replica `ed3d4cc3`. Packet SHA-256 `908a50a41a7ae7f3cfa570913683af2a2ed31118e51f72e677d9cca451fe20b2`; protocol `8057ce596b3643a7e10ccaaa68857a4cb4aa8290cb305ba034fd051d5a0142f0`; implementation `1ea886b52e493c068d6117cda7a04c01bc819243b6ce20956f6538c171134fd0`. No production file was edited, no authorization/attempt marker was created, and no native launch/model import/load/download occurred. Neither development data file was read or parsed; only public generator code, identities, roots, hashes and exposure receipts were used.

## Disposition of prior findings

### Fixed in exact committed source

- **N1-N3:** restart prompt budget, truncation and loaded-tokenizer identity.
- **V1-V5:** source/buffer authority, text-bound token cache, selective eviction, release identity and empty active sites, including pre-existing-buffer coverage.
- **D1-D5:** pilot gradient reset, exact numeric exposure/types, profiler atomicity, permitted source recomputation, and producer boundary. The new direct source-recompute test passes.
- **R1-R4:** branch isolation, cold authority, P clock/current pointer and chunk state.
- **F1-F2, T2, A1-A3, S1, C1:** privacy/outcome semantics, movement/recovery, task reset, epochs, interventions, study attribution and sealed prefix.
- **X0-X2:** seeded structural development independence, derived recovery and torn-ledger refusal/recovery.
- **X3 named terminal issue:** main/restart capacity receipts have distinct paths/stages and final result charging no longer depends on the obsolete consumed-stage fallback. The updated terminal lifecycle test passes.
- **X4 named revocation issue:** generic/P reads and clocked caches respect cross-instance revocation; restoration re-reads the durable marker and is no longer sticky. The previous failing restoration test passes.
- **X5 main cold path, X6-X7, X9-X10, X13-X14:** charged cold reads and process control, frozen/executed tolerance, correction authority, attempt/environment closure, empty-list parity, tracked density runtime module and fitting-close lifecycle.
- **T1 original minimal bypass and R3 pointer:** complete authority mappings are required for authority-only reads, and the P pointer is read/validated/repaired.
- **X8/X11/X12 named repairs:** canonical navigation names coordinator05 consistently, receipt files use temp+fsync+link+parent fsync, and density control costs/feature contracts are enforced. Release pointer staging/rollback and direct-read accounting have direct tests.

### Open or partial

| Finding | Current defect |
|---|---|
| **Native budget follow-up** | `fit_probes_512.writes=291`, but documented work needs **292**: 32×(4 rebuild +3 read +2 one-resume headroom) plus probe, fit-curve, fitting-closed and source-cost receipts. The final stage can therefore fail at close after fitting/probes. Independent arithmetic reproduction fails `291 >= 292`. |
| **Read resume envelope** | `DiagnosticEngineV12.read` permits four caller resumes at two writes each but reserves only one two-write resume. A real-ledger fake-pending read with four resumes consumes **11 writes** versus the five-write per-read envelope; corrected reproduction fails `11 <= 5`. The final probe is under-reserved even before the four-resume worst case. |
| **X12 release rollback** | `_publish_pointer` swallows restore failure and `_revert_pointer_charge` swallows accounting rollback failure. A failed restore can leave the new generation live while the charge is reverted. Release-origin initialization also charges before `atomic_json` without failure refund. |
| **T1 contract validity** | Full authority mapping is now enforced, but `ReadContract(cutoff=None)` still skips validity entirely, allowing expired payloads when identity alone is supplied. Rejected-tier payload bytes are read but excluded from `bytes_read` because only `result.ok` is charged. |
| **X1 quarantine recovery** | `recover()` skips derived-table recomputation whenever any event is quarantined, so an unrelated corrupt payload can conceal deleted tombstone/lineage/revocation rows instead of rebuilding them. |
| **X4 durability residual** | P revocation/restoration semantics are fixed in-process and cross-instance, but marker rename/unlink lacks parent-directory fsync, so crash durability is weaker than the P pointer path. |
| **X11 navigation/receipt residuals** | Navigation replacement is not set-atomic or rollback-safe on directory-fsync failure. Post-link temporary unlink failure can leave debris by design. Canonical manifests contain absolute host paths and a historical repair03 label, weakening portable evidence. |
| **Timing/failure handling** | Density timing records `mps_unavailable` and continues instead of refusing a pass whose declared synchronization did not execute. Fallback receipt writers can still replace an in-flight exception. |
| **Evidence** | Some README/report claims still cite obsolete stage counts or reused development, and some canonical verification artifacts are outside the declared canonical directory. These are evidence-quality gaps, not passing evidence. |

## Exact clean evidence

### Checkout and tests

The new worktree `/private/tmp/replica-independent-v12-bb269e54-qualified` used sparse exclusions before checkout; protected module and checkpoints were absent. The density runtime module is tracked and collection succeeded without overlays. Only owned review tests were copied as external inputs: updated X tests `dab31abd…`, final residuals `588f7ad…`, updated policy tests `5b48d926…`, and new budget edges `3d3b21f0…`.

Final exact suite: **948 passed, 0 failed, 0 skipped** in 286.10 s; outer 286.71 s, max RSS **1,035,091,968 bytes**. Receipts: `full_regression_bb269e54_final.*`.

New budget edges: **2 failed, 0 passed, 0 skipped**, intentionally confirming the two native budget defects; 2.23 s, outer 2.69 s, max RSS **373,932,032 bytes**. Receipts: `budget_edges_bb269e54.*`.

### Integrated and track workflows

The exact integrated tiny-BF16 2/9/36 workflow passed gradients, disabled/masked behavior, two-span live/durable and live/release parity, two-record cold value/logit parity, and exact fresh-process uninterrupted weights/moments with one actual update per arm. Outer 18.71 s, max RSS **794,214,400 bytes**, internal wall 16.117531708 s, artifacts **102,308,561 bytes**. This is synthetic CPU engineering only.

Density profile/pilot/receipt generation completed. Pilot target **60,135,607.615384616 ns**, selected updates 2/1/1, all arms matched under the executable tolerance. Profile 24.84 s/RSS **842,940,416 bytes**; pilot 11.57 s/RSS **1,000,882,176 bytes**. A canonical manifest-bound receipt set was generated. Initial navigation failed because custom input filenames did not match the canonical schema; that failure is preserved, and the corrected canonical directory generation succeeded. D4 is fixed, but the evidence/atomicity/timing residuals above remain open.

Replica workflow completed in 2.62 s/RSS **357,892,096 bytes**, internal 0.176610417 s; `formation_used_new=true`, `task_improvement_requires_write=true`, `chunk_retry_same=true`. Partial-authority and sticky-restoration reproductions now pass.

### Established checks

Migration strict tracked **1482/1482**, 10.83 s/RSS 31,817,728; final product **552/552**, 8.21 s/RSS 36,880,384; bundle **293 inputs, 7,657,081 bytes**, digest `5be29a21022408aca555ba77284940ff42a734d3a4e9bd64909a34e09f1970d3`, 8.12 s/RSS 27,656,192; tracked artifact guard **15,287 files**, no violations, 0.19 s/RSS 36,356,096. Earlier verdicts and historical receipts were not overwritten.

### Native public closure

Independent `--review` validated the exact packet, 62 implementation files, two source-execution files, **8,238,996,247 source bytes**, and environment closure **67 files / 450,061,586 bytes**, with code before model loading. It reports the fresh split public identity and no admission label parsing; 4.44 s/RSS **75,005,952 bytes**. Environment-only returned `environment_verified`, 12 critical files, `model_imported=false`, `native_execution=false`. Static preflight does not exercise the under-reserved final-probe or four-resume paths and cannot establish launch readiness.

No native competence, useful recall, persistent learning, promotion or RSI is established. Exact commands and machine-readable dispositions are in `commands.md` and `verification_summary.json`.
