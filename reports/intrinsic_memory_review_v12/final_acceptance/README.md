# Consolidated exact clean-HEAD acceptance review at 6925b653

## Verdict

**CPU engineering remains unqualified and the native packet is not ready.** Exact committed source collects without overlays and passes **985 tests with zero skips**, including every original, updated, residual, acceptance, budget and Darwin-specific X11 reproduction. Two additional independent defects are confirmed: release-origin creation is rejected by every `fit_probes_*` stage because `artifact_bytes` is capped at zero, and the compound development renderer records an answer in text while the oracle returns `unknown` for the same unrecorded target. The first is a deterministic launch blocker; the second invalidates false-recall and development utility controls.

Reviewed integration: `6925b653`; components `e0cab0df`, `febf1fea`, `0f09672f`, `6925b653`, and Replica `468fad07`. Packet SHA-256 `3e11f0f5beb4c8dadd7f9d73f0ae14833271926d21e91cc6a001dad5d647878c`; protocol `68238c7819f3d8466c16c6a03584a998504410385890f0e6789a943811bb651c`; implementation `44db25c47f944e603b4e04352f1669f31a08cd3802c9acffc66e46ddd4e7ab5a`. No production file was edited, no authorization/attempt marker was created, and no native launch/model import/load/download occurred. Neither development data file was read or parsed; only public generator code, identities, roots, hashes and exposure receipts were used.

## Disposition of prior findings

### Fixed in exact committed source

- **N1-N3:** restart budget, truncation, loaded-tokenizer identity.
- **V1-V5:** source/buffer authority, text cache, selective eviction, release identity and empty active sites.
- **D1-D5:** pilot gradient reset, exact numeric exposure/types, profiler atomicity, permitted source recomputation and producer boundary.
- **R1-R4:** branch isolation, cold authority, P clock/current pointer and chunk state.
- **F1-F2, T1-T2, A1-A3, S1, C1:** privacy/outcome semantics, full manifest identity+validity and rejected-read accounting, movement/recovery, task reset, epochs, interventions, attribution and sealed prefix.
- **X1-X2:** quarantine-aware derived recovery and torn-ledger refusal/recovery.
- **X4 named revocation/restoration:** cross-instance generic/P reads and clocked caches block revoked state; restore re-reads the durable marker and is not sticky.
- **X5 main cold path, X7, X9-X10, X13-X14:** cold staging/process control, verified corrections, attempt/environment closure, density empty-list parity, tracked runtime module and fitting-close lifecycle.
- **Exact consolidated budgets:** one read with four resumes is capped and actually charges **11 writes**; the final probe cap is **484**. Both updated budget reproductions pass.
- **O4-O10 named cases:** corrupt-backup navigation abort/restore, final-receipt retry, unknown-accelerator refusal, quarantined-revocation reconstruction, P pointer restore, fallback attempted bytes and in-flight revocation checks all have working implementations and regressions.
- **Darwin durability:** `F_FULLFSYNC` injection leaves no final/temp file and retry succeeds. MPS timing refuses unavailable/fallback backends.

### Open or partial

| Finding | Current defect |
|---|---|
| **Native origin reservation / O1** | `ProducerReleaseCorpus.__init__` performs `store._charge` then `_reserve_write("release_origin", size)`; `fit_probes_*` has `artifact_bytes=0`, so a real `plan("two")` ledger rejects the first origin at 305 bytes. Independent reproduction raises `BudgetError` before probe evidence. Reservation refusal also leaves the store charge. Root/lock creation is outside accounting, and rollback can refund after a non-durable unlink. |
| **X0 compound independence** | `_render_compound` always emits `The recorded color of {target} is {answer}` even when `target` is the unrecorded entity, while `_compound_answer` returns `unknown`. Public-code-only reproduction confirms the text contains the supposedly unknown answer. Structural family/seed independence alone does not repair gate validity. |
| **X3 metadata/fallback accounting** | Config/template/tokenizer and restart-request/checkpoint manifest reads occur before reservations. Failure-fallback ticket writes use raw `atomic_json` outside the 11-write envelope. Terminal receipts are now charged in normal paths, but these metadata/fallback paths remain uncharged. |
| **O2/O3 cleanup** | Non-ENOENT pending unlink failure can replace the primary exception and skip pointer-charge rollback. Immutable-write cleanup can refund while leaving an orphan file that makes every retry fail. Origin authority cleanup can leave staging files. |
| **X6 cost metric** | Pilot emits `measured_monotonic_wall_nanoseconds`, while the frozen protocol requires synchronized wall; receipt generation copies the mismatch instead of rejecting it. Timing now fails closed for unknown backends, but the evidence label is not bound to the frozen metric. |
| **X8 evidence/navigation** | Canonical coordinator09 receipts still say `repair_cycle=repair03`. Navigation paths are cwd-relative, not reliably repository-relative, and replacement/removal of navigation or final files lacks a following directory fsync on some failure paths. |
| **X11 crash durability** | The named fsync/retry path is fixed, but failed directory-fsync cleanup can remove the final link without fsyncing the parent afterward; a crash can leave a durable final receipt while the caller saw an error. Set-level rollback has the same omission. |
| **X12 release rollback** | Pointer charge is committed before `_stage_pointer`, whose failure does not revert it. A later pending-unlink failure can skip charge rollback entirely. Restore/charge-revert failures are still swallowed in some paths. |
| **X1/O7 recovery residuals** | Quarantined revocation is reconstructed, but quarantined non-revocation correction tombstones are not authenticated/rebuilt, allowing a corrected parent to reappear. Reconstruction also trusts the unauthenticated `events.kind` column rather than binding it to the receipt kind. |
| **R3/O8 pointer recovery** | `tiers.recover()` does not validate/reconcile `p_current`, so it can report ok while every `read_p` fails. `publish_p` cleanup can replace the original pointer-write exception. |
| **O6 residual** | Unknown accelerator fallback is refused in the pilot, but profile wall-time fields remain unsynchronized rather than explicitly labelled unsynchronized. |
| **Durability/evidence** | Revoke/restore applies marker changes before directory fsync and can raise after the effect. Replica workflow still writes canonical measurement JSON directly without no-overwrite/durable creation. |

## Exact clean evidence

### Checkout and tests

The worktree `/private/tmp/replica-independent-v12-6925b653-final-acceptance` used sparse exclusions before checkout; protected module and checkpoints were absent. The density runtime module is tracked and collection succeeded without overlays. Owned tests were supplied as external inputs: X tests `dab31abd…`, final residuals `1bdca80b…`, policy tests `5b48d926…`, budget tests `7ae606bb…`, origin budget `1bc39e1a…`, and public compound check `9261c64b…`.

Final exact suite: **985 passed, 1 failed, 0 skipped** in 288.94 s; outer 289.64 s, max RSS **997,130,240 bytes**. The failure is the intentional release-origin budget reproduction. All original 16, updated X6/X11, final residual, acceptance and exact 11/484 budget tests pass. Separate public compound check: **1 failed**, 0.03 s, max RSS **44,056,576 bytes**, confirming X0. Receipts: `full_regression_6925b653_final.*`, `origin_budget_6925b653.*`, and `compound_public_6925b653.*`.

### Integrated and track workflows

The exact integrated tiny-BF16 2/9/36 workflow passed gradients, disabled/masked behavior, two-span live/durable and live/release parity, two-record cold value/logit parity, and exact fresh-process uninterrupted weights/moments with one actual update per arm. Outer 18.66 s, max RSS **794,787,840 bytes**, internal wall 16.074315208 s, artifacts **102,308,845 bytes**. Synthetic CPU engineering only.

Density profile/pilot/receipt/navigation completed with a canonical manifest-bound set. Pilot target **62,898,569.115384616 ns**, selected updates 2/1/1, all arms matched under executable tolerance. Profile 24.96 s/RSS **842,743,808 bytes**; pilot 11.46 s/RSS **981,745,664 bytes**. No density temporary-store directory remained. These are CPU fixture results, not native cost or capability.

Replica workflow completed in 2.64 s/RSS **359,596,032 bytes**, internal 0.184298375 s; `formation_used_new=true`, `task_improvement_requires_write=true`, `chunk_retry_same=true`. Named O7-O10 paths pass, but recovery/durability residuals above remain open.

### Established checks

Migration strict tracked **1482/1482**, 11.38 s/RSS 31,031,296; final product **552/552**, 8.72 s/RSS 37,765,120; bundle **293 inputs, 7,657,081 bytes**, digest `5be29a21022408aca555ba77284940ff42a734d3a4e9bd64909a34e09f1970d3`, 8.25 s/RSS 27,426,816; tracked artifact guard **16,110 files**, no violations, 0.21 s/RSS 38,502,400. Earlier verdicts and historical receipts were not overwritten.

### Native public closure

Independent `--review` validated the exact packet, 62 implementation files, two source-execution files, **8,238,996,247 source bytes**, and environment closure **67 files / 450,061,586 bytes**, with code before model loading. It reports compound family identity and no admission label parsing; 4.67 s/RSS **74,530,816 bytes**. Environment-only returned `environment_verified`, 12 critical files, `model_imported=false`, `native_execution=false`. Static preflight does not exercise origin creation, metadata fallback writes or compound text/oracle consistency and cannot establish readiness.

No native competence, useful recall, persistent learning, promotion or RSI is established. Exact commands and machine-readable dispositions are in `commands.md` and `verification_summary.json`.
