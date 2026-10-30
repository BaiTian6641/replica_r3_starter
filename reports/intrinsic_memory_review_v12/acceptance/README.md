# Exact clean-HEAD acceptance review at e2c49bd3

## Verdict

**CPU engineering remains unqualified and the native packet is not ready.** The exact committed source now collects without any overlay, and almost all original repairs pass, but four independently reproduced defects remain: partial-authority manifest bypass, sticky cross-instance P restoration, legitimate source recomputation rejected as a side effect, and uncharged terminal parent receipts. Static review also confirms release-pointer rollback/accounting gaps, contradictory canonical evidence, and adjacent process-control/failure-masking paths.

Reviewed integration: `e2c49bd3`; components `7cf57766`, `2ddb6287`, `a3ddcc33`, `e2c49bd3`, and Replica `8cdfaf06`. Native packet SHA-256 `48910b59ed1d3012f7b7dc993206025afa95dd27067d63362968fd75e7a796c0`; protocol `d09bb7e83bec9ce414a3f4b83d6400068703acdabdbbba9248b7365668fe43fa`; implementation `08617be2f136d8c181f659ccaa7019f6c835ce1184c418693398386eff5b972f`. `src/replica_lab/intrinsic_memory_density_receipts_v12.py` is tracked at HEAD and present in the clean checkout. No production file was edited, no authorization/attempt marker was created, and no native launch/model import/load/download occurred. Neither development data file was read or parsed; only public generator code, identities, roots, hashes and exposure receipts were used.

## Disposition of prior findings

### Fixed in exact committed source

- **N1-N3:** restart budget, truncation, loaded-tokenizer identity.
- **V1-V5:** buffer/source authority, text-bound token cache, selective eviction, release source identity, empty active sites. Pre-existing-buffer coverage now exists.
- **D1-D3 and D5:** pilot gradient reset, exact numeric exposure/types, profiler atomicity and producer boundary.
- **R1-R4:** branch isolation, cold authority, P clock/current pointer, chunk state. The P pointer is now read/validated/reconciled and pointer-write failure rolls back the new record.
- **F1-F2:** privacy revocation semantics and outcome cutoff/order/revocation filtering.
- **T2, A1-A3, S1, C1:** movement/recovery, task reset, source epochs, intervention lifecycle, study attribution, sealed prefix bound.
- **X0-X2:** seeded structural development independence and torn-ledger refusal/recovery. Public generator code now varies slots/order/unknowns/extras/palette from the seed; no development labels were inspected.
- **X5 main cold path:** charged staging, healthy-sibling retention and typed failure. `_cold_payloads` now catches `Exception`, not process-control exceptions.
- **X6-X7:** frozen tolerance/epsilon and executed tolerance; verified-successor requirement.
- **X9-X10:** durable attempt marker/full declared environment closure and density empty-list parity.
- **X13-X14:** density runtime module tracked and collection succeeds; fitting-close receipt remains inside the active final probe stage. The corrected independent lifecycle test passes.

### Open or only partially fixed

| Finding | Disposition and evidence |
|---|---|
| **D4** | **OPEN:** `intrinsic_memory_density_v12.py:198-208` still adds positive source `requests/forwards/cache_misses` deltas to `changed` and raises, despite explicitly allowing source recomputation. `test_source_recomputation_is_not_a_forbidden_side_effect` fails. |
| **T1** | **OPEN, narrowed:** an authority containing only `eligible=True` is now refused, but `eligible=True, producer=...` satisfies the `any(identity_fields)` guard and skips representation/scope/cutoff checks. The new partial-authority reproduction returns `ready` for an expired payload. Require a complete contract or all identity fields. |
| **X3** | **OPEN:** `charged_write` consumes one `finalize_parent` write for the first capacity receipt, then writes the restart capacity receipt and final `result.json` with `receipt_charging.charged=false`. The restart capacity write overwrites main-window evidence. The independent consumed-stage reproduction returns `charged=False`. Release-corpus source/authority/payload reads also remain only locally counted or charged to different cost fields, not native `artifact_read_bytes`. |
| **X4** | **OPEN for restoration:** durable revocation now blocks generic/P reads and clocked caches across instances, but `_is_p_revoked` returns cached true without rechecking the marker. After a second instance observes revocation, host restoration by the first instance leaves the second permanently revoked; the new reproduction raises `PermissionError`. |
| **X8** | **OPEN:** fresh coordinator03 receipts are coherent, but `CURRENT.json` names coordinator03 while `NONCANONICAL.json` names repair03 canonical; the stated generator cannot generate either navigation file. Protocol text still freezes 6835 seconds while validated stage allocations total 6905. Default measurement outputs still target historical repair01. |
| **X11** | **OPEN residual:** sibling-temp+fsync+hard-link fixes the named partial-write case and the updated fsync/retry test passes. There is no parent-directory fsync, and an `os.unlink(temporary)` failure after linking leaves a final receipt while the call raises; receipt-set rollback cannot then remove it. |
| **X12** | **OPEN residual:** density control cost and feature-contract enforcement are fixed, but release pointer charge is committed before `atomic_json`; pointer-write failure is not released. A post-replace directory-fsync failure can leave a failed rebuild live. Release corpus cost fields also underreport direct payload reads. |
| **X5 adjacent failures** | **OPEN residual:** cold staging is fixed, but eviction/persist handlers and `finalize_receipt` still catch `BaseException`, potentially swallowing `KeyboardInterrupt`/`SystemExit`. `charged_write` failure during an existing exception can also mask the original error. |

R3 pointer authority and X12 density controls are otherwise fixed. X9 retains only declared operational limits: a locally deletable attempt-marker file and a closure that does not hash every wheel payload file.

## Exact clean evidence

### Source-free checkout and full suite

A new worktree `/private/tmp/replica-independent-v12-e2c49bd3-acceptance` was created with sparse exclusions for the protected module and checkpoints before checkout; both were absent. No code or module was overlaid. Only the three owned review test files were copied as external test inputs: updated X edge tests (SHA-256 `dab31abd0fcbe78b9be3ebd4fb05ec34aab7b09fe2e7f48a8dec85df308b7858`), updated final residuals (SHA-256 `588f7ad574d3c3e493a0bcc059d2999f567e213f61f775d9304a51431d236059`), and new policy edges (SHA-256 `bbb21e234812e26201d35f996313ffb86bd444a78ee0d736860335c41130189c`).

Final exact suite: **928 passed, 2 failed, 0 skipped** in 288.36 s; outer 288.97 s, max RSS **1,008,648,192 bytes**. The two failures are the intentional T1 partial-authority and X4 sticky-restoration reproductions. All original 16, updated X6/X11, final residual, and native lifecycle tests pass. A separate two-test accounting probe also failed both tests as intended: D4 source recomputation and X3 terminal parent charging; 0.56 s, outer 0.83 s, max RSS **229,965,824 bytes**. Receipts: `full_regression_e2c49bd3_final.*` and `residual_accounting_edges_e2c49bd3.*`.

### Integrated and track workflows

The exact integrated tiny-BF16 2/9/36 workflow passed gradients, disabled/masked behavior, two-span live/durable and live/release parity, two-record cold value/logit parity, and exact fresh-process uninterrupted weights/moments with one actual update per arm. Outer 18.67 s, max RSS **793,231,360 bytes**, internal wall 16.074133584 s, artifacts **102,308,340 bytes**. Synthetic CPU engineering only.

Density profile/pilot/receipt generation completed with a manifest-bound receipt set. Pilot target **60,607,931.61538461 ns**, selected updates 2/1/1, all three arms matched under the executable tolerance. Profile 24.66 s/RSS **826,441,728 bytes**; pilot 11.53 s/RSS **981,467,136 bytes**. No density temporary-store directory remained. D4 and X8/X11/X12 residuals prevent density qualification; no native cost/capability is established.

Replica workflow completed in 2.57 s/RSS **356,106,240 bytes**, internal 0.183348084 s; `formation_used_new=true`, `task_improvement_requires_write=true`, `chunk_retry_same=true`. T1 and X4 restoration remain open.

### Established checks

Migration strict tracked **1482/1482**, 11.05 s/RSS 31,195,136; final product **552/552**, 8.38 s/RSS 36,978,688; bundle **293 inputs, 7,657,081 bytes**, digest `5be29a21022408aca555ba77284940ff42a734d3a4e9bd64909a34e09f1970d3`, 7.96 s/RSS 27,607,040; tracked artifact guard **14,742 files**, no violations, 0.19 s/RSS 36,175,872. Earlier verdicts and historical receipts were not overwritten.

### Native public closure

Independent `--review` validated the exact packet, 62 implementation files, two source-execution files, **8,238,996,247 source bytes**, and environment closure **67 files / 450,061,586 bytes**, with code before model loading. It reports the fresh split public identity and no admission label parsing; 4.45 s/RSS **74,874,880 bytes**. Environment-only returned `environment_verified`, 12 critical files, `model_imported=false`, `native_execution=false`. These gates do not exercise parent terminal accounting and therefore do not close X3.

No native competence, useful recall, persistent learning, promotion or RSI is established. Exact commands and machine-readable dispositions are in `commands.md` and `verification_summary.json`.
