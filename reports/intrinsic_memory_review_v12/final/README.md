# Final independent review at aedfc483

## Verdict

**CPU engineering is not qualified and the native packet is not ready.** Two independent release blockers are definitive: the exact HEAD cannot collect the requested suite because a required density runtime module is untracked, and every native launch would fail when `fitting_closed` is written after its stage has closed. The proposed fresh development split is also positionally label-equivalent to the exposed retired split, so it does not restore scientific freshness.

All 16 original reproductions and all seven updated X edge reproductions pass in a documented diagnostic overlay. That overlay adds the untracked module `src/replica_lab/intrinsic_memory_density_receipts_v12.py` (SHA-256 `f19ac2d40d578498696e04c7d7eb26d46b244e4e0d18711251937b1751c27222`) and uses the updated X6 test (SHA-256 `dab31abd0fcbe78b9be3ebd4fb05ec34aab7b09fe2e7f48a8dec85df308b7858`). The overlay passed **899 tests**, but it is not a clean `aedfc483` qualification. Seven additional residual reproductions fail, confirming the native stage-order defect and six other open issues.

Reviewed HEAD: `aedfc483`; component commits `826aac6b`, `257c722e`, `383a7a2e`, `aedfc483`, and Replica `bbd263b8`. Native packet SHA-256 `48963bdc1996526d7512cb6e753901d3562b08610cf67894906f0ec95b988c9e`; protocol `16c7b2f3ff072249e9a370be6e906ac93d8cfced6127122dfa6b6337e09203e7`; implementation receipt `c683db4881346ff0d922fc1744557f00a1328d462550b4facea5239362185d2c`. No production file was edited, no authorization/attempt marker was created, and no native launch/model import/load/download occurred. Neither v11 nor v12 development data was read or parsed in this final review; only public code, manifests, roots, hashes and exposure receipts were used.

## Explicit original-finding disposition

| Finding | Status at aedfc483 |
|---|---|
| N1 restart prompt budget | **FIXED** — exact 67-pass derivation and real-ledger reproduction pass. |
| N2 truncation | **FIXED** — non-termination is truncated and baseline requires termination. |
| N3 tokenizer identity | **FIXED** — loaded origin/files/live properties are checked; post-load vocabulary-content mutation remains a stated limitation. |
| V1 buffer authority | **FIXED code / coverage open** — parameters+buffers are stamped and save revalidates. Tests still add buffers after provider construction rather than mutate a pre-existing buffer. |
| V2 tokenizer-cache text identity | **FIXED** — actual text digest is in the key. |
| V3 selective eviction | **FIXED** — warm and cold paths preserve healthy siblings. |
| V4 release source identity | **FIXED for inference/read/value validation** — training intentionally bypasses release identity under its separate live-source contract. |
| V5 empty active sites | **FIXED** — producer and density consumer normalize empty iterables and preserve baseline. |
| D1 pilot gradient reset | **FIXED** — every pass zeroes gradients. |
| D2 numeric exposure protocol | **FIXED named numbers / new validation defect** — 2 and 8 are frozen, but 2.0/8.0 pass validation and fail in execution; the new non-integer reproduction fails. |
| D3 profiler atomicity | **FIXED** — validation precedes identity mutation and partial hooks roll back. |
| D4 authority side-effect classification | **FIXED** — revalidation is reported separately. |
| D5 producer boundary | **FIXED in boundary / consumer gap** — API and optimizer contracts are enforced, but cost pilot still constructs a profiler directly. |
| R1 branch isolation | **FIXED** — parents, descendants and lineage are scope checked. |
| R2 cold authority resume | **FIXED** — authority/contract persist and are revalidated. |
| R3 P clock | **FIXED core / pointer gap** — clock-versioned latest-valid P works, but the advertised current pointer is write-only and has no authority/failure consumer. |
| R4 chunk state | **FIXED** — pre-chunk recompute, pending guard and invalidation are correct. |
| F1 pin/revocation policy | **FIXED policy / stale docstrings** — privacy revocation is an immediate barrier consistent with external docs; `formation.revoke` and `recover` docstrings still contradict the implementation. |
| F2 outcome/order | **FIXED** — cutoff/epoch filtering and proposal ordinal work. Outcome filtering after revocation is unspecified. |
| T1 manifest read contract | **STILL OPEN, narrowed** — full contracts work, but a minimal authority mapping with only `eligible=True` bypasses producer/representation/scope/validity. New reproduction returns `ready`. |
| T2 tier movement/recovery | **FIXED** — movement accounting, clean eviction and schema quarantine work. |
| A1 task reset | **FIXED** — actual previous workspace is cleared. |
| A2 source epoch | **FIXED** — runtime epoch is admitted and propagated. |
| A3 intervention lifecycle | **FIXED** — explicit order and finally cleanup work. |
| S1 study attribution | **FIXED** — measured identities determine attribution. |
| C1 prefix bound | **FIXED** — exact maximum is explicitly sealed; over-bound is refused. |
| X0 fresh development | **STILL OPEN** — public admission/retirement/hashes work, but public generator code reuses the retired case table, rotations, order and oracle; `DEVELOPMENT_SEED` is unused. Labels are positionally determined by the exposed split without reading either data file. |
| X1 derived recovery | **FIXED** — expected derived tables and event scopes are recomputed and rebuild/degradation is reported. Deleted-tombstone reproduction passes. |
| X2 torn ledger tail | **FIXED** — all normal reads/mutations refuse; explicit audited recovery is required. Original reproduction passes. |
| X3 charged native work | **STILL OPEN, narrowed** — source/closure/model/fit/stage writes are charged, but `main_result`, `restart_result`, admission/result/capacity/failure receipts remain outside charged stages and some actual read bytes are constants rather than observations. |
| X4 P revocation | **STILL OPEN** — same-instance generic reads are blocked, but a marker written after another instance is constructed is ignored by generic APIs; clocked projection/replay entries also survive revocation. Both new reproductions fail. |
| X5 cold failure accounting | **FIXED main path / new exception defect** — charged staging, healthy-sibling retention and typed failure work; the two original edge tests pass. New `_cold_payloads` catches `BaseException`, swallowing `KeyboardInterrupt`/`SystemExit` as invalid payload. |
| X6 tolerance contract | **FIXED premise and execution / residual binding gaps** — mutation is refused and explicit relative tolerance executes; the updated test passes. `SELECTION_EPSILON` is still outside the frozen protocol, and the declared synchronization rule has no executable synchronization boundary. |
| X7 unverified correction | **FIXED** — an unverified successor cannot supersede verified evidence. |
| X8 coherent evidence | **STILL OPEN** — repair02 and fresh final receipts are generator-backed/coherent, but root and repair01 receipts remain contradictory and unmarked; repair02 verification plan cites the removed old X6 test name. |
| X9 attempt/environment closure | **FIXED with declared residuals** — durable external marker and 67-file/450,061,586-byte closure verify. Local marker deletion and deliberately incomplete wheel-file coverage remain operational limits. |
| X10 density empty list | **FIXED** — updated edge test passes. |
| X11 temp/no-overwrite | **FIXED mechanics / atomicity defect** — density temp directories are scoped and existing outputs are refused, but `write_new_json` directly writes the final path; fault injection leaves a partial file that blocks retry. |
| X12 remaining boundaries | **STILL OPEN, partial** — chunk, ordering and release parity are fixed; high-exposure control cost is not charged to the arm/comparison, P pointer authority/failure handling is absent, and some control/feature-contract booleans are copied rather than enforced. |
| X13 clean integration | **OPEN, new P0** — `aedfc483` omits the tracked density receipt module required by its committed tests/scripts. Exact clean collection fails with four import errors; coordinator results depend on untracked bytes. |
| X14 native stage lifecycle | **OPEN, new P0** — `run_main` closes `fit_probes_512`, then calls `engine.write_receipt("fitting_closed")` before entering checkpoint. The real ledger raises `stage is not active`; the independent reproduction confirms certain launch failure after fitting/probes. |

Additional evidence defects: the protocol difference list freezes **6835** seconds while validated caps total **6845**; `preflight_verification_fresh.json` says the current review/environment checks ran while README says they were not run; `reports/replica_successors_v12/final_verification.json` records a failure for the removed old X6 premise. Release-pointer replacement can become live before the surrounding store transaction/accounting completes, leaving fail-open publication on a later failure.

## Evidence obtained

### Exact clean HEAD versus diagnostic overlay

A new protected-absent source-only worktree `/private/tmp/replica-independent-v12-aedfc483-final` was configured with sparse exclusions before checkout. Protected module and checkpoints were absent. The exact HEAD regression command failed during collection in 4.73 s because `replica_lab.intrinsic_memory_density_receipts_v12` was missing; see `full_regression_aedfc483.{log,time}`. This is a failed check, not a skip or pass.

For diagnosis only, the untracked module above was copied into the isolated worktree and the updated X6 edge file was collected as `final_edge_cases.py`. The overlay suite passed **899/899, zero skips** in 286.51 s; outer 287.04 s, max RSS **998,080,512 bytes**. This proves the repaired behaviors in the overlay, not clean HEAD portability.

The separate residual suite failed **7/7**, zero skips, in 2.37 s; outer 2.85 s, max RSS **376,832,000 bytes**: inactive fitting-close stage, cross-instance P revocation, clocked P caches, minimal-authority manifest bypass, non-integer exposure, partial receipt creation, and swallowed process-control exception. Receipts: `residual_edges_aedfc483_overlay.{log,xml,time}`. Test SHA-256: `946c6a50bd599caa517a83c766d85812ef3ba59a7272118dc73c60d18455cea4`.

### Integrated and track workflows

The overlay integrated tiny-BF16 2/9/36 workflow passed all gradient/disabled/mask checks, two-span live/durable parity, actual live-vs-release slots/logits, two-record cold value/logit parity, and exact uninterrupted fresh-process weights/moments with one actual continuation update per arm. Outer 18.32 s, max RSS **794,460,160 bytes**, internal wall 15.769755292 s, artifacts **102,306,661 bytes**. This is synthetic CPU engineering only.

Fresh density profile/pilot/receipt generation completed. Pilot target **63,954,091.34615385 ns**, selected 2/1/1 updates and all three arms matched under the current executable tolerance. Profile 24.60 s/RSS 823,541,760; pilot 11.34 s/RSS 953,548,800. Receipts are internally manifest-bound, but X6/X8/X11/X12 defects remain and no native cost/capability is established. No density temporary-store directory remained; only a torchinductor cache directory was present under the controlled temp root.

Replica workflow completed in 2.57 s/RSS **355,074,048 bytes**, internal 0.177029125 s; `formation_used_new=true`, `task_improvement_requires_write=true`, `chunk_retry_same=true`. These results do not close T1/X4/R3 authority gaps.

### Established checks

Migration strict tracked **1482/1482**, 11.00 s/RSS 31,080,448; final product **552/552**, 8.04 s/RSS 38,174,720; bundle **293 inputs, 7,657,081 bytes**, digest `5be29a21022408aca555ba77284940ff42a734d3a4e9bd64909a34e09f1970d3`, 7.84 s/RSS 28,033,024; tracked artifact guard **14,199 files**, no violations, 0.17 s/RSS 35,962,880. Historical receipts were not overwritten.

### Native public preflight

Independent `--review` validated the exact packet, 62 implementation files, two source-execution files, **8,238,996,247 source bytes**, and environment closure **67 files / 450,061,586 bytes**, with code verification before model loading. It reports fresh-development public identity/admission and no label parsing; 4.43 s/RSS **75,055,104 bytes**. Environment-only returned `environment_verified`, 12 critical files, `model_imported=false`, `native_execution=false`. These static gates do not execute `run_main` and therefore do not detect X14, nor do they make X0 semantically fresh or X3 fully charged.

No native competence, useful recall, persistent learning, promotion or RSI is established. Exact commands and machine-readable dispositions are in `commands.md` and `verification_summary.json`.
