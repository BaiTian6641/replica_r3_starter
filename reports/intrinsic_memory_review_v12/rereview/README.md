# Independent re-review at c95d3e23

## Fresh verdict

**CPU engineering remains unqualified and the native packet is not ready.** All 16 original reproductions now pass and every original N1-N3, V1-V5, R1-R4, F2, T2, A1-A3, S1, C1, and D1-D5 repair is present. F1 is implemented for immutable pins, but its privacy-revocation policy conflicts with another active contract. T1 is only partially closed because authority-only reads can still bypass manifest identity checks. New independent reproductions confirm seven residual defects, and static review confirms additional budget/recovery/evidence defects.

The native closure and environment review pass without model import/load, and the packet hash is `98c3f47abfa12d29222ae6b90e4c2b944396f4fb00bb29ba63ec20236773012d`. That does not cure the torn-ledger append, uncharged declared work, cold failure-path accounting, or exposure/process blockers below. No authorization, native launch, model download/load, production edit, or commit occurred.

**Exposure incident:** the delegated NativeRereview scout read `data/intrinsic_memory_coverage_v11_20261029/development.json` while checking false-recall coverage, contrary to the explicit no-development-label instruction. The historical receipt still says unopened, but it no longer describes this review team. Keep native launch blocked and treat that split as exposed for any successor utility claim informed by this review. No label contents are reproduced in this report. An attempted immediate coordinator notification failed because the available agent channel was read-only; this report is the delivered notification.

Reviewed integration: `c95d3e23958d3b76299b01afc2b31a714395707b`; components `6ba07e1ba0850da6b8329fe8ce357ef07720ee5a`, `cb7d4634660e677b3090cc4c0940bf7dd2ec9b1f`, `c268d56126f44d8940a49f9d15ec43388974ccfc`, and `c95d3e23958d3b76299b01afc2b31a714395707b`. Protocol SHA-256 `b6247f6c0bf46097f864f4c558b90b683d21200c7d89dfb3233c949835cf3f0a`; implementation receipt SHA-256 `e4e7df7fcbac7bf5347bc2d8dbce4bbb0e89cc885d06557b3b5e97e761db806c`. The unrelated dirty helper remains SHA-256 `f43ba05ff6742682fad962bddd450610bf3ccffccacd0a48debf63647a64f9bb`.

## Original finding resolution

| ID | Status | Current evidence |
|---|---|---|
| N1 | **FIXED** | Restart derives and reserves exactly 67 prompt passes; the original real-ledger reproduction passes and pass 68 is refused by the new native test. |
| N2 | **FIXED** | `intrinsic_memory_diagnostic_v12.py:327` reports `not stopped`; baseline also requires termination. Original reproduction passes. |
| N3 | **FIXED** | `intrinsic_memory_native_v12.py:113-141` re-hashes actual tokenizer files and checks loaded origin, vocabulary size, IDs and fast flag. Runtime vocabulary-content mutation after load remains a coverage limitation. |
| V1 | **FIXED** | `contextual_source_features_v12.py:304-333` stamps parameters and buffers and clears C on mismatch; save validates originals. Original reproduction passes. Pre-existing-buffer mutation remains under-tested. |
| V2 | **FIXED** | Token cache key includes actual text digest at `contextual_source_features_v12.py:374-376`; original wrong-text reproduction passes. |
| V3 | **FIXED** | Warm/live failures evict only `ContextualRecordError.record_id`; cold failure probes are selective. Healthy-sibling cold-cache behavior lacks a direct assertion and has new accounting defects X5. |
| V4 | **FIXED** | Release source identity is checked in values, batch validation and read at `contextual_release_v12.py:261-262,294-295,322-323`. |
| V5 | **FIXED** | Empty iterables normalize and skip in producer forward; original list reproduction passes. Density consumer defect X10 remains. |
| R1 | **FIXED** | Formation validates parent, descendants and every lineage ancestor for subject/branch/purpose/mode; original cross-branch reproduction passes. |
| R2 | **FIXED** | Cold tickets retain authority/contract and both resume and fallback revalidate after I/O; original reproduction passes. |
| R3 | **FIXED** | P uses clock-versioned records, monotonic pointer and latest-valid selection; original clock-2 reproduction passes. Generic revocation remains open under RR4. |
| R4 | **FIXED** | Chunk begin guards pending state, invalidate restores committed state, recompute replays recorded operations from pre-chunk state; all three original reproductions pass. |
| F1 | **FIXED code / POLICY OPEN** | Tombstones apply only at generations visible to the pin and the original reproduction passes. `docs/01_System_Design.md:144` instead says a pin does not override later privacy revocation; this must be reconciled before claiming both privacy and immutable snapshots. |
| T1 | **STILL OPEN, partial repair** | `ReadContract` is enforced on hot/cold/fallback, and contract-or-authority is mandatory. However `formation_authority` (`successors/adapters.py:396-405`) does not validate manifest producer/representation/scope, and authority-only reads remain allowed at `successors/tiers.py:509-524`. |
| T2 | **FIXED** | Movement bytes are counted, clean eviction candidates are used, manifest schemas are validated and malformed records quarantined. |
| A1 | **FIXED** | Reset clears actual `runtime.previous_workspace`; no-store regression exists. |
| A2 | **FIXED** | Runtime epoch is admitted, propagated and revalidated for every observation/outcome/proposal. |
| A3 | **FIXED** | Explicit identity/task order and `finally` pin cleanup are implemented. The disjoint C/I reset effects remain structurally difficult to distinguish. |
| S1 | **FIXED** | Study attribution derives from actual new/before/disabled record identities and measured provisional-note absence. |
| C1 | **FIXED** | Initial prefix length/types are bounded and the original over-bound reproduction passes. Exactly 8192 initial tokens is still accepted but permits no later chunk; clarify whether that is an intentionally sealed sequence. |
| D1 | **FIXED** | Every pilot pass zeroes gradients; original reproduction passes. |
| D2 | **FIXED for named numbers** | Common updates 2 and cap 8 are numeric, validated and read from the protocol. Tolerance remains unbound under RR6. |
| D3 | **FIXED** | Instrument validates before identity mutation and hook registration rolls back on partial failure; original reproduction passes. |
| D4 | **FIXED** | Authority revalidation is reported separately from prohibited effects; original durable reproduction passes. |
| D5 | **FIXED** | Boundary enforces encoder/provider methods, NativeFusedAdamW, class/module and non-null release. Cost pilot still constructs the profiler directly rather than using this boundary. |

All 16 original independent reproductions were explicitly identified in the new JUnit and had no failure/skip children.

## New confirmed findings

### P0 — scientific process blocker

**X0. Development exposure invalidates a fresh successor utility claim.** Trigger and consequence are described in the exposure incident above. This was not a production-code defect and no label content is needed to reproduce the process fact: the NativeRereview scout output records reading the development file. Remedy: preserve the old receipt as historical, record the new exposure honestly, and do not launch/use that split as fresh for this review-informed successor.

### P1 — authority, recovery, and budget correctness

**X1. Formation recovery fails open after derived-authority deletion.** `successors/formation.py:1462-1569` rebuilds only `record_index` when unequal and `_derived_errors` checks references, not expected tombstones/lineage/revocations or event/note scope columns. `test_recovery_detects_deleted_revocation_tombstone` deletes a real revocation row; `recover()` returns `ok`, leaving revoked evidence readable. Tampering lineage can also detach descendants from later corrections. Remedy: recompute and authenticate all derived tables and scope columns from payload/receipt authority, or bind them into the receipt chain; add deletion/tamper tests.

**X2. A torn ledger tail is appended to instead of refused.** `intrinsic_memory_budget_v12.py:203-289` records `torn_tail` but reserve/enter/finish never reject it. `test_ledger_refuses_append_after_torn_tail` observes a successful reservation; the next record fuses with the partial line and bricks replay. Remedy: refuse every mutation and non-operator read on a torn tail until explicit recovery, and test reserve/enter/finish.

**X3. Declared native read/write work is still not charged.** `run_spark_intrinsic_coverage_v12.py:148-152,185-189` wraps source preflight/revalidation with no `artifact_read_bytes`; `intrinsic_memory_native_v12.py:181` charges only `model_loads`; per-fit receipts at `:350` and stage source-cost receipts at `intrinsic_memory_diagnostic_v12.py:233-241` use bare `atomic`. Thus the declared 41,297,741,683 read bytes and at least 512 receipt writes are not reserved/enforced despite being counted in packet arithmetic. Remedy: reserve before each actual hash/load/receipt write and record actuals without refund; add a whole-stage ledger-consumption test.

**X4. Revoked P remains consumable through generic tier reads.** `successors/tiers.py:866-917` writes `p_revoked` but generic `_authority_state` only checks `_tombstones`, and recovery ignores the revocation marker. `test_revoked_p_is_not_ready_through_generic_tier_read` returns `ready` for `p_expert_c1` after host revocation. Remedy: enforce P revocation at every storage/read/eviction/recovery boundary and define host restoration explicitly.

### P2 — failure accounting, contract binding, and evidence integrity

**X5. Failed cold resume performs uncharged payload reads and can lose its typed failure.** `contextual_memory_v12.py:261-300,488-492` re-reads healthy text plus four slots with `accounting=None` before recording failure; eviction errors occur before terminal persistence. `test_failed_resume_does_not_re_read_healthy_payloads_uncharged` observes five uncharged reads; `test_failed_cold_resume_preserves_typed_failure` replaces `ContextualReadFailure` with a secondary `RuntimeError`. Remedy: use staged results/authority instead of re-reading or charge every probe; persist the typed terminal failure independently of best-effort eviction.

**X6. Configured density tolerance is not the executed tolerance.** `intrinsic_memory_density_protocols_v12.py:459-461,566-567` validates only numeric type and uses module constant 0.02; the pilot prints the configured value but does not pass it to `cost_tolerance`. `test_configured_tolerance_is_the_executed_tolerance` changes config to 0.5, validation passes, execution remains 0.02. Selection rule/rounding and synchronized timing strings likewise lack executable validation. Remedy: validate exact frozen values and pass the validated protocol into every computation.

**X7. Unverified correction policy can erase verified evidence.** `successors/formation.py:1328-1352` creates an unverified successor without a verifier receipt while tombstoning the verified parent; a verified-only pin filters the successor and sees neither. No explicit policy or test authorizes this result. Remedy: require successor verification/use eligibility at least equal to the parent or specify and test a deliberate fail-closed erasure policy.

**X8. Current checked-in evidence is inconsistent or mislabeled.** Density `repair01/protocol_summary.json` and `verification_results.json` cite `target_cost_ns=59918921.307692304`, while the linked current pilot has `56841349.46153846`; no checked-in generator reproduces the summaries. Integrated `scripts/run_intrinsic_contextual_cpu_v12.py:280-288` names a release-to-release comparison `matches_live_slots`, and child cold checks count spans rather than compare values. True two-record parity exists only in `tests/test_contextual_memory_v12.py:530-565`. Remedy: regenerate coherent receipts through checked-in commands and rename/measure the workflow fields accurately.

**X9. Single-use attempt and environment closure remain incomplete.** Attempt consumption still relies only on a deletable output directory (`run_spark_intrinsic_coverage_v12.py:217-225`); no external append-only consumption registry exists. Environment validation still binds selected files rather than native shared libraries/full wheel integrity, as its own manifest declares. Remedy: durable external attempt marker and explicit full executable dependency closure or a stated qualification limitation.

### P3 — edge behavior and operational evidence

**X10. Density empty-list active sites misses disabled parity.** `intrinsic_memory_density_v12.py:489-497` checks only `== ()`; `test_density_profiler_empty_list_uses_disabled_parity` reaches snapshot and raises. Normalize before the guard.

**X11. Density pilot/profile retain temporary stores and overwrite timing receipts.** Controlled `TMPDIR` execution left **41 directories / 9,760 KiB** after one profile+pilot. Store/profiler construction occurs before `try`, so construction failures can mask the original error and leak path resources. Both scripts overwrite their default repair01 JSON via `write_text`. Remedy: scoped temporary directories, construction inside guarded cleanup, and fresh/no-overwrite outputs.

**X12. Chunk and parity boundaries remain coverage gaps.** Recompute between begin and first propose can replay the previous committed chunk; exactly-max-prefix semantics are unspecified. Same-source two-record order is deterministic only after adding `record_id`, not semantic observation/admission order. Whole-release eviction and pointer/publication failure paths lack direct tests. These require contract tests before broader claims; they were not counted as new high-severity defects.

## Coverage gap disposition

- **Producer:** release `_write` rollback is fixed and tested; pointer/manifest publication failure remains untested. Two-record live/warm/cold/release parity is fixed in unit tests for canonical sorted input; reversed caller order, post-correction parity, and the integrated release/cold claims remain open. Static native model source computes RoPE values on demand; no model execution was performed to assess runtime buffer behavior.
- **Density:** post-update profiles are **fixed**; my fresh profile records `logits_changed_from_pre_update=true` for all six fixed/matched arms. Negative profiler controls pass, but no deliberate out-of-tolerance arm or no-update unchanged-digest control exists. Disjoint native pilot worlds and end-to-end native consumption remain open. Matched records still omit `feature_contract`. Report reproducibility and temp/output safety remain open as above.
- **Replica:** original formation/chunk/tier/adapter contracts pass, but derived recovery authentication, generic P revocation, unverified-correction policy, and privacy-revocation documentation remain open. The fresh workflow reports `formation_used_new=true`, `task_improvement_requires_write=true`, and `chunk_retry_same=true`; these do not prove those missing boundaries.
- **Native:** cached-ledger lock TOCTOU is fixed under cooperating writers. Restart wall is statically provisioned at 150 seconds but no v12 native timing exists. Public generator code establishes non-vacuous unknown-answer controls without this main process reading labels; the delegated exposure incident still stands. Partial native shared-library/wheel origins and output-only attempt consumption remain open.

## Independent evidence at c95d3e23

### Isolation

A new detached worktree `/private/tmp/replica-independent-v12-c95d3e23-rereview` was created with no checkout, then sparse exclusions for the protected module and checkpoints were configured before checkout. Both were absent afterward. No weights were downloaded/copied and no production file was edited. New edge tests and receipts exist only under this rereview directory; `test_rereview_edge_cases.py` SHA-256 is `b9fbe2e837198adc03c4bc04ec95782f6ce65d1353ef23754208502bf195d04a`.

### Tests and workflows

- Full inherited + matched + all new native/density/Replica repairs + original independent reproductions: **859 passed, 0 failed, 0 skipped**. Pytest 284.73 s; outer 285.27 s; max RSS **1,001,111,552 bytes**. Receipts `full_regression_c95d3e23.{log,xml,time}`.
- New residual edge reproductions: **7 failed, 0 passed, 0 skipped**, intentionally confirming X1, X2, X4, X5, X6 and X10. Pytest 2.78 s; outer 3.26 s; max RSS **403,243,008 bytes**. Receipts `edge_reproductions_c95d3e23.{log,xml,time}`.
- Integrated tiny-BF16 2/9/36 workflow: all gradient, disabled, mask, durable, cold recovery and exact uninterrupted continuation checks passed; one actual update per arm. Outer 17.90 s, max RSS **777,371,648 bytes**, internal wall 15.266464416 s, final artifacts **102,218,469 bytes**. This is CPU engineering only.
- Fresh density profile: all six arms changed post-update logits; 24.48 s, RSS **823,771,136 bytes**. Fresh pilot: target **63,576,216.34615384 ns**, selected updates 2/1/1 and all matched under the current executable tolerance; 8.70 s, RSS **951,582,720 bytes**. These do not close X6/X8/X11 or establish native cost.
- Replica workflow: 2.60 s, RSS **357,580,800 bytes**, internal 0.174966791 s; fresh outputs under `replica_successors/`.

### Established checks

Migration strict tracked **1482/1482**, 10.91 s/RSS 31,424,512; final product **552/552**, 8.30 s/RSS 38,813,696; bundle **293 inputs, 7,657,081 bytes**, digest `5be29a21022408aca555ba77284940ff42a734d3a4e9bd64909a34e09f1970d3`, 7.83 s/RSS 27,688,960; tracked artifact guard **13,364 files**, no violations, 0.18 s/RSS 34,488,320. No historical receipt was overwritten.

### Native closure/environment

Independent review validated the exact packet above, **62 implementation files**, two source-execution files and **8,238,996,247 source bytes**, code before model loading; 4.18 s/RSS **76,316,672 bytes**. Environment-only returned `environment_verified`, 12 critical files, `model_imported=false`, `native_execution=false`; 0.06 s/RSS **30,998,528 bytes**. These are static/environment gates only. No native competence, useful recall, P learning, promotion or RSI is established.

Exact commands are in `commands.md`; machine-readable status is in `verification_summary.json`.
