# V12 independent-review repair register

Source verdict: `f98247f8` / `reports/intrinsic_memory_review_v12/README.md` — **incorrect / changes required**. Sixteen executable reproductions failed; additional findings were established by static data/control-flow inspection. No native authorization exists. This register prevents a later suite count from silently hiding an unresolved finding.

| IDs | Owner | Repair cycle | Scope | Status after assignment |
|---|---|---:|---|---|
| N1–N3 | NativeV12Preparation | 2/5 | restart work reservation, truncation, loaded-tokenizer identity | 64/64; environment/review pass; re-review pending |
| V1–V5 | V12Implementation | 3/5 | buffer/source authority, text-bound tokens, targeted eviction, release source identity, empty-site parity | 69/69 + run02; re-review pending |
| D1–D5 | DensityControls | 1/5 | per-pass gradients, numeric protocol binding, atomic profiler, revalidation accounting, producer API | 94/94 + profile/pilot; re-review pending |
| R1–R4 | ReplicaSuccessors | 3/5 | branch lineage, cold authority, P clocks, chunk pre-state/invalidation | 73/73 + workflow; re-review pending |
| F1–F2 | ReplicaSuccessors | 3/5 | immutable pins, outcome cutoff and observation order | 73/73 + workflow; re-review pending |
| T1–T2 | ReplicaSuccessors | 3/5 | tier read contract, movement/eviction/recovery accounting | 73/73 + workflow; re-review pending |
| A1–A3 | ReplicaSuccessors | 3/5 | actual workspace reset, source epochs, intervention order/finally | 73/73 + workflow; re-review pending |
| S1 | ReplicaSuccessors | 3/5 | observed write attribution | 73/73 + workflow; re-review pending |
| C1 | ReplicaSuccessors | 3/5 | initial-prefix token bound | 73/73 + workflow; re-review pending |

## Review coverage that must remain visible

- Two-record live/warm/cold/release ordering and value parity; release-write failure charging.
- Derived tombstone/note/lineage authentication and explicit P-specific revocation policy.
- Post-update density profiles and separation of CPU measurements from native consumers.
- Attempt consumption beyond output-directory existence, ledger stat TOCTOU, source/base hash and receipt-write charging, restart wall headroom, false-recall coverage, and shared-library/wheel closure.
- True live-versus-release and multi-record value parity in the integrated workflow.

Each item must gain focused coverage or remain an explicit unresolved limitation in the final independent verdict. A green happy-path suite is not sufficient to mark it resolved.

Coordinator coverage additions: two-record ordering/value parity and release-write
fault recovery are implemented; derived-table authentication and explicit P
revocation are implemented; post-update density profiles distinguish arms;
multi-record live/warm/cold/release parity is implemented. Native TOCTOU and
hash/read-byte accounting are repaired. Attempt-consumption durability, torn-tail
policy, false-recall breadth, wheel/shared-library closure and all native
behavior remain explicitly unqualified until the final reviewer and native run.

## Required re-verification

1. Owning tracks repair production code and add focused regressions without editing the independent reproduction file.
2. Coordinator runs all focused suites and `tests/test_independent_review_v12_findings.py`; every formerly failing reproduction must pass.
3. Reviewer re-tests repaired findings against a new final commit and issues a new verdict; the negative verdict and old packets remain historical.
4. Native packet is regenerated after all producer/native changes. Environment-only and `--review` must pass again before a new authorization receipt is created.

## Residual re-review findings and exposure decision

The `81df39ae` re-review preserved seven new failing edge reproductions and
additional static findings. A review scout read the v11 development file against
instructions; the user explicitly selected **fresh split + continue** on 29
October 2026. No label content is copied here, the historical receipt is
unchanged, and that split cannot support a fresh v12 utility claim.

| IDs | Owner | Repair cycle | Required closure |
|---|---|---:|---|
| X0 | NativeV12Preparation | 3/5 | new independent development split; exposure receipt; no old-label reads |
| X2, X3, X9 | NativeV12Preparation | 3/5 | torn-tail refusal, real work charging, attempt/environment closure |
| X5, X8, X12-producer | V12Implementation | 4/5 | charged/typed cold failure, truthful integrated parity, publication/order coverage |
| X6, X8, X10, X11, X12-density | DensityControls | 2/5 | executed tolerance, coherent receipts, empty parity, cleanup/no-overwrite, controls |
| X1, X4, X7, F1, T1, X12-Replica | ReplicaSuccessors | 5 + user-authorized 6th | derived recovery, P revocation, correction/privacy policy, authority-only reads, chunk edges |

Any further repair cycle required for Replica reaches the five-cycle cap and
requires the user question box before continuing. Other components have the same
five-cycle limit. No native authorization may be created until the fresh split,
all residual findings and a third independent verdict are complete.

Coordinator residual results: native 80/80 plus environment/review and a fresh
16-world development split; producer 79 checks plus true run03 parity; density
101/101 plus coherent fresh receipts; Replica 86 checks plus workflow and all
original/updated edge reproductions. The user authorized the targeted sixth
Replica fixture repair after the cap. All production fixes are committed;
third-party final review is pending before any authorization.

## Third-review blockers and continuation

The `84c464c3` review found clean-HEAD collection failure (X13), deterministic
native fitting-close failure (X14), positional label equivalence in the proposed
fresh split (X0), and further revocation/authority/atomicity residuals. The user
again selected **continue all repairs**, explicitly authorizing Replica cycle 7.

- Native cycle 4: X14, X0, X3 and packet/receipt consistency.
- Producer cycle 5 (final): process-control exceptions and atomic release publication.
- Density cycle 3: tracked runtime module, exact types, epsilon/sync, atomic writes,
  consumer boundary, control cost and current evidence.
- Replica cycle 7 (authorized): cross-instance P revocation, complete authority,
  pointer semantics, revocation-aware outcomes and current receipts.

No authorization or native launch is permitted until a clean-HEAD suite and an
independent lifecycle/freshness review pass.

## Exact-HEAD residual defects and continuation

The `f507db8a` exact-HEAD review collected successfully and reached 928 passed,
two failed, but confirmed D4, T1, X3 and X4 plus publication/evidence/process
residuals. The user selected **continue all repairs** again, authorizing producer
and Replica work beyond five cycles.

- Native cycle 5 (final): terminal parent charging, distinct capacity evidence and text consistency.
- Producer continuation: release charge rollback, post-replace failures, read accounting and process-control propagation.
- Density cycle 4: permitted source recomputation, canonical navigation and directory-durable receipts.
- Replica continuation: complete authority fields and cross-instance P restoration.

The native packet remains unauthorized and unlaunched.

## Final coordinator repair results

- Producer: 94 checks and run05, including release rollback and direct-read accounting.
- Native: 89/89 plus environment/review; 22 stages, 6925 seconds, 44,627,834,537
  reserved read bytes, 2057 writes; packet `908a50a41a7ae7f3cfa570913683af2a2ed31118e51f72e677d9cca451fe20b2`.
- Density: 114 checks and coherent coordinator05 evidence with durable atomic receipts.
- Replica: 97 checks and workflow, including complete authority and two-instance restoration.

All production changes are committed. Exact clean-HEAD independent qualification
is pending; no attempt marker or authorization exists.

## Native budget cap and authorized continuation

The `11dd40d3` qualification suite passed 948 tests with zero overlays, then
found final-probe 291/292-write under-reservation and four-resume 11/5-write
under-reservation. Static review also retained rollback, validity, quarantine,
durability, navigation and evidence-portability defects. Native reached cycle 5;
the user explicitly authorized cycle 6 and all remaining repairs.

The next gates are exact structural write-envelope tests, failure-atomic release
rollback, complete validity/quarantine recovery, directory-durable markers,
fail-closed timing and portable atomic navigation. Native remains unlaunched.

## Replica cycle budget update

The user explicitly authorized **up to 20 Replica repair cycles** because this is
the core and most difficult track. Subsequent Replica cycles count against that
20-cycle limit; **at cycle 20, stop and use the question box** to ask whether to
continue. Other components retain their existing caps and continuation
requirements. Repair work must also maintain documented online research,
independent findings, exact commands, failed attempts and measured evidence.

## Consolidated audit repairs

The user authorized all remaining cycles after the independent audit batch
(`audit_20261030.md`). Current work includes native cycle 7, density cycle 6,
producer continuation and Replica cycle 8/20.

Required closure: held-out rule families; origin/read/write reservation coverage;
cleanup without error masking; one validated crash-resolved navigation pointer;
Darwin full-sync and MPS-fallback enforcement; quarantine-safe revocation;
P-pointer failure recovery; attempted fallback bytes; and post-I/O P revocation.
Every fix requires focused failure/crash-path tests and updated research links.

## Origin and compound blockers

The `70ebb852` review passed 985 tests before independently confirming
zero-cap origin reservation and answer-bearing `unknown` compound text. Static
review retained metadata/fallback accounting, cleanup rollback, recovery,
cost-metric and durability defects. The user authorized all remaining cycles.

Current work: native origin/oracle/metadata repairs; producer cleanup/charge
rollback; density metric/navigation durability; Replica cycle 9/20 recovery and
durable evidence. No native launch or authorization has occurred.

## Dual-corpus and evidence-integrity repairs

The `7fd17078` release review passed origin/compound checks and 993 tests, then
confirmed 580/579-write probe requirements, 16-write restart and value-only
artifact-cap failures. The user authorized all remaining cycles.

Current work: exact constructor/artifact accounting; metadata/fallback charging;
cleanup without error masking; metric/navigation/crash-durability; Replica cycle
10/20 authenticated recovery, in-flight revocation and versioned evidence.

## Final coordinator gate candidates

- Producer: 111 checks and run09; release cleanup and byte accounting committed.
- Native: 95/95 and static gates; dual-corpus packet `601e11f1f7b3aa3953972a017794b2f63b80fc3548b0561df9a8dfcd8306c9d2`.
- Density: 130 checks and coordinator13 with frozen metrics and crash-safe navigation.
- Replica: 113 checks and workflow with content-addressed evidence; cycle 10/20.

Independent final gate review is pending. No native attempt or authorization exists.
