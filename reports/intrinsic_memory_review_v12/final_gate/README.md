# Exact clean-HEAD final gate review at f9133f11

## Verdict

**CPU engineering remains unqualified and the native packet is not ready.** The origin/compound/budget first checks pass **5/5**, and exact committed source passes **1008 tests with zero overlays and zero skips**. A separate real-call-graph accounting review then confirms a deterministic double charge in every probe world: `DiagnosticEngineV12` reserves three writes for `producer_release_rebuild` while `ProducerReleaseCorpus.rebuild` reserves the same value, manifest and pointer writes again on the active stage. Required probe caps are **676 final / 675 other**, not 580/579.

Reviewed integration: `f9133f11`; components `9a3bd5e2`, `686a04c5`, `f9133f11`, density `58f963a4`, and Replica `3ca3df2f`. Packet SHA-256 `601e11f1f7b3aa3953972a017794b2f63b80fc3548b0561df9a8dfcd8306c9d2`; protocol `b37beae565f33e386cedf922015f17817d7bf5108637a7767f3b7645c8503c13`; implementation `1f81b9700a5e5f5cf86da040ce2c2fe1735b506534ce098919038657655a7147`. No production file was edited, no authorization/attempt marker was created, and no native launch/model import/load/download occurred. Neither development data file was read or parsed; only public generator code, identities, roots, hashes and exposure receipts were used.

## Disposition of prior findings

### Fixed in exact committed source

- **N1-N3:** restart budget, truncation and loaded-tokenizer identity.
- **V1-V5:** source/buffer authority, text-bound cache, selective eviction, release identity and empty active sites.
- **D1-D5:** pilot gradient reset, exact numeric exposure/types, profiler atomicity, permitted source recomputation and producer boundary.
- **R1-R4:** branch isolation, cold authority, P clock/current pointer and chunk state.
- **F1-F2, T1-T2, A1-A3, S1, C1:** privacy/outcome semantics, full manifest identity+validity and rejected-read accounting, movement/recovery, task reset, epochs, interventions, attribution and sealed prefix.
- **X0-X2:** compound renderer/oracle consistency, quarantine receipt-kind recovery and torn-ledger handling.
- **X4-X5, X7, X9-X10, X13-X14:** cross-instance revocation/restoration and clocked caches, cold failure/process control, verified corrections, attempt/environment closure, density empty-list parity, tracked runtime module and fitting-close lifecycle.
- **O1-O10 named cases:** origin reservation/rollback, cleanup/original error handling, corrupt-backup navigation, final-receipt retry, unknown-accelerator refusal, quarantined correction/revocation kinds, P pointer restore/recovery, fallback attempted bytes and in-flight revocation reads.
- **Restart and artifacts:** restart cap 16 is sufficient; full origin/manifest/pointer allowances are present at 524,288 bytes per probe and 16,384 bytes at restart. Dual-corpus construction itself is now visible in the plan.
- **Darwin durability:** `F_FULLFSYNC` injection leaves no final/temp file and retry succeeds; MPS fallback timing is refused.

### Open or partial

| Finding | Current defect |
|---|---|
| **Dual rebuild charge** | Per probe world, the outer measured rebuild charges 3 writes and inner origin/value/manifest/pointer reservations charge 3 more. With first origin 1, second corpus 1 and read 13, actual is **21 per world**, plus four final-stage receipts (three other-stage), requiring **676/675** and **4001 total**, not 580/579/3617. `fit_probes_128` exhausts its 579 cap at world 28. Independent arithmetic reproduction fails `580 >= 676`. |
| **X3 metadata/fallback accounting** | Prepared-data reads, packet/protocol/implementation/source-execution reads and failure-fallback ticket writes remain uncharged. Teacher-cache hash read precedes reservation. Metadata actuals are stat sums rather than observed reads. |
| **Compound structured facts** | Unrecorded target rendering is fixed, but `same_colour` generates `anchor_colour` different from `answer` while its relation and oracle require equality. The stored compound oracle is internally inconsistent even though the rendered text omits that field. Unknown cases also remain fixed to side 1. |
| **O8/P publication recovery** | A record can be durably written before `_enforce_hot` raises `DirtyState`; there is no rollback, and later `read_p()` can select/repair the pointer to the failed generation. Generic `publish(kind="p")` still checks revocation only at entry. |
| **Pointer/directory durability** | Pointer rollback and directory sync failures are swallowed in Replica paths, which still use plain `os.fsync` rather than Darwin `F_FULLFSYNC`. A crash can leave an error-returned publication live or a live pointer to a deleted record. |
| **X1/O7 derived-index preservation** | Quarantined-event `record_index.event_key` is unauthenticated. Relabelling a healthy index row to a quarantined event preserves it over the correctly rebuilt row and hides unrelated healthy evidence. |
| **X6 evidence validation** | Metric is now frozen and executed, but receipt builders do not validate `scored_labels_read` or input `native_execution`, so contradictory input flags can be copied into a clean canonical receipt set. |
| **X11 crash durability** | `_remove_final` can unlink successfully and then fail parent fsync, with no durable retry of that removal. New directory creation is not fsynced in its parent. Cleanup notes can incorrectly claim a file remains after it was unlinked. |
| **X12 release cleanup** | Constructor temp cleanup runs outside `writer.lock` and can delete another writer's pending/restore pointer. Failed release-write unlink is not followed by parent fsync before refund. |
| **O6 timing edge** | Device strings without `.type` are silently labelled `cpu_synchronous` instead of refused as unsupported inputs. Production device objects are unaffected. |
| **Evidence identity** | Some native README tables retain stale read/write/probe caps and preparation outputs are written without durable temp+fsync+rename semantics. These do not validate the corrected budget arithmetic. |

The fitting group length check remains correct: each predeclared eight-row group is evaluated separately and all groups are required.

## Exact clean evidence

### Checkout and tests

The worktree `/private/tmp/replica-independent-v12-f9133f11-final-gate` used sparse exclusions before checkout; protected module and checkpoints were absent. The density runtime module is tracked and collection succeeded without overlays. Owned tests were supplied externally: X tests `dab31abd…`, final residuals `1bdca80b…`, policy tests `5b48d926…`, budget tests `231fbc60…`, origin test `1bc39e1a…`, and compound test `9261c64b…`.

Initial origin/compound/budget checks: **5 passed, 0 failed** in 2.40 s; outer 2.93 s, max RSS **377,372,672 bytes**. Final exact suite: **1008 passed, 0 failed, 0 skipped** in 290.63 s; outer 291.19 s, max RSS **1,047,150,592 bytes**. Separate dual-charge reproduction: **1 failed** in 2.26 s; outer 2.74 s, max RSS **373,473,280 bytes**. Receipts: `origin_compound_budget_f9133f11.*`, `full_regression_f9133f11.*`, and `double_charge_probe_f9133f11.*`.

### Integrated and track workflows

The exact integrated tiny-BF16 2/9/36 workflow passed gradients, disabled/masked behavior, two-span live/durable and live/release parity, two-record cold value/logit parity, and exact fresh-process uninterrupted weights/moments with one actual update per arm. Outer 18.90 s, max RSS **782,155,776 bytes**, internal wall 16.264645375 s, artifacts **102,309,017 bytes**. Synthetic CPU engineering only.

Density profile/pilot/receipt/navigation completed with a canonical manifest-bound set. Pilot target **62,853,424.69230769 ns**, selected updates 2/1/1, all arms matched under executable tolerance. Profile 24.93 s/RSS **837,550,080 bytes**; pilot 11.50 s/RSS **981,483,520 bytes**. No density temporary-store directory remained. Metric is frozen/executed and canonical identities are repository-relative, but the evidence-validation and crash-durability residuals above remain open.

Replica workflow completed in 2.60 s/RSS **356,728,832 bytes**, internal 0.179790917 s, producing content-addressed measurement, protocol and interface receipts. `formation_used_new=true`, `task_improvement_requires_write=true`, `chunk_retry_same=true`. In-flight write/publication and pointer-durability residuals remain open.

### Established checks

Migration strict tracked **1482/1482**, 11.39 s/RSS 30,932,992; final product **552/552**, 8.48 s/RSS 38,141,952; bundle **293 inputs, 7,657,081 bytes**, digest `5be29a21022408aca555ba77284940ff42a734d3a4e9bd64909a34e09f1970d3`, 8.40 s/RSS 27,967,488; tracked artifact guard **17,229 files**, no violations, 0.21 s/RSS 40,435,712. Earlier verdicts and historical receipts were not overwritten.

### Native public closure

Independent `--review` validated the exact packet, 62 implementation files, two source-execution files, **8,238,996,247 source bytes**, and environment closure **67 files / 450,061,586 bytes**, with code before model loading. It reports compound identity and no admission label parsing; 4.46 s/RSS **75,644,928 bytes**. Environment-only returned `environment_verified`, 12 critical files, `model_imported=false`, `native_execution=false`. Static preflight validates plan equality but does not execute a real probe world and therefore misses the outer+inner double charge and metadata/fallback gaps.

No native competence, useful recall, persistent learning, promotion or RSI is established. Exact commands and machine-readable dispositions are in `commands.md` and `verification_summary.json`.
