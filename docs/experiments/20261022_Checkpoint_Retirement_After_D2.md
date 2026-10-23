# Checkpoint retirement after completed D2 qualification

22 October 2026, America/Chicago. The user requested deletion of earlier run
files that no longer serve the project. The D2 package was completed and
committed as `e75fd66`; its capacity/restart and tiny-fit evidence is archived.
This maintenance retires superseded payloads while preserving the original
source, selected results, final negative controls and all run-level evidence.
Training and generation remain idle. No model is instantiated or downloaded.

The [pinned plan](../../reports/checkpoint_retirement_plan_after_D2_20261022.json)
names each target. The
[cleanup journal](../../reports/checkpoint_retirement_after_D2_20261022.json)
records metadata archival, guarded deletion, protected-file checks and measured
filesystem free space. It records **completed reviewed retirement and orphan
pool cleanup**, with the original source, selected results and raw evidence
unchanged.

| Completed measurement | Result |
| --- | ---: |
| Retired payload directories | 19 |
| Retired orphan pool files | 10 |
| Archived small metadata files | 100 |
| Free disk before | 465.759 GiB |
| Free disk after | 588.998 GiB |
| Observed increase in free disk | **123.239 GiB** |
| Retained live pool versions | 4 |
| Model runs / downloads | 0 / 0 |

The journal verifies 35 protected byte claims before the initial cleanup and
again during recovery. All protected-file, original-source, acquisition-receipt
and raw-evidence identities remain unchanged. Training and generation remain
idle.

## Retired payloads and why

The cleanup retired **19 directories totaling 222,846,077,633 logical bytes
(207.542 GiB)**. It keeps the parent run directories and every report, profile,
generation, guard and log outside these payload subtrees.

| Payload group | Exact subdirectories | Count | Reason |
| --- | --- | ---: | --- |
| D2 capacity 256, chunks 01 and 02 | `checkpoint` and `training_checkpoint_0001` / `training_checkpoint_0002` | 4 | Completed two-update capacity/restart gate; independent audits and raw evidence archived. |
| D2 capacity 512, chunks 01 and 02 | `checkpoint` and `training_checkpoint_0001` / `training_checkpoint_0002` | 4 | Same completed gate at actual 512 tokens; no planned continuation parent. |
| D2 tiny, chunk 01 | `checkpoint` and `training_checkpoint_0004` | 2 | Successfully resumed to retained update 8; update-4 answers and scope/restart evidence remain archived. |
| D1 mask-1, both chunk-01 rate arms | `checkpoint` | 2 | Intermediate update-8 exports superseded by retained successful update 28 and negative update 64. |
| Native adapter and augmented generation qualification | `adapter_export` | 2 | Completed compatibility diagnostics; archived reports retain the findings and provenance. |
| D0 balanced mask-1, chunk 02 | `checkpoint` | 1 | Update-12 continuation probe is complete; selected update 8 remains intact. |
| D0 capacity 256 and 512, chunks 01 and 02 | `checkpoint` | 4 | Completed infrastructure checks; optimizer/runtime copies were retired by the earlier cleanup. |

The machine-readable plan contains complete paths and file identities. The
earlier [retention inventory](../../reports/checkpoint_retention_inventory_20261022.json)
and [first cleanup](20261022_Checkpoint_Cleanup_Applied.md) remain historical
records. Their conservative retention of all exports and then-undocumented D2
payloads is superseded only for the targets in this new plan.

## Protected local artifacts

- The immutable original at `checkpoints/minicpm5_source`, its acquisition
  receipt, sealed locks, tokenizer and template. Continue loading these same
  local MiniCPM5-2B bytes; routine experiments must never reacquire the model.
- D0 selected update 8: `minicpm5_balanced_mask1_B8_lr1e4_chunk01_20261022`,
  including export and `training_checkpoint_0008`.
- D1 successful update 28: `minicpm5_D1_mask1_B8_lr1e2_chunk02_20261022`,
  including export and `training_checkpoint_0028`.
- D2 tiny update 8: `minicpm5_D2_tiny_chunk02_20261022`, including export and
  `training_checkpoint_0008`.
- Final negative exports: the historical collapsed D0 preflight and D1
  lr1e-3 update 64. Their generation evidence and failure interpretation remain
  useful controls.
- Every run-level raw record, committed report, audit, answer and small
  provenance archive. No scientific result or failure record is erased.

All run names above are under `runs/`. Keeping a diagnostic result does not
promote it to the developmental model; the D2 scope and interpretation limits
in [its results](20261022_D2_Results.md) remain unchanged.

## Evidence and resume boundary

Before deletion, 100 small manifests, receipts and configuration files were
archived with verified hashes. Existing native, sidecar, optimizer-contract and
saved-answer audit records retain their original claims. The cleanup journal
binds the archival and deletion identities; it distinguishes fresh checks from
hashes recorded by earlier audits.

Deleted weights and optimizer/RNG state cannot be reconstructed from those
small records. Archived validations describe files present when the checks ran;
they remain evidence of completed gates, but cannot be rerun against removed
payloads. Exact continuation from those retired updates is no longer available.
The retained selected exports and runtime checkpoints remain the reusable
local results. A later fresh diagnostic would load the preserved original and
count as a new experiment, never an automatic rerun or download.

## Pool garbage collection and disk accounting

After the 19 directory retirements, **10 canonical native-weight pool files**
were orphaned and deleted. Garbage collection follows actual surviving
payload files and their manifests/receipts. Historical report references deliberately preserve
retired hashes and therefore do not keep a pool object live. Retain canonical
weights for D0 update 8, the D1 FP32 frozen base, D2 tiny update 8 and the
collapsed D0 negative export.

The orphan pool files total **100,670,697,200 logical bytes (93.757 GiB)**.
Those files can share APFS extents with the retired payloads. Their logical
size and the 207.542 GiB directory total must not be added and presented as
physical recovery. Only the journal's before/after filesystem readings measure
observed free-space change; APFS sharing, snapshots and unrelated system
activity can affect it.

The initial guard stopped after retiring 19 payload directories and one pool
file when an unused pool object's change timestamp (`ctime`) changed during
verification. Other identity fields were unchanged. A
[fresh full-hash recheck](../../reports/checkpoint_retirement_pool_metadata_recheck_20261022.json)
confirmed the remaining unused objects matched their content addresses and
were stable; a tested, narrow recovery then removed the nine remaining orphan
objects. The cause of the timestamp change is unassigned. The
[initial partial journal](../../reports/checkpoint_retirement_initial_after_D2_20261022.json)
is preserved and hash-bound by the final journal; the original guards remain
unchanged. All protected-byte checks passed again before recovery.

Targeted storage and retirement verification passed **75 tests** in 0.77 s:
[test results](../../reports/tests_checkpoint_retirement_20261022.xml) and
[raw output](../../reports/test_output_checkpoint_retirement_20261022.txt).

## Retention at the end of each experiment

Archive raw trainer/evaluator reports, answers, memory/guard traces, exact
settings and receipt/manifest hashes before marking any checkpoint obsolete.
Keep the original once, one explicitly retained result with its completed-update
runtime per active arm, selected comparison parents and useful final negatives.
Retire completed capacity checkpoints and superseded intermediate exports only
after their continuation/export checks and evidence archival pass. Preserve
undocumented or incomplete artifacts until their purpose is resolved.

Inspect free disk before launching another bounded run. Use verified APFS
weight sharing for byte-identical saves and explicit orphan-pool cleanup after
retirement; unique learned weights and optimizer states still consume space.
The serialization headroom and guard requirements in each protocol continue
to apply. This cleanup authorizes no new training or generation.
