# Spark archive measurement and migration preparation v16

Status: **bounded inventory and interruption-safe dry run complete; no production migration or deletion executed.** Formation, persistent learning, density and RSI remain gated by useful native recall.

## Measured archive and release costs

Source: preserved v12 attempt10 `stores/` and its 128 rebuild events. Existing per-release evidence remains `reports/intrinsic_memory_successors_v13/release_costs_attempt10.json`.

| Measurement | Result |
|---|---:|
| Earliest retained logical releases | 232,024 bytes, 64 files, 32/32 restorable |
| Current-pointer logical releases | 231,928 bytes, 64 files, 32/32 restorable |
| Current complete archive | 8,557,737 bytes, 736 files |
| Retained release generations | 128 across 32 stores |
| Release rebuild wall/write cost | 281.676 seconds; 986,112 written bytes; 813,032 read bytes |

“Initial” means earliest retained generation, not a preserved pre-training archive baseline. Logical bytes are not physical APFS blocks or SSD write amplification. Initial/final manifest sizes differ slightly; all payload and pointer hashes validate, so no overwritten-payload defect is claimed.

## Interruption-safe migration preparation

`src/replica_lab/intrinsic_memory_archive_migration_v16.py` copies one generation through payloads → manifest → independent destination verification → atomic **v12 authority pointer**. Pointer format, scope and authority epoch are preserved exactly. Every source copy is hashed while read, destinations are checked against recorded manifest hashes, and source/destination identities are revalidated before authority. Existing exact files survive interruption without recopying; mismatches, traversal and symlinks are refused.

The dry run selected `v11_fit_00` generation 4, injected an interruption after the manifest, confirmed no pointer existed, resumed to completion with zero bytes/files recopied, and confirmed an idempotent second completion. Thirteen focused tests cover crash recovery, the real `ProducerReleaseCorpus` consumer, rollback, traversal, source mutation, symlink substitution, malformed scopes, boolean/zero/noncanonical generations, nonexistent pointer targets, malformed unpublished history, fresh-process cold reads with the original payload unavailable, correction/revocation, index rebuild and repeat recovery. Dry-run success requires the interruption, absent pre-recovery pointer, completed recovery and zero recopied bytes/files.

## Retention policy

- Preserve raw source archives, unsuccessful trajectories and all historical manifests.
- Only independently restorable generations with valid pointer, manifest and payload hashes are rollback-eligible.
- Retain the current release and previous independently restorable working release when available.
- Historical evidence with any future payload mismatch would be evidence-only, never rollback authority.
- Migration does not authorize deletion. Any deletion candidate requires separate explicit approval after independent verification; none is proposed or performed here.

Full receipt: `reports/intrinsic_memory_migration_v16/preparation_attempt04.json`. Large-scale or incremental production migration remains unmeasured; this preparation establishes bounded correctness and cost evidence only.

## Independent review and attempt02 repair

The independent reviewer rejected attempt01's migration correctness: its pointer was
not consumable by the real release reader, payload names permitted traversal,
copy/final verification had a source-mutation race, resume accepted symlinks,
inventory validation was incomplete and assumed the numerical latest generation,
and dry-run success did not require an actual interruption. Attempt01 is preserved
as `preparation_attempt01.json`; it is not used as passing evidence.

Attempt02 implements all migration corrections and adds real-consumer and
adversarial tests. Rollback pointers select the authority-named generation rather
than the newest directory. No production migration or deletion occurred. A
separate reviewer finding about native result-schema fields is tracked outside
this migration evidence and does not affect the content-key mechanism or archive
contracts.

Final re-review additionally required real `MemoryScope` validation and exact
positive-integer/canonical generation checks. Attempt03 includes those repairs;
attempts01/02 and all review findings remain preserved, not passing evidence.

The final recovery cases also prove that a nonexistent pointer target is
quarantined rather than replaced with another generation, malformed unpublished
history does not hide a healthy committed generation, and a fresh production
process can read migrated cold payloads when the original payload location is
unavailable. Canonical authority, scope, epoch and payload hashes remain bound.
