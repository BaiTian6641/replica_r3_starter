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

`src/replica_lab/intrinsic_memory_archive_migration_v16.py` copies one generation through payloads → manifest → independent destination verification → atomic pointer. Each file is hash-verified and durable before the next stage. Existing exact files survive interruption without recopying; mismatches are refused. The source is read-only and the pointer is never committed before complete verification.

The dry run selected `v11_fit_00` generation 4, injected an interruption after the manifest, confirmed no pointer existed, resumed to completion with zero bytes recopied, and confirmed an idempotent second completion. The temporary destination was removed. Three focused tests cover crash recovery, tamper refusal and earliest/current inventory distinctions.

## Retention policy

- Preserve raw source archives, unsuccessful trajectories and all historical manifests.
- Only independently restorable generations with valid pointer, manifest and payload hashes are rollback-eligible.
- Retain the current release and previous independently restorable working release when available.
- Historical evidence with any future payload mismatch would be evidence-only, never rollback authority.
- Migration does not authorize deletion. Any deletion candidate requires separate explicit approval after independent verification; none is proposed or performed here.

Full receipt: `reports/intrinsic_memory_migration_v16/preparation.json`. Large-scale or incremental production migration remains unmeasured; this preparation establishes bounded correctness and cost evidence only.
