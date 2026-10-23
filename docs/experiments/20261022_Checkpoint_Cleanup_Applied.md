# Applied checkpoint cleanup and shared future saves

**Later storage status:** [post-D2 retirement](20261022_Checkpoint_Retirement_After_D2.md)
removes 19 newly obsolete payload directories and 10 orphan pool versions after
D2 qualification. Selected results/source and raw evidence remain. Counts and
retention decisions below describe the earlier maintenance snapshot.

22 October 2026, America/Chicago. The user authorized exact-byte deduplication,
deletion of the previously marked old checkpoints, preservation of original and
selected results, and disk measurements. **Training remains paused.** No
MiniCPM model was loaded, trained, evaluated or downloaded during this operation.

## Applied scope and measured disk space

The cleanup deleted **16 obsolete runtime directories** (179.38 GiB logical)
and deduplicated **nine retained weight files** (84.38 GiB logical) using APFS
clones. The pool was then seeded with **ten existing weight versions**.

| Measurement phase | Free GiB |
|---|---:|
| Before cleanup | 264.03 |
| After nine retained-file clones | 348.39 |
| After old-directory deletion and verification | 527.79 |
| After pool seeding | 527.79 |

**Observed net free-space increase: 263.76 GiB**
(283,211,587,584 bytes), including pool seeding. The pool's 93.76 GiB of logical
weight filenames consumed no additional full weight copies; measured free
space changed by only about 1.77 MiB during seeding, including metadata/system activity.
Free-space observations are system-wide; logical size and `du` totals cannot
measure unique APFS shared extents. See the
[results record](../../reports/checkpoint_storage_results_20261022.json).


The [cleanup journal](../../reports/checkpoint_cleanup_apply_20261022.json)
records each pending and completed operation. It pins the three approved
inventory/audit files by SHA-256, validates the 32 archived metadata copies,
checks that model workers are inactive, and refuses paths outside the approved
16-directory / 9-file scope. Its metadata-only
[plan](../../reports/checkpoint_cleanup_plan_20261022.json) is also retained.

The operation keeps one verified native weight file per exact byte group.
Each retained duplicate keeps its own regular-file inode, original path, mode,
mtime and bytes while sharing APFS blocks. There are no hardlinks or symlinks.
Verified rollback clones remain until installed bytes pass their hash check.
Obsolete runtime directories are individually rechecked, renamed to a private
quarantine name in the same run and removed with fd-relative recursive deletion.
The journal names temporary/recovery paths before these actions. Its writes are
atomic process-progress records, not a promise of power-loss durability.

The earlier inventory and duplicate reports remain immutable historical
snapshots. Their `NOT_DELETED` and candidate status describe the state before
this operation; use this completion journal for current cleanup state.
The old runtime directories' optimizer/RNG states and exact resume points are
now retired. Their 32 receipts/manifests and raw run-level evidence remain.

## What was preserved

- Original BF16 `checkpoints/minicpm5_source`, acquisition receipt and metadata /
  sealed locks. The source was validated offline before and after cleanup.
- Entire selected D0 update-8 run, including inference export and resumable state.
- Selected D1 update-28 export, sidecars and resumable state.
- Both final negative exports, all other retained inference exports, and all
  recent D2 capacity evidence and resumable states.
- Run-level reports, answers, profiles, logs, tokenizer/config files and archived
  metadata. Deduplication applies only to matching native weight bytes.

All four retained training checkpoint receipts are checked against their full
payloads before deletion and again afterward. The original and other protected
file identities remain unchanged; the nine explicitly deduplicated weights
change inode/ctime while preserving SHA-256 bytes. No generated-answer result
is re-scored or promoted by a storage operation.

## Future saves share storage automatically

[checkpoint_storage.py](../../src/replica_lab/checkpoint_storage.py) is called by
`LlamaMemoryAdapter.save_pretrained` after ordinary native serialization and
before sidecar/manifest and training-receipt publication. On this Mac, exports
inside `runs/` use a SHA-256-addressed pool at `runs/.checkpoint_storage/`.
Same-byte native shards share verified APFS clone extents; changed trained
versions receive different entries. A per-hash lock serializes publication.
The decision uses complete file hashes, never tensor identity, version counters,
filename, size or a claim that a backbone was frozen.

The [pool seed record](../../reports/checkpoint_weight_pool_seed_20261022.json)
initializes the pool from the ten retained audited byte groups through COW clones.
This lets future matching saves share blocks with the existing results as well.
Pool entries are additional logical filenames, not additional full physical
copies. Checkpoints remain independently loadable if the pool is later removed.

Existing self-contained paths, shard indexes and payload receipts stay valid.
Sidecars and optimizer/RNG state remain independent. The manifest records the
storage backend/action. Unsupported clone operations keep a valid ordinary
save and record that fallback; hash or integrity conflicts prevent publication.
No backend substitutes an ordinary copy or hardlink for an APFS clone.

Normal serialization still needs temporary disk space and I/O before deduplication.
The pool retains each unique version's blocks until separate, explicit garbage
collection; future cleanup must account for pool entries as well as checkpoint
paths. Do not remove pool contents merely from their size or filename.
Exports outside this workspace's `runs/`, and non-macOS environments, retain
ordinary-save behavior; the later CUDA filesystem backend is not qualified here.

The new helper is included in future training implementation hashes. Historical
runtime receipts retain their original strict code contract. Exact continuation
uses the recorded Git implementation, or a separately qualified migration; do
not relabel old D2 receipts to pretend the implementation is unchanged.

## Validation and reproduction

The complete fixture/unit suite passes **446 tests**, with **one skip**:
`test_interrupted_training_matches_uninterrupted_with_rng_batches[mps]`
reports MPS unavailable in the restricted test context. No native MiniCPM test
or fallback CPU model run was substituted. Real tiny-file APFS cloning and
private writes passed. See the
[test evidence](../../reports/tests_checkpoint_storage_20261022.xml) and
[delivery validation](../../reports/bundle_validation_checkpoint_storage_20261022.json).


No native MPS training/generation or ANE work was resumed. Small test fixtures
exercise serialization, COW private writes, pool independence, changed weight
versions, concurrency, corrupt/failed clones, rollback and receipt compatibility.
The real cleanup additionally verifies full saved payload bytes without tensor
or model imports.

The applied driver is intentionally pinned and one-shot. Repeating its command
now refuses missing retired paths; future cleanup requires a fresh inventory.
The commands executed for this operation were:

```sh
.venv/bin/python scripts/apply_checkpoint_storage_cleanup.py \
  --output reports/checkpoint_cleanup_plan_20261022.json
.venv/bin/python scripts/apply_checkpoint_storage_cleanup.py --apply \
  --output reports/checkpoint_cleanup_apply_20261022.json
.venv/bin/python scripts/seed_checkpoint_weight_pool.py --apply
```

The two interrupted D2/runtime source drafts remain outside this storage
commit. The earlier compaction helper and its safety tests are adopted here;
its original narrow CLI has not been broadened into a recursive cleanup tool.
