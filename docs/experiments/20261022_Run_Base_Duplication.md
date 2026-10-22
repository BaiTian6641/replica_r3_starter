# Repeated native backbones in run checkpoints

22 October 2026, America/Chicago. Training remains paused. This audit reads
checkpoint bytes and writes reports only; it does not delete, replace, clone,
load or download model artifacts.

## Why `base/` appears so often

`LlamaMemoryAdapter.save_pretrained` always serializes the entire native model
into `base/`, then saves any added modules into `sidecars.safetensors`.
`save_training_checkpoint` calls that same exporter for every saved training
state. The final inference export also calls it. This creates full backbone
files in both resumable checkpoints and inference exports.

D1 uses `modules_only`: the original backbone is frozen while added memory and
workspace modules learn. Those runs repeatedly save the same original values
as FP32 files. Qualification runs perform zero optimizer updates; their manifest
can still say `full_augmented`, because it describes parameter eligibility,
not proof that the original weights changed. Current file hashes are the
criterion for byte sharing.

The immutable source weight is BF16, about 4.69 GiB. Each saved FP32 native
weight file is about 9.38 GiB. Frozen FP32 exports contain the same numerical
source values, as checked by the earlier
[D1 audit](../../reports/minicpm5_D1_checkpoint_audit_20261022.json), but they
are different files from the original BF16 source. Preserve both the original
source and the exact saved dtype/bytes needed by existing checkpoints.

D0 full fine-tuning and D2 `full_augmented` train the original backbone.
Their `base/` directories contain learned weights. They may duplicate another
export of the same completed update; they must not be replaced with the
original source or another update merely because filenames and sizes match.

## Exact file audit

Run the [read-only audit](../../scripts/audit_checkpoint_storage.py) with:

```sh
.venv/bin/python scripts/audit_checkpoint_storage.py \
  --output reports/base_checkpoint_duplication_audit_20261022.json
```

The [audit report](../../reports/base_checkpoint_duplication_audit_20261022.json)
records full-file SHA-256 hashes, file identities, saved regime, dtype, local
checkpoint receipt hashes when available, exact duplicate groups and proposed
canonical files. Hashes use two bounded streams of 8 MiB chunks. No tensor or
model library is imported. Source and checkpoint metadata are checked before
and after the audit.

A manifest's `run.model_lock.snapshot_files_sha256` describes the parent
source. It does **not** prove that a fine-tuned export still has the original
weights. A checkpoint receipt's `files_sha256` describes that checkpoint's
actual payload. This audit verifies current payload bytes, including final
exports without training receipts.

The completed pass verified **35 weight files** (328.15 GiB logical) and found
**10 distinct byte groups**. All 18 D1/qualification source-value FP32 files
are identical. Eight additional groups each contain two identical exports of
a trained update. The original negative D0 checkpoint forms the remaining
single-file group.

| Storage category | Files | Logical GiB |
|---|---:|---:|
| Repeated weights beyond one file per byte group | 25 | 234.39 |
| Repeated weights already inside the earlier cleanup directories | 16 | 150.01 |
| Additional repeated weights outside that cleanup list | 9 | 84.38 |
| Earlier whole-directory cleanup, including runtime state | 16 directories | 179.38 |
| Combined cleanup and sharing, with overlap removed | — | 263.76 |

These are logical upper bounds. No physical space has been reclaimed or measured.
The original 16-directory deletion list is unchanged; the 25-file list marks
sharing candidates, not additional files to delete. The
[storage validation report](../../reports/base_checkpoint_storage_validation_20261022.json)
checks the group totals, receipt claims, current file identities and candidate
mapping without repeating the large-file hash pass.

## Retention and compatible storage sharing

The existing [whole-directory cleanup list](../../reports/checkpoint_delete_candidates_20261022.txt)
marks obsolete training states. A second
[deduplication candidate list](../../reports/base_checkpoint_dedup_candidates_20261022.json)
marks individual repeated weight files whose paths and bytes must be preserved.
These are different operations. Their overlap is calculated in the audit;
do not add both totals without subtracting that overlap.

Existing loaders expect the `base/` path, config/tokenizer files, sidecars and
runtime files. Existing receipts hash those bytes. Deleting a repeated
`model.safetensors` file from an otherwise retained checkpoint breaks it.
Deleting a whole directory also loses its optimizer/RNG state and resume point.
Keep selected D0/D1 artifacts, final negative exports, recent D2 checkpoints,
raw reports and the original source according to the existing retention policy.

APFS copy-on-write clones are a compatible storage option: separate files can
share data blocks, while subsequent writes remain private to each file.
[Apple's `clonefile(2)` manual](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/man/man2/clonefile.2)
documents these semantics. An implementation must keep current paths and exact
bytes, check SHA-256 before and after replacement, and operate while all
checkpoint users are inactive. Symlinks make retention/portability dependent
on another path. Hardlinks couple writes. Substituting BF16 source files for
FP32 files changes dtype, serialization and receipt hashes.

The earlier uncommitted compaction draft covers only the four D1 runs and two
qualification runs, using one reference hash. It does not cover arbitrary
D0/D2 groups. It remains a paused draft; this audit does not apply or broaden it.
Physical recovery has not been measured. Existing APFS sharing, snapshots and
open files can make recovered space smaller than the logical upper bound.

## Work before resuming model experiments

1. Retain this hash inventory and the existing cleanup evidence. Keep marked
   paths present until the corresponding storage operation is performed.
2. For retained checkpoints, prepare exact-hash groups for APFS file cloning,
   preserving the original source, independent sidecars and optimizer/RNG state.
   Verify receipt inventories and loader compatibility after each replacement.
3. Change future checkpoint publication to reuse a verified immutable FP32
   backbone for frozen D1 saves, and one native weight snapshot per completed
   update for full fine-tuning. Use independent copy-on-write files before
   receipt publication. Different learned updates require different snapshots.
4. Validate that writer change on tiny files and then with the selected local
   checkpoint on MPS when experiments resume. No new source download is needed.
   Record logical size and actual free-space changes separately.

No checkpoint payload is changed by this documentation or audit. Storage
sharing and the future writer change remain work to execute before further
large checkpoint-producing experiments.
