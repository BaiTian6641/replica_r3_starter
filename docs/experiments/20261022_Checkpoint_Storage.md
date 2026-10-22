# Byte-identical checkpoint storage preparation — 22 October 2026

**Historical preparation:** the narrower tool described below supplied tested
clone/hash primitives for the subsequent [authorized cleanup](20261022_Checkpoint_Cleanup_Applied.md).
The pinned apply driver now records the actual nine retained replacements and
16 retired runtime directories. The 18-file inventory below predates cleanup.

Read-only inventory found **18 native FP32 weight files**, each 10,067,069,720
bytes (9.3757 GiB): 16 across the four completed D1 run trees and one in each
native adapter/generation qualification export. Their logical total is 168.7624
GiB. Keeping one canonical file and replacing 17 identical duplicates with APFS
clones could release **at most 159.3867 GiB** of duplicate extents. This is an
upper bound, not measured recovery: existing shared extents, APFS snapshots and
open old files can reduce or delay the disk-free increase. The inventory's
`df -k .` reading was 130,801,024 KiB available, about 124.74 GiB.

The [D1 audit](../../reports/minicpm5_D1_checkpoint_audit_20261022.json) proves
the four final D1 exports have identical native bytes. Intermediate checkpoints
and qualification exports still require fresh streaming hashes before any
replacement; equal file sizes alone are insufficient. The known final native
SHA256 is `07db90adf475eae2201fc3f3fbe76849c12cf1777c4e271114af142a048dfbb8`.

## Tool and safety boundary

[compact_identical_checkpoints.py](../../scripts/compact_identical_checkpoints.py)
defaults to planning. It requires explicit run roots and a reference native file;
the CLI allowlist contains only these four D1 runs and the two completed
qualification runs. It rejects unknown/broad roots, incomplete reports, visible
processes naming a selected run, symlinks, path traversal, existing hard links
and cross-filesystem candidates. Only regular FP32 native
`{checkpoint,training_checkpoint_NNNN,adapter_export}/base/model*.safetensors`
files qualify. Sidecars, optimizer/RNG runtime files, manifests, receipts and
sealed source checkpoints are never replacement targets. The reference remains
untouched.

Apple documents clone files as separate objects whose later writes are private.
They share storage with copy-on-write semantics. The public `clonefile(2)` call
fails on unsupported filesystems; the tool calls it directly through the system
library. This avoids `cp -c`, which can silently fall back to a full copy.
There is **no hard-link or ordinary-copy fallback**. See Apple's
[clonefile manual](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/man/man2/clonefile.2)
and [cp manual](https://github.com/apple-oss-distributions/file_cmds/blob/main/cp/cp.1).

Every same-size candidate is streamed through SHA256. Apply rechecks original
file identity and bytes, creates a fresh clone in a private temporary directory
inside the target's parent, verifies that clone's bytes and independent inode,
and keeps a verified COW rollback copy of the old target. It preserves target
mode and mtime, then uses `os.replace` and verifies the installed bytes again.
Clone/verification failure before replacement leaves the old file intact;
post-replacement verification failure rolls back. An OS error during rollback
retains the original COW copy and reports its recovery path. Atomicity is per
file: already completed replacements are retained if a later file fails.

Successful replacement changes the inode/ctime, while identical file bytes keep
existing SHA256 receipts valid. This is not a generic metadata archival tool.
The process scan and identity checks require quiescent checkpoints; they are not
a lock against uncooperative writers or a concurrent directory replacement.
Keep all selected runs inactive throughout planning and apply.

## Reviewable commands

Run from the workspace root. These arguments name all eligible inventory roots;
there is no recursive whole-workspace selection.

```sh
compaction_args=(
  --run-root runs/minicpm5_D1_mask1_B8_lr1e3_chunk01_20261022
  --run-root runs/minicpm5_D1_mask1_B8_lr1e3_chunk02_20261022
  --run-root runs/minicpm5_D1_mask1_B8_lr1e2_chunk01_20261022
  --run-root runs/minicpm5_D1_mask1_B8_lr1e2_chunk02_20261022
  --run-root runs/minicpm5_augmented_generation_qualification256_20261022
  --run-root runs/minicpm5_native_adapter_qualification256_20261022
  --reference runs/minicpm5_augmented_generation_qualification256_20261022/adapter_export/base/model.safetensors
  --expected-sha256 07db90adf475eae2201fc3f3fbe76849c12cf1777c4e271114af142a048dfbb8
)
.venv/bin/python scripts/compact_identical_checkpoints.py \
  "${compaction_args[@]}" --output reports/checkpoint_storage_plan_20261022.json
```

After reviewing the plan, the explicit mutation command rebuilds and verifies
its own plan and writes a fresh report:

```sh
.venv/bin/python scripts/compact_identical_checkpoints.py \
  "${compaction_args[@]}" --apply --output reports/checkpoint_storage_apply_20261022.json
```

Reports include every candidate's planned action/hash/replacement flag, verified
before/after hashes and identities for replacements, duplicate-byte upper bound,
before/after disk-free readings, completed replacements and failures. The disk-free
delta is system-wide and is not solely attributable to this operation.

Preparation validation: **30 file-only CPU tests passed** in
[test_checkpoint_compaction.py](../../tests/test_checkpoint_compaction.py), using
tiny mock copies to test byte/receipt preservation, path restrictions, failed
clones, corrupt clones, hard-link rejection, stale plans, active-process rejection
and rollback/recovery. These tests do not demonstrate APFS sharing or physical
space recovery. No existing checkpoint was replaced during preparation; the
actual APFS apply and resulting free space must be recorded separately.
