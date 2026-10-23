# Git artifacts, source reuse and checkpoint retention

**Later storage status:** [post-D2 retirement](20261022_Checkpoint_Retirement_After_D2.md)
removes 19 newly obsolete payload directories and 10 orphan pool versions after
D2 qualification. Selected results/source and raw evidence remain. Counts and
retention decisions below describe the earlier maintenance snapshot.

**Current status:** the subsequent [authorized cleanup](20261022_Checkpoint_Cleanup_Applied.md)
applied the 16-directory retirement and verified weight sharing. The findings
and initial marking below describe the preceding maintenance snapshot.

22 October 2026, America/Chicago. This maintenance package does not resume
model training, generation, checkpoint compaction or deletion.

## Git findings

The audit at commit `9bc719b` examined all 619 stored Git objects and every tree
in the 19 available local commits, including refs/reflogs. No model/checkpoint
files were tracked in the index or history. The largest blob was an 887,469-byte
memory report. The Git directory used about 4.6 MiB on disk. No LFS objects,
pointers, configuration or remotes were present. A history rewrite and LFS
migration are unnecessary for these local objects.
The [Git object audit](../../reports/git_artifact_audit_20261022.json) records
the baseline before this maintenance commit.

The large local directories are separate from Git: `runs/` contains about
375 GiB of ignored experiments, and the original source is about 4.7 GiB.
Ignoring a file does not remove it from disk. Untracking a file does not recover
local disk space either. The Git index was checked and artifact-directory
untracking was applied with `--cached --ignore-unmatch`; zero model paths
needed removal, and no working files were deleted.

## Protection against future additions

The root [.gitignore](../../.gitignore) excludes local model/run/private/data
directories, caches, environments, weight/state formats and compiled model
packages. Source code, small synthetic fixtures, raw result reports, receipts,
provenance and documentation remain shareable.

The [index guard](../../scripts/check_git_artifacts.py) reads staged Git objects,
not their possibly different working files. It rejects model/state paths even
after `git add -f`, and rejects blobs above the **project limit of 50 MiB**.
This is a local repository policy. It does not rewrite history or delete data.

The [.githooks/pre-commit](../../.githooks/pre-commit) hook is enabled locally.
Git does not distribute local configuration, so enable it again after cloning:

```bash
git config --local core.hooksPath .githooks
.venv/bin/python scripts/check_git_artifacts.py --scope tracked
```

Commit source and reports through Git. Do not upload the entire working folder
including ignored `runs/`, `checkpoints/`, environments or caches.

## Reuse the original source offline

**Keep `checkpoints/minicpm5_source/` and
`checkpoints/minicpm5_source.download.json`.** Keep the metadata/sealed locks
under `configs/`. The existing source revision is
`f97400052a43d642bbc6e9975e2397e3ae6a6b52`.

“Fresh from the sealed parent” means loading these existing local bytes into
a new experiment. It does not mean downloading another copy. All native
trainers/evaluators already load with `local_files_only=True`.

The acquisition helper now defaults to offline verification/reuse:

```bash
.venv/bin/python scripts/download_model.py \
  --model-lock configs/minicpm5_metadata_lock.local.json \
  --sealed-lock configs/minicpm5_sealed_lock.local.json \
  --output checkpoints/minicpm5_source
```

A valid sealed source is reused without importing the download client,
contacting the network, rewriting its receipt or changing source files. A
missing/corrupt/incomplete source fails by default. Only an intentional initial
acquisition or repair uses `--allow-network`; source-identity conflicts still
require investigation rather than automatic replacement. Explicit repair forces
only missing or corrupt sealed repository files; valid original files are
excluded from the acquisition request.

The [actual offline reuse check](../../reports/source_reuse_verification_20261022.json)
verified all nine sealed files with socket access and model/download library
imports blocked. All 36 source files and the acquisition receipt remained
unchanged. No model was loaded or executed.

## Marked cleanup candidates

The [retention inventory](../../reports/checkpoint_retention_inventory_20261022.json)
separates protected artifacts from `DELETE_ELIGIBLE_NOT_DELETED` paths. Small
checkpoint receipts/manifests are archived alongside that inventory before any
future removal. No deletion command or automatic cleanup is run by this package.

There are **16 marked runtime checkpoint directories**, totaling
**192,610,937,126 logical bytes (179.38 GiB)**. The
[plain path list](../../reports/checkpoint_delete_candidates_20261022.txt)
contains only these candidates. Their 32 small receipts/manifests were copied
to `reports/checkpoint_metadata_archive_20261022/` (128,336 bytes), and every
copy's SHA-256 matches its source.

Protected artifacts include the immutable source, selected D0 update-8
checkpoint/export, selected D1 update-28 checkpoint/export, final negative
exports, recent D2 runs whose evidence is not yet fully archived, and run-level
reports/profiles/generations/logs. Redundant completed runtime checkpoints are
marked only when their run is documented and the selected artifacts are
elsewhere. The inventory uses exact nonoverlapping paths.

Deleting a marked runtime directory later removes its optimizer/RNG state and
ability to resume that exact point; the preserved metadata is not a weights
backup. Logical byte totals are estimates of removable file content, not
promised physical recovery on APFS with snapshots or shared extents. This
package does not apply the previously drafted compaction tool.

The subsequent [run-backbone duplication audit](20261022_Run_Base_Duplication.md)
checks repeated native weights inside retained checkpoints as well. Its
dedup-only candidates must preserve file paths and bytes. Those logical totals
overlap this whole-directory cleanup list; use the audit's nonoverlapping
calculation instead of adding the two headline totals.

Training and the paused D2/runtime construction work remain paused until the
user resumes them. Unfinished source drafts are preserved separately from this
maintenance commit.

## Validation

The [scoped validation report](../../reports/git_hygiene_validation_20261022.json)
records **92 passing tests** for index/ignore protection and offline acquisition
reuse, plus source reuse and metadata archive checks. These checks do not resume
the paused model experiments or certify unfinished D2/runtime drafts.
