# Repository review and staged commit policy

25 October 2026, America/Chicago. This maintenance records existing work and
protects future commits; it does not launch training or acquire model payloads.

## What belongs in Git

Track implementation, tests, configuration and source locks, documentation,
small synthetic fixtures, and experiment evidence. In particular, preserve
failed attempts, guard logs, host samples, verdicts, JUnit results, receipts and
provenance under `reports/`. They support the recorded scientific claims and
are not disposable merely because they are generated. Small procedural PNGs
and diagram renders are intentional source fixtures and documentation assets.

Keep checkpoints, model weights, optimizer/runtime state, private data,
downloaded corpora, local LFS object stores, environment files, virtual
environments, caches, build output, editor backups and ad hoc output outside
tracking. Use `data/raw/`, `data/processed/`, `data/downloads/`, `data/external/`,
`data/cache/`, `runs/`, `checkpoints/`, `logs/` or `reports/local/` as appropriate.
The protected `src/replica_lab/runtime_pilot.py` stays local and ignored.
Ignoring a file does not delete it or remove previously committed bytes.

## Large files and LFS

The audit of the starting repository (`3f1433b`) found 3,391 tracked files and
5,649 stored blobs, including unreachable objects. No stored blob exceeded the
50 MiB project limit. The largest was the 8,148,167-byte synthetic fixture
`data/e2_capacity8192/records.jsonl`. No tracked file matched the initial ignore
rules, and no tracked cache or compiled Python file required removal.

Git LFS is not installed and this repository has no configured remote. The
current policy deliberately keeps model/checkpoint payloads entirely outside
Git, including LFS; it does not require an LFS migration. LFS still versions
pointer files and needs separate object storage. If a future reviewed asset
must be versioned using LFS, install and configure LFS, declare its exact path
in `.gitattributes`, and update the guard and storage policy together. A broad
`*.bin`/weight LFS rule must not silently admit local model state.

No history rewrite, garbage collection, payload deletion or push is part of
this maintenance. The index guard rejects force-added payloads and local
scratch paths as well as oversized Git blobs. It reads staged object sizes,
so shrinking a working file after staging cannot evade the size check.

The later clean-export check found two absolute symlinks under
`reports/memory_reliability_v2_attempt03/smoke_fixture/run/`: `candidate_extension`
and `images`. Their targets are ignored local `runs/` payloads. Both links are
now ignored and removed from the index with `git rm --cached`; the working links,
targets and adjacent historical reports remain intact. Old commits contain only
the 88-byte and 75-byte link texts, not their payloads. The guard now checks
staged link targets, rejecting absolute paths, repository escapes and links to
excluded local artifacts while allowing relative links to source files.

## Commit each completed package

1. Inspect `git status --short` and the diff; identify the files owned by the
   package. Preserve concurrent work and the protected pilot.
2. Finish one coherent package and run the checks relevant to it. Record the
   command, result and limits in `execution_state.md` or its dated evidence.
3. Stage explicit paths, then review `git diff --cached --stat`,
   `git diff --cached --check` and the staged diff. Avoid repository-wide
   staging in a shared, changing checkout.
4. Run the index guard and commit promptly after that package passes. Split
   artifact policy, historical evidence, design/research, source preparation
   and data preparation into separate commits. A timed auto-commit must not
   capture incomplete code or unreviewed output.
5. Recheck status after every commit. Commit experimental protocols before
   launching the experiment; record later results in a separate commit.
   Committing old results now does not retroactively preregister a protocol.

Enable the versioned hook in every clone:

```bash
git config --local core.hooksPath .githooks
.venv/bin/python scripts/check_git_artifacts.py --scope tracked
.venv/bin/python -m pytest tests/test_git_artifacts.py -q -p no:cacheprovider
```

The final maintenance check passes 74 tests, covering normal evidence additions,
forced additions of ignored artifacts, staged versus working file sizes,
full-index scans and staged symbolic-link targets. The expanded guard passes
all 4,752 tracked files after the two local links are untracked.
Native model execution is outside this check.

## Verify a relocated checkout

A clean export at `d163909` passed 111 focused tests, then exposed a separate
visual-data audit defect: the stored catalog/registry locators contained the
original absolute checkout path. The follow-up makes new allowlists use relative
paths and permits historical absolute locators only when both share one checkout
prefix and retain the configured relative source paths. Source hashes, admission
fields and every other allowlist value still have to match recomputation.
Existing assets, manifests and historical reports are not rewritten.

After cloning or exporting, audit the committed fixtures into a fresh local
report path:

```bash
.venv/bin/python scripts/prepare_visual_data.py --audit-only \
  --report reports/local/visual-data-audit.json
```

The focused visual-data suite passes 19 tests, including relocated checkout
audit/regeneration and rejection of changed source paths, hashes, mixed checkout
roots and acquisition flags. This qualifies fixture portability, not native
visual model behavior or training admission.

The final clean export at `7925cfd` passes **116 focused tests** and the complete
visual-data audit. The [audit record](../../reports/repository_hygiene_20261025/audit.json)
includes the document inventory, all-source syntax checks, historical hash
verification, stored-blob sizes, preserved local links and validation scope.
The [JUnit evidence](../../reports/repository_hygiene_20261025/clean_export_tests.xml)
comes from the relocated export, excluding active concurrent V0 implementation.
