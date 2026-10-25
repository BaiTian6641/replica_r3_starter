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

The maintenance check passes 65 tests, covering normal evidence additions,
forced additions of ignored artifacts, staged versus working file sizes,
and full-index scans. The full starting index passes the expanded guard.
Native model execution is outside this check.
