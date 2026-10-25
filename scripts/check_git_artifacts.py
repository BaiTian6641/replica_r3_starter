#!/usr/bin/env python3
"""Reject local model artifacts and oversized blobs from the actual Git index.

No working file is removed or read as model data. The local 50 MiB limit is a
project policy. This is a commit guard, not a history rewriter or LFS migrator.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import PurePosixPath
import subprocess
import sys

MAX_BLOB_BYTES = 50 * 1024 * 1024
LOCAL_ROOTS = {"checkpoints", "models", "runs", "private", ".venv", ".cache",
               ".hf_cache", ".uv-cache", ".lfs", "lfs", "artifacts", "logs",
               ".agents", ".codex"}
GENERATED_DIRECTORIES = {"venv", ".tox", ".nox", "__pycache__", ".pytest_cache",
                         ".mypy_cache", ".ruff_cache", ".hypothesis",
                         ".ipynb_checkpoints", "htmlcov", "build", "dist", ".host_tmp",
                         ".cache", ".hf_cache", ".uv-cache"}
WEIGHT_SUFFIXES = {".safetensors", ".pt", ".pth", ".ckpt", ".gguf", ".ggml",
                   ".onnx", ".mlmodel", ".weights", ".h5", ".hdf5", ".npy",
                   ".npz", ".arrow", ".parquet", ".pkl", ".pickle", ".joblib",
                   ".tflite", ".engine", ".ort"}
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".7z", ".zst")


def git(*arguments, input=None):
    return subprocess.run(["git", *arguments], input=input, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True).stdout


def artifact_reason(name):
    path = PurePosixPath(name)
    parts = tuple(part.casefold() for part in path.parts)
    if (parts and (parts[0] in LOCAL_ROOTS
                   or parts[0].startswith(".venv"))):
        return "local artifact directory"
    if parts[:2] in {("data", name) for name in ("raw", "processed", "downloads", "external", "cache")}:
        return "local dataset directory"
    if parts[:2] in {("reports", "local"), ("reports", "tmp")}:
        return "ad hoc local report directory"
    if (any(part in GENERATED_DIRECTORIES or part.startswith(".venv")
            or part.endswith(".egg-info") for part in parts[:-1])):
        return "generated environment, cache or build directory"
    if parts == ("src", "replica_lab", "runtime_pilot.py"):
        return "protected local pilot draft"
    basename = path.name.casefold()
    if basename != ".env.example" and (basename == ".env" or basename.startswith(".env.")):
        return "local environment or credential file"
    if (basename in {".ds_store", "thumbs.db", ".coverage", "coverage.xml"}
            or basename.startswith(".coverage.")
            or basename.endswith((".pyc", ".pyo", ".pyd", ".swp", ".swo", "~"))):
        return "generated cache, coverage or editor file"
    if path.suffix.lower().startswith(".sqlite"):
        return "local database"
    if basename.endswith(ARCHIVE_SUFFIXES):
        return "bulk archive or backup"
    if (path.suffix.lower() in WEIGHT_SUFFIXES
            or any(p.lower().endswith((".mlpackage", ".mlmodelc")) for p in path.parts)
            or any(fnmatch.fnmatch(path.name.lower(), p) for p in
                   ("pytorch_model*.bin", "adapter_model*.bin", "optimizer*.bin", "training_args.bin"))):
        return "model/state or binary dataset artifact"
    return None


def audit_index(scope="staged", max_bytes=MAX_BLOB_BYTES):
    changed = None
    if scope == "staged":
        changed = set(git("diff", "--cached", "--name-only", "--no-renames",
                          "--diff-filter=ACMRT", "-z").decode().split("\0"))
    entries, violations = [], []
    for record in git("ls-files", "--stage", "-z").decode().split("\0"):
        if not record:
            continue
        metadata, name = record.split("\t", 1)
        mode, oid, stage = metadata.split()
        if changed is not None and name not in changed:
            continue
        if stage != "0":
            violations.append({"path": name, "reason": "unmerged index entry"})
        elif mode != "160000":  # A submodule entry stores a commit ID, not its files.
            entries.append((name, oid))
    sizes = {}
    if entries:
        objects = sorted({oid for _, oid in entries})
        response = git("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
                       input=("\n".join(objects) + "\n").encode()).decode()
        for line in response.splitlines():
            fields = line.split()
            if len(fields) == 3 and fields[1] == "blob":
                sizes[fields[0]] = int(fields[2])
    for name, oid in entries:
        reasons = []
        if artifact_reason(name):
            reasons.append(artifact_reason(name))
        if oid not in sizes:
            reasons.append("index object is missing or is not a blob")
        elif sizes[oid] > max_bytes:
            reasons.append("blob exceeds the project size limit")
        if reasons:
            violations.append({"path": name, "object": oid,
                               "bytes": sizes.get(oid), "reason": "; ".join(reasons)})
    return {"status": "passed" if not violations else "failed", "scope": scope,
            "max_blob_bytes": max_bytes, "checked_files": len(entries),
            "largest_checked_blob_bytes": max(sizes.values(), default=0),
            "violations": violations, "working_files_changed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("staged", "tracked"), default="staged")
    parser.add_argument("--max-bytes", type=int, default=MAX_BLOB_BYTES)
    args = parser.parse_args()
    if args.max_bytes < 1:
        parser.error("max-bytes must be positive")
    report = audit_index(args.scope, args.max_bytes)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
