#!/usr/bin/env python3
"""Profile full v12 WorkLedger replay on a disposable copy of preserved evidence.

Run only while no native worker is active. The source journal is hashed and never
modified. No model import, authorization, native execution, or capability claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import tempfile
import time

from replica_lab.intrinsic_memory_budget_v12 import WorkLedger, atomic


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile(source: Path, output: Path, repeats: int) -> dict:
    if output.exists():
        raise ValueError("profile receipt must be fresh")
    source_sha256 = sha256(source)
    with source.open("rb") as stream:
        first = stream.readline()
    plan = json.loads(first)
    identity = plan["identity"]
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="v12-ledger-profile-") as directory:
        copied = Path(directory) / source.name
        shutil.copyfile(source, copied)
        ledger = WorkLedger(copied, identity=identity)
        samples = []
        document = None
        for _ in range(repeats):
            before = time.perf_counter()
            document = ledger.snapshot()  # uncached: the current full replay path
            samples.append(time.perf_counter() - before)
        result = {
            "format": "replica.intrinsic_memory.ledger_replay_profile.v1",
            "source_path": str(source),
            "source_sha256": source_sha256,
            "source_bytes": source.stat().st_size,
            "identity_sha256": hashlib.sha256(identity.encode()).hexdigest(),
            "events": len(document["events"]),
            "journal_records": document["journal"]["append_calls"],
            "stages_replayed": len(document["stages"]),
            "repeats": repeats,
            "full_replay_seconds": samples,
            "full_replay_seconds_total": sum(samples),
            "wall_seconds": time.perf_counter() - started,
            "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "source_modified": sha256(source) != source_sha256,
            "native_execution": False,
            "note": "Each sample reparses and validates every complete journal record; successor comparisons must use the same preserved source and repeat count.",
        }
    result["disposable_copy_removed"] = not copied.exists()
    atomic(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("at least one replay sample required")
    result = profile(args.ledger.resolve(), args.output.resolve(), args.repeats)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
