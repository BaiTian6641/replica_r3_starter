#!/usr/bin/env python3
"""Measure release rebuild cost and retained archives from preserved v12 evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from replica_lab.intrinsic_memory_budget_v12 import atomic
from replica_lab.intrinsic_memory_budget_v13 import IncrementalWorkLedger


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(run: Path):
    ledger_path = run / "work_ledger.jsonl"
    with ledger_path.open("rb") as stream:
        identity = json.loads(stream.readline())["identity"]
    document = IncrementalWorkLedger(ledger_path, identity=identity).snapshot(cached=True)
    rebuilds = [event for event in document["events"]
                if event["operation"] == "producer_release_rebuild"]
    stages = {}
    for event in rebuilds:
        stage = stages.setdefault(event["stage"], {
            "rebuilds": 0, "wall_seconds": 0.0, "artifact_write_bytes": 0,
            "artifact_read_bytes": 0, "artifact_write_files": 0, "publications": 0,
            "wall_seconds_per_rebuild": [],
        })
        cost = event["detail"]["corpus_cost"]
        stage["rebuilds"] += 1
        stage["wall_seconds"] += event["elapsed_seconds"]
        stage["wall_seconds_per_rebuild"].append(event["elapsed_seconds"])
        for key in ("artifact_write_bytes", "artifact_read_bytes", "artifact_write_files", "publications"):
            stage[key] += cost[key]
    for stage in stages.values():
        count = stage["rebuilds"]
        stage["wall_seconds_mean"] = stage["wall_seconds"] / count
        stage["wall_seconds_min"] = min(stage["wall_seconds_per_rebuild"])
        stage["wall_seconds_max"] = max(stage["wall_seconds_per_rebuild"])
        stage["artifact_write_bytes_mean"] = stage["artifact_write_bytes"] / count
    stores = run / "stores"
    store_files = [path for path in stores.rglob("*") if path.is_file()]
    generations = [path for path in stores.rglob("*")
                   if path.is_dir() and len(path.name) == 8 and path.name.isdigit()]
    slots = [path for path in store_files if path.suffix == ".slots"]
    manifests = [path for path in generations for path in [path / "manifest.json"] if path.is_file()]
    return {
        "format": "replica.intrinsic_memory.release_cost_measurement.v13",
        "run_root": str(run), "ledger_sha256": sha256(ledger_path),
        "rebuilds": len(rebuilds),
        "stages": stages,
        "totals": {
            "wall_seconds": sum(event["elapsed_seconds"] for event in rebuilds),
            "artifact_write_bytes": sum(event["detail"]["corpus_cost"]["artifact_write_bytes"]
                                        for event in rebuilds),
            "artifact_read_bytes": sum(event["detail"]["corpus_cost"]["artifact_read_bytes"]
                                       for event in rebuilds),
            "artifact_write_files": sum(event["detail"]["corpus_cost"]["artifact_write_files"]
                                        for event in rebuilds),
            "publications": sum(event["detail"]["corpus_cost"]["publications"]
                                for event in rebuilds),
        },
        "current_archive": {
            "stores": len([path for path in stores.iterdir() if path.is_dir()]),
            "files": len(store_files), "bytes": sum(path.stat().st_size for path in store_files),
            "retained_release_generations": len(generations),
            "slot_files": len(slots), "slot_bytes": sum(path.stat().st_size for path in slots),
            "release_manifest_files": len(manifests),
            "release_manifest_bytes": sum(path.stat().st_size for path in manifests),
            "release_generations_per_world": sorted({len(list(path.glob("000000*")))
                for path in stores.glob("*/artifacts/contextual_v12/releases/*")}),
        },
        "scope": {
            "native_fitting_and_restart_evidence": True,
            "large_scale_migration_measured": False,
            "incremental_migration_measured": False,
            "initial_archive_baseline_preserved": False,
            "interpretation": "Bounded rebuild wall/write cost and current retained size are measured. A growth rate or incremental migration claim requires a separate corpus with initial/final snapshots.",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("release cost receipt must be fresh")
    result = measure(args.run.resolve())
    atomic(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
