#!/usr/bin/env python3
"""Compare v12 full replay and v13 incremental appends on identical evidence copies."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import tempfile
import time

from replica_lab.intrinsic_memory_budget_v12 import WorkLedger, atomic
from replica_lab.intrinsic_memory_budget_v13 import IncrementalWorkLedger


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reanchor_copy(path: Path) -> None:
    """Shift copy timestamps by one offset; relative budgets stay identical."""
    records = [json.loads(line) for line in path.read_text().splitlines()]
    clock = records[0]["clock"]
    monotonic_delta = time.monotonic() - clock["started_monotonic"]
    unix_delta = time.time() - clock["started_unix"]
    for key in ("started_monotonic", "deadline_monotonic"):
        clock[key] += monotonic_delta
    for key in ("started_unix", "deadline_unix"):
        clock[key] += unix_delta
    for record in records[1:]:
        for key in ("started", "deadline", "at"):
            if key in record and record[key] is not None:
                record[key] += monotonic_delta
        if "deadline_unix" in record:
            record["deadline_unix"] += unix_delta
    path.write_text("".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                            for record in records))

def measured_appends(ledger, stage: str, count: int):
    reads = 0
    original = ledger._read
    def counted():
        nonlocal reads
        reads += 1
        return original()
    ledger._read = counted
    ledger.snapshot(cached=True)  # warm the initial preserved state
    samples = []
    for index in range(count):
        before = time.perf_counter()
        token = ledger.reserve(stage, f"successor_profile_{index}", writes=1)
        ledger.finish(token, actual={"writes": 1})
        samples.append(time.perf_counter() - before)
    return samples, reads, ledger.snapshot(cached=True)

def semantic_state(doc):
    return {
        "plan": doc["plan"],
        "reserved": doc["reserved"],
        "reserved_wall_seconds": doc["reserved_wall_seconds"],
        "torn_records": doc["torn_records"],
        "stages": {name: {"status": state["status"], "reserved": state["reserved"]}
                   for name, state in doc["stages"].items()},
        "events": [{key: value for key, value in event.items()
                    if key not in ("started", "elapsed_seconds", "pid")}
                   for event in doc["events"]],
    }


def compare(source: Path, output: Path, count: int) -> dict:
    if output.exists():
        raise ValueError("comparison receipt must be fresh")
    source_sha256 = sha256(source)
    with source.open("rb") as stream:
        identity = json.loads(stream.readline())["identity"]
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="v13-ledger-compare-") as directory:
        first = Path(directory) / "v12.jsonl"
        second = Path(directory) / "v13.jsonl"
        shutil.copyfile(source, first)
        shutil.copyfile(source, second)
        reanchor_copy(first)
        reanchor_copy(second)
        old = WorkLedger(first, identity=identity)
        new = IncrementalWorkLedger(second, identity=identity)
        initial = old.snapshot(cached=True)
        active = [name for name, state in initial["stages"].items()
                  if state["status"] == "active"]
        if len(active) != 1:
            raise ValueError("comparison requires exactly one active stage")
        stage = active[0]
        remaining = initial["plan"][stage]["writes"] - initial["stages"][stage]["reserved"]["writes"]
        if remaining < count:
            raise ValueError("insufficient declared write capacity for comparison")
        old_seconds, old_reads, old_doc = measured_appends(old, stage, count)
        new_seconds, new_reads, new_doc = measured_appends(new, stage, count)
        # Independent full replay must agree with both incremental/cached views.
        independent = WorkLedger(second, identity=identity).snapshot()
        result = {
            "format": "replica.intrinsic_memory.ledger_replay_comparison.v13",
            "source_path": str(source), "source_sha256": source_sha256,
            "source_bytes": source.stat().st_size,
            "initial_events": len(initial["events"]),
            "initial_journal_records": initial["journal"]["append_calls"],
            "stage": stage, "append_pairs_per_implementation": count,
            "v12_seconds": old_seconds, "v13_seconds": new_seconds,
            "v12_seconds_total": sum(old_seconds), "v13_seconds_total": sum(new_seconds),
            "speedup_v12_over_v13": (sum(old_seconds) / sum(new_seconds)
                                     if sum(new_seconds) > 0 else None),
            "v12_full_reads_after_warm_including_final": old_reads,
            "v13_full_reads_after_warm_including_final": new_reads,
            "final_semantic_state_equal": (semantic_state(old_doc) == semantic_state(new_doc)
                                           == semantic_state(independent)),
            "events_added_per_implementation": len(new_doc["events"]) - len(initial["events"]),
            "source_modified": sha256(source) != source_sha256,
            "wall_seconds": time.perf_counter() - started,
            "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "native_execution": False, "disposable_copies_reanchored_for_profile": True,
        }
    result["disposable_copies_removed"] = not first.exists() and not second.exists()
    atomic(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--append-pairs", type=int, default=10)
    args = parser.parse_args()
    if args.append_pairs < 1:
        raise ValueError("at least one append pair required")
    result = compare(args.ledger.resolve(), args.output.resolve(), args.append_pairs)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
