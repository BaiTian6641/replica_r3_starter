#!/usr/bin/env python3
"""Run the CPU-only delayed visual-memory contract checks.

This command never loads a model, executes an optimizer, runs a gradient or
starts a native job. The exact-source control lazily reuses the existing SQLite
ledger runtime. ``--native`` is accepted only to report the declared gate and
exits nonzero without running anything.

The CPU report separates the deterministic reference adapter's actual outcome,
corrections, unknowns, provenance and storage costs from mechanical restart,
journal and stale-source evidence. It is not a learned-memory or perception
quality claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.visual_memory_episodes import (  # noqa: E402
    ADMISSION_FORMAT,
    NativeEvaluationGated,
    assert_native_evaluation_allowed,
    run_visual_memory_cpu,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fresh_output(path: Path, parser) -> Path:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        parser.error("Use a fresh run output path")
    path.mkdir(parents=True, exist_ok=True)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared", type=Path,
        default=ROOT / "data/visual_memory_20261025/prepared01")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/visual_memory_20261025/cpu01")
    parser.add_argument(
        "--native", action="store_true",
        help="report the native gate and exit without running a native job")
    parser.add_argument(
        "--no-admission", action="store_true",
        help="verify prepared bytes without requiring admission.json "
             "(development checks only)")
    parser.add_argument(
        "--max-wall-seconds", type=int, default=None,
        help="override the declared CPU wall-clock bound")
    args = parser.parse_args(argv)
    prepared = args.prepared.resolve()
    if not prepared.is_dir():
        parser.error("prepared input directory does not exist")
    protocol = json.loads((prepared / "protocol.json").read_text(encoding="utf-8"))
    if args.native:
        try:
            assert_native_evaluation_allowed(protocol)
        except NativeEvaluationGated as error:
            print(json.dumps({
                "status": "native_evaluation_gated",
                "native_evaluated": False,
                "gate": protocol.get("native_gate"),
                "reason": str(error),
            }, indent=2, sort_keys=True))
            return 2
    admission_path = prepared / "admission.json"
    if not args.no_admission:
        if not admission_path.is_file():
            parser.error(
                "CPU evaluation requires admission.json; run "
                "scripts/prepare_visual_memory.py --admit")
        admission = json.loads(admission_path.read_text(encoding="utf-8"))
        if (admission.get("format") != ADMISSION_FORMAT
                or admission.get("status") != "admitted"):
            parser.error("preparation admission is not an admitted receipt")
    output = _fresh_output(args.output.resolve(), parser)
    report = run_visual_memory_cpu(
        prepared, output, require_admission=not args.no_admission,
        max_wall_seconds=args.max_wall_seconds)
    report_path = output / "report.json"
    summary = {
        "format": report["format"],
        "status": report["status"],
        "native_evaluated": report["native_evaluated"],
        "counts": report["counts"],
        "actual_outcome": report["actual_outcome"],
        "corrections": report["corrections"],
        "unknowns": report["unknowns"],
        "provenance": report["provenance"],
        "storage": report["storage"],
        "mechanical_gates": report["mechanical_persistence"]["gates"],
        "native_gate": report["native_gate"],
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
