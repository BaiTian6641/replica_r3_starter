#!/usr/bin/env python3
"""Validate current or archived vision-curriculum scores without native work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.vision_score_replay import (  # noqa: E402
    VALIDATION_VERSIONS, replay_score_report)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-version", choices=VALIDATION_VERSIONS,
                        required=True)
    parser.add_argument("--score-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--reload-report", type=Path, required=True)
    parser.add_argument("--outer-guard-report", type=Path,
                        help="per-arm outer guard required by historical v1 replay")
    parser.add_argument("--supplemental-lineage", type=Path)
    parser.add_argument("--output", type=Path, required=True,
                        help="fresh validation receipt; never an input report")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a fresh output path")
    resolved = [args.score_report, args.protocol, args.run_report,
                args.reload_report, args.supplemental_lineage,
                args.outer_guard_report]
    if args.output.resolve() in {path.resolve() for path in resolved if path}:
        parser.error("Output may not overwrite an input artifact")
    receipt = replay_score_report(
        score_path=args.score_report, protocol_path=args.protocol,
        run_report_path=args.run_report, reload_report_path=args.reload_report,
        validation_version=args.validation_version,
        lineage_path=args.supplemental_lineage,
        outer_guard_report_path=args.outer_guard_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": receipt["status"],
                      "validation_version": receipt["validation_version"],
                      "learned_quality_pass": receipt["learned_quality_pass"],
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
