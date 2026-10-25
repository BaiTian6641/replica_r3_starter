#!/usr/bin/env python3
"""Replay a frozen vision diagnostic decision without rerunning native inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.vision_diagnostic import decide  # noqa: E402
from replica_lab.vision_qualification import require, sha256_file  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a fresh replay output path")
    run_report_path = args.run / "run_report.json"
    diagnostic_path = args.run / "diagnostic.json"
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    require(run_report.get("status") == "completed", "Diagnostic run did not complete")
    require(diagnostic == run_report.get("diagnostic"),
            "Standalone diagnostic differs from its native run report")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    require(sha256_file(args.protocol) == run_report.get("protocol_sha256"),
            "Replay protocol differs from the native diagnostic")
    previous = diagnostic["verdict"]
    replayed = decide(diagnostic["readout"], diagnostic["feature_statistics"],
                      **protocol["decision"])
    result = {
        "format": "replica.vision_diagnostic_decision_replay.v1",
        "parent_run_report_sha256": sha256_file(run_report_path),
        "parent_diagnostic_sha256": sha256_file(diagnostic_path),
        "protocol_sha256": sha256_file(args.protocol),
        "implementation_sha256": {
            "src/replica_lab/vision_diagnostic.py": sha256_file(
                ROOT / "src/replica_lab/vision_diagnostic.py"),
            "scripts/replay_vision_diagnostic.py": sha256_file(Path(__file__)),
        },
        "readout": diagnostic["readout"],
        "previous_verdict": previous,
        "replayed_verdict": replayed,
        "interpretation_changed": previous != replayed,
        "native_inference_repeated": False,
        "quality_claim": ("Retrospective decision-logic repair only; raw measurements "
                          "are unchanged and useful visual contribution is not assumed"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"previous": previous["decision"],
                      "replayed": replayed["decision"],
                      "interpretation_changed": result["interpretation_changed"],
                      "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
