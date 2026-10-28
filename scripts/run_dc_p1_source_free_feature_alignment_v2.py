#!/usr/bin/env python3
"""Run the authorized source-free v2 fit diagnostic behind the capacity guard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dc_protocol_common_20261027 import ProtocolError  # noqa: E402
from replica_lab.dc_p1_source_free_native import run_source_free_fit  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/dc_p1_source_free_feature_alignment_v2.json"
DEFAULT_PREPARED = ROOT / "data/dc_p1_native_response_v2_20261027"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps",), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        result = run_source_free_fit(
            config_path=args.config, prepared=args.prepared,
            authorization_path=args.authorization, output=args.output,
            resume=bool(args.resume), require_head=True)
        payload = {"status": result.get("status", "complete"), **result}
    except (ProtocolError, OSError, ValueError, RuntimeError) as error:
        detail = error.as_dict() if hasattr(error, "as_dict") else {
            "check": "runner", "detail": str(error)}
        print(json.dumps({"status": "refused", **detail}, indent=2, sort_keys=True))
        return 2
    if args.report is not None:
        if args.report.exists():
            print(json.dumps({"status": "refused", "check": "fresh_report",
                              "detail": f"report exists: {args.report}"}, indent=2))
            return 2
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("native_execution_admissible") is True else 4


if __name__ == "__main__":
    raise SystemExit(main())
