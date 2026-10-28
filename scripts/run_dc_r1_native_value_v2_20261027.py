#!/usr/bin/env python3
"""Run one authorized DC-R1 v2 native readout stage on MPS.

The preparation package remains review-only.  This separate command verifies
an external authorization receipt, exact config and prepared bytes, then runs
real sequential per-layer generation.  It never creates authorization evidence,
loads a model on CPU, trains, or writes a checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replica_lab.dc_r1_native_runner import STAGES, run_stage  # noqa: E402
from dc_protocol_common_20261027 import ProtocolError  # noqa: E402

DEFAULT_CONFIG = "configs/dc_r1_native_value_protocol_v2.json"
DEFAULT_PREPARED = "data/dc_r1_native_value_v2_20261027"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--device", choices=("mps",), required=True,
                        help="native MPS is required; CPU fallback is disabled")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / DEFAULT_CONFIG)
    parser.add_argument("--prepared", type=Path, default=ROOT / DEFAULT_PREPARED)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        result = run_stage(
            args.stage, config_path=args.config, prepared=args.prepared,
            authorization_path=args.authorization, output=args.output,
            resume=bool(args.resume))
        payload = {"status": result.get("status", "complete"), **result}
    except (ProtocolError, OSError, ValueError, RuntimeError) as error:
        detail = error.as_dict() if hasattr(error, "as_dict") else {
            "check": "runner", "detail": str(error)}
        print(json.dumps({"status": "refused", **detail}, indent=2, sort_keys=True))
        return 2
    if args.report is not None:
        if args.report.exists():
            print(json.dumps({"status": "refused", "check": "fresh_report",
                              "detail": f"report already exists: {args.report}"},
                             indent=2, sort_keys=True))
            return 2
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("admitted") is True else 4


if __name__ == "__main__":
    raise SystemExit(main())
