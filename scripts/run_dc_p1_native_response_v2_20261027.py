#!/usr/bin/env python3
"""Run one DC-P1 v2 native stage: fit, development or final.

The command verifies the external authorization receipt, exact config and
prepared-manifest bytes, protocol commit and runner bytes before importing
torch.  Fit trains the declared arms and scores fit evidence; development
reloads those arm checkpoints and scores development evidence; final trains the
one-shot treatment confirmation.  Native execution is refused without an
authorization receipt.  The canonical invocation runs under
``scripts/run_capacity_preflight.py --entry-point``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replica_lab.dc_p1_native_runner import (  # noqa: E402
    STAGES, Refused, RunnerError, run_stage,
)
from dc_protocol_common_20261027 import ProtocolError  # noqa: E402


DEFAULT_CONFIG = ("configs/dc_p1_native_response_protocol_v2.json")
DEFAULT_PREPARED = "data/dc_p1_native_response_v2_20261027"
DEFAULT_OUTPUT = "runs/dc_p1_native_response_v2_20261027/native01"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--device", choices=("mps",), required=True,
                        help="native MPS is required; there is no CPU fallback")
    parser.add_argument("--output", type=Path, required=True,
                        help="stage output directory (native01/<stage>)")
    parser.add_argument("--config", type=Path, default=ROOT / DEFAULT_CONFIG)
    parser.add_argument("--prepared", type=Path,
                        default=ROOT / DEFAULT_PREPARED)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--protocol-commit", required=True,
                        help="full 40-hex commit bound by the authorization")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted stage from its journal")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.device != "mps":
        parser.error("DC-P1 v2 requires --device mps; no fallback is allowed")
    try:
        result = run_stage(
            args.stage,
            config_path=args.config,
            prepared=args.prepared,
            authorization_path=args.authorization,
            protocol_commit=args.protocol_commit,
            output=args.output,
            resume=bool(args.resume),
            require_head=True,
        )
        payload = {"status": result.get("status", "complete"), **result}
    except (Refused, ProtocolError, RunnerError, OSError, ValueError) as error:
        if isinstance(error, (Refused, ProtocolError)):
            detail = error.as_dict() if hasattr(error, "as_dict") else {
                "check": "refused", "detail": str(error)}
        else:
            detail = {"check": "runner", "detail": str(error)}
        print(json.dumps({"status": "refused", **detail},
                         indent=2, sort_keys=True))
        return 2
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        if args.report.exists():
            print(json.dumps({"status": "refused", "check": "fresh_report",
                              "detail": f"report already exists: {args.report}"},
                             indent=2, sort_keys=True))
            return 2
        args.report.write_text(json.dumps(payload, indent=2, sort_keys=True)
                              + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if (payload.get("status") in ("passed", "complete")
                 and payload.get("admitted") is True) else 4


if __name__ == "__main__":
    raise SystemExit(main())
