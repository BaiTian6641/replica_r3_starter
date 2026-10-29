#!/usr/bin/env python3
"""Explicit future-launch wall supervisor; this CLI grants no experiment authority."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from replica_lab.wall_deadline_v1 import WallDeadlineError, WallPolicy, run_wall_bounded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true", help="execute only with separately established authorization")
    parser.add_argument("--authorization-identity", required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--shutdown-grace-seconds", type=float, default=10.0)
    parser.add_argument("--reap-reserve-seconds", type=float, default=1.0)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        result = run_wall_bounded(command, cwd=args.cwd, output=args.output,
            policy=WallPolicy(args.wall_seconds, args.shutdown_grace_seconds, args.reap_reserve_seconds),
            launch_authorized=args.launch, authorization_identity=args.authorization_identity)
    except WallDeadlineError as error:
        parser.error(str(error))
    print(json.dumps({"status": result.receipt["status"], "receipt": str(args.output / "receipt.json"),
                      "child_exit_code": result.receipt["child_exit_code"]}))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
