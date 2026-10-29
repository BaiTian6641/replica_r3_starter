#!/usr/bin/env python3
"""Harmless CPU child for capacity-guard subprocess tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--mode", choices=("success", "missing", "stale"), default="success")
    args, _ = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def shutdown(signum, frame):
        del frame
        (args.output / "shutdown.json").write_text(
            json.dumps({"signal": signum, "mode": args.mode}) + "\n")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    if args.mode != "missing":
        with (args.output / "profile_samples.jsonl").open("w") as stream:
            stream.write(json.dumps({"mps_driver_bytes": 128}) + "\n")
            stream.flush()
    if args.mode == "success":
        print("fake child completed", flush=True)
        return 0
    print(f"fake child waiting for guard: {args.mode}", flush=True)
    time.sleep(60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
