#!/usr/bin/env python3
"""Run the deterministic explicit-record address-bank control.

This is a CPU-only control.  It verifies the pinned prepared04 episodes,
revocations and labels, freezes and writes the dev decisions before parsing any
evaluation label, applies the predeclared dev endpoint, and writes heldout
decisions only if dev passes.  It loads no model, tokenizer or checkpoint, runs
no native job and makes no generation claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.memory_address_bank import (  # noqa: E402
    CONFIG_FORMAT, REPORT_FORMAT, run_prepared_control,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/memory_address_bank_v1.json")
    parser.add_argument(
        "--prepared", type=Path,
        help="Optional prepared root override; pinned hashes still apply")
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Fresh output directory for immutable JSONL decisions and report.json")
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config.get("format") != CONFIG_FORMAT:
        parser.error(f"configuration format must be {CONFIG_FORMAT}")
    report = run_prepared_control(
        config, prepared=args.prepared, output=args.output,
        config_path=config_path)
    if report.get("format") != REPORT_FORMAT:
        raise RuntimeError("Address-bank report format differs")
    path = Path(args.output) / "report.json"
    print(json.dumps({
        "outcome": report["outcome"],
        "dev_gate": report["gating"]["dev"],
        "heldout": {
            "status": report["gating"]["heldout"]["status"],
            "decisions": report["gating"]["heldout"]["decisions"],
        },
        "report": str(path),
        "report_sha256": _sha(path),
        "decisions": report["counts"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
