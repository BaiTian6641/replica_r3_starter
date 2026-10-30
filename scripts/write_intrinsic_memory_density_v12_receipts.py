#!/usr/bin/env python3
"""Regenerate coherent repair02 protocol/verification receipts from measured JSON.

Inputs are the profile and pilot JSON written by the two checked-in measurement
scripts.  Outputs are derived deterministically; the destination must be new and
no existing receipt is overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from replica_lab.intrinsic_memory_density_receipts_v12 import write_repair_receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--cpu-summary", type=Path, required=True)
    parser.add_argument("--cost-pilot", type=Path, required=True)
    parser.add_argument("--protocols-directory", type=Path, default=ROOT / "configs")
    parser.add_argument("--repair-cycle", default=None)
    args = parser.parse_args()
    manifest = write_repair_receipts(
        args.directory, cpu_summary_path=args.cpu_summary,
        cost_pilot_path=args.cost_pilot,
        protocols_directory=args.protocols_directory,
        repair_cycle=args.repair_cycle)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
