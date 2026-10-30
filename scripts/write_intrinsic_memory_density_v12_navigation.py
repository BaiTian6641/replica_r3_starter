#!/usr/bin/env python3
"""Generate consistent CURRENT/NONCANONICAL density navigation receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from replica_lab.intrinsic_memory_density_receipts_v12 import write_current_navigation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--root", type=Path,
                        default=ROOT / "reports/intrinsic_memory_density_v12")
    args = parser.parse_args()
    current = write_current_navigation(
        args.root, args.canonical,
        generator="scripts/write_intrinsic_memory_density_v12_navigation.py",
        x6_test=("tests/test_intrinsic_memory_density_v12_protocols.py::"
                 "test_measured_cost_fit_selection_and_tolerance_are_deterministic"))
    print(json.dumps(current, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
