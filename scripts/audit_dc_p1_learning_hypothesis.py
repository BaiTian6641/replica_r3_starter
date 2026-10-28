#!/usr/bin/env python3
"""Run the bounded CPU DC-P1 learning-hypothesis audit and seal its evidence.

This command uses small CPU tensors and the real objective, optimizer, clipping
and feature functions. It loads no MiniCPM5 model, reads no evaluation labels,
and does not authorize or execute the alternative native protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dc_protocol_common_20261027 import (  # noqa: E402
    ProtocolError, require_fresh_path, write_json_atomic,
)
from replica_lab.dc_p1_learning_audit import run_cpu_audit  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True,
                        help="fresh JSON artifact for the CPU audit")
    args = parser.parse_args(argv)
    try:
        report = run_cpu_audit(args.receipt.parent / "fit")
        target = require_fresh_path(args.receipt, "learning_audit_receipt")
        write_json_atomic(target, report)
    except (ProtocolError, OSError, ValueError) as error:
        detail = error.as_dict() if hasattr(error, "as_dict") else {
            "check": "learning_audit", "detail": str(error)}
        print(json.dumps({"status": "refused", **detail}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
