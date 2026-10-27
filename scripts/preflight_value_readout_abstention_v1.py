#!/usr/bin/env python3
"""Review-only preflight for the value-readout abstention protocol.

``--review`` validates the fixture and writes a fresh deterministic receipt;
it never executes native work.  ``--launch`` is deliberately fail-closed: it
requires a separate authorization receipt, and even a valid authorization is
recorded as refused because this preparation package contains no native
admission runner.  A separate later authorization and runner are mandatory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from value_readout_abstention_contract_v1 import (  # noqa: E402
    CONFIG_PATH, PREPARED_DIR, ProtocolError, preflight_report,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--prepared", type=Path, default=PREPARED_DIR)
    parser.add_argument("--receipt", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--review", action="store_true",
                      help="validate and write a review-only receipt")
    mode.add_argument("--launch", action="store_true",
                      help="fail closed without a separate admission runner")
    parser.add_argument("--authorization-receipt", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        report = preflight_report(
            config_path=args.config, prepared_dir=args.prepared,
            receipt_path=args.receipt,
            authorization_path=args.authorization_receipt,
            launch=bool(args.launch))
    except ProtocolError as error:
        print(json.dumps({"status": "refused", "check": error.check,
                          "detail": error.detail}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
