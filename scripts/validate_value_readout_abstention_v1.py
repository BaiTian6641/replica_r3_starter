#!/usr/bin/env python3
"""Deterministic CPU validation for the value-readout abstention fixture.

The validator rebuilds every authoritative label from the synthesized episode
timeline, checks the one-step schedule, training/evaluation disjointness, the
sealed heldout labels, the seven distinct control arms, budgets and closure.
It optionally admits ordered fit/development/heldout receipts.  It constructs
no model, tokenizer, optimizer or native process.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from value_readout_abstention_contract_v1 import (  # noqa: E402
    CONFIG_PATH, PREPARED_DIR, ProtocolError, _write_json_atomic,
    admit_development, admit_fit_receipt, admit_heldout, read_json,
    sha256_json, validate_prepared,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--prepared", type=Path, default=PREPARED_DIR)
    parser.add_argument("--receipt", type=Path, default=None,
                        help="fresh path for a deterministic validation receipt")
    parser.add_argument("--fit-receipt", type=Path, default=None)
    parser.add_argument("--dev-receipt", type=Path, default=None)
    parser.add_argument("--heldout-receipt", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        report = validate_prepared(args.prepared, args.config)
        admissions = {}
        fit_receipt = None
        if args.fit_receipt is not None:
            fit_receipt = read_json(args.fit_receipt)
            admissions["fit"] = admit_fit_receipt(
                fit_receipt, args.prepared, args.config)
        if args.dev_receipt is not None:
            if fit_receipt is None:
                raise ProtocolError("development_receipt",
                                    "a development receipt requires --fit-receipt")
            admissions["development"] = admit_development(
                read_json(args.dev_receipt), fit_receipt=fit_receipt,
                prepared_dir=args.prepared, config_path=args.config)
        if args.heldout_receipt is not None:
            if fit_receipt is None or args.dev_receipt is None:
                raise ProtocolError(
                    "heldout_receipt",
                    "a heldout receipt requires --fit-receipt and --dev-receipt")
            admissions["heldout"] = admit_heldout(
                read_json(args.heldout_receipt),
                development_receipt=read_json(args.dev_receipt),
                fit_receipt=fit_receipt, prepared_dir=args.prepared,
                config_path=args.config)
        if admissions:
            report["admissions"] = admissions
            report["validation_sha256"] = sha256_json(
                {key: value for key, value in report.items()
                 if key != "validation_sha256"})
        if args.receipt is not None:
            target = Path(args.receipt)
            if target.exists():
                raise ProtocolError("receipt",
                                    f"receipt path is not fresh: {target}")
            _write_json_atomic(target, report)
    except ProtocolError as error:
        print(json.dumps({"status": "invalid", "check": error.check,
                          "detail": error.detail}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
