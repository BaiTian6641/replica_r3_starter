#!/usr/bin/env python3
"""Deterministic CPU validation and fail-closed admission for the fixture.

``--construction-audit`` (default) is the trusted all-split audit used while
building/validating the fixture.  ``--runtime-precheck`` hashes artifacts and
reads no JSONL rows.  Passing ``--fit-receipt`` and ``--fit-evidence`` admits
the treatment fit from preserved artifacts; development and held-out labels are
then opened only through the post-admission label vault.  No model, tokenizer,
optimizer, native process or download is constructed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from value_readout_abstention_contract_v1 import (  # noqa: E402
    CONFIG_PATH, PREPARED_DIR, AccessSpy, LabelVault, ProtocolError,
    _write_json_atomic, admit_development, admit_fit_receipt, admit_heldout,
    read_json, runtime_precheck, sha256_json, validate_prepared,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--prepared", type=Path, default=PREPARED_DIR)
    parser.add_argument("--receipt", type=Path, default=None,
                        help="fresh path for a deterministic validation receipt")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--construction-audit", action="store_true",
                      help="trusted all-split construction audit")
    mode.add_argument("--runtime-precheck", action="store_true",
                      help="hash-only runtime precheck; parses no JSONL")
    parser.add_argument("--fit-receipt", type=Path, default=None)
    parser.add_argument("--fit-evidence", type=Path, default=None)
    parser.add_argument("--dev-receipt", type=Path, default=None)
    parser.add_argument("--dev-evidence", type=Path, default=None)
    parser.add_argument("--heldout-receipt", type=Path, default=None)
    parser.add_argument("--heldout-evidence", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        spy = AccessSpy()
        vault = LabelVault(args.prepared, config_path=args.config, spy=spy)
        if args.runtime_precheck:
            report = runtime_precheck(args.prepared, args.config)
        else:
            report = validate_prepared(args.prepared, args.config)
        admissions = {}
        fit_receipt = None
        if args.fit_receipt is not None:
            fit_receipt = read_json(args.fit_receipt)
            admissions["fit"] = admit_fit_receipt(
                fit_receipt, args.prepared, args.config,
                evidence_dir=args.fit_evidence, spy=spy)
        if args.dev_receipt is not None:
            if fit_receipt is None:
                raise ProtocolError("development_receipt",
                                    "a development receipt requires --fit-receipt")
            admissions["development"] = admit_development(
                read_json(args.dev_receipt), fit_receipt=fit_receipt,
                prepared_dir=args.prepared, config_path=args.config,
                evidence_dir=args.dev_evidence,
                fit_evidence_dir=args.fit_evidence,
                label_vault=vault, spy=spy)
        if args.heldout_receipt is not None:
            if fit_receipt is None or args.dev_receipt is None:
                raise ProtocolError(
                    "heldout_receipt",
                    "a heldout receipt requires --fit-receipt and --dev-receipt")
            admissions["heldout"] = admit_heldout(
                read_json(args.heldout_receipt),
                development_receipt=read_json(args.dev_receipt),
                fit_receipt=fit_receipt, prepared_dir=args.prepared,
                config_path=args.config,
                evidence_dir=args.heldout_evidence,
                dev_evidence_dir=args.dev_evidence,
                fit_evidence_dir=args.fit_evidence,
                label_vault=vault, spy=spy)
        if admissions:
            report["admissions"] = admissions
            report["admission_flow"] = {
                "stages": [stage for stage in
                           ("fit", "development", "heldout")
                           if stage in admissions],
                "dataset_audit": report["dataset_audit"],
                "label_access": [
                    {"split": "dev",
                     "opened": any(row["split"] == "dev" for row in vault.opened),
                     "after_admission": True},
                    {"split": "heldout",
                     "opened": any(row["split"] == "heldout"
                                   for row in vault.opened),
                     "after_admission": True},
                ],
                "model_evaluation": report["model_evaluation"],
                "access_spy": spy.events,
            }
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
