#!/usr/bin/env python3
"""Validate DC-P1/DC-R1 configs and sealed fixtures without native work.

``--construction-audit`` checks all rows, including sealed labels, and is the
trusted preparation audit.  ``--hash-only`` verifies manifests/file hashes and
parses no JSONL rows; it is the runtime-precheck shape.  Neither mode loads a
model or tokenizer or writes a historical receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dc_p1_native_response_20261027 import (  # noqa: E402
    CONFIG_PATH as P1_CONFIG, PREPARED_DIR as P1_PREPARED,
    PROTOCOL_ID as P1_ID, validate_config as validate_p1_config,
    validate_prepared as validate_p1_prepared,
)
from dc_r1_native_value_20261027 import (  # noqa: E402
    CONFIG_PATH as R1_CONFIG, PREPARED_DIR as R1_PREPARED,
    PROTOCOL_ID as R1_ID, validate_config as validate_r1_config,
    validate_prepared as validate_r1_prepared,
)
from dc_protocol_common_20261027 import (  # noqa: E402
    read_json, require_fresh_path, sha256_file, sha256_json, validate_manifest,
    write_json_atomic,
)

P1_FILES = (
    "teaching.jsonl", "fit_cases.jsonl", "dev_cases.jsonl",
    "sealed_final_teaching.jsonl", "sealed_final_fit_cases.jsonl",
    "sealed_final_cases.jsonl", "sealed_labels_fit.jsonl",
    "sealed_labels_dev.jsonl", "sealed_final_labels_fit.jsonl",
    "sealed_labels_final.jsonl",
)
R1_FILES = (
    "dev_cases.jsonl", "sealed_final_cases.jsonl",
    "dev_conditions_exact_context.jsonl",
    "dev_conditions_selected_literal.jsonl",
    "dev_conditions_compressed_record.jsonl",
    "dev_conditions_wrong_value.jsonl",
    "dev_conditions_no_memory.jsonl",
    "sealed_final_conditions_exact_context.jsonl",
    "sealed_final_conditions_selected_literal.jsonl",
    "sealed_final_conditions_compressed_record.jsonl",
    "sealed_final_conditions_wrong_value.jsonl",
    "sealed_final_conditions_no_memory.jsonl",
    "sealed_labels_dev.jsonl", "sealed_labels_final.jsonl", "codec.json",
)


def _hash_precheck(config_path: Path, prepared: Path, protocol_id: str,
                   files: tuple[str, ...]) -> dict:
    config = read_json(config_path)
    manifest = validate_manifest(
        prepared, config, expected_files=files,
        manifest_format=config["format"].replace(".protocol.", ".prepared."),
        protocol_id=protocol_id)
    return {
        "status": "hash_valid",
        "protocol_id": protocol_id,
        "manifest_sha256": manifest["manifest_sha256"],
        "rows_parsed": 0,
        "model_loaded": False,
        "tokenizer_loaded": False,
        "native_computation": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("p1", "r1", "all"),
                        default="all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--construction-audit", action="store_true",
                      help="trusted all-split preparation audit")
    mode.add_argument("--hash-only", action="store_true",
                      help="manifest/file hash check; parses no JSONL rows")
    parser.add_argument("--receipt", type=Path, default=None,
                        help="fresh path for a deterministic validation receipt")
    args = parser.parse_args(argv)
    try:
        report: dict = {"format": "replica.dc_protocol.validation.v1"}
        if args.protocol in ("p1", "all"):
            if args.hash_only:
                validate_p1_config(read_json(P1_CONFIG))
                report["p1"] = _hash_precheck(
                    P1_CONFIG, P1_PREPARED, P1_ID, P1_FILES)
            else:
                report["p1"] = validate_p1_prepared(P1_PREPARED, P1_CONFIG)
        if args.protocol in ("r1", "all"):
            if args.hash_only:
                validate_r1_config(read_json(R1_CONFIG))
                report["r1"] = _hash_precheck(
                    R1_CONFIG, R1_PREPARED, R1_ID, R1_FILES)
            else:
                report["r1"] = validate_r1_prepared(R1_PREPARED, R1_CONFIG)
        report["mode"] = "hash_only" if args.hash_only else "construction_audit"
        report["valid"] = True
        report["validation_sha256"] = sha256_json(report)
        if args.receipt is not None:
            target = require_fresh_path(args.receipt, "validation_receipt")
            write_json_atomic(target, report)
            report["receipt"] = str(target)
            report["receipt_sha256"] = sha256_file(target)
    except Exception as error:  # noqa: BLE001 - CLI refuses with a stable code
        print(json.dumps({"status": "invalid", "error": str(error)},
                         indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
