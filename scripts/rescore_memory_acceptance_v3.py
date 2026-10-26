#!/usr/bin/env python3
"""Rescore immutable memory rankings through memory.acceptance.v3.

Candidate quality uses only the actual query_key_value readout.  exact_key_value
is a separate mixture diagnostic (never an upper bound) and oracle rows are a
separate gold-target population.  Expected inventory is pinned by the config;
missing, malformed, duplicate or extra rows fail closed.  A fresh v3 receipt and
v3 manifest are written without overwriting any v2 receipt or manifest.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.memory_acceptance_v3 import (  # noqa: E402
    MANIFEST_FORMAT, RECEIPT_FORMAT, build_manifest, load_current_receipt,
    load_expected_inventory, receipt_digest, rescore_ranking_file,
    validate_config, validate_receipt)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/memory_acceptance_v3.json")
    parser.add_argument("--inventory", type=Path,
                        help="optional explicit inventory; must match the config pin")
    parser.add_argument("--generation", type=Path,
                        help="optional historical generation JSONL, reference only")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest-output", type=Path,
                        help="write a fresh v3 manifest beside the receipt")
    parser.add_argument("--check-current", type=Path, metavar="MANIFEST",
                        help="validate the manifest's current v3 receipt and exit")
    args = parser.parse_args(argv)
    config = validate_config(json.loads(args.config.read_text()))
    if args.check_current is not None:
        receipt = load_current_receipt(args.check_current, config=config)
        print(json.dumps({
            "status": "current",
            "manifest_format": MANIFEST_FORMAT,
            "receipt_digest": receipt_digest(receipt),
            "metric_version": receipt["metric_version"],
            "primary_quality_arm": receipt["primary_quality_arm"],
            "decision": receipt["verdict"]["decision"],
        }, sort_keys=True))
        return 0
    if args.rankings is None or args.output is None:
        parser.error("--rankings and --output are required unless --check-current is used")
    expected = config["expected_inventory"]
    if args.inventory is not None:
        inventory_path = _resolve(args.inventory)
        if "path" not in expected:
            parser.error("An inline inventory config cannot be overridden")
        declared_path = _resolve(expected["path"])
        if inventory_path.resolve() != declared_path.resolve():
            parser.error("--inventory must be the exact inventory pinned by the config")
        if _sha(inventory_path) != expected["sha256"]:
            parser.error("--inventory bytes differ from the config pin")
    inventory = load_expected_inventory(config)
    output = _resolve(args.output)
    if output.exists():
        parser.error("Use a fresh output receipt")
    receipt = rescore_ranking_file(
        args.rankings, config, generation_path=args.generation,
        inventory=inventory)
    receipt["config"]["file_sha256"] = _sha(args.config)
    validate_receipt(receipt, config=config)
    if receipt["format"] != RECEIPT_FORMAT:
        raise RuntimeError("Unexpected receipt format")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    result = {
        "decision": receipt["verdict"]["decision"],
        "primary_quality_arm": receipt["primary_quality_arm"],
        "candidate_variant": receipt["candidate_variant"],
        "accepted_correct": receipt["metrics"]["variants"][
            receipt["candidate_variant"]]["quality"]["accepted_correct"],
        "accepted_correct_rate": receipt["metrics"]["variants"][
            receipt["candidate_variant"]]["quality"]["accepted_correct_rate"],
        "accepted_precision": receipt["metrics"]["variants"][
            receipt["candidate_variant"]]["quality"]["accepted_precision"],
        "acceptance_coverage": receipt["metrics"]["variants"][
            receipt["candidate_variant"]]["quality"]["acceptance_coverage"],
        "output": str(output),
        "receipt_digest": receipt_digest(receipt),
    }
    if args.manifest_output is not None:
        manifest_path = _resolve(args.manifest_output)
        if manifest_path.exists():
            parser.error("Use a fresh manifest output path")
        if manifest_path.parent.resolve() != output.parent.resolve():
            parser.error("Manifest and receipt must share a directory")
        historical = [
            {"path": str(_resolve(item["path"])), "reason": item["reason"]}
            for item in config.get("historical_artifacts", [])]
        manifest = build_manifest(
            receipt_path=output, receipt=receipt, historical=historical,
            updated=dt.date.today().isoformat())
        for entry in manifest["receipts"]:
            if entry["status"] != "current" and entry.get("not_v3") is not True:
                raise RuntimeError("Historical artifact is not explicitly non-v3")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        result["manifest"] = str(manifest_path)
        result["manifest_current"] = manifest["authoritative"]["path"]
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
