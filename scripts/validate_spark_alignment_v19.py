#!/usr/bin/env python3
"""Validate the Stage-1 V19 alignment packet and generate the V20 master CPU validation receipt.

Supports:
- `--environment-only`: verify the bound environment closure for V19.
- `--review`: verify V19 packet, protocol, fresh fitting/diagnostic splits,
  parent V16 checkpoint bindings, and V18.2 predecessor receipt without
  authorizing or launching a native worker.
- `--master-v20`: run `run_master_spark_v20_pipeline` and write
  `reports/spark_master_plan_v20/cpu_master_validation.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from launch_spark_intrinsic_readout_environment_v1 import validate_environment
from replica_lab.intrinsic_memory_alignment_v19 import (
    ALIGNMENT_PROTOCOL,
    validate_alignment_pairs_v19,
)
from replica_lab.intrinsic_memory_budget_v12 import atomic
from replica_lab.intrinsic_memory_execution_v5 import bound_json, contained, hash_file
from replica_lab.spark_wake_sleep_cognition_v20 import run_master_spark_v20_pipeline

PACKET_FORMAT = "replica.intrinsic_memory.alignment_launch_packet.v19"


def validate_v19_packet(packet_path: Path) -> dict:
    raw_packet = json.loads(packet_path.read_text())
    if raw_packet.get("format") != PACKET_FORMAT:
        raise ValueError("unexpected V19 packet format")
    if raw_packet.get("launch_authorized") is not False:
        raise ValueError("prepared V19 packet must keep launch_authorized=false")
    if raw_packet.get("native_workers") != 1 or raw_packet.get("cpu_fallback") is not False:
        raise ValueError("V19 packet requires 1 native worker and cpu_fallback=false")

    protocol = bound_json(ROOT, raw_packet["protocol"])
    if protocol.get("format") != ALIGNMENT_PROTOCOL:
        raise ValueError("unexpected V19 protocol format")

    env_report = validate_environment(
        bound_json(ROOT, protocol["environment"]), root=ROOT
    )

    fit_payload = bound_json(ROOT, protocol["data"]["fit"])
    diag_payload = bound_json(ROOT, protocol["data"]["diagnostic_fit"])
    fit_rows = validate_alignment_pairs_v19(fit_payload["rows"])
    diag_rows = validate_alignment_pairs_v19(diag_payload["rows"])
    if {r["world_id"] for r in fit_rows} & {r["world_id"] for r in diag_rows}:
        raise ValueError("fit and diagnostic_fit world IDs must be disjoint")

    v18_pred = protocol["predecessor_v18_outcome"]
    v18_res_path = contained(ROOT, v18_pred["result_path"])
    if hash_file(v18_res_path) != v18_pred["result_sha256"]:
        raise ValueError("predecessor V18.2 result digest mismatch")

    ckpt = protocol["checkpoint"]
    manifest_path = contained(ROOT, str(Path(ckpt["path"]) / "manifest.json"))
    if hash_file(manifest_path) != ckpt["manifest_sha256"]:
        raise ValueError("parent V16 checkpoint manifest digest mismatch")
    ckpt_manifest = json.loads(manifest_path.read_text())
    if ckpt_manifest.get("execution_identity") != ckpt["execution_identity"]:
        raise ValueError("parent V16 checkpoint execution identity mismatch")
    comp_manifest = bound_json(ROOT, protocol["checkpoint_components"])
    if ckpt_manifest.get("components_manifest_sha256") != protocol["checkpoint_components"]["sha256"]:
        raise ValueError("parent V16 checkpoint component digest mismatch")
    if comp_manifest.get("adapter_contract") != ckpt_manifest.get("adapter_contract"):
        raise ValueError("parent V16 checkpoint component adapter_contract mismatch")

    output_path = contained(ROOT, raw_packet["output_root"])
    review_receipt = {
        "format": "replica.intrinsic_memory.alignment_packet_review.v19",
        "packet_path": str(packet_path.relative_to(ROOT)),
        "packet_sha256": hash_file(packet_path),
        "protocol_sha256": raw_packet["protocol"]["sha256"],
        "predecessor_v18_result_sha256": v18_pred["result_sha256"],
        "checkpoint_manifest_sha256": ckpt["manifest_sha256"],
        "checkpoint_components_sha256": protocol["checkpoint_components"]["sha256"],
        "fit_data_sha256": protocol["data"]["fit"]["sha256"],
        "diagnostic_fit_data_sha256": protocol["data"]["diagnostic_fit"]["sha256"],
        "environment_qualified": True,
        "environment_directory": env_report["environment_directory"],
        "output_root": raw_packet["output_root"],
        "output_root_unused": not output_path.exists(),
        "launch_authorized": False,
        "native_model_loads": 0,
    }
    review_out = ROOT / "reports/spark_alignment_v19/packet_review.json"
    atomic(review_out, review_receipt)
    return review_receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        type=Path,
        default=ROOT / "reports/spark_alignment_v19/native_packet.json",
    )
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--master-v20", action="store_true")
    args = parser.parse_args()

    out: dict = {}
    if args.review or not args.master_v20:
        out["v19_packet_review"] = validate_v19_packet(args.packet.resolve())
    if args.master_v20:
        with tempfile.TemporaryDirectory() as tmp:
            v20_report = run_master_spark_v20_pipeline(Path(tmp))
        v20_path = ROOT / "reports/spark_master_plan_v20/cpu_master_validation.json"
        v20_path.parent.mkdir(parents=True, exist_ok=True)
        atomic(v20_path, v20_report)
        out["v20_master_validation"] = {
            "path": str(v20_path.relative_to(ROOT)),
            "sha256": hash_file(v20_path),
            "all_categories_and_stages_cpu_verified": v20_report[
                "all_categories_and_stages_cpu_verified"
            ],
        }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
