#!/usr/bin/env python3
"""Prepare the exact v13 fitting-only diagnostic packet without launching it."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.intrinsic_memory_budget_v12 import sha_json
from replica_lab.intrinsic_memory_counterfactual_protocol_v13 import (
    CONTROLS, OUTER_WALL_SECONDS, PACKET, PROTOCOL, diagnostic_plan, totals,
    validate_bindings, validate_protocol)
from replica_lab.intrinsic_memory_execution_v5 import binding, hash_file, local_closure

BASE = "configs/spark_intrinsic_coverage_v12.json"
REPORT = "reports/intrinsic_memory_counterfactual_v13"
CONFIG = "configs/spark_counterfactual_diagnostic_v13.json"
OUTPUT = f"{REPORT}/native_bf16_pair_diagnostic_attempt01"
CHECKPOINT = "reports/intrinsic_memory_native_v12/native_bf16_two_v12_attempt10/checkpoint"


def save_new(relative, value):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return path


def prepare():
    for relative in (CONFIG, f"{REPORT}/implementation.json",
                     f"{REPORT}/native_packet.json", f"{REPORT}/authorization.json", OUTPUT):
        if (ROOT / relative).exists():
            raise ValueError("diagnostic preparation already exists: " + relative)
    base = json.loads((ROOT / BASE).read_text())
    components_path = ROOT / CHECKPOINT / "components/manifest.json"
    components = json.loads(components_path.read_text())
    checkpoint = {
        "path": CHECKPOINT,
        "manifest_sha256": hash_file(ROOT / CHECKPOINT / "manifest.json"),
        "weights_sha256": components["weights_sha256"],
        "optimizer_sha256": components["optimizer_sha256"],
    }
    checkpoint_components = f"{CHECKPOINT}/components/manifest.json"
    plan = diagnostic_plan()
    protocol = {
        "format": PROTOCOL,
        "source_repo": base["source_repo"], "source_revision": base["source_revision"],
        "source": base["source"], "source_execution": base["source_execution"],
        "source_feature_binding": base["source_feature_binding"],
        "source_features": base["source_features"],
        "slot_policy_v12": base["slot_policy_v12"],
        "generation": base["generation"], "training": base["training"],
        "environment": base["environment"], "environment_closure": base["environment_closure"],
        "standing_authorization_text": base["standing_authorization_text"],
        "experiment_variant": base["experiment_variant"],
        "diagnostic_predecessor": binding(
            ROOT, "reports/intrinsic_memory_native_v12/native_bf16_two_v12_attempt10/main_result.json"),
        "data": {"fit": base["data"]["fit"]},
        "diagnostic": {
            "controls": list(CONTROLS), "pairs": 16, "rows": 32, "updates": 0,
            "development": False, "output_root": OUTPUT,
            "checkpoint": checkpoint,
            "checkpoint_components": binding(ROOT, checkpoint_components),
            "parent_reconciliation": binding(
                ROOT, "reports/intrinsic_memory_native_v12/attempt10_terminal_reconciliation.json"),
            "purpose": "locate_distinction_loss_before_choosing_one_successor_mechanism",
        },
        "development_labels_used": False, "parameter_updates": 0,
        "arms": {"two": {
            "budget": plan, "total_reservations": totals(plan),
            "wall_seconds": OUTER_WALL_SECONDS, "output_root": OUTPUT,
            "resource_limits": base["arms"]["two"]["resource_limits"],
        }},
    }
    entries = [
        "scripts/run_spark_counterfactual_v13.py",
        "src/replica_lab/intrinsic_memory_counterfactual_v13.py",
        "src/replica_lab/intrinsic_memory_counterfactual_protocol_v13.py",
        "src/replica_lab/intrinsic_memory_budget_v13.py",
    ]
    save_new(f"{REPORT}/implementation.json", {
        "entries": entries, "files": local_closure(ROOT, entries)})
    protocol["implementation"] = binding(ROOT, f"{REPORT}/implementation.json")
    validate_protocol(protocol)
    save_new(CONFIG, protocol)
    packet = {
        "format": PACKET, "arm": "two", "attempts": 1, "retries": 0,
        "experiment_version": 13, "native_engine_version": 13,
        "attempt_marker": "runs/intrinsic_memory_counterfactual_v13/attempts.jsonl",
        "authorization_required": f"{REPORT}/authorization.json",
        "authorization_status": "external_receipt_required_not_created",
        "output_root": OUTPUT,
        "protocol": binding(ROOT, CONFIG),
        "commands": {
            "review": [".venv-spark-bf16/bin/python", "scripts/run_spark_counterfactual_v13.py",
                       "--packet", f"{REPORT}/native_packet.json", "--review"],
            "launch": [".venv-spark-bf16/bin/python", "scripts/run_spark_counterfactual_v13.py",
                       "--packet", f"{REPORT}/native_packet.json", "--authorization",
                       f"{REPORT}/authorization.json", "--launch"],
        },
    }
    save_new(f"{REPORT}/native_packet.json", packet)
    _, _, review = validate_bindings(ROOT, ROOT / f"{REPORT}/native_packet.json",
                                    check_source=False)
    result = {
        "prepared": True, "native_run": False, "authorization_created": False,
        "packet_sha256": hash_file(ROOT / f"{REPORT}/native_packet.json"),
        "protocol_sha256": packet["protocol"]["sha256"],
        "implementation_sha256": protocol["implementation"]["sha256"],
        "environment_sha256": protocol["environment"]["sha256"],
        "environment_closure_sha256": protocol["environment_closure"]["sha256"],
        "feature_binding_sha256": sha_json(protocol["source_feature_binding"]),
        "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
        "budget": protocol["arms"]["two"]["total_reservations"],
        "review": review,
    }
    save_new(f"{REPORT}/preparation.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2, sort_keys=True))
