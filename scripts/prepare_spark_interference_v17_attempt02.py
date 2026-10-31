#!/usr/bin/env python3
"""Prepare corrected attempt02 with top-level content-key binding."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_spark_reader_query_v15 import BASE, save_new
from replica_lab.intrinsic_memory_budget_v12 import sha_json
from replica_lab.intrinsic_memory_interference_protocol_v17 import (
    OUTER_WALL_SECONDS, PACKET, PROTOCOL, PAIR_INDICES, diagnostic_plan, totals,
    validate_bindings, validate_protocol)
from replica_lab.intrinsic_memory_execution_v5 import binding, hash_file, local_closure

REPORT = "reports/intrinsic_memory_interference_v17"
CONFIG = "configs/spark_interference_diagnostic_v17_attempt02.json"
OUTPUT = f"{REPORT}/native_bf16_interference_attempt02"
IMPLEMENTATION = f"{REPORT}/implementation_attempt02.json"
PACKET_PATH = f"{REPORT}/native_packet_attempt02.json"
AUTHORIZATION = f"{REPORT}/authorization_attempt02.json"
CHECKPOINT = "reports/intrinsic_memory_content_key_v16/native_bf16_two_attempt01/checkpoint"


def prepare():
    for relative in (CONFIG, IMPLEMENTATION, PACKET_PATH, AUTHORIZATION, OUTPUT):
        if (ROOT / relative).exists():
            raise ValueError("interference attempt02 preparation already exists: " + relative)
    base = json.loads((ROOT / BASE).read_text())
    components = json.loads((ROOT / CHECKPOINT / "components/manifest.json").read_text())
    outer = json.loads((ROOT / CHECKPOINT / "manifest.json").read_text())
    checkpoint = {
        "path": CHECKPOINT,
        "manifest_sha256": hash_file(ROOT / CHECKPOINT / "manifest.json"),
        "weights_sha256": components["weights_sha256"],
        "optimizer_sha256": components["optimizer_sha256"],
        "execution_identity": outer["execution_identity"],
    }
    plan = diagnostic_plan()
    protocol = {
        "format": PROTOCOL, "intrinsic_reader_key_mode": "content",
        "source_repo": base["source_repo"], "source_revision": base["source_revision"],
        "source": base["source"], "source_execution": base["source_execution"],
        "source_feature_binding": base["source_feature_binding"],
        "source_features": base["source_features"], "slot_policy_v12": base["slot_policy_v12"],
        "generation": base["generation"], "training": base["training"],
        "environment": base["environment"], "environment_closure": base["environment_closure"],
        "standing_authorization_text": base["standing_authorization_text"],
        "experiment_variant": base["experiment_variant"],
        "diagnostic_predecessor": binding(
            ROOT, "reports/intrinsic_memory_content_key_v16/attempt01_terminal_reconciliation.json"),
        "data": {"fit": base["data"]["fit"]},
        "diagnostic": {
            "pair_indices": list(PAIR_INDICES), "pairs": 4, "rows": 8,
            "update_arms_per_pair": 2, "updates": 8, "development": False,
            "generated_tokens": 0, "intrinsic_reader_key_mode": "content",
            "failed_attempt01": binding(
                ROOT, f"{REPORT}/native_bf16_interference_attempt01/failure.json"),
            "output_root": OUTPUT, "checkpoint": checkpoint,
            "checkpoint_components": binding(ROOT, f"{CHECKPOINT}/components/manifest.json"),
            "parent_reconciliation": binding(
                ROOT, "reports/intrinsic_memory_content_key_v16/attempt01_terminal_reconciliation.json"),
            "purpose": "measure_sequence_margins_value_read_interventions_and_paired_gradient_interference_before_candidate_selection",
        },
        "development_labels_used": False, "parameter_updates": 8,
        "arms": {"two": {
            "budget": plan, "total_reservations": totals(plan),
            "wall_seconds": OUTER_WALL_SECONDS, "output_root": OUTPUT,
            "resource_limits": base["arms"]["two"]["resource_limits"],
        }},
    }
    entries = [
        "scripts/run_spark_interference_v17.py",
        "src/replica_lab/intrinsic_memory_interference_v17.py",
        "src/replica_lab/intrinsic_memory_interference_protocol_v17.py",
        "src/replica_lab/intrinsic_memory_budget_v13.py",
    ]
    save_new(IMPLEMENTATION, {"entries": entries, "files": local_closure(ROOT, entries)})
    protocol["implementation"] = binding(ROOT, IMPLEMENTATION)
    validate_protocol(protocol)
    save_new(CONFIG, protocol)
    packet = {
        "format": PACKET, "arm": "two", "attempts": 1, "retries": 0,
        "experiment_version": 17, "native_engine_version": 17,
        "attempt_marker": "runs/intrinsic_memory_interference_v17/attempts.jsonl",
        "authorization_required": AUTHORIZATION,
        "authorization_status": "external_receipt_required_not_created",
        "output_root": OUTPUT, "protocol": binding(ROOT, CONFIG),
        "commands": {
            "review": [".venv-spark-bf16/bin/python", "scripts/run_spark_interference_v17.py",
                       "--packet", PACKET_PATH, "--review"],
            "launch": [".venv-spark-bf16/bin/python", "scripts/run_spark_interference_v17.py",
                       "--packet", PACKET_PATH, "--authorization", AUTHORIZATION, "--launch"],
        },
    }
    save_new(PACKET_PATH, packet)
    _, _, review = validate_bindings(ROOT, ROOT / PACKET_PATH, check_source=False)
    result = {
        "prepared": True, "native_run": False, "authorization_created": False,
        "packet_sha256": hash_file(ROOT / PACKET_PATH),
        "protocol_sha256": packet["protocol"]["sha256"],
        "implementation_sha256": protocol["implementation"]["sha256"],
        "environment_sha256": protocol["environment"]["sha256"],
        "environment_closure_sha256": protocol["environment_closure"]["sha256"],
        "feature_binding_sha256": sha_json(protocol["source_feature_binding"]),
        "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
        "budget": protocol["arms"]["two"]["total_reservations"], "review": review,
        "failed_attempt01_preserved": True,
    }
    save_new(f"{REPORT}/preparation_attempt02.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2, sort_keys=True))
