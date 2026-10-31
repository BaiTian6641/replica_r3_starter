#!/usr/bin/env python3
"""Prepare corrected attempt02 after the preserved shared-counter budget failure."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_spark_reader_query_v15 import BASE, CHECKPOINT, REPORT, save_new
from replica_lab.intrinsic_memory_budget_v12 import sha_json
from replica_lab.intrinsic_memory_reader_query_protocol_v15 import (
    CONTROLS, OUTER_WALL_SECONDS, PACKET, PROTOCOL, diagnostic_plan, totals,
    validate_bindings, validate_protocol)
from replica_lab.intrinsic_memory_execution_v5 import binding, hash_file, local_closure

CONFIG = "configs/spark_reader_query_diagnostic_v15_attempt02.json"
OUTPUT = f"{REPORT}/native_bf16_reader_query_attempt02"
IMPLEMENTATION = f"{REPORT}/implementation_attempt02.json"
PACKET_PATH = f"{REPORT}/native_packet_attempt02.json"
AUTHORIZATION = f"{REPORT}/authorization_attempt02.json"


def prepare():
    for relative in (CONFIG, IMPLEMENTATION, PACKET_PATH, AUTHORIZATION, OUTPUT):
        if (ROOT / relative).exists():
            raise ValueError("reader/query attempt02 preparation already exists: " + relative)
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
    plan = diagnostic_plan(revision=2)
    protocol = {
        "format": PROTOCOL,
        "source_repo": base["source_repo"], "source_revision": base["source_revision"],
        "source": base["source"], "source_execution": base["source_execution"],
        "source_feature_binding": base["source_feature_binding"],
        "source_features": base["source_features"], "slot_policy_v12": base["slot_policy_v12"],
        "generation": base["generation"], "training": base["training"],
        "environment": base["environment"], "environment_closure": base["environment_closure"],
        "standing_authorization_text": base["standing_authorization_text"],
        "experiment_variant": base["experiment_variant"],
        "diagnostic_predecessor": binding(
            ROOT, "reports/intrinsic_memory_pre_mlp_v14/attempt01_terminal_reconciliation.json"),
        "data": {"fit": base["data"]["fit"]},
        "diagnostic": {
            "controls": list(CONTROLS), "pairs": 16, "rows": 32, "updates": 0,
            "development": False, "forwards_per_row": len(CONTROLS), "generated_tokens": 0,
            "budget_revision": 2,
            "budget_revision_reason": "attempt01 proved source-feature forwards and tokens share the forwards/input_tokens counters",
            "failed_attempt01": binding(
                ROOT, f"{REPORT}/native_bf16_reader_query_attempt01/failure.json"),
            "output_root": OUTPUT, "checkpoint": checkpoint,
            "checkpoint_components": binding(ROOT, f"{CHECKPOINT}/components/manifest.json"),
            "parent_reconciliation": binding(
                ROOT, "reports/intrinsic_memory_native_v12/attempt10_terminal_reconciliation.json"),
            "purpose": "measure_actual_query_key_value_weights_read_and_interference_before_mechanism_selection",
        },
        "development_labels_used": False, "parameter_updates": 0,
        "arms": {"two": {
            "budget": plan, "total_reservations": totals(plan),
            "wall_seconds": OUTER_WALL_SECONDS, "output_root": OUTPUT,
            "resource_limits": base["arms"]["two"]["resource_limits"],
        }},
    }
    entries = [
        "scripts/run_spark_reader_query_v15.py",
        "src/replica_lab/intrinsic_memory_reader_query_v15.py",
        "src/replica_lab/intrinsic_memory_reader_query_protocol_v15.py",
        "src/replica_lab/intrinsic_memory_budget_v13.py",
    ]
    save_new(IMPLEMENTATION, {"entries": entries, "files": local_closure(ROOT, entries)})
    protocol["implementation"] = binding(ROOT, IMPLEMENTATION)
    validate_protocol(protocol)
    save_new(CONFIG, protocol)
    packet = {
        "format": PACKET, "arm": "two", "attempts": 1, "retries": 0,
        "experiment_version": 15, "native_engine_version": 15,
        "attempt_marker": "runs/intrinsic_memory_reader_query_v15/attempts.jsonl",
        "authorization_required": AUTHORIZATION,
        "authorization_status": "external_receipt_required_not_created",
        "output_root": OUTPUT, "protocol": binding(ROOT, CONFIG),
        "commands": {
            "review": [".venv-spark-bf16/bin/python", "scripts/run_spark_reader_query_v15.py",
                       "--packet", PACKET_PATH, "--review"],
            "launch": [".venv-spark-bf16/bin/python", "scripts/run_spark_reader_query_v15.py",
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
