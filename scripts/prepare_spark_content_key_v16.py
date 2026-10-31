#!/usr/bin/env python3
"""Prepare one content-key reader arm using the sealed v12 training controller."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.architecture.intrinsic_memory_layout_v4 import PLACEMENT
from replica_lab.intrinsic_memory_budget_v12 import sha_json
from replica_lab.intrinsic_memory_execution_v5 import binding, hash_file, local_closure
from replica_lab.intrinsic_memory_execution_v12 import validate_bindings

BASE = "configs/spark_intrinsic_coverage_v12.json"
REPORT = "reports/intrinsic_memory_content_key_v16"
CONFIG = "configs/spark_content_key_v16.json"
OUTPUT = f"{REPORT}/native_bf16_two_attempt01"
PACKET = "replica.intrinsic_memory.launch_packet.v12"


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
                     f"{REPORT}/native_packet_two.json", f"{REPORT}/authorization_two.json", OUTPUT):
        if (ROOT / relative).exists():
            raise ValueError("content-key preparation already exists: " + relative)
    protocol = deepcopy(json.loads((ROOT / BASE).read_text()))
    protocol["intrinsic_placement"] = PLACEMENT
    protocol["intrinsic_reader_key_mode"] = "content"
    protocol["experiment_variant"]["current_mechanism"] = (
        "content_only_attention_keys_ordered_values_v16_source_slots_placement_objective_gates_unchanged")
    protocol["mechanism_diagnostic"] = binding(
        ROOT, "reports/intrinsic_memory_reader_query_v15/native_bf16_reader_query_attempt02/result.json")
    protocol["arms"]["two"]["output_root"] = OUTPUT
    entries = [
        "scripts/run_spark_intrinsic_coverage_v12.py",
        "src/replica_lab/intrinsic_memory_native_v12.py",
        "src/replica_lab/contextual_memory_layout_v12.py",
        "src/replica_lab/contextual_memory_v12.py",
        "src/replica_lab/architecture/spark_adapter.py",
        "src/replica_lab/architecture/intrinsic_memory_layout_v4.py",
        "src/replica_lab/architecture/spine.py",
    ]
    save_new(f"{REPORT}/implementation.json", {
        "entries": entries, "files": local_closure(ROOT, entries)})
    protocol["implementation"] = binding(ROOT, f"{REPORT}/implementation.json")
    save_new(CONFIG, protocol)
    packet = {
        "format": PACKET, "arm": "two", "attempts": 1, "retries": 0,
        "experiment_version": 12, "native_engine_version": 12,
        "mechanism_revision": "content_key_v16",
        "attempt_marker": "runs/intrinsic_memory_content_key_v16/attempts.jsonl",
        "authorization_required": f"{REPORT}/authorization_two.json",
        "authorization_status": "external_receipt_required_not_created",
        "output_root": OUTPUT, "protocol": binding(ROOT, CONFIG),
        "gated_arms_not_prepared": ["full_attention_nine", "all"],
        "commands": {
            "review": [".venv-spark-bf16/bin/python", "scripts/run_spark_intrinsic_coverage_v12.py",
                       "--packet", f"{REPORT}/native_packet_two.json", "--review"],
            "launch": [".venv-spark-bf16/bin/python", "scripts/run_spark_intrinsic_coverage_v12.py",
                       "--packet", f"{REPORT}/native_packet_two.json", "--authorization",
                       f"{REPORT}/authorization_two.json", "--launch"],
        },
    }
    save_new(f"{REPORT}/native_packet_two.json", packet)
    _, _, review = validate_bindings(ROOT, ROOT / f"{REPORT}/native_packet_two.json",
                                    check_source=False)
    result = {
        "prepared": True, "native_run": False, "authorization_created": False,
        "mechanism_revision": "content_key_v16",
        "packet_sha256": hash_file(ROOT / f"{REPORT}/native_packet_two.json"),
        "protocol_sha256": packet["protocol"]["sha256"],
        "implementation_sha256": protocol["implementation"]["sha256"],
        "environment_sha256": protocol["environment"]["sha256"],
        "environment_closure_sha256": protocol["environment_closure"]["sha256"],
        "feature_binding_sha256": sha_json(protocol["source_feature_binding"]),
        "budget": protocol["arms"]["two"]["total_reservations"],
        "review": review,
    }
    save_new(f"{REPORT}/preparation.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2, sort_keys=True))
