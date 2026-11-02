#!/usr/bin/env python3
"""Seal a reviewable V18.2 source snapshot and native packet; never launch."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from replica_lab.intrinsic_memory_budget_v12 import atomic, sha_json
from replica_lab.intrinsic_memory_execution_v5 import binding, local_closure
from replica_lab.intrinsic_memory_paired_protocol_v18 import (
    PROTOCOL, PACKET, AUTHORIZATION, OUTER_WALL_SECONDS, diagnostic_plan, parent_plan, totals, validate_protocol, budget_derivation,
    training_contract)

REPORT = "reports/spark_repairs_v18/preparation_attempt04"


def save(relative, value):
    path = ROOT / relative
    if path.exists():
        raise ValueError("immutable preparation already exists: " + relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic(path, value)
    return binding(ROOT, relative)


def prepare():
    protocol = copy.deepcopy(json.loads((ROOT / "configs/spark_interference_diagnostic_v17_attempt05.json").read_text()))
    protocol.update(format=PROTOCOL, protocol_revision=2, parameter_updates=12)
    protocol["execution_preparation"] = {
        "revision": 4,
        "previous_packet": binding(ROOT, "reports/spark_repairs_v18/preparation_attempt03/native_packet.json"),
        "reason": (
            "Repair MPS-to-CPU float64 reduction corrupting tensor_statistics/module_statistics/probe "
            "after preparation_attempt03 native_bf16_comparison_attempt01 failed at gradient_measurement; "
            "scientific contract unchanged"
        )}
    protocol["inherited_protocol"] = binding(ROOT, "configs/spark_interference_diagnostic_v17_attempt05.json")
    protocol.pop("standing_authorization_text", None)
    # Historical training and evaluation metadata remain in the bound parent;
    # no inherited 512-update schedule or development pin is active in V18.2.
    protocol.pop("experiment_variant", None)
    protocol["parent_training"] = protocol["training"]
    protocol["training"] = training_contract()
    protocol["intrinsic_placement"] = "block_input_query_to_post_block_residual"
    protocol["comparison"] = {"arms": ["sequential", "paired_joint"], "steps": [8, 4],
        "primary": "mean_final_pair_minimum_total_margin", "guard": "worlds_below_own_baseline",
        "loss": "per_world_mean_answer_plus_EOS_CE", "joint_coefficients": [0.5, 0.5],
        "primary_strictly_greater": True, "guard_no_greater": True}
    output = f"{REPORT}/native_bf16_comparison_attempt01"
    protocol["diagnostic"] = {key: value for key, value in protocol["diagnostic"].items()
                              if key in ("checkpoint", "checkpoint_components", "parent_reconciliation")}
    protocol["diagnostic"].update(pair_indices=[0, 1, 4, 5], pairs=4, rows=8, updates=12,
        development=False, generated_tokens=0, intrinsic_reader_key_mode="content", output_root=output)
    plan = diagnostic_plan()
    protocol["arms"]["two"].update(sites=[3, 7], budget=plan, total_reservations=totals(plan),
                                     wall_seconds=OUTER_WALL_SECONDS, output_root=output)
    protocol["budget_derivation"] = budget_derivation()
    protocol["total_with_preflight"] = {key: value + totals(parent_plan())[key] for key, value in totals(plan).items()}
    entries = ["scripts/run_spark_paired_update_v18.py", "scripts/prepare_spark_paired_update_v18.py",
               "scripts/validate_spark_repairs_v18.py",
               "scripts/validate_spark_correction_generation_v18.py",
               "src/replica_lab/intrinsic_memory_cpu_fixture_v18.py"]
    files = local_closure(ROOT, entries)
    snapshot = f"{REPORT}/source_snapshot"
    destination = ROOT / snapshot
    destination.mkdir(parents=True, exist_ok=False)
    for relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        target.chmod(0o444)
    protocol["implementation"] = save(f"{REPORT}/implementation.json", {"entries": entries, "files": files})
    protocol["source_snapshot"] = save(f"{REPORT}/source_snapshot.json", {
        "root": snapshot, "files": files,
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_bytes": sum((destination / relative).stat().st_size for relative in files),
        "contains_base_checkpoint": False, "review_status": "independent_review_pending"})
    validate_protocol(protocol)
    config = save("configs/spark_paired_update_v18_2_preparation04.json", protocol)
    packet = {"format": PACKET, "arm": "two", "attempts": 1, "retries": 0,
        "experiment_version": 18, "native_engine_version": 18,
        "attempt_marker": "runs/spark_paired_update_v18/attempts.jsonl", "output_root": output,
        "protocol": config, "authorization_required": f"{REPORT}/authorization.json",
        "authorization_status": "not_created_native_launch_not_authorized",
        "cpu_fallback": False, "native_workers": 1,
        "commands": {"review": [".venv-spark-bf16/bin/python", "scripts/run_spark_paired_update_v18.py",
            "--packet", f"{REPORT}/native_packet.json", "--review"],
            "launch_after_separate_review_and_authorization": [".venv-spark-bf16/bin/python",
            "scripts/run_spark_paired_update_v18.py", "--packet", f"{REPORT}/native_packet.json",
            "--authorization", f"{REPORT}/authorization.json", "--launch"]}}
    pin = save(f"{REPORT}/native_packet.json", packet)
    save(f"{REPORT}/independent_review.template.json", {
        "decision": "pending", "reviewer": None, "findings": [],
        "packet_sha256": pin["sha256"], "source_snapshot_sha256": protocol["source_snapshot"]["sha256"],
        "checklist": "docs/experiments/20261031_Spark_Correction_Generation_V18_Review.md"})
    save(f"{REPORT}/authorization.template.json", {
        "format": AUTHORIZATION, "decision": "not_authorized", "packet_sha256": pin["sha256"],
        "protocol_sha256": config["sha256"], "arm": "two", "source_kind": "official_bf16",
        "output_root": output, "attempts": 1, "retries": 0, "budget": protocol["total_with_preflight"],
        "implementation_sha256": protocol["implementation"]["sha256"],
        "environment_sha256": protocol["environment"]["sha256"],
        "environment_closure_sha256": protocol["environment_closure"]["sha256"],
        "feature_binding_sha256": sha_json(protocol["source_feature_binding"]),
        "checkpoint_manifest_sha256": protocol["diagnostic"]["checkpoint"]["manifest_sha256"],
        "outer_wall_seconds": OUTER_WALL_SECONDS,
        "source_snapshot_sha256": protocol["source_snapshot"]["sha256"],
        "independent_review": {"path": f"{REPORT}/independent_review.json", "sha256": None},
        "human_approval_reference": None, "human_approval_text": None})
    print(json.dumps({"packet": pin, "source_files": len(files), "native_run": False,
                      "authorization_created": False, "budget": protocol["total_with_preflight"]}, indent=2))


if __name__ == "__main__":
    prepare()
