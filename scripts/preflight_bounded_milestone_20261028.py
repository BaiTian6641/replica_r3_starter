#!/usr/bin/env python3
"""Build committed DC-R1/DC-P1 milestone readiness without native execution.

The report binds implementation bytes to a full Git commit, carries exact work
and artifact budgets, and preserves the Spark/data-preparation evidence.  It
never creates authorization, loads a model or tokenizer, or opens evaluation
labels.  A native command is emitted only when the external DC-R1 receipt is
present and valid; the source-free alternative remains preparation-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dc_protocol_common_20261027 import (  # noqa: E402
    ProtocolError, read_json, require, require_committed_bytes,
    require_fresh_path, sha256_file, write_json_atomic,
)
import dc_r1_native_value_v2_20261027 as r1  # noqa: E402
import preflight_dc_p1_source_free_feature_alignment_v1 as source_free  # noqa: E402

R1_CONFIG = ROOT / "configs/dc_r1_native_value_protocol_v2.json"
R1_PREPARED = ROOT / "data/dc_r1_native_value_v2_20261027"
R1_RUNNER = "scripts/run_dc_r1_native_value_v2_20261027.py"
R1_AUTH = ROOT / "reports/dc_r1_native_value_v2_20261028/authorization.json"
AUDIT_PATH = ROOT / "reports/dc_p1_learning_audit_20261028/audit_attempt01.json"
SPARK_VERIFY = ROOT / "reports/sparkx_v5_1_review_20261027/verification.json"
SPARK_FINAL = ROOT / "reports/sparkx_final_product_20261026/verification_v2.json"
SPARK_GATES = ROOT / "reports/sparkx_v5_1_review_20261027/native_gates.json"
CAPTION_IDENTITY = ROOT / "reports/caption_admission_preparation_20261027/partition_identity.json"

COMMIT_FILES = (
    "configs/retired_runners.json",
    "configs/dc_r1_native_value_protocol_v2.json",
    "data/dc_r1_native_value_v2_20261027/manifest.json",
    "configs/dc_p1_source_free_feature_alignment_v1.json",
    "configs/dc_p1_native_response_protocol_v2.json",
    "data/dc_p1_native_response_v2_20261027/manifest.json",
    "scripts/dc_protocol_common_20261027.py",
    "scripts/dc_r1_native_value_v2_20261027.py",
    "scripts/preflight_dc_r1_native_value_v2_20261027.py",
    R1_RUNNER,
    "src/replica_lab/dc_r1_native_backend.py",
    "src/replica_lab/dc_r1_native_runner.py",
    "src/replica_lab/dc_p1_learning_audit.py",
    "src/replica_lab/dc_p1_native_runner.py",
    "src/replica_lab/dc_p1_evidence.py",
    "src/replica_lab/artifact_publication.py",
    "src/replica_lab/learning_evidence.py",
    "src/replica_lab/model_identity.py",
    "src/replica_lab/native_readout.py",
    "src/replica_lab/runner_registry.py",
    "scripts/audit_dc_p1_learning_hypothesis.py",
    "scripts/preflight_dc_p1_source_free_feature_alignment_v1.py",
    "scripts/score_dc_p1_generated_v2_20261027.py",
    "scripts/preflight_bounded_milestone_20261028.py",
    "reports/dc_p1_learning_audit_20261028/audit_attempt01.json",
    "reports/dc_p1_source_free_feature_alignment_v1_20261028/review.json",
    "reports/sparkx_v5_1_review_20261027/verification.json",
    "reports/sparkx_final_product_20261026/verification_v2.json",
    "reports/sparkx_v5_1_review_20261027/native_gates.json",
    "reports/caption_admission_preparation_20261027/partition_identity.json",
)


def _missing_fields(path: Path, declared: list[str]) -> list[str]:
    if not path.is_file():
        return list(declared)
    payload = read_json(path)
    return [name for name in declared if name not in payload]


def _r1_readiness() -> dict:
    config = read_json(R1_CONFIG)
    manifest = read_json(R1_PREPARED / "manifest.json")
    review = r1.preflight()
    auth_present = R1_AUTH.is_file()
    missing = _missing_fields(R1_AUTH, config["launch"]["authorization_fields"])
    native_command = None
    launch = {}
    authorization_valid = False
    authorization_error = None
    if auth_present and not missing:
        try:
            launch = r1.preflight(
                launch=True, authorization=R1_AUTH,
                native_runner=ROOT / R1_RUNNER)
            native_command = launch["next_command"]
            authorization_valid = launch["status"] == "launch_admitted"
        except ProtocolError as error:
            authorization_error = error.as_dict()
    status = ("retired_minicpm_only" if launch.get("status") == "retired_minicpm_only"
              else "launch_admitted_external_receipt" if authorization_valid
              else "blocked_invalid_authorization" if authorization_error
              else "blocked_missing_authorization")
    return {
        "protocol_id": r1.PROTOCOL_ID,
        "status": status,
        "config_sha256": sha256_file(R1_CONFIG),
        "prepared_manifest_sha256": manifest["manifest_sha256"],
        "runner_path": R1_RUNNER,
        "runner_sha256": sha256_file(ROOT / R1_RUNNER),
        "review_command": (
            "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python "
            "scripts/preflight_dc_r1_native_value_v2_20261027.py --review "
            "--receipt reports/bounded_milestone_20261028/dc_r1_review.json"),
        "launch_preflight_command": (
            "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python "
            "scripts/preflight_dc_r1_native_value_v2_20261027.py --launch "
            "--receipt reports/bounded_milestone_20261028/dc_r1_launch_review.json "
            "--authorization reports/dc_r1_native_value_v2_20261028/authorization.json"),
        "native_command": native_command,
        "native_execution": False,
        "authorization_path": str(R1_AUTH.relative_to(ROOT)),
        "authorization_receipt_present": auth_present,
        "authorization_valid": authorization_valid,
        "authorization_error": authorization_error,
        "missing_receipt_fields": missing,
        "output_root": config["launch"]["output_root"],
        "fresh_output_available": not (ROOT / config["launch"]["output_root"]).exists(),
        "workload": r1.workload_derivation()["total"],
        "physical_limits": config["budget"]["physical_limits"],
        "authorized_run_ceiling": config["budget"]["authorized_run_ceiling"],
        "review_status": review["status"],
        "native_compressed_gate": config["compressed_channel"]["native_gate_status"],
        "retirement": launch.get("retirement"),
    }


def _source_free_readiness() -> dict:
    config = read_json(source_free.CONFIG_PATH)
    review = source_free.preflight()
    audit = read_json(AUDIT_PATH)
    require(audit.get("format") == "replica.dc_p1_learning_audit.v1",
            "audit", "learning audit format differs")
    require(audit.get("alternative_protocol") == source_free.PROTOCOL_ID,
            "audit", "learning audit does not justify this alternative")
    auth_path = ROOT / config["launch"]["authorization_path"]
    missing = _missing_fields(auth_path, config["launch"]["authorization_fields"])
    return {
        "protocol_id": source_free.PROTOCOL_ID,
        "status": "preparation_only_no_runner_or_authorization",
        "config_sha256": sha256_file(source_free.CONFIG_PATH),
        "prepared_manifest_sha256": review["prepared_manifest_sha256"],
        "review_command": review["review_command"],
        "audit_command": (
            "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python "
            "scripts/audit_dc_p1_learning_hypothesis.py --receipt "
            "reports/dc_p1_learning_audit_20261028/audit_attempt01.json"),
        "audit": {
            "path": str(AUDIT_PATH.relative_to(ROOT)),
            "sha256": sha256_file(AUDIT_PATH),
            "gradient_ratio": audit["gradient"]["gradient_ratio"],
            "source_free_feature_shift_l2": audit["feature"]["source_free_feature_shift_l2"],
            "selected_expert_max_logit_delta": audit["selected_expert_max_logit_delta"],
            "generated_changed_token_count": sum(
                row["changed_token_count"] for row in audit["updates"]),
            "alternative_justified": audit["alternative_justified"],
            "capability_qualified": False,
        },
        "work_units": review["work_units"],
        "artifact_bytes_max": review["artifact_bytes_max"],
        "native_command": None,
        "native_execution": False,
        "runner_status": review["runner_status"],
        "launch_allowed": False,
        "authorization_path": config["launch"]["authorization_path"],
        "authorization_receipt_present": auth_path.is_file(),
        "missing_receipt_fields": missing,
        "development_labels": config["prepared"]["development_labels"],
        "sealed_final_labels": config["prepared"]["sealed_final_labels"],
    }


def _spark_readiness() -> dict:
    verification = read_json(SPARK_VERIFY)
    final = read_json(SPARK_FINAL)
    gates = read_json(SPARK_GATES)
    caption = read_json(CAPTION_IDENTITY)
    require(verification["passed"] == verification["checks"] == 721,
            "spark", "v5.1 verification is not 721/721")
    require(final["passed"] == final["checks_count"] == 500
            and not final["failed"], "spark", "final product is not 500/500")
    require(gates["summary"] == {"total": 10, "unresolved": 10,
                                 "executed": 0, "passed": 0},
            "spark", "native gate disposition differs")
    require(caption["counts"]["admitted"] == 9
            and caption["counts"]["exclusions"] == 21
            and caption["counts"]["exposed"] == 8
            and caption["counts"]["diagnostic_eval_scenes"] == 24,
            "caption", "caption partition counts differ")
    return {
        "status": "preparation_verified_native_gates_unresolved",
        "verification": {
            "path": str(SPARK_VERIFY.relative_to(ROOT)),
            "sha256": sha256_file(SPARK_VERIFY), "passed": 721, "checks": 721,
        },
        "final_product": {
            "path": str(SPARK_FINAL.relative_to(ROOT)),
            "sha256": sha256_file(SPARK_FINAL), "passed": 500, "checks": 500,
        },
        "caption_partition": {
            "path": str(CAPTION_IDENTITY.relative_to(ROOT)),
            "sha256": sha256_file(CAPTION_IDENTITY),
            "counts": caption["counts"],
        },
        "native_gates": {
            "path": str(SPARK_GATES.relative_to(ROOT)),
            "sha256": sha256_file(SPARK_GATES),
            "summary": gates["summary"],
        },
        "rsi_or_personality_executed": False,
    }


def build_report(protocol_commit: str, *, commit_verifier: Callable | None = None) -> dict:
    bindings = {}
    for relative in COMMIT_FILES:
        digest = sha256_file(ROOT / relative)
        require_committed_bytes(
            protocol_commit, relative, digest, verifier=commit_verifier)
        bindings[relative] = digest
    r1_report = _r1_readiness()
    source_report = _source_free_readiness()
    spark_report = _spark_readiness()
    status = ("refused_invalid_authorization"
              if r1_report["authorization_error"] else
              "review_ready_retired_minicpm_only"
              if r1_report["status"] == "retired_minicpm_only" else
              "review_ready_external_authorization_admitted"
              if r1_report["authorization_valid"] else
              "review_ready_native_blocked_missing_authorization")
    r1_prerequisites = ([
        "do not launch the retired MiniCPM-only v2 runner as new work; use the v3 replacement or explicitly labeled historical replay",
        "keep separate authorization and capacity-guard requirements for any permitted replay or replacement run",
    ] if r1_report["status"] == "retired_minicpm_only" else [
        "issue and independently review the exact external DC-R1 authorization receipt",
        "run the emitted capacity-guarded command once only after that receipt admits it",
    ])
    return {
        "format": "replica.bounded_milestone.readiness.v1",
        "protocol_commit": protocol_commit,
        "status": status,
        "commit_bindings": bindings,
        "dc_r1": r1_report,
        "dc_p1_source_free_alternative": source_report,
        "spark_and_data_preparation": spark_report,
        "native_execution": False,
        "native_jobs": 0,
        "model_loaded": False,
        "tokenizer_loaded": False,
        "evaluation_labels_read": False,
        "authorization_created": False,
        "capability_qualified": False,
        "learning_or_promotion_admitted": False,
        "remaining_prerequisites": [
            *r1_prerequisites,
            "implement and separately review the source-free alternative runner before any alternative native execution",
            "preserve all native gates, RSI and personality gates as unresolved",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-commit", required=True,
                        help="full 40-hex commit containing every bound input")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = build_report(args.protocol_commit)
        target = require_fresh_path(args.receipt, "milestone_readiness_receipt")
        write_json_atomic(target, report)
    except (ProtocolError, OSError, ValueError) as error:
        detail = error.as_dict() if hasattr(error, "as_dict") else {
            "check": "milestone_readiness", "detail": str(error)}
        print(json.dumps({"status": "refused", **detail}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["status"] == "refused_invalid_authorization" else 0


if __name__ == "__main__":
    raise SystemExit(main())
