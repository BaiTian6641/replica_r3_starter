#!/usr/bin/env python3
"""Review and launch-preflight for the executable source-free v2 diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from dc_protocol_common_20261027 import (  # noqa: E402
    ProtocolError, read_json, require, require_fresh_path, sha256_file,
    write_json_atomic,
)
from preflight_dc_p1_source_free_feature_alignment_v1 import (  # noqa: E402
    validate_prepared,
)
from replica_lab import dc_p1_source_free_native as runner  # noqa: E402

CONFIG_PATH = ROOT / "configs/dc_p1_source_free_feature_alignment_v2.json"
PREPARED_DIR = ROOT / "data/dc_p1_native_response_v2_20261027"
PROTOCOL_ID = "dc_p1_source_free_feature_alignment_v2"
EXPECTED_FORMAT = "replica.dc_p1_source_free_feature_alignment.protocol.v2"


def derive_work(config: dict) -> dict[str, int]:
    require(len(config["controls"]["arms"]) == 4, "controls",
            "four matched arms are required")
    updates = sum(int(arm["updates"]) for arm in config["controls"]["arms"])
    require(updates == 18, "budget", "total updates must be 18")
    generation = config["generation"]
    max_new = int(generation["max_new_tokens_per_call"])
    prompt = int(generation["evaluation_prompt_tokens_max"])
    teaching = int(generation["teaching_tokens_per_event_max"])
    parity = int(generation["parity_probe_tokens_max"])
    requests = len(config["controls"]["arms"]) * int(
        generation["fit_behavior_requests_per_arm"])
    baseline_requests = len(config["controls"]["arms"]) * int(
        generation["baseline_behavior_requests_per_arm"])
    require(requests == 24 and baseline_requests == 24, "budget",
            "baseline and post-fit generation schedules differ")
    generation_positions = (requests + baseline_requests) * (
        max_new * prompt + max_new * (max_new - 1) // 2)
    recovery_positions = max_new * prompt + max_new * (max_new - 1) // 2
    training_positions = updates * teaching * 2
    return {
        "model_loads": 2,
        "tokenizer_loads": 2,
        "training_forwards": updates * 2,
        "source_free_query_forwards": updates,
        "backward_calls": updates,
        "optimizer_steps": updates,
        "generation_requests": requests + baseline_requests,
        "generation_prefill_forwards": requests + baseline_requests,
        "generation_decode_forwards": (requests + baseline_requests) * (max_new - 1),
        "autoregressive_forwards": (requests + baseline_requests) * max_new,
        "generated_tokens": (requests + baseline_requests) * max_new,
        "teaching_processed_positions": training_positions,
        "source_conditioned_processed_positions": updates * teaching,
        "source_free_query_processed_positions": updates * prompt,
        "generation_processed_positions": generation_positions,
        "baseline_generation_processed_positions": baseline_requests * (
            max_new * prompt + max_new * (max_new - 1) // 2),
        "parity_probe_forwards": 3,
        "parity_processed_positions": 3 * parity,
        "restart_training_forwards": 0,
        "restart_backward_calls": 0,
        "restart_optimizer_steps": 0,
        "restart_processed_text_positions": recovery_positions,
        "restart_model_loads": 1,
        "restart_tokenizer_loads": 1,
        "recovery_generation_requests": 1,
        "recovery_generation_prefill_forwards": 1,
        "recovery_generation_decode_forwards": max_new - 1,
        "recovery_forward_calls": max_new,
        "recovery_generated_tokens": max_new,
        "recovery_processed_positions": recovery_positions,
        "recovery_training_forwards": 0,
        "recovery_backward_calls": 0,
        "recovery_optimizer_steps": 0,
        "recovery_processed_text_positions": recovery_positions,
        "restarts": 1,
        "forward_calls": updates * 2 + (requests + baseline_requests) * max_new
            + 3 + max_new,
        "processed_text_positions": training_positions + generation_positions
            + 3 * parity + recovery_positions,
        "per_event_checkpoint_writes": updates,
        "artifact_checkpoint_writes": 1,
        "checkpoint_writes": updates + 1,
        "retained_checkpoint_artifacts": 4,
        "temporary_storage_bytes": 33554432,
        "artifact_bytes": 33554432,
        "wall_seconds": 1200,
        "downloads": 0,
        "retries": 0,
        "workers": 1,
        "control_evaluations": 0,
    }


def validate_config(config: dict) -> dict[str, int]:
    require(config.get("format") == EXPECTED_FORMAT, "config", "format differs")
    require(config.get("protocol_id") == PROTOCOL_ID, "config", "protocol id differs")
    parent = config["parent"]
    require(parent["config_sha256"] == sha256_file(ROOT / parent["config_path"]),
            "lineage", "parent config hash differs")
    require(config["prepared"]["development_labels"] == "closed"
            and config["prepared"]["sealed_final_labels"] == "closed",
            "access", "development and sealed-final labels must stay closed")
    require(config["objective"]["both_objective_forwards_required"] is True,
            "objective", "both objective forwards are required")
    require(config["objective"]["treatment_weight"] == 0.25
            and config["objective"]["matched_trained_weight"] == 0.0,
            "objective", "source-free objective weights differ")
    require(config["objective"]["span_specific_supervision"] is False
            and config["objective"]["memory_repair_claimed"] is False,
            "objective", "forbidden claim detected")
    require(config["optimizer_policy"]["primary"] == "NativeFusedAdamW"
            and config["optimizer_policy"]["default_name"] == "adamw_fused"
            and config["optimizer_policy"]["changed"] is False
            and config["optimizer_policy"]["mps_fallback_enabled"] is False,
            "optimizer", "optimizer identity changed")
    require(config["launch"]["runner_status"] == "implemented"
            and config["launch"]["native_execution"] is False,
            "launch", "runner status or native execution declaration differs")
    derived = derive_work(config)
    declared = config["budget"]["physical_limits"]
    for name, value in derived.items():
        require(declared.get(f"max_{name}") == value, "budget",
                f"max_{name}={declared.get(f'max_{name}')} differs from derived {value}")
    require(config["budget"]["artifact_bytes_max"] == derived["artifact_bytes"],
            "budget", "artifact cap differs")
    return derived


def _next_command(config: dict, output: Path) -> str:
    authorization = ROOT / config["launch"]["authorization_path"]
    report_root = output.parent
    return shlex.join([
        "PYTHONDONTWRITEBYTECODE=1", "python3",
        "scripts/run_capacity_preflight.py", "--entry-point",
        "run_dc_p1_source_free_feature_alignment_v2.py", "--report",
        str(report_root / "guard_fit01.json"), "--", "--device", "mps",
        "--output", str(output), "--authorization", str(authorization),
        "--config", str(CONFIG_PATH), "--prepared", str(PREPARED_DIR),
        "--report", str(report_root / "fit_result01.json"),
    ])


def preflight(config_path: Path = CONFIG_PATH, prepared: Path = PREPARED_DIR,
              *, authorization: Path | None = None,
              output: Path | None = None, require_head: bool = True,
              commit_verifier=None, launch: bool = False) -> dict:
    config = read_json(config_path)
    derived = validate_config(config)
    prepared_report = validate_prepared(prepared, config)
    output = (Path(output) if output is not None
              else ROOT / config["launch"]["output_root"])
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    report = {
        "format": "replica.dc_p1_source_free_feature_alignment.review.v2",
        "protocol_id": PROTOCOL_ID,
        "status": "review_ready_preparation_only",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "prepared_manifest_path": str(prepared / "manifest.json"),
        "prepared_manifest_sha256": config["prepared"]["manifest_sha256"],
        "prepared_validation": prepared_report,
        "work_units": derived,
        "artifact_bytes_max": config["budget"]["artifact_bytes_max"],
        "native_execution": False,
        "model_loaded": False,
        "tokenizer_loaded": False,
        "labels_read": False,
        "downloads": 0,
        "launch_allowed": False,
        "runner_status": config["launch"]["runner_status"],
        "authorization_receipt_present": authorization is not None,
        "missing_receipt_fields": (
            config["launch"]["missing_receipt_fields"] if authorization is None else []),
        "review_command": (
            "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python "
            "scripts/preflight_dc_p1_source_free_feature_alignment_v2.py "
            "--receipt reports/dc_p1_source_free_feature_alignment_v2_20261028/review.json"),
        "next_eligible_command": _next_command(config, output),
        "native_command": None,
        "output_root": str(output),
        "fresh_output_available": not output.exists(),
        "remaining_prerequisites": [
            "issue an external authorization receipt with every declared field",
            "bind exact config, prepared manifest, protocol commit, attempt and output",
            "preserve the 50 GiB / 512 MiB / 15% host-free exclusive-MPS guards",
        ],
    }
    if launch:
        if authorization is None:
            raise ProtocolError("authorization", "an external authorization receipt is required")
        context, _ = runner.verify_source_free_authorization(
            config_path=config_path, prepared=prepared,
            authorization_path=authorization, output=output,
            require_head=require_head, commit_verifier=commit_verifier,
            register=False)
        report.update({
            "status": "launch_admitted",
            "launch_allowed": True,
            "native_command": _next_command(config, output),
            "authorization_receipt_sha256": context.authorization_sha256,
            "protocol_commit": context.protocol_commit,
            "attempt_id": context.attempt_id,
            "exit_code": 0,
        })
    else:
        report["native_execution_status"] = "refused_preparation_only"
        report["exit_code"] = 0
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--prepared", type=Path, default=PREPARED_DIR)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = preflight(
            args.config, args.prepared, authorization=args.authorization,
            output=args.output, launch=bool(args.launch))
        target = require_fresh_path(args.receipt, "review_receipt")
        write_json_atomic(target, report)
    except (ProtocolError, OSError, ValueError) as error:
        detail = error.as_dict() if hasattr(error, "as_dict") else {
            "check": "preflight", "detail": str(error)}
        print(json.dumps({"status": "refused", **detail}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(report.get("exit_code", 2))


if __name__ == "__main__":
    raise SystemExit(main())
