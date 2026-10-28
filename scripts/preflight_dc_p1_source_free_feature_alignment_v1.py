#!/usr/bin/env python3
"""Review-only preflight for the source-free feature-alignment protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dc_protocol_common_20261027 import (  # noqa: E402
    ProtocolError, read_json, read_jsonl, require, require_fresh_path,
    sha256_file, sha256_json, write_json_atomic,
)

CONFIG_PATH = ROOT / "configs/dc_p1_source_free_feature_alignment_v1.json"
PREPARED_DIR = ROOT / "data/dc_p1_native_response_v2_20261027"
PROTOCOL_ID = "dc_p1_source_free_feature_alignment_v1"
EXPECTED_FORMAT = "replica.dc_p1_source_free_feature_alignment.protocol.v1"


def derive_work(config: dict) -> dict[str, int]:
    arms = config["controls"]["arms"]
    require(len(arms) == 4, "controls", "four matched arms are required")
    updates = sum(int(arm["updates"]) for arm in arms)
    requests = sum(int(config["generation"]["fit_behavior_requests_per_arm"])
                   for _ in [0]) * 4
    max_new = int(config["generation"]["max_new_tokens_per_call"])
    prompt = int(config["generation"]["evaluation_prompt_tokens_max"])
    teaching = int(config["generation"]["teaching_tokens_per_event_max"])
    parity = int(config["generation"]["parity_probe_tokens_max"])
    require(updates == 18, "budget", "total updates must be 18")
    require(requests == 24, "budget", "fit behavior requests must be 24")
    return {
        "model_loads": 2,
        "tokenizer_loads": 2,
        "training_forwards": updates,
        "source_free_query_forwards": updates,
        "backward_calls": updates,
        "optimizer_steps": updates,
        "generation_requests": requests,
        "generation_prefill_forwards": requests,
        "generation_decode_forwards": requests * (max_new - 1),
        "autoregressive_forwards": requests * max_new,
        "generated_tokens": requests * max_new,
        "teaching_processed_positions": updates * teaching,
        "source_free_query_processed_positions": updates * prompt,
        "generation_processed_positions": requests * (
            max_new * prompt + max_new * (max_new - 1) // 2),
        "parity_probe_forwards": 3,
        "parity_processed_positions": 3 * parity,
        "restart_training_forwards": 1,
        "restart_backward_calls": 1,
        "restart_processed_text_positions": teaching,
        "restart_model_loads": 1,
        "restart_tokenizer_loads": 1,
        "restarts": 1,
        "forward_calls": 2 * updates + requests * max_new + 3 + 1,
        "processed_text_positions": (
            updates * teaching + updates * prompt
            + requests * (max_new * prompt + max_new * (max_new - 1) // 2)
            + 3 * parity + teaching),
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
    require(config["objective"]["treatment_weight"] == 0.25
            and config["objective"]["matched_trained_weight"] == 0.0,
            "objective", "source-free objective weights differ")
    require(config["objective"]["span_specific_supervision"] is False,
            "objective", "span-specific supervision is forbidden")
    require(config["objective"]["memory_repair_claimed"] is False,
            "objective", "memory repair is not a permitted claim")
    require(config["optimizer_policy"]["primary"] == "NativeFusedAdamW"
            and config["optimizer_policy"]["default_name"] == "adamw_fused"
            and config["optimizer_policy"]["changed"] is False,
            "optimizer", "optimizer identity changed")
    require(config["optimizer_policy"]["mps_fallback_enabled"] is False,
            "optimizer", "CPU fallback must remain disabled")
    require(config["launch"]["runner_status"] == "not_implemented"
            and config["launch"]["native_execution"] is False,
            "launch", "this protocol cannot claim an executable native runner")
    derived = derive_work(config)
    declared = config["budget"]["physical_limits"]
    for name, value in derived.items():
        require(declared.get(f"max_{name}") == value, "budget",
                f"max_{name}={declared.get(f'max_{name}')} differs from derived {value}")
    require(config["budget"]["artifact_bytes_max"] == derived["artifact_bytes"],
            "budget", "artifact cap differs")
    return derived


def validate_prepared(prepared: Path, config: dict) -> dict:
    manifest = read_json(prepared / "manifest.json")
    body = {key: value for key, value in manifest.items()
            if key != "manifest_sha256"}
    require(manifest["manifest_sha256"] == sha256_json(body),
            "prepared", "manifest self-integrity digest differs")
    require(manifest.get("format") == "replica.dc_p1_native_response.prepared.v2"
            and manifest.get("protocol_id") == "dc_p1_native_response_v2",
            "prepared", "prepared format or protocol differs")
    require(manifest.get("heldout_opened") is False
            and manifest.get("native_computation") is False
            and manifest.get("model_loaded") is False
            and manifest.get("tokenizer_loaded") is False
            and manifest.get("downloads") == 0,
            "prepared", "prepared access or execution declaration differs")
    for name, digest in manifest["files"].items():
        path = prepared / name
        require(path.is_file(), "consumed_artifact",
                f"missing consumed artifact {path}")
        require(sha256_file(path) == digest, "consumed_artifact",
                f"consumed artifact hash differs: {name}")
    require(manifest["manifest_sha256"] == config["prepared"]["manifest_sha256"],
            "prepared", "prepared manifest digest differs")
    require(manifest["manifest_sha256"] == config["parent"]["prepared_manifest_sha256"],
            "lineage", "parent prepared manifest digest differs")
    groups = set(config["prepared"]["source_groups"])
    teaching = [row for row in read_jsonl(prepared / "teaching.jsonl")
                if row["source_group"] in groups]
    require([int(row["event_ordinal"]) for row in teaching]
            == config["prepared"]["event_ordinals"], "prepared",
            "bounded teaching event ordinals differ")
    require([row["role"] for row in teaching]
            == ["primary", "correction", "abstention"] * 2, "prepared",
            "bounded teaching roles differ")
    cases = {row["case_id"]: row for row in read_jsonl(prepared / "fit_cases.jsonl")
             if row["source_group"] in groups}
    require(list(cases) == config["prepared"]["fit_case_ids"], "prepared",
            "bounded fit case IDs differ")
    for case_id in config["prepared"]["fit_case_ids"]:
        require(cases[case_id]["source_context_present"] is False
                and cases[case_id]["source_kv_loaded"] is False,
                "prepared", f"{case_id} is not source-free")
    return {"teaching_rows": len(teaching), "fit_cases": len(cases)}


def preflight(config_path: Path = CONFIG_PATH, prepared: Path = PREPARED_DIR,
              receipt_path: Path | None = None) -> dict:
    config = read_json(config_path)
    derived = validate_config(config)
    prepared_report = validate_prepared(prepared, config)
    report = {
        "format": "replica.dc_p1_source_free_feature_alignment.review.v1",
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
        "authorization_receipt_present": False,
        "missing_receipt_fields": config["launch"]["missing_receipt_fields"],
        "review_command": (
            "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python "
            "scripts/preflight_dc_p1_source_free_feature_alignment_v1.py "
            "--config configs/dc_p1_source_free_feature_alignment_v1.json "
            "--receipt reports/dc_p1_source_free_feature_alignment_v1_20261028/review.json"),
        "native_command": None,
        "remaining_prerequisites": [
            "issue an external authorization receipt with every declared field",
            "bind exact config, prepared manifest and protocol commit",
            "preserve the 50 GiB / 512 MiB / 15% host-free exclusive-MPS guards",
        ],
    }
    if receipt_path is not None:
        target = require_fresh_path(receipt_path, "review_receipt")
        write_json_atomic(target, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--prepared", type=Path, default=PREPARED_DIR)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = preflight(args.config, args.prepared, args.receipt)
    except ProtocolError as error:
        print(json.dumps({"status": "refused", **error.as_dict()},
                         indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
