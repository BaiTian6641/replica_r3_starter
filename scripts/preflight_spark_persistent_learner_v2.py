#!/usr/bin/env python3
"""Review and authorization preflight for the separately versioned learner v2."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping

from preflight_spark_text_memory_v1 import (
    ProtocolError, _current_head, _require, _resolve, _source_validation,
    read_json, sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/spark_persistent_learner_protocol_v2.json"
PROTOCOL_ID = "spark_persistent_learner_v2"
EXPECTED_FORMAT = "replica.spark_persistent_learner.protocol.v2"
ARMS = {
    "full_eligible_context", "selective_exact_and_procedure", "spine_E_only",
    "source_free_P_only", "E_plus_P", "no_update", "reset_P",
    "shuffled_memory", "inapplicable_procedure", "matched_replay",
}
PROFILES = {"P1", "P4", "P16", "P32", "P64", "P128_stretch"}


def derive_work(config: Mapping[str, Any]) -> dict[str, int]:
    schedule = config["work_and_cost"]["native_schedule"]
    runtime = config["runtime"]
    fit = int(schedule["fit_events"])
    cases = int(schedule["development_cases_per_arm"])
    arms = len(config["controls"]["arms"])
    requests = (int(schedule["parity_requests"])
                + int(schedule["restart_parity_requests"]) + arms * cases)
    prompt = int(runtime["context_tokens_max"])
    new_tokens = int(runtime["generation"]["max_new_tokens"])
    positions_per_request = prompt * new_tokens + new_tokens * (new_tokens - 1) // 2
    restarts = int(schedule["restarts"])
    replay = int(schedule["replay_events_max"])
    query = 2 * fit + requests * new_tokens + 1
    expected = {
        "development_cases": arms * cases,
        "arms": arms,
        "model_loads_max": 1 + restarts,
        "tokenizer_loads_max": 1 + restarts,
        "training_forwards_max": 2 * fit,
        "matched_nonlearning_fit_events": fit,
        "matched_nonlearning_forward_calls_max": fit,
        "matched_nonlearning_backward_calls_max": fit,
        "matched_nonlearning_processed_positions_max": fit * prompt,
        "direct_response_ce_objective_forwards_max": 2 * fit,
        "direct_response_ce_backward_calls_max": 2 * fit,
        "direct_response_ce_feature_vectors_max": 2 * fit * prompt,
        "plastic_forward_calls_max": 0,
        "plastic_backward_calls_max": 0,
        "query_sensing_forwards_max": query,
        "native_query_passes_max": query,
        "backward_calls_max": 2 * fit,
        "backward_processed_positions_max": 2 * fit * prompt,
        "plastic_feature_vectors_max": 0,
        "plastic_backward_feature_vectors_max": 0,
        "plastic_fit_evaluation_forwards_max": 0,
        "optimizer_steps_max": 2 * fit,
        "response_spine_optimizer_steps_max": 0,
        "response_port_optimizer_steps_max": fit,
        "selected_P_optimizer_steps_max": fit,
        "generation_requests_max": requests,
        "autoregressive_forwards_max": requests * new_tokens,
        "forward_calls_max": 2 * fit + query + requests * new_tokens + replay + 4,
        "generated_tokens_max": requests * new_tokens,
        "processed_text_positions_max": (
            5 * fit * prompt + 2 * requests * positions_per_request + 5 * prompt),
        "initialization_parity_forwards_max": 2,
        "initialization_parity_processed_positions_max": 2 * prompt,
        "influence_check_query_sensing_max": 1,
        "influence_check_forwards_max": 2,
        "influence_check_processed_positions_max": 2 * prompt,
        "replay_forward_calls_max": replay,
        "replay_processed_positions_max": replay * prompt,
        "retrieval_calls_max": 3 * fit + requests + 1,
        "verification_calls_max": (3 * fit + arms * cases
            + int(schedule["parity_requests"]) + int(schedule["restart_parity_requests"])),
        "checkpoint_writes_max": fit + 1,
        "retained_checkpoint_artifacts_max": fit,
        "reloads_max": restarts,
        "cache_hits_max": 0,
        "cache_misses_max": 3 * fit + requests + 1,
    }
    for key, value in expected.items():
        _require(schedule.get(key) == value, "budget",
                 f"{key}={schedule.get(key)} differs from derived {value}")
    _require(replay == fit, "budget", "matched replay must be one replay per fit event")
    _require(restarts == 1, "budget", "exactly one restart is required")
    for key in ("teacher_passes_max", "consolidation_events_max", "native_retries"):
        _require(schedule.get(key) == 0, "budget", f"{key} must be zero for this pilot")
    _require(schedule["artifact_bytes_max"] > 0 and schedule["wall_seconds_max"] > 0,
             "budget", "artifact and wall ceilings must be positive")
    return expected


def validate_config(config: Mapping[str, Any], root: Path = ROOT) -> dict[str, Any]:
    _require(config.get("format") == EXPECTED_FORMAT, "config", "format differs")
    _require(config.get("protocol_id") == PROTOCOL_ID, "config", "protocol id differs")
    _require(config["active_direction"]["source_baseline"] == "official_fp8",
             "source", "active baseline must remain official_fp8")
    _require(config["active_direction"]["internal_spine"] == "early_permanent_required",
             "spine", "internal spine must be early and permanent")
    _require(config["active_direction"]["initial_profile"] == "P1",
             "capacity", "v2 must begin with P1")
    trainable = config["trainable_groups"]
    _require(trainable["optimizer"] == "NativeFusedAdamW", "optimizer",
             "only NativeFusedAdamW is admitted")
    _require(trainable["frozen_original_backbone"] is True, "optimizer",
             "pilot must keep the original backbone frozen")
    _require(trainable["updates_per_fit_event"] == 2
             and trainable["update_scopes"] == {"response_spine_non_port": 0,
             "response_port_output": 1, "selected_P_direct_response_ce": 1},
             "optimizer", "v2 needs separate port-output and selected-P owners")
    _require(trainable["response_spine_optimizer_steps"] == 0
             and trainable["response_port_optimizer_steps"] == 4
             and trainable["local_mse_plastic_optimizer_steps"] == 0,
             "optimizer", "non-port spine and local-MSE updates are separate arms")
    _require("extension.spine.ports.*.output_projection" in trainable["groups"]
             and trainable["response_port_learning_rate"] > 0,
             "optimizer", "zero-init port outputs need a declared trainable owner")
    memory = config["memory_contract"]
    for key in ("evidence_storage_records_max", "retained_history_versions_max",
                "active_read_slots_max", "readable_value_slots_per_record",
                "readable_value_width", "snapshot_cache_entries_max", "metadata_bytes_max",
                "replay_bytes_max", "journal_records_max", "journal_bytes_max",
                "checkpoint_transient_bytes_max", "identity_reservation_bytes"):
        _require(type(memory[key]) is int and memory[key] > 0,
                 "memory", f"{key} must be positive")
    _require(memory["active_read_slots_max"] <= 8, "memory",
             "pilot read budget cannot exceed eight slots")
    _require(memory["evidence_storage_records_max"] > memory["active_read_slots_max"],
             "memory", "storage and read budgets must be distinct")
    _require(memory["representation_codec_mismatch"] == "reject_without_silent_conversion",
             "memory", "incompatible representations must be rejected")
    _require(memory["identity_component_separate"] is True
             and memory["identity_removed_from_P_only_controls"] is True
             and memory["p_only_text_and_identity_removed"] is True,
             "identity", "P-only controls must remove text and identity")
    tiers = config["tiered_storage"]
    for key in ("hot_bytes_max", "warm_bytes_max", "cold_bytes_max",
                "identity_bytes_max", "replay_bytes_max"):
        _require(type(tiers[key]) is int and tiers[key] > 0, "tiers",
                 f"{key} must be positive")
    _require(tiers["inactive_shards_on_gpu"] is False
             and tiers["allocate_inactive_bank_on_gpu"] is False
             and tiers["active_experts_initial"] == 1,
             "tiers", "inactive banks must remain lazy and off GPU")
    profiles = {row["id"]: row for row in config["capacity_profiles"]}
    _require(set(profiles) == PROFILES, "capacity", "profile catalog differs")
    _require(profiles["P1"]["experts"] == 1 and profiles["P1"]["hidden_width"] == 512,
             "capacity", "P1 geometry differs")
    for profile_id, row in profiles.items():
        expected = row["experts"] * 3 * row["width"] * row["hidden_width"]
        _require(row["full_expert_parameters"] == expected, "capacity",
                 f"{profile_id} expert parameter count differs")
        _require(row["fp32_weight_bytes"] == expected * 4, "capacity",
                 f"{profile_id} weight bytes differ")
        _require(row["all_expert_fp32_weight_gradient_adam_bytes"] == expected * 16,
                 "capacity", f"{profile_id} training-state bytes differ")
    runtime = config["runtime"]
    _require(runtime["decoder_geometry"]["layers"] == 36, "spine",
             "active source geometry must be the sealed 36-layer SparkX2.6")
    for sites in (runtime["spine_sites"], runtime["candidate_spine_sites"]):
        _require(len(sites) == 2 and len(set(sites)) == 2
                 and all(type(site) is int and 0 <= site < 36 for site in sites),
                 "spine", "reader sites must be distinct valid zero-based indices")
    _require(runtime["generation"]["greedy"] is True
             and runtime["generation"]["sampling"] is False
             and runtime["generation"]["enable_thinking"] is False,
             "generation", "pilot must use greedy thinking-off decoding")
    _require(config["dataset"]["sealed_final_labels_closed"] is True
             and config["dataset"]["parent_control_is_sealed"] is True,
             "access", "development and sealed controls must remain separated")
    required_arms = {arm["id"] for arm in config["controls"]["arms"]}
    _require(required_arms == ARMS, "controls", "required comparison arms differ")
    _require(config["controls"]["p_only_removes"] == [
        "source_text", "exact_spans", "current_workspace_answers", "approved_identity_records"],
        "controls", "P-only information removal is incomplete")
    _require(config["success_criteria"]["reasoning_reuse"]["accuracy_gain_required"] is False,
             "success", "reasoning noninferiority must not require an accuracy gain")
    _require(config["learned_writer"]["online_parameter"] == "D_only"
             and config["learned_writer"]["native_execution_in_this_protocol"] is False,
             "writer", "learned writer must remain a separate D-only offline study")
    guards = config["work_and_cost"]["guards"]
    _require(guards["driver_limit_bytes"] == 53687091200
             and guards["sustained_swap_growth_limit_bytes"] == 536870912
             and guards["minimum_host_free_fraction"] == 0.15
             and guards["exclusive_native_worker"] is True
             and guards["cpu_fallback"] is False,
             "guards", "native resource guard changed")
    launch = config["launch"]
    _require(launch["runner_status"] in {"preparation_only", "implemented"},
             "launch", "unknown runner status")
    for relative in (launch["runner"], launch["preflight"], *launch["implementation_files"]):
        _require((root / relative).is_file(), "implementation", f"missing {relative}")
    work_units = derive_work(config)
    return {
        "source": _source_validation(config, root, full_weight_hash=False),
        "implementation_sha256": {
            relative: sha256_file(root / relative)
            for relative in launch["implementation_files"]
        },
        "work_units": work_units,
        "native_execution": False,
        "model_loaded": False,
        "tokenizer_loaded": False,
        "labels_read": False,
        "downloads": 0,
    }


def _missing_bindings(config: Mapping[str, Any], report: Mapping[str, Any],
                      authorization: Path | None) -> list[str]:
    missing = []
    launch = config["launch"]
    if launch.get("protocol_commit") is None:
        missing.append("protocol_commit")
    if launch["runner_status"] != "implemented":
        missing.append("implemented_v2_runner")
    if config["dataset"].get("manifest_sha256") is None:
        missing.append("prepared_manifest_sha256")
    if authorization is None:
        missing.append("external_authorization_receipt")
    if not launch.get("implementation_sha256"):
        missing.append("implementation_sha256")
    if not _resolve(ROOT, launch["backend_evidence_path"]).is_file():
        missing.append("native_fp8_backend_evidence")
    if report["source"]["required_backend"] and not launch["required_execution_binding"].get(
            "execution_backend"):
        missing.append("execution_backend")
    return missing


def _validate_authorization(config: Mapping[str, Any], authorization: Path,
                            output: Path, root: Path,
                            implementation_hashes: Mapping[str, str], *,
                            config_path: Path = CONFIG_PATH,
                            allow_existing_output: bool = False) -> dict[str, Any]:
    launch = config["launch"]
    _require(launch["runner_status"] == "implemented", "launch",
             "v2 runner is not implemented")
    receipt = read_json(authorization)
    for key in launch["required_authorization_fields"]:
        _require(key in receipt and receipt[key] not in (None, "", []),
                 "authorization", f"missing {key}")
    _require(receipt["protocol_id"] == PROTOCOL_ID, "authorization", "protocol id differs")
    _require(receipt["protocol_sha256"] == sha256_file(config_path),
             "authorization", "protocol hash differs")
    head = _current_head(root)
    protocol_commit = launch.get("protocol_commit")
    _require(isinstance(protocol_commit, str) and len(protocol_commit) == 40,
             "authorization", "protocol implementation commit is missing")
    ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", protocol_commit, head],
                              cwd=root, capture_output=True, check=False)
    _require(ancestry.returncode == 0, "authorization",
             "protocol implementation commit is not an ancestor of HEAD")
    drift = subprocess.run(["git", "diff", "--quiet", protocol_commit, head, "--",
                            *launch["implementation_files"]], cwd=root,
                           capture_output=True, check=False)
    _require(drift.returncode == 0, "authorization",
             "implementation files changed after the protocol commit")
    _require(receipt["protocol_commit"] == protocol_commit,
             "authorization", "receipt and protocol implementation commits differ")
    _require(receipt["implementation_sha256"] == dict(implementation_hashes),
             "authorization", "implementation hashes differ")
    _require(launch.get("implementation_sha256") == dict(implementation_hashes),
             "authorization", "config implementation binding differs")
    _require(receipt["prepared_manifest_sha256"] == config["dataset"]["manifest_sha256"],
             "authorization", "prepared manifest hash differs")
    manifest_path = _resolve(root, config["dataset"]["manifest"])
    _require(manifest_path.is_file(), "prepared", "prepared manifest is missing")
    _require(sha256_file(manifest_path) == config["dataset"]["manifest_sha256"],
             "prepared", "prepared manifest content hash differs")
    for key, expected in launch["required_execution_binding"].items():
        _require(receipt[key] == expected, "authorization",
                 f"execution binding {key} differs; no precision/backend substitution is allowed")
    backend_path = _resolve(root, launch["backend_evidence_path"])
    _require(backend_path.is_file(), "authorization", "native FP8 backend evidence is missing")
    backend = read_json(backend_path)
    for key, expected in launch["required_backend_evidence"].items():
        _require(backend.get(key) == expected, "authorization",
                 f"native FP8 backend evidence {key} differs")
    _require(receipt["backend_evidence_sha256"] == sha256_file(backend_path),
             "authorization", "backend evidence hash differs")
    _require(receipt["source_identity"] == "official_fp8", "authorization",
             "active source must remain official_fp8")
    _require(receipt["output_root"] == str(output), "authorization", "output root differs")
    _require(allow_existing_output or not output.exists(), "output", "native output must be fresh")
    _require(isinstance(receipt["issued_by"], str) and receipt["issued_by"].strip(),
             "authorization", "issued_by is required")
    try:
        expiry = datetime.fromisoformat(receipt["expires_at_utc"].replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise ProtocolError("authorization", "expires_at_utc is invalid") from error
    _require(expiry.tzinfo is not None and expiry > datetime.now(timezone.utc),
             "authorization", "receipt is expired")
    _source_validation(config, root, full_weight_hash=True)
    return receipt


def guarded_command(config: Mapping[str, Any], output: Path,
                    authorization: Path, *, config_path: Path = CONFIG_PATH) -> str:
    report = output.parent / "guard_native01.json"
    return shlex.join([
        "PYTHONDONTWRITEBYTECODE=1", "PYTORCH_ENABLE_MPS_FALLBACK=0",
        "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "HF_DATASETS_OFFLINE=1",
        ".venv/bin/python",
        "scripts/run_capacity_preflight.py", "--entry-point", "run_spark_persistent_learner_v2.py",
        "--report", str(report), "--", "--device", "mps", "--output", str(output),
        "--authorization", str(authorization), "--config", str(config_path),
        "--prepared", str(_resolve(ROOT, config["dataset"]["root"])),
    ])


def preflight(config_path: Path = CONFIG_PATH, *, authorization: Path | None = None,
              output: Path | None = None, launch: bool = False) -> dict[str, Any]:
    config = read_json(config_path)
    root = config_path.parent.parent if config_path.parent.name == "configs" else ROOT
    report = validate_config(config, root)
    output = (_resolve(ROOT, output) if output is not None
              else _resolve(ROOT, config["launch"]["output_root"]))
    missing = _missing_bindings(config, report, authorization)
    report.update({
        "format": "replica.spark_persistent_learner.review.v2",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(config_path),
        "protocol_commit": _current_head(ROOT),
        "status": "review_ready_preparation_only",
        "native_execution": False,
        "launch_allowed": False,
        "native_command": None,
        "output_root": str(output),
        "fresh_output_available": not output.exists(),
    "missing_bindings": missing,
    "remaining_prerequisites": [
        *(["commit the v2 implementation and bind protocol_commit and implementation_sha256"]
          if "protocol_commit" in missing or "implementation_sha256" in missing else []),
        *(["materialize and hash the independent fit/development fixture without opening sealed-final data"]
          if "prepared_manifest_sha256" in missing else []),
        *(["obtain exact official-FP8 forward/backward/single-update backend evidence on MPS"]
          if "native_fp8_backend_evidence" in missing else []),
        *(["issue an external authorization for the exact source, backend, implementation and output"]
          if "external_authorization_receipt" in missing else []),
        "preserve one exclusive worker and the 50 GiB / 512 MiB / 15% host-free guards",
    ],
    })
    if launch:
        _require(authorization is not None, "authorization",
                 "an external authorization receipt is required")
        receipt = _validate_authorization(config, authorization, output, root,
                                          report["implementation_sha256"], config_path=config_path)
        report.update({
            "status": "launch_admitted", "launch_allowed": True,
            "native_command": guarded_command(config, output, authorization, config_path=config_path),
            "authorization_id": receipt["authorization_id"], "attempt_id": receipt["attempt_id"],
            "execution_format": receipt["execution_format"],
            "execution_backend": receipt["execution_backend"], "weight_sha256_checked": True,
        })
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--authorization", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = preflight(args.config, authorization=args.authorization,
                           output=args.output, launch=args.launch)
        if args.receipt.exists():
            raise ProtocolError("receipt", "receipt path must be fresh")
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (ProtocolError, OSError, ValueError, subprocess.CalledProcessError) as error:
        detail = error.as_dict() if isinstance(error, ProtocolError) else {
            "check": "preflight", "detail": str(error)}
        print(json.dumps({"status": "refused", **detail}, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
