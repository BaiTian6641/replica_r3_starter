#!/usr/bin/env python3
"""Review preflight for the FP8-derived BF16 P1/spine experiment.

This module is preparation-only. It validates the derivative manifest, every
declared shard, the parent FP8 lineage and the current P1/spine settings before
any model is loaded. It never downloads weights, never executes the parent FP8
identity, never substitutes the official BF16 release and never manufactures an
authorization. A launch is refused until a derivative-specific orchestration
runner exists and an external receipt binds this protocol, manifest, artifact,
qualification evidence, implementation and fresh output root.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
for _relative in (ROOT / "src", ROOT / "scripts"):
    if str(_relative) not in sys.path:
        sys.path.insert(0, str(_relative))

from preflight_spark_text_memory_v1 import (  # noqa: E402
    ProtocolError, _current_head, _require, _resolve, read_json, sha256_file,
)
from replica_lab.architecture.spark2_5 import (  # noqa: E402
    CURRENT_SOURCE_LAYERS, DEQUANTIZED_FP8_DERIVATIVE,
    DERIVATIVE_ACTIVATION_SEMANTICS, DERIVATIVE_CLAIMS_FALSE,
    DERIVATIVE_EXECUTION_FORMAT, LATE_SPINE_SITES, OFFICIAL_NATIVE_FP8,
    OFFICIAL_PARAMETER_ELEMENTS, P1_MEMORY_SIZE, P1_PORT_HEADS,
    P1_SPINE_SITES, P1_SPINE_SLOTS, dequantized_fp8_derivative,
    derivative_artifact_digest, derivative_precision_binding,
    evaluate_derivative_p1_spine, fingerprint, load_spark2_5_config,
    official_native_fp8_precision,
)
from replica_lab.execution_manifest import (  # noqa: E402
    ExecutionManifestError, verify_execution_manifest,
)

CONFIG_PATH = ROOT / "configs/spark_persistent_learner_derivative_bf16_placement_protocol_v2.json"
PROTOCOL_ID = "spark_persistent_learner_derivative_bf16_v1"
EXPECTED_FORMAT = "replica.spark_persistent_learner_derivative.protocol.v1"
MANIFEST_SCHEMA = "replica.spark2_5_dequantized_derivative.v1"
DERIVATIVE_SOURCE = "dequantized_fp8_derivative"
OFFICIAL_FP8 = "official_fp8"
OFFICIAL_BF16 = "official_bf16"
PROFILES = {"P1", "P4", "P16", "P32", "P64", "P128_stretch"}
ARMS = {
    "full_eligible_context", "selective_exact_and_procedure", "spine_E_only",
    "reader_trained_E_only", "source_free_P_only", "E_plus_P", "no_update", "reset_P",
    "shuffled_memory", "inapplicable_procedure", "matched_replay",
}
RUNNER_STATUSES = {"implemented", "absent_derivative_orchestration"}


def derive_work(config: Mapping[str, Any]) -> dict[str, int]:
    """Derive the bounded native schedule declared by this derivative protocol."""
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
    readout_queries = 3 * fit + requests + 1
    memory = config["memory_contract"]
    full_records = int(memory["full_context_records_max"])
    candidate_key_reads = int(memory["readout_candidate_key_reads_max_per_selection"])
    selection_attempts = int(memory["readout_selection_attempts_max_per_query"])
    selection_reads = readout_queries * selection_attempts * (
        candidate_key_reads + 5 * full_records)
    payload_reads = readout_queries * 2 * 5 * full_records
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
        "retrieval_calls_max": readout_queries,
        "verification_calls_max": (3 * fit + arms * cases
            + int(schedule["parity_requests"]) + int(schedule["restart_parity_requests"])),
        "checkpoint_writes_max": fit + 1,
        "retained_checkpoint_artifacts_max": fit,
        "reloads_max": restarts,
        "cache_hits_max": 2 * full_records * readout_queries,
        "cache_misses_max": 2 * full_records * readout_queries,
        "readout_selection_calls_max": selection_attempts * readout_queries,
        "readout_cache_bytes_max": int(memory["readout_cache_bytes_max"]),
        "readout_result_bytes_max": (int(memory["exact_text_bytes_max"])
            + int(memory["full_context_value_slots_max"]) * 1024),
        "readout_file_reads_max": selection_reads + payload_reads,
        "readout_io_bytes_max": 1073741824,
        "cold_recall_tickets_max": full_records * readout_queries,
        "cold_recall_resumes_max": full_records * readout_queries,
        "revalidation_failures_max": 0,
        "payload_read_calls_max": 2 * full_records * readout_queries,
        "payload_file_reads_max": payload_reads,
        "payload_io_bytes_max": 1073741824,
        "revalidations_max": 3 * full_records * readout_queries,
    }
    calibration = 8
    expected["competence_cases"] = calibration
    expected["competence_forwards_max"] = calibration * new_tokens
    for name in ("generation_requests_max", "verification_calls_max"):
        expected[name] += calibration
    for name in ("forward_calls_max", "autoregressive_forwards_max", "generated_tokens_max"):
        expected[name] += calibration * new_tokens
    expected["processed_text_positions_max"] += calibration * positions_per_request
    expected["readout_steps_max"] = 33
    expected["readout_cases_max"] = 40
    for name in ("training_forwards_max", "backward_calls_max", "optimizer_steps_max"):
        expected[name] += 33
    expected["backward_processed_positions_max"] += 33 * prompt
    expected["forward_calls_max"] += 35 + 40 * new_tokens
    expected["processed_text_positions_max"] += 35 * prompt + 40 * positions_per_request
    for name in ("autoregressive_forwards_max", "generated_tokens_max"):
        expected[name] += 40 * new_tokens
    for name in ("generation_requests_max", "verification_calls_max"):
        expected[name] += 40
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


def _mapping(value: Any, name: str) -> Mapping:
    _require(isinstance(value, Mapping), "config", f"{name} must be an object")
    return value


def validate_derivative_source(config: Mapping[str, Any], root: Path = ROOT, *,
                               full_hash: bool = True,
                               require_official_geometry: bool = True) -> dict[str, Any]:
    """Validate parent lineage, conversion identity, shards and config bytes."""
    sources = _mapping(config.get("source_identities"), "source_identities")
    _require(set(sources) == {OFFICIAL_FP8, OFFICIAL_BF16, DERIVATIVE_SOURCE},
             "source", "exactly the three precision identities are declared")
    parent = _mapping(sources[OFFICIAL_FP8], "official_fp8")
    source = _mapping(sources[DERIVATIVE_SOURCE], "dequantized_fp8_derivative")
    _require(source.get("substitution_allowed") is False, "source",
             "derivative substitution is not allowed")
    _require(parent.get("substitution_allowed") is False, "source",
             "official FP8 substitution is not allowed")
    _require(sources[OFFICIAL_BF16].get("substitution_allowed") is False, "source",
             "official BF16 substitution is not allowed")
    _require(source.get("execution_format") == DERIVATIVE_EXECUTION_FORMAT, "source",
             "derivative execution format differs")

    source_root = _resolve(root, source["local_path"])
    manifest_path = _resolve(root, source["manifest_path"])
    _require(manifest_path.is_file(), "source", "derivative manifest is missing")
    _require(sha256_file(manifest_path) == source["manifest_sha256"], "source",
             "derivative manifest hash differs")
    manifest = read_json(manifest_path)
    body = {key: value for key, value in manifest.items() if key != "fingerprint"}
    _require(manifest.get("fingerprint") == fingerprint(body), "source",
             "derivative manifest fingerprint mismatch")
    _require(manifest.get("schema") == MANIFEST_SCHEMA == source["manifest_schema"],
             "source", "derivative manifest schema differs")
    _require(manifest.get("derivative_index") == source["derivative_index"] == 1,
             "source", "exactly one derivative identity is authorized")
    claims = _mapping(manifest.get("claims"), "claims")
    for key in DERIVATIVE_CLAIMS_FALSE:
        _require(claims.get(key) is False, "source", f"derivative claim {key} must be false")
    _require(manifest["claims"]["official_native_fp8"] is False
             and manifest["claims"]["official_bf16_release"] is False, "source",
             "the derivative cannot claim either official identity")

    source_block = _mapping(manifest.get("source"), "manifest.source")
    derivative = _mapping(manifest.get("derivative"), "manifest.derivative")
    _require(source_block.get("repo_id") == parent["repo_id"] ==
             source["parent_repo_id"], "source", "derivative parent repository differs")
    _require(source_block.get("revision") == parent["revision"] ==
             source["parent_revision"], "source", "derivative parent revision differs")
    _require(source_block.get("config_sha256") == parent["config_sha256"] ==
             source["parent_config_sha256"], "source", "derivative parent config differs")
    _require(source_block.get("artifact_sha256") ==
             source["parent_artifact_sha256"], "source",
             "derivative parent artifact digest differs")
    _require(source_block.get("precision_kind") == OFFICIAL_NATIVE_FP8, "source",
             "derivative parent is not the official native FP8 identity")
    _require(source_block.get("quantization_fingerprint") ==
             parent["quantization_fingerprint"], "source",
             "derivative parent quantization fingerprint differs")

    parent_root = _resolve(root, parent["local_path"])
    parent_config_path = parent_root / "config.json"
    parent_weight_path = parent_root / "model.safetensors"
    _require(parent_config_path.is_file(), "source",
             f"missing sealed FP8 parent config {parent_config_path}")
    _require(sha256_file(parent_config_path) == parent["config_sha256"], "source",
             "official FP8 parent config hash differs")
    _require(parent_weight_path.is_file(), "source",
             f"missing sealed FP8 parent payload {parent_weight_path}")
    parent_files = source_block.get("source_files")
    _require(isinstance(parent_files, list) and len(parent_files) == 1, "source",
             "the parent lineage must declare exactly one sealed shard")
    parent_file = _mapping(parent_files[0], "source.source_files[0]")
    _require(parent_file.get("filename") == "model.safetensors", "source",
             "the sealed parent payload must be model.safetensors")
    _require(parent_file.get("sha256") == parent["weight_sha256"], "source",
             "parent shard pin differs from the sealed weight hash")
    _require(parent_file.get("bytes") == parent_weight_path.stat().st_size, "source",
             "sealed parent payload byte count differs")
    parent_artifact = derivative_artifact_digest(parent_files)
    _require(parent_artifact == source["parent_artifact_sha256"], "source",
             "parent artifact digest differs from the sealed shard pins")
    if full_hash:
        _require(sha256_file(parent_weight_path) == parent["weight_sha256"], "source",
                 "official FP8 parent weight hash differs")
    parent_config_raw = read_json(parent_config_path)
    parent_config = load_spark2_5_config(
        parent_config_raw, require_official_geometry=require_official_geometry)
    parent_identity = official_native_fp8_precision(
        parent_config, artifact_sha256=parent_artifact)
    _require(parent_identity.kind == OFFICIAL_NATIVE_FP8, "source",
             "parent identity is not official native FP8")
    _require(source_block.get("parent_precision") == parent_identity.to_dict(), "source",
             "manifest parent precision provenance differs from the sealed config")

    dequantizer = ROOT / "scripts" / "dequantize_spark_fp8.py"
    _require(dequantizer.is_file(), "source", "the checked-in dequantizer is missing")
    dequantizer_sha = sha256_file(dequantizer)
    _require(derivative.get("dequantization_sha256") == dequantizer_sha ==
             source["dequantization_sha256"], "source",
             "dequantizer script hash differs from the conversion provenance")
    _require(derivative.get("activation_semantics") == source["activation_semantics"] ==
             DERIVATIVE_ACTIVATION_SEMANTICS, "source",
             "derivative activation semantics differ")
    _require(derivative.get("equivalent_to_bf16_release") is False, "source",
             "the derivative cannot equal the official BF16 release")
    _require(derivative.get("storage") == "bf16"
             and derivative.get("weight_dtype") == "BF16", "source",
             "derivative storage is not BF16")
    precision_block = _mapping(derivative.get("precision"), "derivative.precision")
    _require(precision_block.get("kind") == DEQUANTIZED_FP8_DERIVATIVE, "source",
             "derivative precision kind differs")
    _require(precision_block.get("storage") == "bf16"
             and precision_block.get("weight_dtype") == "BF16", "source",
             "derivative precision storage differs")
    _require(precision_block.get("dequantized") is True
             and precision_block.get("activation_quantized") is False, "source",
             "derivative precision must be dequantized without activation quantization")
    _require(precision_block.get("derivative_of") == parent_artifact, "source",
             "derivative precision does not name the sealed parent artifact")
    _require(precision_block.get("dequantization_sha256") == dequantizer_sha, "source",
             "derivative precision dequantizer hash differs")
    _require(precision_block.get("activation") == DERIVATIVE_ACTIVATION_SEMANTICS, "source",
             "derivative precision activation semantics differ")
    _require(precision_block.get("equivalent_to_bf16_release") is False, "source",
             "derivative precision claims BF16-release equivalence")

    files = manifest.get("files")
    _require(isinstance(files, list) and len(files) == source["shard_count"] == 8,
             "source", "exactly eight derivative shards must be declared")
    declared_names = {f"model-{index:05d}-of-00008.safetensors"
                      for index in range(1, 9)}
    _require({item.get("filename") for item in files} == declared_names, "source",
             "derivative shard names differ from the eight-shard inventory")
    observed_names = {path.name for path in source_root.glob("*.safetensors")}
    _require(observed_names == declared_names, "source",
             "observed derivative shard inventory differs from its manifest")
    verified = []
    for item in files:
        item = _mapping(item, "files[]")
        path = source_root / item["filename"]
        _require(path.is_file(), "source", f"missing derivative shard {item['filename']}")
        _require(path.stat().st_size == item.get("bytes"), "source",
                 f"{item['filename']}: byte count changed")
        if full_hash:
            digest = sha256_file(path)
            _require(digest == item.get("sha256"), "source",
                     f"{item['filename']}: SHA256 changed")
        else:
            digest = item.get("sha256")
        verified.append({"filename": item["filename"], "bytes": item["bytes"],
                         "sha256": digest})
    artifact = derivative_artifact_digest(verified)
    _require(artifact == derivative.get("artifact_sha256") ==
             source["artifact_sha256"], "source",
             "derivative artifact digest differs from its manifest")
    _require(artifact != parent_artifact, "source",
             "the derivative reused the parent artifact digest")

    config_path = source_root / "config.json"
    _require(config_path.is_file(), "source", "derivative config.json is missing")
    _require(sha256_file(config_path) == derivative.get("config_sha256") ==
             source["config_sha256"], "source",
             "derivative config bytes differ from its manifest")
    config_raw = read_json(config_path)
    _require("quantization_config" not in config_raw, "source",
             "implicit_fp8_dequantization: derivative still declares FP8 storage")
    _require(config_raw.get("dtype") == "bfloat16", "source",
             "derivative config must declare dtype=bfloat16")
    declared_precision = _mapping(config_raw.get("replica_precision"),
                                  "config.replica_precision")
    _require(declared_precision.get("kind") == DEQUANTIZED_FP8_DERIVATIVE, "source",
             "derivative config does not declare the derivative precision kind")
    _require(declared_precision.get("parent_kind") == OFFICIAL_NATIVE_FP8, "source",
             "derivative config parent kind differs")
    _require(declared_precision.get("parent_artifact_sha256") == parent_artifact, "source",
             "derivative config does not name the sealed parent artifact")
    _require(declared_precision.get("dequantization_sha256") == dequantizer_sha, "source",
             "derivative config dequantizer hash differs")
    _require(declared_precision.get("activation_semantics") ==
             DERIVATIVE_ACTIVATION_SEMANTICS, "source",
             "derivative config activation semantics differ")
    _require(declared_precision.get("equivalent_to_bf16_release") is False, "source",
             "derivative config claims BF16-release equivalence")
    derivative_config = load_spark2_5_config(
        config_raw, require_official_geometry=require_official_geometry)

    inventory = _mapping(manifest.get("tensor_inventory"), "tensor_inventory")
    _require(inventory.get("parameter_elements") ==
             derivative_config.geometry.parameter_elements ==
             source["parameter_elements"], "source",
             "derivative parameter inventory differs from the config")
    _require(inventory.get("dtypes") == ["BF16"], "source",
             "derivative inventory must contain only BF16 storage")
    if derivative_config.geometry.is_official_geometry:
        _require(inventory.get("parameter_elements") == OFFICIAL_PARAMETER_ELEMENTS, "source",
                 "official derivative parameter count differs")
        _require(inventory.get("tensor_count") == source["tensor_count"] == 290, "source",
                 "official derivative tensor count differs")
    else:
        _require(type(inventory.get("tensor_count")) is int
                 and inventory["tensor_count"] > 0, "source",
                 "derivative tensor count must be positive")

    tokenizer = _mapping(config["runtime"].get("tokenizer"), "runtime.tokenizer")
    for filename, expected in (
        ("tokenizer.json", tokenizer["tokenizer_json_sha256"]),
        ("tokenizer_config.json", tokenizer["tokenizer_config_sha256"]),
        ("chat_template.jinja", tokenizer["chat_template_sha256"]),
    ):
        path = source_root / filename
        _require(path.is_file(), "tokenizer", f"missing derivative tokenizer asset {filename}")
        _require(sha256_file(path) == expected, "tokenizer",
                 f"derivative tokenizer asset {filename} hash differs")

    precision = dequantized_fp8_derivative(
        derivative_config, parent=parent_identity, artifact_sha256=artifact,
        dequantization_sha256=dequantizer_sha,
        activation_semantics=DERIVATIVE_ACTIVATION_SEMANTICS, storage="bf16")
    _require(precision.kind == DEQUANTIZED_FP8_DERIVATIVE, "source",
             "derivative precision identity was not admitted")
    _require(precision.digest == derivative.get("precision_digest") ==
             source["precision_digest"], "source",
             "derivative precision digest differs from its manifest")
    derivative_precision_binding(derivative_config, precision)
    try:
        execution = verify_execution_manifest(
            source_root, source.get("execution_manifest", {}),
            repository_root=root, artifact_sha256=artifact)
    except ExecutionManifestError as error:
        raise ProtocolError("execution_manifest", str(error)) from error
    return {
        "execution_manifest": execution,
        "source_identity": DERIVATIVE_SOURCE,
        "execution_format": DERIVATIVE_EXECUTION_FORMAT,
        "source_root": str(source_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": source["manifest_sha256"],
        "manifest_fingerprint": manifest["fingerprint"],
        "derivative_index": manifest["derivative_index"],
        "artifact_sha256": artifact,
        "config_sha256": source["config_sha256"],
        "dequantization_sha256": dequantizer_sha,
        "precision_digest": precision.digest,
        "activation_semantics": precision.activation,
        "storage": precision.storage,
        "weight_dtype": precision.weight_dtype,
        "equivalent_to_official_bf16": precision.equivalent_to_bf16_release,
        "activation_quantized": precision.activation_quantized,
        "substitution_allowed": False,
        "parent": {
            "repo_id": parent["repo_id"],
            "revision": parent["revision"],
            "artifact_sha256": parent_artifact,
            "config_sha256": parent["config_sha256"],
            "weight_sha256": parent["weight_sha256"],
            "quantization_fingerprint": parent["quantization_fingerprint"],
            "precision_kind": parent_identity.kind,
            "weight_sha256_checked": bool(full_hash),
        },
        "shards": verified,
        "shard_count": len(verified),
        "tensor_inventory": dict(inventory),
        "parameter_elements": derivative_config.geometry.parameter_elements,
        "tokenizer_checked": True,
        "full_weight_hash_checked": bool(full_hash),
        "config": derivative_config,
        "precision": precision,
    }


def _json_source(source: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in source.items()
            if key not in ("config", "precision")}


def _validate_admission(config: Mapping[str, Any], source: Mapping[str, Any]) -> dict:
    report = evaluate_derivative_p1_spine(
        source["precision"], source["config"], device="mps",
        spine_sites=tuple(config["runtime"]["spine_sites"]),
        spine_slots=int(config["runtime"]["spine_slot_read_budget"]),
        memory_size=int(config["runtime"]["memory_port_size"]),
        heads=int(config["runtime"]["memory_port_heads"]))
    _require(report["admitted"], "admission", "; ".join(report["reasons"]))
    _require(report["precision_kind"] == DEQUANTIZED_FP8_DERIVATIVE, "admission",
             "admission changed the precision identity")
    return report


def validate_config(config: Mapping[str, Any], root: Path = ROOT, *,
                    full_hash: bool = True,
                    require_official_geometry: bool = True) -> dict[str, Any]:
    _require(config.get("format") == EXPECTED_FORMAT, "config", "format differs")
    _require(config.get("protocol_id") == PROTOCOL_ID, "config", "protocol id differs")
    _require(config.get("version") == 1, "config", "protocol version differs")
    classification = _mapping(config.get("classification"), "classification")
    for key in ("preparation_only", "official_fp8_unchanged", "official_bf16_unchanged"):
        _require(classification.get(key) is True, "classification", f"{key} must be true")
    for key in ("native_execution", "model_download", "heldout_opened", "promotion",
                "lifelong_learning_claim", "reasoning_improvement_claim",
                "speedup_claim", "rsi_claim", "useful_learning_claim",
                "official_fp8_substitution", "official_bf16_substitution"):
        _require(classification.get(key) is False, "classification", f"{key} must be false")
    _require(classification.get("precision_identity") ==
             DERIVATIVE_EXECUTION_FORMAT, "classification",
             "precision identity differs")

    active = _mapping(config.get("active_direction"), "active_direction")
    _require(active.get("source_baseline") == DERIVATIVE_SOURCE, "source",
             "active baseline must be the dequantized derivative")
    _require(active.get("source_parent") == OFFICIAL_FP8, "source",
             "derivative parent must remain official FP8")
    _require(active.get("execution_format") == DERIVATIVE_EXECUTION_FORMAT, "source",
             "execution format differs")
    _require(active.get("internal_spine") == "early_permanent_required", "spine",
             "internal spine must be early and permanent")
    _require(active.get("initial_profile") == "P1", "capacity", "P1 must be the initial profile")
    _require(active.get("full_context_first") is True, "controls",
             "the untouched full-context baseline must run first")
    _require("official_fp8" in active.get("deferred", [])
             and "official_bf16" in active.get("deferred", []), "source",
             "both official identities must remain deferred")

    contract = _mapping(config.get("derivative_contract"), "derivative_contract")
    _require(contract.get("identity") == DERIVATIVE_EXECUTION_FORMAT, "contract",
             "derivative contract identity differs")
    _require(contract.get("parent_identity") == OFFICIAL_FP8, "contract",
             "derivative contract parent differs")
    _require(contract.get("official_bf16_substitute") is False
             and contract.get("official_fp8_substitute") is False, "contract",
             "no precision substitution is admitted")
    _require(contract.get("activation_semantics") == DERIVATIVE_ACTIVATION_SEMANTICS,
             "contract", "derivative activation semantics differ")
    for key in ("zero_init_parity", "finite_gradients_into_P_and_ports",
                "one_native_fused_adamw_update_with_moments_and_clock",
                "frozen_base_integrity", "extension_only_fresh_process_reload",
                "untouched_full_eligible_context_first",
                "matched_early_3_7_vs_late_23_31_only_when_justified"):
        _require(contract.get("required_checks", {}).get(key) is True, "contract",
                 f"required derivative check {key} is missing")

    runtime = _mapping(config.get("runtime"), "runtime")
    _require(runtime.get("device") == "mps", "runtime", "native backend must remain MPS")
    _require(runtime.get("native_workers") == 1 and runtime.get("cpu_fallback") is False
             and runtime.get("offline") is True, "runtime",
             "one exclusive offline worker without CPU fallback is required")
    _require(runtime.get("decoder_geometry", {}).get("layers") == CURRENT_SOURCE_LAYERS,
             "spine", "the derivative source must be the sealed 36-layer geometry")
    _require(tuple(runtime.get("spine_sites", ())) == P1_SPINE_SITES, "spine",
             "current P1 spine sites must be 3/7")
    _require(tuple(runtime.get("candidate_spine_sites", ())) == LATE_SPINE_SITES,
             "spine", "matched late sites must remain 23/31")
    _require(runtime.get("spine_slot_read_budget") == P1_SPINE_SLOTS, "spine",
             "spine read budget differs")
    _require(runtime.get("memory_port_size") == P1_MEMORY_SIZE
             and runtime.get("memory_port_heads") == P1_PORT_HEADS, "spine",
             "memory-port geometry differs from current P1")
    _require(runtime.get("attention_implementation") == "eager"
             and runtime.get("use_cache_for_training_reference") is False, "runtime",
             "eager attention without training cache is required")
    generation = _mapping(runtime.get("generation"), "runtime.generation")
    _require(generation.get("greedy") is True and generation.get("sampling") is False
             and generation.get("enable_thinking") is False, "generation",
             "greedy thinking-off decoding is required")

    trainable = _mapping(config.get("trainable_groups"), "trainable_groups")
    _require(trainable.get("optimizer") == "NativeFusedAdamW", "optimizer",
             "only NativeFusedAdamW is admitted")
    _require(trainable.get("frozen_original_backbone") is True, "optimizer",
             "the original backbone must stay frozen")
    _require(trainable.get("updates_per_fit_event") == 2
             and trainable.get("update_scopes") == {
                 "response_spine_non_port": 0, "response_port_output": 1,
                 "selected_P_direct_response_ce": 1},
             "optimizer", "P and port-output owners must remain separate")
    _require(trainable.get("response_spine_optimizer_steps") == 0
             and trainable.get("local_mse_plastic_optimizer_steps") == 0,
             "optimizer", "non-port spine and local-MSE updates are separate arms")
    _require(trainable.get("single_parameter_owner") is True
             and trainable.get("response_port_owner") ==
             "dedicated_NativeFusedAdamW_output_projection_only", "optimizer",
             "single ownership is required")
    _require(set(trainable.get("groups", ())) == {
        "extension.spine.ports.*.output_projection", "plastic.expert_0.gate",
        "plastic.expert_0.up", "plastic.expert_0.down"}, "optimizer",
        "P1 trainable ownership differs")

    profiles = {row["id"]: row for row in config["capacity_profiles"]}
    _require(set(profiles) == PROFILES, "capacity", "profile catalog differs")
    _require(profiles["P1"]["experts"] == 1 and profiles["P1"]["width"] == 256
             and profiles["P1"]["hidden_width"] == 512, "capacity", "P1 geometry differs")
    for profile_id, row in profiles.items():
        expected = row["experts"] * 3 * row["width"] * row["hidden_width"]
        _require(row["full_expert_parameters"] == expected
                 and row["fp32_weight_bytes"] == expected * 4
                 and row["all_expert_fp32_weight_gradient_adam_bytes"] == expected * 16,
                 "capacity", f"{profile_id} training-state bytes differ")
    for profile_id in ("P4", "P16", "P32", "P64", "P128_stretch"):
        status = str(profiles[profile_id]["status"])
        _require(status != "initial_direct_response_ce_pilot"
                 and ("lazy" in status or "not_preallocated" in status), "capacity",
                 "P4+ must remain lazy and not preallocated")

    memory = _mapping(config.get("memory_contract"), "memory_contract")
    _require(memory.get("full_context_selection") ==
             "all_eligible_without_semantic_score_filter", "memory",
             "full-context selection changed")
    _require(memory.get("verification_filter") ==
             "exclude_failed_check_and_quarantined_before_key_reads", "memory",
             "verification filter changed")
    _require(memory.get("full_context_records_max") == 16
             and memory.get("full_context_value_slots_max") == 64, "memory",
             "full-context capacity changed")
    _require(memory.get("readout_candidate_key_reads_max_per_selection") == 64
             and memory.get("readout_selection_attempts_max_per_query") == 2, "memory",
             "readout caps changed")
    for key in ("readout_cache_bytes_max", "exact_text_bytes_max",
                "readout_result_bytes_max"):
        _require(type(memory.get(key)) is int and memory[key] > 0, "memory",
                 f"{key} must be positive")
    _require(memory.get("representation_codec_mismatch") ==
             "reject_without_silent_conversion", "memory",
             "representation mismatch must be rejected")
    _require(memory.get("port_state_restoration") ==
             "exact_output_projection_identity_set_required", "memory",
             "exact port-state restoration is required")
    tiers = _mapping(config.get("tiered_storage"), "tiered_storage")
    _require(tiers.get("active_experts_initial") == 1
             and tiers.get("inactive_shards_on_gpu") is False
             and tiers.get("allocate_inactive_bank_on_gpu") is False, "tiers",
             "inactive banks must remain lazy and off GPU")

    controls = _mapping(config.get("controls"), "controls")
    _require({arm["id"] for arm in controls["arms"]} == ARMS, "controls",
             "required comparison arms differ")
    _require(controls.get("untouched_baseline") == "full_eligible_context", "controls",
             "the untouched full-context baseline must be explicit")
    _require(controls.get("p_only_removes") == [
        "source_text", "exact_spans", "current_workspace_answers",
        "approved_identity_records"], "controls", "P-only removal is incomplete")
    _require(set(controls.get("frozen_trained_reader_comparisons", ())) == {
        "reader_trained_E_only", "no_update", "reset_P"}, "controls",
        "frozen trained-reader comparison set differs")

    dataset = _mapping(config.get("dataset"), "dataset")
    _require(dataset.get("sealed_final_labels_closed") is True, "access",
             "sealed final labels must remain closed")
    manifest_path = _resolve(root, dataset["manifest"])
    _require(manifest_path.is_file(), "prepared", "prepared dataset manifest is missing")
    _require(sha256_file(manifest_path) == dataset["manifest_sha256"], "prepared",
             "prepared dataset manifest hash differs")

    guards = config["work_and_cost"]["guards"]
    _require(guards.get("driver_limit_bytes") == 53687091200
             and guards.get("sustained_swap_growth_limit_bytes") == 536870912
             and guards.get("minimum_host_free_fraction") == 0.15
             and guards.get("exclusive_native_worker") is True
             and guards.get("cpu_fallback") is False
             and guards.get("full_backbone_fp32_adamw_admitted") is False, "guards",
             "native resource guard changed")

    launch = _mapping(config.get("launch"), "launch")
    _require(launch.get("runner_status") in RUNNER_STATUSES, "launch",
             "unknown derivative runner status")
    _require(launch.get("preflight") ==
             "scripts/preflight_spark_intrinsic_readout_placement_v2.py", "launch",
             "preflight path differs")
    _require(launch.get("no_launch_without_exact_authorization") is True
             and launch.get("no_launch_without_derivative_runner") is True
             and launch.get("authorization_must_be_external") is True, "launch",
             "launch guards differ")
    for relative in (launch["preflight"], *launch.get("implementation_files", [])):
        _require((root / relative).is_file(), "implementation", f"missing {relative}")
    implementation_sha256 = {
        relative: sha256_file(root / relative)
        for relative in launch.get("implementation_files", [])
    }
    pinned = launch.get("implementation_sha256") or {}
    if pinned:
        _require(pinned == implementation_sha256, "implementation",
                 "implementation hashes differ from the protocol binding")

    source = validate_derivative_source(
        config, root, full_hash=full_hash,
        require_official_geometry=require_official_geometry)
    admission = _validate_admission(config, source)
    from types import SimpleNamespace
    from replica_lab.baseline_competence import load_spec
    from replica_lab.contextual_readout_training import load_readout_spec
    try:
        load_spec(SimpleNamespace(config=config, root=root))
        load_readout_spec(SimpleNamespace(config=config, root=root))
    except (KeyError, ValueError, OSError) as error:
        raise ProtocolError("learning_stages", str(error)) from error
    _require(launch["required_qualification_evidence"]["artifact_sha256"] == source["artifact_sha256"],
             "qualification", "qualification does not bind the active derivative artifact")
    return {
        "source": _json_source(source),
        "derivative_config": source["config"],
        "precision_identity": source["precision"],
        "admission": admission,
        "implementation_sha256": implementation_sha256,
        "work_units": derive_work(config),
        "native_execution": False,
        "model_loaded": False,
        "tokenizer_loaded": False,
        "labels_read": False,
        "downloads": 0,
    }


def _qualification_evidence(launch: Mapping[str, Any], root: Path) -> dict[str, Any]:
    evidence = _mapping(launch.get("required_qualification_evidence"),
                        "required_qualification_evidence")
    path = _resolve(root, evidence["path"])
    _require(path.is_file(), "qualification", "derivative qualification evidence is missing")
    _require(sha256_file(path) == evidence["sha256"], "qualification",
             "derivative qualification evidence hash differs")
    payload = read_json(path)
    run = payload.get("selected_runs", {}).get(DERIVATIVE_SOURCE)
    _require(isinstance(run, Mapping), "qualification",
             "selected derivative qualification run is missing")
    _require(run.get("run_id") == evidence["run_id"]
             and run.get("precision_kind") == DEQUANTIZED_FP8_DERIVATIVE, "qualification",
             "qualified derivative run identity differs")
    _require(run.get("status") == "completed"
             and run.get("completion_scope") == evidence["completion_scope"],
             "qualification", "the qualified derivative run is incomplete")
    for stage, expected in evidence["stage_status"].items():
        _require(run.get("stage_status", {}).get(stage) == expected, "qualification",
                 f"qualified derivative stage {stage} differs")
    _require(run.get("base_identity", {}).get("artifact_sha256") ==
             evidence["artifact_sha256"], "qualification",
             "qualified derivative artifact differs")
    return {"path": str(path), "sha256": evidence["sha256"],
            "run_id": run["run_id"], "status": run["status"],
            "completion_scope": run["completion_scope"]}


def runner_binding(config: Mapping[str, Any], root: Path = ROOT) -> dict[str, Any]:
    """Describe the declared derivative runner and its pinned hash."""
    launch = config["launch"]
    runner = _resolve(root, launch["runner"])
    pinned = (launch.get("implementation_sha256") or {}).get(launch["runner"])
    observed = sha256_file(runner) if runner.is_file() else None
    return {
        "path": str(runner),
        "relative": launch["runner"],
        "exists": runner.is_file(),
        "status": launch.get("runner_status"),
        "sha256": observed,
        "pinned_sha256": pinned,
        "hash_matches": observed is not None and pinned == observed,
        "implemented": (launch.get("runner_status") == "implemented"
                        and runner.is_file()),
    }


def _missing_bindings(config: Mapping[str, Any], report: Mapping[str, Any],
                      authorization: Path | None, root: Path = ROOT) -> list[str]:
    launch = config["launch"]
    missing: list[str] = []
    if launch.get("protocol_commit") is None:
        missing.append("protocol_commit")
    runner = _resolve(root, launch["runner"])
    if launch.get("runner_status") != "implemented" or not runner.is_file():
        missing.append("implemented_derivative_runner")
    if launch.get("implementation_sha256") != report["implementation_sha256"]:
        missing.append("implementation_sha256")
    dataset = config["dataset"]
    prepared = _resolve(root, dataset["manifest"])
    if (dataset.get("manifest_sha256") is None or not prepared.is_file()
            or sha256_file(prepared) != dataset["manifest_sha256"]):
        missing.append("prepared_manifest_sha256")
    if authorization is None:
        missing.append("external_derivative_authorization")
    try:
        _qualification_evidence(launch, root)
    except ProtocolError:
        missing.append("derivative_qualification_evidence")
    if not report["admission"].get("admitted"):
        missing.append("derivative_p1_spine_admission")
    return missing


def _validate_authorization(config: Mapping[str, Any], authorization: Path,
                            output: Path, root: Path,
                            implementation_hashes: Mapping[str, str], *,
                            config_path: Path = CONFIG_PATH,
                            allow_existing_output: bool = False) -> dict[str, Any]:
    launch = config["launch"]
    _require(launch["runner_status"] == "implemented"
             and _resolve(root, launch["runner"]).is_file(), "launch",
             "no derivative orchestration runner is implemented; refusing before model load")
    receipt = read_json(authorization)
    for key in launch["required_authorization_fields"]:
        _require(key in receipt and receipt[key] not in (None, "", []),
                 "authorization", f"missing {key}")
    _require(receipt["protocol_id"] == PROTOCOL_ID, "authorization", "protocol id differs")
    _require(receipt["protocol_sha256"] == sha256_file(config_path), "authorization",
             "protocol hash differs")
    head = _current_head(root)
    protocol_commit = launch.get("protocol_commit")
    _require(isinstance(protocol_commit, str) and len(protocol_commit) == 40,
             "authorization", "protocol implementation commit is missing")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol_commit, head],
        cwd=root, capture_output=True, check=False)
    _require(ancestry.returncode == 0, "authorization",
             "protocol implementation commit is not an ancestor of HEAD")
    drift = subprocess.run(
        ["git", "diff", "--quiet", protocol_commit, head, "--",
         *launch["implementation_files"]], cwd=root, capture_output=True, check=False)
    _require(drift.returncode == 0, "authorization",
             "implementation files changed after the protocol commit")
    _require(receipt["protocol_commit"] == protocol_commit, "authorization",
             "receipt and protocol implementation commits differ")
    _require(receipt["implementation_sha256"] == dict(implementation_hashes),
             "authorization", "implementation hashes differ")
    _require(launch.get("implementation_sha256") == dict(implementation_hashes),
             "authorization", "config implementation binding differs")
    _require(receipt["prepared_manifest_sha256"] == config["dataset"]["manifest_sha256"],
             "authorization", "prepared manifest hash differs")
    for key, expected in launch["required_execution_binding"].items():
        _require(receipt[key] == expected, "authorization",
                 f"execution binding {key} differs; no precision substitution is allowed")
    source = config["source_identities"][DERIVATIVE_SOURCE]
    _require(receipt["source_identity"] == DERIVATIVE_SOURCE, "authorization",
             "active source must remain the dequantized derivative")
    _require(receipt["derivative_manifest_sha256"] == source["manifest_sha256"],
             "authorization", "derivative manifest hash differs")
    _require(receipt["derivative_artifact_sha256"] == source["artifact_sha256"],
             "authorization", "derivative artifact hash differs")
    _require(receipt.get("execution_manifest_sha256") ==
             source["execution_manifest"]["sha256"], "authorization",
             "execution manifest binding differs")
    evidence = _qualification_evidence(launch, root)
    _require(receipt["qualification_evidence_sha256"] == evidence["sha256"],
             "authorization", "qualification evidence hash differs")
    _require(receipt["output_root"] == str(output), "authorization", "output root differs")
    _require(allow_existing_output or not output.exists(), "output",
             "native output must be fresh")
    _require(isinstance(receipt["issued_by"], str) and receipt["issued_by"].strip(),
             "authorization", "issued_by is required")
    try:
        expiry = datetime.fromisoformat(receipt["expires_at_utc"].replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise ProtocolError("authorization", "expires_at_utc is invalid") from error
    _require(expiry.tzinfo is not None and expiry > datetime.now(timezone.utc),
             "authorization", "receipt is expired")
    return receipt


def guarded_command(config: Mapping[str, Any], output: Path,
                    authorization: Path, *, config_path: Path = CONFIG_PATH) -> str:
    report = output.parent / "guard_derivative_native03.json"
    guards = config["work_and_cost"]["guards"]
    return shlex.join([
        "PYTHONDONTWRITEBYTECODE=1", "PYTORCH_ENABLE_MPS_FALLBACK=0",
        "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "HF_DATASETS_OFFLINE=1",
        str(ROOT / ".venv-spark-bf16/bin/python"), "scripts/run_capacity_preflight_spark_placement_v2.py",
        "--entry-point", Path(config["launch"]["runner"]).name,
        "--driver-limit-gib", str(guards["driver_limit_bytes"] / 1024 ** 3),
        "--swap-growth-limit-mib",
        str(guards["sustained_swap_growth_limit_bytes"] / 1024 ** 2),
        "--minimum-free-percent", str(guards["minimum_host_free_fraction"] * 100),
        "--sustained-host-samples", str(guards["sustained_host_samples"]),
        "--report", str(report), "--", "--device", "mps", "--output", str(output),
        "--stage", "readout",
        "--authorization", str(authorization), "--config", str(config_path),
        "--prepared", str(_resolve(ROOT, config["dataset"]["root"])),
    ])


def _next_gate(missing: list[str]) -> str:
    if "implemented_derivative_runner" in missing:
        return ("implement the derivative P1/spine orchestration runner and bind "
                "protocol_commit plus implementation hashes")
    if "protocol_commit" in missing:
        return ("commit the implemented derivative runner and bind protocol_commit; "
                "implementation hashes are already pinned")
    if "implementation_sha256" in missing:
        return "refresh the stale implementation hashes after the integrating commit"
    if "prepared_manifest_sha256" in missing:
        return "materialize and hash the independent fit/development fixture"
    if "derivative_qualification_evidence" in missing:
        return "restore or reissue the derivative artifact qualification evidence"
    if "external_derivative_authorization" in missing:
        return ("obtain an external authorization bound to this protocol, derivative "
                "manifest/artifact, qualification evidence and fresh output root")
    if "derivative_p1_spine_admission" in missing:
        return "repair derivative backend admission before any launch"
    return "run the guarded derivative P1 pilot once, with no retries"


def preflight(config_path: Path = CONFIG_PATH, *, authorization: Path | None = None,
              output: Path | None = None, launch: bool = False,
              full_hash: bool = True) -> dict[str, Any]:
    config_path = Path(config_path)
    config = read_json(config_path)
    root = config_path.parent.parent if config_path.parent.name == "configs" else ROOT
    report = validate_config(config, root, full_hash=full_hash)
    output = (_resolve(root, output) if output is not None
              else _resolve(root, config["launch"]["output_root"]))
    missing = _missing_bindings(config, report, authorization, root)
    result = {
        "format": "replica.spark_persistent_learner_derivative.review.v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(config_path),
        "protocol_commit": _current_head(root),
        "runner": runner_binding(config, root),
        "status": "review_ready_preparation_only",
        "native_execution": False,
        "model_loaded": False,
        "tokenizer_loaded": False,
        "labels_read": False,
        "downloads": 0,
        "launch_allowed": False,
        "native_command": None,
        "source": report["source"],
        "admission": report["admission"],
        "work_units": report["work_units"],
        "authorization": {
            "path": str(_resolve(root, config["launch"]["authorization_path"])),
            "present": _resolve(root, config["launch"]["authorization_path"]).is_file(),
            "status": "absent_or_not_validated",
        },
        "output_root": str(output),
        "fresh_output_available": not output.exists(),
        "missing_bindings": missing,
        "next_gate": _next_gate(missing),
        "remaining_prerequisites": [
            *(["commit the implemented derivative runner and bind protocol_commit; implementation hashes are already pinned"]
              if "protocol_commit" in missing else []),
            *(["refresh the stale implementation hashes after the integrating commit"]
              if "implementation_sha256" in missing else []),
            *(["implement the derivative P1/spine orchestration runner"] if "implemented_derivative_runner" in missing else []),
            *(["materialize and hash the independent fit/development fixture without opening sealed-final data"]
              if "prepared_manifest_sha256" in missing else []),
            *(["issue an external authorization for the exact derivative manifest, artifact, qualification evidence, implementation and output"]
              if "external_derivative_authorization" in missing else []),
            "preserve one exclusive worker and the 50 GiB / 512 MiB / 15% host-free guards",
            "run the required P1 numerical checks at experiment scope before any learning claim",
        ],
        "non_claims": [
            "no official FP8 execution or substitution is inferred",
            "no official BF16 equivalence or substitution is inferred",
            "no useful learning, accuracy, speed or RSI claim follows from preparation",
            "no model was loaded and no native job was launched",
        ],
    }
    if launch:
        gates = _missing_bindings(config, report, authorization, root)
        if authorization is None:
            gates.append("external_derivative_authorization")
        _require(not gates, "launch",
                 "refused before model load: " + ", ".join(dict.fromkeys(gates)))
        receipt = _validate_authorization(
            config, authorization, output, root, report["implementation_sha256"],
            config_path=config_path)
        result.update({
            "status": "launch_admitted",
            "launch_allowed": True,
            "native_command": guarded_command(config, output, authorization,
                                              config_path=config_path),
            "authorization": {"path": str(authorization), "present": True,
                              "status": "validated"},
            "authorization_id": receipt["authorization_id"],
            "attempt_id": receipt["attempt_id"],
            "execution_format": receipt["execution_format"],
            "execution_backend": receipt["execution_backend"],
            "artifact_sha256_checked": True,
        })
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--authorization", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--skip-full-hash", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = preflight(args.config, authorization=args.authorization,
                           output=args.output, launch=args.launch,
                           full_hash=not args.skip_full_hash)
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
