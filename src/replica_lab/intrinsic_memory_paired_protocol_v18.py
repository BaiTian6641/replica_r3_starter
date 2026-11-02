"""Exact packet, binding and authorization contracts for paired-update v18.2."""
from __future__ import annotations

import json
from pathlib import Path
import time

from .execution_manifest import build_execution_manifest
from .intrinsic_memory_budget_v12 import COUNTERS, sha_json, validate_plan
from .intrinsic_memory_coverage_v12 import ENVIRONMENT_CLOSURE_BYTES, SOURCE_FILE_BYTES
from .intrinsic_memory_execution_v5 import bound_json, contained, hash_file, local_closure
from .intrinsic_memory_execution_v12 import (
    validate_environment_closure, validate_source_feature_binding)

PACKET = "replica.intrinsic_memory.paired_update_launch_packet.v18.2"
PROTOCOL = "replica.intrinsic_memory.paired_update_protocol.v18.2"
AUTHORIZATION = "replica.intrinsic_memory.paired_update_authorization.v18.2"
OUTER_WALL_SECONDS = 1800
CHECKPOINT_READ_BOUND = 96 * 1024**2
METADATA_READ_BOUND = 64 * 1024**2
PAIR_INDICES = (0, 1, 4, 5)


def training_contract():
    return {"optimizer": "NativeFusedAdamW", "lr": 0.0001, "seed": 17,
            "updates": 12, "teacher_updates": 0, "restart_updates": 0,
            "schedule": "fixed_pairs_0_1_4_5_one_exposure_per_world_per_arm",
            "batch_size_by_arm": {"sequential": 1, "paired_joint": 2}}


def _stage(wall, **counts):
    return {**{key: 0 for key in COUNTERS}, "wall_seconds": wall, **counts}


def diagnostic_plan():
    # 2 baselines + 12 post-step global measurements, 8 worlds, 2 candidates.
    measurements = (2 + 8 + 4) * 8 * 2
    train_forwards, source_forwards = 2 * 8, 8
    return {
        "admission": _stage(180, artifact_read_bytes=SOURCE_FILE_BYTES + ENVIRONMENT_CLOSURE_BYTES
                            + CHECKPOINT_READ_BOUND + METADATA_READ_BOUND,
                            artifact_bytes=4*1024**2, writes=8),
        "load": _stage(240, model_loads=1, restart_restores=1, writes=18, artifact_bytes=1024**2,
                       artifact_read_bytes=SOURCE_FILE_BYTES + CHECKPOINT_READ_BOUND + METADATA_READ_BOUND),
        "diagnose": _stage(1200, forwards=measurements+train_forwards+source_forwards,
            backwards=16, updates=12, input_tokens=(measurements+train_forwards)*256 + 8*512,
            source_encodes=8, source_feature_forwards=8, tokenizer_calls=8,
            prompt_tokenizer_calls=16, source_token_bytes=8*256, cache_misses=8,
            cache_evictions=8, cache_recomputations=8, artifact_read_bytes=16*1024**2,
            artifact_bytes=64*1024**2, writes=5408),
        "finalize": _stage(60, artifact_bytes=2*1024**2, writes=4),
    }


def budget_derivation():
    return {"candidate_forwards": 14*8*2, "training_forwards": 2*8,
            "frozen_source_forwards": 8, "backwards": 2*8, "updates": {"sequential": 8, "paired_joint": 4},
            "all_world_measurements": 14, "producer_computations": 14*8+16,
            "token_ceiling": "240 response forwards * 256 + 8 source spans * (256 tokenize + 256 forward)",
            "telemetry_append_cap": 1200*4 + 1200//2,
            "diagnose_artifact_bytes": {"trajectory_receipts": 4*1024**2, "telemetry": 8*1024**2,
                                        "ledger_and_temporary_overlap": 52*1024**2},
            "pre_attempt_validation": "separate parent ledger; admission and supervisor artifacts included in total_with_preflight",
            "elapsed_wall_ceiling_seconds": OUTER_WALL_SECONDS + 180,
            "reserved_stage_seconds_overlap_parent_supervision": True,
            "snapshot": "source files only; exact size and digest in snapshot manifest; no model copies"}


def parent_plan():
    return {"admission": diagnostic_plan()["admission"],
            "supervise": _stage(OUTER_WALL_SECONDS, writes=32, artifact_bytes=4*1024**2,
                                artifact_read_bytes=16*1024**2)}


def totals(plan):
    return {key: sum(stage[key] for stage in plan.values()) for key in (*COUNTERS, "wall_seconds")}


def validate_protocol(protocol):
    if (protocol.get("format") != PROTOCOL or set(protocol.get("arms", {})) != {"two"}
            or protocol.get("intrinsic_reader_key_mode") != "content"
            or protocol.get("development_labels_used") is not False
            or protocol.get("parameter_updates") != 12
            or protocol.get("protocol_revision") != 2
            or protocol.get("source_repo") != "XHToken/Spark-X2.6-4B"
            or protocol.get("source_revision") != "66e3ae7bee54e0a523d692cb8ef841f35409807b"
            or protocol.get("training") != training_contract()
            or protocol.get("intrinsic_placement") != "block_input_query_to_post_block_residual"):
        raise ValueError("invalid paired-update numerical or source identity")
    diagnostic = protocol.get("diagnostic", {})
    if (diagnostic.get("pair_indices") != list(PAIR_INDICES)
            or diagnostic.get("updates") != 12 or diagnostic.get("development") is not False
            or diagnostic.get("generated_tokens") != 0
            or diagnostic.get("intrinsic_reader_key_mode") != "content"
            or protocol.get("comparison") != {
                "arms": ["sequential", "paired_joint"], "steps": [8, 4],
                "primary": "mean_final_pair_minimum_total_margin", "guard": "worlds_below_own_baseline",
                "loss": "per_world_mean_answer_plus_EOS_CE", "joint_coefficients": [0.5, 0.5],
                "primary_strictly_greater": True, "guard_no_greater": True}):
        raise ValueError("paired-update comparison differs")
    if set(protocol.get("data", {})) != {"fit"}:
        raise ValueError("only fitting data may be bound")
    arm = protocol["arms"]["two"]
    validate_plan(arm["budget"])
    if (arm.get("sites") != [3, 7] or arm["budget"] != diagnostic_plan() or arm["total_reservations"] != totals(diagnostic_plan())
            or arm["wall_seconds"] != OUTER_WALL_SECONDS
            or arm["output_root"] != diagnostic["output_root"]):
        raise ValueError("derived paired-update budget differs")
    combined = {key: value + totals(parent_plan())[key]
                for key, value in totals(diagnostic_plan()).items()}
    if protocol.get("total_with_preflight") != combined or protocol.get("budget_derivation") != budget_derivation():
        raise ValueError("pre-attempt accounting derivation differs")
    if arm["resource_limits"] != {"driver_bytes": 50*1024**3, "swap_growth_bytes": 512*1024**2,
            "minimum_host_free_percent": 15, "sustained_host_samples": 3, "telemetry_timeout_seconds": 30}:
        raise ValueError("native resource limits differ")


def validate_implementation_snapshot(root, protocol):
    implementation = bound_json(root, protocol["implementation"])
    if local_closure(root, implementation["entries"]) != implementation["files"]:
        raise ValueError("executable implementation closure differs")
    snapshot = bound_json(root, protocol["source_snapshot"])
    snapshot_root = contained(root, snapshot["root"])
    if snapshot["files"] != implementation["files"] or local_closure(snapshot_root, implementation["entries"]) != snapshot["files"]:
        raise ValueError("stable execution snapshot differs")
    if sum(contained(snapshot_root, name).stat().st_size for name in snapshot["files"]) != snapshot["source_bytes"]:
        raise ValueError("source snapshot byte count differs")
    return implementation, snapshot


def validate_bindings(root, packet_path, *, check_source=True, closure_interpreter=None):
    root = Path(root)
    packet_path = Path(packet_path)
    packet = json.loads(packet_path.read_text())
    protocol = bound_json(root, packet["protocol"])
    if (packet.get("format") != PACKET or packet.get("arm") != "two"
            or packet.get("attempts") != 1 or packet.get("retries") != 0
            or packet.get("experiment_version") != 18
            or packet.get("native_engine_version") != 18
            or packet.get("native_workers") != 1 or packet.get("cpu_fallback") is not False):
        raise ValueError("invalid single-attempt paired-update packet")
    validate_protocol(protocol)
    parent = bound_json(root, protocol["inherited_protocol"])
    if protocol["inherited_protocol"]["sha256"] != "970155e0684f66d5e32439a21e1ef584db3659e4dba2060a0ad5b0e49ed1012d":
        raise ValueError("inherited V17 numerical contract differs")
    for key in ("source_repo", "source_revision", "source", "source_execution", "source_features",
                "source_feature_binding", "generation", "environment", "environment_closure", "data", "slot_policy_v12"):
        if protocol[key] != parent[key]:
            raise ValueError("inherited numerical contract differs: " + key)
    if protocol["parent_training"] != parent["training"] or protocol["arms"]["two"]["sites"] != [3, 7]:
        raise ValueError("parent training identity or reader sites differ")
    if protocol["diagnostic"]["checkpoint"] != parent["diagnostic"]["checkpoint"]:
        raise ValueError("fixed V16 checkpoint differs")
    if packet["output_root"] != protocol["arms"]["two"]["output_root"]:
        raise ValueError("output binding differs")
    contained(root, packet["output_root"])
    # The pinned parent owns historical qualification metadata. V18 carries
    # only its identical active source contract, without old evaluation pins.
    feature_binding = validate_source_feature_binding(root, parent)
    implementation, snapshot = validate_implementation_snapshot(root, protocol)
    environment = bound_json(root, protocol["environment"])
    environment_closure = validate_environment_closure(
        root, protocol, interpreter=closure_interpreter)
    source = bound_json(root, protocol["source"])
    source_root = contained(root, source["source_root"])
    if (source.get("source_identity") != "official_bf16"
            or source.get("repo_id") != protocol["source_repo"]
            or source.get("revision") != protocol["source_revision"]
            or source.get("derivative") is not False):
        raise ValueError("official BF16 source lineage differs")
    source_execution = bound_json(root, protocol["source_execution"])
    if build_execution_manifest(source_root, artifact_sha256=sha_json(source["files"])) != source_execution:
        raise ValueError("actual model/configuration/local execution code differs")
    started = time.monotonic()
    bytes_checked = 0
    if check_source:
        for item in source["files"]:
            path = contained(root, item["path"])
            if path.stat().st_size != item["bytes"] or hash_file(path) != item["sha256"]:
                raise ValueError("source file content differs: " + item["path"])
            bytes_checked += item["bytes"]
    fit = protocol["data"]["fit"]
    fit_path = contained(root, fit["path"])
    if hash_file(fit_path) != fit["sha256"]:
        raise ValueError("prepared fitting data differs")
    from .intrinsic_memory_paired_update_v18 import validate_pairs
    validate_pairs(json.loads(fit_path.read_text())["rows"])
    checkpoint = protocol["diagnostic"]["checkpoint"]
    checkpoint_root = contained(root, checkpoint["path"])
    manifest = checkpoint_root / "manifest.json"
    if hash_file(manifest) != checkpoint["manifest_sha256"]:
        raise ValueError("interference checkpoint manifest differs")
    components = bound_json(root, protocol["diagnostic"]["checkpoint_components"])
    for key, name in (("weights_sha256", "extension.safetensors"),
                      ("optimizer_sha256", "optimizer.pt")):
        path = checkpoint_root / "components" / name
        if hash_file(path) != checkpoint[key] or components[key] != checkpoint[key]:
            raise ValueError("interference checkpoint component differs")
    if json.loads(manifest.read_text()).get("reader_key_mode") != "content":
        raise ValueError("interference parent must use content keys")
    return packet, protocol, {
        "packet_sha256": hash_file(packet_path),
        "source_bytes_hashed": bytes_checked,
        "source_hash_wall_seconds": time.monotonic() - started,
        "source_qualified_for_loading": check_source,
        "code_before_model_loading": True,
        "implementation_files": len(implementation["files"]),
        "source_snapshot_sha256": protocol["source_snapshot"]["sha256"],
        "pair_membership_validated_before_model_load": True,
        "source_execution_files": len(source_execution["files"]),
        "environment": environment, "environment_closure": environment_closure,
        "feature_binding": {"tokenizer_sha256": feature_binding["tokenizer_sha256"],
                            "original_parameters_sha256": feature_binding[
                                "original_parameters_sha256"]},
        "fitting_data_sha256": fit["sha256"],
        "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
        "development_labels_parsed": False,
        "native_capability": False,
    }


def validate_authorization(root, packet_path, receipt_path, *, allow_existing=False):
    packet = json.loads(Path(packet_path).read_text())
    protocol = bound_json(root, packet["protocol"])
    arm = protocol["arms"]["two"]
    receipt = json.loads(Path(receipt_path).read_text())
    expected = {
        "format": AUTHORIZATION, "decision": "authorized",
        "packet_sha256": hash_file(Path(packet_path)),
        "protocol_sha256": packet["protocol"]["sha256"], "arm": "two",
        "source_kind": "official_bf16", "output_root": packet["output_root"],
        "attempts": 1, "retries": 0, "budget": protocol["total_with_preflight"],
        "implementation_sha256": protocol["implementation"]["sha256"],
        "environment_sha256": protocol["environment"]["sha256"],
        "environment_closure_sha256": protocol["environment_closure"]["sha256"],
        "feature_binding_sha256": sha_json(protocol["source_feature_binding"]),
        "checkpoint_manifest_sha256": protocol["diagnostic"]["checkpoint"]["manifest_sha256"],
        "outer_wall_seconds": arm["wall_seconds"],
        "source_snapshot_sha256": protocol["source_snapshot"]["sha256"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("missing or different exact interference authorization")
    if not receipt.get("human_approval_reference") or not receipt.get("human_approval_text"):
        raise ValueError("direct human approval evidence required")
    review = bound_json(root, receipt["independent_review"])
    if (review.get("decision") != "pass" or review.get("source_snapshot_sha256") != protocol["source_snapshot"]["sha256"]
            or review.get("packet_sha256") != hash_file(Path(packet_path))):
        raise ValueError("independent review must bind this exact source snapshot and packet")
    output = contained(root, packet["output_root"])
    if allow_existing and not output.is_dir():
        raise ValueError("authorized interference output is missing")
    if not allow_existing and output.exists():
        raise ValueError("interference attempt output already exists")
    return receipt
