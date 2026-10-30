"""Exact packet, binding and authorization contracts for the v13 pair diagnostic."""
from __future__ import annotations

import json
from pathlib import Path
import time

from .execution_manifest import build_execution_manifest
from .intrinsic_memory_budget_v12 import COUNTERS, sha_json, validate_plan
from .intrinsic_memory_coverage_v12 import (
    ENVIRONMENT_CLOSURE_BYTES, SOURCE_FILE_BYTES)
from .intrinsic_memory_execution_v5 import (
    bound_json, contained, hash_file, local_closure)
from .intrinsic_memory_execution_v12 import (
    validate_environment_closure, validate_source_feature_binding)

PACKET = "replica.intrinsic_memory.counterfactual_launch_packet.v13"
PROTOCOL = "replica.intrinsic_memory.counterfactual_protocol.v13"
AUTHORIZATION = "replica.intrinsic_memory.counterfactual_authorization.v13"
CONTROLS = ("contextual", "row_permutation", "shuffled", "wrong", "zero", "absent", "exact_text")
OUTER_WALL_SECONDS = 2400
CHECKPOINT_READ_BOUND = 96 * 1024**2
METADATA_READ_BOUND = 64 * 1024**2


def _stage(wall, **counts):
    return {**{key: 0 for key in COUNTERS}, "wall_seconds": wall, **counts}


def diagnostic_plan():
    admission_reads = (SOURCE_FILE_BYTES + ENVIRONMENT_CLOSURE_BYTES
                       + CHECKPOINT_READ_BOUND + METADATA_READ_BOUND)
    load_reads = SOURCE_FILE_BYTES + CHECKPOINT_READ_BOUND + METADATA_READ_BOUND
    return {
        "admission": _stage(120, artifact_read_bytes=admission_reads,
                            artifact_bytes=1024**2, writes=4),
        "load": _stage(180, model_loads=1, restart_restores=1, writes=2,
                       artifact_read_bytes=load_reads),
        "diagnose": _stage(
            1800, forwards=1824, input_tokens=475136, generated_tokens=1792,
            source_encodes=32, source_feature_forwards=32, tokenizer_calls=32,
            prompt_tokenizer_calls=448, source_token_bytes=8192,
            cache_misses=32, cache_evictions=32, cache_recomputations=32,
            artifact_read_bytes=16 * 1024**2, artifact_bytes=256 * 1024**2, writes=40),
        "finalize": _stage(30, artifact_bytes=1024**2, writes=4),
    }


def totals(plan):
    return {key: sum(stage[key] for stage in plan.values()) for key in (*COUNTERS, "wall_seconds")}


def validate_protocol(protocol):
    if (protocol.get("format") != PROTOCOL or set(protocol.get("arms", {})) != {"two"}
            or protocol.get("development_labels_used") is not False
            or protocol.get("parameter_updates") != 0):
        raise ValueError("invalid fitting-only diagnostic protocol")
    diagnostic = protocol.get("diagnostic", {})
    if (diagnostic.get("controls") != list(CONTROLS) or diagnostic.get("pairs") != 16
            or diagnostic.get("rows") != 32 or diagnostic.get("updates") != 0
            or diagnostic.get("development") is not False):
        raise ValueError("diagnostic scope differs")
    if set(protocol.get("data", {})) != {"fit"}:
        raise ValueError("only fitting data may be bound")
    arm = protocol["arms"]["two"]
    plan = arm["budget"]
    validate_plan(plan)
    if plan != diagnostic_plan() or arm["total_reservations"] != totals(plan):
        raise ValueError("diagnostic budget differs from derivation")
    if arm["wall_seconds"] != OUTER_WALL_SECONDS or totals(plan)["wall_seconds"] > OUTER_WALL_SECONDS - 11:
        raise ValueError("diagnostic outer wall differs")
    if arm["output_root"] != protocol["diagnostic"]["output_root"]:
        raise ValueError("diagnostic output binding differs")
    limits = arm["resource_limits"]
    if (limits["driver_bytes"] != 50 * 1024**3
            or limits["swap_growth_bytes"] != 512 * 1024**2
            or limits["minimum_host_free_percent"] != 15):
        raise ValueError("native resource limits differ")


def validate_bindings(root, packet_path, *, check_source=True, closure_interpreter=None):
    root = Path(root)
    packet_path = Path(packet_path)
    packet = json.loads(packet_path.read_text())
    protocol = bound_json(root, packet["protocol"])
    if (packet.get("format") != PACKET or packet.get("arm") != "two"
            or packet.get("attempts") != 1 or packet.get("retries") != 0
            or packet.get("experiment_version") != 13
            or packet.get("native_engine_version") != 13):
        raise ValueError("invalid single-attempt diagnostic packet")
    validate_protocol(protocol)
    if packet["output_root"] != protocol["arms"]["two"]["output_root"]:
        raise ValueError("output binding differs")
    contained(root, packet["output_root"])
    feature_binding = validate_source_feature_binding(root, protocol)
    implementation = bound_json(root, protocol["implementation"])
    if local_closure(root, implementation["entries"]) != implementation["files"]:
        raise ValueError("executable implementation closure differs")
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
    checkpoint = protocol["diagnostic"]["checkpoint"]
    checkpoint_root = contained(root, checkpoint["path"])
    manifest = checkpoint_root / "manifest.json"
    if hash_file(manifest) != checkpoint["manifest_sha256"]:
        raise ValueError("diagnostic checkpoint manifest differs")
    components = bound_json(root, protocol["diagnostic"]["checkpoint_components"])
    for key, name in (("weights_sha256", "extension.safetensors"),
                      ("optimizer_sha256", "optimizer.pt")):
        path = checkpoint_root / "components" / name
        if hash_file(path) != checkpoint[key] or components[key] != checkpoint[key]:
            raise ValueError("diagnostic checkpoint component differs")
    return packet, protocol, {
        "packet_sha256": hash_file(packet_path),
        "source_bytes_hashed": bytes_checked,
        "source_hash_wall_seconds": time.monotonic() - started,
        "source_qualified_for_loading": check_source,
        "code_before_model_loading": True,
        "implementation_files": len(implementation["files"]),
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
        "attempts": 1, "retries": 0, "budget": arm["total_reservations"],
        "implementation_sha256": protocol["implementation"]["sha256"],
        "environment_sha256": protocol["environment"]["sha256"],
        "environment_closure_sha256": protocol["environment_closure"]["sha256"],
        "feature_binding_sha256": sha_json(protocol["source_feature_binding"]),
        "checkpoint_manifest_sha256": protocol["diagnostic"]["checkpoint"]["manifest_sha256"],
        "outer_wall_seconds": arm["wall_seconds"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("missing or different exact diagnostic authorization")
    if not receipt.get("human_approval_reference") or not receipt.get("human_approval_text"):
        raise ValueError("direct human approval evidence required")
    if receipt["human_approval_text"] != protocol["standing_authorization_text"]:
        raise ValueError("authorization text differs from the recorded standing instruction")
    output = contained(root, packet["output_root"])
    if allow_existing and not output.is_dir():
        raise ValueError("authorized diagnostic output is missing")
    if not allow_existing and output.exists():
        raise ValueError("diagnostic attempt output already exists")
    return receipt
