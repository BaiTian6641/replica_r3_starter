#!/usr/bin/env python3
"""Build and verify the review-only smaller-step launch/evidence packet.

The packet is a deterministic JSON document assembled from already-committed
preparation artifacts.  It records the exact config and implementation hashes,
parent/evidence lineage, the full offline launch command, the fresh output
path, the separate authorization contract, preflight checks and every native
step that remains unverified.

It never imports a native runtime, loads a model, runs a launch command, writes
an authorization receipt or authorizes development, held-out, semantic-audit,
V0, promotion or retry access.  Rebuilding from identical repository bytes
produces byte-identical output.

Commands:

``build``
    Recompute and write the packet.  Refuses on protocol drift, wrong limits or
    gates, or any implementation/evidence/prepared hash mismatch.

``verify``
    Recompute the packet and refuse if the checked-in document is stale.

``preflight``
    Emit the review-only launch disposition.  The packet is never an
    authorization receipt, so the disposition is always closed until an
    external receipt bound to a committed protocol passes the real
    ``review-launch`` gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

FORMAT = "replica.vision_smaller_step.launch_packet.v1"
VERSION = "1"
KIND = "preparation_only_launch_evidence"
DEFAULT_CONFIG = "configs/vision_smaller_step_lr3e5_v1.json"
DEFAULT_PACKET = ("reports/vision_smaller_step_launch_packet_v1_20261027/"
                  "launch_packet.json")
RUNNER_PATH = "scripts/vision_smaller_step_v1.py"
CONTRACT_TEST_PATH = "tests/test_vision_smaller_step_v1.py"
BUILDER_PATH = "scripts/build_vision_smaller_step_launch_packet_v1.py"
PACKET_TEST_PATH = "tests/test_vision_smaller_step_launch_packet_v1.py"

EXPECTED_LIMITS = {
    "decoder_forward_calls": 4824,
    "visual_observation_calls": 168,
    "backward_calls": 4,
    "optimizer_steps": 1,
    "wall_seconds": 1200,
    "artifact_bytes": 33554432,
}
EXPECTED_ARTIFACT_INCLUDES_REPORT = True
EXPECTED_MODEL_LOADS = 1
EXPECTED_DOWNLOADS = 0
EXPECTED_RETRIES = 0
EXPECTED_WORKERS = 1
EXPECTED_GENERATED_TOKENS = 3800
EXPECTED_IMAGE_CONTROLS = ["correct", "sibling", "shuffled", "noimage"]
EXPECTED_GENERATION_ROLES = ["answer", "caption"]
EXPECTED_READOUTS = ["prefix_0", "prefix_1"]
EXPECTED_UPDATE_IDS = [
    "caption:vba2-w013-p0-a",
    "caption:vba2-w025-p0-a",
    "answer:vba2-w000-p0-a",
    "answer:vba2-w036-p0-b",
]
EXPECTED_SEED = 29
EXPECTED_LR = 3e-05
EXPECTED_COMPLETED_LR = 0.0003
EXPECTED_OPTIMIZER = "NativeFusedAdamW"
EXPECTED_TENSORS = [
    "vision_projector.0.weight",
    "vision_projector.0.bias",
    "vision_projector.1.weight",
    "vision_projector.1.bias",
    "vision_projector.3.weight",
    "vision_projector.3.bias",
]
EXPECTED_OPTIMIZATION_GATES = {
    "max_update_batch_mean_nll_delta": 0.25,
    "max_wider_answer_mean_nll_delta": 0.25,
    "max_wider_caption_mean_nll_delta": 0.25,
    "update_count": 1,
    "backward_calls": 4,
    "optimizer_steps": 1,
    "optimizer_owned_tensors": 6,
    "backbone_gradient_tensors": 0,
}
EXPECTED_FUNCTIONAL_GATES = {
    "answer_correct_min": 1,
    "answer_correct_denominator": 14,
    "answer_prefill_gold_min": 1,
    "caption_token_f1_gain_min": 0.1,
    "caption_semantic_f1_min": 0.05,
    "correct_vs_sibling_accuracy_margin_min": 0.05,
    "correct_vs_shuffled_accuracy_margin_min": 0.05,
    "correct_vs_noimage_accuracy_margin_min": 0.05,
    "eos_only_rate_max": 0.5,
    "whitespace_only_rate_max": 0.1,
}
EXPECTED_CLOSED_DECISION_FLAGS = (
    "development_authorized_here",
    "heldout_authorized",
    "v0_authorized",
    "promotion",
    "retry_authorized",
)
EVIDENCE_FILES = (
    ("config", "config", "config_sha256"),
    ("report", "report", "report_sha256"),
    ("training_trace", "training_trace", "training_trace_sha256"),
    ("prefix_0_records", "prefix_0_records", "prefix_0_records_sha256"),
    ("prefix_1_records", "prefix_1_records", "prefix_1_records_sha256"),
)
PREPARED_FILES = (
    ("protocol", "protocol", "protocol_sha256"),
    ("manifest", "manifest", "manifest_sha256"),
    ("training", "training", "training_sha256"),
)
EVIDENCE_VALUES = (
    "lr",
    "pre_mean_response_nll",
    "post_mean_response_nll",
    "post_mean_response_nll_rounded",
    "delta_post_minus_pre",
    "answer_correct_cases",
    "answer_correct_denominator",
    "answer_prefill_gold_cases",
    "caption_semantic_f1",
    "post_source",
)
PACKET_KEYS = {
    "format", "version", "kind", "deterministic", "native_execution",
    "model_loads", "authorization_created", "protocol", "implementation",
    "lineage", "scope", "gates", "limits", "launch", "authorization",
    "preflight", "unverified_native_steps", "limitations", "packet_sha256",
}
IMPLEMENTATION_KEYS = {
    "config", "runner", "contract_test", "packet_builder", "packet_test",
    "pins",
}
LINEAGE_KEYS = {
    "parent", "initialization", "precondition", "evidence", "prepared",
    "schedule", "update_batch",
}
LIMITS_KEYS = set(EXPECTED_LIMITS) | {
    "artifact_includes_final_report", "model_loads", "downloads", "retries",
    "workers", "cpu_fallback",
}
AUTHORIZATION_KEYS = {
    "required", "created", "present_at_build", "path", "format", "status",
    "protocol_commit_required", "protocol_commit", "commit_binding_required",
    "commit_bound", "config_sha256_binding_required", "config_sha256",
    "native_execution_authorized", "schema", "separate_from_packet",
    "minted_by_packet", "closed_roles", "verification",
}


class PacketError(ValueError):
    """The launch/evidence packet is incomplete or its sources drifted."""


class StalePacketError(PacketError):
    """The checked-in packet no longer matches its sources."""


class LaunchRefused(PacketError):
    """The packet cannot authorize a native launch."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise PacketError(message)


def _exact_keys(value: Any, keys: set[str], label: str) -> None:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    _require(set(value) == keys,
             f"{label} fields differ: {sorted(set(value) ^ keys)}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise PacketError(f"cannot read JSON {path}: {error}") from error
    _require(isinstance(payload, dict), f"JSON must be an object: {path}")
    return dict(payload)


def _relative(path: str | Path) -> str:
    return Path(path).as_posix()


def _file_entry(root: Path, relative: str) -> dict[str, Any]:
    path = Path(root) / relative
    _require(path.is_file(), f"packet source is missing: {relative}")
    payload = path.read_bytes()
    return {
        "path": _relative(relative),
        "sha256": _sha256_bytes(payload),
        "bytes": len(payload),
    }


def _verified_entry(root: Path, relative: str, expected: str,
                    label: str) -> dict[str, Any]:
    _require(isinstance(expected, str) and len(expected) == 64,
             f"declared hash is invalid: {label}")
    entry = _file_entry(root, relative)
    _require(entry["sha256"] == expected,
             f"packet source hash differs: {label}: {relative}")
    entry["declared_sha256"] = expected
    entry["verified"] = True
    return entry


def _protocol_module():
    name = "vision_smaller_step_launch_packet_dependency"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, ROOT / RUNNER_PATH)
    _require(spec is not None and spec.loader is not None,
             f"cannot load protocol runner: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _learn_module():
    from replica_lab import vision_learnability_diagnostic as learn
    return learn


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise PacketError(f"declared {label} differs: {actual!r} != "
                          f"{expected!r}")


def _derive_limits(config: Mapping[str, Any]) -> dict[str, Any]:
    limits = config["limits"]
    campaign = config["accounting"]["campaign_cumulative"]
    work = config["readouts"]["expected_measured_work"]
    observed = {
        "decoder_forward_calls": [
            limits["max_decoder_forwards"],
            campaign["decoder_forward_calls"],
        ],
        "visual_observation_calls": [
            limits["max_visual_encoder_calls"],
            campaign["visual_observation_calls"],
            work["visual_observation_calls"],
        ],
        "backward_calls": [
            limits["max_total_backward_calls"],
            campaign["backward_calls"],
            work["backward_calls"],
        ],
        "optimizer_steps": [
            limits["max_optimizer_steps"],
            campaign["optimizer_steps"],
            work["optimizer_steps"],
        ],
        "wall_seconds": [
            limits["max_seconds"],
            config["accounting"]["wall_limit_seconds"],
        ],
        "artifact_bytes": [
            limits["artifact_bytes"],
            config["accounting"]["artifact_bytes"],
        ],
    }
    for key, expected in EXPECTED_LIMITS.items():
        for value in observed[key]:
            _require_equal(value, expected, f"limit:{key}")
    _require_equal(limits["model_loads"], EXPECTED_MODEL_LOADS,
                   "limit:model_loads")
    _require_equal(limits["downloads"], EXPECTED_DOWNLOADS,
                   "limit:downloads")
    _require_equal(limits["retries"], EXPECTED_RETRIES, "limit:retries")
    _require_equal(limits["workers"], EXPECTED_WORKERS, "limit:workers")
    _require_equal(config["policy"]["cpu_fallback"], False,
                   "limit:cpu_fallback")
    _require_equal(config["policy"]["memory_guard"], {
        "driver_bytes": 53687091200,
        "swap_growth_bytes": 536870912,
        "host_free_fraction": 0.15,
        "exclusive_worker": True,
        "cpu_fallback": False,
    }, "memory_guard")
    return {
        **EXPECTED_LIMITS,
        "artifact_includes_final_report": EXPECTED_ARTIFACT_INCLUDES_REPORT,
        "model_loads": EXPECTED_MODEL_LOADS,
        "downloads": EXPECTED_DOWNLOADS,
        "retries": EXPECTED_RETRIES,
        "workers": EXPECTED_WORKERS,
        "cpu_fallback": False,
    }


def _implementation_block(config: Mapping[str, Any], config_path: str,
                          root: Path) -> dict[str, Any]:
    config_entry = _file_entry(root, _relative(config_path))
    runner_entry = _file_entry(root, RUNNER_PATH)
    contract_entry = _file_entry(root, CONTRACT_TEST_PATH)
    builder_entry = _file_entry(root, BUILDER_PATH)
    packet_test_entry = _file_entry(root, PACKET_TEST_PATH)
    pins: dict[str, Any] = {}
    declared = config["implementation_reuse"]["pins"]
    _require(isinstance(declared, Mapping) and declared,
             "implementation pins are missing")
    for relative, expected in declared.items():
        pins[_relative(relative)] = _verified_entry(
            root, relative, expected, f"implementation pin:{relative}")
    _require_equal(config["implementation_reuse"]["source_edits"], 0,
                   "implementation source_edits")
    _require_equal(config["implementation_reuse"]["native_execution"], False,
                   "implementation native_execution")
    return {
        "config": config_entry,
        "runner": runner_entry,
        "contract_test": contract_entry,
        "packet_builder": builder_entry,
        "packet_test": packet_test_entry,
        "pins": pins,
    }


def _parent_block(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    source = config["source"]
    lock = _verified_entry(root, source["model_lock"],
                           source["model_lock_sha256"], "model_lock")
    manifest = _verified_entry(
        root, source["vision_source_manifest"],
        source["vision_source_manifest_sha256"], "vision_source_manifest")
    model_path = Path(root) / source["model_path"]
    return {
        "model_path": source["model_path"],
        "model_path_present": model_path.exists(),
        "model_lock": lock,
        "model_revision": source["model_revision"],
        "vision_source_manifest": manifest,
        "sealed_parent": True,
        "downloads": 0,
        "parent_weights_hashed_here": False,
        "note": "The sealed parent is never loaded by this packet builder.",
    }


def _initialization_block(config: Mapping[str, Any]) -> dict[str, Any]:
    initialization = config["initialization"]
    _require_equal(initialization["init_seed"], EXPECTED_SEED,
                   "initialization seed")
    _require_equal(initialization["warm_start"], None,
                   "initialization warm_start")
    _require_equal(list(initialization["projector_tensors"]),
                   EXPECTED_TENSORS, "initialization projector tensors")
    return {
        "init_seed": initialization["init_seed"],
        "recipe": initialization["recipe"],
        "warm_start": None,
        "parameter_count": initialization["parameter_count"],
        "expected_init_sha256": initialization["expected_init_sha256"],
        "projector_tensors": list(initialization["projector_tensors"]),
        "forbidden_parent_weights_sha256":
            list(initialization["forbidden_parent_weights_sha256"]),
        "verification": "pending_native_initialization_hash_assertion",
    }


def _precondition_block(config: Mapping[str, Any],
                        root: Path) -> dict[str, Any]:
    precondition = config["precondition"]
    report_path = precondition["report"]
    entry = _file_entry(root, report_path)
    _verified_entry(root, precondition["config"],
                    precondition["config_sha256"], "precondition config")
    report = _read_json(Path(root) / report_path)
    learn = _learn_module()
    validation = learn.ablation_precondition_report(
        report,
        required_decision=precondition["required_decision"],
        required_mode=precondition["required_mode"],
        config_sha256=precondition["config_sha256"])
    _require(validation["ok"],
             f"precondition is not satisfied: {validation['violations']}")
    decision = report.get("decision") or {}
    return {
        "run": precondition["run"],
        "report": entry,
        "required_decision": precondition["required_decision"],
        "required_mode": precondition["required_mode"],
        "observed_decision": decision.get("decision"),
        "observed_mode": report.get("mode"),
        "config": precondition["config"],
        "config_sha256": precondition["config_sha256"],
        "validation": validation,
        "satisfied": True,
        "note": precondition["note"],
    }


def _evidence_block(config: Mapping[str, Any],
                    root: Path) -> dict[str, Any]:
    evidence = config["evidence"]
    arm = evidence["completed_arm"]
    update = evidence["completed_update_batch"]
    files = {
        label: _verified_entry(root, arm[path_key], arm[hash_key],
                               f"completed_arm:{label}")
        for label, path_key, hash_key in EVIDENCE_FILES
    }
    values = {key: update[key] for key in EVIDENCE_VALUES if key != "lr"}
    values["lr"] = arm["lr"]
    _require_equal(values["lr"], EXPECTED_COMPLETED_LR,
                   "completed-arm lr")
    return {
        "completed_arm": {
            "files": files,
            "values": values,
            "anchor_policy": evidence["anchor_policy"],
            "values_are_comparison_anchors_not_gates": True,
        },
        "completed_update_batch": {
            "example_ids": list(update["example_ids"]),
            "case_ids": list(update["case_ids"]),
            "pre_mean_response_nll": update["pre_mean_response_nll"],
            "post_mean_response_nll": update["post_mean_response_nll"],
            "delta_post_minus_pre": update["delta_post_minus_pre"],
            "answer_correct_cases": update["answer_correct_cases"],
            "answer_correct_denominator":
                update["answer_correct_denominator"],
            "caption_semantic_f1": update["caption_semantic_f1"],
            "post_source": update["post_source"],
        },
    }


def _prepared_block(config: Mapping[str, Any],
                    root: Path) -> dict[str, Any]:
    prepared = config["prepared"]
    files = {
        label: _verified_entry(root, prepared[path_key],
                               prepared[hash_key],
                               f"prepared:{label}")
        for label, path_key, hash_key in PREPARED_FILES
    }
    return {
        "root": prepared["root"],
        "files": files,
        "training_examples": prepared["training_examples"],
        "answer_examples": prepared["answer_examples"],
        "caption_examples": prepared["caption_examples"],
        "answer_pairs": prepared["answer_pairs"],
        "caption_pair_ids": prepared["caption_pair_ids"],
        "read_only_training": prepared["read_only_training"],
        "forbidden_access": list(prepared["forbidden_access"]),
    }


def _schedule_block(config: Mapping[str, Any],
                    root: Path) -> dict[str, Any]:
    prepared = config["prepared"]
    schedule = config["schedule"]
    source = _verified_entry(
        root, f"{prepared['root']}/{schedule['source']}",
        schedule["source_sha256"], "schedule source")
    labels = _verified_entry(
        root, f"{prepared['root']}/{schedule['caption_labels']}",
        schedule["caption_labels_sha256"], "schedule caption labels")
    return {
        "source": source,
        "caption_labels": labels,
        "update_index": schedule["update_index"],
        "updates_planned": schedule["updates_planned"],
        "batch_size": schedule["batch_size"],
        "answer_per_update": schedule["answer_per_update"],
        "caption_per_update": schedule["caption_per_update"],
        "unique_rows": schedule["unique_rows"],
        "slots": schedule["slots"],
        "expected_answer_rows": schedule["expected_answer_rows"],
        "expected_caption_rows": schedule["expected_caption_rows"],
        "expected_repeated_rows": schedule["expected_repeated_rows"],
        "smallest_full_coverage_prefix":
            schedule["smallest_full_coverage_prefix"],
        "completed_batch_reused": schedule["completed_batch_reused"],
        "sequence": list(config["readouts"]["sequence"]),
    }


def _update_batch_block(config: Mapping[str, Any],
                        evidence: Mapping[str, Any]) -> dict[str, Any]:
    update = config["update_batch"]
    _require_equal(list(update["example_ids"]), EXPECTED_UPDATE_IDS,
                   "update-batch example ids")
    _require_equal(update["rows"], 4, "update-batch rows")
    _require_equal(dict(update["roles"]), {"answer": 2, "caption": 2},
                   "update-batch roles")
    completed = evidence["completed_update_batch"]
    _require_equal(list(completed["example_ids"]), EXPECTED_UPDATE_IDS,
                   "completed update-batch example ids")
    return {
        "source": update["source"],
        "update_index": update["update_index"],
        "example_ids": list(update["example_ids"]),
        "case_ids": list(update["case_ids"]),
        "rows": update["rows"],
        "roles": dict(update["roles"]),
        "pre_evaluation_label": update["pre_evaluation_label"],
        "post_evaluation_label": update["post_evaluation_label"],
        "evaluation_scope": update["evaluation_scope"],
        "evaluation_controls": list(update["evaluation_controls"]),
        "evaluation_records": update["evaluation_records"],
        "separate_from_wider_cohort": update["separate_from_wider_cohort"],
        "loss_metric": update["loss_metric"],
        "completed_batch_reused": True,
        "completed_pre_mean_response_nll":
            completed["pre_mean_response_nll"],
        "completed_post_mean_response_nll":
            completed["post_mean_response_nll"],
        "completed_delta_post_minus_pre":
            completed["delta_post_minus_pre"],
    }


def _lineage_block(config: Mapping[str, Any], root: Path,
                   evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "parent": _parent_block(config, root),
        "initialization": _initialization_block(config),
        "precondition": _precondition_block(config, root),
        "evidence": evidence,
        "prepared": _prepared_block(config, root),
        "schedule": _schedule_block(config, root),
        "update_batch": _update_batch_block(config, evidence),
    }


def _scope_block(config: Mapping[str, Any]) -> dict[str, Any]:
    optimizer = config["optimizer"]
    scope = config["gradient_scope"]
    _require_equal(optimizer["lr"], EXPECTED_LR, "optimizer lr")
    _require_equal(config["hypothesis"]["completed_lr"],
                   EXPECTED_COMPLETED_LR, "completed lr")
    _require_equal(optimizer["wrapper"], EXPECTED_OPTIMIZER,
                   "optimizer wrapper")
    _require_equal(list(scope["trainable_tensors"]), EXPECTED_TENSORS,
                   "gradient scope tensors")
    _require_equal(scope["optimizer_owned_parameters"], 6,
                   "optimizer ownership")
    _require_equal(scope["backbone_gradient_tensors_allowed"], 0,
                   "backbone gradients")
    _require_equal(list(scope["nonprojector_gradient_names_allowed"]), [],
                   "non-projector gradient names")
    _require_equal(scope["backward_calls"], 4, "backward calls")
    _require_equal(scope["optimizer_steps"], 1, "optimizer steps")
    return {
        "optimizer": dict(optimizer),
        "gradient_scope": dict(scope),
        "single_variable": {
            "changed": "optimizer.lr",
            "completed": EXPECTED_COMPLETED_LR,
            "arm": EXPECTED_LR,
            "statement": config["hypothesis"]["single_discriminating_change"],
        },
        "preserved": {
            "parent": config["source"]["model_path"],
            "seed": EXPECTED_SEED,
            "update_batch_example_ids": list(EXPECTED_UPDATE_IDS),
            "projector_tensors": list(EXPECTED_TENSORS),
            "optimizer_wrapper": EXPECTED_OPTIMIZER,
            "response_only_labels": True,
            "gradient_scope_unchanged": True,
            "completed_reproduction_repeated": False,
        },
    }


def _raw_generation_block(config: Mapping[str, Any]) -> dict[str, Any]:
    readouts = config["readouts"]
    wider = readouts["wider"]
    _require_equal(list(wider["controls"]), EXPECTED_IMAGE_CONTROLS,
                   "image controls")
    _require_equal(list(wider["generation_roles"]),
                   EXPECTED_GENERATION_ROLES, "generation roles")
    _require_equal(list(wider["teacher_forced_roles"]),
                   EXPECTED_GENERATION_ROLES, "teacher-forced roles")
    _require_equal(wider["eos_only_not_counted_correct"], True,
                   "EOS-only accounting")
    _require_equal(wider["whitespace_only_not_counted_correct"], True,
                   "whitespace-only accounting")
    _require_equal(EXPECTED_GENERATED_TOKENS,
                   config["accounting"]["campaign_cumulative"]
                   ["generated_tokens"], "generated tokens")
    per_readout = []
    for label in EXPECTED_READOUTS:
        base = f"{config['output']}/readouts/{label}"
        per_readout.append({
            "label": label,
            "readout": f"{base}/readout.json",
            "records": f"{base}/records.jsonl",
            "answer_generations": f"{base}/train_answer_generations.jsonl",
            "caption_generations":
                f"{base}/train_caption_generations.jsonl",
            "controls": list(wider["controls"]),
            "records_expected": wider["records_per_readout"],
            "available": wider["available_per_readout"],
            "placeholders": wider["placeholders_per_readout"],
            "logical_scoring_cells":
                wider["logical_scoring_cells_per_readout"],
            "logical_scoring_operations":
                wider["logical_scoring_operations_per_readout"],
        })
    return {
        "required": True,
        "raw_generation_required": True,
        "output_root": config["output"],
        "readouts": list(EXPECTED_READOUTS),
        "generation_roles": list(EXPECTED_GENERATION_ROLES),
        "per_readout": per_readout,
        "update_batch_generation": False,
        "update_batch_records": [
            f"{config['output']}/readouts/update_batch_pre/records.jsonl",
            f"{config['output']}/readouts/update_batch_post/records.jsonl",
        ],
        "max_new_tokens_by_kind": dict(wider["max_new_tokens_by_kind"]),
        "expected_generated_tokens": EXPECTED_GENERATED_TOKENS,
        "empty_text_policy": wider["empty_text_policy"],
        "eos_only_not_counted_correct": True,
        "whitespace_only_not_counted_correct": True,
        "controls_required": list(EXPECTED_IMAGE_CONTROLS),
    }


def _gates_block(config: Mapping[str, Any]) -> dict[str, Any]:
    gates = config["gates"]
    optimization = gates["optimization"]
    functional = gates["functional_fit"]
    decision = gates["decision"]
    for key, value in EXPECTED_OPTIMIZATION_GATES.items():
        _require_equal(optimization[key], value, f"optimization gate:{key}")
    for key, value in EXPECTED_FUNCTIONAL_GATES.items():
        _require_equal(functional[key], value, f"functional gate:{key}")
    for key in EXPECTED_CLOSED_DECISION_FLAGS:
        _require_equal(decision[key], False, f"decision flag:{key}")
    _require_equal(decision["requires_all_gates"], True,
                   "decision requires_all_gates")
    return {
        "optimization": dict(optimization),
        "functional_fit": dict(functional),
        "decision": dict(decision),
        "image_controls": list(EXPECTED_IMAGE_CONTROLS),
        "raw_generation_evidence": _raw_generation_block(config),
        "expected_measured_work": dict(
            config["readouts"]["expected_measured_work"]),
        "decoder_forward_accounting": {
            "limit": config["limits"]["max_decoder_forwards"],
            "campaign_cumulative":
                config["accounting"]["campaign_cumulative"]
                ["decoder_forward_calls"],
            "declared_expected_measured_work":
                config["readouts"]["expected_measured_work"]
                ["decoder_forward_calls"],
            "note": ("The enforceable cap and campaign cumulative are 4824 "
                     "decoder forwards. The config's separate "
                     "expected_measured_work field records 4804 and is not "
                     "the limit."),
        },
    }


def _launch_block(config: Mapping[str, Any], config_path: str,
                  module: Any) -> dict[str, Any]:
    output = config["output"]
    _require(output.startswith("runs/"),
             f"fresh output must live under runs/: {output}")
    receipt = config["authorization"]["receipt_path"]
    argv = [
        "python", RUNNER_PATH, "run",
        "--config", _relative(config_path),
        "--output", output,
        "--model-path", config["source"]["model_path"],
        "--model-lock", config["source"]["model_lock"],
        "--authorization", receipt,
    ]
    environment = dict(module.NATIVE_ENVIRONMENT)
    _require_equal(environment, {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    }, "native environment")
    prefix = " ".join(f"{key}={shlex.quote(value)}"
                      for key, value in environment.items())
    return {
        "cwd": ".",
        "argv": argv,
        "shell": f"{prefix} {shlex.join(argv)}",
        "environment": environment,
        "environment_override_policy":
            "forced before native imports; caller values are replaced",
        "offline_required": True,
        "mps_cpu_fallback_disabled": True,
        "fresh_output": {
            "path": output,
            "must_not_exist": True,
            "checked_before_model_load": True,
        },
        "authorization_path": receipt,
        "authorization_before_native_imports": True,
        "native_imports_before_authorization": 0,
        "model_loads_before_authorization": 0,
        "preflight_command": [
            "python", RUNNER_PATH, "review-launch",
            "--config", _relative(config_path),
            "--authorization", receipt,
        ],
    }


def _authorization_block(config: Mapping[str, Any], config_path: str,
                         config_sha256: str,
                         root: Path) -> dict[str, Any]:
    authorization = config["authorization"]
    _require_equal(authorization["native_execution_authorized"], False,
                   "predeclared native authorization")
    _require_equal(authorization["requires_protocol_commit"], True,
                   "protocol commit requirement")
    _require_equal(authorization["requires_exact_config_sha256"], True,
                   "config hash requirement")
    _require_equal(authorization["requires_commit_binding"], True,
                   "commit binding requirement")
    _require_equal(authorization["protocol_commit"], None,
                   "predeclared protocol commit")
    receipt = authorization["receipt_path"]
    _require(receipt != _relative(config_path),
             "authorization receipt must be separate from the protocol")
    required = list(authorization["required_receipt_fields"])
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": {
            "format": {"const": authorization["receipt_format"]},
            "status": {"const": authorization["required_status"]},
            "protocol_name": {"const": config["name"]},
            "protocol_path": {"const": _relative(config_path)},
            "protocol_commit": {
                "type": "string",
                "pattern": "^[0-9a-f]{40}$",
            },
            "config_sha256": {"const": config_sha256},
            "native_execution_authorized": {"const": True},
            "dev_authorized": {"const": False},
            "heldout_authorized": {"const": False},
            "v0_authorized": {"const": False},
            "semantic_audit_authorized": {"const": False},
            "promotion_authorized": {"const": False},
            "retry_authorized": {"const": False},
            "issued_at_utc": {"type": "string", "minLength": 1},
        },
    }
    return {
        "required": True,
        "created": False,
        "present_at_build": (root / receipt).is_file(),
        "path": receipt,
        "format": authorization["receipt_format"],
        "status": "absent",
        "protocol_commit_required": True,
        "protocol_commit": None,
        "commit_binding_required": True,
        "commit_bound": False,
        "config_sha256_binding_required": True,
        "config_sha256": config_sha256,
        "native_execution_authorized": False,
        "schema": schema,
        "separate_from_packet": True,
        "minted_by_packet": False,
        "closed_roles": {
            "development": False,
            "heldout": False,
            "v0": False,
            "semantic_audit": False,
            "promotion": False,
            "retry": False,
        },
        "verification": ("scripts/vision_smaller_step_v1.py review-launch "
                         "and run validate this receipt, the commit-bound "
                         "config bytes and the fresh output before any "
                         "native import or model load"),
    }


def _preflight_block() -> list[dict[str, Any]]:
    return [
        {"id": "protocol_contract", "status": "passed",
         "requirement": "declared protocol validates exactly on CPU"},
        {"id": "config_sha256", "status": "passed",
         "requirement": "packet binds the exact config bytes"},
        {"id": "implementation_pins", "status": "passed",
         "requirement": "runner, tests, builder and reused modules match "
                        "their declared hashes"},
        {"id": "evidence_lineage", "status": "passed",
         "requirement": "completed arm and update-batch evidence hashes "
                        "match"},
        {"id": "prepared_inputs", "status": "passed",
         "requirement": "prepared protocol, manifest, training and schedule "
                        "hashes match"},
        {"id": "precondition_decision", "status": "passed",
         "requirement": "preserved ablation report records "
                        "spatial_contribution_observed"},
        {"id": "single_variable_scope", "status": "passed",
         "requirement": "only optimizer.lr changes, 3e-05 versus 3e-04"},
        {"id": "limits_exact", "status": "passed",
         "requirement": "4824 decoder forwards, 168 visual observations, "
                        "4 backward calls, 1 optimizer step, 1200 seconds, "
                        "33554432 artifact bytes including the report"},
        {"id": "gates_declared", "status": "passed",
         "requirement": "optimization and functional gates are declared and "
                        "decision access remains closed"},
        {"id": "protocol_commit", "status": "pending",
         "requirement": "a 40-character commit containing the exact config "
                        "bytes must be recorded in the receipt"},
        {"id": "authorization_receipt", "status": "pending",
         "requirement": "a separate external receipt must authorize native "
                        "execution; the packet never creates one"},
        {"id": "fresh_output", "status": "pending",
         "requirement": "the output path must not exist at launch"},
        {"id": "native_execution", "status": "not_performed",
         "requirement": "no model load, backward call, optimizer step or "
                        "native command is run during packet preparation"},
        {"id": "native_environment", "status": "declared",
         "requirement": "offline hub/datasets/transformers and disabled MPS "
                        "CPU fallback are forced before native imports"},
    ]


def _unverified_native_steps() -> list[str]:
    return [
        "Commit the protocol config (and this packet) and record the "
        "resulting full 40-character commit SHA in the separate "
        "authorization receipt.",
        "Issue the external authorization receipt bound to the exact config "
        "SHA-256 and commit; this builder never mints one.",
        "Run review-launch: commit binding, receipt fields and the fresh "
        "output path must pass before any native import.",
        "Acquire the exclusive MPS worker and prove the capacity guard: "
        "50 GiB driver, 512 MiB sustained swap growth, 15% host free, no "
        "CPU fallback.",
        "Load the sealed MiniCPM5 parent once and frozen SigLIP2, then "
        "verify the seed-29 projector initialization hash natively.",
        "Perform exactly 4 backward calls and 1 NativeFusedAdamW optimizer "
        "step at lr 3e-05, proving only the six projector tensors change.",
        "Collect prefix_0, update_batch_pre, update_1, update_batch_post and "
        "prefix_1 evidence, including raw answer/caption generations and "
        "correct/sibling/shuffled/no-image controls.",
        "Measure decoder forwards, visual observations, backward calls, "
        "optimizer steps, wall seconds and artifact bytes including the "
        "final report against the declared caps.",
        "Evaluate the optimization and functional-fit gates; no outcome "
        "authorizes development, held-out, semantic-audit, V0, promotion "
        "or retry.",
    ]


def _limitations() -> list[str]:
    return [
        "Preparation only: no native command, model load, backward call, "
        "optimizer step or generation has been executed by this packet.",
        "The authorization receipt is absent and the protocol commit is "
        "unbound, so the launch is refused before native loading.",
        "Numeric losses, gate outcomes, wall time and artifact bytes are "
        "unobserved; 4824 forwards, 168 observations, 4 backward calls, "
        "1 optimizer step, 1200 seconds and 33554432 bytes are declarations "
        "with exact equality checks, not measurements.",
        "Native memory-guard behavior, fused AdamW availability on this MPS "
        "host and the seed-29 initialization hash remain unverified.",
        "Generated-answer correctness, caption quality, EOS-only and "
        "whitespace-only accounting, and image-control separation remain "
        "unverified until raw generations exist.",
        "A passing optimization gate is only absence of the declared loss "
        "excursion; it is not generated-answer conversion or perceptual "
        "learning.",
        "Development, held-out, semantic-audit, V0, promotion and retry "
        "access remain closed regardless of this packet or any result.",
        "This packet is evidence and review material, never an "
        "authorization receipt or a promotion decision.",
    ]


def _packet_sha256(packet: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in packet.items()
               if key != "packet_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True).encode("utf-8")
    return _sha256_bytes(canonical)


def _protocol_block(config: Mapping[str, Any], config_path: str,
                    root: Path) -> dict[str, Any]:
    entry = _file_entry(root, _relative(config_path))
    return {
        "name": config["name"],
        "mode": config["mode"],
        "version": config["version"],
        "config_path": entry["path"],
        "config_sha256": entry["sha256"],
        "config_bytes": entry["bytes"],
        "declared_at_utc": config["declared_at_utc"],
        "lr": config["optimizer"]["lr"],
        "completed_lr": config["hypothesis"]["completed_lr"],
        "single_discriminating_change":
            config["hypothesis"]["single_discriminating_change"],
        "hypothesis_scope": config["claims"]["hypothesis_scope"],
        "native_execution_authorized": False,
        "development_authorized": False,
    }


def _build_packet_body(config_path: str | Path = DEFAULT_CONFIG, *,
                       root: str | Path = ROOT) -> dict[str, Any]:
    root_path = Path(root).resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root_path / config_path
    config_path = config_path.resolve()
    _relative_path = config_path.relative_to(root_path).as_posix()
    try:
        module = _protocol_module()
        config = module.load_protocol(config_path, root=root_path)
    except Exception as error:  # noqa: BLE001 - protocol drift refusal
        if isinstance(error, PacketError):
            raise
        raise PacketError(f"protocol validation refused packet: {error}") \
            from error
    digest = _sha256_bytes(config_path.read_bytes())
    _require_equal(config["name"], module.NAME, "protocol name")
    _require_equal(config["optimizer"]["lr"], module.LR, "protocol lr")
    _require_equal(config["optimizer"]["wrapper"], EXPECTED_OPTIMIZER,
                   "optimizer wrapper")
    _require_equal(list(config["gradient_scope"]["trainable_tensors"]),
                   list(module.PROJECTOR_TENSORS),
                   "projector tensors")
    _require_equal(list(config["update_batch"]["example_ids"]),
                   list(module.UPDATE_EXAMPLE_IDS), "update example ids")
    limits = _derive_limits(config)
    implementation = _implementation_block(config, _relative_path, root_path)
    _require_equal(implementation["config"]["sha256"], digest,
                   "implementation config hash")
    evidence = _evidence_block(config, root_path)
    body = {
        "format": FORMAT,
        "version": VERSION,
        "kind": KIND,
        "deterministic": True,
        "native_execution": False,
        "model_loads": 0,
        "authorization_created": False,
        "protocol": _protocol_block(config, _relative_path, root_path),
        "implementation": implementation,
        "lineage": _lineage_block(config, root_path, evidence),
        "scope": _scope_block(config),
        "gates": _gates_block(config),
        "limits": limits,
        "launch": _launch_block(config, _relative_path, module),
        "authorization": _authorization_block(
            config, _relative_path, digest, root_path),
        "preflight": _preflight_block(),
        "unverified_native_steps": _unverified_native_steps(),
        "limitations": _limitations(),
    }
    body["packet_sha256"] = _packet_sha256(body)
    return body


def build_packet(config_path: str | Path = DEFAULT_CONFIG, *,
                 root: str | Path = ROOT) -> dict[str, Any]:
    """Build the deterministic packet without writing it anywhere."""
    packet = _build_packet_body(config_path, root=root)
    _require(set(packet) == PACKET_KEYS, "packet fields differ")
    _require_equal(set(packet["implementation"]), IMPLEMENTATION_KEYS,
                   "implementation fields")
    _require_equal(set(packet["lineage"]), LINEAGE_KEYS, "lineage fields")
    _require_equal(set(packet["limits"]), LIMITS_KEYS, "limits fields")
    _require_equal(set(packet["authorization"]), AUTHORIZATION_KEYS,
                   "authorization fields")
    return packet


def packet_bytes(packet: Mapping[str, Any]) -> bytes:
    return (json.dumps(packet, indent=2, sort_keys=True,
                       ensure_ascii=True) + "\n").encode("utf-8")


def write_packet(packet: Mapping[str, Any],
                 path: str | Path = DEFAULT_PACKET) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = packet_bytes(packet)
    target.write_bytes(payload)
    return _sha256_bytes(payload)


def verify_packet(packet: Mapping[str, Any], *,
                  config_path: str | Path = DEFAULT_CONFIG,
                  root: str | Path = ROOT) -> dict[str, Any]:
    """Refuse a packet whose hashes, limits, gates or sources drifted."""
    _exact_keys(packet, PACKET_KEYS, "packet")
    declared_hash = packet["packet_sha256"]
    if not isinstance(declared_hash, str) or len(declared_hash) != 64:
        raise StalePacketError("packet_sha256 is invalid")
    if _packet_sha256(packet) != declared_hash:
        raise StalePacketError(
            "packet sha256 differs from its canonical payload")
    try:
        expected = _build_packet_body(config_path, root=root)
    except PacketError as error:
        raise StalePacketError(f"packet sources are stale: {error}") from error
    for key in sorted(PACKET_KEYS - {"packet_sha256"}):
        if packet[key] != expected[key]:
            raise StalePacketError(f"stale packet section: {key}")
    return dict(packet)


def launch_preflight(packet: Mapping[str, Any], *,
                     config_path: str | Path = DEFAULT_CONFIG,
                     root: str | Path = ROOT) -> dict[str, Any]:
    """Return the review-only launch disposition; never executes anything."""
    verify_packet(packet, config_path=config_path, root=root)
    authorization = packet["authorization"]
    launch = packet["launch"]
    refusals = [
        "packet_is_not_an_authorization_receipt",
        "packet_is_preparation_only",
    ]
    if not (Path(root) / authorization["path"]).is_file():
        refusals.append("authorization_receipt_missing")
    if authorization["protocol_commit"] is None:
        refusals.append("protocol_commit_unbound")
    if authorization["commit_bound"] is not True:
        refusals.append("commit_binding_unverified")
    if authorization["native_execution_authorized"] is not True:
        refusals.append("native_execution_not_authorized")
    if (Path(root) / launch["fresh_output"]["path"]).exists():
        refusals.append("fresh_output_exists")
    return {
        "format": "replica.vision_smaller_step.launch_preflight.v1",
        "status": "review_only_launch_refused",
        "launch_allowed": False,
        "native_execution": False,
        "model_loads": 0,
        "packet_sha256": packet["packet_sha256"],
        "config_sha256": packet["protocol"]["config_sha256"],
        "protocol_commit": authorization["protocol_commit"],
        "authorization_path": authorization["path"],
        "fresh_output": launch["fresh_output"],
        "refusals": refusals,
        "required_review_command": launch["preflight_command"],
        "authorized_launch_command": launch["argv"],
        "note": ("This packet is evidence, not authorization. Even a full "
                 "preflight disposition here cannot authorize a launch; the "
                 "real review-launch gate binds a committed protocol to a "
                 "separate external receipt."),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify the preparation-only smaller-step "
                    "launch/evidence packet")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="write the deterministic packet")
    build.add_argument("--config", default=DEFAULT_CONFIG)
    build.add_argument("--output", default=DEFAULT_PACKET)
    build.add_argument("--root", default=str(ROOT))
    build.add_argument("--json", action="store_true")
    verify = sub.add_parser("verify", help="refuse a stale packet")
    verify.add_argument("--config", default=DEFAULT_CONFIG)
    verify.add_argument("--packet", default=DEFAULT_PACKET)
    verify.add_argument("--root", default=str(ROOT))
    preflight = sub.add_parser(
        "preflight", help="emit the review-only launch disposition")
    preflight.add_argument("--config", default=DEFAULT_CONFIG)
    preflight.add_argument("--packet", default=DEFAULT_PACKET)
    preflight.add_argument("--root", default=str(ROOT))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            packet = build_packet(args.config, root=args.root)
            digest = write_packet(packet, Path(args.root) / args.output
                                  if not Path(args.output).is_absolute()
                                  else args.output)
            summary = {
                "status": "launch_packet_written_preparation_only",
                "packet": args.output,
                "packet_sha256": digest,
                "canonical_sha256": packet["packet_sha256"],
                "config_sha256": packet["protocol"]["config_sha256"],
                "native_execution": False,
                "model_loads": 0,
                "authorization_created": False,
                "limits": packet["limits"],
            }
            if args.json:
                print(json.dumps(summary, indent=2, sort_keys=True))
            else:
                print(f"packet_written {args.output}")
                print(f"packet_sha256={digest}")
                print("native_execution=false model_loads=0 "
                      "authorization_created=false")
            return 0
        path = Path(args.packet)
        if not path.is_absolute():
            path = Path(args.root) / path
        packet = _read_json(path)
        if args.command == "verify":
            verify_packet(packet, config_path=args.config, root=args.root)
            print(f"packet_current {path}")
            print(f"packet_sha256={packet['packet_sha256']}")
            return 0
        disposition = launch_preflight(
            packet, config_path=args.config, root=args.root)
        print(json.dumps(disposition, indent=2, sort_keys=True))
        return 0
    except PacketError as error:
        print(f"{args.command} refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
