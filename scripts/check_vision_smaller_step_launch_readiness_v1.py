#!/usr/bin/env python3
"""Read-only launch-readiness checker for the lr 3e-05 smaller-step arm.

The checker binds the declared protocol, the preparation-only launch packet,
the reviewed git commit, the preserved limits/gates/raw-generation contract and
the (currently absent) external authorization receipt.  It emits a
deterministic readiness assessment and never:

* mints, edits or infers an authorization receipt,
* imports a native runtime or loads a model,
* runs a launch command, backward call, optimizer step or generation,
* opens development, held-out, semantic-audit, V0, promotion or retry access.

``build`` writes the assessment, ``verify`` refuses a stale assessment and
``status`` prints it without writing.  A readiness assessment can show that
preparation is complete; it can never show that a launch is allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DECLARATION_PATH = "configs/vision_smaller_step_launch_readiness_v1.json"
REPORT_PATH = ("reports/vision_smaller_step_launch_readiness_v1_20261027/"
               "readiness.json")
PACKET_BUILDER_PATH = "scripts/build_vision_smaller_step_launch_packet_v1.py"
FORMAT = "replica.vision_smaller_step.launch_readiness.assessment.v1"
VERSION = "1"
DECLARATION_FORMAT = "replica.vision_smaller_step.launch_readiness.v1"

DECLARATION_KEYS = {
    "format", "version", "name", "declared_at_utc", "purpose", "subject",
    "commit_binding", "authorization", "launch", "preserved_contract",
    "remaining_requirements", "closed_roles", "decision", "verification",
}
SUBJECT_KEYS = {
    "protocol_path", "protocol_sha256", "lr", "completed_lr",
    "update_batch_example_ids", "packet_path", "packet_file_sha256",
    "packet_canonical_sha256", "packet_format", "packet_kind",
    "packet_native_execution", "packet_model_loads",
    "packet_authorization_created",
}
COMMIT_BINDING_KEYS = {
    "required", "reviewed_commit", "reviewed_paths", "config_introducing_commit",
    "config_commit_blob_sha256", "receipt_protocol_commit",
    "recorded_in_receipt", "verification_rule", "note",
}
AUTHORIZATION_KEYS = {
    "required", "path", "format", "required_status", "status_at_declaration",
    "present_at_declaration", "receipt_sha256_at_declaration",
    "protocol_commit_at_declaration", "native_execution_authorized_at_declaration",
    "minted_here", "separate_from_packet", "required_fields", "closed_fields",
    "commit_binding_required", "config_sha256_binding_required", "authority",
    "note", "inspected_existing_receipts",
}
LAUNCH_KEYS = {
    "argv", "review_command", "environment", "output",
    "output_must_not_exist", "output_exists_at_declaration",
    "authorization_checked_before_native_imports",
    "disposition_at_declaration", "refusals_at_declaration",
}
PRESERVED_KEYS = {
    "limits", "campaign_cumulative", "optimization_gate",
    "functional_fit_gate", "decision", "raw_generation",
}
RAW_GENERATION_KEYS = {
    "required", "raw_generation_required", "generation_roles", "controls",
    "readouts", "records_per_readout", "available_per_readout",
    "placeholders_per_readout", "logical_scoring_cells_per_readout",
    "logical_scoring_operations_per_readout", "expected_generated_tokens",
    "max_new_tokens_by_kind", "empty_text_policy",
    "eos_only_not_counted_correct", "whitespace_only_not_counted_correct",
    "update_batch_generation", "required_paths",
}
INSPECTED_KEYS = {"path", "protocol_name", "protocol_commit",
                  "config_sha256", "binds_subject_config"}
REQUIREMENT_KEYS = {"id", "blocking", "requirement", "owner",
                    "satisfied_by_readiness"}
EXPECTED_REQUIREMENT_IDS = [
    "authorization_receipt_issued",
    "protocol_commit_recorded",
    "review_launch_gate_passed",
    "fresh_output_available",
    "exclusive_worker_and_capacity_guard",
    "native_execution_within_declared_budget",
]
REPORT_KEYS = {
    "format", "version", "name", "deterministic", "native_execution",
    "model_loads", "authorization_created", "declaration", "subject",
    "preserved_contract", "commit_binding", "authorization", "launch",
    "checks", "remaining_requirements", "closed_roles", "decision",
    "readiness_sha256",
}


class ReadinessError(ValueError):
    """The readiness declaration is incomplete or drifted from the sources."""


class StaleReadinessError(ReadinessError):
    """The checked-in assessment no longer matches the repository."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ReadinessError(message)


def _exact_keys(value: Any, keys: set[str], label: str) -> None:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    _require(set(value) == keys,
             f"{label} fields differ: {sorted(set(value) ^ keys)}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str | Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ReadinessError(f"cannot read JSON {path}: {error}") from error
    _require(isinstance(payload, dict), f"JSON must be an object: {path}")
    return dict(payload)


def _packet_builder():
    name = "vision_smaller_step_readiness_packet_dependency"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    path = ROOT / PACKET_BUILDER_PATH
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None,
             f"cannot load packet builder: {PACKET_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git_blob_sha256(root: Path, commit: str, relative: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "cat-file", "blob",
             f"{commit}:{relative}"],
            check=False, capture_output=True)
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return _sha256_bytes(completed.stdout)


def _require_commit(value: Any, label: str) -> None:
    _require(isinstance(value, str) and len(value) == 40
             and all(character in "0123456789abcdef" for character in value),
             f"{label} must be a full 40-character lowercase commit SHA")


def _load_declaration(path: str | Path) -> dict[str, Any]:
    declaration = _read_json(path)
    _exact_keys(declaration, DECLARATION_KEYS, "readiness declaration")
    _require(declaration["format"] == DECLARATION_FORMAT,
             "readiness declaration format differs")
    _exact_keys(declaration["subject"], SUBJECT_KEYS, "subject")
    _exact_keys(declaration["commit_binding"], COMMIT_BINDING_KEYS,
                "commit_binding")
    _exact_keys(declaration["authorization"], AUTHORIZATION_KEYS,
                "authorization")
    _exact_keys(declaration["launch"], LAUNCH_KEYS, "launch")
    _exact_keys(declaration["preserved_contract"], PRESERVED_KEYS,
                "preserved_contract")
    _exact_keys(declaration["preserved_contract"]["raw_generation"],
                RAW_GENERATION_KEYS, "preserved_contract.raw_generation")
    requirements = declaration["remaining_requirements"]
    _require(isinstance(requirements, list) and requirements,
             "remaining requirements are missing")
    for requirement in requirements:
        _exact_keys(requirement, REQUIREMENT_KEYS, "remaining requirement")
        _require(requirement["blocking"] is True,
                 f"remaining requirement must block: {requirement['id']}")
    _require([item["id"] for item in requirements]
             == EXPECTED_REQUIREMENT_IDS,
             "remaining requirement ids or order differ")
    _require(declaration["decision"]["launch_allowed"] is False,
             "readiness declaration cannot allow a launch")
    _require(declaration["decision"]["native_execution"] is False,
             "readiness declaration cannot authorize native execution")
    _require(declaration["decision"]["authorization_created"] is False,
             "readiness declaration cannot create authorization")
    for role, value in declaration["closed_roles"].items():
        _require(value is False, f"readiness declaration opens role: {role}")
    for role, value in declaration["authorization"]["closed_fields"].items():
        _require(value is False,
                 f"readiness declaration opens receipt field: {role}")
    inspected = declaration["authorization"]["inspected_existing_receipts"]
    _require(isinstance(inspected, list) and inspected,
             "inspected authorization receipts are missing")
    for entry in inspected:
        _exact_keys(entry, INSPECTED_KEYS, "inspected authorization")
        _require(entry["binds_subject_config"] is False,
                 "an inspected receipt cannot bind the subject config")
        _require_commit(entry["protocol_commit"], "inspected protocol_commit")
        _require(isinstance(entry["config_sha256"], str)
                 and len(entry["config_sha256"]) == 64,
                 "inspected config SHA-256 is invalid")
    return declaration


def _authorization_state(declaration: Mapping[str, Any], protocol: Mapping[str, Any],
                         root: Path, *,
                         git_root: str | Path | None = None) -> dict[str, Any]:
    """Observe and validate any receipt without ever creating one."""
    binding_root = Path(git_root) if git_root is not None else root
    path = Path(root) / declaration["authorization"]["path"]
    if not path.is_file():
        return {
            "path": declaration["authorization"]["path"],
            "present": False,
            "status": "absent",
            "receipt_sha256": None,
            "protocol_commit": None,
            "commit_bound": False,
            "config_sha256_bound": False,
            "native_execution_authorized": False,
            "closed_roles_respected": None,
            "violations": ["authorization_receipt_missing"],
            "note": "The readiness checker never mints or infers a receipt.",
        }
    payload = path.read_bytes()
    violations: list[str] = []
    receipt: dict[str, Any] = {}
    try:
        receipt = _read_json(path)
    except ReadinessError as error:
        violations.append(f"unreadable_receipt:{error}")
    if receipt:
        required = set(declaration["authorization"]["required_fields"])
        if set(receipt) != required:
            violations.append("receipt_fields_differ")
        if receipt.get("format") != declaration["authorization"]["format"]:
            violations.append("receipt_format_differs")
        if receipt.get("status") != declaration["authorization"]["required_status"]:
            violations.append("receipt_status_not_authorized")
        if receipt.get("protocol_name") != protocol["name"]:
            violations.append("receipt_protocol_name_differs")
        if receipt.get("protocol_path") != declaration["subject"]["protocol_path"]:
            violations.append("receipt_protocol_path_differs")
        if receipt.get("config_sha256") != declaration["subject"]["protocol_sha256"]:
            violations.append("receipt_config_sha256_differs")
        if receipt.get("native_execution_authorized") is not True:
            violations.append("receipt_does_not_authorize_native_execution")
        for field, expected in declaration["authorization"]["closed_fields"].items():
            if receipt.get(field) is not expected:
                violations.append(f"receipt_opens_closed_field:{field}")
        if not isinstance(receipt.get("issued_at_utc"), str) or \
                not receipt["issued_at_utc"].strip():
            violations.append("receipt_issued_at_missing")
        try:
            _require_commit(receipt.get("protocol_commit"),
                            "authorization protocol_commit")
        except ReadinessError as error:
            violations.append(str(error))
    commit = receipt.get("protocol_commit")
    bound = False
    if isinstance(commit, str) and len(commit) == 40:
        blob = _git_blob_sha256(binding_root, commit,
                                declaration["subject"]["protocol_path"])
        bound = blob == declaration["subject"]["protocol_sha256"]
        if not bound:
            violations.append("receipt_commit_does_not_contain_protocol_bytes")
    status = "present_valid" if not violations else "present_invalid"
    return {
        "path": declaration["authorization"]["path"],
        "present": True,
        "status": status,
        "receipt_sha256": _sha256_bytes(payload),
        "protocol_commit": commit if isinstance(commit, str) else None,
        "commit_bound": bound,
        "config_sha256_bound":
            receipt.get("config_sha256")
            == declaration["subject"]["protocol_sha256"],
        "native_execution_authorized":
            receipt.get("native_execution_authorized") is True,
        "closed_roles_respected": not any(
            item.startswith("receipt_opens_closed_field")
            for item in violations),
        "violations": violations,
        "note": ("A valid receipt is still not a launch decision: only "
                 "review-launch may pass the launch gate."),
    }


def _scan_authorizations(root: Path, subject: Mapping[str, Any]) -> list[dict[str, Any]]:
    """List every receipt-shaped authorization file and whether it binds us."""
    records: list[dict[str, Any]] = []
    reports = root / "reports"
    if not reports.is_dir():
        return records
    for path in sorted(reports.glob("**/*authorization*.json")):
        try:
            payload = _read_json(path)
        except ReadinessError:
            continue
        if "protocol_name" not in payload and "config_sha256" not in payload:
            continue
        binds = (payload.get("config_sha256") == subject["protocol_sha256"]
                 or payload.get("protocol_path") == subject["protocol_path"])
        records.append({
            "path": path.relative_to(root).as_posix(),
            "protocol_name": payload.get("protocol_name"),
            "status": payload.get("status"),
            "protocol_commit": payload.get("protocol_commit"),
            "config_sha256": payload.get("config_sha256"),
            "binds_subject_config": binds,
        })
    return records


def _verify_inspected(declaration: Mapping[str, Any],
                      subject: Mapping[str, Any],
                      root: Path) -> list[dict[str, Any]]:
    """Prove no existing receipt binds the smaller-step protocol."""
    declared = {entry["path"]: entry
                for entry in declaration["authorization"]
                ["inspected_existing_receipts"]}
    observed = _scan_authorizations(root, subject)
    observed_paths = {entry["path"] for entry in observed}
    _require(observed_paths == set(declared),
             "inspected authorization set differs from the repository: "
             f"{sorted(observed_paths ^ set(declared))}")
    _require(declaration["authorization"]["path"] not in observed_paths,
             "a receipt exists at the smaller-step authorization path")
    for entry in observed:
        _require(entry["binds_subject_config"] is False,
                 f"an existing receipt binds the smaller-step config: "
                 f"{entry['path']}")
        _require(entry["config_sha256"] != subject["protocol_sha256"],
                 f"existing receipt config matches the subject: {entry['path']}")
        expected = declared[entry["path"]]
        for key in ("protocol_name", "protocol_commit", "config_sha256"):
            _require(entry[key] == expected[key],
                     f"inspected receipt drifted: {entry['path']}: {key}")
        _require(entry["binds_subject_config"]
                 is expected["binds_subject_config"],
                 f"inspected binding state differs: {entry['path']}")
    return observed


def _packet_raw_paths(raw: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for entry in raw["per_readout"]:
        for key in ("readout", "records", "answer_generations",
                    "caption_generations"):
            paths.append(entry[key])
    paths.extend(raw["update_batch_records"])
    return paths


def _launch_state(declaration: Mapping[str, Any],
                  authorization: Mapping[str, Any],
                  root: Path) -> dict[str, Any]:
    """Observe the launch prerequisites; never permit or perform a launch."""
    output = declaration["launch"]["output"]
    output_exists = (root / output).exists()
    refusals = [
        "packet_is_not_an_authorization_receipt",
        "packet_is_preparation_only",
    ]
    if not authorization["present"]:
        refusals.append("authorization_receipt_missing")
    if authorization["protocol_commit"] is None:
        refusals.append("protocol_commit_unbound")
    if authorization["commit_bound"] is not True:
        refusals.append("commit_binding_unverified")
    if authorization["native_execution_authorized"] is not True:
        refusals.append("native_execution_not_authorized")
    if output_exists:
        refusals.append("fresh_output_exists")
    return {
        "argv": declaration["launch"]["argv"],
        "review_command": declaration["launch"]["review_command"],
        "environment": declaration["launch"]["environment"],
        "output": output,
        "output_exists": output_exists,
        "authorization_checked_before_native_imports": True,
        "disposition": ("review_only_launch_refused" if refusals
                        else "no_launch_performed_here"),
        "refusals": refusals,
    }


def assess(declaration_path: str | Path = DECLARATION_PATH, *,
           root: str | Path = ROOT) -> dict[str, Any]:
    """Recompute the readiness assessment from declared and live state."""
    root_path = Path(root).resolve()
    declaration_file = Path(declaration_path)
    if not declaration_file.is_absolute():
        declaration_file = root_path / declaration_file
    declaration = _load_declaration(declaration_file)
    subject = declaration["subject"]

    protocol_file = root_path / subject["protocol_path"]
    _require(protocol_file.is_file(),
             f"protocol is missing: {subject['protocol_path']}")
    protocol_sha = _sha256_file(protocol_file)
    _require(protocol_sha == subject["protocol_sha256"],
             "declared protocol SHA-256 differs from the current config")
    protocol = _read_json(protocol_file)

    builder = _packet_builder()
    packet_file = root_path / subject["packet_path"]
    _require(packet_file.is_file(),
             f"packet is missing: {subject['packet_path']}")
    packet_bytes = packet_file.read_bytes()
    packet_sha = _sha256_bytes(packet_bytes)
    _require(packet_sha == subject["packet_file_sha256"],
             "declared packet file SHA-256 differs")
    packet = _read_json(packet_file)
    _require(packet["packet_sha256"] == subject["packet_canonical_sha256"],
             "declared packet canonical SHA-256 differs")
    builder.verify_packet(packet, config_path=subject["protocol_path"],
                          root=root_path)
    _require(packet["format"] == subject["packet_format"],
             "packet format differs")
    _require(packet["kind"] == subject["packet_kind"],
             "packet kind differs")
    _require(packet["native_execution"] is subject["packet_native_execution"],
             "packet native_execution differs")
    _require(packet["model_loads"] == subject["packet_model_loads"],
             "packet model_loads differs")
    _require(packet["authorization_created"]
             is subject["packet_authorization_created"],
             "packet authorization_created differs")

    protocol_block = packet["protocol"]
    _require(protocol_block["config_path"] == subject["protocol_path"],
             "packet binds a different protocol path")
    _require(protocol_block["config_sha256"] == protocol_sha,
             "packet binds a different protocol SHA-256")
    _require(protocol_block["lr"] == subject["lr"],
             "packet binds a different lr")
    _require(protocol_block["completed_lr"] == subject["completed_lr"],
             "packet binds a different completed lr")
    _require(list(packet["scope"]["preserved"]["update_batch_example_ids"])
             == subject["update_batch_example_ids"],
             "packet binds a different update batch")

    preserved = declaration["preserved_contract"]
    _require(packet["limits"] == preserved["limits"],
             "preserved limits differ from the packet")
    campaign = protocol["accounting"]["campaign_cumulative"]
    for key, expected in preserved["campaign_cumulative"].items():
        _require(campaign.get(key) == expected,
                 f"campaign accounting differs: {key}")
    packet_gates = packet["gates"]
    for key in ("optimization", "functional_fit", "decision"):
        declared_gate = preserved[
            "optimization_gate" if key == "optimization"
            else "functional_fit_gate" if key == "functional_fit"
            else "decision"]
        for gate_key, expected in declared_gate.items():
            _require(packet_gates[key].get(gate_key) == expected,
                     f"preserved {key} gate differs: {gate_key}")
    _require(protocol["gates"]["optimization"]
             ["max_update_batch_mean_nll_delta"]
             == preserved["optimization_gate"]["max_update_batch_mean_nll_delta"],
             "protocol optimization gate differs")
    _require(protocol["gates"]["functional_fit"]["answer_correct_min"]
             == preserved["functional_fit_gate"]["answer_correct_min"],
             "protocol functional gate differs")
    _require(protocol["limits"]["max_decoder_forwards"]
             == preserved["limits"]["decoder_forward_calls"],
             "protocol decoder forward limit differs")
    _require(protocol["limits"]["max_visual_encoder_calls"]
             == preserved["limits"]["visual_observation_calls"],
             "protocol visual observation limit differs")
    _require(protocol["limits"]["max_total_backward_calls"]
             == preserved["limits"]["backward_calls"],
             "protocol backward limit differs")
    _require(protocol["limits"]["max_optimizer_steps"]
             == preserved["limits"]["optimizer_steps"],
             "protocol optimizer step limit differs")
    _require(protocol["limits"]["max_seconds"]
             == preserved["limits"]["wall_seconds"],
             "protocol wall limit differs")
    _require(protocol["limits"]["artifact_bytes"]
             == preserved["limits"]["artifact_bytes"],
             "protocol artifact byte limit differs")
    _require(protocol["accounting"]["artifact_bytes"]
             == preserved["limits"]["artifact_bytes"],
             "protocol accounting artifact bytes differ")

    raw = packet_gates["raw_generation_evidence"]
    declared_raw = preserved["raw_generation"]
    _require(raw["required"] is True
             and raw["raw_generation_required"] is True,
             "packet does not require raw generation evidence")
    for key in ("generation_roles", "readouts"):
        _require(list(raw[key]) == list(declared_raw[key]),
                 f"raw generation {key} differ")
    _require(sorted(raw["controls_required"])
             == sorted(declared_raw["controls"]),
             "raw generation controls differ")
    for key in ("expected_generated_tokens", "max_new_tokens_by_kind",
                "empty_text_policy", "eos_only_not_counted_correct",
                "whitespace_only_not_counted_correct",
                "update_batch_generation"):
        _require(raw[key] == declared_raw[key],
                 f"raw generation requirement differs: {key}")
    _require(sorted(_packet_raw_paths(raw))
             == sorted(declared_raw["required_paths"]),
             "raw generation evidence paths differ")
    _require([entry["label"] for entry in raw["per_readout"]]
             == list(declared_raw["readouts"]),
             "raw generation per-readout labels differ")
    for entry in raw["per_readout"]:
        _require(sorted(entry["controls"])
                 == sorted(declared_raw["controls"]),
                 "raw generation controls differ per readout")
        _require(entry["records_expected"]
                 == declared_raw["records_per_readout"],
                 "raw generation record count differs")
        _require(entry["available"] == declared_raw["available_per_readout"],
                 "raw generation available count differs")
        _require(entry["placeholders"]
                 == declared_raw["placeholders_per_readout"],
                 "raw generation placeholder count differs")
        _require(entry["logical_scoring_cells"]
                 == declared_raw["logical_scoring_cells_per_readout"],
                 "raw generation scoring cells differ")
        _require(entry["logical_scoring_operations"]
                 == declared_raw["logical_scoring_operations_per_readout"],
                 "raw generation scoring operations differ")

    binding = declaration["commit_binding"]
    _require_commit(binding["reviewed_commit"], "reviewed_commit")
    _require_commit(binding["config_introducing_commit"],
                    "config_introducing_commit")
    reviewed: list[dict[str, Any]] = []
    for relative in binding["reviewed_paths"]:
        working = root_path / relative
        _require(working.is_file(), f"reviewed path is missing: {relative}")
        working_sha = _sha256_file(working)
        blob_sha = _git_blob_sha256(root_path, binding["reviewed_commit"],
                                    relative)
        _require(blob_sha == working_sha,
                 f"reviewed commit does not contain current bytes: {relative}")
        reviewed.append({"path": relative, "sha256": working_sha,
                         "commit_blob_sha256": blob_sha,
                         "matches": True})
    config_blob = _git_blob_sha256(root_path,
                                   binding["config_introducing_commit"],
                                   subject["protocol_path"])
    _require(config_blob == binding["config_commit_blob_sha256"],
             "config-introducing commit blob differs")
    _require(config_blob == protocol_sha,
             "config-introducing commit does not contain the protocol bytes")
    _require(binding["receipt_protocol_commit"] is None
             and binding["recorded_in_receipt"] is False,
             "readiness declaration cannot claim a recorded protocol commit")

    authorization = _authorization_state(declaration, protocol, root_path)
    inspected = _verify_inspected(
        declaration,
        {"protocol_sha256": protocol_sha,
         "protocol_path": subject["protocol_path"]},
        root_path)
    authorization = {**authorization,
                     "inspected_existing_receipts": inspected}
    _require(not (authorization["present"]
                  and authorization["status"] == "present_valid"
                  and authorization["native_execution_authorized"] is False),
             "valid receipt must authorize native execution")

    output = declaration["launch"]["output"]
    _require(packet["launch"]["fresh_output"]["path"] == output,
             "declared launch output differs from the packet")
    _require(packet["launch"]["argv"] == declaration["launch"]["argv"],
             "declared launch argv differs from the packet")
    _require(packet["launch"]["preflight_command"]
             == declaration["launch"]["review_command"],
             "declared review command differs from the packet")
    _require(packet["launch"]["environment"]
             == declaration["launch"]["environment"],
             "declared launch environment differs from the packet")
    _require(declaration["launch"]["output_must_not_exist"] is True,
             "launch must require a fresh output")
    _require(declaration["launch"]["authorization_checked_before_native_imports"]
             is True,
             "launch must check authorization before native imports")
    launch_state = _launch_state(declaration, authorization, root_path)
    output_exists = launch_state["output_exists"]

    requirements: list[dict[str, Any]] = []
    for requirement in declaration["remaining_requirements"]:
        identifier = requirement["id"]
        if identifier == "authorization_receipt_issued":
            satisfied = authorization["status"] == "present_valid"
        elif identifier == "protocol_commit_recorded":
            satisfied = (authorization["protocol_commit"] is not None
                         and authorization["commit_bound"] is True)
        elif identifier == "fresh_output_available":
            satisfied = not output_exists
        else:
            satisfied = False
        requirements.append({**requirement, "satisfied": satisfied})

    if authorization["status"] == "present_valid":
        reason = ("receipt_present_but_readiness_is_not_a_launch_authority_"
                  "use_review_launch_before_native_imports")
    elif authorization["present"]:
        reason = "authorization_receipt_present_but_invalid"
    else:
        reason = ("preparation_is_complete_but_the_external_authorization_"
                  "receipt_is_absent_and_the_protocol_commit_is_unrecorded")

    checks = [
        {"id": "declaration_schema", "status": "passed",
         "requirement": "readiness declaration matches its declared schema"},
        {"id": "protocol_config_sha256", "status": "passed",
         "requirement": "the exact lr 3e-05 protocol bytes are bound"},
        {"id": "packet_current", "status": "passed",
         "requirement": "the launch packet recomputes from the repository "
                        "with no drift"},
        {"id": "packet_subject_binding", "status": "passed",
         "requirement": "packet protocol, lr, completed lr and update batch "
                        "match the declaration"},
        {"id": "preserved_limits", "status": "passed",
         "requirement": "4824 decoder forwards, 168 visual observations, "
                        "4 backward calls, 1 optimizer step, 1200 seconds and "
                        "33554432 artifact bytes including the report"},
        {"id": "preserved_campaign_accounting", "status": "passed",
         "requirement": "campaign cumulative accounting matches the protocol"},
        {"id": "preserved_optimization_gate", "status": "passed",
         "requirement": "the 0.25 nat optimization gate and projector-only "
                        "scope are unchanged"},
        {"id": "preserved_functional_fit_gate", "status": "passed",
         "requirement": "generated-answer, caption, control-margin, EOS-only "
                        "and whitespace-only gates are unchanged"},
        {"id": "preserved_decision_closure", "status": "passed",
         "requirement": "development, held-out, V0, promotion and retry remain "
                        "closed"},
        {"id": "preserved_raw_generation", "status": "passed",
         "requirement": "all raw generation and control evidence paths, "
                        "counts and empty-text rules are preserved"},
        {"id": "reviewed_commit_blobs", "status": "passed",
         "requirement": f"reviewed commit {binding['reviewed_commit']} "
                        "contains the exact current protocol, runner, builder, "
                        "both contract tests and packet bytes"},
        {"id": "config_commit_blob", "status": "passed",
         "requirement": "the config-introducing commit contains the exact "
                        "protocol bytes"},
        {"id": "authorization_state", "status": "observed",
         "requirement": f"receipt state is {authorization['status']}; a "
                        "receipt is never minted or inferred"},
        {"id": "existing_authorizations_inspected", "status": "passed",
         "requirement": f"{len(inspected)} existing receipt(s) were inspected; "
                        "all bind vision_learnability_step1_reproduction_v1 "
                        "and a different config, and none binds the "
                        "smaller-step protocol"},
        {"id": "fresh_output", "status": "observed",
         "requirement": f"output {output} exists={output_exists}"},
        {"id": "no_native_execution", "status": "passed",
         "requirement": "no model load, backward call, optimizer step, "
                        "generation or launch command is performed"},
    ]

    report: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "name": declaration["name"],
        "deterministic": True,
        "native_execution": False,
        "model_loads": 0,
        "authorization_created": False,
        "declaration": {
            "path": Path(declaration_path).as_posix(),
            "sha256": _sha256_file(declaration_file),
        },
        "subject": {
            "protocol_path": subject["protocol_path"],
            "protocol_sha256": protocol_sha,
            "packet_path": subject["packet_path"],
            "packet_file_sha256": packet_sha,
            "packet_canonical_sha256": packet["packet_sha256"],
            "packet_current": True,
            "lr": subject["lr"],
            "completed_lr": subject["completed_lr"],
            "update_batch_example_ids": subject["update_batch_example_ids"],
        },
        "preserved_contract": {
            "limits": packet["limits"],
            "campaign_cumulative": preserved["campaign_cumulative"],
            "optimization_gate": preserved["optimization_gate"],
            "functional_fit_gate": preserved["functional_fit_gate"],
            "decision": preserved["decision"],
            "raw_generation": {
                "required_paths": declared_raw["required_paths"],
                "controls": list(raw["controls_required"]),
                "generation_roles": list(raw["generation_roles"]),
                "readouts": list(raw["readouts"]),
                "expected_generated_tokens":
                    raw["expected_generated_tokens"],
                "empty_text_policy": raw["empty_text_policy"],
                "eos_only_not_counted_correct":
                    raw["eos_only_not_counted_correct"],
                "whitespace_only_not_counted_correct":
                    raw["whitespace_only_not_counted_correct"],
            },
        },
        "commit_binding": {
            "reviewed_commit": binding["reviewed_commit"],
            "reviewed_paths": reviewed,
            "config_introducing_commit": binding["config_introducing_commit"],
            "config_blob_sha256": config_blob,
            "receipt_protocol_commit": authorization["protocol_commit"],
            "commit_bound": authorization["commit_bound"],
            "recorded_in_receipt": authorization["commit_bound"],
        },
        "authorization": authorization,
        "launch": launch_state,
        "checks": checks,
        "remaining_requirements": requirements,
        "closed_roles": declaration["closed_roles"],
        "decision": {
            "preparation_complete": True,
            "launch_allowed": False,
            "native_execution": False,
            "model_loads": 0,
            "authorization_created": False,
            "reason": reason,
        },
    }
    report["readiness_sha256"] = _readiness_sha256(report)
    _exact_keys(report, REPORT_KEYS, "readiness assessment")
    return report


def _readiness_sha256(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items()
               if key != "readiness_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True).encode("utf-8")
    return _sha256_bytes(canonical)


def report_bytes(report: Mapping[str, Any]) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True,
                       ensure_ascii=True) + "\n").encode("utf-8")


def write_report(report: Mapping[str, Any], path: str | Path = REPORT_PATH,
                 *, root: str | Path = ROOT) -> str:
    target = Path(path)
    if not target.is_absolute():
        target = Path(root) / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(report_bytes(report))
    return _sha256_bytes(report_bytes(report))


def verify_report(report: Mapping[str, Any] | None = None, *,
                  declaration_path: str | Path = DECLARATION_PATH,
                  report_path: str | Path = REPORT_PATH,
                  root: str | Path = ROOT) -> dict[str, Any]:
    """Refuse a missing, malformed or stale readiness assessment."""
    root_path = Path(root)
    if report is None:
        path = Path(report_path)
        if not path.is_absolute():
            path = root_path / path
        _require(path.is_file(), f"readiness assessment is missing: {path}")
        report = _read_json(path)
    _exact_keys(report, REPORT_KEYS, "readiness assessment")
    _require(report["format"] == FORMAT, "readiness assessment format differs")
    _require(report["readiness_sha256"] == _readiness_sha256(report),
             "readiness assessment hash differs from its payload")
    expected = assess(declaration_path, root=root_path)
    for key in sorted(REPORT_KEYS - {"readiness_sha256"}):
        if report[key] != expected[key]:
            raise StaleReadinessError(
                f"stale readiness section: {key}")
    return dict(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only launch readiness for the smaller-step protocol")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="write the readiness assessment")
    build.add_argument("--declaration", default=DECLARATION_PATH)
    build.add_argument("--output", default=REPORT_PATH)
    build.add_argument("--root", default=str(ROOT))
    build.add_argument("--json", action="store_true")
    verify = sub.add_parser("verify", help="refuse a stale assessment")
    verify.add_argument("--declaration", default=DECLARATION_PATH)
    verify.add_argument("--report", default=REPORT_PATH)
    verify.add_argument("--root", default=str(ROOT))
    status = sub.add_parser("status", help="print without writing")
    status.add_argument("--declaration", default=DECLARATION_PATH)
    status.add_argument("--root", default=str(ROOT))
    return parser


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    unsatisfied = [item["id"] for item in report["remaining_requirements"]
                   if not item["satisfied"]]
    return {
        "status": "readiness_current",
        "readiness_sha256": report["readiness_sha256"],
        "config_sha256": report["subject"]["protocol_sha256"],
        "packet_sha256": report["subject"]["packet_canonical_sha256"],
        "authorization_status": report["authorization"]["status"],
        "protocol_commit": report["authorization"]["protocol_commit"],
        "output_exists": report["launch"]["output_exists"],
        "refusals": report["launch"]["refusals"],
        "unsatisfied_requirements": unsatisfied,
        "decision": report["decision"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        root = Path(args.root).resolve()
        if args.command == "build":
            report = assess(args.declaration, root=root)
            digest = write_report(report, args.output, root=root)
            if args.json:
                print(json.dumps({**_summary(report),
                                  "report_sha256": digest},
                                 indent=2, sort_keys=True))
            else:
                print(f"readiness_written {args.output}")
                print(f"readiness_sha256={report['readiness_sha256']}")
                print(f"report_sha256={digest}")
                print("native_execution=false model_loads=0 "
                      "authorization_created=false launch_allowed=false")
            return 0
        if args.command == "verify":
            report = verify_report(declaration_path=args.declaration,
                                  report_path=args.report, root=root)
            print("readiness_current")
            print(f"readiness_sha256={report['readiness_sha256']}")
            print(f"authorization_status={report['authorization']['status']}")
            print("launch_allowed=false")
            return 0
        report = assess(args.declaration, root=root)
        print(json.dumps(_summary(report), indent=2, sort_keys=True))
        return 0
    except ReadinessError as error:
        print(f"{args.command} refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
