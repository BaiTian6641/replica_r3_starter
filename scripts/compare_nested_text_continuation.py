#!/usr/bin/env python3
"""Exact-only comparison of uninterrupted and resumed nested-text learning.

This is a pure CPU receipt tool.  It never loads a model, builds a learner or
runs a second in-process reference.  The single exact verdict comes from the
declared artifacts: two independent uninterrupted 28-event arms (A/A) and one
resume that continues the preserved 27-event parent checkpoint (A/B).

Any missing artifact, hash drift, guard stop, event-sequence defect or canonical
state difference fails the comparison.  A receipt is still written so the
failure is preserved; the exit code is nonzero unless every gate passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.nested_continuation import FORMAT, compare_states  # noqa: E402
from replica_lab import source_provenance as provenance  # noqa: E402

RECEIPT_FORMAT = "replica.nested_text_continuation.comparison.v1"
SOURCE_MODES = provenance.MODES
CURRENT_SOURCE_RECEIPT_FORMAT = (
    "replica.nested_text.continuation_current_receipt.v1")
GUARD_BINDING_FORMAT = "replica.nested_text.guard_binding.v1"
HISTORICAL_SOURCE_SNAPSHOT = (
    "reports/nested_text_continuation_comparison_20261026/"
    "source_snapshot.json")
CURRENT_SOURCE_RECEIPT = (
    "configs/nested_text_continuation_current_receipt_v1.json")
GUARD_BINDING = "configs/nested_text_continuation_guard_binding_v1.json"
RUN_FORMAT = "replica.nested_text.run.v1"
CONFIG_FORMAT = "replica.nested_text.nl1.v1"
P_ONLY_FORMAT = "replica.nested_learning.text_p_only.v1"
CONFIG_SCHEMA_VERSION = 1

EXPECTED_EVENT_IDS = tuple(
    [f"teach-{index:04d}" for index in range(24)]
    + [f"correction-{index:04d}" for index in range(4)])
PARENT_EVENTS = 27
PARENT_STATE_INDEX = 27
CONTINUOUS_EVENTS = 28
RESUME_EVENT_INDEX = CONTINUOUS_EVENTS
RESUME_EVENT_ID = EXPECTED_EVENT_IDS[RESUME_EVENT_INDEX - 1]
FORBIDDEN_P_ONLY_KEYS = (
    "optimizer", "moments", "dedup", "clock", "revision", "coordinator",
    "observations", "sources", "kv")
REFERENCE_KEYS = (
    "reference_after_state", "reference_outcome", "reference_state",
    "expected_state", "reference_before_state")
SECTIONS = (
    "controller", "controller.bank", "controller.bank.optimizer",
    "controller.clock", "controller.dedup", "controller.observation",
    "controller.revision", "controller.personality", "text", "text.protocol",
    "text.route", "text.feature_producer", "text.read_path",
    "text.personality", "text.event_fingerprints",
)
STAGE_PRIORITY = {
    "event_sequence": 0, "canonical_validation": 1, "outcome": 2,
    "aa_canonical": 3, "aa_outcome": 4, "state_outcome_binding": 5,
    "boundary": 6, "final": 7,
}

HEX64 = re.compile(r"^[0-9a-f]{64}$")
PARAMETER_LEAF = re.compile(r"^controller\.bank\.parameters\[(\d+)\]\[1\]$")
OPTIMIZER_LEAF = re.compile(
    r"^controller\.bank\.optimizer\.state\.(\d+)\.exp_avg$")
OPTIMIZER_SQ_LEAF = re.compile(
    r"^controller\.bank\.optimizer\.state\.(\d+)\.exp_avg_sq$")
OPTIMIZER_STEP_LEAF = re.compile(
    r"^controller\.bank\.optimizer\.state\.(\d+)\.step$")
OBSERVATION_INDEX = re.compile(
    r"^controller\.observation\.observations\[(\d+)\]\.event_id$")
CHUNK_INDEX = re.compile(
    r"^controller\.observation\.chunks\[(\d+)\]\.fingerprint$")
DEDUP_INDEX = re.compile(r"^controller\.dedup\.records\[(\d+)\]\.event_id$")
EVENT_FINGERPRINT_PREFIX = "text.event_fingerprints."


class ComparisonError(RuntimeError):
    """An artifact cannot be compared at all."""


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON numeric value {value!r}")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_json(value: Any) -> str:
    return _sha_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8"))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"),
                      parse_constant=_reject_constant)


def _load_jsonl(path: Path) -> list[Any]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line, parse_constant=_reject_constant))
        except ValueError as error:
            raise ComparisonError(f"{path}:{number}: {error}") from error
    return rows


def _path(value: str | Path, root: str | Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (Path(root) / path)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _is_hex(value: Any) -> bool:
    return isinstance(value, str) and bool(HEX64.match(value))


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    raise TypeError(f"unsupported scalar {type(value)!r}")


def _leaf_value(leaves: Mapping[str, Any], path: str) -> Any:
    leaf = leaves.get(path)
    if not isinstance(leaf, Mapping) or "value" not in leaf:
        return _MISSING
    return leaf["value"]


_MISSING = object()


def _short(value: Any, limit: int = 160) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _first_difference(left: Any, right: Any, prefix: str = "") -> dict | None:
    """Return the first exact difference in two JSON-compatible values."""
    if left == right:
        return None
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right), key=str):
            if key not in left or key not in right:
                path = f"{prefix}.{key}" if prefix else str(key)
                return {"path": path,
                        "left": _short(left.get(key, "<missing>")),
                        "right": _short(right.get(key, "<missing>"))}
            found = _first_difference(left[key], right[key],
                                      f"{prefix}.{key}" if prefix else str(key))
            if found is not None:
                return found
    if (isinstance(left, list) and isinstance(right, list)):
        for index in range(max(len(left), len(right))):
            path = f"{prefix}[{index}]"
            if index >= len(left) or index >= len(right):
                return {"path": path,
                        "left": _short(left[index] if index < len(left) else "<missing>"),
                        "right": _short(right[index] if index < len(right) else "<missing>")}
            found = _first_difference(left[index], right[index], path)
            if found is not None:
                return found
    path = prefix or "<root>"
    return {"path": path, "left": _short(left), "right": _short(right)}


def _section_hashes(leaves: Mapping[str, Any]) -> dict[str, str]:
    result = {}
    for section in SECTIONS:
        selected = {path: value for path, value in leaves.items()
                    if path == section or path.startswith(section + ".")
                    or path.startswith(section + "[")}
        if selected:
            result[section] = _sha_json(selected)
    return result


def _index_set(leaves: Mapping[str, Any], pattern: re.Pattern) -> set[int]:
    found = set()
    for path in leaves:
        match = pattern.match(path)
        if match is not None:
            found.add(int(match.group(1)))
    return found


class Report:
    """Accumulates fail-closed findings and exact first divergence."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.resource_failures = 0
        self.resource_invalid = False
        self.aa_compared = 0
        self.aa_mismatch = False
        self._earliest: tuple[Any, int, str, dict] | None = None

    def fail(self, gate: str, detail: str, *, event_index: int | None = None,
             event_id: str | None = None, path: str | None = None,
             stage: str | None = None, resource: bool = False,
             invalid: bool = False) -> bool:
        prefix = gate
        if event_index is not None:
            prefix += f" event_index={event_index}"
        if event_id is not None:
            prefix += f" event={event_id}"
        self.failures.append(f"{prefix}: {detail}")
        if resource:
            self.resource_failures += 1
            self.resource_invalid = self.resource_invalid or invalid
        if stage is not None and event_index is not None:
            entry = {"stage": stage, "event_index": event_index,
                     "event_id": event_id, "path": path, "detail": detail}
            key = (event_index, STAGE_PRIORITY.get(stage, 99),
                   path or "", entry)
            if self._earliest is None or key[:3] < self._earliest[:3]:
                self._earliest = key
        return False

    def note(self, stage: str, event_index: int, event_id: str | None,
             path: str | None, detail: str) -> None:
        """Record a divergence location without adding a duplicate failure."""
        entry = {"stage": stage, "event_index": event_index,
                 "event_id": event_id, "path": path, "detail": detail}
        key = (event_index, STAGE_PRIORITY.get(stage, 99), path or "", entry)
        if self._earliest is None or key[:3] < self._earliest[:3]:
            self._earliest = key

    def record_aa(self, equal: bool) -> None:
        self.aa_compared += 1
        if not equal:
            self.aa_mismatch = True

    @property
    def earliest(self) -> dict | None:
        return None if self._earliest is None else self._earliest[3]

    @property
    def exact_status(self) -> str:
        # Exact certification is all-or-nothing: a missing receipt, drift or
        # guard stop prevents an exact claim just as a state difference does.
        return "passed" if not self.failures else "failed"

    @property
    def resource_status(self) -> str:
        if self.resource_invalid:
            return "invalid"
        if self.resource_failures:
            return "exceeded"
        return "within_declared_limits"

    @property
    def variation_status(self) -> str:
        if self.aa_compared == 0:
            return "not_measured"
        return "variation_detected" if self.aa_mismatch else "none"

    @property
    def status(self) -> str:
        if self.failures:
            return "failed"
        return "exact_match"


def _require_mapping(report: Report, value: Any, gate: str,
                     detail: str) -> Mapping | None:
    if not isinstance(value, Mapping):
        report.fail(gate, detail)
        return None
    return value


def _protocol_context(protocol_path: Path, report: Report,
                      artifact_root: Path = ROOT) -> dict | None:
    try:
        config = _load_json(protocol_path)
    except (OSError, ValueError) as error:
        report.fail("protocol", f"cannot read protocol config: {error}")
        return None
    if not isinstance(config, Mapping):
        report.fail("protocol", "protocol config is not a JSON object")
        return None
    if config.get("format") != CONFIG_FORMAT:
        report.fail("protocol", f"protocol format must be {CONFIG_FORMAT!r}")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        report.fail("protocol", "protocol schema_version must be 1")
    for name, expected in (
            ("native_execution_admitted", True), ("training_admitted", True),
            ("promotion", False), ("one_clock_control_only", True),
            ("adaptive_router", False), ("rsi_enabled", False),
            ("replay_bytes", 0)):
        if config.get(name) != expected:
            report.fail("protocol", f"protocol {name} must be {expected!r}")
    modes = config.get("allowed_native_modes")
    if not (isinstance(modes, list) and "teach" in modes and "resume" in modes):
        report.fail("protocol",
                    "allowed_native_modes must admit teach and resume")
    comparison = _require_mapping(
        report, config.get("comparison"), "protocol",
        "protocol comparison block is missing")
    runtime = _require_mapping(
        report, config.get("runtime"), "protocol",
        "protocol runtime block is missing")
    storage = _require_mapping(
        report, config.get("storage_policy"), "protocol",
        "protocol storage_policy block is missing")
    bank = _require_mapping(report, config.get("bank"), "protocol",
                            "protocol bank block is missing")
    features = _require_mapping(report, config.get("features"), "protocol",
                                "protocol features block is missing")
    routing = _require_mapping(report, config.get("routing"), "protocol",
                               "protocol routing block is missing")
    read_path = _require_mapping(report, config.get("read_path"), "protocol",
                                 "protocol read_path block is missing")
    transaction = _require_mapping(
        report, config.get("transaction"), "protocol",
        "protocol transaction block is missing")
    episodes = _require_mapping(report, config.get("episodes"), "protocol",
                                "protocol episodes block is missing")
    if (comparison is None or runtime is None or storage is None
            or bank is None or features is None or routing is None
            or read_path is None or transaction is None or episodes is None):
        return None

    implementation_paths = config.get("implementation_paths")
    if not (isinstance(implementation_paths, list)
            and implementation_paths
            and all(isinstance(item, str) and item
                    for item in implementation_paths)):
        report.fail("protocol", "implementation_paths must be a nonempty string list")
        implementation_paths = []

    declared = comparison.get("protocol_sha256")
    if not _is_hex(declared):
        report.fail("protocol", "comparison.protocol_sha256 must be a SHA-256")
    if comparison.get("canonical_state_format") != FORMAT:
        report.fail("protocol", "canonical_state_format differs from the module")
    for name, expected in (
            ("parent_event_count", PARENT_EVENTS),
            ("continuous_events", CONTINUOUS_EVENTS),
            ("continuous_arms", 2), ("resume_arms", 1)):
        if comparison.get(name) != expected:
            report.fail("protocol", f"comparison.{name} must be {expected}")
    if comparison.get("boundary_event") != RESUME_EVENT_ID:
        report.fail("protocol", "comparison.boundary_event must be correction-0003")
    for name in ("require_exact_aa", "require_exact_ab",
                 "require_gradient_norms", "require_guard_receipts",
                 "require_full_dedup_and_coordinator"):
        if comparison.get(name) is not True:
            report.fail("protocol", f"comparison.{name} must be true")
    if comparison.get("allow_in_process_reference_verdict") is not False:
        report.fail("protocol", "in-process reference verdicts must be disallowed")
    if comparison.get("allow_numeric_envelope") is not False:
        report.fail("protocol", "numeric envelopes must be disallowed")
    if comparison.get("stop_on_any_mismatch") is not True:
        report.fail("protocol", "stop_on_any_mismatch must be true")
    if comparison.get("heldout_opened") is not False:
        report.fail("protocol", "heldout must remain closed")

    manifest_sha = episodes.get("manifest_sha256")
    if not _is_hex(manifest_sha):
        report.fail("protocol", "episodes.manifest_sha256 must be a SHA-256")
    parent_value = comparison.get("parent_checkpoint")
    if not isinstance(parent_value, str) or not parent_value:
        report.fail("protocol", "comparison.parent_checkpoint is missing")
        return None
    parent_path = _path(parent_value, artifact_root)
    parent_sha = comparison.get("parent_checkpoint_sha256")
    parent_bytes = comparison.get("parent_checkpoint_bytes")
    if not _is_hex(parent_sha):
        report.fail("protocol", "parent_checkpoint_sha256 must be a SHA-256")
    if not _is_int(parent_bytes) or parent_bytes <= 0:
        report.fail("protocol", "parent_checkpoint_bytes must be a positive integer")

    parent = {"declared_path": parent_value, "path": parent_path,
              "declared_sha256": parent_sha, "declared_bytes": parent_bytes,
              "sha256": None, "bytes": None}
    try:
        parent["sha256"] = _sha_file(parent_path)
        parent["bytes"] = parent_path.stat().st_size
    except OSError as error:
        report.fail("parent_checkpoint", f"cannot read parent checkpoint: {error}")
    else:
        if parent["sha256"] != parent_sha:
            report.fail("parent_checkpoint",
                        "parent checkpoint sha256 differs from the declaration")
        if parent["bytes"] != parent_bytes:
            report.fail("parent_checkpoint",
                        "parent checkpoint byte size differs from the declaration")

    return {
        "config": config, "comparison": comparison, "runtime": runtime,
        "storage": storage, "bank": bank, "features": features,
        "routing": routing, "read_path": read_path,
        "transaction": transaction, "episodes": episodes,
        "implementation_paths": implementation_paths,
        "protocol_sha256": declared, "manifest_sha256": manifest_sha,
        "config_sha256": _sha_file(protocol_path),
        "artifact_root": Path(artifact_root),
        "parent": parent,
    }


def _validate_protocol_body(report: Report, body: Any, label: str,
                            ctx: Mapping) -> bool:
    if not isinstance(body, Mapping):
        return report.fail("protocol", f"{label} protocol.json is not an object")
    declared = ctx["protocol_sha256"]
    if body.get("protocol_sha256") != declared:
        report.fail("protocol",
                    f"{label} protocol.json protocol_sha256 differs")
    payload = {key: value for key, value in body.items()
               if key != "protocol_sha256"}
    try:
        computed = _sha_json(payload)
    except (TypeError, ValueError) as error:
        return report.fail("protocol",
                           f"{label} protocol.json cannot be hashed: {error}")
    if computed != declared:
        report.fail("protocol",
                    f"{label} protocol.json body hash differs from protocol_sha256")

    bank = ctx["bank"]
    features = ctx["features"]
    routing = ctx["routing"]
    transaction = ctx["transaction"]
    expected = {
        "expert_count": bank.get("experts"),
        "width": bank.get("width"),
        "hidden_width": bank.get("hidden_width"),
        "feature_sites": features.get("sites"),
        "target_projection_seed": features.get("target_projection_seed"),
        "read_gain": ctx["read_path"].get("gain"),
        "route_mode": routing.get("mode"),
        "route_seed": routing.get("seed"),
        "p_init": bank.get("p_init"),
        "steps_per_event": transaction.get("steps_per_event"),
        "total_step_cap": transaction.get("total_step_cap"),
        "clock_period": transaction.get("clock_period"),
        "gradient_norm_max": transaction.get("gradient_norm_max"),
    }
    for key, value in expected.items():
        if body.get(key) != value:
            report.fail("protocol",
                        f"{label} protocol.json {key} differs from the config")
    sites = body.get("feature_sites")
    if isinstance(sites, list):
        sites = tuple(sites)
    if sites != tuple(features.get("sites") or ()):
        report.fail("protocol", f"{label} protocol.json feature sites differ")
    for key in ("protocol_sha256", "target_projection_sha256",
                "read_path_sha256", "theta_lock_sha256",
                "parameter_stamp_sha256", "tokenizer_sha256"):
        if not _is_hex(body.get(key)):
            report.fail("protocol", f"{label} protocol.json {key} is not a SHA-256")
    if body.get("target_projection_sha256") != body.get("read_path_sha256"):
        report.fail("protocol",
                    f"{label} protocol read decoder differs from the target projection")
    for key in ("world_uid", "scope", "theta_revision"):
        if not isinstance(body.get(key), str) or not body.get(key):
            report.fail("protocol", f"{label} protocol.json {key} is empty")
    return True


def _validate_implementation(report: Report, mapping: Any, label: str,
                             expected_paths: list[str],
                             expected_map: Mapping | None,
                             source_label: str) -> None:
    if expected_map is None:
        # The source snapshot or receipt already failed authentication.
        return
    if not isinstance(mapping, Mapping):
        report.fail("implementation", f"{label} implementation_sha256 is missing")
        return
    if set(mapping) != set(expected_paths):
        report.fail("implementation",
                    f"{label} implementation path set differs from the protocol")
    for relative in sorted(set(expected_paths)):
        expected = expected_map.get(relative)
        if expected is None:
            report.fail("implementation",
                        f"{source_label} does not pin {relative}")
            continue
        if mapping.get(relative) != expected:
            report.fail("implementation",
                        f"{label} implementation hash differs from the "
                        f"{source_label} at {relative}")


def _validate_guard(report: Report, path: Path, label: str,
                    runtime: Mapping, binding: Mapping | None,
                    expected_output_name: str | None = None) -> dict | None:
    try:
        guard = _load_json(path)
    except (OSError, ValueError) as error:
        report.fail("guard", f"{label} cannot be read: {error}", resource=True,
                    invalid=True)
        return None
    if not isinstance(guard, Mapping):
        report.fail("guard", f"{label} is not a JSON object", resource=True,
                    invalid=True)
        return None
    if binding is not None:
        declared = guard.get("implementation_sha256")
        if not isinstance(declared, Mapping):
            report.fail("guard",
                        f"{label} guard.implementation_sha256 is missing",
                        resource=True, invalid=True)
        elif dict(declared) != dict(binding["implementation_sha256"]):
            report.fail("guard",
                        f"{label} guard implementation map differs from the "
                        "declared launcher/child binding", resource=True,
                        invalid=True)
        command = guard.get("command")
        if not (isinstance(command, list) and command
                and all(isinstance(item, str) and item for item in command)):
            report.fail("guard", f"{label} guard.command is not a string list",
                        resource=True, invalid=True)
        else:
            executable = Path(command[0]).name
            if not executable.startswith("python"):
                report.fail("guard",
                            f"{label} guard.command[0] is not a Python launcher "
                            f"(saw {executable!r})", resource=True,
                            invalid=True)
            child = Path(binding["child"]).name
            matches = [item for item in command[1:]
                       if Path(item).name == child]
            if len(matches) != 1:
                report.fail("guard",
                            f"{label} guard.command must name exactly one declared "
                            f"child {child!r}", resource=True, invalid=True)
        if expected_output_name is not None:
            output = guard.get("output")
            if not isinstance(output, str) or Path(output).name != \
                    expected_output_name:
                report.fail("guard",
                            f"{label} guard.output does not name the declared "
                            f"artifact {expected_output_name!r}", resource=True,
                            invalid=True)
    status = guard.get("status")
    if status != "completed_within_declared_sampled_limits":
        report.fail("guard", f"{label} status must be "
                    "completed_within_declared_sampled_limits "
                    f"(saw {status!r})", resource=True)
    body = guard.get("guard")
    if not isinstance(body, Mapping):
        report.fail("guard", f"{label} guard block is missing", resource=True,
                    invalid=True)
        return None
    if body.get("stop_reason") is not None:
        report.fail("guard", f"{label} guard.stop_reason must be null "
                    f"(saw {body.get('stop_reason')!r})", resource=True)
    policy = body.get("policy")
    if not isinstance(policy, Mapping):
        report.fail("guard", f"{label} guard.policy is missing", resource=True,
                    invalid=True)
        policy = {}
    expected_policy = {
        "driver_limit_bytes": runtime.get("max_driver_bytes"),
        "swap_growth_limit_bytes": runtime.get("max_swap_growth_bytes"),
        "minimum_free_percent": runtime.get("minimum_free_percent"),
        "sustained_host_samples": runtime.get("sustained_host_samples"),
    }
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            report.fail("guard", f"{label} policy.{key} differs from the protocol",
                        resource=True)
    peak_swap = body.get("peak_swap_growth_bytes")
    peak_driver = body.get("peak_driver_bytes")
    free_percent = body.get("minimum_observed_free_percent")
    if not _is_int(peak_swap) or peak_swap < 0:
        report.fail("guard", f"{label} peak_swap_growth_bytes is invalid",
                    resource=True, invalid=True)
    elif peak_swap > runtime.get("max_swap_growth_bytes", 0):
        report.fail("guard", f"{label} peak_swap_growth_bytes exceeds the limit",
                    resource=True)
    if not _is_int(peak_driver) or peak_driver < 0:
        report.fail("guard", f"{label} peak_driver_bytes is invalid",
                    resource=True, invalid=True)
    elif peak_driver > runtime.get("max_driver_bytes", 0):
        report.fail("guard", f"{label} peak_driver_bytes exceeds the limit",
                    resource=True)
    if not _is_number(free_percent):
        report.fail("guard", f"{label} minimum_observed_free_percent is invalid",
                    resource=True, invalid=True)
    elif free_percent < runtime.get("minimum_free_percent", 0):
        report.fail("guard", f"{label} minimum_observed_free_percent is below the limit",
                    resource=True)
    for key in ("host_samples", "driver_samples"):
        value = body.get(key)
        if not _is_int(value) or value < 1:
            report.fail("guard", f"{label} {key} must be a positive integer",
                        resource=True, invalid=True)
    if "child_exit_code" in guard and guard.get("child_exit_code") != 0:
        report.fail("guard", f"{label} child_exit_code is not zero",
                    resource=True)
    try:
        sha = _sha_file(path)
    except OSError:
        sha = None
    return {
        "path": str(path), "sha256": sha, "status": status,
        "stop_reason": body.get("stop_reason"),
        "peak_driver_bytes": peak_driver,
        "peak_swap_growth_bytes": peak_swap,
        "minimum_observed_free_percent": free_percent,
        "implementation_sha256": (dict(guard.get("implementation_sha256"))
                                  if isinstance(
                                      guard.get("implementation_sha256"),
                                      Mapping) else None),
    }


def _normalise_binding(value: Any, label: str, *,
                       require_format: bool) -> dict | None:
    fields = {"launcher", "child", "implementation_sha256"}
    if require_format:
        fields |= {"format", "reason"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise provenance.SourceProvenanceError(
            f"{label} fields differ from the declared guard binding")
    if require_format:
        if value.get("format") != GUARD_BINDING_FORMAT:
            raise provenance.SourceProvenanceError(
                f"{label} format differs")
        if not isinstance(value.get("reason"), str) or not value["reason"].strip():
            raise provenance.SourceProvenanceError(f"{label} lacks a reason")
    launcher = provenance.safe_relative(value["launcher"],
                                        f"{label} launcher").as_posix()
    child = provenance.safe_relative(value["child"],
                                     f"{label} child").as_posix()
    if Path(launcher).suffix != ".py" or Path(child).suffix != ".py":
        raise provenance.SourceProvenanceError(
            f"{label} launcher/child must be Python entry points")
    mapping = value["implementation_sha256"]
    if not isinstance(mapping, Mapping) or not mapping:
        raise provenance.SourceProvenanceError(
            f"{label} implementation map is missing")
    result = {}
    for relative, digest in mapping.items():
        safe = provenance.safe_relative(relative,
                                        f"{label} implementation path")
        if not provenance.is_sha256(digest):
            raise provenance.SourceProvenanceError(
                f"{label} implementation pin is invalid: {safe.as_posix()}")
        result[safe.as_posix()] = digest
    if launcher not in result:
        raise provenance.SourceProvenanceError(
            f"{label} does not authenticate its own launcher")
    return {"launcher": launcher, "child": child,
            "implementation_sha256": result}


def _bind_guard_to_source(report: Report, binding: dict, ctx: Mapping,
                          label: str) -> bool:
    expected = ctx.get("expected_implementation")
    if expected is None:
        return False
    for relative, digest in binding["implementation_sha256"].items():
        if expected.get(relative) != digest:
            report.fail("guard",
                        f"{label} implementation pin differs from the reviewed "
                        f"source map at {relative}", resource=True,
                        invalid=True)
            return False
    return True


def _load_guard_binding(path: Path, ctx: Mapping,
                        report: Report) -> dict | None:
    try:
        document = provenance.load_json_object(path, "Guard binding")
        binding = _normalise_binding(document, "Guard binding",
                                     require_format=True)
    except provenance.SourceProvenanceError as error:
        report.fail("guard", str(error), resource=True, invalid=True)
        return None
    if not _bind_guard_to_source(report, binding, ctx, "Guard binding"):
        return None
    return binding


def _resolve_implementation_source(
        mode: str, ctx: Mapping, *, artifact_root: Path,
        protocol_path: Path, source_root: Path | None,
        current_receipt: Path | None, guard_binding: Path | None,
        report: Report) -> tuple[dict, dict | None]:
    """Authenticate the artifact implementation map and guard binding."""
    paths = ctx["implementation_paths"]

    def resolve(value: Path | None, default: Path) -> Path:
        if value is None:
            return default
        return value if value.is_absolute() else artifact_root / value

    try:
        if mode == "historical":
            snapshot_path = resolve(source_root,
                                    ROOT / HISTORICAL_SOURCE_SNAPSHOT)
            snapshot = provenance.load_source_snapshot(snapshot_path)
            expected = provenance.historical_map(snapshot, paths)
            ctx["expected_implementation"] = expected
            source = {
                "mode": "historical",
                "snapshot_sha256": snapshot["sha256"],
                "receipt_sha256": None,
                "paths": expected,
            }
            binding_path = resolve(guard_binding, ROOT / GUARD_BINDING)
            binding = _load_guard_binding(Path(binding_path), ctx, report)
            return source, binding
        if mode != "current":
            raise provenance.SourceProvenanceError(
                f"Unknown implementation source mode: {mode}")
        if current_receipt is None:
            raise provenance.SourceProvenanceError(
                "Current implementation verification requires a reviewed "
                "current receipt")
        receipt_path = (Path(current_receipt)
                        if Path(current_receipt).is_absolute()
                        else artifact_root / Path(current_receipt))
        receipt = provenance.load_json_object(receipt_path,
                                             "Current continuation receipt")
        if set(receipt) != {
                "format", "mode", "declared_at_utc", "reason", "config",
                "implementation_sha256", "guard", "receipt_digest"}:
            raise provenance.SourceProvenanceError(
                "Current continuation receipt fields differ")
        if receipt.get("format") != CURRENT_SOURCE_RECEIPT_FORMAT:
            raise provenance.SourceProvenanceError(
                "Current continuation receipt format differs")
        if receipt.get("mode") != provenance.CURRENT_RECEIPT_MODE:
            raise provenance.SourceProvenanceError(
                "Current continuation receipt must be explicitly reviewed")
        if not isinstance(receipt.get("reason"), str) or not receipt["reason"].strip():
            raise provenance.SourceProvenanceError(
                "Current continuation receipt lacks a reason")
        if receipt.get("receipt_digest") != provenance.receipt_digest(receipt):
            raise provenance.SourceProvenanceError(
                "Current continuation receipt digest differs")
        config = receipt.get("config")
        if not isinstance(config, Mapping) or set(config) != {"path", "sha256"}:
            raise provenance.SourceProvenanceError(
                "Current continuation receipt config pin is malformed")
        declared_config = provenance.safe_relative(
            config["path"], "Current receipt config path")
        if Path(declared_config).name != protocol_path.name:
            raise provenance.SourceProvenanceError(
                "Current continuation receipt pins a different config")
        if config["sha256"] != ctx["config_sha256"]:
            raise provenance.SourceProvenanceError(
                "Current continuation receipt config pin is stale")
        declared_map = receipt.get("implementation_sha256")
        if not isinstance(declared_map, Mapping) or \
                set(declared_map) != set(paths):
            raise provenance.SourceProvenanceError(
                "Current continuation receipt does not pin the declared "
                "implementation inventory")
        for relative, digest in declared_map.items():
            if not provenance.is_sha256(digest):
                raise provenance.SourceProvenanceError(
                    f"Current continuation receipt pin is invalid: {relative}")
        live = provenance.compute_implementation_map(artifact_root, paths)
        if dict(declared_map) != live:
            stale = sorted(relative for relative in paths
                           if declared_map.get(relative) != live[relative])
            raise provenance.SourceProvenanceError(
                "Current continuation receipt does not match the live "
                f"implementation bytes: {stale}")
        ctx["expected_implementation"] = live
        binding = _normalise_binding(receipt.get("guard"),
                                     "Current receipt guard binding",
                                     require_format=False)
        source = {
            "mode": "current",
            "snapshot_sha256": None,
            "receipt_sha256": provenance.sha256_file(receipt_path),
            "paths": dict(declared_map),
        }
        if not _bind_guard_to_source(report, binding, ctx,
                                     "Current receipt guard binding"):
            binding = None
        return source, binding
    except provenance.SourceProvenanceError as error:
        report.fail("implementation", str(error))
        return {"mode": mode, "snapshot_sha256": None,
                "receipt_sha256": None, "paths": None}, None


def _validate_canonical(report: Report, state: Any, *, label: str,
                        event_index: int, expected_ids: tuple[str, ...],
                        ctx: Mapping, protocol_body: Mapping) -> bool:
    before = len(report.failures)
    state_event_id = (expected_ids[event_index - 1]
                      if 1 <= event_index <= len(expected_ids) else None)
    if not isinstance(state, Mapping):
        report.fail("canonical", f"{label} canonical state is missing",
                    event_index=event_index)
        report.note("canonical_validation", event_index, state_event_id, None,
                    "canonical state missing")
        return False
    if state.get("format") != FORMAT:
        report.fail("canonical", f"{label} canonical format differs",
                    event_index=event_index)
        report.note("canonical_validation", event_index, state_event_id, None,
                    "canonical format differs")
        return False
    leaves = state.get("leaves")
    sections = state.get("sections")
    counts = state.get("counts")
    digest = state.get("sha256")
    if not isinstance(leaves, Mapping) or not isinstance(sections, Mapping) \
            or not isinstance(counts, Mapping) or not isinstance(digest, str):
        report.fail("canonical", f"{label} canonical state is incomplete",
                    event_index=event_index)
        report.note("canonical_validation", event_index, state_event_id, None,
                    "canonical state incomplete")
        return False
    try:
        computed_digest = _sha_json(leaves)
        computed_sections = _section_hashes(leaves)
    except (TypeError, ValueError) as error:
        report.fail("canonical",
                    f"{label} canonical state cannot be hashed: {error}",
                    event_index=event_index)
        report.note("canonical_validation", event_index, state_event_id, None,
                    "canonical state unhashable")
        return False
    if computed_digest != digest:
        report.fail("canonical", f"{label} canonical sha256 differs from its leaves",
                    event_index=event_index)
    missing_sections = [name for name in SECTIONS if name not in sections]
    if missing_sections:
        report.fail("canonical",
                    f"{label} canonical sections missing: {missing_sections}",
                    event_index=event_index)
    for name in SECTIONS:
        if sections.get(name) != computed_sections.get(name):
            report.fail("canonical",
                        f"{label} canonical section {name} hash differs",
                        event_index=event_index)

    bank = ctx["bank"]
    features = ctx["features"]
    transaction = ctx["transaction"]
    experts = int(bank.get("experts", 0))
    width = int(bank.get("width", 0))
    hidden_width = int(bank.get("hidden_width", 0))
    expected_parameters = 3 * experts
    expected_elements = 3 * experts * width * hidden_width

    parameter_indices = _index_set(leaves, PARAMETER_LEAF)
    optimizer_indices = _index_set(leaves, OPTIMIZER_LEAF)
    optimizer_sq_indices = _index_set(leaves, OPTIMIZER_SQ_LEAF)
    optimizer_step_indices = _index_set(leaves, OPTIMIZER_STEP_LEAF)
    observation_indices = _index_set(leaves, OBSERVATION_INDEX)
    chunk_indices = _index_set(leaves, CHUNK_INDEX)
    dedup_indices = _index_set(leaves, DEDUP_INDEX)
    expected_slice = expected_ids[:event_index]
    fingerprint_keys = {
        key[len(EVENT_FINGERPRINT_PREFIX):]
        for key in leaves if key.startswith(EVENT_FINGERPRINT_PREFIX)}

    if counts.get("parameter_tensors") != expected_parameters:
        report.fail("canonical",
                    f"{label} count parameter_tensors must be {expected_parameters}",
                    event_index=event_index)
    if counts.get("parameter_elements") != expected_elements:
        report.fail("canonical",
                    f"{label} count parameter_elements must be {expected_elements}",
                    event_index=event_index)
    if parameter_indices != set(range(expected_parameters)):
        report.fail("canonical",
                    f"{label} does not contain all {expected_parameters} parameter tensors",
                    event_index=event_index)
    for position in sorted(parameter_indices):
        leaf = leaves.get(f"controller.bank.parameters[{position}][1]")
        if not isinstance(leaf, Mapping) or leaf.get("kind") != "tensor" \
                or not _is_hex(leaf.get("sha256")):
            report.fail("canonical",
                        f"{label} parameter tensor {position} is not hashed",
                        event_index=event_index)
    if counts.get("optimizer_state_entries") != len(optimizer_indices) \
            or not optimizer_indices:
        report.fail("canonical",
                    f"{label} optimizer moments are incomplete",
                    event_index=event_index)
    if (optimizer_sq_indices != optimizer_indices
            or optimizer_step_indices != optimizer_indices):
        report.fail("canonical",
                    f"{label} optimizer moment triples are incomplete",
                    event_index=event_index)
    for position in sorted(optimizer_indices):
        for name in ("exp_avg", "exp_avg_sq"):
            leaf = leaves.get(
                f"controller.bank.optimizer.state.{position}.{name}")
            if not isinstance(leaf, Mapping) or leaf.get("kind") != "tensor" \
                    or not _is_hex(leaf.get("sha256")):
                report.fail("canonical",
                            f"{label} optimizer {name} {position} is not hashed",
                            event_index=event_index)
    if counts.get("observation_count") != event_index \
            or observation_indices != set(range(event_index)):
        report.fail("canonical",
                    f"{label} observation count differs from event_index",
                    event_index=event_index)
    if counts.get("chunk_count") != event_index \
            or chunk_indices != set(range(event_index)):
        report.fail("canonical",
                    f"{label} coordinator chunk count differs from event_index",
                    event_index=event_index)
    chunk_hashes = []
    for index in sorted(chunk_indices):
        value = _leaf_value(
            leaves, f"controller.observation.chunks[{index}].fingerprint")
        if not _is_hex(value):
            report.fail("canonical",
                        f"{label} chunk fingerprint {index} is invalid",
                        event_index=event_index)
        else:
            chunk_hashes.append(value)
    if counts.get("event_fingerprint_count") != event_index \
            or fingerprint_keys != set(expected_slice):
        report.fail("canonical",
                    f"{label} event fingerprint set differs from the sequence",
                    event_index=event_index)
    for event_id in expected_slice:
        value = _leaf_value(leaves,
                            f"{EVENT_FINGERPRINT_PREFIX}{event_id}")
        if not _is_hex(value):
            report.fail("canonical",
                        f"{label} event fingerprint for {event_id} is invalid",
                        event_index=event_index)
    if dedup_indices != set(range(event_index)):
        report.fail("canonical",
                    f"{label} dedup record count differs from event_index",
                    event_index=event_index)
    dedup_ids = set()
    for index in sorted(dedup_indices):
        event_id = _leaf_value(leaves,
                               f"controller.dedup.records[{index}].event_id")
        fingerprint = _leaf_value(
            leaves, f"controller.dedup.records[{index}].fingerprint")
        dedup_ids.add(event_id)
        if not _is_hex(fingerprint):
            report.fail("canonical",
                        f"{label} dedup fingerprint {index} is invalid",
                        event_index=event_index)
    if dedup_ids != set(expected_slice):
        report.fail("canonical",
                    f"{label} dedup event ids differ from the sequence",
                    event_index=event_index)
    if counts.get("leaf_count") != len(leaves):
        report.fail("canonical", f"{label} count leaf_count differs",
                    event_index=event_index)

    sealed = _leaf_value(leaves, "controller.observation.latest_sealed_at")
    if not _is_number(sealed):
        report.fail("canonical",
                    f"{label} latest_sealed_at is missing or non-finite",
                    event_index=event_index)
    if _leaf_value(leaves, "controller.revision") != event_index:
        report.fail("canonical",
                    f"{label} controller revision differs from event_index",
                    event_index=event_index)
    if _leaf_value(leaves, "controller.clock.eligible_events") != event_index:
        report.fail("canonical",
                    f"{label} teaching clock differs from event_index",
                    event_index=event_index)
    for name, expected in (
            ("controller.max_steps", transaction.get("max_steps")),
            ("controller.total_step_cap", transaction.get("total_step_cap")),
            ("controller.clock_period", transaction.get("clock_period")),
            ("controller.gradient_norm_max",
             transaction.get("gradient_norm_max"))):
        if _leaf_value(leaves, name) != expected:
            report.fail("canonical",
                        f"{label} {name} differs from the protocol",
                        event_index=event_index)
    for name, expected in (
            ("controller.bank.expert_count", experts),
            ("controller.bank.width", width),
            ("controller.bank.hidden_width", hidden_width),
            ("controller.bank.seed", bank.get("seed"))):
        if _leaf_value(leaves, name) != expected:
            report.fail("canonical",
                        f"{label} {name} differs from the protocol",
                        event_index=event_index)

    # Every event state must embed the same declared TextProtocol body.
    for key, expected in protocol_body.items():
        if isinstance(expected, list):
            for position, item in enumerate(expected):
                path = f"text.protocol.{key}[{position}]"
                _check_scalar_leaf(report, leaves, path, item, label,
                                   event_index, "protocol")
        else:
            _check_scalar_leaf(report, leaves, f"text.protocol.{key}", expected,
                               label, event_index, "protocol")
    _check_scalar_leaf(report, leaves, "text.feature_producer.hidden_size",
                       protocol_body.get("hidden_size"), label, event_index,
                       "feature producer")
    _check_scalar_leaf(report, leaves, "text.feature_producer.width",
                       protocol_body.get("width"), label, event_index,
                       "feature producer")
    _check_scalar_leaf(report, leaves,
                       "text.feature_producer.target_projection_seed",
                       protocol_body.get("target_projection_seed"), label,
                       event_index, "feature producer")
    _check_scalar_leaf(report, leaves,
                       "text.feature_producer.target_projection_sha256",
                       protocol_body.get("target_projection_sha256"), label,
                       event_index, "feature producer")
    for position, site in enumerate(protocol_body.get("feature_sites") or ()):
        _check_scalar_leaf(report, leaves,
                           f"text.feature_producer.sites[{position}]", site,
                           label, event_index, "feature producer")
    _check_scalar_leaf(report, leaves, "text.read_path.gain",
                       protocol_body.get("read_gain"), label, event_index,
                       "read path")
    _check_scalar_leaf(report, leaves, "text.read_path.decoder_sha256",
                       protocol_body.get("read_path_sha256"), label,
                       event_index, "read path")
    _check_scalar_leaf(report, leaves, "text.route.mode",
                       protocol_body.get("route_mode"), label, event_index,
                       "route")
    _check_scalar_leaf(report, leaves, "text.route.seed",
                       protocol_body.get("route_seed"), label, event_index,
                       "route")
    _check_scalar_leaf(report, leaves, "text.route.expert_count", experts,
                       label, event_index, "route")
    permutation = sorted(
        int(match.group(1)) for key in leaves
        if (match := re.match(r"^text\.route\.permutation\[(\d+)\]$", key)))
    if permutation != list(range(experts)):
        report.fail("canonical", f"{label} route permutation is incomplete",
                    event_index=event_index)
    if len(report.failures) > before:
        report.note("canonical_validation", event_index, state_event_id, None,
                    f"{label} canonical state differs from the declared shape")
    return len(report.failures) == before


def _check_scalar_leaf(report: Report, leaves: Mapping, path: str,
                       expected: Any, label: str, event_index: int,
                       what: str) -> None:
    leaf = leaves.get(path)
    if not isinstance(leaf, Mapping):
        report.fail("canonical", f"{label} {what} leaf {path} is missing",
                    event_index=event_index)
        return
    if leaf.get("kind") != _kind(expected) or leaf.get("value") != expected:
        report.fail("canonical",
                    f"{label} {what} leaf {path} differs from the protocol",
                    event_index=event_index)


def _validate_outcome(report: Report, row: Any, index: int, event_id: str,
                      label: str, ctx: Mapping) -> bool:
    before = len(report.failures)
    if not isinstance(row, Mapping):
        report.fail("outcome", f"{label} transaction is not an object",
                    event_index=index, event_id=event_id)
        report.note("outcome", index, event_id, None,
                    f"{label} transaction is not an object")
        return False
    required = ("status", "event_id", "expert_id", "route_mode",
                "feature_sha256", "target_sha256", "retried", "transaction",
                "chunk", "gradient_norm_before_clipping")
    missing = [key for key in required if key not in row]
    if missing:
        report.fail("outcome",
                    f"{label} transaction is missing {missing}",
                    event_index=index, event_id=event_id)
        report.note("outcome", index, event_id, ",".join(missing),
                    f"{label} transaction is missing required fields")
        return False
    if row.get("event_id") != event_id:
        report.fail("outcome", f"{label} transaction event_id differs",
                    event_index=index, event_id=event_id)
    if row.get("status") != "committed":
        report.fail("outcome", f"{label} transaction status is not committed",
                    event_index=index, event_id=event_id)
    if row.get("route_mode") != "fixed":
        report.fail("outcome", f"{label} route_mode is not fixed",
                    event_index=index, event_id=event_id)
    if row.get("retried") is not False:
        report.fail("outcome", f"{label} transaction must not be retried",
                    event_index=index, event_id=event_id)
    for key in ("feature_sha256", "target_sha256"):
        if not _is_hex(row.get(key)):
            report.fail("outcome", f"{label} {key} is not a SHA-256",
                        event_index=index, event_id=event_id)
    expert_id = row.get("expert_id")
    experts = int(ctx["bank"].get("experts", 0))
    if not _is_int(expert_id) or not 0 <= expert_id < experts:
        report.fail("outcome", f"{label} expert_id is invalid",
                    event_index=index, event_id=event_id)
    gradient = row.get("gradient_norm_before_clipping")
    if not _is_number(gradient) or gradient < 0:
        report.fail("outcome",
                    f"{label} gradient_norm_before_clipping must be a finite "
                    "nonnegative number", event_index=index, event_id=event_id)

    transaction = row.get("transaction")
    if not isinstance(transaction, Mapping):
        report.fail("outcome", f"{label} nested transaction is missing",
                    event_index=index, event_id=event_id)
    else:
        for key, expected in (
                ("status", "committed"), ("event_id", event_id),
                ("expert_id", expert_id), ("revision", index),
                ("eligible_events", index), ("steps", 1)):
            if transaction.get(key) != expected:
                report.fail("outcome",
                            f"{label} transaction field {key} differs",
                            event_index=index, event_id=event_id)
        if not _is_hex(transaction.get("fingerprint")):
            report.fail("outcome",
                        f"{label} transaction fingerprint is not a SHA-256",
                        event_index=index, event_id=event_id)
        if not _is_number(transaction.get("loss")):
            report.fail("outcome", f"{label} transaction loss is not finite",
                        event_index=index, event_id=event_id)

    chunk = row.get("chunk")
    if not isinstance(chunk, Mapping):
        report.fail("outcome", f"{label} chunk outcome is missing",
                    event_index=index, event_id=event_id)
    else:
        for key, expected in (
                ("status", "committed"), ("pre_revision", index - 1),
                ("post_revision", index), ("transition_count", 1)):
            if chunk.get(key) != expected:
                report.fail("outcome", f"{label} chunk field {key} differs",
                            event_index=index, event_id=event_id)
        if not _is_hex(chunk.get("fingerprint")):
            report.fail("outcome",
                        f"{label} chunk fingerprint is not a SHA-256",
                        event_index=index, event_id=event_id)
        if not _is_number(chunk.get("sealed_at")):
            report.fail("outcome", f"{label} chunk sealed_at is not finite",
                        event_index=index, event_id=event_id)
        if not (isinstance(chunk.get("observation_ids"), list)
                and len(chunk["observation_ids"]) == 1):
            report.fail("outcome",
                        f"{label} chunk must contain exactly one observation",
                        event_index=index, event_id=event_id)
        if not (isinstance(chunk.get("reader_ids"), list)
                and len(chunk["reader_ids"]) == 2):
            report.fail("outcome", f"{label} chunk reader ids differ",
                        event_index=index, event_id=event_id)
        for key in ("aggregate_key", "aggregate_value"):
            values = chunk.get(key)
            if not (isinstance(values, list) and values
                    and all(_is_number(item) for item in values)):
                report.fail("outcome", f"{label} chunk {key} is invalid",
                            event_index=index, event_id=event_id)
    if len(report.failures) > before:
        report.note("outcome", index, event_id, None,
                    f"{label} transaction differs from the declared contract")
    return True


def _bind_outcome_to_state(report: Report, state: Mapping, row: Mapping,
                           index: int, event_id: str, label: str) -> None:
    before = len(report.failures)
    leaves = state.get("leaves") if isinstance(state, Mapping) else None
    if not isinstance(leaves, Mapping):
        return
    transaction = row.get("transaction")
    chunk = row.get("chunk")
    if not isinstance(transaction, Mapping) or not isinstance(chunk, Mapping):
        return
    record_index = None
    for position in sorted(_index_set(leaves, DEDUP_INDEX)):
        if _leaf_value(leaves,
                       f"controller.dedup.records[{position}].event_id") == event_id:
            record_index = position
            break
    if record_index is None:
        report.fail("state_outcome_binding",
                    f"{label} canonical state lacks the dedup record",
                    event_index=index, event_id=event_id)
        report.note("state_outcome_binding", index, event_id, None,
                    f"{label} dedup record is missing")
        return
    base = f"controller.dedup.records[{record_index}]"
    checks = {
        f"{base}.fingerprint": transaction.get("fingerprint"),
        f"{base}.outcome.fingerprint": transaction.get("fingerprint"),
        f"{base}.outcome.event_id": event_id,
        f"{base}.outcome.status": "committed",
        f"{base}.outcome.expert_id": transaction.get("expert_id"),
        f"{base}.outcome.revision": index,
        f"{base}.outcome.eligible_events": index,
        f"{base}.outcome.steps": 1,
        f"{base}.outcome.loss": transaction.get("loss"),
    }
    for path, expected in checks.items():
        if _leaf_value(leaves, path) != expected:
            report.fail("state_outcome_binding",
                        f"{label} {path} differs from the transaction",
                        event_index=index, event_id=event_id, path=path)
    chunk_fingerprint = chunk.get("fingerprint")
    chunk_matches = [
        position for position in _index_set(leaves, CHUNK_INDEX)
        if _leaf_value(
            leaves,
            f"controller.observation.chunks[{position}].fingerprint")
        == chunk_fingerprint]
    if not chunk_matches:
        report.fail("state_outcome_binding",
                    f"{label} chunk fingerprint is absent from the state",
                    event_index=index, event_id=event_id)
    else:
        position = chunk_matches[0]
        base = f"controller.observation.chunks[{position}]"
        if _leaf_value(leaves, f"{base}.post_revision") != index:
            report.fail("state_outcome_binding",
                        f"{label} chunk post_revision differs",
                        event_index=index, event_id=event_id)
        if _leaf_value(leaves, f"{base}.transition_count") != 1:
            report.fail("state_outcome_binding",
                        f"{label} chunk transition_count differs",
                        event_index=index, event_id=event_id)
    if len(report.failures) > before:
        report.note("state_outcome_binding", index, event_id, None,
                    f"{label} state does not bind the reported outcome")


def _compare_canonical(report: Report, left: Mapping, right: Mapping,
                       stage: str, index: int, event_id: str,
                       label: str, *, aa: bool = False) -> bool:
    try:
        result = compare_states(left, right)
    except (TypeError, ValueError) as error:
        return report.fail(stage, f"{label} compare_states failed: {error}",
                           event_index=index, event_id=event_id, stage=stage)
    if aa:
        report.record_aa(result["equal"])
    if result["equal"]:
        return True
    first = result.get("first_difference") or {}
    path = first.get("path")
    detail = (f"{label} canonical states differ"
              + (f" at {path}" if path else ""))
    if result.get("section_differences"):
        detail += f" sections={result['section_differences'][:4]}"
    report.fail(stage, detail, event_index=index, event_id=event_id,
                path=path, stage=stage)
    return False


def _compare_outcome(report: Report, left: Mapping, right: Mapping,
                     index: int, event_id: str, label: str, *,
                     aa: bool = False) -> None:
    equal = left == right
    if aa:
        report.record_aa(equal)
    if equal:
        return
    difference = _first_difference(left, right) or {}
    report.fail("aa_outcome",
                f"{label} per-event transaction differs at "
                f"{difference.get('path')}: left={difference.get('left')} "
                f"right={difference.get('right')}",
                event_index=index, event_id=event_id,
                path=difference.get("path"), stage="aa_outcome")


def _validate_cost(report: Report, path: Path, label: str, ctx: Mapping,
                   checkpoint_bytes: int | None, p_only_bytes: int | None) -> dict | None:
    try:
        cost = _load_json(path)
    except (OSError, ValueError) as error:
        report.fail("cost", f"{label} cost.json cannot be read: {error}")
        return None
    if not isinstance(cost, Mapping):
        report.fail("cost", f"{label} cost.json is not an object")
        return None
    if cost.get("protocol_sha256") != ctx["protocol_sha256"]:
        report.fail("cost", f"{label} cost protocol_sha256 differs")
    if cost.get("events") != CONTINUOUS_EVENTS:
        report.fail("cost", f"{label} cost events must be {CONTINUOUS_EVENTS}")
    if cost.get("eligible_events") != CONTINUOUS_EVENTS:
        report.fail("cost", f"{label} cost eligible_events must be "
                    f"{CONTINUOUS_EVENTS}")
    if cost.get("optimizer_steps") != CONTINUOUS_EVENTS:
        report.fail("cost", f"{label} cost optimizer_steps must be "
                    f"{CONTINUOUS_EVENTS}")
    if cost.get("teacher_requests") != CONTINUOUS_EVENTS:
        report.fail("cost", f"{label} cost teacher_requests must be "
                    f"{CONTINUOUS_EVENTS}")
    runtime = ctx["runtime"]
    if cost.get("model_loads") != runtime.get("model_loads_per_process", 1):
        report.fail("cost", f"{label} cost model_loads differs")
    if cost.get("tokenizer_loads") != runtime.get("tokenizer_loads_per_process", 1):
        report.fail("cost", f"{label} cost tokenizer_loads differs")
    wall = cost.get("wall_seconds")
    if not _is_number(wall) or wall < 0:
        report.fail("cost", f"{label} cost wall_seconds is invalid",
                    resource=True, invalid=True)
    elif wall > runtime.get("max_wall_seconds", 0):
        report.fail("cost", f"{label} cost wall_seconds exceeds the limit",
                    resource=True)
    peak = cost.get("peak_driver_bytes")
    if not _is_int(peak) or peak < 0:
        report.fail("cost", f"{label} cost peak_driver_bytes is invalid",
                    resource=True, invalid=True)
    elif peak > runtime.get("max_driver_bytes", 0):
        report.fail("cost", f"{label} cost peak_driver_bytes exceeds the limit",
                    resource=True)
    if checkpoint_bytes is not None and cost.get("checkpoint_bytes") != checkpoint_bytes:
        report.fail("cost", f"{label} cost checkpoint_bytes differs from the file")
    if p_only_bytes is not None and cost.get("p_only_bytes") != p_only_bytes:
        report.fail("cost", f"{label} cost p_only_bytes differs from the file")
    storage = ctx["storage"]
    gates = ctx["config"].get("gates")
    cost_gate = gates.get("cost") if isinstance(gates, Mapping) else {}
    if not isinstance(cost_gate, Mapping):
        cost_gate = {}
    max_checkpoint = storage.get("max_checkpoint_bytes",
                                 cost_gate.get("max_checkpoint_bytes"))
    max_p_only = storage.get("max_p_only_bytes",
                             cost_gate.get("max_p_only_bytes"))
    if checkpoint_bytes is not None and _is_int(max_checkpoint) \
            and checkpoint_bytes > max_checkpoint:
        report.fail("artifacts", f"{label} checkpoint exceeds max_checkpoint_bytes",
                    resource=True)
    if p_only_bytes is not None and _is_int(max_p_only) \
            and p_only_bytes > max_p_only:
        report.fail("artifacts", f"{label} P-only export exceeds max_p_only_bytes",
                    resource=True)
    return cost


def _validate_parameter_audit(report: Report, path: Path, label: str,
                              ctx: Mapping) -> dict | None:
    try:
        audit = _load_json(path)
    except (OSError, ValueError) as error:
        report.fail("theta_audit", f"{label} parameter audit cannot be read: {error}")
        return None
    if not isinstance(audit, Mapping):
        report.fail("theta_audit", f"{label} parameter audit is not an object")
        return None
    if audit.get("theta_stamp_unchanged") is not True:
        report.fail("theta_audit",
                    f"{label} theta_stamp_unchanged must be true")
    if audit.get("read_path_trainable_parameters") != 0:
        report.fail("theta_audit",
                    f"{label} read path must have zero trainable parameters")
    experts = int(ctx["bank"].get("experts", 0))
    expected_names = [
        f"expert_{index}.{name}"
        for index in range(experts)
        for name in ("gate", "up", "down")]
    if audit.get("plastic_parameter_count") != len(expected_names):
        report.fail("theta_audit",
                    f"{label} plastic_parameter_count differs")
    if audit.get("plastic_parameters") != expected_names:
        report.fail("theta_audit",
                    f"{label} plastic parameter names differ")
    steps = audit.get("expert_steps")
    if not (isinstance(steps, list) and len(steps) == experts
            and all(_is_int(item) and item >= 0 for item in steps)):
        report.fail("theta_audit", f"{label} expert_steps is invalid")
    elif sum(steps) != CONTINUOUS_EVENTS:
        report.fail("theta_audit",
                    f"{label} expert_steps must sum to {CONTINUOUS_EVENTS}")
    return dict(audit)


def _dir_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _read_rows(report: Report, path: Path, label: str) -> list | None:
    try:
        return _load_jsonl(path)
    except (OSError, ValueError, ComparisonError) as error:
        report.fail("artifacts", f"{label} cannot be read: {error}")
        return None


def _load_continuous(report: Report, directory: Path, label: str,
                     ctx: Mapping) -> dict | None:
    try:
        run = _load_json(directory / "run.json")
    except (OSError, ValueError) as error:
        report.fail("process_reports",
                    f"{label} run.json cannot be read: {error}")
        return None
    if not isinstance(run, Mapping):
        report.fail("process_reports", f"{label} run.json is not an object")
        return None
    if run.get("format") != RUN_FORMAT:
        report.fail("process_reports", f"{label} run format differs")
    if run.get("mode") != "teach":
        report.fail("process_reports", f"{label} mode must be teach")
    if run.get("status") != "completed":
        report.fail("process_reports", f"{label} status must be completed")
    if run.get("protocol_sha256") != ctx["protocol_sha256"]:
        report.fail("process_reports",
                    f"{label} run protocol_sha256 differs")
    if run.get("episodes_manifest_sha256") != ctx["manifest_sha256"]:
        report.fail("process_reports",
                    f"{label} episodes manifest sha256 differs")
    if run.get("config_sha256") != ctx["config_sha256"]:
        report.fail("process_reports",
                    f"{label} config_sha256 differs from the passed protocol")
    if run.get("theta_stamp_unchanged") is not True:
        report.fail("theta_audit", f"{label} run theta_stamp_unchanged must be true")
    if run.get("updates") != CONTINUOUS_EVENTS \
            or run.get("learning_events") != CONTINUOUS_EVENTS:
        report.fail("process_reports",
                    f"{label} updates/learning_events must be {CONTINUOUS_EVENTS}")
    if run.get("state_history") != "state_history.jsonl":
        report.fail("process_reports",
                    f"{label} state_history filename differs")
    if run.get("final_state") != "final_state.json":
        report.fail("process_reports", f"{label} final_state filename differs")
    if run.get("parameter_audit") != "parameter_audit.json":
        report.fail("process_reports",
                    f"{label} parameter_audit filename differs")
    if run.get("legacy_state_digest") != ctx["comparison"].get("legacy_after_state"):
        report.fail("process_reports",
                    f"{label} legacy_state_digest differs from the declaration")
    checkpoint_field = run.get("checkpoint")
    if not isinstance(checkpoint_field, str) or not checkpoint_field:
        report.fail("process_reports", f"{label} checkpoint path is missing")
    elif Path(checkpoint_field).name != "full_checkpoint.pt":
        report.fail("process_reports",
                    f"{label} checkpoint path does not name full_checkpoint.pt")
    p_only_field = run.get("p_only")
    if not isinstance(p_only_field, str) or not p_only_field:
        report.fail("process_reports", f"{label} p_only path is missing")
    elif Path(p_only_field).name != "state.pt" \
            or Path(p_only_field).parent.name != "p_only":
        report.fail("process_reports",
                    f"{label} p_only path does not name p_only/state.pt")
    _validate_implementation(report, run.get("implementation_sha256"), label,
                             ctx["implementation_paths"],
                             ctx.get("expected_implementation"),
                             ctx.get("implementation_source_label", "source"))

    try:
        protocol_body = _load_json(directory / "protocol.json")
    except (OSError, ValueError) as error:
        report.fail("protocol", f"{label} protocol.json cannot be read: {error}")
        return None
    if not _validate_protocol_body(report, protocol_body, label, ctx):
        return None

    history = _read_rows(report, directory / "state_history.jsonl", label)
    transactions = _read_rows(report, directory / "transactions.jsonl", label)
    if history is None or transactions is None:
        return None
    try:
        final_state = _load_json(directory / "final_state.json")
    except (OSError, ValueError) as error:
        report.fail("canonical", f"{label} final_state.json cannot be read: {error}")
        final_state = None
    try:
        checkpoint = directory / "full_checkpoint.pt"
        checkpoint_bytes = checkpoint.stat().st_size
        checkpoint_sha = _sha_file(checkpoint)
        if run.get("checkpoint_bytes") != checkpoint_bytes:
            report.fail("artifacts",
                        f"{label} checkpoint_bytes differs from the file")
        if run.get("checkpoint_sha256") != checkpoint_sha:
            report.fail("artifacts",
                        f"{label} checkpoint_sha256 differs from the file")
    except OSError as error:
        report.fail("artifacts", f"{label} checkpoint is unreadable: {error}")
        checkpoint_bytes = None
        checkpoint_sha = None

    p_only_path = directory / "p_only" / "state.pt"
    p_only_manifest_path = directory / "p_only" / "manifest.json"
    try:
        p_only_bytes = p_only_path.stat().st_size
        p_only_sha = _sha_file(p_only_path)
        manifest = _load_json(p_only_manifest_path)
    except (OSError, ValueError) as error:
        report.fail("p_only", f"{label} P-only export is unreadable: {error}")
        p_only_bytes = None
        p_only_sha = None
        manifest = None
    if isinstance(manifest, Mapping):
        if manifest.get("format") != P_ONLY_FORMAT:
            report.fail("p_only", f"{label} P-only manifest format differs")
        if manifest.get("bytes") != p_only_bytes:
            report.fail("p_only",
                        f"{label} P-only manifest bytes differ from the file")
        if manifest.get("sha256") != p_only_sha:
            report.fail("p_only",
                        f"{label} P-only manifest sha256 differs from the file")
        if manifest.get("forbidden_absent") is not True:
            report.fail("p_only",
                        f"{label} P-only manifest must confirm forbidden keys absent")
        if manifest.get("source_inputs") != 0:
            report.fail("p_only",
                        f"{label} P-only manifest source_inputs must be zero")
        keys = manifest.get("keys")
        if not isinstance(keys, list) or any(
                key in FORBIDDEN_P_ONLY_KEYS for key in keys):
            report.fail("p_only", f"{label} P-only manifest keys are invalid")
    elif manifest is not None:
        report.fail("p_only", f"{label} P-only manifest is not an object")
    if p_only_sha is not None and run.get("p_only_sha256") != p_only_sha:
        report.fail("p_only",
                    f"{label} run p_only_sha256 differs from the file")

    cost = _validate_cost(report, directory / "cost.json", label, ctx,
                          checkpoint_bytes, p_only_bytes)
    audit = _validate_parameter_audit(
        report, directory / "parameter_audit.json", label, ctx)
    if run.get("canonical_state_sha256") != (
            final_state.get("sha256") if isinstance(final_state, Mapping) else None):
        report.fail("canonical",
                    f"{label} run canonical_state_sha256 differs from final_state")

    storage = ctx["storage"]
    max_new = storage.get("max_new_artifact_bytes_per_arm")
    try:
        total_bytes = _dir_bytes(directory)
    except OSError as error:
        total_bytes = None
        report.fail("artifacts", f"{label} artifact directory unreadable: {error}",
                    resource=True)
    if total_bytes is not None and _is_int(max_new) and total_bytes > max_new:
        report.fail("artifacts",
                    f"{label} new artifact bytes {total_bytes} exceed {max_new}",
                    resource=True)

    return {
        "label": label, "directory": directory, "run": run,
        "protocol_body": protocol_body, "history": history,
        "transactions": transactions, "final_state": final_state,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": checkpoint_bytes,
        "p_only_sha256": p_only_sha, "p_only_bytes": p_only_bytes,
        "p_only_manifest": manifest, "cost": cost, "audit": audit,
        "artifact_bytes": total_bytes,
        "run_sha256": _safe_sha(directory / "run.json"),
    }


def _safe_sha(path: Path) -> str | None:
    try:
        return _sha_file(path)
    except OSError:
        return None


def _load_resume(report: Report, directory: Path, ctx: Mapping) -> dict | None:
    label = "resume"
    try:
        run = _load_json(directory / "run.json")
        manifest = _load_json(directory / "resume.json")
    except (OSError, ValueError) as error:
        report.fail("process_reports",
                    f"{label} report cannot be read: {error}")
        return None
    if not isinstance(run, Mapping) or not isinstance(manifest, Mapping):
        report.fail("process_reports", f"{label} reports are not objects")
        return None
    for name, mapping in (("run.json", run), ("resume.json", manifest)):
        if mapping.get("format") != RUN_FORMAT:
            report.fail("process_reports", f"{label} {name} format differs")
        if mapping.get("mode") != "resume":
            report.fail("process_reports", f"{label} {name} mode must be resume")
        if mapping.get("status") != "completed_pending_independent_comparison":
            report.fail("in_process_reference_verdict",
                        f"{label} {name} status must be "
                        "completed_pending_independent_comparison "
                        f"(saw {mapping.get('status')!r})")
        if "restart_equal" not in mapping or mapping.get("restart_equal") is not None:
            report.fail("in_process_reference_verdict",
                        f"{label} {name} restart_equal must be present and "
                        "null; an in-process reference verdict is not admissible")
        for key in REFERENCE_KEYS:
            if key in mapping:
                report.fail("in_process_reference_verdict",
                            f"{label} {name} contains forbidden {key}")
    if manifest.get("next_event") != RESUME_EVENT_ID:
        report.fail("process_reports", f"{label} next_event must be correction-0003")
    if manifest.get("protocol_sha256") != ctx["protocol_sha256"]:
        report.fail("process_reports", f"{label} resume protocol_sha256 differs")
    if manifest.get("episodes_manifest_sha256") != ctx["manifest_sha256"]:
        report.fail("process_reports",
                    f"{label} resume episodes manifest sha256 differs")
    if manifest.get("config_sha256") != ctx["config_sha256"]:
        report.fail("process_reports",
                    f"{label} resume config_sha256 differs from the passed protocol")
    if manifest.get("before_state") != ctx["comparison"].get("legacy_before_state"):
        report.fail("process_reports",
                    f"{label} before_state differs from the declaration")
    if manifest.get("after_state") != ctx["comparison"].get("legacy_after_state"):
        report.fail("process_reports",
                    f"{label} after_state differs from the declaration")
    if manifest.get("transaction_retried") is not False:
        report.fail("process_reports",
                    f"{label} transaction_retried must be false")
    if manifest.get("descriptive_only") is not True:
        report.fail("process_reports",
                    f"{label} resume report must be descriptive_only")
    for mapping, name in ((run, "run.json"), (manifest, "resume.json")):
        _validate_implementation(
            report, mapping.get("implementation_sha256"),
            f"{label} {name}", ctx["implementation_paths"],
            ctx.get("expected_implementation"),
            ctx.get("implementation_source_label", "source"))
    if run.get("config_sha256") != ctx["config_sha256"]:
        report.fail("process_reports",
                    f"{label} run config_sha256 differs from the passed protocol")
    if run.get("before_state") != ctx["comparison"].get("legacy_before_state"):
        report.fail("process_reports",
                    f"{label} run before_state differs from the declaration")
    if run.get("after_state") != ctx["comparison"].get("legacy_after_state"):
        report.fail("process_reports",
                    f"{label} run after_state differs from the declaration")
    if manifest.get("before_state_file") != "before_state.json":
        report.fail("process_reports",
                    f"{label} before_state_file filename differs")
    if manifest.get("after_state_file") != "after_state.json":
        report.fail("process_reports",
                    f"{label} after_state_file filename differs")
    if manifest.get("state_history") != "state_history.jsonl":
        report.fail("process_reports",
                    f"{label} state_history filename differs")
    if run.get("theta_stamp_unchanged") is not True:
        report.fail("theta_audit",
                    f"{label} run theta_stamp_unchanged must be true")
    if run.get("canonical_state_sha256") != manifest.get("after_canonical_sha256"):
        report.fail("process_reports",
                    f"{label} run/resume canonical_state_sha256 differ")
    if run.get("state_history") != "state_history.jsonl":
        report.fail("process_reports",
                    f"{label} run state_history filename differs")
    if run.get("final_state") != "after_state.json":
        report.fail("process_reports",
                    f"{label} run final_state filename differs")
    if run.get("parameter_audit") != "parameter_audit.json":
        report.fail("process_reports",
                    f"{label} run parameter_audit filename differs")

    parent = ctx["parent"]
    checkpoint_value = manifest.get("checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        report.fail("parent_checkpoint",
                    f"{label} checkpoint path is missing")
    else:
        checkpoint_path = _path(checkpoint_value, ctx["artifact_root"])
        try:
            if checkpoint_path.resolve() != parent["path"].resolve():
                report.fail("parent_checkpoint",
                            f"{label} checkpoint does not point at the declared parent")
        except OSError:
            pass
    if manifest.get("checkpoint_sha256") != parent["declared_sha256"]:
        report.fail("parent_checkpoint",
                    f"{label} checkpoint_sha256 differs from the declared parent")
    if manifest.get("checkpoint_bytes") != parent["declared_bytes"]:
        report.fail("parent_checkpoint",
                    f"{label} checkpoint_bytes differs from the declared parent")
    if run.get("checkpoint_sha256") != parent["declared_sha256"]:
        report.fail("parent_checkpoint",
                    f"{label} run checkpoint_sha256 differs from the declared parent")

    try:
        before_state = _load_json(directory / "before_state.json")
        after_state = _load_json(directory / "after_state.json")
    except (OSError, ValueError) as error:
        report.fail("canonical", f"{label} canonical states unreadable: {error}")
        return None
    if isinstance(before_state, Mapping) \
            and manifest.get("before_canonical_sha256") != before_state.get("sha256"):
        report.fail("canonical",
                    f"{label} before_canonical_sha256 differs from before_state.json")
    if isinstance(after_state, Mapping) \
            and manifest.get("after_canonical_sha256") != after_state.get("sha256"):
        report.fail("canonical",
                    f"{label} after_canonical_sha256 differs from after_state.json")
    history = _read_rows(report, directory / "state_history.jsonl", label)
    audit = _validate_parameter_audit(
        report, directory / "parameter_audit.json", label, ctx)
    if history is None:
        return None
    if len(history) != 1:
        report.fail("event_sequence",
                    f"{label} state_history must contain exactly one event",
                    event_index=CONTINUOUS_EVENTS,
                    event_id=EXPECTED_EVENT_IDS[-1], stage="event_sequence")
    try:
        total_bytes = _dir_bytes(directory)
    except OSError:
        total_bytes = None
    max_new = ctx["storage"].get("max_new_artifact_bytes_per_arm")
    if total_bytes is not None and _is_int(max_new) and total_bytes > max_new:
        report.fail("artifacts",
                    f"{label} new artifact bytes {total_bytes} exceed {max_new}",
                    resource=True)
    return {
        "label": label, "directory": directory, "run": run,
        "resume": manifest, "before_state": before_state,
        "after_state": after_state, "history": history, "audit": audit,
        "artifact_bytes": total_bytes,
        "run_sha256": _safe_sha(directory / "run.json"),
        "resume_sha256": _safe_sha(directory / "resume.json"),
    }


def _check_continuous_sequence(report: Report, arm: Mapping,
                               ctx: Mapping) -> None:
    label = arm["label"]
    history = arm["history"]
    transactions = arm["transactions"]
    if not isinstance(history, list) or not isinstance(transactions, list):
        return
    if len(history) != CONTINUOUS_EVENTS:
        report.fail("event_sequence",
                    f"{label} state_history has {len(history)} rows, expected "
                    f"{CONTINUOUS_EVENTS}",
                    event_index=min(len(history) + 1, CONTINUOUS_EVENTS),
                    event_id=(EXPECTED_EVENT_IDS[len(history)]
                              if len(history) < len(EXPECTED_EVENT_IDS) else None),
                    stage="event_sequence")
    if len(transactions) != CONTINUOUS_EVENTS:
        report.fail("event_sequence",
                    f"{label} transactions.jsonl has {len(transactions)} rows, "
                    f"expected {CONTINUOUS_EVENTS}",
                    event_index=min(len(transactions) + 1, CONTINUOUS_EVENTS),
                    event_id=(EXPECTED_EVENT_IDS[len(transactions)]
                              if len(transactions) < len(EXPECTED_EVENT_IDS) else None),
                    stage="event_sequence")
    for position in range(min(len(history), len(transactions),
                              len(EXPECTED_EVENT_IDS))):
        expected_id = EXPECTED_EVENT_IDS[position]
        index = position + 1
        row = history[position]
        transaction = transactions[position]
        if not isinstance(row, Mapping):
            report.fail("event_sequence",
                        f"{label} state_history row {index} is not an object",
                        event_index=index, event_id=expected_id,
                        stage="event_sequence")
            continue
        if row.get("event_index") != index:
            report.fail("event_sequence",
                        f"{label} state_history event_index must be {index} "
                        f"(saw {row.get('event_index')!r})",
                        event_index=index, event_id=expected_id,
                        stage="event_sequence")
            continue
        if row.get("event_id") != expected_id:
            report.fail("event_sequence",
                        f"{label} state_history event {index} is "
                        f"{row.get('event_id')!r}, expected {expected_id!r}",
                        event_index=index, event_id=expected_id,
                        stage="event_sequence")
        if row.get("transaction") != transaction:
            difference = _first_difference(row.get("transaction"), transaction) or {}
            report.fail("event_sequence",
                        f"{label} state_history/transactions differ at "
                        f"{difference.get('path')} for event {index}",
                        event_index=index, event_id=expected_id,
                        path=difference.get("path"), stage="event_sequence")
        legacy = row.get("legacy_state_digest")
        if not _is_hex(legacy):
            report.fail("event_sequence",
                        f"{label} legacy_state_digest is invalid at event {index}",
                        event_index=index, event_id=expected_id,
                        stage="event_sequence")
        elif index == PARENT_STATE_INDEX and legacy != ctx["comparison"].get(
                "legacy_before_state"):
            report.fail("boundary",
                        f"{label} event_index {PARENT_STATE_INDEX} legacy digest "
                        "differs from the declared legacy_before_state",
                        event_index=index, event_id=expected_id,
                        stage="boundary")
        elif index == CONTINUOUS_EVENTS and legacy != ctx["comparison"].get(
                "legacy_after_state"):
            report.fail("final",
                        f"{label} final legacy digest differs from the declared "
                        "legacy_after_state",
                        event_index=index, event_id=expected_id, stage="final")


def _check_resume_sequence(report: Report, resume: Mapping) -> None:
    label = "resume"
    history = resume["history"]
    if not isinstance(history, list) or not history:
        report.fail("event_sequence",
                    f"{label} state_history is empty", event_index=CONTINUOUS_EVENTS,
                    event_id=EXPECTED_EVENT_IDS[-1], stage="event_sequence")
        return
    row = history[0]
    if not isinstance(row, Mapping):
        report.fail("event_sequence", f"{label} state_history row is not an object",
                    event_index=CONTINUOUS_EVENTS,
                    event_id=EXPECTED_EVENT_IDS[-1], stage="event_sequence")
        return
    if row.get("event_index") != CONTINUOUS_EVENTS:
        report.fail("event_sequence",
                    f"{label} event_index must be {CONTINUOUS_EVENTS}",
                    event_index=CONTINUOUS_EVENTS,
                    event_id=EXPECTED_EVENT_IDS[-1], stage="event_sequence")
    if row.get("event_id") != EXPECTED_EVENT_IDS[-1]:
        report.fail("event_sequence",
                    f"{label} event_id must be correction-0003",
                    event_index=CONTINUOUS_EVENTS,
                    event_id=EXPECTED_EVENT_IDS[-1], stage="event_sequence")
    manifest = resume["resume"]
    if row.get("transaction") != manifest.get("resumed_outcome"):
        difference = _first_difference(
            row.get("transaction"), manifest.get("resumed_outcome")) or {}
        report.fail("event_sequence",
                    f"{label} state_history transaction differs from "
                    f"resumed_outcome at {difference.get('path')}",
                    event_index=CONTINUOUS_EVENTS,
                    event_id=EXPECTED_EVENT_IDS[-1],
                    path=difference.get("path"), stage="event_sequence")
    if row.get("legacy_state_digest") != manifest.get("after_state"):
        report.fail("event_sequence",
                    f"{label} state_history legacy digest differs from after_state",
                    event_index=CONTINUOUS_EVENTS,
                    event_id=EXPECTED_EVENT_IDS[-1], stage="event_sequence")
    if isinstance(resume["after_state"], Mapping) \
            and row.get("canonical_state") != resume["after_state"]:
        difference = _first_difference(
            row.get("canonical_state"), resume["after_state"]) or {}
        report.fail("event_sequence",
                    f"{label} state_history canonical state differs from "
                    f"after_state.json at {difference.get('path')}",
                    event_index=CONTINUOUS_EVENTS,
                    event_id=EXPECTED_EVENT_IDS[-1],
                    path=difference.get("path"), stage="event_sequence")


def _compare_arms(report: Report, arms: Mapping[str, Mapping],
                  resume: Mapping, ctx: Mapping) -> None:
    a, b = arms.get("continuous-a"), arms.get("continuous-b")
    if a is None or b is None or resume is None:
        return
    protocol_body = a["protocol_body"]
    if a["protocol_body"] != b["protocol_body"]:
        difference = _first_difference(a["protocol_body"], b["protocol_body"]) or {}
        report.fail("protocol", "continuous arms carry different protocol.json "
                    f"bodies at {difference.get('path')}")
    if a["run"].get("world_uid") != b["run"].get("world_uid"):
        report.fail("process_reports", "continuous arms carry different world_uid")
    if a["run"].get("world_uid") != protocol_body.get("world_uid"):
        report.fail("process_reports",
                    "run world_uid differs from protocol.json world_uid")
    if a.get("audit") is not None and b.get("audit") is not None \
            and a["audit"] != b["audit"]:
        report.fail("theta_audit",
                    "continuous arms carry different parameter audits")
    if resume.get("audit") is not None and a.get("audit") is not None \
            and resume["audit"] != a["audit"]:
        report.fail("theta_audit",
                    "resume parameter audit differs from the continuous arms")

    history_a = a["history"]
    history_b = b["history"]
    if not isinstance(history_a, list) or not isinstance(history_b, list):
        return
    common = min(len(history_a), len(history_b), len(EXPECTED_EVENT_IDS))
    for position in range(common):
        index = position + 1
        event_id = EXPECTED_EVENT_IDS[position]
        row_a, row_b = history_a[position], history_b[position]
        if not isinstance(row_a, Mapping) or not isinstance(row_b, Mapping):
            continue
        _compare_outcome(report, row_a.get("transaction"),
                         row_b.get("transaction"), index, event_id,
                         "continuous-a/continuous-b", aa=True)
        state_a = row_a.get("canonical_state")
        state_b = row_b.get("canonical_state")
        if isinstance(state_a, Mapping) and isinstance(state_b, Mapping):
            _compare_canonical(report, state_a, state_b, "aa_canonical", index,
                               event_id, "continuous-a/continuous-b", aa=True)
        if row_a.get("legacy_state_digest") != row_b.get("legacy_state_digest"):
            report.fail("aa_canonical",
                        "continuous arms legacy digests differ",
                        event_index=index, event_id=event_id, stage="aa_canonical")

    final_a, final_b = a.get("final_state"), b.get("final_state")
    if isinstance(final_a, Mapping) and isinstance(final_b, Mapping):
        _compare_canonical(report, final_a, final_b, "aa_canonical",
                           CONTINUOUS_EVENTS, EXPECTED_EVENT_IDS[-1],
                           "continuous-a/continuous-b final_state", aa=True)
    for label, arm in (("continuous-a", a), ("continuous-b", b)):
        final_state = arm.get("final_state")
        history = arm["history"]
        if isinstance(final_state, Mapping) and history \
                and isinstance(history[-1], Mapping) \
                and history[-1].get("canonical_state") != final_state:
            difference = _first_difference(
                history[-1].get("canonical_state"), final_state) or {}
            report.fail("final",
                        f"{label} final_state.json differs from the last "
                        f"state_history row at {difference.get('path')}",
                        event_index=CONTINUOUS_EVENTS,
                        event_id=EXPECTED_EVENT_IDS[-1],
                        path=difference.get("path"), stage="final")

    # A/A P-only export equality is required; checkpoint bytes are not.
    if a.get("p_only_sha256") is not None and b.get("p_only_sha256") is not None \
            and a["p_only_sha256"] != b["p_only_sha256"]:
        report.fail("p_only",
                    "continuous arms export different P-only SHA-256 values")

    before = resume.get("before_state")
    after = resume.get("after_state")
    if isinstance(before, Mapping):
        for label, arm in (("continuous-a", a), ("continuous-b", b)):
            history = arm["history"]
            if len(history) < PARENT_STATE_INDEX:
                report.fail("boundary",
                            f"{label} has no event_index {PARENT_STATE_INDEX} state",
                            event_index=PARENT_STATE_INDEX,
                            event_id=EXPECTED_EVENT_IDS[PARENT_STATE_INDEX - 1],
                            path=f"state_history[{PARENT_STATE_INDEX - 1}]",
                            stage="boundary")
                continue
            row = history[PARENT_STATE_INDEX - 1]
            state = row.get("canonical_state") if isinstance(row, Mapping) else None
            if not isinstance(state, Mapping):
                report.fail("boundary",
                            f"{label} event_index {PARENT_STATE_INDEX} canonical "
                            "state is missing", event_index=PARENT_STATE_INDEX,
                            event_id=EXPECTED_EVENT_IDS[PARENT_STATE_INDEX - 1],
                            stage="boundary")
                continue
            _compare_canonical(report, state, before, "boundary",
                               PARENT_STATE_INDEX,
                               EXPECTED_EVENT_IDS[PARENT_STATE_INDEX - 1],
                               f"{label}/resume.before_state")
            if isinstance(row, Mapping) and row.get("legacy_state_digest") != \
                    resume["resume"].get("before_state"):
                report.fail("boundary",
                            f"{label} event_index {PARENT_STATE_INDEX} legacy digest "
                            "differs from resume.before_state",
                            event_index=PARENT_STATE_INDEX,
                            event_id=EXPECTED_EVENT_IDS[PARENT_STATE_INDEX - 1],
                            stage="boundary")
    if isinstance(after, Mapping):
        for label, arm in (("continuous-a", a), ("continuous-b", b)):
            final_state = arm.get("final_state")
            if not isinstance(final_state, Mapping):
                report.fail("final",
                            f"{label} final_state is missing for the resume "
                            "comparison", event_index=CONTINUOUS_EVENTS,
                            event_id=EXPECTED_EVENT_IDS[-1], stage="final")
                continue
            _compare_canonical(report, final_state, after, "final",
                               CONTINUOUS_EVENTS, EXPECTED_EVENT_IDS[-1],
                               f"{label}/resume.after_state")
    # The resumed event must itself match the uninterrupted event 28 outcome.
    resume_history = resume.get("history") or []
    if resume_history and isinstance(resume_history[0], Mapping):
        resumed_outcome = resume_history[0].get("transaction")
        for label, arm in (("continuous-a", a), ("continuous-b", b)):
            history = arm["history"]
            if len(history) >= CONTINUOUS_EVENTS:
                row = history[CONTINUOUS_EVENTS - 1]
                if isinstance(row, Mapping):
                    _compare_outcome(report, row.get("transaction"),
                                     resumed_outcome, CONTINUOUS_EVENTS,
                                     EXPECTED_EVENT_IDS[-1],
                                     f"{label}/resume")
    # Checkpoint byte equality is deliberately not a gate; only canonical
    # state and P-only export equality are.  The receipt reports the observed
    # sizes for the record.


def _verify_artifact_states(report: Report, arms: Mapping[str, Mapping],
                            resume: Mapping, ctx: Mapping) -> None:
    protocol_body = arms["continuous-a"]["protocol_body"]
    for label in ("continuous-a", "continuous-b"):
        arm = arms[label]
        history = arm["history"]
        for position, row in enumerate(history):
            if position >= len(EXPECTED_EVENT_IDS):
                break
            index = position + 1
            event_id = EXPECTED_EVENT_IDS[position]
            if not isinstance(row, Mapping):
                continue
            _validate_outcome(report, row.get("transaction"), index, event_id,
                              label, ctx)
            state = row.get("canonical_state")
            valid = _validate_canonical(
                report, state, label=label, event_index=index,
                expected_ids=EXPECTED_EVENT_IDS, ctx=ctx,
                protocol_body=protocol_body)
            if valid and isinstance(state, Mapping):
                _bind_outcome_to_state(report, state, row.get("transaction") or {},
                                       index, event_id, label)
        final_state = arm.get("final_state")
        if isinstance(final_state, Mapping):
            _validate_canonical(
                report, final_state, label=f"{label} final_state",
                event_index=CONTINUOUS_EVENTS,
                expected_ids=EXPECTED_EVENT_IDS, ctx=ctx,
                protocol_body=protocol_body)
    before = resume.get("before_state")
    if isinstance(before, Mapping):
        _validate_canonical(report, before, label="resume before_state",
                            event_index=PARENT_STATE_INDEX,
                            expected_ids=EXPECTED_EVENT_IDS, ctx=ctx,
                            protocol_body=protocol_body)
    after = resume.get("after_state")
    if isinstance(after, Mapping):
        _validate_canonical(report, after, label="resume after_state",
                            event_index=CONTINUOUS_EVENTS,
                            expected_ids=EXPECTED_EVENT_IDS, ctx=ctx,
                            protocol_body=protocol_body)
    history = resume.get("history") or []
    if history and isinstance(history[0], Mapping):
        _validate_outcome(report, history[0].get("transaction"),
                          CONTINUOUS_EVENTS, RESUME_EVENT_ID, "resume", ctx)
        state = history[0].get("canonical_state")
        valid = _validate_canonical(
            report, state, label="resume state_history",
            event_index=CONTINUOUS_EVENTS, expected_ids=EXPECTED_EVENT_IDS,
            ctx=ctx, protocol_body=protocol_body)
        if valid and isinstance(state, Mapping):
            _bind_outcome_to_state(
                report, state, history[0].get("transaction") or {},
                CONTINUOUS_EVENTS, RESUME_EVENT_ID, "resume")


def _record_path(value: Any, artifact_root: Path,
                 record_inputs: str) -> str | None:
    if value is None or record_inputs == "omit":
        return None
    path = Path(value)
    if record_inputs == "relative":
        try:
            return path.relative_to(artifact_root).as_posix()
        except ValueError:
            return str(path)
    return str(path)


def _finalize(report: Report, ctx: Mapping | None,
              arms: Mapping[str, Mapping], resume: Mapping | None,
              guards: Mapping[str, Mapping | None],
              protocol_path: Path, paths: Mapping[str, Path], *,
              source: Mapping, artifact_root: Path,
              record_inputs: str) -> dict:
    recorded_inputs = {
        "protocol": _record_path(protocol_path, artifact_root, record_inputs),
        "continuous_a": _record_path(paths.get("continuous-a"), artifact_root,
                                     record_inputs),
        "continuous_b": _record_path(paths.get("continuous-b"), artifact_root,
                                     record_inputs),
        "resume": _record_path(paths.get("resume"), artifact_root,
                               record_inputs),
        "guard_a": _record_path(paths.get("guard-a"), artifact_root,
                                record_inputs),
        "guard_b": _record_path(paths.get("guard-b"), artifact_root,
                                record_inputs),
        "guard_resume": _record_path(paths.get("guard-resume"), artifact_root,
                                     record_inputs),
    }
    recorded_guards = {}
    for name, entry in guards.items():
        if entry is None:
            recorded_guards[name] = None
            continue
        recorded = dict(entry)
        recorded["path"] = _record_path(entry.get("path"), artifact_root,
                                        record_inputs)
        recorded_guards[name] = recorded
    receipt = {
        "format": RECEIPT_FORMAT,
        "status": report.status,
        "exact_status": report.exact_status,
        "variation_status": report.variation_status,
        "resource_status": report.resource_status,
        "exact_only": True,
        "numeric_envelope": False,
        "aa_tolerance": 0,
        "in_process_reference_verdict": False,
        "failures": list(report.failures),
        "failure_count": len(report.failures),
        "earliest_divergence": report.earliest,
        "events": {
            "expected_sequence": list(EXPECTED_EVENT_IDS),
            "continuous_events": CONTINUOUS_EVENTS,
            "parent_events": PARENT_EVENTS,
            "aa_states_compared": report.aa_compared,
            "boundary_event": RESUME_EVENT_ID,
            "boundary_event_index": RESUME_EVENT_INDEX,
            "parent_state_index": PARENT_STATE_INDEX,
        },
        "guards": recorded_guards,
        "artifacts": {},
        "implementation_source": {
            "mode": source.get("mode"),
            "snapshot_sha256": source.get("snapshot_sha256"),
            "receipt_sha256": source.get("receipt_sha256"),
            "implementation_pins": (len(source["paths"])
                                    if isinstance(source.get("paths"),
                                                  Mapping) else 0),
        },
    }
    if record_inputs != "omit":
        receipt["inputs"] = recorded_inputs
    if ctx is not None:
        receipt["protocol"] = {
            "config_sha256": ctx["config_sha256"],
            "protocol_sha256": ctx["protocol_sha256"],
            "episodes_manifest_sha256": ctx["manifest_sha256"],
            "implementation_sha256": dict(source.get("paths") or {}),
        }
        receipt["parent_checkpoint"] = {
            "declared_path": ctx["parent"]["declared_path"],
            "sha256": ctx["parent"]["sha256"],
            "bytes": ctx["parent"]["bytes"],
            "declared_sha256": ctx["parent"]["declared_sha256"],
            "declared_bytes": ctx["parent"]["declared_bytes"],
            "retained": ctx["parent"]["path"].exists(),
        }
    for label in ("continuous-a", "continuous-b"):
        arm = arms.get(label)
        if arm is None:
            continue
        receipt["artifacts"][label] = {
            "path": _record_path(arm["directory"], artifact_root,
                                 record_inputs),
            "run_sha256": arm.get("run_sha256"),
            "checkpoint_sha256": arm.get("checkpoint_sha256"),
            "checkpoint_bytes": arm.get("checkpoint_bytes"),
            "p_only_sha256": arm.get("p_only_sha256"),
            "p_only_bytes": arm.get("p_only_bytes"),
            "artifact_bytes": arm.get("artifact_bytes"),
            "canonical_state_sha256": (
                arm["final_state"].get("sha256")
                if isinstance(arm.get("final_state"), Mapping) else None),
        }
    if resume is not None:
        before = resume.get("before_state")
        after = resume.get("after_state")
        receipt["artifacts"]["resume"] = {
            "path": _record_path(resume["directory"], artifact_root,
                                 record_inputs),
            "run_sha256": resume.get("run_sha256"),
            "resume_sha256": resume.get("resume_sha256"),
            "before_canonical_sha256": (
                before.get("sha256") if isinstance(before, Mapping) else None),
            "after_canonical_sha256": (
                after.get("sha256") if isinstance(after, Mapping) else None),
            "checkpoint_sha256": (
                resume["resume"].get("checkpoint_sha256")
                if isinstance(resume.get("resume"), Mapping) else None),
            "checkpoint_bytes": (
                resume["resume"].get("checkpoint_bytes")
                if isinstance(resume.get("resume"), Mapping) else None),
            "artifact_bytes": resume.get("artifact_bytes"),
        }
    if arms.get("continuous-a") and arms.get("continuous-b"):
        a, b = arms["continuous-a"], arms["continuous-b"]
        receipt["comparisons"] = {
            "p_only_equal": (a.get("p_only_sha256") is not None
                             and a.get("p_only_sha256") == b.get("p_only_sha256")),
            "checkpoint_bytes_equal": (a.get("checkpoint_bytes") is not None
                                       and a.get("checkpoint_bytes")
                                       == b.get("checkpoint_bytes")),
            "boundary": {
                "event": RESUME_EVENT_ID,
                "event_index": RESUME_EVENT_INDEX,
                "parent_state_index": PARENT_STATE_INDEX,
                "continuous_a_before_sha256": _history_state_sha(
                    a.get("history"), PARENT_STATE_INDEX),
                "continuous_b_before_sha256": _history_state_sha(
                    b.get("history"), PARENT_STATE_INDEX),
                "resume_before_sha256": (
                    resume["before_state"].get("sha256")
                    if resume is not None
                    and isinstance(resume.get("before_state"), Mapping) else None),
                "continuous_a_after_sha256": (
                    a["final_state"].get("sha256")
                    if isinstance(a.get("final_state"), Mapping) else None),
                "continuous_b_after_sha256": (
                    b["final_state"].get("sha256")
                    if isinstance(b.get("final_state"), Mapping) else None),
                "resume_after_sha256": (
                    resume["after_state"].get("sha256")
                    if resume is not None
                    and isinstance(resume.get("after_state"), Mapping) else None),
            },
        }
    receipt["claims"] = {
        "promotion": False,
        "heldout_opened": False,
        "numeric_envelope": False,
        "exact_only": True,
        "in_process_reference_verdict": False,
    }
    return receipt


def _history_state_sha(history: Any, index: int) -> str | None:
    if not isinstance(history, list) or len(history) < index:
        return None
    row = history[index - 1]
    if not isinstance(row, Mapping):
        return None
    state = row.get("canonical_state")
    return state.get("sha256") if isinstance(state, Mapping) else None


def compare_artifacts(*, protocol: str | Path, continuous_a: str | Path,
                      continuous_b: str | Path, resume: str | Path,
                      guard_a: str | Path, guard_b: str | Path,
                      guard_resume: str | Path,
                      artifact_root: str | Path = ROOT,
                      source_root: str | Path | None = None,
                      mode: str = "historical",
                      current_receipt: str | Path | None = None,
                      guard_binding: str | Path | None = None,
                      record_inputs: str = "absolute") -> dict:
    """Compare declared artifacts and return a deterministic receipt.

    Historical mode authenticates the artifact implementation maps against the
    immutable source snapshot; live drift cannot change the verdict.  Current
    mode requires a reviewed current receipt and authenticates the live tree.
    Relative inputs resolve against ``artifact_root`` so relocated replay does
    not depend on the original checkout path.
    """
    if record_inputs not in ("absolute", "relative", "omit"):
        raise ValueError("record_inputs must be absolute, relative or omit")
    artifact_root = Path(artifact_root)
    report = Report()
    paths = {
        "continuous-a": _path(continuous_a, artifact_root),
        "continuous-b": _path(continuous_b, artifact_root),
        "resume": _path(resume, artifact_root),
        "guard-a": _path(guard_a, artifact_root),
        "guard-b": _path(guard_b, artifact_root),
        "guard-resume": _path(guard_resume, artifact_root),
    }
    if paths["continuous-a"].resolve() == paths["continuous-b"].resolve():
        report.fail("process_reports",
                    "the two uninterrupted arms must be distinct directories")
    if paths["resume"].resolve() in (paths["continuous-a"].resolve(),
                                     paths["continuous-b"].resolve()):
        report.fail("process_reports",
                    "the resume directory must be distinct from the arms")
    guard_paths = [paths[name].resolve()
                   for name in ("guard-a", "guard-b", "guard-resume")]
    if len(set(guard_paths)) != 3:
        report.fail("guard",
                    "each process requires a distinct guard receipt",
                    resource=True, invalid=True)
    protocol_path = _path(protocol, artifact_root)
    ctx = _protocol_context(protocol_path, report, artifact_root)
    guards = {}
    source = {"mode": mode, "snapshot_sha256": None,
              "receipt_sha256": None, "paths": None}
    binding = None
    if ctx is None:
        # Without the declared runtime limits no guard can be judged.
        report.resource_invalid = True
        if mode == "current" and current_receipt is None:
            report.fail("implementation",
                        "Current implementation verification requires a "
                        "reviewed current receipt")
    else:
        ctx["config_path"] = protocol_path
        source, binding = _resolve_implementation_source(
            mode, ctx, artifact_root=artifact_root,
            protocol_path=protocol_path,
            source_root=(Path(source_root) if source_root is not None else None),
            current_receipt=(Path(current_receipt)
                             if current_receipt is not None else None),
            guard_binding=(Path(guard_binding)
                           if guard_binding is not None else None),
            report=report)
        ctx["expected_implementation"] = source.get("paths")
        ctx["implementation_source_label"] = (
            "historical source snapshot" if mode == "historical"
            else "reviewed current receipt")
        guard_artifacts = {"guard-a": "continuous-a",
                           "guard-b": "continuous-b",
                           "guard-resume": "resume"}
        for name in ("guard-a", "guard-b", "guard-resume"):
            guards[name] = _validate_guard(
                report, paths[name], name, ctx["runtime"], binding,
                expected_output_name=paths[guard_artifacts[name]].name)
    arms: dict[str, Mapping] = {}
    resume_arm = None
    if ctx is not None:
        for label, name in (("continuous-a", "continuous-a"),
                            ("continuous-b", "continuous-b")):
            directory = paths[name]
            if not directory.is_dir():
                report.fail("process_reports",
                            f"{label} output directory is missing")
                continue
            arm = _load_continuous(report, directory, label, ctx)
            if arm is not None:
                arms[label] = arm
        if paths["resume"].is_dir():
            resume_arm = _load_resume(report, paths["resume"], ctx)
        else:
            report.fail("process_reports", "resume output directory is missing")
        # Cross-check that every report was produced by today's implementation.
        maps = []
        for arm in arms.values():
            if isinstance(arm.get("run"), Mapping):
                maps.append(arm["run"].get("implementation_sha256"))
        if resume_arm is not None:
            if isinstance(resume_arm.get("run"), Mapping):
                maps.append(resume_arm["run"].get("implementation_sha256"))
            if isinstance(resume_arm.get("resume"), Mapping):
                maps.append(resume_arm["resume"].get("implementation_sha256"))
        if len(maps) >= 2 and any(mapping != maps[0] for mapping in maps[1:]):
            report.fail("implementation",
                        "process implementation maps differ between reports")
        for arm in arms.values():
            _check_continuous_sequence(report, arm, ctx)
        if resume_arm is not None:
            _check_resume_sequence(report, resume_arm)
        if len(arms) == 2 and resume_arm is not None:
            _verify_artifact_states(report, arms, resume_arm, ctx)
            _compare_arms(report, arms, resume_arm, ctx)
        elif len(arms) == 2:
            report.fail("boundary", "resume artifacts are required for A/B",
                        event_index=PARENT_STATE_INDEX,
                        event_id=EXPECTED_EVENT_IDS[PARENT_STATE_INDEX - 1],
                        stage="boundary")
    return _finalize(report, ctx, arms, resume_arm, guards, protocol_path,
                     paths, source=source, artifact_root=artifact_root,
                     record_inputs=record_inputs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--continuous-a", type=Path, required=True)
    parser.add_argument("--continuous-b", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--guard-a", type=Path, required=True)
    parser.add_argument("--guard-b", type=Path, required=True)
    parser.add_argument("--guard-resume", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=ROOT,
                        help="root for relative artifact/protocol inputs")
    parser.add_argument("--source-root", type=Path, default=None,
                        help="historical source snapshot manifest or directory")
    parser.add_argument("--mode", choices=SOURCE_MODES, default="historical")
    parser.add_argument("--current-receipt", type=Path, default=None,
                        help="reviewed current receipt (required by --mode current)")
    parser.add_argument("--guard-binding", type=Path, default=None,
                        help="declared launcher/child guard binding")
    parser.add_argument("--record-inputs",
                        choices=("absolute", "relative", "omit"),
                        default="absolute",
                        help="how to record input/artifact paths")
    args = parser.parse_args()
    try:
        if args.output.exists():
            parser.error("comparison receipt path already exists")
        receipt = compare_artifacts(
            protocol=args.protocol, continuous_a=args.continuous_a,
            continuous_b=args.continuous_b, resume=args.resume,
            guard_a=args.guard_a, guard_b=args.guard_b,
            guard_resume=args.guard_resume,
            artifact_root=args.artifact_root, source_root=args.source_root,
            mode=args.mode, current_receipt=args.current_receipt,
            guard_binding=args.guard_binding,
            record_inputs=args.record_inputs)
    except (OSError, ValueError, KeyError, TypeError) as error:
        receipt = {
            "format": RECEIPT_FORMAT, "status": "failed",
            "exact_status": "failed", "variation_status": "not_measured",
            "resource_status": "invalid",
            "failures": [f"internal_error: {error}"], "failure_count": 1,
            "earliest_divergence": None,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": receipt.get("status"),
        "exact_status": receipt.get("exact_status"),
        "variation_status": receipt.get("variation_status"),
        "resource_status": receipt.get("resource_status"),
        "earliest_divergence": receipt.get("earliest_divergence"),
        "failure_count": receipt.get("failure_count"),
        "output": str(args.output),
    }, sort_keys=True))
    return 0 if receipt.get("status") == "exact_match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
