"""Explicit current and historical vision-curriculum score validation.

Current v1 reports always require ``reload_report_sha256``.  A small set of
archived v1 development reports predates that field; they are readable only
through ``historical-v1-supplemental-lineage``.  That route verifies the
original score bytes, run, reload and supplemental lineage before adding the
missing hash to an in-memory copy.  It never rewrites historical evidence or
weakens :func:`validate_score_report` for new reports.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .contracts import fingerprint
from .vision_curriculum import (
    IMPLEMENTATION_FILES,
    SCORE_REPORT_FORMAT,
    VisionCurriculumError,
    load_protocol,
    load_run_report,
    sha256_file,
    validate_protocol,
    validate_reload_report,
    validate_run_report,
    validate_score_report,
)

CURRENT_V1 = "current-v1"
HISTORICAL_V1 = "historical-v1-supplemental-lineage"
VALIDATION_VERSIONS = (CURRENT_V1, HISTORICAL_V1)
REPLAY_FORMAT = "replica.vision_curriculum.score_validation_replay.v1"
LINEAGE_FORMAT = "replica.vision_curriculum.supplemental_lineage.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VisionCurriculumError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def _historical_protocol(protocol_path: Path,
                         lineage: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate archived structure without changing its original pin inventory."""
    protocol = _load(protocol_path)
    pins = protocol.get("implementation_sha256")
    _require(isinstance(pins, dict), "Historical protocol implementation pins are missing")
    expected = set(IMPLEMENTATION_FILES)
    actual = set(pins)
    missing = expected - actual
    _require(not actual - expected and missing <= {"scripts/run_capacity_preflight.py"},
             "Historical protocol implementation inventory is not a recognized v1 layout")
    normalized = copy.deepcopy(protocol)
    if missing:
        launch = lineage.get("native_launch", {})
        _require(launch.get("guard_entry_point") == "run_capacity_preflight.py"
                 and launch.get("entry_point") == "run_vision_curriculum.py",
                 "Lineage does not bind the omitted historical guard launcher")
        guard_hash = launch.get("guard_sha256")
        _require(isinstance(guard_hash, str) and len(guard_hash) == 64,
                 "Historical guard launcher hash is missing")
        normalized["implementation_sha256"] = {
            **pins, "scripts/run_capacity_preflight.py": guard_hash}
    validate_protocol(normalized)
    return protocol, sorted(missing)


def _historical_run(run: dict[str, Any], outer_path: Path,
                    protocol: dict[str, Any], protocol_path: Path,
                    run_report_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind an archived outer guard without editing the original run report."""
    outer = _load(outer_path)
    command = outer.get("command")
    _require(isinstance(command, list) and "train" in command,
             "Outer guard is not a curriculum training launch")

    def option(name):
        values = [command[index + 1] for index, value in enumerate(command[:-1])
                  if value == name]
        _require(len(values) == 1, f"Outer guard requires exactly one {name}")
        return values[0]

    _require(option("--arm") == run["arm"] and option("--device") == "mps",
             "Outer guard arm or device differs")
    _require(Path(option("--protocol")).resolve() == protocol_path.resolve()
             and Path(option("--output")).resolve() == run_report_path.parent.resolve(),
             "Outer guard protocol or run output differs")
    _require(outer.get("status") == "completed_within_declared_sampled_limits"
             and outer.get("child_exit_code") == 0,
             "Outer guard did not complete successfully")
    guard = outer.get("guard", {})
    policy = guard.get("policy", {})
    expected = protocol["resource_policy"]
    for key in ("driver_limit_bytes", "sustained_swap_growth_bytes",
                "minimum_host_free_percent", "sustained_host_samples"):
        outer_key = {"sustained_swap_growth_bytes": "swap_growth_limit_bytes",
                     "minimum_host_free_percent": "minimum_free_percent"}.get(key, key)
        _require(policy.get(outer_key) == expected[key],
                 f"Outer guard resource policy differs: {key}")
    _require(guard.get("stop_reason") is None,
             "Outer guard records a stop reason")
    outer_peak = guard.get("peak_driver_bytes")
    _require(outer_peak == run["costs"].get("peak_driver_bytes"),
             "Outer guard peak differs from run costs")
    internal = run.get("implementation_runtime", {}).get("guard", {})
    internal_peak = internal.get("driver_peak", internal.get("peak_driver_bytes"))
    _require(isinstance(internal_peak, int) and internal_peak <= outer_peak,
             "Outer guard peak is below the internal guard peak")
    events = internal.get("events", [])
    sampled = [event.get("mps_driver_bytes") for event in events
               if event.get("mps_driver_bytes") is not None]
    _require(not sampled or max(sampled) <= outer_peak,
             "Internal guard sample exceeds the outer peak")
    normalized = copy.deepcopy(run)
    normalized_guard = normalized["implementation_runtime"]["guard"]
    normalized_guard["peak_driver_bytes"] = outer_peak
    if "driver_peak" in normalized_guard:
        normalized_guard["driver_peak"] = outer_peak
    return normalized, {
        "outer_guard_report_sha256": sha256_file(outer_path),
        "internal_peak_driver_bytes": internal_peak,
        "outer_peak_driver_bytes": outer_peak,
        "outer_guard_policy_verified": True,
        "outer_guard_command_verified": True,
    }


def replay_score_report(*, score_path: str | Path,
                        protocol_path: str | Path,
                        run_report_path: str | Path,
                        reload_report_path: str | Path,
                        validation_version: str,
                        lineage_path: str | Path | None = None,
                        outer_guard_report_path: str | Path | None = None
                        ) -> dict[str, Any]:
    """Validate one score report through an explicit versioned replay route."""
    _require(validation_version in VALIDATION_VERSIONS,
             f"Unknown score validation version: {validation_version}")
    score_path = Path(score_path)
    protocol_path = Path(protocol_path)
    run_report_path = Path(run_report_path)
    reload_report_path = Path(reload_report_path)
    original_score_bytes = score_path.read_bytes()
    score_hash = sha256_file(score_path)
    protocol_hash = sha256_file(protocol_path)
    run_hash = sha256_file(run_report_path)
    reload_hash = sha256_file(reload_report_path)

    lineage = None
    original_lineage_bytes = None
    if validation_version == HISTORICAL_V1:
        _require(lineage_path is not None,
                 "Historical v1 replay requires supplemental lineage")
        lineage_path = Path(lineage_path)
        original_lineage_bytes = lineage_path.read_bytes()
        lineage = _load(lineage_path)
        _require(lineage.get("format") == LINEAGE_FORMAT,
                 "Unknown supplemental lineage format")
        protocol, protocol_inventory_overlay = _historical_protocol(
            protocol_path, lineage)
    else:
        protocol = load_protocol(protocol_path)
        protocol_inventory_overlay = []
    run = load_run_report(run_report_path)
    reload_report = _load(reload_report_path)
    score = _load(score_path)
    _require(score.get("format") == SCORE_REPORT_FORMAT,
             "Historical replay accepts only vision-curriculum score v1")
    _require(score.get("protocol_sha256") == protocol_hash
             and run.get("protocol_sha256") == protocol_hash
             and reload_report.get("protocol_sha256") == protocol_hash,
             "Score, run or reload belongs to another protocol")
    _require(score.get("run_report_sha256") == run_hash
             and reload_report.get("run_report_sha256") == run_hash,
             "Score or reload belongs to another run")
    _require(score.get("arm") == run.get("arm") == reload_report.get("arm"),
             "Score, run and reload arms differ")
    _require(score.get("data_pins") == protocol["data"]["manifest"],
             "Score data pins differ from the protocol")
    if validation_version == HISTORICAL_V1:
        _require(outer_guard_report_path is not None,
                 "Historical v1 replay requires its per-arm outer guard")
        outer_guard_report_path = Path(outer_guard_report_path)
        run_for_validation, guard_overlay = _historical_run(
            run, outer_guard_report_path, protocol, protocol_path, run_report_path)
    else:
        _require(outer_guard_report_path is None,
                 "Current v1 validation does not accept an outer-guard overlay")
        run_for_validation = run
        guard_overlay = None
    validation = validate_run_report(run_for_validation, protocol)
    _require(score["engineering"]["pass"] == validation["engineering_pass"]
             and score["engineering"]["checks"] == validation["checks"]
             and score["engineering"]["failures"] == validation["engineering_failures"],
             "Score engineering evidence differs from the validated run")
    reload_for_validation = copy.deepcopy(reload_report)
    reload_payload_overlay = (validation_version == HISTORICAL_V1
                              and "payload_sha256" not in reload_for_validation)
    if reload_payload_overlay:
        reload_for_validation["payload_sha256"] = fingerprint(reload_for_validation)
    validate_reload_report(reload_for_validation, run, protocol)

    normalized = copy.deepcopy(score)
    lineage_hash = sha256_file(lineage_path) if lineage_path is not None else None
    if validation_version == CURRENT_V1:
        _require(lineage_path is None,
                 "Current v1 validation does not accept a historical lineage overlay")
        _require(score.get("reload_report_sha256") == reload_hash,
                 "Current v1 score reload hash is missing or differs")
    else:
        _require(lineage_path is not None,
                 "Historical v1 replay requires supplemental lineage")
        _require("reload_report_sha256" not in score,
                 "Historical overlay is only for v1 reports lacking the reload hash")
        _require(lineage_path.read_bytes() == original_lineage_bytes,
                 "Historical lineage changed during replay")
        arm = score["arm"]
        run_binding = lineage.get("run_reports", {}).get(arm)
        reload_binding = lineage.get("fresh_process_reload", {}).get(arm)
        _require(isinstance(run_binding, dict) and isinstance(reload_binding, dict),
                 "Supplemental lineage does not cover this arm")
        _require(lineage.get("protocol_sha256") == protocol_hash
                 and lineage.get("manifest_sha256")
                 == protocol["data"]["manifest"]["sha256"],
                 "Supplemental lineage protocol or data manifest differs")
        _require(run_binding == {"run_report_sha256": run_hash,
                                 "score_report_sha256": score_hash},
                 "Supplemental lineage run/score byte binding differs")
        _require(reload_binding.get("independent_reload_report_sha256") == reload_hash
                 and reload_binding.get("run_local_copy_sha256") == reload_hash
                 and reload_binding.get("copies_byte_identical") is True,
                 "Supplemental lineage reload byte binding differs")
        normalized["reload_report_sha256"] = reload_hash
        _require(lineage_path.read_bytes() == original_lineage_bytes,
                 "Historical lineage changed during replay")

    validate_score_report(normalized, protocol)
    _require(score_path.read_bytes() == original_score_bytes,
             "Historical score report changed during replay")
    return {
        "format": REPLAY_FORMAT,
        "validation_version": validation_version,
        "status": "validated",
        "arm": normalized["arm"],
        "protocol_sha256": protocol_hash,
        "run_report_sha256": run_hash,
        "reload_report_sha256": reload_hash,
        "original_score_report_sha256": score_hash,
        "supplemental_lineage_sha256": lineage_hash,
        "guard_overlay": guard_overlay,
        "reload_payload_sha256_overlay": reload_payload_overlay,
        "protocol_inventory_overlay": protocol_inventory_overlay,
        "normalized_score_payload_fingerprint": fingerprint(normalized),
        "engineering_pass": normalized["engineering"]["pass"],
        "learned_quality_pass": normalized["learned_quality"]["pass"],
        "eligibility": copy.deepcopy(normalized["eligibility"]),
        "claims": copy.deepcopy(normalized["claims"]),
        "historical_score_modified": False,
        "historical_run_modified": False,
        "supplemental_lineage_modified": False,
        "native_execution": False,
        "route": ("strict current v1 validation" if validation_version == CURRENT_V1
                  else "archived v1 plus authenticated reload lineage; in-memory normalization only"),
    }
