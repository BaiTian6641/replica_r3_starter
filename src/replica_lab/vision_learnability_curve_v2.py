"""Measured accounting and readout audits for the vision learning curve.

This module is the corrected companion to the preserved v1 curve receipt.  It
does not rewrite the v1 module, its config, its tests or the attempt04 receipt.
Instead it:

* validates v2 accounting declarations with explicit units, per-readout and
  cumulative call-time budgets;
* validates measured readouts against the physical phase sum, prefill and
  teacher-forced call counts, record-level generation lengths and the
  visual-position/cache/projector identities;
* separates EOS-only responses from whitespace-only decoded text and audits
  response masks, caption/answer target equality and first-token predictions;
* records projector output scale and correct/sibling/shuffled control feature
  distinguishability when projected tensors are available;
* preserves the v1 verdict ``curve_no_optimization_progress`` and refuses any
  development, held-out, V0, promotion or retry implication;
* predeclares exactly one bounded training-only step-1 reproduction and
  validates its fail-closed authorization contract.

No function in this module loads a model or executes a native forward.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import vision_forward_accounting as fwd
from . import vision_learnability_curve as curve_v1
from . import vision_learnability_diagnostic as learn
from .vision_collapse_diagnostic import CONTROL_NAMES


VERSION = "replica.vision_learnability_curve.v2"
CURVE_MODE = "curve_v2"
STEP1_MODE = "curve_step1"
CURVE_CONFIG_FORMAT = "replica.vision_learnability_curve_config.v2"
STEP1_CONFIG_FORMAT = "replica.vision_learnability_step1_config.v1"
READOUT_FORMAT = "replica.vision_learnability_curve.readout.v2"
STEP1_READOUT_FORMAT = "replica.vision_learnability_step1.readout.v1"
CORRECTION_FORMAT = fwd.HISTORICAL_FORMAT
REPORT_FORMAT = "replica.vision_learnability_curve.report.v2"
STEP1_REPORT_FORMAT = "replica.vision_learnability_step1.report.v1"
DERIVED_STEP1_FORMAT = "replica.vision_learnability_step1.derived_receipt.v1"
DERIVED_STEP1_CLASS = "derived_validator_correction"
AUTHORIZATION_FORMAT = "replica.vision_learnability_step1.authorization.v1"

V1_VERDICT = "curve_no_optimization_progress"
V1_SUMMARY = "reports/vision_learnability_curve_v1_20261026/summary.json"
V1_RUN = "runs/vision_learnability_curve_v1_20261026_attempt04"
V1_CONFIG = "configs/vision_learnability_curve_v1.json"

HISTORICAL_PREFIXES = (0, 1, 2, 4, 7)
HISTORICAL_GENERATION_LENGTHS = (1362, 1442, 1396, 721, 324)
HISTORICAL_RECORD_TOKEN_SUMS = (1362, 1442, 1396, 721, 324)
HISTORICAL_SCORING_PASSES = (1852, 1932, 1886, 1211, 814)
HISTORICAL_TOTAL_SCORING_PASSES = 7695
HISTORICAL_PREPARES = 1470
HISTORICAL_TRAINING_PROBE_EVENTS = 60
HISTORICAL_GRAND_TOTAL = 7755
HISTORICAL_AVAILABLE = 98
HISTORICAL_LOGICAL_OPERATIONS = 1470

STEP1_NAME = "vision_learnability_step1_reproduction_v1"
STEP1_SEED = 29
STEP1_LR = 0.0003
STEP1_WEIGHT_DECAY = 0.01
STEP1_INIT_SHA256 = (
    "511ba5c7648851424649962467972f0e926303d64873dfce08ecd3de299314b9")
STEP1_PREFIX1_SHA256 = (
    "e6f1d25e968f0ee9914d1e0c42f92a8f2db2c0442109098bc322b8601de46cf4")

STEP1_ANCHORS = {
    "prefix0_teacher_forced_loss": {
        "answer": 10.704498972211566,
        "caption": 9.773420969645182,
    },
    "prefix1_teacher_forced_loss": {
        "answer": 11.74190446308681,
        "caption": 13.448152462641398,
    },
    "prefix0_projector_sha256": STEP1_INIT_SHA256,
    "prefix1_projector_sha256": STEP1_PREFIX1_SHA256,
    "first_update_grad_norm": 345.9010314941406,
    "first_update_projector_norm": 1.0185893941275908,
    "teacher_forced_abs_tolerance": 1e-5,
    "repeat_abs_tolerance": 1e-6,
}

STEP1_EXPECTED_WORK = {
    "decoder_forward_calls": 5644,
    "generated_tokens": 4166,
    "prepare_sense_passes": 882,
    "explicit_logits_forwards": 588,
    "update_forwards": 8,
    "backward_calls": 4,
    "optimizer_steps": 1,
    "readouts": 3,
    "generation_lengths_by_readout": [1362, 1362, 1442],
    "declared_readout_decoder_passes": [1852, 1852, 1932],
    "visual_observation_calls": 220,
    "visual_encoder_calls_max": 220,
    "visual_encoder_cache_misses_max": 220,
    "projector_passes_max": 5644,
}

STEP1_STATUS_SUPPORTED = "step1_reproduction_supported"
STEP1_STATUS_REFUTED = "step1_reproduction_refuted"
STEP1_STATUS_NONDETERMINISTIC = "readout_nondeterminism_detected"
STEP1_STATUS_ENGINEERING = "stopped_engineering_failure"
STEP1_STATUSES = (
    STEP1_STATUS_SUPPORTED,
    STEP1_STATUS_REFUTED,
    STEP1_STATUS_NONDETERMINISTIC,
    STEP1_STATUS_ENGINEERING,
)

NEW_FILE_PINS = (
    "src/replica_lab/vision_forward_accounting.py",
    "src/replica_lab/vision_learnability_curve_v2.py",
    "scripts/run_vision_learnability_curve_v2.py",
)

CURVE_KEYS = {
    "format", "version", "mode", "name", "declared_at_utc", "purpose",
    "prepared", "source", "precondition", "schedule", "initialization",
    "optimizer", "probes", "readouts", "decision", "access", "v0_gate",
    "answer_max_new_tokens", "caption_max_new_tokens", "stop_tokens",
    "target_eos_token", "controls", "prompt_contract", "accounting",
    "expectations", "limits", "policy", "implementation_pins",
    "authorization", "output", "report_root", "claims", "limitations",
}
STEP1_KEYS = {
    "format", "version", "mode", "name", "declared_at_utc", "purpose",
    "prepared", "source", "precondition", "schedule", "initialization",
    "optimizer", "experiment", "readouts", "historical_anchors_as_hypotheses",
    "nondeterminism_stop", "expected_measured_work", "accounting", "limits",
    "policy", "prompt_contract", "implementation_pins", "authorization",
    "statuses", "claims", "limitations", "output", "report_root",
}

AUTHORIZATION_KEYS = {
    "native_execution_authorized", "requires_protocol_commit",
    "protocol_commit", "receipt_path", "receipt_format", "required_status",
}
READOUT_COUNT_KEYS = {
    "records", "available", "placeholders", "generations",
    "prefill_forwards", "teacher_forced_cells", "generated_tokens",
}


def require(condition: Any, message: str) -> None:
    learn.require(condition, message)


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]


def _require_sha256(value: Any, name: str) -> None:
    require(isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value),
            f"{name} is not a SHA-256 digest")


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


# ---------------------------------------------------------------------------
# Config loading and validation
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    require(isinstance(payload, Mapping), "Config must be a JSON object")
    mode = payload.get("mode")
    if mode == CURVE_MODE:
        return validate_curve_config(payload)
    if mode == STEP1_MODE:
        return validate_step1_config(payload)
    raise learn.LearnabilityDiagnosticError(
        f"Unknown vision v2 config mode: {mode!r}")


def _validate_authorization(authorization: Mapping[str, Any]) -> None:
    require(isinstance(authorization, Mapping)
            and set(authorization) == AUTHORIZATION_KEYS,
            "Authorization fields differ")
    require(authorization["native_execution_authorized"] is False,
            "A predeclared protocol must not embed native authorization")
    require(authorization["requires_protocol_commit"] is True,
            "A protocol commit is required before native execution")
    require(authorization["protocol_commit"] is None,
            "The predeclared protocol has no committed identity yet")
    require(isinstance(authorization["receipt_path"], str)
            and authorization["receipt_path"].startswith("reports/"),
            "An authorization receipt path is required")
    require(authorization["receipt_format"] == AUTHORIZATION_FORMAT,
            "Authorization receipt format differs")
    require(authorization["required_status"] == "authorized",
            "Authorization receipt status differs")


def _validate_implementation_pins(pins: Mapping[str, Any]) -> None:
    require(isinstance(pins, Mapping), "Implementation pins are missing")
    require(set(pins) == {"prepared_protocol_pins_checked", "mode",
                          "reason", "migrations", "new_files"},
            "Implementation pin fields differ")
    require(pins["prepared_protocol_pins_checked"] is True
            and pins["mode"] == "explicit_read_only_migration",
            "Explicit read-only implementation migrations are required")
    require(isinstance(pins["reason"], str) and pins["reason"].strip(),
            "Implementation migration reason is missing")
    migrations = pins["migrations"]
    require(isinstance(migrations, Mapping)
            and set(migrations) == set(curve_v1.MIGRATION_PATHS),
            "The exact prepared04 implementation migrations are required")
    for relative, expected in curve_v1.EXPECTED_MIGRATIONS.items():
        actual = migrations[relative]
        require(isinstance(actual, Mapping)
                and set(actual) == {"parent_sha256", "current_sha256",
                                    "reason"},
                f"Malformed migration: {relative}")
        require(actual["parent_sha256"] == expected[0],
                f"Migration parent differs: {relative}")
        require(actual["current_sha256"] == expected[1],
                f"Migration current hash differs: {relative}")
        _require_sha256(actual["parent_sha256"], "parent_sha256")
        _require_sha256(actual["current_sha256"], "current_sha256")
        require(isinstance(actual["reason"], str)
                and actual["reason"].strip(),
                f"Migration lacks a reason: {relative}")
    new_files = pins["new_files"]
    require(isinstance(new_files, Mapping)
            and set(new_files) == set(NEW_FILE_PINS),
            "New-file implementation pins differ")
    for relative, digest in new_files.items():
        _require_sha256(digest, relative)


def _validate_readout_contract(readouts: Mapping[str, Any]) -> None:
    require(isinstance(readouts, Mapping), "Readout contract is missing")
    require(readouts.get("format") == READOUT_FORMAT,
            "Readout contract format differs")
    require(readouts.get("prefixes") == [0, 1, 2, 4, 7],
            "Readout prefixes differ")
    require(readouts.get("available_per_readout") == HISTORICAL_AVAILABLE
            and readouts.get("records_per_readout") == 104
            and readouts.get("placeholders_per_readout") == 6,
            "Readout matrix counts differ")
    require(readouts.get("teacher_forced_roles") == ["answer", "caption"],
            "Teacher-forced roles differ")
    require(readouts.get("empty_text_policy")
            == "separate_eos_only_from_whitespace_only",
            "Empty-text policy differs")
    require(readouts.get("logical_operations_per_readout") == 294,
            "Logical scoring operations differ")
    require(readouts.get("physical_units_required") is True,
            "Measured physical units are required")
    require(readouts.get("endpoint_gates_decision_input") is False,
            "Endpoint gates cannot decide the curve")


def _validate_expectations(expected: Mapping[str, Any]) -> None:
    require(isinstance(expected, Mapping), "Expectations are missing")
    require(expected.get("rows") == 26
            and expected.get("available_per_readout") == 98
            and expected.get("records_per_readout") == 104
            and expected.get("placeholders_per_readout") == 6,
            "Expectation matrix differs")
    require(expected.get("prepares_per_readout") == 294
            and expected.get("explicit_logits_per_readout") == 196
            and expected.get("prefill_forwards_per_readout") == 98
            and expected.get("teacher_forced_cells_per_readout") == 98,
            "Measured per-readout phase expectations differ")
    require(expected.get("total_prepares") == HISTORICAL_PREPARES
            and expected.get("total_explicit_logits") == 980,
            "Measured cumulative phase expectations differ")
    require(expected.get("logical_scoring_cells_per_readout") == 98
            and expected.get("logical_scoring_operations_per_readout") == 294
            and expected.get("total_logical_operations") == 1470,
            "Logical cell/operation expectations differ")
    require(expected.get("historical_generation_lengths")
            == list(HISTORICAL_GENERATION_LENGTHS)
            and expected.get("historical_scoring_decoder_passes")
            == list(HISTORICAL_SCORING_PASSES),
            "Historical reconstruction anchors differ")
    require(expected.get("historical_total_scoring_decoder_passes")
            == HISTORICAL_TOTAL_SCORING_PASSES
            and expected.get("historical_training_probe_events")
            == HISTORICAL_TRAINING_PROBE_EVENTS
            and expected.get("historical_grand_total_events")
            == HISTORICAL_GRAND_TOTAL,
            "Historical reconstruction totals differ")
    require(expected.get("measured_by_old_run") is False,
            "The historical reconstruction cannot be labelled measured")
    require(expected.get("prefill_counts_must_equal_available") is True
            and expected.get("generation_lengths_must_reconcile") is True,
            "Prefill/generation reconciliation must be mandatory")


def _validate_curve_limits(limits: Mapping[str, Any]) -> None:
    require(isinstance(limits, Mapping), "Limits are missing")
    required = {
        "max_updates": 7,
        "max_optimizer_steps": 7,
        "max_update_backward_calls": 28,
        "max_probe_backward_calls": 2,
        "max_total_backward_calls": 30,
        "max_readouts": 5,
        "max_scoring_generations": 490,
        "max_teacher_forced_cells": 490,
        "max_decoder_forwards_per_readout": 2400,
        "max_decoder_forwards": 12000,
        "max_prepares_per_readout": 294,
        "max_prepares": 1470,
        "max_explicit_logits_per_readout": 196,
        "max_explicit_logits": 980,
        "max_generated_tokens_per_readout": 1888,
        "max_generated_tokens": 9440,
        "max_visual_encoder_calls": 390,
        "max_visual_encoder_calls_per_readout": 78,
        "max_projector_passes": 12000,
        "readout_weight_writes": 0,
        "max_seconds": 1200,
        "artifact_bytes": 67108864,
        "model_loads": 1,
        "extension_loads": 0,
        "extension_writes": 0,
        "checkpoint_writes": 0,
        "downloads": 0,
        "workers": 1,
        "grad_accum": 1,
    }
    for key, value in required.items():
        require(limits.get(key) == value,
                f"Curve v2 limit differs: {key}")
    learn._validate_token_limits(limits)


def validate_curve_config(config: Mapping[str, Any]) -> dict[str, Any]:
    require(config.get("mode") == CURVE_MODE,
            f"Unknown curve v2 mode: {config.get('mode')!r}")
    require(set(config) == CURVE_KEYS,
            f"Curve v2 config fields differ: "
            f"{sorted(set(config) ^ CURVE_KEYS)}")
    require(config["format"] == CURVE_CONFIG_FORMAT
            and config["version"] == VERSION
            and config["name"] == "vision_learnability_curve_v2",
            "Curve v2 identity differs")
    require(isinstance(config["purpose"], str)
            and "measured" in config["purpose"].lower()
            and "accounting" in config["purpose"].lower(),
            "Curve v2 purpose must describe measured accounting")
    learn._validate_prepared(config["prepared"])
    curve_v1._validate_source(config["source"])
    curve_v1._validate_precondition(config["precondition"])
    curve_v1._validate_schedule(config["schedule"])
    curve_v1._validate_initialization(config["initialization"])
    curve_v1._validate_optimizer(config["optimizer"])
    curve_v1._validate_probes(config["probes"])
    _validate_readout_contract(config["readouts"])
    require(isinstance(config["decision"], Mapping)
            and config["decision"].get("trailing_slope")
            == "descriptive_only",
            "Decision contract differs")
    require(config["v0_gate"] == {
        "planning_authorized_on_pass": False,
        "run_v0": False,
        "requires_separate_predeclared_job": True,
        "heldout_sealed": True,
    }, "V0 gate contract differs")
    learn._validate_prompt_contract(config["prompt_contract"])
    require(config["answer_max_new_tokens"] == 8
            and config["caption_max_new_tokens"] == 32,
            "Generation token budgets differ")
    require(tuple(config["stop_tokens"]) == learn.STOP_TOKENS
            and config["target_eos_token"] == learn.TARGET_EOS_TOKEN,
            "Stop/EOS contract differs")
    require(tuple(config["controls"]) == learn.CONTROL_NAMES,
            "Control inventory differs")
    curve_v1._validate_access(config["access"])
    curve_v1._validate_policy(config["policy"])
    _validate_implementation_pins(config["implementation_pins"])
    _validate_authorization(config["authorization"])
    _validate_expectations(config["expectations"])
    _validate_curve_limits(config["limits"])
    try:
        fwd.validate_budget_declaration(
            config["accounting"], available_per_readout=HISTORICAL_AVAILABLE,
            expected_readouts=5)
    except fwd.AccountingError as error:
        raise learn.LearnabilityDiagnosticError(str(error)) from error
    claims = config["claims"]
    require(claims.get("verdict") == V1_VERDICT
            and claims.get("verdict_preserved") is True
            and claims.get("dev_authorized") is False
            and claims.get("heldout_authorized") is False
            and claims.get("v0_authorized") is False
            and claims.get("promotion") is False
            and claims.get("retry_authorized") is False,
            "Curve v2 must preserve the v1 verdict and seals")
    require(str(config["output"]).startswith("runs/")
            and str(config["report_root"]).startswith("reports/"),
            "Output and report roots must be declared paths")
    require(isinstance(config["limitations"], list) and config["limitations"],
            "Limitations are required")
    return dict(config)


def _validate_step1_limits(limits: Mapping[str, Any]) -> None:
    required = {
        "max_updates": 1,
        "max_optimizer_steps": 1,
        "max_update_backward_calls": 4,
        "max_probe_backward_calls": 0,
        "max_total_backward_calls": 4,
        "max_readouts": 3,
        "max_scoring_generations": 294,
        "max_teacher_forced_cells": 294,
        "max_decoder_forwards_per_readout": 2400,
        "max_decoder_forwards": 7200,
        "max_prepares_per_readout": 294,
        "max_prepares": 882,
        "max_explicit_logits_per_readout": 196,
        "max_explicit_logits": 588,
        "max_generated_tokens_per_readout": 1900,
        "max_generated_tokens": 5700,
        "max_visual_encoder_calls": 220,
        "max_visual_encoder_calls_per_readout": 78,
        "max_projector_passes": 5644,
        "readout_weight_writes": 0,
        "max_seconds": 900,
        "artifact_bytes": 16777216,
        "max_prompt_tokens": 256,
        "max_sequence_tokens": 288,
        "max_new_tokens": 8,
        "caption_max_new_tokens": 32,
        "model_loads": 1,
        "extension_loads": 0,
        "extension_writes": 0,
        "checkpoint_writes": 0,
        "downloads": 0,
        "workers": 1,
        "grad_accum": 1,
    }
    require(isinstance(limits, Mapping), "Step-1 limits are missing")
    for key, value in required.items():
        require(limits.get(key) == value,
                f"Step-1 limit differs: {key}")
    learn._validate_token_limits(limits)


def _validate_step1_experiment(experiment: Mapping[str, Any]) -> None:
    require(isinstance(experiment, Mapping), "Step-1 experiment is missing")
    required = {
        "init_seed": STEP1_SEED,
        "prepared_root": "data/vision_balanced_alignment_v2_20261026/"
                         "prepared04",
        "projector_tensors": list(curve_v1.learn.PROJECTOR_TENSORS),
        "projector_parameter_count": 11542016,
        "optimizer": "NativeFusedAdamW",
        "lr": STEP1_LR,
        "weight_decay": STEP1_WEIGHT_DECAY,
        "clip_grad_norm": 1.0,
        "update_source": "training/batches.jsonl",
        "update_index": 1,
        "rows_per_update": 4,
        "backward_calls": 4,
        "optimizer_steps": 1,
        "probes": 0,
        "prefixes_after_update": [1],
        "forbidden_prefixes": [2, 4, 7],
        "model_loads": 1,
        "extension_writes": 0,
        "checkpoint_writes": 0,
        "dev_access": False,
        "heldout_access": False,
        "v0": False,
        "promotion": False,
        "retry": False,
    }
    for key, value in required.items():
        require(experiment.get(key) == value,
                f"Step-1 experiment differs: {key}")
    require(experiment.get("readout_sequence")
            == ["prefix_0", "prefix_0_null", "update_1", "prefix_1"],
            "Step-1 readout sequence differs")


def validate_step1_config(config: Mapping[str, Any]) -> dict[str, Any]:
    require(config.get("mode") == STEP1_MODE,
            f"Unknown step-1 mode: {config.get('mode')!r}")
    require(set(config) == STEP1_KEYS,
            f"Step-1 config fields differ: "
            f"{sorted(set(config) ^ STEP1_KEYS)}")
    require(config["format"] == STEP1_CONFIG_FORMAT
            and config["version"] == VERSION
            and config["name"] == STEP1_NAME,
            "Step-1 identity differs")
    learn._validate_prepared(config["prepared"])
    curve_v1._validate_source(config["source"])
    curve_v1._validate_precondition(config["precondition"])
    curve_v1._validate_schedule(config["schedule"])
    curve_v1._validate_initialization(config["initialization"])
    curve_v1._validate_optimizer(config["optimizer"])
    _validate_step1_experiment(config["experiment"])
    require(config["readouts"] == {
        "format": STEP1_READOUT_FORMAT,
        "prefixes": [0, 0, 1],
        "labels": ["prefix_0", "prefix_0_null", "prefix_1"],
        "available_per_readout": 98,
        "records_per_readout": 104,
        "placeholders_per_readout": 6,
        "logical_scoring_operations_per_readout": 294,
        "teacher_forced_roles": ["answer", "caption"],
        "empty_text_policy":
            "separate_eos_only_from_whitespace_only",
        "physical_units_required": True,
    }, "Step-1 readout contract differs")
    require(config["historical_anchors_as_hypotheses"] == STEP1_ANCHORS,
            "Step-1 historical anchors differ")
    require(config["nondeterminism_stop"] == {
        "compare": ["prefix_0", "prefix_0_null"],
        "teacher_forced_abs_tolerance": 1e-6,
        "generation_ids": "exact",
        "projector_sha256": "exact",
        "on_mismatch": STEP1_STATUS_NONDETERMINISTIC,
        "stop_before_update": True,
    }, "Step-1 nondeterminism stop differs")
    require(config["expected_measured_work"] == STEP1_EXPECTED_WORK,
            "Step-1 measured-work expectations differ")
    _validate_step1_limits(config["limits"])
    curve_v1._validate_policy(config["policy"])
    require(config["policy"].get("memory_guard") == {
        "driver_bytes": 53687091200,
        "swap_growth_bytes": 536870912,
        "host_free_fraction": 0.15,
        "exclusive_worker": True,
        "cpu_fallback": False,
    }, "Step-1 memory guard differs")
    learn._validate_prompt_contract(config["prompt_contract"])
    _validate_implementation_pins(config["implementation_pins"])
    _validate_authorization(config["authorization"])
    try:
        fwd.validate_budget_declaration(
            config["accounting"], available_per_readout=98,
            expected_readouts=3)
    except fwd.AccountingError as error:
        raise learn.LearnabilityDiagnosticError(str(error)) from error
    require(tuple(config["statuses"]) == STEP1_STATUSES,
            "Step-1 statuses differ")
    claims = config["claims"]
    require(claims == {
        "curve_v1_verdict": V1_VERDICT,
        "curve_verdict_preserved": True,
        "verdict_change": False,
        "dev_authorized": False,
        "heldout_authorized": False,
        "v0_authorized": False,
        "promotion": False,
        "retry_authorized": False,
        "generated_answer_conversion_claim": False,
        "rank_improvement_is_conversion": False,
    }, "Step-1 claims differ")
    require(isinstance(config["limitations"], list) and config["limitations"],
            "Step-1 limitations are required")
    return dict(config)


# ---------------------------------------------------------------------------
# Measured readout validation
# ---------------------------------------------------------------------------


def _measured_counters(readout: Mapping[str, Any]) -> Mapping[str, Any]:
    measured = readout.get("measured")
    require(isinstance(measured, Mapping), "Readout lacks measured counters")
    counters = measured.get("counters")
    require(isinstance(counters, Mapping), "Measured counters are missing")
    missing = set(fwd.COUNTER_KEYS) - set(counters)
    require(not missing, f"Measured counters missing: {sorted(missing)}")
    units = measured.get("units")
    require(isinstance(units, Mapping), "Measured units are missing")
    for key in fwd.COUNTER_KEYS:
        require(units.get(key) == fwd.UNIT_BY_COUNTER[key],
                f"Measured unit differs: {key}")
    for key, value in counters.items():
        require(type(value) is int and value >= 0,
                f"Measured counter {key} is not a nonnegative integer")
    return counters


def validate_measured_readout(
        readout: Mapping[str, Any], *,
        available_per_readout: int = HISTORICAL_AVAILABLE,
        prefixes: Sequence[int] = HISTORICAL_PREFIXES,
        prefix_position: int = 0,
        records: Sequence[Mapping[str, Any]] | None = None,
        max_new_tokens_by_kind: Mapping[str, int] | None = None,
        endpoint: bool = False,
) -> dict[str, Any]:
    """Validate one physical readout against measured counters and records."""
    violations: list[str] = []
    prefix = readout.get("prefix")
    if prefix_position < len(prefixes):
        observed = -1 if prefix is None else int(prefix)
        if observed != int(prefixes[prefix_position]):
            violations.append(
                f"prefix:{prefix}!={prefixes[prefix_position]}")
    measured = readout.get("measured")
    if not isinstance(measured, Mapping):
        return {"ok": False, "violations": ["measured_missing"]}
    try:
        counters = _measured_counters(readout)
    except learn.LearnabilityDiagnosticError as error:
        return {"ok": False, "violations": [f"counters_invalid:{error}"]}
    label = f"prefix:{prefix}"
    decoder = int(counters["decoder_forward_calls"])
    prepares = int(counters["prepare_sense_passes"])
    logits = int(counters["explicit_logits_forwards"])
    prefill = int(counters["generation_prefill_calls"])
    decode = int(counters["generation_decode_calls"])
    generated = int(counters["generated_tokens"])
    if decoder != prepares + logits + prefill + decode:
        violations.append(f"{label}:decoder_phase_sum")
    if generated != prefill + decode:
        violations.append(f"{label}:generated_prefill_decode")
    if prepares != 3 * available_per_readout:
        violations.append(f"{label}:prepare_count:{prepares}")
    if logits != 2 * available_per_readout:
        violations.append(f"{label}:logits_count:{logits}")
    if prefill != available_per_readout:
        violations.append(f"{label}:prefill_count:{prefill}")
    if int(counters["processed_visual_positions"]) != 196 * int(
            counters["visual_forward_calls"]):
        violations.append(f"{label}:visual_positions")
    if int(counters["projector_passes"]) != int(
            counters["visual_forward_calls"]):
        violations.append(f"{label}:projector_passes")
    if int(counters["visual_encoder_calls"]) != int(
            counters["visual_encoder_cache_misses"]):
        violations.append(f"{label}:encoder_miss_identity")
    if int(counters["visual_observation_calls"]) != int(
            counters["visual_encoder_cache_hits"]) + int(
            counters["visual_encoder_cache_misses"]):
        violations.append(f"{label}:cache_identity")
    if int(counters["logical_scoring_cells"]) != available_per_readout:
        violations.append(f"{label}:logical_cells")
    if int(counters["logical_scoring_operations"]) != 3 * (
            available_per_readout):
        violations.append(f"{label}:logical_operations")
    logical = measured.get("logical")
    if not isinstance(logical, Mapping):
        violations.append(f"{label}:logical_units_missing")
    else:
        if logical.get("scoring_cells") != available_per_readout:
            violations.append(f"{label}:logical_cells_report")
        if logical.get("scoring_operations") != 3 * available_per_readout:
            violations.append(f"{label}:logical_operations_report")
        if logical.get("physical_decoder_passes") != decoder:
            violations.append(f"{label}:physical_decoder_report")
    counts = readout.get("counts")
    if not isinstance(counts, Mapping):
        violations.append(f"{label}:counts_missing")
    else:
        missing = READOUT_COUNT_KEYS - set(counts)
        if missing:
            violations.append(f"{label}:counts_missing:{sorted(missing)}")
        if counts.get("available") != available_per_readout:
            violations.append(f"{label}:counts_available")
        if counts.get("records") != 104:
            violations.append(f"{label}:counts_records")
        if counts.get("placeholders") != 6:
            violations.append(f"{label}:counts_placeholders")
        if counts.get("prefill_forwards") != prefill:
            violations.append(f"{label}:counts_prefill")
        if counts.get("teacher_forced_cells") != available_per_readout:
            violations.append(f"{label}:counts_teacher_forced")
        if counts.get("generated_tokens") != generated:
            violations.append(f"{label}:counts_generated")
        if counts.get("generations") != available_per_readout:
            violations.append(f"{label}:counts_generations")
    summary = readout.get("generation_summary")
    if not isinstance(summary, Mapping):
        violations.append(f"{label}:generation_summary_missing")
    elif summary.get("generated_tokens") != generated:
        violations.append(f"{label}:generation_summary_tokens")
    if records is not None:
        available = [record for record in records
                     if record.get("available") is not False]
        if len(available) != available_per_readout:
            violations.append(f"{label}:record_available")
        record_tokens = sum(len(record.get("generated_ids") or [])
                            for record in available)
        if record_tokens != generated:
            violations.append(
                f"{label}:record_token_sum:{record_tokens}!={generated}")
        limits = dict(max_new_tokens_by_kind or {"answer": 8, "caption": 32})
        for record in available:
            kind = str(record.get("kind"))
            ids = record.get("generated_ids") or []
            if not ids:
                violations.append(
                    f"{label}:empty_ids:{record.get('case_id')}:"
                    f"{record.get('control')}")
            if kind in limits and len(ids) > int(limits[kind]):
                violations.append(
                    f"{label}:generation_length:{kind}:{len(ids)}")
            if record.get("prefill_argmax_matches_first_token") is not None:
                prefill_report = record.get("prefill") or {}
                expected = bool(ids) and prefill_report.get(
                    "argmax_token") == int(ids[0])
                if bool(record.get(
                        "prefill_argmax_matches_first_token")) != expected:
                    violations.append(
                        f"{label}:prefill_self_consistency:"
                        f"{record.get('case_id')}:{record.get('control')}")
    gates = readout.get("gates")
    if endpoint:
        if not isinstance(gates, Mapping):
            violations.append(f"{label}:endpoint_gates_missing")
        elif gates.get("endpoint_context_only") is not True \
                or gates.get("curve_decision_input") is not False:
            violations.append(f"{label}:endpoint_gate_scope")
    elif gates is not None:
        violations.append(f"{label}:unexpected_gates")
    return {
        "ok": not violations,
        "violations": violations,
        "prefix": prefix,
        "decoder_forward_calls": decoder,
        "generated_tokens": generated,
        "prepare_sense_passes": prepares,
        "explicit_logits_forwards": logits,
        "generation_prefill_calls": prefill,
        "generation_decode_calls": decode,
        "logical_scoring_cells": int(counters["logical_scoring_cells"]),
        "logical_scoring_operations":
            int(counters["logical_scoring_operations"]),
    }


def validate_measured_readouts(
        readouts: Sequence[Mapping[str, Any]], *,
        prefixes: Sequence[int] = HISTORICAL_PREFIXES,
        available_per_readout: int = HISTORICAL_AVAILABLE,
        records_by_prefix: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
        endpoint_prefix: int = 7,
) -> dict[str, Any]:
    violations: list[str] = []
    totals = {
        "decoder_forward_calls": 0,
        "generated_tokens": 0,
        "prepare_sense_passes": 0,
        "explicit_logits_forwards": 0,
        "logical_scoring_cells": 0,
        "logical_scoring_operations": 0,
    }
    reports = []
    if len(readouts) != len(prefixes):
        violations.append(f"readouts:{len(readouts)}")
    for position, readout in enumerate(readouts):
        prefix = int(readout.get("prefix", -1))
        report = validate_measured_readout(
            readout, available_per_readout=available_per_readout,
            prefixes=prefixes, prefix_position=position,
            records=(None if records_by_prefix is None
                     else records_by_prefix.get(prefix)),
            endpoint=prefix == endpoint_prefix)
        reports.append(report)
        violations.extend(report["violations"])
        for key in totals:
            totals[key] += int(report.get(key, 0))
    return {
        "ok": not violations,
        "violations": sorted(set(violations)),
        "readouts": reports,
        "totals": totals,
    }


# ---------------------------------------------------------------------------
# EOS / mask / target / first-token audits
# ---------------------------------------------------------------------------


def response_audit(
        records: Sequence[Mapping[str, Any]], *,
        rows_by_case: Mapping[str, Mapping[str, Any]] | None = None,
        caption_labels: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Audit EOS-only vs whitespace-only text, targets and first tokens."""
    violations: list[str] = []
    eos_only = 0
    whitespace_only = 0
    by_prefix: dict[int, dict[str, int]] = {}
    by_kind_control: dict[str, int] = {}
    target_checked = 0
    answer_target_checked = 0
    caption_target_checked = 0
    self_consistency_checked = 0
    prefill_gold_checked = 0
    prefill_gold_correct = 0
    for record in records:
        if record.get("available") is False:
            continue
        case_id = str(record.get("case_id"))
        control = str(record.get("control"))
        kind = str(record.get("kind"))
        ids = [int(value) for value in record.get("generated_ids") or []]
        stops = record.get("stop_tokens") or list(learn.STOP_TOKENS)
        is_eos = fwd.eos_only_ids(ids, stops)
        is_whitespace = (not is_eos
                         and fwd.whitespace_only_text(
                             record.get("generated_text")))
        prefix = int(record.get("prefix", -1))
        bucket = by_prefix.setdefault(prefix, {
            "eos_only": 0, "whitespace_only": 0, "available": 0})
        bucket["available"] += 1
        if is_eos:
            eos_only += 1
            bucket["eos_only"] += 1
            key = f"{kind}:{control}:eos_only"
            by_kind_control[key] = by_kind_control.get(key, 0) + 1
        if is_whitespace:
            whitespace_only += 1
            bucket["whitespace_only"] += 1
            key = f"{kind}:{control}:whitespace_only"
            by_kind_control[key] = by_kind_control.get(key, 0) + 1
        if not ids:
            violations.append(
                f"empty_ids:{case_id}:{control}")
        target_text = record.get("target_text")
        if kind == "answer" and rows_by_case is not None:
            row = rows_by_case.get(case_id)
            if row is None:
                violations.append(f"missing_row:{case_id}")
            else:
                answer_target_checked += 1
                target_checked += 1
                if target_text != row.get("response"):
                    violations.append(
                        f"answer_target:{case_id}:{control}")
        elif kind == "caption" and caption_labels is not None:
            label = caption_labels.get(case_id)
            if label is None:
                violations.append(f"missing_caption_label:{case_id}")
            else:
                caption_target_checked += 1
                target_checked += 1
                if target_text != label.get("answer"):
                    violations.append(
                        f"caption_target:{case_id}:{control}")
        prefill = record.get("prefill") or {}
        argmax = prefill.get("argmax_token")
        if ids and argmax is not None:
            self_consistency_checked += 1
            expected = int(argmax) == ids[0]
            recorded = record.get("prefill_argmax_matches_first_token")
            if recorded is not None and bool(recorded) != expected:
                violations.append(
                    f"prefill_self_consistency:{case_id}:{control}")
        gold = prefill.get("gold_token")
        target_first = record.get("target_first_token")
        if gold is not None and target_first is not None:
            prefill_gold_checked += 1
            if int(gold) == int(target_first) and argmax is not None \
                    and int(argmax) == int(gold):
                prefill_gold_correct += 1
    return {
        "ok": not violations,
        "violations": sorted(set(violations)),
        "available_records": sum(bucket["available"]
                                 for bucket in by_prefix.values()),
        "eos_only": eos_only,
        "whitespace_only": whitespace_only,
        "eos_only_policy":
            "reported separately and scored as a generated response",
        "by_prefix": {str(prefix): bucket
                      for prefix, bucket in sorted(by_prefix.items())},
        "by_kind_control": dict(sorted(by_kind_control.items())),
        "target_equality": {
            "checked": target_checked,
            "answer_checked": answer_target_checked,
            "caption_checked": caption_target_checked,
            "answer_target_source": "row.response",
            "caption_target_source": "sealed_caption_answer",
        },
        "prefill_self_consistency": {
            "checked": self_consistency_checked,
            "label": "not an independent first-token accuracy claim",
        },
        "prefill_gold": {
            "checked": prefill_gold_checked,
            "correct": prefill_gold_correct,
        },
    }


def validate_response_masks(
        fit_contract_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    violations: list[str] = []
    seen = set()
    occurrences = 0
    for row in fit_contract_rows:
        occurrences += 1
        label = f"{row.get('example_id')}"
        seen.add(label)
        if row.get("prompt_masked") is not True:
            violations.append(f"prompt_not_masked:{label}")
        if row.get("response_only_labels") is not True \
                or row.get("label_shift") != 1:
            violations.append(f"label_contract:{label}")
        if row.get("contract_valid") is not True:
            violations.append(f"contract_invalid:{label}")
        if row.get("prompt_masked_tokens") != row.get("prompt_tokens"):
            violations.append(f"prompt_mask_width:{label}")
        if row.get("supervised_label_tokens") != row.get("response_tokens"):
            violations.append(f"supervised_width:{label}")
        if row.get("loss_tokens") != row.get("response_tokens"):
            violations.append(f"loss_width:{label}")
        if row.get("response_eos") is not True \
                or row.get("eos_in_stop_set") is not True:
            violations.append(f"eos_supervision:{label}")
        if row.get("pad_free_response") is not True:
            violations.append(f"pad_in_response:{label}")
        if row.get("kind") == "caption" \
                and row.get("target_text") is not None \
                and row.get("label_target") is not None \
                and row.get("target_text") != row.get("label_target"):
            violations.append(f"caption_target:{label}")
    return {
        "ok": not violations,
        "violations": sorted(set(violations)),
        "rows_checked": occurrences,
        "unique_rows": len(seen),
        "scheduled_repeats_allowed": True,
        "prompt_fully_masked": not any(
            violation.startswith("prompt_not_masked")
            for violation in violations),
        "response_only_labels": True,
        "label_shift": 1,
        "visual_tokens_never_supervised": True,
    }


# ---------------------------------------------------------------------------
# Projector scale and feature distinguishability
# ---------------------------------------------------------------------------


def projector_scale_report(
        projected: Any, *, text_embedding_rms: float | None = None
) -> dict[str, Any]:
    """Measure projected-token scale; caller supplies an already projected tensor."""
    import torch
    require(torch.is_tensor(projected) and projected.ndim == 3,
            "Projected features must be [1,patches,width]")
    tensor = projected.detach().float()
    require(tensor.shape[0] == 1 and tensor.shape[1] > 0,
            "Projected features must carry at least one patch")
    require(bool(torch.isfinite(tensor).all()),
            "Projected features are nonfinite")
    rms = float(tensor.pow(2).mean().sqrt().cpu())
    norms = tensor[0].norm(dim=-1)
    report = {
        "format": "replica.vision_forward_accounting.projector_scale.v1",
        "shape": list(tensor.shape),
        "rms": rms,
        "mean_token_norm": float(norms.mean().cpu()),
        "min_token_norm": float(norms.min().cpu()),
        "max_token_norm": float(norms.max().cpu()),
        "within_image_variance": float(tensor[0].var(dim=0,
                                                       unbiased=False).mean()
                                       .cpu()),
        "finite": True,
    }
    if text_embedding_rms is not None:
        text_rms = float(text_embedding_rms)
        require(text_rms > 0, "Text embedding RMS must be positive")
        report["text_embedding_rms"] = text_rms
        report["projected_to_text_rms_ratio"] = rms / text_rms
    return report


def feature_distinguishability_report(
        projected_by_control: Mapping[str, Mapping[str, Any]], *,
        text_embedding_rms: float | None = None
) -> dict[str, Any]:
    """Compare correct/sibling/shuffled projected features per case.

    ``projected_by_control`` maps case ID to control to a projected tensor.
    No-image has no visual tensor and is reported as absent.
    """
    import torch
    require(isinstance(projected_by_control, Mapping)
            and projected_by_control,
            "Control feature tensors are required")
    cases = []
    sibling_cosines: list[float] = []
    shuffled_cosines: list[float] = []
    rms_values: list[float] = []
    for case_id, controls in sorted(projected_by_control.items()):
        require(isinstance(controls, Mapping), "Control map differs")
        correct = controls.get("correct")
        require(torch.is_tensor(correct),
                f"Correct projected tensor missing: {case_id}")
        correct_mean = correct.detach().float()[0].mean(dim=0)
        case = {
            "case_id": str(case_id),
            "controls_present": sorted(controls),
            "noimage_present": "noimage" in controls
                               and controls["noimage"] is not None,
        }
        rms_values.append(float(correct.detach().float().pow(2).mean()
                                .sqrt().cpu()))
        for control, values in (("sibling", sibling_cosines),
                                ("shuffled", shuffled_cosines)):
            tensor = controls.get(control)
            if tensor is None:
                continue
            require(torch.is_tensor(tensor), f"{control} tensor differs")
            mean = tensor.detach().float()[0].mean(dim=0)
            denominator = float(correct_mean.norm() * mean.norm()) + 1e-12
            cosine = float((correct_mean * mean).sum().cpu()) / denominator
            values.append(cosine)
            case[f"correct_{control}_cosine"] = cosine
        cases.append(case)
    require(sibling_cosines and shuffled_cosines,
            "Correct/sibling/shuffled projected features are required")
    summary_sibling = _mean(sibling_cosines)
    summary_shuffled = _mean(shuffled_cosines)
    distinct = (summary_sibling is not None and summary_sibling < 0.999
                and summary_shuffled is not None
                and summary_shuffled < 0.999)
    report = {
        "format": "replica.vision_forward_accounting.feature_controls.v1",
        "cases": cases,
        "mean_correct_sibling_cosine": summary_sibling,
        "mean_correct_shuffled_cosine": summary_shuffled,
        "mean_projected_rms": _mean(rms_values),
        "feature_distinguishable": bool(distinct),
        "noimage_policy": "no visual tensor; reported absent, not zeroed",
    }
    if text_embedding_rms is not None:
        require(float(text_embedding_rms) > 0,
                "Text embedding RMS must be positive")
        report["text_embedding_rms"] = float(text_embedding_rms)
        report["projected_to_text_rms_ratio"] = (
            float(report["mean_projected_rms"]) / float(text_embedding_rms))
    return report


# ---------------------------------------------------------------------------
# Historical control evidence and correction payload
# ---------------------------------------------------------------------------


def control_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        if record.get("available") is False:
            continue
        key = (int(record.get("prefix", -1)), str(record.get("kind")),
               str(record.get("control")))
        grouped.setdefault(key, []).append(record)
    result: dict[str, Any] = {}
    for (prefix, kind, control), rows in sorted(grouped.items()):
        losses = [_finite(row.get("teacher_forced_loss")) for row in rows]
        losses = [value for value in losses if value is not None]
        ranks = [_finite((row.get("prefill") or {}).get("gold_rank"))
                 for row in rows]
        ranks = [value for value in ranks if value is not None]
        args = [(row.get("prefill") or {}).get("argmax_token")
                for row in rows]
        eos = 0
        for row in rows:
            ids = row.get("generated_ids") or []
            stops = row.get("stop_tokens") or list(learn.STOP_TOKENS)
            eos += int(fwd.eos_only_ids(ids, stops))
        result.setdefault(str(prefix), {}).setdefault(kind, {})[control] = {
            "cases": len(rows),
            "mean_teacher_forced_loss": _mean(losses),
            "mean_prefill_gold_rank": _mean(ranks),
            "prefill_argmax_eos_rate": (
                sum(1 for token in args
                    if token in set(learn.STOP_TOKENS)) / len(args)
                if args else None),
            "eos_only": eos,
        }
    # Descriptive margins only: a rank gain is not generated-answer conversion.
    for prefix, kinds in result.items():
        for kind, controls in kinds.items():
            correct = controls.get("correct") or {}
            noimage = controls.get("noimage") or {}
            if correct.get("mean_teacher_forced_loss") is not None \
                    and noimage.get("mean_teacher_forced_loss") is not None:
                correct["loss_margin_vs_noimage"] = (
                    correct["mean_teacher_forced_loss"]
                    - noimage["mean_teacher_forced_loss"])
            sibling = controls.get("sibling") or {}
            shuffled = controls.get("shuffled") or {}
            if correct.get("mean_teacher_forced_loss") is not None:
                if sibling.get("mean_teacher_forced_loss") is not None:
                    correct["loss_margin_vs_sibling"] = (
                        correct["mean_teacher_forced_loss"]
                        - sibling["mean_teacher_forced_loss"])
                if shuffled.get("mean_teacher_forced_loss") is not None:
                    correct["loss_margin_vs_shuffled"] = (
                        correct["mean_teacher_forced_loss"]
                        - shuffled["mean_teacher_forced_loss"])
    return result


def build_historical_correction(
        *,
        root: str | Path,
        run: str | Path,
        summary: str | Path,
        v1_config: str | Path = V1_CONFIG,
        step1_config: str | Path | None = None,
        prepared_root: str | Path | None = None,
        replay_command: str | None = None,
) -> dict[str, Any]:
    """Build the explicit accounting correction over preserved v1 receipts."""
    root = Path(root)
    run_path = root / run
    summary_path = root / summary
    v1_path = root / v1_config
    require(run_path.is_dir(), "Preserved v1 run is missing")
    require(summary_path.is_file(), "Preserved v1 summary is missing")
    require(v1_path.is_file(), "Preserved v1 config is missing")
    summary_payload = _read_json(summary_path)
    require(summary_payload.get("status") == V1_VERDICT,
            "The preserved v1 verdict changed")
    readout_root = run_path / "readouts"
    records_by_prefix: dict[int, list[dict[str, Any]]] = {}
    record_sums: dict[int, int] = {}
    all_records: list[dict[str, Any]] = []
    source_receipts = {}
    for path in (summary_path, v1_path, run_path / "report.json",
                 run_path / "training_trace.jsonl",
                 run_path / "backward_probes.json"):
        require(path.is_file(), f"Preserved receipt is missing: {path}")
        source_receipts[str(path.relative_to(root))] = _sha256_file(path)
    for prefix in HISTORICAL_PREFIXES:
        directory = readout_root / f"prefix_{prefix:03d}"
        require(directory.is_dir(), f"Readout is missing: {directory}")
        readout_records = _read_jsonl(directory / "records.jsonl")
        records_by_prefix[prefix] = readout_records
        record_sums[prefix] = sum(
            len(record.get("generated_ids") or [])
            for record in readout_records
            if record.get("available") is not False)
        all_records.extend({**record, "prefix": prefix}
                           for record in readout_records)
        for name in ("records.jsonl", "readout.json",
                     "train_answer_generations.jsonl",
                     "train_caption_generations.jsonl"):
            path = directory / name
            require(path.is_file(), f"Preserved readout file missing: {path}")
            source_receipts[str(path.relative_to(root))] = _sha256_file(path)
    generation_lengths = {
        prefix: int(length)
        for prefix, length in zip(HISTORICAL_PREFIXES,
                                  HISTORICAL_GENERATION_LENGTHS)}
    record_token_sums = {
        prefix: int(record_sums[prefix]) for prefix in HISTORICAL_PREFIXES}
    reconstruction = fwd.reconstruct_scoring_accounting(
        list(HISTORICAL_GENERATION_LENGTHS))
    validation = fwd.validate_reconstruction(reconstruction)
    require(validation["ok"], "Historical reconstruction failed")
    prepared_root = (Path(prepared_root) if prepared_root is not None
                     else root / "data/vision_balanced_alignment_v2_"
                                 "20261026/prepared04")
    rows_by_case: dict[str, Mapping[str, Any]] = {}
    caption_labels: dict[str, Mapping[str, Any]] = {}
    alignment_path = prepared_root / "training/alignment.jsonl"
    labels_path = prepared_root / "training/train_caption_labels.jsonl"
    if alignment_path.is_file() and labels_path.is_file():
        rows_by_case = {str(row["case_id"]): row
                        for row in _read_jsonl(alignment_path)}
        caption_labels = {str(row["case_id"]): row
                          for row in _read_jsonl(labels_path)}
        source_receipts[str(alignment_path.relative_to(root))] = (
            _sha256_file(alignment_path))
        source_receipts[str(labels_path.relative_to(root))] = (
            _sha256_file(labels_path))
    audit = response_audit(
        all_records, rows_by_case=rows_by_case or None,
        caption_labels=caption_labels or None)
    trace = _read_jsonl(run_path / "training_trace.jsonl")
    fit_rows = [row for record in trace
                for row in record.get("fit_contract_rows") or []]
    mask_audit = validate_response_masks(fit_rows)
    controls = control_metrics(all_records)
    sibling_checks = []
    shuffled_checks = []
    if rows_by_case:
        from .vision_collapse_diagnostic import control_plan
        plan = control_plan(list(rows_by_case.values()))
        for case_id, row in rows_by_case.items():
            plan_row = plan[str(case_id)]
            correct_sha = row.get("image_sha256")
            if plan_row.get("sibling_available"):
                sibling_row = rows_by_case.get(
                    str(plan_row["sibling_case_id"]))
                if sibling_row is not None:
                    sibling_checks.append(
                        correct_sha != sibling_row.get("image_sha256"))
            shuffled_row = rows_by_case.get(
                str(plan_row["shuffled_case_id"]))
            if shuffled_row is not None:
                shuffled_checks.append(
                    correct_sha != shuffled_row.get("image_sha256"))
    sibling_separated = bool(sibling_checks) and all(sibling_checks)
    shuffled_separated = bool(shuffled_checks) and all(shuffled_checks)
    superseded_path = root / (
        "reports/remediation_20261027/superseded/"
        "vision_learnability_curve_v1_accounting_correction_declared_5254.json")
    superseded = None
    if superseded_path.is_file():
        superseded_sha = _sha256_file(superseded_path)
        source_receipts[str(superseded_path.relative_to(root))] = (
            superseded_sha)
        superseded = {
            "path": str(superseded_path.relative_to(root)),
            "sha256": superseded_sha,
            "status": "superseded_by_measured_attempt02",
            "declared_generation_lengths": [1370, 1442, 1396, 721, 325],
            "declared_total_generated_tokens": 5254,
            "declared_scoring_decoder_passes": [
                1860, 1932, 1886, 1211, 815],
            "declared_total_scoring_decoder_passes": 7704,
            "declared_grand_total_events": 7764,
            "reason": (
                "raw saved generation IDs include terminal stops and sum to "
                "1,362/1,442/1,396/721/324 = 5,245"),
        }
    attempt02_evidence = {
        "run": "runs/vision_learnability_step1_reproduction_v1_20261027_"
               "attempt02",
        "preserved_copy": "reports/remediation_20261027/step1_attempt02",
        "prefix0": {
            "generated_tokens": 1362,
            "decoder_calls": 1852,
            "visual_encoder_misses": 72,
            "answer_nll": 10.704498972211566,
            "caption_nll": 9.773420969645182,
            "projector_sha256": STEP1_INIT_SHA256,
        },
        "prefix0_null": {
            "generated_tokens": 1362,
            "decoder_calls": 1852,
            "visual_encoder_misses": 72,
            "answer_nll": 10.704498972211566,
            "caption_nll": 9.773420969645182,
            "projector_sha256": STEP1_INIT_SHA256,
        },
        "update": {
            "loss": 10.964820146560669,
            "grad_norm": 345.9010314941406,
            "projector_update_norm": 1.0185893941275908,
            "backward_calls": 4,
            "optimizer_steps": 1,
        },
        "prefix1_failure": (
            "cumulative visual_encoder_cache_misses reached 217 because the "
            "216 cap omitted four update-phase image observations"),
        "prefix1_final_readout_exists": False,
        "hypothesis_verdict_exists": False,
    }
    step1_identity = None
    if step1_config is not None:
        step1_path = root / step1_config
        require(step1_path.is_file(), "Step-1 protocol is missing")
        step1_identity = {
            "path": str(step1_config),
            "sha256": _sha256_file(step1_path),
            "name": STEP1_NAME,
            "status": "predeclared_pending_protocol_commit",
            "native_execution_authorized": False,
            "expected_measured_work": STEP1_EXPECTED_WORK,
            "statuses": list(STEP1_STATUSES),
        }
        source_receipts[str(step1_config)] = _sha256_file(step1_path)
    command = replay_command or (
        "python scripts/run_vision_learnability_curve_v2.py "
        "replay-accounting --run runs/vision_learnability_curve_v1_20261026_"
        "attempt04 --summary reports/vision_learnability_curve_v1_20261026/"
        "summary.json --config configs/vision_learnability_curve_v1.json "
        "--step1-config configs/vision_learnability_step1_reproduction_v1.json")
    return {
        "format": CORRECTION_FORMAT,
        "version": VERSION,
        "subject": "vision_learnability_curve_v1",
        "status": "accounting_correction_only_verdict_preserved",
        "v1_verdict": V1_VERDICT,
        "v1_verdict_preserved": True,
        "correction": {
            "original_declared_scoring_forwards": HISTORICAL_PREPARES,
            "original_unit_is": fwd.PREPARE_UNIT,
            "original_label_was": "scoring_forwards",
            "classification": "reconstructed_not_measured_by_old_run",
            "measured_by_old_run": False,
            "generation_lengths_by_prefix": {
                str(prefix): generation_lengths[prefix]
                for prefix in HISTORICAL_PREFIXES},
            "preserved_record_token_sums": {
                str(prefix): record_token_sums[prefix]
                for prefix in HISTORICAL_PREFIXES},
            "record_token_sum_delta": {
                str(prefix): generation_lengths[prefix]
                - record_token_sums[prefix]
                for prefix in HISTORICAL_PREFIXES},
            "delta_note": (
                "The remediation accounting convention records "
                "1,362/1,442/1,396/721/324 generated tokens = 5,245. An earlier "
                "correction incorrectly declared 1,370/1,442/1,396/721/325 "
                "= 5,254 (7,704 scoring / 7,764 events); that artifact is "
                "preserved under reports/remediation_20261027/superseded/. "
                "The corrected scoring total is 7,695 and grand total 7,755."),
            "reconstruction": reconstruction,
            "declared_readout_decoder_passes": {
                str(prefix): pass_count
                for prefix, pass_count in zip(HISTORICAL_PREFIXES,
                                              HISTORICAL_SCORING_PASSES)},
            "training_probe_events": HISTORICAL_TRAINING_PROBE_EVENTS,
            "grand_total_events": HISTORICAL_GRAND_TOTAL,
            "prepares_are_a_subset_of_scoring_passes": True,
            "logical_operations_total": HISTORICAL_LOGICAL_OPERATIONS,
            "supersedes": superseded,
        },
        "attempt02_evidence": attempt02_evidence,
        "audits": {
            "response": audit,
            "caption_prefill_target_alignment": {
                "status": "v1_prefill_used_row_response_for_every_kind",
                "prefill_source": "row.response",
                "teacher_forced_caption_source": "sealed_caption_answer",
                "equality_asserted_by_v1": False,
                "v2_fix": (
                    "the v2 measured readout scores the prefill gold token "
                    "against the exact teacher-forced target text"),
            },
            "response_masks": mask_audit,
            "control_metrics": controls,
            "projector_output_scale": {
                "status": "not_measurable_from_preserved_receipts",
                "reason": (
                    "attempt04 did not persist projected tensors or encoder "
                    "features; the ablation arm's 83x ratio belongs to a "
                    "different arm and is not substituted here"),
                "required_in_future_readout": True,
            },
            "feature_distinguishability": {
                "status": "measured_from_control_losses_only",
                "correct_sibling_distinct_image_shas": bool(
                    sibling_separated),
                "correct_shuffled_distinct_image_shas": bool(
                    shuffled_separated),
                "rank_improvement_is_not_conversion": True,
                "controls": controls,
            },
            "readout_nondeterminism": {
                "status": "not_measured_by_old_run",
                "step1_null_repeat_required": True,
            },
        },
        "preserved_source_receipts": dict(sorted(source_receipts.items())),
        "step1_protocol": step1_identity,
        "replay": {
            "command": command,
            "deterministic": True,
            "excludes": ["timestamps", "native execution"],
            "checks": [
                "v1 verdict unchanged",
                "source receipt hashes",
                "generation lengths and record sums",
                "7,695 reconstruction",
                "EOS/whitespace counts",
                "target equality",
                "response masks",
                "control metrics",
            ],
        },
        "preservation": {
            "v1_files_byte_identical": True,
            "attempt04_receipt_unchanged": True,
            "dev_accessed": False,
            "heldout_accessed": False,
            "v0_run": False,
            "promotion": False,
            "retry_authorized": False,
            "next_gate": (
                "accounting correction and one predeclared step-1 protocol; "
                "no execution is authorized by this evidence"),
        },
    }


def validate_historical_correction(evidence: Mapping[str, Any]) -> dict[str, Any]:
    require(evidence.get("format") == CORRECTION_FORMAT,
            "Correction format differs")
    require(evidence.get("v1_verdict") == V1_VERDICT
            and evidence.get("v1_verdict_preserved") is True,
            "The v1 verdict is not preserved")
    correction = evidence.get("correction") or {}
    require(correction.get("measured_by_old_run") is False,
            "Historical accounting must be reconstructed")
    require(correction.get("declared_readout_decoder_passes")
            == {str(prefix): value
                for prefix, value in zip(HISTORICAL_PREFIXES,
                                         HISTORICAL_SCORING_PASSES)},
            "Declared per-prefix scoring passes differ")
    require(correction.get("training_probe_events") == 60
            and correction.get("grand_total_events") == 7755,
            "Historical totals differ")
    require((correction.get("reconstruction") or {}).get("totals", {})
            .get("scoring_decoder_passes") == 7695,
            "Reconstruction total differs")
    superseded = correction.get("supersedes")
    require(isinstance(superseded, Mapping)
            and superseded.get("declared_total_scoring_decoder_passes") == 7704
            and superseded.get("declared_total_generated_tokens") == 5254,
            "The superseded declared-5,254 artifact is not recorded")
    audits = evidence.get("audits") or {}
    require((audits.get("response_masks") or {}).get("ok") is True,
            "Response mask audit failed")
    require((audits.get("response") or {}).get("ok") is True,
            "Response/target audit failed")
    require("raw" not in evidence, "Raw payloads are not permitted")
    return {"ok": True, "v1_verdict": V1_VERDICT,
            "reconstructed": True, "measured": False}


# ---------------------------------------------------------------------------
# Step-1 protocol and accounting
# ---------------------------------------------------------------------------


def validate_step1_accounting(
        readouts: Sequence[Mapping[str, Any]],
        training: Mapping[str, Any], *,
        expected: Mapping[str, Any] = STEP1_EXPECTED_WORK,
) -> dict[str, Any]:
    """Reconcile the three readouts plus the one-update block to 5,644."""
    violations: list[str] = []
    totals = {
        "decoder_forward_calls": 0,
        "generated_tokens": 0,
        "prepare_sense_passes": 0,
        "explicit_logits_forwards": 0,
    }
    for readout in readouts:
        measured = readout.get("measured") or {}
        counters = measured.get("counters") or {}
        for key in totals:
            totals[key] += int(counters.get(key, 0))
    training_counters = (training.get("phase_totals") or {}).get(
        "training") or {}
    training_global = training.get("counters") or {}
    update_forwards = int(training_counters.get("forward_calls", 0))
    totals["update_forwards"] = update_forwards
    totals["decoder_forward_calls"] += update_forwards
    totals["backward_calls"] = int(training_global.get("backward_calls", 0))
    totals["optimizer_steps"] = int(
        training_global.get("optimizer_steps", 0))
    # These are whole-run cumulative counters in accounting.json.  Adding the
    # per-readout counters again double-counted every readout observation.
    for key in ("visual_observation_calls", "visual_encoder_calls",
                "visual_encoder_cache_misses",
                "visual_encoder_cache_hits", "projector_passes"):
        value = training_global.get(key)
        if type(value) is not int or value < 0:
            violations.append(f"{key}:missing_or_invalid")
            totals[key] = -1
        else:
            totals[key] = value
    for key in ("decoder_forward_calls", "generated_tokens",
                "prepare_sense_passes", "explicit_logits_forwards",
                "update_forwards", "backward_calls", "optimizer_steps"):
        expected_value = int(expected[key])
        if int(totals[key]) != expected_value:
            violations.append(f"{key}:{totals[key]}!={expected_value}")
    if len(readouts) != int(expected["readouts"]):
        violations.append(f"readouts:{len(readouts)}")
    lengths = [int((readout.get("generation_summary") or {}).get(
        "generated_tokens", -1)) for readout in readouts]
    if lengths != list(expected["generation_lengths_by_readout"]):
        violations.append(f"generation_lengths:{lengths}")
    observations = int(totals["visual_observation_calls"])
    encoder = int(totals["visual_encoder_calls"])
    misses = int(totals["visual_encoder_cache_misses"])
    hits = int(totals["visual_encoder_cache_hits"])
    if observations != int(expected["visual_observation_calls"]):
        violations.append(
            f"visual_observation_calls:{observations}!="
            f"{expected['visual_observation_calls']}")
    if encoder != misses:
        violations.append(f"visual_encoder_calls!=cache_misses:{encoder}!={misses}")
    if misses + hits != observations:
        violations.append(
            f"cache_misses+hits!=observations:{misses}+{hits}!={observations}")
    if encoder > int(expected["visual_encoder_calls_max"]):
        violations.append(
            f"visual_encoder_calls:{encoder}>"
            f"{expected['visual_encoder_calls_max']}")
    if misses > int(expected["visual_encoder_cache_misses_max"]):
        violations.append(
            f"visual_encoder_cache_misses:{misses}>"
            f"{expected['visual_encoder_cache_misses_max']}")
    projectors = int(totals["projector_passes"])
    if projectors > int(expected["projector_passes_max"]):
        violations.append(f"projector_passes:{projectors}")
    return {
        "ok": not violations,
        "violations": violations,
        "totals": totals,
        "expected": dict(expected),
        "visual_observation_calls": observations,
        "visual_encoder_calls": encoder,
        "visual_encoder_cache_misses": misses,
        "visual_encoder_cache_hits": hits,
        "projector_passes": projectors,
        "arithmetic":
            "4,166 generated + 882 prepares + 588 logits + 8 update "
            "forwards = 5,644 decoder passes",
    }


def readout_nondeterminism_report(
        prefix0: Mapping[str, Any], prefix0_null: Mapping[str, Any],
        *, tolerance: float = 1e-6,
) -> dict[str, Any]:
    violations: list[str] = []
    fields = []
    for role in ("answer", "caption"):
        first = _finite(((prefix0.get("teacher_forced") or {}).get(role)))
        second = _finite(((prefix0_null.get("teacher_forced") or {}).get(role)))
        if first is None or second is None:
            violations.append(f"{role}:missing")
            continue
        delta = abs(first - second)
        fields.append({"role": role, "first": first, "second": second,
                       "absolute_delta": delta})
        if delta > float(tolerance):
            violations.append(f"{role}:loss_delta:{delta}")
    for field in ("projector_state_sha256", "generation_digest"):
        first = prefix0.get(field)
        second = prefix0_null.get(field)
        if first is not None and first != second:
            violations.append(f"{field}:changed")
    return {
        "ok": not violations,
        "violations": violations,
        "tolerance": float(tolerance),
        "fields": fields,
        "deterministic": not violations,
    }


def step1_decision(
        *, prefix0: Mapping[str, Any], prefix0_null: Mapping[str, Any],
        prefix1: Mapping[str, Any], update: Mapping[str, Any],
        accounting: Mapping[str, Any], anchors: Mapping[str, Any] = STEP1_ANCHORS,
        nondeterminism: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one step-1 status while preserving the v1 verdict."""
    base = {
        "curve_v1_verdict": V1_VERDICT,
        "curve_verdict_preserved": True,
        "verdict_change": False,
        "dev_authorized": False,
        "heldout_authorized": False,
        "v0_authorized": False,
        "promotion": False,
        "retry_authorized": False,
        "generated_answer_conversion_claim": False,
    }
    nondeterminism = nondeterminism or readout_nondeterminism_report(
        prefix0, prefix0_null,
        tolerance=float(anchors["repeat_abs_tolerance"]))
    if not nondeterminism["ok"]:
        return {**base, "status": STEP1_STATUS_NONDETERMINISTIC,
                "nondeterminism": nondeterminism,
                "update_performed": False,
                "reason": "the repeated prefix-0 readout is nondeterministic"}
    if accounting.get("ok") is not True:
        return {**base, "status": STEP1_STATUS_ENGINEERING,
                "nondeterminism": nondeterminism,
                "accounting": accounting,
                "reason": "measured work does not reconcile"}
    if update.get("optimizer_steps") != 1 or update.get("backward_calls") != 4:
        return {**base, "status": STEP1_STATUS_ENGINEERING,
                "nondeterminism": nondeterminism,
                "accounting": accounting,
                "reason": "the single update did not execute exactly once"}
    if prefix0.get("projector_state_sha256") != anchors[
            "prefix0_projector_sha256"]:
        return {**base, "status": STEP1_STATUS_ENGINEERING,
                "nondeterminism": nondeterminism,
                "accounting": accounting,
                "reason": "prefix-0 projector digest differs from the anchor"}
    if prefix1.get("projector_state_sha256") != anchors[
            "prefix1_projector_sha256"]:
        return {**base, "status": STEP1_STATUS_ENGINEERING,
                "nondeterminism": nondeterminism,
                "accounting": accounting,
                "reason": "prefix-1 projector digest differs from the anchor"}
    first = prefix0.get("teacher_forced") or {}
    second = prefix1.get("teacher_forced") or {}
    increased = all(
        _finite(first.get(role)) is not None
        and _finite(second.get(role)) is not None
        and float(second[role]) > float(first[role])
        for role in ("answer", "caption"))
    return {
        **base,
        "status": (STEP1_STATUS_SUPPORTED if increased
                   else STEP1_STATUS_REFUTED),
        "nondeterminism": nondeterminism,
        "accounting": accounting,
        "prefix0_teacher_forced": first,
        "prefix1_teacher_forced": second,
        "overshoot_observed": increased,
        "reason": ("the first NativeFusedAdamW step increases both "
                   "readout losses" if increased
                   else "the first step does not increase both readout "
                        "losses"),
    }


def build_derived_step1_receipt(
        *, config: Mapping[str, Any],
        readouts: Sequence[Mapping[str, Any]],
        accounting: Mapping[str, Any],
        update: Mapping[str, Any],
        original_report: Mapping[str, Any],
        source_bindings: Mapping[str, Mapping[str, Any]],
        replay_command: str,
) -> dict[str, Any]:
    """Recompute a corrected step-1 receipt from preserved attempt artifacts.

    The original report is never modified.  This function is CPU-only and
    requires the corrected accounting validator to reconcile before it records
    a derived decision.
    """
    for path, binding in source_bindings.items():
        _require_sha256(binding.get("sha256"), f"source sha256:{path}")
    require(original_report.get("status") == STEP1_STATUS_ENGINEERING,
            "The preserved report is not the expected failed original")
    nondeterminism = readout_nondeterminism_report(
        readouts[0], readouts[1],
        tolerance=float(config["nondeterminism_stop"]
                        ["teacher_forced_abs_tolerance"]))
    accounting_report = validate_step1_accounting(
        readouts, accounting,
        expected=config["expected_measured_work"])
    require(accounting_report["ok"],
            f"Corrected accounting does not reconcile: "
            f"{accounting_report['violations']}")
    decision = step1_decision(
        prefix0=readouts[0], prefix0_null=readouts[1],
        prefix1=readouts[2], update=update,
        accounting=accounting_report, nondeterminism=nondeterminism)
    require(decision["status"] == STEP1_STATUS_SUPPORTED,
            f"Corrected decision is not supported: {decision['status']}")
    return {
        "format": DERIVED_STEP1_FORMAT,
        "version": VERSION,
        "receipt_class": DERIVED_STEP1_CLASS,
        "status": DERIVED_STEP1_CLASS,
        "original_status": STEP1_STATUS_ENGINEERING,
        "original_status_preserved": True,
        "original_report_unchanged": True,
        "no_native_execution": True,
        "native_execution": False,
        "source_bindings": {
            str(path): dict(binding)
            for path, binding in sorted(source_bindings.items())
        },
        "validator_correction": {
            "problem": (
                "validate_step1_accounting summed per-readout visual counters "
                "and then added the same cumulative training_global counters"),
            "resolution": (
                "visual observation/encoder/projector totals now come from "
                "the whole-run cumulative counters exactly once"),
            "expected_visual_observation_calls": 220,
            "encoder_and_miss_cap": 220,
            "cache_identity": "misses == encoder_calls; misses + hits == observations",
        },
        "accounting": accounting_report,
        "nondeterminism": nondeterminism,
        "decision": decision,
        "corrected_status": decision["status"],
        "curve_v1_verdict": V1_VERDICT,
        "curve_verdict_preserved": True,
        "generated_answer_correctness": {
            "claimed": False,
            "note": "rank/loss movement is not generated-answer conversion",
        },
        "image_dependence": {
            "claimed": False,
            "note": "the derived receipt audits accounting, not perceptual quality",
        },
        "dev_authorized": False,
        "dev_accessed": False,
        "heldout_authorized": False,
        "heldout_accessed": False,
        "v0_authorized": False,
        "v0_run": False,
        "promotion": False,
        "retry_authorized": False,
        "replay": {
            "command": replay_command,
            "deterministic": True,
            "native_execution": False,
            "compares": [
                "source_bindings",
                "corrected accounting",
                "nondeterminism",
                "derived decision",
            ],
        },
    }


def validate_derived_step1_receipt(
        receipt: Mapping[str, Any], *, root: str | Path | None = None
) -> dict[str, Any]:
    require(isinstance(receipt, Mapping), "Derived receipt is missing")
    require(receipt.get("format") == DERIVED_STEP1_FORMAT,
            "Derived receipt format differs")
    require(receipt.get("receipt_class") == DERIVED_STEP1_CLASS
            and receipt.get("status") == DERIVED_STEP1_CLASS,
            "Derived receipt is not labelled derived_validator_correction")
    require(receipt.get("original_status") == STEP1_STATUS_ENGINEERING
            and receipt.get("original_status_preserved") is True,
            "Original failed status is not preserved")
    require(receipt.get("original_report_unchanged") is True
            and receipt.get("no_native_execution") is True
            and receipt.get("native_execution") is False,
            "Derived receipt must not claim native execution")
    bindings = receipt.get("source_bindings")
    require(isinstance(bindings, Mapping) and bindings,
            "Derived receipt lacks source bindings")
    for path, binding in bindings.items():
        require(isinstance(binding, Mapping)
                and isinstance(binding.get("sha256"), str)
                and len(binding["sha256"]) == 64,
                f"Source binding differs: {path}")
        if root is not None:
            bound_path = Path(root) / str(binding.get("path"))
            require(bound_path.is_file()
                    and _sha256_file(bound_path) == binding["sha256"],
                    f"Bound source bytes differ: {path}")
    require((receipt.get("accounting") or {}).get("ok") is True,
            "Derived accounting is not ok")
    decision = receipt.get("decision") or {}
    require(decision.get("status") == STEP1_STATUS_SUPPORTED,
            "Derived decision is not supported")
    require(decision.get("curve_v1_verdict") == V1_VERDICT
            and decision.get("curve_verdict_preserved") is True,
            "Derived decision changes the v1 verdict")
    for key in ("dev_authorized", "heldout_authorized", "v0_authorized",
                "promotion", "retry_authorized"):
        require(decision.get(key) is False,
                f"Derived decision authorizes {key}")
    first = decision.get("prefix0_teacher_forced") or {}
    second = decision.get("prefix1_teacher_forced") or {}
    require(first.get("answer") == STEP1_ANCHORS[
        "prefix0_teacher_forced_loss"]["answer"]
        and first.get("caption") == STEP1_ANCHORS[
            "prefix0_teacher_forced_loss"]["caption"],
        "Prefix-0 derived losses differ")
    require(second.get("answer") == STEP1_ANCHORS[
        "prefix1_teacher_forced_loss"]["answer"]
        and second.get("caption") == STEP1_ANCHORS[
            "prefix1_teacher_forced_loss"]["caption"],
        "Prefix-1 derived losses differ")
    return {
        "ok": True,
        "receipt_class": DERIVED_STEP1_CLASS,
        "corrected_status": STEP1_STATUS_SUPPORTED,
        "original_status": STEP1_STATUS_ENGINEERING,
        "sources": len(bindings),
    }



def validate_authorization_receipt(
        receipt: Mapping[str, Any], *,
        config_sha256: str,
        config: Mapping[str, Any],
) -> dict[str, Any]:
    _require_sha256(config_sha256, "config_sha256")
    require(isinstance(receipt, Mapping), "Authorization receipt is missing")
    require(receipt.get("format") == AUTHORIZATION_FORMAT,
            "Authorization receipt format differs")
    require(receipt.get("status") == "authorized",
            "Native execution is not authorized")
    require(receipt.get("config_sha256") == config_sha256,
            "Authorization receipt does not bind the exact config")
    commit = receipt.get("protocol_commit")
    require(isinstance(commit, str) and len(commit) >= 7,
            "Authorization receipt lacks a protocol commit")
    require(receipt.get("protocol_name") == config["name"],
            "Authorization receipt names a different protocol")
    require(receipt.get("native_execution_authorized") is True,
            "Authorization receipt does not authorize native execution")
    require(receipt.get("dev_access") is False
            and receipt.get("heldout_access") is False,
            "Authorization receipt opens sealed data")
    return dict(receipt)
