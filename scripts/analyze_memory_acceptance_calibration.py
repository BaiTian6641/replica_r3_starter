#!/usr/bin/env python3
"""Development-only acceptance/address calibration over immutable rankings.

This CLI replays the predeclared protocol in
``configs/memory_acceptance_calibration_v1.json`` against the pinned
causal-fresh ranking artifact.  It never runs a native job, never loads a
model, never relabels a row, never rewrites the raw rankings or the pinned
inventory, and never scores a held-out row.

The runtime acceptance decision is recomputed from readout fields only:
``readout_norm``, ``top_cosine``, ``second_cosine``, the arm-specific
``cosine_floor``/``margin_floor`` and the existing policy block.  Gold
labels (``top1`` and the artifact evaluator labels) are preserved only for
scoring after that decision.  Zero and tied readouts receive no
recoverability or acceptance credit.  Policy-blocked rows are never
accepted, although the number of blocked rows that would otherwise pass the
cosine thresholds is reported.

Value replay applies to ``query_key_value``, ``exact_key_value`` and
``context_removed``.  ``exact_key_value`` is a separate mixture diagnostic
and is never blended into candidate quality.  Address replay is restricted
to ``query_key_address``: it changes only the existing address-readout
acceptance and never reselects a candidate, regenerates a refresh attempt or
recomputes downstream refresh state.

The output is a fresh derived receipt.  It is not an authority and it is not
a native or held-out result.  Malformed, duplicate, extra, missing or
out-of-scope rows fail closed.

``--diagnosis`` selects a second, receipt-only mode.  It derives fact-only
address/value correctness, abstention, exact two-layer aggregation, separate
controls and declared gradient scope from an existing calibration receipt.
That mode never reads raw rankings, never parses historical v4 metric blocks
(which may include heldout rows), never loads a model and never trains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.memory_acceptance_v3 import (  # noqa: E402
    ALL_ARMS,
    ARM_CASE_COUNTS,
    IDENTITY_KEYS,
    PRIMARY_QUALITY_ARM,
    _expected_identity,
    _groups,
    _normalized_inventory,
    _rate,
    _sha_json,
)
from replica_lab.memory_acceptance_v4 import (  # noqa: E402
    CONFIG_FORMAT as V4_CONFIG_FORMAT,
    LAYER_COMBINATION,
    ORACLE_POPULATION,
    REQUIRED_GATES,
    VERSION as V4_VERSION,
    _block,
    _classify,
    _query_row,
    validate_config as validate_v4_config,
    validate_ranking_inventory,
)

RECEIPT_FORMAT = "replica.memory_acceptance.calibration.receipt.v1"
CONFIG_FORMAT = "replica.memory_acceptance.calibration.v1"
METRIC_VERSION = "memory.acceptance.calibration.v1"

DIAGNOSIS_FORMAT = "replica.memory_acceptance.remediation.diagnosis.v1"
DIAGNOSIS_METRIC_VERSION = "memory.acceptance.remediation.diagnosis.v1"
DIAGNOSIS_OUTPUT = (
    "reports/remediation_20261027/"
    "memory_acceptance_diagnosis_20261027.json")
SUPPORTING_V4_RECEIPTS = (
    ("native02_v4_first_process",
     "reports/memory_acceptance_20261026/"
     "causal_fresh_native02_acceptance_v4.json"),
    ("restart02_v4",
     "reports/memory_acceptance_20261026/"
     "causal_fresh_restart02_acceptance_v4.json"),
)
GRADIENT_PARAMETERS = (
    "expert_<selected>.gate",
    "expert_<selected>.up",
    "expert_<selected>.down",
)
GRADIENT_SCOPE = {
    "hypothesis": (
        "one_step_per_event_future_answer_correction_unaffected_retention"),
    "status": "declared_not_executed",
    "current_path": {
        "training_objective": (
            "one event response cross-entropy or selected-expert "
            "half-sum squared error"),
        "steps_per_event": 1,
        "gradient_parameters": list(GRADIENT_PARAMETERS),
        "gradient_parameter_count_per_event": len(GRADIENT_PARAMETERS),
        "expert_matrices_per_expert": 3,
        "expert_bias": False,
        "selected_expert_only": True,
        "unselected_experts_receive_gradient": False,
        "decoder_receives_gradient": False,
        "read_projections_receive_gradient": False,
        "writers_receive_gradient": False,
        "backbone_receives_gradient": False,
        "personality_receives_gradient": False,
        "unaffected_rows_receive_teaching_gradient": False,
        "unaffected_rows_current_teaching_gradient": 0,
        "frozen_view_requires_no_autograd_graph": True,
    },
    "next_hypothesis_scope": {
        "name": (
            "one_step_per_event_future_answer_correction_"
            "unaffected_retention"),
        "allowed_gradient_parameters": list(GRADIENT_PARAMETERS),
        "must_not_add_gradient_targets": [
            "decoder", "read_projections", "writers", "backbone",
            "personality", "unselected_experts"],
        "teaching_targets": ["future_answer", "correction"],
        "unaffected_retention_requires_explicit_teaching_signal": True,
        "current_unaffected_teaching_gradient": 0,
        "new_training_arm": False,
        "one_step_per_event_contract_unchanged": True,
        "native_execution_permitted_by_this_receipt": False,
    },
    "evidence": [
        {
            "path": "src/replica_lab/nested_learning.py",
            "symbol": "GatedExpert",
            "statement": (
                "The expert owns exactly gate, up and down parameters and "
                "has no bias."),
        },
        {
            "path": "src/replica_lab/nested_learning.py",
            "symbol": "expert_loss_gradients",
            "statement": (
                "Gradients are taken with respect to tuple(expert.parameters()) "
                "only."),
        },
        {
            "path": "src/replica_lab/nested_learning.py",
            "symbol": (
                "PlasticLearningController.apply_teaching_transaction"),
            "statement": (
                "Only the selected expert receives gradients; inactive "
                "experts are required to have no gradient."),
        },
        {
            "path": "src/replica_lab/nested_text.py",
            "symbol": "TextLearner.learn_event_response_ce",
            "statement": (
                "Response cross-entropy gradients are taken with respect to "
                "the selected expert's parameters through the frozen decoder "
                "path."),
        },
        {
            "path": "src/replica_lab/nested_learning.py",
            "symbol": "FrozenView",
            "statement": (
                "A frozen view is readonly and must not carry an autograd "
                "graph."),
        },
        {
            "path": "src/replica_lab/nested_learning.py",
            "symbol": "PersonalityIdentity",
            "statement": (
                "The personality identity is versioned conditioning and is "
                "never rewritten by learning."),
        },
    ],
}

ADDRESS_ARM = "query_key_address"
VALUE_ARMS = ("query_key_value", "exact_key_value", "context_removed")
WRONG_ADDRESS_CATEGORIES = (
    "other_source",
    "independent_world",
    "unknown_address",
    "revoked_target_other_live",
)
GATE_KEYS = tuple(REQUIRED_GATES) + ("revoked_false_recall_max",)
LAYER_SEMANTICS = {
    "raw_top1": "any_layer_valid_top1",
    "raw_top1_all": "all_layers_valid_top1",
    "accepted": "any_layer_accepted",
    "accepted_correct": "all_layers_accepted_and_correct",
    "accepted_wrong": "accepted_and_not_all_correct",
    "zero_or_tied": "all_layers_zero_or_tied",
    "zero_or_tied_credit": "none",
    "layers_are_independent_samples": False,
    "report_layer_and_query_views": True,
}
COMPACT_KEYS = (
    "n",
    "fact_n",
    "unknown_n",
    "revoked_n",
    "observed_top1",
    "raw_top1",
    "raw_top1_rate",
    "fact_recovered",
    "recoverability_rate",
    "accepted",
    "accepted_correct",
    "accepted_correct_fact",
    "accepted_correct_rate",
    "accepted_precision",
    "acceptance_coverage",
    "abstention_rate",
    "accepted_wrong",
    "false_recall",
    "false_recall_rate",
    "false_recall_kinds",
    "unknown_abstention_rate",
    "unknown_false_recall",
    "unknown_false_recall_rate",
    "revoked_abstention_rate",
    "revoked_false_recall",
    "revoked_false_recall_rate",
    "revoked_target_other_live",
    "zero_or_tied",
    "zero_or_tied_rate",
    "zero_or_tied_observed_top1",
    "zero_or_tied_recoverability_credit",
    "zero_or_tied_acceptance_credit",
    "policy_blocked",
    "policy_blocked_acceptance",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _resolve(path, root):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def _require_keys(value, keys, *, label):
    _require(isinstance(value, dict), f"{label} must be an object")
    for key in keys:
        _require(key in value, f"{label} is missing: {key}")


def _require_number(value, *, label, low=0.0, high=1.0):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{label} must be a number")
    _require(low <= value <= high, f"{label} must be within [{low}, {high}]")


def _require_sha(value, *, label):
    _require(isinstance(value, str) and len(value) == 64
             and all(char in "0123456789abcdef" for char in value.lower()),
             f"{label} must be a 64-character sha256")


def _require_bool(value, *, label, expected):
    _require(value is expected, f"{label} must be {expected!r}")


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_floors(spec, *, label, max_points, expected_arms):
    _require_keys(spec, ("cosine_floors", "margin_floors", "max_points",
                         "applies_to_arms"), label=label)
    for key in ("cosine_floors", "margin_floors"):
        values = spec[key]
        _require(isinstance(values, list) and values,
                 f"{label} {key} must be a nonempty list")
        for value in values:
            _require_number(value, label=f"{label} {key} entry")
        _require(len(values) == len(set(values)),
                 f"{label} {key} entries must be unique")
    _require(type(spec["max_points"]) is int and spec["max_points"] > 0,
             f"{label} max_points must be a positive integer")
    points = len(spec["cosine_floors"]) * len(spec["margin_floors"])
    _require(points <= spec["max_points"],
             f"{label} declares {points} points above max_points")
    _require(points == max_points,
             f"{label} must declare exactly {max_points} frontier points")
    _require(tuple(spec["applies_to_arms"]) == tuple(expected_arms),
             f"{label} applies_to_arms differs from the declared arms")


def validate_calibration_config(config):
    """Validate the predeclared calibration protocol without reading sources."""
    _require(isinstance(config, dict), "Calibration config must be an object")
    _require(config.get("format") == CONFIG_FORMAT,
             "Unsupported calibration config format")
    _require(config.get("metric_version") == METRIC_VERSION,
             "Unsupported calibration metric version")
    for key in ("declared_at_utc", "purpose"):
        _require(isinstance(config.get(key), str) and config[key],
                 f"Calibration config is missing: {key}")

    source = config.get("source")
    _require_keys(source, ("rankings", "rankings_sha256", "inventory",
                           "inventory_sha256", "expected_rows"),
                  label="Calibration source")
    for key in ("rankings", "inventory"):
        _require(isinstance(source[key], str) and source[key],
                 f"Calibration source {key} must be a path")
    _require_sha(source["rankings_sha256"], label="rankings_sha256")
    _require_sha(source["inventory_sha256"], label="inventory_sha256")
    _require(type(source["expected_rows"]) is int and source["expected_rows"] > 0,
             "Calibration expected_rows must be a positive integer")

    scope = config.get("scope")
    _require_keys(scope, ("splits", "phases", "variants", "arms",
                          "heldout_rows_scored", "heldout_metrics_reported",
                          "prepared04_heldout_reused", "layers"),
                  label="Calibration scope")
    for key in ("splits", "phases", "variants", "arms", "layers"):
        _require(isinstance(scope[key], list) and scope[key],
                 f"Calibration scope {key} must be a nonempty list")
    for key in ("splits", "phases", "variants"):
        _require(all(isinstance(item, str) and item for item in scope[key]),
                 f"Calibration scope {key} entries must be nonempty strings")
        _require(len(scope[key]) == len(set(scope[key])),
                 f"Calibration scope {key} entries must be unique")
    _require(all(type(item) is int for item in scope["layers"]),
             "Calibration scope layers must be integers")
    _require(len(scope["layers"]) == len(set(scope["layers"])),
             "Calibration scope layers must be unique")
    _require(tuple(scope["arms"]) == tuple(ALL_ARMS),
             "Calibration arms differ from the shared arm inventory")
    _require("heldout" not in scope["splits"],
             "Calibration scope must not score the heldout split")
    _require_bool(scope["heldout_rows_scored"], label="heldout_rows_scored",
                  expected=False)
    _require_bool(scope["heldout_metrics_reported"],
                  label="heldout_metrics_reported", expected=False)
    _require_bool(scope["prepared04_heldout_reused"],
                  label="prepared04_heldout_reused", expected=False)

    thresholds = config.get("native_thresholds")
    _require_keys(thresholds, ("value", "address"),
                  label="Calibration native_thresholds")
    for name in ("value", "address"):
        _require_keys(thresholds[name], ("cosine_floor", "margin_floor"),
                      label=f"Calibration native_thresholds.{name}")
        _require_number(thresholds[name]["cosine_floor"],
                        label=f"native {name} cosine_floor")
        _require_number(thresholds[name]["margin_floor"],
                        label=f"native {name} margin_floor")

    frontier = config.get("frontier")
    _require_keys(frontier, ("value", "address"),
                  label="Calibration frontier")
    _validate_floors(frontier["value"], label="Calibration value frontier",
                     max_points=25, expected_arms=VALUE_ARMS)
    _validate_floors(frontier["address"],
                     label="Calibration address frontier", max_points=25,
                     expected_arms=(ADDRESS_ARM,))
    _require_bool(frontier["value"]["recomputes_existing_readout_acceptance_only"],
                  label="value frontier recomputes_existing_readout_acceptance_only",
                  expected=True)
    _require_bool(frontier["address"]["address_arm_only"],
                  label="address frontier address_arm_only", expected=True)
    _require_bool(frontier["address"]["downstream_refresh_recomputed"],
                  label="address frontier downstream_refresh_recomputed",
                  expected=False)
    _require_bool(frontier["address"]["candidate_policy_reselected"],
                  label="address frontier candidate_policy_reselected",
                  expected=False)

    semantics = config.get("layer_semantics")
    _require(isinstance(semantics, dict), "Calibration layer_semantics must be an object")
    _require(dict(semantics) == LAYER_SEMANTICS,
             "Calibration layer semantics differ from v4 query semantics")

    controls = config.get("controls")
    _require_keys(controls, ("candidate_variant", "oracle_variant", "baselines",
                             "exact_key_value_is_mixture_diagnostic_not_upper_bound",
                             "wrong_address_categories",
                             "unknown_and_revoked_separate",
                             "policy_blocking_preserved", "no_relabeling",
                             "no_gold_inputs_to_threshold_rule"),
                  label="Calibration controls")
    _require(controls["candidate_variant"] in scope["variants"],
             "Candidate variant is not declared")
    _require(controls["oracle_variant"] in scope["variants"],
             "Oracle variant is not declared")
    _require(isinstance(controls["baselines"], list) and controls["baselines"],
             "Calibration baselines must be a nonempty list")
    _require(len(controls["baselines"]) == len(set(controls["baselines"])),
             "Calibration baselines must be unique")
    for variant in controls["baselines"]:
        _require(variant in scope["variants"],
                 f"Baseline variant is not declared: {variant}")
    _require(tuple(controls["wrong_address_categories"])
             == WRONG_ADDRESS_CATEGORIES,
             "Calibration wrong-address categories differ")
    for key in ("exact_key_value_is_mixture_diagnostic_not_upper_bound",
                "unknown_and_revoked_separate", "policy_blocking_preserved",
                "no_relabeling", "no_gold_inputs_to_threshold_rule"):
        _require_bool(controls[key], label=f"controls.{key}", expected=True)

    gates = config.get("gates")
    _require(isinstance(gates, dict), "Calibration gates must be an object")
    for gate in GATE_KEYS:
        _require(gate in gates, f"Calibration gate is missing: {gate}")
        _require_number(gates[gate], label=f"gate {gate}")

    rules = config.get("decision_rules")
    _require_keys(rules, ("oracle_cannot_meet_gates",
                          "oracle_passes_candidate_fails",
                          "candidate_passes_development",
                          "prepared04_heldout"),
                  label="Calibration decision_rules")
    for key, value in rules.items():
        _require(isinstance(value, str) and value,
                 f"Calibration decision rule {key} must be text")
    _require("spent" in rules["prepared04_heldout"],
             "Calibration must declare the prepared04 heldout as spent")

    limits = config.get("limits")
    _require_keys(limits, ("max_source_rows", "max_frontier_points_per_arm",
                           "max_output_bytes", "native_execution",
                           "model_loads", "tokenizer_loads", "gradients",
                           "optimizer_updates", "checkpoint_writes",
                           "new_generation"),
                  label="Calibration limits")
    for key in ("max_source_rows", "max_frontier_points_per_arm",
                "max_output_bytes"):
        _require(type(limits[key]) is int and limits[key] > 0,
                 f"Calibration limit {key} must be a positive integer")
    _require(limits["native_execution"] is False,
             "Calibration must not permit native execution")
    for key in ("model_loads", "tokenizer_loads", "gradients",
                "optimizer_updates", "checkpoint_writes"):
        _require(limits[key] == 0, f"Calibration limit {key} must be zero")
    _require(limits["new_generation"] is False,
             "Calibration must not permit a new generation")
    _require(limits["max_frontier_points_per_arm"] >= 25,
             "Calibration frontier limit is below the declared grid")

    output = config.get("output")
    _require_keys(output, ("receipt", "raw_rankings_modified",
                           "raw_labels_modified",
                           "receipt_is_derived_not_authority",
                           "heldout_opened"), label="Calibration output")
    _require(isinstance(output["receipt"], str) and output["receipt"].endswith(".json"),
             "Calibration receipt path must be a .json path")
    _require_bool(output["raw_rankings_modified"],
                  label="raw_rankings_modified", expected=False)
    _require_bool(output["raw_labels_modified"],
                  label="raw_labels_modified", expected=False)
    _require_bool(output["receipt_is_derived_not_authority"],
                  label="receipt_is_derived_not_authority", expected=True)
    _require_bool(output["heldout_opened"], label="heldout_opened",
                  expected=False)

    non_claims = config.get("non_claims")
    _require(isinstance(non_claims, list) and non_claims,
             "Calibration non_claims must be a nonempty list")
    _require(all(isinstance(item, str) and item for item in non_claims),
             "Calibration non_claims entries must be text")
    return config


def _load_rankings(path):
    data = Path(path).read_bytes()
    rows = []
    for lineno, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed ranking JSON at line {lineno}: {exc}") from exc
        _require(isinstance(row, dict),
                 f"Ranking line {lineno} is not an object")
        rows.append(row)
    _require(rows, "Ranking artifact is empty")
    return data, rows


def _load_inventory(path):
    data = Path(path).read_bytes()
    try:
        document = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed inventory JSON: {exc}") from exc
    _require(isinstance(document, dict), "Inventory must be an object")
    _require(isinstance(document.get("rows"), list) and document["rows"],
             "Inventory rows are missing")
    _require(isinstance(document.get("units"), dict),
             "Inventory units are missing")
    return data, document


def _v4_config(config, document):
    """Build the v4-compatible view used only to reuse strict row validation."""
    scope = config["scope"]
    return {
        "format": V4_CONFIG_FORMAT,
        "metric_version": V4_VERSION,
        "primary_quality_arm": PRIMARY_QUALITY_ARM,
        "diagnostic_arms": [arm for arm in ALL_ARMS
                            if arm != PRIMARY_QUALITY_ARM],
        "candidate_variant": config["controls"]["candidate_variant"],
        "oracle_variant": config["controls"]["oracle_variant"],
        "oracle_population": dict(ORACLE_POPULATION),
        "layer_combination": dict(LAYER_COMBINATION),
        "phase_scoping": {
            "default": "all",
            "subset_recomputes_units": True,
            "parent_inventory_pin_preserved": True,
            "revocation_required_only_when_selected": True,
        },
        "expected_inventory": {
            "rows": document["rows"],
            "units": document.get("units") or {},
            "variants": list(scope["variants"]),
            "layers": list(scope["layers"]),
            "arms": list(ALL_ARMS),
            "phases": list(scope["phases"]),
        },
        "gates": dict(config["gates"]),
        "bootstrap": {"seed": 17, "resamples": 1, "confidence": 0.9},
    }


def _scope_audit(config, document, rankings, scored, out_of_scope):
    scope = config["scope"]
    normalized = _normalized_inventory(document)
    inventory_rows = normalized["rows"]
    inventory_splits = sorted({row["split"] for row in inventory_rows})
    inventory_phases = sorted({row["phase"] for row in inventory_rows})
    declared_splits = list(scope["splits"])
    declared_phases = list(scope["phases"])
    selected_splits = set(declared_splits)
    units = document.get("units") or {}
    if "variants" in units:
        _require(units["variants"] == len(scope["variants"]),
                 "Inventory variant count differs from the calibration scope")
    if "layers" in units:
        _require(units["layers"] == len(scope["layers"]),
                 "Inventory layer count differs from the calibration scope")
    if "arm_case_counts" in units:
        _require(dict(units["arm_case_counts"]) == ARM_CASE_COUNTS,
                 "Inventory arm case counts differ from the shared inventory")

    _require(set(declared_splits) <= set(inventory_splits),
             "Declared splits are not present in the inventory")
    _require(all(split != "heldout" for split in declared_splits),
             "Calibration scope must exclude the heldout split")
    _require(set(declared_phases) == set(inventory_phases),
             "Calibration phase scope differs from the inventory phases")
    _require(tuple(scope["arms"]) == tuple(ALL_ARMS),
             "Calibration arm inventory differs")
    _require(len(scope["variants"]) == len(set(scope["variants"])),
             "Calibration variants are duplicated")

    expected_subset = [row for row in inventory_rows
                       if row["split"] in selected_splits]
    _require(expected_subset, "Scored split scope contains no inventory rows")
    expected_keys = _expected_identity(
        expected_subset, list(scope["variants"]), list(scope["layers"]),
        list(ALL_ARMS))
    scored_keys = {tuple(row[name] for name in IDENTITY_KEYS)
                   for row in scored}
    _require(len(scored_keys) == len(scored),
             "Scored rows contain duplicate identities")
    missing = sorted(set(expected_keys) - scored_keys)
    extra = sorted(scored_keys - set(expected_keys))
    _require(not missing,
             f"Scored rows are missing {len(missing)} expected identities")
    _require(not extra,
             f"Scored rows contain {len(extra)} unexpected identities")

    phase_counts = {}
    split_counts = {}
    for row in scored:
        phase_counts[row["phase"]] = phase_counts.get(row["phase"], 0) + 1
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    for phase in declared_phases:
        _require(phase_counts.get(phase, 0) > 0,
                 f"Scored rows do not cover phase: {phase}")
    for split in declared_splits:
        _require(split_counts.get(split, 0) > 0,
                 f"Scored rows do not cover split: {split}")

    out_by_split = {}
    for row in out_of_scope:
        out_by_split[row["split"]] = out_by_split.get(row["split"], 0) + 1
    _require(out_by_split.get("heldout", 0) == len(out_of_scope),
             "Out-of-scope rows are not all heldout rows")
    _require(len(scored) + len(out_of_scope) == len(rankings),
             "Scored and out-of-scope rows do not reconcile")
    expected_scored = sum(
        ARM_CASE_COUNTS[row["kind"]] * len(scope["variants"])
        * len(scope["layers"])
        for row in expected_subset)
    _require(len(scored) == expected_scored,
             "Scored row count does not match the scoped inventory units")

    arma = {row["arm"] for row in scored}
    _require(arma == set(ALL_ARMS),
             "Scored rows do not cover every declared arm")
    variants = {row["variant"] for row in scored}
    _require(variants == set(scope["variants"]),
             "Scored rows do not cover every declared variant")
    layers = {row["layer"] for row in scored}
    _require(layers == set(scope["layers"]),
             "Scored rows do not cover every declared layer")

    return {
        "declared_splits": declared_splits,
        "inventory_splits": inventory_splits,
        "declared_phases": declared_phases,
        "inventory_phases": inventory_phases,
        "scored_rows": len(scored),
        "out_of_scope_rows": len(out_of_scope),
        "out_of_scope_by_split": dict(sorted(out_by_split.items())),
        "heldout_rows_present": out_by_split.get("heldout", 0),
        "heldout_rows_scored": 0,
        "scored_rows_by_split": dict(sorted(split_counts.items())),
        "scored_rows_by_phase": {
            phase: phase_counts.get(phase, 0) for phase in declared_phases},
        "scored_rows_by_phase_all": dict(sorted(phase_counts.items())),
        "expected_scored_rows": len(expected_keys),
        "scored_rows_match_inventory": True,
        "full_split_scope_used": selected_splits == set(inventory_splits),
        "phases_in_scope": declared_phases,
        "revocation_in_scope": "revocation" in declared_phases,
        "splits": declared_splits,
        "phases": declared_phases,
        "variants": list(scope["variants"]),
        "arms": list(ALL_ARMS),
        "layers": list(scope["layers"]),
    }


def _would_accept(row, floors):
    """Threshold decision from readout fields only, ignoring policy blocking.

    ``top1``, ``target_entry_id``, ``false_recall_kind``, ``cross_world_selected``
    and the artifact ``accepted`` flag are deliberately not read here.  They
    are scoring inputs consumed only after this decision has been made.
    """
    norm = row["readout_norm"]
    _require(_is_number(norm) and norm >= 0,
             "Runtime readout_norm must be nonnegative")
    top = row.get("top_cosine")
    second = row.get("second_cosine")
    _require(_is_number(top), "Runtime top_cosine must be numeric")
    _require(second is None or _is_number(second),
             "Runtime second_cosine must be numeric or null")
    usable = norm != 0 and not (second is not None and top == second)
    if not usable:
        return False
    margin = top - second if second is not None else top
    return (top >= floors["cosine_floor"]
            and margin >= floors["margin_floor"])


def _policy_block(row):
    policy_blocked = row.get("policy_blocked")
    if policy_blocked is None:
        if row.get("stale_blocked"):
            policy_blocked = "stale"
        elif row.get("revoked_blocked"):
            policy_blocked = "revoked"
    return policy_blocked


def _runtime_decision(row, floors):
    """Runtime acceptance decision; policy blocking is applied after thresholds."""
    if not _would_accept(row, floors):
        return False
    return _policy_block(row) is None


def _wrong_category(row):
    """Scoring-only wrong-address category; never a runtime threshold input."""
    kind = row["kind"]
    if kind == "revoked":
        return "revoked_target_other_live"
    if kind == "unknown":
        return "unknown_address"
    if row.get("cross_world_selected"):
        return "independent_world"
    return "other_source"


def _base_rows(scored):
    """Classify rows once, with runtime acceptance recomputed per threshold."""
    bases = []
    for row in scored:
        _require(_is_number(row.get("top_cosine")),
                 "Scored row is missing a numeric top_cosine")
        _require(row.get("second_cosine") is None
                 or _is_number(row.get("second_cosine")),
                 "Scored row has a malformed second_cosine")
        shadow = dict(row)
        shadow["accepted"] = False  # runtime acceptance is recomputed per grid point
        classified = _classify(shadow)
        classified["top_cosine"] = float(row["top_cosine"])
        classified["second_cosine"] = (None if row.get("second_cosine") is None
                                       else float(row["second_cosine"]))
        classified["readout_norm"] = float(row["readout_norm"])
        classified["artifact_accepted"] = bool(row["accepted"])
        classified["would_accept"] = False
        classified["policy_blocked_would_accept"] = False
        classified["wrong_categories"] = ()
        bases.append(classified)
    return bases


def _consent(base, floors):
    """Return a classified row with runtime acceptance applied."""
    top = base["top_cosine"]
    second = base["second_cosine"]
    margin = top - second if second is not None else top
    would = (not base["zero_or_tied"]
             and top >= floors["cosine_floor"]
             and margin >= floors["margin_floor"])
    accepted = would and base["policy_blocked"] is None
    row = dict(base)
    row["accepted"] = accepted
    row["accepted_correct"] = accepted and base["raw_top1"]
    row["accepted_wrong"] = accepted and not row["accepted_correct"]
    row["false_recall"] = row["accepted_wrong"]
    row["false_recall_kind"] = (_wrong_category(base)
                                if row["accepted_wrong"] else None)
    row["would_accept"] = would
    row["policy_blocked_would_accept"] = bool(
        base["policy_blocked"] is not None and would)
    row["wrong_categories"] = tuple(
        sorted({_wrong_category(base)})) if row["accepted_wrong"] else ()
    return row


def _replay_query_rows(group_rows, floors):
    rows = []
    for key in sorted(group_rows):
        group = [_consent(base, floors) for base in group_rows[key]]
        query = _query_row(group)
        query["would_accept"] = any(row["would_accept"] for row in group)
        query["policy_blocked_would_accept"] = any(
            row["policy_blocked_would_accept"] for row in group)
        query["wrong_categories"] = tuple(sorted({
            category for row in group for category in row["wrong_categories"]}))
        rows.append(query)
    return rows


def _compact(block, query_rows=None):
    compact = {key: block.get(key) for key in COMPACT_KEYS}
    if query_rows is not None:
        compact["would_accept"] = sum(row["would_accept"]
                                      for row in query_rows)
        compact["policy_blocked_would_accept"] = sum(
            row["policy_blocked_would_accept"] for row in query_rows)
    return compact


def _categories(query_rows):
    counts = {category: 0 for category in WRONG_ADDRESS_CATEGORIES}
    for query in query_rows:
        if query["false_recall_kind"]:
            counts[query["false_recall_kind"]] = counts.get(
                query["false_recall_kind"], 0) + 1
    return counts


def _layer_categories(layer_rows):
    """Wrong-address categories over accepted-but-incorrect readout layers.

    The query view combines layers and can mark a query as not-all-correct when
    one layer abstains while the other is correct.  Wrong-address categories
    therefore use the readout layer view, where an accepted wrong address is
    observable, and report it alongside the query view rather than conflating
    the two.
    """
    counts = {category: 0 for category in WRONG_ADDRESS_CATEGORIES}
    for row in layer_rows:
        if row["accepted_wrong"]:
            category = row["false_recall_kind"]
            counts[category] = counts.get(category, 0) + 1
    return counts


def _layer_rows(group_rows, floors):
    rows = []
    for key in sorted(group_rows):
        for base in group_rows[key]:
            rows.append(_consent(base, floors))
    return sorted(rows, key=lambda row: (row["variant"], row["arm"],
                                         row["phase"], row["episode_id"],
                                         row["state"], row["case_id"],
                                         row["layer"]))


def _counts(query_rows, layer_rows):
    return {
        "query_groups": len(query_rows),
        "layer_rows": len(layer_rows),
        "facts": sum(row["kind"] == "fact" for row in query_rows),
        "unknown": sum(row["kind"] == "unknown" for row in query_rows),
        "revoked": sum(row["kind"] == "revoked" for row in query_rows),
        "episodes": len({(row["episode_id"], row["state"])
                         for row in query_rows}),
        "worlds": len({row["world_id"] for row in query_rows}),
        "cohorts": len({row["cohort"] for row in query_rows}),
        "layers": len({row["layer"] for row in layer_rows}),
    }


def _address_diagnostics(query_rows, layer_rows):
    fact_n = sum(row["kind"] == "fact" for row in query_rows)
    accepted = sum(row["accepted"] for row in query_rows)
    accepted_hit = sum(row["accepted_correct"] for row in query_rows)
    accepted_any_hit = sum(
        row["accepted"] and row["raw_top1"] for row in query_rows)
    layer_accepted = sum(row["accepted"] for row in layer_rows)
    layer_accepted_hit = sum(row["accepted_correct"] for row in layer_rows)
    disputes = sum(len(row["wrong_categories"]) > 1 for row in query_rows)
    layer_disputes = 0
    grouped = _groups(layer_rows, lambda row: (
        row["phase"], row["episode_id"], row["state"], row["case_id"]))
    for group in grouped.values():
        category_sets = {row["wrong_categories"] for row in group
                         if row["accepted_wrong"]}
        if len(category_sets) > 1:
            layer_disputes += 1
    unknown = [row for row in query_rows if row["kind"] == "unknown"]
    revoked = [row for row in query_rows if row["kind"] == "revoked"]
    return {
        "address_hit": sum(row["raw_top1"] for row in query_rows),
        "address_hit_rate": _rate(
            sum(row["raw_top1"] for row in query_rows), fact_n),
        "address_accepted": accepted,
        "address_accepted_hit": accepted_hit,
        "address_accepted_any_layer_hit": accepted_any_hit,
        "address_accepted_hit_rate": _rate(accepted_hit, fact_n),
        "address_precision": _rate(accepted_hit, accepted),
        "address_query_accepted_not_all_layers_correct": sum(
            row["accepted_wrong"] for row in query_rows),
        "address_layer_accepted": layer_accepted,
        "address_layer_accepted_hit": layer_accepted_hit,
        "address_layer_precision": _rate(layer_accepted_hit, layer_accepted),
        "wrong_address_categories": _layer_categories(layer_rows),
        "query_false_recall_kinds": _categories(query_rows),
        "query_category_disagreements": disputes,
        "layer_category_disagreements": layer_disputes,
        "unknown_n": len(unknown),
        "unknown_accepted": sum(row["accepted"] for row in unknown),
        "unknown_accepted_rate": _rate(
            sum(row["accepted"] for row in unknown), len(unknown)),
        "unknown_false_recall_rate": _rate(
            sum(row["false_recall"] for row in unknown), len(unknown)),
        "revoked_n": len(revoked),
        "revoked_accepted": sum(row["accepted"] for row in revoked),
        "revoked_accepted_rate": _rate(
            sum(row["accepted"] for row in revoked), len(revoked)),
        "revoked_false_recall_rate": _rate(
            sum(row["false_recall"] for row in revoked), len(revoked)),
        "note": ("Address acceptance is a query_key_address lookup diagnostic "
                 "over existing readout rows.  It does not reselect candidates, "
                 "regenerate refresh attempts or recompute downstream state.  "
                 "Wrong-address categories use accepted-but-incorrect readout "
                 "layers; the query view additionally reports when not all "
                 "layers accepted and hit."),
    }


def _native_metrics(config, group_index):
    scope = config["scope"]
    thresholds = config["native_thresholds"]
    arms = {}
    for arm in scope["arms"]:
        floors = (thresholds["address"] if arm == ADDRESS_ARM
                  else thresholds["value"])
        variants = {}
        for variant in scope["variants"]:
            key = (arm, variant)
            _require(group_index.get(key),
                     f"Missing arm/variant coverage: {arm}/{variant}")
            query_rows = _replay_query_rows(group_index[key], floors)
            layer_rows = _layer_rows(group_index[key], floors)
            entry = {
                "query": _block(query_rows),
                "readout": _block(layer_rows),
                "correction": _block(
                    row for row in query_rows
                    if row["target_role"] == "correction_target"),
                "unaffected": _block(
                    row for row in query_rows
                    if row["target_role"] == "retained"),
                "unknown": _block(row for row in query_rows
                                  if row["kind"] == "unknown"),
                "revoked": _block(row for row in query_rows
                                  if row["kind"] == "revoked"),
                "by_layer": {
                    str(layer): _block(row for row in layer_rows
                                       if row["layer"] == layer)
                    for layer in scope["layers"]},
                "by_cohort": {
                    cohort: _block(row for row in query_rows
                                   if row["cohort"] == cohort)
                    for cohort in sorted({row["cohort"]
                                          for row in query_rows})},
                "counts": _counts(query_rows, layer_rows),
            }
            if arm == ADDRESS_ARM:
                entry["address_diagnostics"] = _address_diagnostics(
                    query_rows, layer_rows)
            variants[variant] = entry
        arms[arm] = {"variants": variants}
    return arms


def _value_gate_checks(config, quality, correction, unaffected, unknown,
                       revoked, readout, oracle_quality):
    gates = config["gates"]
    zero_credit_excluded = all(
        block["zero_or_tied_recoverability_credit"] == 0
        and block["zero_or_tied_acceptance_credit"] == 0
        for block in (quality, readout, oracle_quality))
    checks = {
        "accepted_correct_rate": (
            quality["accepted_correct_rate"] is not None
            and quality["accepted_correct_rate"]
            >= gates["accepted_correct_rate_min"]),
        "accepted_precision": (
            quality["accepted_precision"] is not None
            and quality["accepted_precision"]
            >= gates["accepted_precision_min"]),
        "correction": (
            correction["accepted_correct_rate"] is not None
            and correction["accepted_correct_rate"]
            >= gates["correction_accepted_correct_rate_min"]),
        "unaffected": (
            unaffected["accepted_correct_rate"] is not None
            and unaffected["accepted_correct_rate"]
            >= gates["unaffected_accepted_correct_rate_min"]),
        "unknown_false_recall": (
            unknown["unknown_false_recall_rate"] is not None
            and unknown["unknown_false_recall_rate"]
            <= gates["unknown_false_recall_max"]),
        "revoked_false_recall": (
            revoked["revoked_false_recall_rate"] is not None
            and revoked["revoked_false_recall_rate"]
            <= gates["revoked_false_recall_max"]),
        "zero_or_tied_credit_excluded": zero_credit_excluded,
        "oracle_answerable_fact_recoverability": (
            oracle_quality["recoverability_rate"] is not None
            and oracle_quality["recoverability_rate"]
            >= gates["oracle_raw_top1_rate_min"]),
    }
    observed = {
        "accepted_correct_rate": quality["accepted_correct_rate"],
        "accepted_precision": quality["accepted_precision"],
        "correction_accepted_correct_rate": correction[
            "accepted_correct_rate"],
        "unaffected_accepted_correct_rate": unaffected[
            "accepted_correct_rate"],
        "unknown_false_recall_rate": unknown["unknown_false_recall_rate"],
        "revoked_false_recall_rate": revoked["revoked_false_recall_rate"],
        "oracle_answerable_fact_recoverability": oracle_quality[
            "recoverability_rate"],
        "acceptance_coverage": quality["acceptance_coverage"],
    }
    return checks, observed


def _value_checks(config, query_rows, layer_rows, oracle_quality):
    quality = _block(query_rows)
    return _value_gate_checks(
        config, quality,
        _block(row for row in query_rows
               if row["target_role"] == "correction_target"),
        _block(row for row in query_rows
               if row["target_role"] == "retained"),
        _block(row for row in query_rows if row["kind"] == "unknown"),
        _block(row for row in query_rows if row["kind"] == "revoked"),
        _block(layer_rows), oracle_quality)


def _frontier_quality(entry):
    return {key: entry[key] for key in COMPACT_KEYS}


def _baseline_labels(baselines):
    labels = {}
    used = set()
    for index, variant in enumerate(baselines):
        if "shuffled" in variant and "shuffled" not in used:
            label = "shuffled"
        elif variant.endswith("/disabled") and "disabled" not in used:
            label = "disabled"
        elif variant.endswith("/full") and "full" not in used:
            label = "full"
        else:
            label = f"baseline_{index + 1}"
        used.add(label)
        labels[label] = variant
    return labels


def _delta(left, right):
    if left is None or right is None:
        return None
    return left - right


def _value_frontier(config, group_index, oracle_quality):
    declared = config["frontier"]["value"]
    candidate = config["controls"]["candidate_variant"]
    oracle = config["controls"]["oracle_variant"]
    points = []
    for cosine_floor in declared["cosine_floors"]:
        for margin_floor in declared["margin_floors"]:
            floors = {"cosine_floor": cosine_floor,
                      "margin_floor": margin_floor}
            arms_out = {}
            candidate_query_rows = None
            oracle_query_rows = None
            for arm in VALUE_ARMS:
                variants_out = {}
                for variant in config["scope"]["variants"]:
                    key = (arm, variant)
                    _require(group_index.get(key),
                             f"Missing frontier arm/variant coverage: {arm}/{variant}")
                    query_rows = _replay_query_rows(group_index[key], floors)
                    variants_out[variant] = _compact(
                        _block(query_rows), query_rows)
                    if variant == candidate and arm == PRIMARY_QUALITY_ARM:
                        candidate_query_rows = query_rows
                    if variant == oracle and arm == PRIMARY_QUALITY_ARM:
                        oracle_query_rows = query_rows
                arms_out[arm] = {"variants": variants_out}
            _require(candidate_query_rows is not None,
                     "Candidate frontier rows are missing")
            _require(oracle_query_rows is not None,
                     "Oracle frontier rows are missing")
            checks, observed = _value_checks(
                config, candidate_query_rows,
                _layer_rows(group_index[(PRIMARY_QUALITY_ARM, candidate)],
                            floors), oracle_quality)
            oracle_checks, oracle_observed = _value_checks(
                config, oracle_query_rows,
                _layer_rows(group_index[(PRIMARY_QUALITY_ARM, oracle)],
                            floors), oracle_quality)
            value_checks = {key: checks[key] for key in (
                "accepted_correct_rate", "accepted_precision", "correction",
                "unaffected", "unknown_false_recall", "revoked_false_recall",
                "zero_or_tied_credit_excluded")}
            oracle_value_checks = {key: oracle_checks[key] for key in (
                "accepted_correct_rate", "accepted_precision", "correction",
                "unaffected", "unknown_false_recall", "revoked_false_recall",
                "zero_or_tied_credit_excluded")}
            points.append({
                "cosine_floor": cosine_floor,
                "margin_floor": margin_floor,
                "arms": arms_out,
                "candidate_checks": value_checks,
                "candidate_observed": observed,
                "candidate_passes_value_gates": all(value_checks.values()),
                "oracle_checks": oracle_value_checks,
                "oracle_observed": oracle_observed,
                "oracle_passes_value_gates": all(oracle_value_checks.values()),
                "oracle_check": checks["oracle_answerable_fact_recoverability"],
                "passes_all_gates": (all(value_checks.values())
                                     and all(oracle_value_checks.values())
                                     and checks["oracle_answerable_fact_recoverability"]),
            })
    _require(len(points) == declared["max_points"],
             "Value frontier point count differs from max_points")
    return {
        "declared": {
            "cosine_floors": list(declared["cosine_floors"]),
            "margin_floors": list(declared["margin_floors"]),
            "max_points": declared["max_points"],
            "applies_to_arms": list(declared["applies_to_arms"]),
        },
        "point_count": len(points),
        "applies_to_arms": list(VALUE_ARMS),
        "candidate_variant": candidate,
        "candidate_arm": PRIMARY_QUALITY_ARM,
        "recomputes_existing_readout_acceptance_only": True,
        "blends_exact_key_value_into_candidate": False,
        "points": points,
    }


def _address_frontier(config, group_index):
    declared = config["frontier"]["address"]
    candidate = config["controls"]["candidate_variant"]
    points = []
    for cosine_floor in declared["cosine_floors"]:
        for margin_floor in declared["margin_floors"]:
            floors = {"cosine_floor": cosine_floor,
                      "margin_floor": margin_floor}
            variants_out = {}
            candidate_query_rows = None
            candidate_layer_rows = None
            for variant in config["scope"]["variants"]:
                key = (ADDRESS_ARM, variant)
                _require(group_index.get(key),
                         f"Missing address arm/variant coverage: {variant}")
                query_rows = _replay_query_rows(group_index[key], floors)
                layer_rows = _layer_rows(group_index[key], floors)
                compact = _compact(_block(query_rows), query_rows)
                compact["wrong_address_categories"] = _layer_categories(
                    layer_rows)
                compact["query_false_recall_kinds"] = _categories(query_rows)
                variants_out[variant] = compact
                if variant == candidate:
                    candidate_query_rows = query_rows
                    candidate_layer_rows = layer_rows
            _require(candidate_query_rows is not None,
                     "Address candidate rows are missing")
            diagnostics = _address_diagnostics(
                candidate_query_rows, candidate_layer_rows)
            points.append({
                "cosine_floor": cosine_floor,
                "margin_floor": margin_floor,
                "address_only": True,
                "downstream_refresh_recomputed": False,
                "candidate_policy_reselected": False,
                "arms": {ADDRESS_ARM: {"variants": variants_out}},
                "candidate_address_diagnostics": diagnostics,
            })
    _require(len(points) == declared["max_points"],
             "Address frontier point count differs from max_points")
    return {
        "declared": {
            "cosine_floors": list(declared["cosine_floors"]),
            "margin_floors": list(declared["margin_floors"]),
            "max_points": declared["max_points"],
            "applies_to_arms": list(declared["applies_to_arms"]),
        },
        "point_count": len(points),
        "applies_to_arms": [ADDRESS_ARM],
        "candidate_variant": candidate,
        "address_arm_only": True,
        "downstream_refresh_recomputed": False,
        "candidate_policy_reselected": False,
        "points": points,
    }


def _best_point(points, path):
    best = None
    for index, point in enumerate(points):
        value = point
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if value is None:
            continue
        if best is None or value > best["value"]:
            best = {"point_index": index, "value": value,
                    "cosine_floor": point["cosine_floor"],
                    "margin_floor": point["margin_floor"]}
    return best


def _frontier_summary(value_frontier, address_frontier, config):
    value_points = value_frontier["points"]
    passing = [index for index, point in enumerate(value_points)
               if point["candidate_passes_value_gates"]]
    passing_all = [index for index, point in enumerate(value_points)
                   if point["passes_all_gates"]]
    oracle_passing = [index for index, point in enumerate(value_points)
                      if point["oracle_passes_value_gates"]
                      and point["oracle_check"]]
    hit_min = config["gates"]["accepted_correct_rate_min"]
    precision_min = config["gates"]["accepted_precision_min"]
    address_joint = [
        index for index, point in enumerate(address_frontier["points"])
        if point["candidate_address_diagnostics"]["address_accepted_hit_rate"] >= hit_min
        and point["candidate_address_diagnostics"]["address_precision"] is not None
        and point["candidate_address_diagnostics"]["address_precision"] >= precision_min]
    return {
        "value_point_count": len(value_points),
        "address_point_count": len(address_frontier["points"]),
        "address_operating_targets": {
            "accepted_hit_rate_min": hit_min,
            "accepted_precision_min": precision_min,
            "descriptive_not_a_new_gate": True,
        },
        "address_points_meeting_hit_and_precision": len(address_joint),
        "address_point_indices_meeting_hit_and_precision": address_joint,
        "value_points_passing_candidate_value_gates": len(passing),
        "value_point_indices_passing_candidate_value_gates": passing,
        "value_points_passing_all_gates": len(passing_all),
        "value_point_indices_passing_all_gates": passing_all,
        "value_points_passing_oracle_gates": len(oracle_passing),
        "value_point_indices_passing_oracle_gates": oracle_passing,
        "best_query_key_value_accepted_correct_rate": _best_point(
            value_points,
            ("arms", PRIMARY_QUALITY_ARM, "variants",
             value_frontier["candidate_variant"],
             "accepted_correct_rate")),
        "best_query_key_value_accepted_precision": _best_point(
            value_points,
            ("arms", PRIMARY_QUALITY_ARM, "variants",
             value_frontier["candidate_variant"],
             "accepted_precision")),
        "best_query_key_value_acceptance_coverage": _best_point(
            value_points,
            ("arms", PRIMARY_QUALITY_ARM, "variants",
             value_frontier["candidate_variant"],
             "acceptance_coverage")),
        "best_address_accepted_hit_rate": _best_point(
            address_frontier["points"],
            ("candidate_address_diagnostics",
             "address_accepted_hit_rate")),
        "best_address_precision": _best_point(
            address_frontier["points"],
            ("candidate_address_diagnostics", "address_precision")),
        "address_points_with_acceptance": len([
            point for point in address_frontier["points"]
            if point["candidate_address_diagnostics"]["address_accepted"] > 0]),
        "note": ("Value gates are diagnostic development gates.  Address "
                 "points are address-arm lookup diagnostics and are not value "
                 "gates, candidate reselection or downstream refresh replay."),
    }


def _artifact_acceptance_audit(bases, config):
    examples = []
    mismatches = 0
    for base in bases:
        floors = (config["native_thresholds"]["address"]
                  if base["arm"] == ADDRESS_ARM
                  else config["native_thresholds"]["value"])
        runtime = _runtime_decision(base, floors)
        if runtime != base["artifact_accepted"]:
            mismatches += 1
            if len(examples) < 10:
                examples.append({
                    "arm": base["arm"],
                    "variant": base["variant"],
                    "phase": base["phase"],
                    "episode_id": base["episode_id"],
                    "state": base["state"],
                    "case_id": base["case_id"],
                    "layer": base["layer"],
                    "runtime": runtime,
                    "artifact": base["artifact_accepted"],
                })
    return {
        "comparison": "native runtime rule versus artifact accepted flag",
        "is_gate": False,
        "rows_compared": len(bases),
        "mismatches": mismatches,
        "mismatch_examples": examples,
        "reproduced": mismatches == 0,
        "note": ("The artifact accepted flag is a cross-check only; the "
                 "calibration decision is independently recomputed from "
                 "readout fields and the declared floors."),
    }


def _native_summary(arms, config):
    scope = config["scope"]
    candidate = config["controls"]["candidate_variant"]
    arm_summary = {}
    for arm in scope["arms"]:
        entry = arms[arm]["variants"][candidate]
        arm_summary[arm] = {
            "variant": candidate,
            "query": _compact(entry["query"]),
            "zero_or_tied": {
                "query": entry["query"]["zero_or_tied"],
                "readout": entry["readout"]["zero_or_tied"],
                "query_observed_top1_excluded": entry["query"][
                    "zero_or_tied_observed_top1"],
                "query_recoverability_credit": entry["query"][
                    "zero_or_tied_recoverability_credit"],
                "query_acceptance_credit": entry["query"][
                    "zero_or_tied_acceptance_credit"],
            },
            "policy_blocking": {
                "query_policy_blocked": entry["query"]["policy_blocked"],
                "query_policy_blocked_acceptance": entry["query"][
                    "policy_blocked_acceptance"],
                "blocked_rows_never_accepted": True,
            },
        }
        if arm == ADDRESS_ARM:
            arm_summary[arm]["address_diagnostics"] = entry[
                "address_diagnostics"]
    return arm_summary


def _arm_separation(arms, config):
    candidate = config["controls"]["candidate_variant"]
    metrics = {}
    for arm in config["scope"]["arms"]:
        query = arms[arm]["variants"][candidate]["query"]
        metrics[arm] = {
            "accepted_correct_rate": query["accepted_correct_rate"],
            "accepted_precision": query["accepted_precision"],
            "acceptance_coverage": query["acceptance_coverage"],
            "raw_top1_rate": query["raw_top1_rate"],
            "zero_or_tied_rate": query["zero_or_tied_rate"],
        }
    gaps = {}
    arm_names = list(config["scope"]["arms"])
    for index, left in enumerate(arm_names):
        for right in arm_names[index + 1:]:
            gap = {}
            distinct = False
            for metric in ("accepted_correct_rate", "accepted_precision",
                           "acceptance_coverage", "raw_top1_rate",
                           "zero_or_tied_rate"):
                value = _delta(metrics[left][metric], metrics[right][metric])
                gap[metric] = value
                if value is not None and value != 0:
                    distinct = True
            gaps[f"{left}|{right}"] = {"distinct": distinct, **gap}
    return {
        "variant": candidate,
        "by_arm": metrics,
        "pairwise_gaps": gaps,
        "arms_are_measurably_distinct": any(
            item["distinct"] for item in gaps.values()),
        "exact_key_value_blended_into_candidate_quality": False,
        "address_arm_is_not_value_quality": True,
    }


def _control_comparison(arms, config):
    candidate = config["controls"]["candidate_variant"]
    oracle = config["controls"]["oracle_variant"]
    labels = _baseline_labels(config["controls"]["baselines"])

    def compact(variant, arm=PRIMARY_QUALITY_ARM):
        return _compact(arms[arm]["variants"][variant]["query"])

    candidate_compact = compact(candidate)
    comparison = {
        "candidate": {
            "variant": candidate,
            "arm": PRIMARY_QUALITY_ARM,
            "compact": candidate_compact,
        },
        "oracle": {
            "variant": oracle,
            "arm": PRIMARY_QUALITY_ARM,
            "compact": compact(oracle),
            "is_legal_policy": False,
            "is_diagnostic_only": True,
        },
        "baselines": {},
        "exact_key_value_diagnostic": {
            "variant": candidate,
            "arm": "exact_key_value",
            "compact": compact(candidate, "exact_key_value"),
            "is_upper_bound": False,
            "blended_into_candidate_quality": False,
        },
    }
    for label, variant in labels.items():
        comparison["baselines"][label] = {
            "variant": variant,
            "arm": PRIMARY_QUALITY_ARM,
            "compact": compact(variant),
        }
    comparison["deltas_vs_candidate"] = {}
    for label, entry in comparison["baselines"].items():
        comparison["deltas_vs_candidate"][label] = {
            "accepted_correct_rate": _delta(
                entry["compact"]["accepted_correct_rate"],
                candidate_compact["accepted_correct_rate"]),
            "accepted_precision": _delta(
                entry["compact"]["accepted_precision"],
                candidate_compact["accepted_precision"]),
            "acceptance_coverage": _delta(
                entry["compact"]["acceptance_coverage"],
                candidate_compact["acceptance_coverage"]),
        }
    comparison["candidate_equals_full"] = (
        labels.get("full") is not None
        and comparison["baselines"]["full"]["compact"]
        == candidate_compact)
    comparison["candidate_equals_shuffled"] = (
        labels.get("shuffled") is not None
        and comparison["baselines"]["shuffled"]["compact"]
        == candidate_compact)
    comparison["candidate_equals_disabled"] = (
        labels.get("disabled") is not None
        and comparison["baselines"]["disabled"]["compact"]
        == candidate_compact)
    comparison["candidate_equals_oracle"] = (
        comparison["oracle"]["compact"] == candidate_compact)
    comparison["note"] = ("Exact-key is a separate mixture diagnostic and is "
                          "never blended into candidate quality.  The oracle "
                          "is a gold-target diagnostic, not a legal policy.")
    return comparison


def _gate_classification(config, arms, value_frontier):
    candidate = config["controls"]["candidate_variant"]
    oracle = config["controls"]["oracle_variant"]
    arm = PRIMARY_QUALITY_ARM
    entry = arms[arm]["variants"][candidate]
    oracle_entry = arms[arm]["variants"][oracle]
    coverage = {
        "candidate": entry["query"]["n"] > 0
        and entry["query"]["fact_n"] > 0,
        "correction": entry["correction"]["n"] > 0,
        "unaffected": entry["unaffected"]["n"] > 0,
        "unknown": entry["unknown"]["n"] > 0,
        "revoked": entry["revoked"]["n"] > 0,
        "oracle": oracle_entry["query"]["n"] > 0
        and oracle_entry["query"]["fact_n"] > 0,
    }
    missing = [name for name, ok in coverage.items() if not ok]
    _require(not missing,
             f"Calibration gate coverage is missing: {', '.join(missing)}")
    checks, observed = _value_gate_checks(
        config, entry["query"], entry["correction"], entry["unaffected"],
        entry["unknown"], entry["revoked"], entry["readout"],
        oracle_entry["query"])
    value_checks = {key: checks[key] for key in (
        "accepted_correct_rate", "accepted_precision", "correction",
        "unaffected", "unknown_false_recall", "revoked_false_recall",
        "zero_or_tied_credit_excluded")}
    oracle_checks, oracle_observed = _value_gate_checks(
        config, oracle_entry["query"], oracle_entry["correction"],
        oracle_entry["unaffected"], oracle_entry["unknown"],
        oracle_entry["revoked"], oracle_entry["readout"], oracle_entry["query"])
    oracle_value_checks = {key: oracle_checks[key] for key in (
        "accepted_correct_rate", "accepted_precision", "correction",
        "unaffected", "unknown_false_recall", "revoked_false_recall",
        "zero_or_tied_credit_excluded")}
    candidate_meets = all(value_checks.values())
    oracle_meets = (all(oracle_value_checks.values())
                    and oracle_checks["oracle_answerable_fact_recoverability"])
    if not oracle_meets:
        rule = "oracle_cannot_meet_gates"
    elif not candidate_meets:
        rule = "oracle_passes_candidate_fails"
    else:
        rule = "candidate_passes_development"
    thresholds = config["native_thresholds"]
    native_index = next((
        index for index, point in enumerate(value_frontier["points"])
        if point["cosine_floor"] == thresholds["value"]["cosine_floor"]
        and point["margin_floor"] == thresholds["value"]["margin_floor"]), None)
    grid_match = None
    if native_index is not None:
        point = value_frontier["points"][native_index]
        grid_match = (point["candidate_checks"] == value_checks
                      and point["oracle_checks"] == oracle_value_checks
                      and point["oracle_passes_value_gates"]
                      == all(oracle_value_checks.values()))
        _require(grid_match,
                 "Native point disagrees with the value-frontier replay")
    rules = config["decision_rules"]
    return {
        "native_point": {
            "value_cosine_floor": thresholds["value"]["cosine_floor"],
            "value_margin_floor": thresholds["value"]["margin_floor"],
            "address_cosine_floor": thresholds["address"]["cosine_floor"],
            "address_margin_floor": thresholds["address"]["margin_floor"],
            "candidate_variant": candidate,
            "candidate_arm": arm,
            "oracle_variant": oracle,
            "coverage": coverage,
            "coverage_valid": True,
            "fail_closed_on_missing_coverage": True,
            "checks": checks,
            "value_checks": value_checks,
            "oracle_checks": oracle_checks,
            "oracle_observed": oracle_observed,
            "oracle_value_checks": oracle_value_checks,
            "observed": observed,
            "candidate_meets_value_gates": candidate_meets,
            "oracle_meets_gates": oracle_meets,
            "decision": "accepted" if candidate_meets and oracle_meets
            else "rejected",
        },
        "gate_values": {
            key: config["gates"][key] for key in GATE_KEYS},
        "oracle_gate_population": "answerable_fact",
        "oracle_gate_numerator": "raw_valid_top1",
        "oracle_gate_denominator": "fact_n",
        "oracle_is_legal_policy": False,
        "zero_or_tied_credit_excluded": checks[
            "zero_or_tied_credit_excluded"],
        "exact_key_value_blended_into_candidate_quality": False,
        "address_arm_value_gated": False,
        "native_point_in_value_grid": native_index,
        "native_point_matches_value_grid": grid_match,
        "disposition": {
            "selected_rule": rule,
            "rule_text": rules[rule],
            "oracle_meets_gates": oracle_meets,
            "candidate_meets_value_gates": candidate_meets,
            "stop": True,
            "stop_reason": ("Development-only calibration; no native "
                            "confirmation, writer training, relabeling or "
                            "held-out scoring is authorized by this receipt."),
            "native_execution_permitted": False,
            "writer_training_permitted": False,
            "relabeling_permitted": False,
            "heldout_confirmation_permitted": False,
            "prepared04_heldout_disposition": rules["prepared04_heldout"],
        },
    }


def _receipt_content_digest(receipt, *, label):
    body = {key: value for key, value in receipt.items()
            if key != "content_sha256"}
    digest = receipt.get("content_sha256")
    _require(isinstance(digest, str) and len(digest) == 64,
             f"{label}: content_sha256 is missing or malformed")
    _require(_sha_json(body) == digest,
             f"{label}: content_sha256 does not cover the receipt body")
    return digest


def _validate_diagnosis_source(receipt, *, label):
    _require(isinstance(receipt, dict), f"{label}: receipt must be an object")
    _require(receipt.get("format") == RECEIPT_FORMAT,
             f"{label}: format is not {RECEIPT_FORMAT}")
    _require(receipt.get("metric_version") == METRIC_VERSION,
             f"{label}: metric_version is not {METRIC_VERSION}")
    for key in ("source", "native", "value_frontier", "address_frontier",
                "gate_classification", "frontier_summary"):
        _require(key in receipt, f"{label}: missing {key}")
    scope = receipt["source"].get("scope")
    _require(isinstance(scope, dict),
             f"{label}: source scope is missing")
    _require(scope.get("heldout_rows_scored") == 0,
             f"{label}: heldout rows were scored")
    heldout_flags = []
    for holder in (scope, receipt.get("audit") or {}):
        if "heldout_metrics_reported" in holder:
            heldout_flags.append(holder["heldout_metrics_reported"])
    _require(
        bool(heldout_flags)
        and all(flag is False for flag in heldout_flags),
        f"{label}: heldout metrics were reported or unverifiable")
    _receipt_content_digest(receipt, label=label)


def _native_point_index(frontier, floors, *, label):
    points = frontier.get("points") if isinstance(frontier, dict) else None
    _require(isinstance(points, list) and points,
             f"{label} frontier has no points")
    matches = [index for index, point in enumerate(points)
               if point.get("cosine_floor") == floors["cosine_floor"]
               and point.get("margin_floor") == floors["margin_floor"]]
    _require(len(matches) == 1,
             f"{label} frontier does not contain exactly one native point")
    return matches[0]


def _nonfact_accepted(block, *, label):
    unknown_n = int(block.get("unknown_n") or 0)
    revoked_n = int(block.get("revoked_n") or 0)
    unknown_accepted = int(block.get("unknown_false_recall") or 0)
    revoked_accepted = int(block.get("revoked_false_recall") or 0)
    _require(0 <= unknown_accepted <= unknown_n,
             f"{label}: unknown false recall exceeds unknown rows")
    _require(0 <= revoked_accepted <= revoked_n,
             f"{label}: revoked false recall exceeds revoked rows")
    for count, accepted, rate, part in (
            (unknown_n, unknown_accepted,
             block.get("unknown_abstention_rate"), "unknown"),
            (revoked_n, revoked_accepted,
             block.get("revoked_abstention_rate"), "revoked")):
        if count == 0:
            _require(accepted == 0,
                     f"{label}: {part} accepted without {part} rows")
            continue
        _require(rate is not None,
                 f"{label}: {part} abstention rate is missing")
        expected = count * (1.0 - float(rate))
        _require(abs(expected - accepted) <= 1e-6,
                 f"{label}: {part} abstention rate disagrees with false recall")
    return unknown_accepted + revoked_accepted


def _wrong_kind_counts(block, *, label):
    kinds = block.get("false_recall_kinds") or {}
    _require(isinstance(kinds, dict),
             f"{label}: false_recall_kinds must be an object")
    unexpected = sorted(set(kinds) - set(WRONG_ADDRESS_CATEGORIES)
                        - {"cross_world"})
    _require(not unexpected,
             f"{label}: unexpected false-recall categories {unexpected}")
    counts = {category: int(kinds.get(category) or 0)
              for category in WRONG_ADDRESS_CATEGORIES}
    counts["cross_world_alias"] = int(kinds.get("cross_world") or 0)
    counts["fact_wrong_count"] = (
        counts["independent_world"] + counts["other_source"]
        + counts["cross_world_alias"])
    counts["non_fact_wrong_count"] = (
        counts["unknown_address"] + counts["revoked_target_other_live"])
    return counts


def _abstention_diagnostics(block, *, label):
    unknown_n = int(block.get("unknown_n") or 0)
    revoked_n = int(block.get("revoked_n") or 0)
    unknown_accepted = int(block.get("unknown_false_recall") or 0)
    revoked_accepted = int(block.get("revoked_false_recall") or 0)
    available = (unknown_n + revoked_n) > 0
    return {
        "available": available,
        "unknown_n": unknown_n,
        "unknown_accepted": unknown_accepted,
        "unknown_abstained": unknown_n - unknown_accepted,
        "unknown_abstention_rate": _rate(
            unknown_n - unknown_accepted, unknown_n),
        "unknown_false_recall": unknown_accepted,
        "unknown_false_recall_rate": _rate(unknown_accepted, unknown_n),
        "revoked_n": revoked_n,
        "revoked_accepted": revoked_accepted,
        "revoked_abstained": revoked_n - revoked_accepted,
        "revoked_abstention_rate": _rate(
            revoked_n - revoked_accepted, revoked_n),
        "revoked_false_recall": revoked_accepted,
        "revoked_false_recall_rate": _rate(revoked_accepted, revoked_n),
        "false_recall_total": unknown_accepted + revoked_accepted,
        "false_recall_rate": _rate(
            unknown_accepted + revoked_accepted, unknown_n + revoked_n),
        "note": (
            "Unknown and revoked rows are separated from fact acceptance.  "
            "Every accepted unknown or revoked row is a false recall by "
            "construction and never receives correctness credit."),
    }


def _per_layer_diagnostics(entry, *, layers, label):
    """Per-layer fact-only readout counts for one arm or variant."""
    by_layer = entry.get("by_layer")
    _require(isinstance(by_layer, dict),
             f"{label}: per-layer blocks are missing")
    result = {}
    accepted_total = 0
    correct_total = 0
    fact_total = 0
    for layer in layers:
        block = by_layer.get(str(layer))
        _require(isinstance(block, dict),
                 f"{label}: missing layer {layer}")
        nonfact = _nonfact_accepted(
            block, label=f"{label} layer {layer}")
        accepted = int(block.get("accepted") or 0) - nonfact
        correct = int(block.get("accepted_correct_fact")
                      if "accepted_correct_fact" in block
                      else block.get("accepted_correct") or 0)
        fact_n = int(block.get("fact_n") or 0)
        _require(0 <= correct <= accepted,
                 f"{label} layer {layer}: correct layers exceed accepted")
        result[str(layer)] = {
            "layer": layer,
            "fact_readouts": fact_n,
            "fact_accepted_layers": accepted,
            "fact_accepted_correct_layers": correct,
            "fact_wrong_layers": accepted - correct,
            "fact_layer_precision": _rate(correct, accepted),
            "fact_layer_acceptance_rate": _rate(accepted, fact_n),
        }
        accepted_total += accepted
        correct_total += correct
        fact_total += fact_n
    return {
        "layers": result,
        "fact_readouts": fact_total,
        "fact_accepted_layers": accepted_total,
        "fact_accepted_correct_layers": correct_total,
        "fact_wrong_layers": accepted_total - correct_total,
        "fact_layer_precision": _rate(correct_total, accepted_total),
        "layer_sums_are_not_query_groups": True,
    }


def _aggregation_diagnostics(query, readout, *, layers, label,
                             accepted_any_hit=None,
                             layer_view_available=True):
    """Explicit fact-only query/layer aggregation for one arm or variant.

    The historical receipt reports ``accepted`` over every query group (or
    every readout layer), including unknown and revoked rows, while
    ``accepted_correct`` is fact-only.  This view keeps those historical
    fields unchanged and derives the fact-only denominators plus the exact
    two-layer decomposition.  ``accepted`` remains any-layer acceptance and
    ``accepted_correct`` remains all-layers accepted and correct.
    """
    _require(isinstance(query, dict),
             f"{label}: query block must be an object")
    _require(isinstance(layers, list) and len(layers) >= 2,
             f"{label}: at least two layers are required")
    if layer_view_available:
        _require(isinstance(readout, dict),
                 f"{label}: readout block must be an object")
    fact_n = int(query.get("fact_n") or 0)
    _require(fact_n > 0, f"{label}: no fact query groups")
    query_nonfact = _nonfact_accepted(query, label=f"{label} query")
    query_accepted = int(query.get("accepted") or 0)
    _require(query_nonfact <= query_accepted,
             f"{label}: non-fact acceptance exceeds total acceptance")
    fact_accepted = query_accepted - query_nonfact
    strict = int(query.get("accepted_correct_fact")
                 if "accepted_correct_fact" in query
                 else query.get("accepted_correct") or 0)
    _require(0 <= strict <= fact_accepted,
             f"{label}: strict groups exceed fact accepted groups")
    _require(int(query.get("accepted_correct") or 0) == strict,
             f"{label}: accepted_correct includes non-fact rows")
    not_all = fact_accepted - strict
    query_kinds = _wrong_kind_counts(query, label=f"{label} query")
    _require(query_kinds["non_fact_wrong_count"] == query_nonfact,
             f"{label}: non-fact categories disagree with abstention counts")
    wrong_groups = query_kinds["fact_wrong_count"]
    _require(wrong_groups <= not_all,
             f"{label}: wrong-layer groups exceed accepted-not-all groups")
    partial = not_all - wrong_groups
    no_accepted = fact_n - fact_accepted
    _require(no_accepted >= 0,
             f"{label}: fact acceptance exceeds fact groups")
    any_hit = int(query.get("raw_top1") or 0)
    _require(0 <= any_hit <= fact_n,
             f"{label}: any-layer correctness exceeds fact groups")
    if accepted_any_hit is not None:
        accepted_any_hit = int(accepted_any_hit)
        _require(strict <= accepted_any_hit <= fact_accepted,
                 f"{label}: accepted-any-layer-hit exceeds accepted groups")

    query_aggregation = {
        "fact_accepted_groups": fact_accepted,
        "fact_any_layer_correct_groups": any_hit,
        "fact_any_layer_correct_rate": _rate(any_hit, fact_n),
        "fact_any_layer_acceptance_rate": _rate(fact_accepted, fact_n),
        "fact_strict_all_layers_groups": strict,
        "fact_strict_group_rate": _rate(strict, fact_n),
        "fact_accepted_not_all_layers_correct_groups": not_all,
        "fact_partial_groups": partial,
        "fact_partial_group_rate": _rate(partial, fact_n),
        "fact_wrong_layer_groups": wrong_groups,
        "fact_wrong_layer_group_rate": _rate(wrong_groups, fact_n),
        "fact_no_accepted_layer_groups": no_accepted,
        "fact_no_accepted_layer_rate": _rate(no_accepted, fact_n),
        "fact_query_precision": _rate(strict, fact_accepted),
        "fact_accepted_any_layer_correct_groups": accepted_any_hit,
        "fact_accepted_any_layer_correct_rate": (
            _rate(accepted_any_hit, fact_accepted)
            if accepted_any_hit is not None else None),
        "fact_accepted_any_layer_correct_available": (
            accepted_any_hit is not None),
        "fact_partial_exactly_one_layer_groups": (
            partial if len(layers) == 2 else None),
        "fact_wrong_layer_categories": {
            "independent_world": query_kinds["independent_world"],
            "other_source": query_kinds["other_source"],
            "cross_world_alias": query_kinds["cross_world_alias"],
        },
    }

    if not layer_view_available:
        return {
            "denominators": {
                "fact_query_groups": fact_n,
                "fact_layer_readouts": None,
                "layers": list(layers),
                "layers_per_group": len(layers),
                "layers_are_independent_samples": False,
                "query_acceptance_semantics": "any_layer_accepted",
                "query_correctness_semantics":
                    "all_layers_accepted_and_correct",
            },
            "query_aggregation": query_aggregation,
            "layer_correctness": {
                "available": False,
                "reason": (
                    "role subsets are reported at query aggregation only; "
                    "this receipt has no per-role layer block"),
            },
            "decomposition": {
                "available": False,
                "reason": (
                    "the source receipt has no per-role readout block, so "
                    "the layer contribution of partial groups is unavailable"),
            },
            "historical_reconciliation": {
                "query_accepted_including_non_fact": query_accepted,
                "query_accepted_wrong": int(
                    query.get("accepted_wrong") or 0),
                "non_fact_accepted_groups": query_nonfact,
                "fact_accepted_groups": fact_accepted,
                "query_accepted_correct": strict,
                "query_raw_top1": any_hit,
                "readout_accepted_including_non_fact": None,
                "readout_accepted_wrong": None,
                "non_fact_accepted_layers": None,
                "fact_accepted_layers": None,
                "readout_accepted_correct": None,
                "note": (
                    "Historical accepted fields are preserved.  The fact-only "
                    "view subtracts accepted unknown/revoked rows; it does not "
                    "relabel or rewrite the historical fields."),
            },
            "abstention": _abstention_diagnostics(
                query, label=f"{label} query"),
        }

    layer_fact_n = int(readout.get("fact_n") or 0)
    _require(layer_fact_n == fact_n * len(layers),
             f"{label}: layer readouts do not equal fact groups * layers")
    readout_nonfact = _nonfact_accepted(
        readout, label=f"{label} readout")
    readout_accepted = int(readout.get("accepted") or 0)
    _require(readout_nonfact <= readout_accepted,
             f"{label}: non-fact layer acceptance exceeds total")
    accepted_layers = readout_accepted - readout_nonfact
    correct_layers = int(readout.get("accepted_correct_fact")
                         if "accepted_correct_fact" in readout
                         else readout.get("accepted_correct") or 0)
    _require(0 <= correct_layers <= accepted_layers,
             f"{label}: correct layers exceed accepted fact layers")
    wrong_layers = accepted_layers - correct_layers
    readout_kinds = _wrong_kind_counts(
        readout, label=f"{label} readout")
    _require(readout_kinds["non_fact_wrong_count"] == readout_nonfact,
             f"{label}: non-fact layer categories disagree with abstention")
    _require(readout_kinds["fact_wrong_count"] == wrong_layers,
             f"{label}: fact wrong-layer categories disagree with "
             "accepted-correct subtraction")

    layer_correctness = {
        "available": True,
        "fact_accepted_layers": accepted_layers,
        "fact_accepted_correct_layers": correct_layers,
        "fact_wrong_layers": wrong_layers,
        "fact_layer_precision": _rate(correct_layers, accepted_layers),
        "fact_layer_acceptance_coverage": _rate(
            accepted_layers, layer_fact_n),
        "fact_wrong_layer_rate": _rate(wrong_layers, layer_fact_n),
        "non_fact_accepted_layers": readout_nonfact,
        "fact_wrong_layer_categories": {
            "independent_world": readout_kinds["independent_world"],
            "other_source": readout_kinds["other_source"],
            "cross_world_alias": readout_kinds["cross_world_alias"],
        },
    }

    if len(layers) == 2:
        correct_from_strict = strict * 2
        correct_from_partial = partial
        _require(correct_from_strict + correct_from_partial
                 <= correct_layers,
                 f"{label}: strict/partial layer credit exceeds correct "
                 "layers")
        correct_from_wrong = (correct_layers - correct_from_strict
                              - correct_from_partial)
        accepted_from_wrong = (accepted_layers - correct_from_strict
                               - correct_from_partial)
        _require(accepted_from_wrong == correct_from_wrong + wrong_layers,
                 f"{label}: accepted-layer decomposition does not balance")
        decomposition = {
            "available": True,
            "exact": True,
            "strict_groups": strict,
            "partial_groups": partial,
            "wrong_layer_groups": wrong_groups,
            "accepted_correct_layers_from_strict_groups":
                correct_from_strict,
            "accepted_correct_layers_from_partial_groups":
                correct_from_partial,
            "accepted_correct_layers_from_wrong_layer_groups":
                correct_from_wrong,
            "accepted_wrong_layers_from_wrong_layer_groups": wrong_layers,
            "fact_accepted_layers": accepted_layers,
            "fact_accepted_correct_layers": correct_layers,
            "fact_wrong_layers": wrong_layers,
            "identity": (
                "2*strict + partial + correct_in_wrong_layer_groups == "
                "accepted_correct_layers"),
            "identity_holds": (
                correct_from_strict + correct_from_partial
                + correct_from_wrong == correct_layers),
            "accepted_identity_holds": (
                correct_from_strict + correct_from_partial
                + correct_from_wrong + wrong_layers == accepted_layers),
            "layers_per_group": 2,
        }
    else:
        decomposition = {
            "available": True,
            "exact": False,
            "reason": (
                "partial groups can contribute between one and layers-1 "
                "correct layers when layers_per_group > 2; aggregate receipt "
                "fields do not determine that split"),
            "strict_groups": strict,
            "partial_groups": partial,
            "wrong_layer_groups": wrong_groups,
            "fact_accepted_layers": accepted_layers,
            "fact_accepted_correct_layers": correct_layers,
            "fact_wrong_layers": wrong_layers,
            "layers_per_group": len(layers),
        }

    return {
        "denominators": {
            "fact_query_groups": fact_n,
            "fact_layer_readouts": layer_fact_n,
            "layers": list(layers),
            "layers_per_group": len(layers),
            "layers_are_independent_samples": False,
            "query_acceptance_semantics": "any_layer_accepted",
            "query_correctness_semantics":
                "all_layers_accepted_and_correct",
            "layer_view_semantics":
                "accepted_and_correct_per_readout_layer",
        },
        "query_aggregation": query_aggregation,
        "layer_correctness": layer_correctness,
        "decomposition": decomposition,
        "historical_reconciliation": {
            "query_accepted_including_non_fact": query_accepted,
            "query_accepted_wrong": int(
                query.get("accepted_wrong") or 0),
            "non_fact_accepted_groups": query_nonfact,
            "fact_accepted_groups": fact_accepted,
            "query_accepted_correct": strict,
            "query_raw_top1": any_hit,
            "readout_accepted_including_non_fact": readout_accepted,
            "readout_accepted_wrong": int(
                readout.get("accepted_wrong") or 0),
            "non_fact_accepted_layers": readout_nonfact,
            "fact_accepted_layers": accepted_layers,
            "readout_accepted_correct": correct_layers,
            "note": (
                "Historical accepted fields are preserved.  The fact-only "
                "view subtracts accepted unknown/revoked rows; it does not "
                "relabel or rewrite the historical fields."),
        },
        "abstention": _abstention_diagnostics(
            query, label=f"{label} query"),
    }


def _control_delta(base, other):
    base_query = base["query_aggregation"]
    other_query = other["query_aggregation"]
    base_layer = base["layer_correctness"]
    other_layer = other["layer_correctness"]
    return {
        "fact_strict_group_rate": _delta(
            other_query["fact_strict_group_rate"],
            base_query["fact_strict_group_rate"]),
        "fact_query_precision": _delta(
            other_query["fact_query_precision"],
            base_query["fact_query_precision"]),
        "fact_layer_precision": _delta(
            other_layer.get("fact_layer_precision"),
            base_layer.get("fact_layer_precision")),
        "fact_wrong_layers": _delta(
            other_layer.get("fact_wrong_layers"),
            base_layer.get("fact_wrong_layers")),
        "fact_accepted_groups": _delta(
            other_query["fact_accepted_groups"],
            base_query["fact_accepted_groups"]),
        "unknown_false_recall_rate": _delta(
            other["abstention"]["unknown_false_recall_rate"],
            base["abstention"]["unknown_false_recall_rate"]),
        "revoked_false_recall_rate": _delta(
            other["abstention"]["revoked_false_recall_rate"],
            base["abstention"]["revoked_false_recall_rate"]),
    }


def _selected_record(entry, *, label):
    query = entry["query"]
    rows = int(query.get("runtime_rows") or 0)
    accepted = int(query.get("runtime_accepted") or 0)
    correct = int(query.get("runtime_correct") or 0)
    _require(0 <= correct <= accepted <= rows,
             f"{label}: runtime selected-record counts are inconsistent")
    return {
        "runtime_rows": rows,
        "runtime_accepted": accepted,
        "runtime_correct": correct,
        "runtime_precision": _rate(correct, accepted),
        "selected_other_live": int(
            query.get("selected_other_live") or 0),
        "runtime_selection_available": bool(
            query.get("runtime_selection_available")),
        "source": "artifact runtime fields, not threshold replay",
    }


def _wrong_world_diagnostics(entry, *, label):
    query = entry["query"]
    readout = entry["readout"]
    query_kinds = _wrong_kind_counts(query, label=f"{label} query")
    layer_kinds = _wrong_kind_counts(readout, label=f"{label} readout")
    return {
        "query_cross_world_false_recall": int(
            query.get("cross_world_false_recall") or 0),
        "layer_cross_world_false_recall": int(
            readout.get("cross_world_false_recall") or 0),
        "query_independent_world_groups": query_kinds["independent_world"],
        "layer_independent_world_groups": layer_kinds["independent_world"],
        "query_other_source_groups": query_kinds["other_source"],
        "layer_other_source_groups": layer_kinds["other_source"],
        "source": "scoring-only cross_world_selected and category fields",
    }


def _provenance_receipt_record(path, *, role):
    """Hash a historical receipt without reading its metric fields.

    The historical v4 receipts cover all rows, including heldout rows, so
    their metric blocks are never parsed or used here.  Only the file hash is
    recorded as provenance.
    """
    path = Path(path)
    return {
        "role": role,
        "path": str(path),
        "available": path.is_file(),
        "sha256": _sha(path) if path.is_file() else None,
        "format": "replica.memory_acceptance.metrics.v4",
        "metric_version": V4_VERSION,
        "metrics_used": False,
        "reason": (
            "provenance hash only; the receipt may include heldout rows and "
            "its metric blocks are not parsed or used"),
    }


def _source_receipt_record(path, *, role, receipt=None):
    path = Path(path)
    record = {
        "role": role,
        "path": str(path),
        "available": path.is_file(),
        "sha256": None,
        "content_sha256": None,
        "format": None,
        "metric_version": None,
    }
    if not path.is_file():
        return record
    document = (json.loads(path.read_text()) if receipt is None else receipt)
    record.update({
        "sha256": _sha(path),
        "format": document.get("format"),
        "metric_version": document.get("metric_version"),
    })
    if "content_sha256" in document:
        record["content_sha256"] = document["content_sha256"]
    else:
        record["content_sha256"] = _sha_json(document)
    return record


def analyze_memory_diagnosis(receipt, *, receipt_path=None, root=ROOT,
                             supporting_receipts=None):
    """Derive the remediation diagnosis from a calibration receipt.

    This is a pure replay over committed receipt fields.  It does not read
    rankings, load a model, run a native command, train a writer, relabel a
    row or score a held-out row.
    """
    _validate_diagnosis_source(receipt, label="calibration receipt")
    native = receipt["native"]
    arms = native["arms"]
    controls_source = native["control_comparison"]
    candidate = controls_source["candidate"]["variant"]
    oracle = controls_source["oracle"]["variant"]
    for arm in (ADDRESS_ARM, PRIMARY_QUALITY_ARM):
        _require(arm in arms, f"calibration receipt is missing arm {arm}")
        _require(candidate in arms[arm]["variants"],
                 f"calibration receipt is missing {arm}/{candidate}")
    _require(oracle in arms[PRIMARY_QUALITY_ARM]["variants"],
             "calibration receipt is missing the oracle variant")
    layers = list(native["views"]["layer_view"]["layers"])
    thresholds = native["thresholds"]
    value_index = _native_point_index(
        receipt["value_frontier"], thresholds["value"], label="value")
    address_index = _native_point_index(
        receipt["address_frontier"], thresholds["address"], label="address")
    gates = receipt["gate_classification"]
    if "native_point_in_value_grid" in gates:
        _require(gates["native_point_in_value_grid"] == value_index,
                 "native value point index disagrees with the frontier")
    _require(gates.get("native_point_matches_value_grid") is True,
             "native value point does not match the value frontier")

    def arm_variant(arm, variant):
        entry = (arms.get(arm) or {}).get("variants", {}).get(variant)
        _require(isinstance(entry, dict),
                 f"calibration receipt is missing {arm}/{variant}")
        for key in ("query", "readout"):
            _require(isinstance(entry.get(key), dict),
                     f"calibration receipt is missing {arm}/{variant}/{key}")
        return entry

    address_entry = arm_variant(ADDRESS_ARM, candidate)
    value_entry = arm_variant(PRIMARY_QUALITY_ARM, candidate)
    oracle_entry = arm_variant(PRIMARY_QUALITY_ARM, oracle)
    address_point = receipt["address_frontier"]["points"][address_index]
    address_diag = address_point["candidate_address_diagnostics"]

    address = _aggregation_diagnostics(
        address_entry["query"], address_entry["readout"],
        layers=layers, label="address",
        accepted_any_hit=address_diag["address_accepted_any_layer_hit"])
    value = _aggregation_diagnostics(
        value_entry["query"], value_entry["readout"],
        layers=layers, label="value")
    oracle_diag = _aggregation_diagnostics(
        oracle_entry["query"], oracle_entry["readout"],
        layers=layers, label="oracle")
    address_by_layer = _per_layer_diagnostics(
        address_entry, layers=layers, label="address")
    value_by_layer = _per_layer_diagnostics(
        value_entry, layers=layers, label="value")
    oracle_by_layer = _per_layer_diagnostics(
        oracle_entry, layers=layers, label="oracle")
    for label, layer_view, aggregate in (
            ("address", address_by_layer, address),
            ("value", value_by_layer, value),
            ("oracle", oracle_by_layer, oracle_diag)):
        _require(
            layer_view["fact_accepted_layers"]
            == aggregate["layer_correctness"]["fact_accepted_layers"],
            f"{label}: per-layer accepted layers do not sum to the aggregate")
        _require(
            layer_view["fact_accepted_correct_layers"]
            == aggregate["layer_correctness"]["fact_accepted_correct_layers"],
            f"{label}: per-layer correct layers do not sum to the aggregate")
        _require(
            layer_view["fact_wrong_layers"]
            == aggregate["layer_correctness"]["fact_wrong_layers"],
            f"{label}: per-layer wrong layers do not sum to the aggregate")

    _require(
        address["query_aggregation"]["fact_accepted_groups"]
        == address_diag["address_accepted"],
        "address query acceptance disagrees with address diagnostics")
    _require(
        address["query_aggregation"]["fact_strict_all_layers_groups"]
        == address_diag["address_accepted_hit"],
        "address strict groups disagree with address diagnostics")
    _require(
        address["query_aggregation"][
            "fact_accepted_not_all_layers_correct_groups"]
        == address_diag["address_query_accepted_not_all_layers_correct"],
        "address partial groups disagree with address diagnostics")
    _require(
        address["query_aggregation"]["fact_any_layer_correct_groups"]
        == address_diag["address_hit"],
        "address any-layer hit disagrees with address diagnostics")
    _require(
        address["layer_correctness"]["fact_accepted_layers"]
        == address_diag["address_layer_accepted"],
        "address accepted layers disagree with address diagnostics")
    _require(
        address["layer_correctness"]["fact_accepted_correct_layers"]
        == address_diag["address_layer_accepted_hit"],
        "address correct layers disagree with address diagnostics")
    _require(
        address["layer_correctness"]["fact_layer_precision"]
        == address_diag["address_layer_precision"],
        "address layer precision disagrees with address diagnostics")
    _require(
        address["query_aggregation"]["fact_query_precision"]
        == address_diag["address_precision"],
        "address query precision disagrees with address diagnostics")
    _require(
        sum(int(value_) for value_ in
            address_diag["wrong_address_categories"].values())
        == address["layer_correctness"]["fact_wrong_layers"],
        "address wrong-layer categories disagree with accepted layers")

    address_accepted_layers = address["layer_correctness"][
        "fact_accepted_layers"]
    address_correct_layers = address["layer_correctness"][
        "fact_accepted_correct_layers"]
    address_wrong_layers = address["layer_correctness"]["fact_wrong_layers"]
    address_strict = address["query_aggregation"][
        "fact_strict_all_layers_groups"]
    address_partial = address["query_aggregation"]["fact_partial_groups"]
    address_no_accepted = address["query_aggregation"][
        "fact_no_accepted_layer_groups"]
    address_fact_n = address["denominators"]["fact_query_groups"]
    address.update({
        "arm": ADDRESS_ARM,
        "variant": candidate,
        "thresholds": dict(thresholds["address"]),
        "by_layer": address_by_layer,
        "layer_precision_headline": (
            f"{address_correct_layers}/{address_accepted_layers}"),
        "strict_group_headline": f"{address_strict}/{address_fact_n}",
        "headline": (
            f"{address_correct_layers}/{address_accepted_layers} accepted "
            "address layer readouts are correct "
            f"({address_wrong_layers} wrong-address layers) but only "
            f"{address_strict}/{address_fact_n} fact query groups are "
            f"strict; {address_partial} groups are partial and "
            f"{address_no_accepted} groups accept no layer."),
        "comparison": {
            "accepted_layer_result": (
                f"{address_correct_layers}/{address_accepted_layers}"),
            "strict_group_result": f"{address_strict}/{address_fact_n}",
            "strict_groups": address_strict,
            "partial_groups": address_partial,
            "no_accepted_layer_groups": address_no_accepted,
            "wrong_address_layers": address_wrong_layers,
        },
        "status": "not_repaired",
        "explanation": (
            "The layer view and the query view answer different questions.  "
            "accepted=any_layer_accepted is preserved, so an accepted fact "
            "group needs only one accepted layer; accepted_correct="
            "all_layers_accepted_and_correct is preserved, so a strict group "
            "needs every layer accepted and correct.  All "
            f"{address_accepted_layers} accepted address layers are correct "
            f"and {address_wrong_layers} accepted layers are wrong, so the "
            f"{address_partial} partial groups are not wrong-address "
            "retrievals at layer level; with two layers they are groups with "
            "exactly one accepted correct layer and one abstaining layer.  "
            "The historical query accepted_wrong value is preserved and is "
            "an aggregation label, not a wrong-address label.  Address "
            f"acceptance is still unrepaired: {address_no_accepted}/"
            f"{address_fact_n} fact groups never accept a layer and strict "
            f"coverage is {address_strict}/{address_fact_n}."),
    })

    value_accepted_layers = value["layer_correctness"][
        "fact_accepted_layers"]
    value_correct_layers = value["layer_correctness"][
        "fact_accepted_correct_layers"]
    value_wrong_layers = value["layer_correctness"]["fact_wrong_layers"]
    value_strict = value["query_aggregation"]["fact_strict_all_layers_groups"]
    value_partial = value["query_aggregation"]["fact_partial_groups"]
    value_wrong_groups = value["query_aggregation"]["fact_wrong_layer_groups"]
    value_no_accepted = value["query_aggregation"][
        "fact_no_accepted_layer_groups"]
    value_fact_n = value["denominators"]["fact_query_groups"]
    value.update({
        "arm": PRIMARY_QUALITY_ARM,
        "variant": candidate,
        "thresholds": dict(thresholds["value"]),
        "by_layer": value_by_layer,
        "headline": (
            f"Candidate value readout accepts {value_accepted_layers} fact "
            f"layers with only {value_correct_layers} correct "
            f"({value_wrong_layers} wrong) and reaches {value_strict}/"
            f"{value_fact_n} strict fact groups; unknown false recall is "
            f"{value['abstention']['unknown_accepted']}/"
            f"{value['abstention']['unknown_n']} and revoked false recall "
            f"is {value['abstention']['revoked_accepted']}/"
            f"{value['abstention']['revoked_n']}."),
        "status": "next_bottleneck",
        "explanation": (
            "Unlike address, the value arm accepts incorrect fact layers: "
            f"{value_wrong_layers} accepted fact value layers are wrong.  "
            "The remaining strict group gap also includes abstention: "
            f"{value_partial} accepted groups have one correct accepted "
            "layer and one abstaining layer, "
            f"{value_wrong_groups} accepted groups contain at least one "
            f"accepted wrong layer, and {value_no_accepted} fact groups "
            "accept no layer.  Unknown and revoked rows are accepted at "
            f"{value['abstention']['unknown_accepted']}/"
            f"{value['abstention']['unknown_n']} and "
            f"{value['abstention']['revoked_accepted']}/"
            f"{value['abstention']['revoked_n']} respectively, so abstention "
            "is not a working safety margin."),
    })

    oracle_accepted_layers = oracle_diag["layer_correctness"][
        "fact_accepted_layers"]
    oracle_correct_layers = oracle_diag["layer_correctness"][
        "fact_accepted_correct_layers"]
    oracle_wrong_layers = oracle_diag["layer_correctness"][
        "fact_wrong_layers"]
    oracle_strict = oracle_diag["query_aggregation"][
        "fact_strict_all_layers_groups"]
    oracle_partial = oracle_diag["query_aggregation"]["fact_partial_groups"]
    oracle_no_accepted = oracle_diag["query_aggregation"][
        "fact_no_accepted_layer_groups"]
    oracle_fact_n = oracle_diag["denominators"]["fact_query_groups"]
    oracle_diag.update({
        "arm": PRIMARY_QUALITY_ARM,
        "variant": oracle,
        "thresholds": dict(thresholds["value"]),
        "by_layer": oracle_by_layer,
        "is_legal_policy": False,
        "is_diagnostic_only": True,
        "headline": (
            f"Oracle fact value layers are {oracle_correct_layers}/"
            f"{oracle_accepted_layers} correct with {oracle_wrong_layers} "
            f"wrong fact value layers, yet only {oracle_strict}/"
            f"{oracle_fact_n} fact groups are strict and {oracle_partial} "
            f"are partial; {oracle_no_accepted} groups accept no layer."),
        "explanation": (
            "The oracle is a gold-target diagnostic, not a legal policy.  Its "
            "accepted-layer fact precision with "
            f"{oracle_partial} partial groups isolates layer aggregation and "
            "abstention as a remaining limit even when value readout is "
            "correct.  It does not repair the candidate value readout."),
    })

    variant_names = sorted(arms[PRIMARY_QUALITY_ARM]["variants"])

    def variant_with(predicate, label):
        matches = [name for name in variant_names if predicate(name)]
        _require(len(matches) == 1,
                 f"calibration receipt does not isolate {label}")
        return matches[0]

    disabled_variant = variant_with(
        lambda name: name.endswith("/disabled"), "disabled/reset")
    shuffled_variant = variant_with(
        lambda name: "shuffled" in name, "shuffled-value")
    full_variant = variant_with(
        lambda name: name.endswith("/full"), "full")

    def value_variant(name):
        entry = arm_variant(PRIMARY_QUALITY_ARM, name)
        return _aggregation_diagnostics(
            entry["query"], entry["readout"], layers=layers,
            label=f"value:{name}")

    disabled = value_variant(disabled_variant)
    shuffled = value_variant(shuffled_variant)
    full = value_variant(full_variant)
    exact_entry = arm_variant("exact_key_value", candidate)
    exact_context = _aggregation_diagnostics(
        exact_entry["query"], exact_entry["readout"], layers=layers,
        label="exact_context")
    context_entry = arm_variant("context_removed", candidate)
    context_removed = _aggregation_diagnostics(
        context_entry["query"], context_entry["readout"], layers=layers,
        label="context_removed")

    selected_by_variant = {
        name: _selected_record(
            arm_variant(PRIMARY_QUALITY_ARM, name),
            label=f"selected-record:{name}")
        for name in variant_names}

    def separation_metric(selector):
        return {label: selector(item)
                for label, item in (
                    ("candidate", value), ("oracle", oracle_diag),
                    ("full", full), ("shuffled_value", shuffled),
                    ("disabled_reset", disabled),
                    ("exact_context", exact_context))}

    strict_rates = separation_metric(
        lambda item: item["query_aggregation"]["fact_strict_group_rate"])
    precisions = separation_metric(
        lambda item: item["query_aggregation"]["fact_query_precision"])
    layer_precisions = separation_metric(
        lambda item: item["layer_correctness"]["fact_layer_precision"])
    accepted_groups = separation_metric(
        lambda item: item["query_aggregation"]["fact_accepted_groups"])
    unknown_false_recall = separation_metric(
        lambda item: item["abstention"]["unknown_false_recall_rate"])

    address_variants = {}
    for name in variant_names:
        entry = arm_variant(ADDRESS_ARM, name)
        address_variants[name] = {
            "fact_accepted_groups": int(entry["query"]["accepted"]) - int(
                entry["query"].get("unknown_false_recall") or 0) - int(
                entry["query"].get("revoked_false_recall") or 0),
            "fact_strict_groups": int(
                entry["query"].get("accepted_correct_fact") or 0),
            "fact_accepted_layers": int(entry["readout"]["accepted"]) - int(
                entry["readout"].get("unknown_false_recall") or 0) - int(
                entry["readout"].get("revoked_false_recall") or 0),
            "fact_correct_layers": int(
                entry["readout"].get("accepted_correct_fact") or 0),
        }
    address_invariant = len({
        tuple(item.items()) for item in address_variants.values()}) == 1

    controls = {
        "candidate": {
            "variant": candidate,
            "arm": PRIMARY_QUALITY_ARM,
            "metrics": value,
        },
        "oracle": {
            "variant": oracle,
            "arm": PRIMARY_QUALITY_ARM,
            "metrics": oracle_diag,
            "is_legal_policy": False,
            "is_diagnostic_only": True,
        },
        "exact_context": {
            "control": "exact_key_value_arm",
            "variant": candidate,
            "arm": "exact_key_value",
            "available": True,
            "metrics": exact_context,
            "abstention_available": exact_context["abstention"]["available"],
            "abstention_unavailable_reason": (
                None if exact_context["abstention"]["available"] else
                "exact-key rows exist only for fact targets; this arm has "
                "no unknown or revoked rows"),
            "is_upper_bound": False,
            "blended_into_candidate_quality": False,
            "delta_vs_candidate": _control_delta(value, exact_context),
        },
        "context_removed": {
            "control": "context_removed_arm",
            "variant": candidate,
            "arm": "context_removed",
            "available": True,
            "metrics": context_removed,
            "delta_vs_candidate": _control_delta(value, context_removed),
            "interpretation": (
                "With context removed every readout is zero or tied; the "
                "control accepts no fact group and receives no recoverability "
                "or acceptance credit."),
        },
        "selected_record": {
            "control": "artifact_runtime_selected_record",
            "available": any(
                item["runtime_selection_available"]
                for item in selected_by_variant.values()),
            "candidate": selected_by_variant[candidate],
            "oracle": selected_by_variant[oracle],
            "address_candidate": _selected_record(
                address_entry, label="selected-record:address"),
            "by_variant": selected_by_variant,
            "note": (
                "Runtime selected-record fields come from the artifact and "
                "are not recomputed by threshold replay.  They separate the "
                "selection decision from the value readout correctness."),
        },
        "disabled_reset": {
            "control": "shared_mean/disabled",
            "variant": disabled_variant,
            "available": True,
            "metrics": disabled,
            "delta_vs_candidate": _control_delta(value, disabled),
            "interpretation": (
                "The disabled/reset control has zero acceptance and a "
                "zero-or-tied rate of 1.0; observed top-1 evidence receives "
                "no recoverability or acceptance credit."),
        },
        "shuffled_value": {
            "control": "shuffled_value_variant",
            "variant": shuffled_variant,
            "available": True,
            "metrics": shuffled,
            "delta_vs_candidate": _control_delta(value, shuffled),
            "interpretation": (
                "Shuffled values do not restore fact value correctness: the "
                "control remains at "
                f"{shuffled['query_aggregation']['fact_strict_all_layers_groups']}"
                f"/{shuffled['denominators']['fact_query_groups']} strict "
                "fact groups and its query precision is lower than the "
                "candidate.  It is a control, not a recovered policy."),
        },
        "full": {
            "control": "full_state_variant",
            "variant": full_variant,
            "available": True,
            "metrics": full,
            "delta_vs_candidate": _control_delta(value, full),
        },
        "wrong_world": {
            "control": "cross_world_selected",
            "candidate": _wrong_world_diagnostics(
                value_entry, label="wrong-world:candidate"),
            "oracle": _wrong_world_diagnostics(
                oracle_entry, label="wrong-world:oracle"),
            "exact_context": _wrong_world_diagnostics(
                exact_entry, label="wrong-world:exact"),
            "address_candidate": _wrong_world_diagnostics(
                address_entry, label="wrong-world:address"),
            "available": True,
            "note": (
                "cross_world_selected and the category fields are "
                "scoring-only and never threshold inputs.  For the address "
                "arm the query view counts cross-world evidence in a "
                "non-accepted layer of a partial group; the accepted-layer "
                "view has zero wrong address layers."),
        },
        "separation": {
            "fact_strict_group_rate": strict_rates,
            "fact_query_precision": precisions,
            "fact_layer_precision": layer_precisions,
            "fact_accepted_groups": accepted_groups,
            "unknown_false_recall_rate": unknown_false_recall,
            "controls_are_distinct": len({
                (strict_rates[label], precisions[label],
                 accepted_groups[label])
                for label in strict_rates}) > 1,
            "candidate_equals_full": (
                value["query_aggregation"]
                == full["query_aggregation"]),
            "candidate_equals_shuffled": (
                value["query_aggregation"]
                == shuffled["query_aggregation"]),
            "candidate_equals_disabled": (
                value["query_aggregation"]
                == disabled["query_aggregation"]),
            "address_arm_invariant_across_value_variants": (
                address_invariant),
            "address_arm_by_variant": address_variants,
        },
    }

    def role_diagnostics(name):
        block = value_entry.get(name)
        _require(isinstance(block, dict),
                 f"calibration receipt is missing the value {name} block")
        return _aggregation_diagnostics(
            block, None, layers=layers, label=f"value:{name}",
            layer_view_available=False)

    by_role = {
        "correction_target": {
            "available": "correction" in value_entry,
            "metrics": role_diagnostics("correction"),
            "oracle_metrics": None,
        },
        "retained_unaffected": {
            "available": "unaffected" in value_entry,
            "metrics": role_diagnostics("unaffected"),
            "oracle_metrics": None,
        },
        "note": (
            "Role subsets are query-level only in the calibration receipt.  "
            "Unaffected/retained rows are scored after an event; they are not "
            "an additional teaching gradient in the current path."),
    }
    if "correction" in oracle_entry:
        by_role["correction_target"]["oracle_metrics"] = (
            _aggregation_diagnostics(
                oracle_entry["correction"], None, layers=layers,
                label="oracle:correction", layer_view_available=False))
    if "unaffected" in oracle_entry:
        by_role["retained_unaffected"]["oracle_metrics"] = (
            _aggregation_diagnostics(
                oracle_entry["unaffected"], None, layers=layers,
                label="oracle:unaffected", layer_view_available=False))

    frontier = receipt["frontier_summary"]
    disposition = receipt["gate_classification"]["disposition"]
    calibration_evidence = {
        "value_point_count": int(frontier["value_point_count"]),
        "value_points_passing_candidate_value_gates": int(
            frontier["value_points_passing_candidate_value_gates"]),
        "value_points_passing_all_gates": int(
            frontier["value_points_passing_all_gates"]),
        "value_points_passing_oracle_gates": int(
            frontier["value_points_passing_oracle_gates"]),
        "address_point_count": int(frontier["address_point_count"]),
        "address_points_meeting_hit_and_precision": int(
            frontier["address_points_meeting_hit_and_precision"]),
        "best_candidate_accepted_correct_rate": frontier[
            "best_query_key_value_accepted_correct_rate"],
        "best_candidate_precision": frontier[
            "best_query_key_value_accepted_precision"],
        "selected_rule": disposition["selected_rule"],
        "candidate_meets_value_gates": bool(
            disposition["candidate_meets_value_gates"]),
        "oracle_meets_gates": bool(disposition["oracle_meets_gates"]),
        "calibration_is_a_solution": False,
        "note": (
            "No value point in the 25-point frontier passes the candidate or "
            "oracle value gates, and no address point meets both descriptive "
            "hit and precision targets.  Threshold calibration cannot repair "
            "the value readout or abstention."),
    }

    bottleneck = {
        "selected": "value_readout_and_abstention",
        "components": ["value_readout", "abstention"],
        "scope": "train_dev_only",
        "train_dev_scope": {
            "splits": list(receipt["source"]["scope"].get(
                "splits") or receipt["source"]["scope"].get(
                    "declared_splits") or []),
            "phases": list(receipt["source"]["scope"].get(
                "phases") or receipt["source"]["scope"].get(
                    "declared_phases") or []),
            "heldout_rows_scored": 0,
            "heldout_metrics_reported": False,
            "per_split_decomposition_available": False,
            "per_split_decomposition_note": (
                "The source calibration receipt aggregates train and dev "
                "native blocks; it does not expose a train/dev split of the "
                "same arm and variant.  No per-split claim is made and no "
                "heldout row is scored."),
        },
        "value_evidence": {
            "fact_accepted_groups": value["query_aggregation"][
                "fact_accepted_groups"],
            "fact_strict_groups": value["query_aggregation"][
                "fact_strict_all_layers_groups"],
            "fact_partial_groups": value["query_aggregation"][
                "fact_partial_groups"],
            "fact_wrong_layer_groups": value["query_aggregation"][
                "fact_wrong_layer_groups"],
            "fact_accepted_layers": value["layer_correctness"][
                "fact_accepted_layers"],
            "fact_accepted_correct_layers": value["layer_correctness"][
                "fact_accepted_correct_layers"],
            "fact_wrong_layers": value["layer_correctness"][
                "fact_wrong_layers"],
            "fact_layer_precision": value["layer_correctness"][
                "fact_layer_precision"],
            "fact_strict_group_rate": value["query_aggregation"][
                "fact_strict_group_rate"],
        },
        "abstention_evidence": {
            "unknown_n": value["abstention"]["unknown_n"],
            "unknown_accepted": value["abstention"]["unknown_accepted"],
            "unknown_false_recall_rate": value["abstention"][
                "unknown_false_recall_rate"],
            "revoked_n": value["abstention"]["revoked_n"],
            "revoked_accepted": value["abstention"]["revoked_accepted"],
            "revoked_false_recall_rate": value["abstention"][
                "revoked_false_recall_rate"],
        },
        "address_evidence": {
            "fact_accepted_groups": address["query_aggregation"][
                "fact_accepted_groups"],
            "fact_strict_groups": address["query_aggregation"][
                "fact_strict_all_layers_groups"],
            "fact_partial_groups": address["query_aggregation"][
                "fact_partial_groups"],
            "fact_accepted_layers": address["layer_correctness"][
                "fact_accepted_layers"],
            "fact_accepted_correct_layers": address["layer_correctness"][
                "fact_accepted_correct_layers"],
            "fact_wrong_layers": address["layer_correctness"][
                "fact_wrong_layers"],
            "fact_layer_precision": address["layer_correctness"][
                "fact_layer_precision"],
            "fact_strict_group_rate": address["query_aggregation"][
                "fact_strict_group_rate"],
            "status": "not_repaired",
        },
        "oracle_evidence": {
            "fact_accepted_groups": oracle_diag["query_aggregation"][
                "fact_accepted_groups"],
            "fact_strict_groups": oracle_diag["query_aggregation"][
                "fact_strict_all_layers_groups"],
            "fact_partial_groups": oracle_diag["query_aggregation"][
                "fact_partial_groups"],
            "fact_accepted_layers": oracle_diag["layer_correctness"][
                "fact_accepted_layers"],
            "fact_accepted_correct_layers": oracle_diag["layer_correctness"][
                "fact_accepted_correct_layers"],
            "fact_wrong_layers": oracle_diag["layer_correctness"][
                "fact_wrong_layers"],
            "fact_layer_precision": oracle_diag["layer_correctness"][
                "fact_layer_precision"],
            "is_legal_policy": False,
        },
        "retention_evidence": {
            "candidate_unaffected_fact_groups": by_role[
                "retained_unaffected"]["metrics"]["denominators"][
                    "fact_query_groups"],
            "candidate_unaffected_accepted_groups": by_role[
                "retained_unaffected"]["metrics"]["query_aggregation"][
                    "fact_accepted_groups"],
            "candidate_unaffected_strict_groups": by_role[
                "retained_unaffected"]["metrics"]["query_aggregation"][
                    "fact_strict_all_layers_groups"],
            "oracle_unaffected_strict_groups": (
                (by_role["retained_unaffected"]["oracle_metrics"] or {})
                .get("query_aggregation", {})
                .get("fact_strict_all_layers_groups")),
            "unaffected_rows_are_teaching_targets": False,
            "unaffected_rows_current_teaching_gradient": 0,
            "note": (
                "Unaffected/retained rows are scored after an event and "
                "currently receive no teaching gradient.  They are part of "
                "the next hypothesis scope, not a new training arm in this "
                "receipt."),
        },
        "address_readout_repaired": False,
        "calibration_is_a_solution": False,
        "why_this_bottleneck": [
            (
                "Address layer precision is "
                f"{address_correct_layers}/{address_accepted_layers} with "
                f"{address_wrong_layers} wrong accepted address layers, so "
                "accepted address layers are not mis-addressed; the address "
                "gap is low strict coverage and partial layer agreement, not "
                "a wrong-address mechanism."),
            (
                "Value layer precision is only "
                f"{value_correct_layers}/{value_accepted_layers} and "
                f"{value_wrong_layers} accepted fact value layers are wrong, "
                "so the value readout itself is the dominant accepted error."),
            (
                "Unknown false recall is "
                f"{value['abstention']['unknown_accepted']}/"
                f"{value['abstention']['unknown_n']} and revoked false "
                "recall is "
                f"{value['abstention']['revoked_accepted']}/"
                f"{value['abstention']['revoked_n']}, so abstention is also "
                "failing."),
            (
                f"The oracle has {oracle_correct_layers}/"
                f"{oracle_accepted_layers} correct accepted fact value "
                f"layers but still only {oracle_strict}/{oracle_fact_n} "
                f"strict groups with {oracle_partial} partial groups, which "
                "isolates layer abstention/aggregation as a remaining limit "
                "independent of wrong fact values."),
            (
                "No calibration point repairs the value gates; threshold "
                "movement cannot create a correct value readout or a working "
                "abstention rule."),
        ],
        "next_hypothesis": (
            "one_step_per_event_future_answer_correction_unaffected_"
            "retention"),
        "next_hypothesis_status": "declared_not_executed",
        "new_training_arm": False,
        "calibration_evidence": calibration_evidence,
        "not_claimed": [
            "Address acceptance is not repaired.",
            "Calibration is not a solution.",
            "The oracle is not a legal policy or an achievable bound.",
            "No value readout, abstention, writer or retention capability is "
            "demonstrated by this receipt.",
        ],
    }

    primary_record = _source_receipt_record(
        receipt_path, role="primary_train_dev_calibration",
        receipt=receipt) if receipt_path is not None else {
            "role": "primary_train_dev_calibration",
            "path": None,
            "available": True,
            "sha256": None,
            "content_sha256": receipt["content_sha256"],
            "format": receipt["format"],
            "metric_version": receipt["metric_version"],
        }
    primary_record.update({
        "declared_at_utc": receipt["config"].get("declared_at_utc")
        if isinstance(receipt.get("config"), dict) else None,
        "content_sha256_verified": True,
    })

    supporting = []
    declared_supporting = (
        SUPPORTING_V4_RECEIPTS if supporting_receipts is None
        else supporting_receipts)
    for role, relative in declared_supporting:
        path = Path(relative)
        if not path.is_absolute():
            path = Path(root) / path
        supporting.append(_provenance_receipt_record(path, role=role))

    body = {
        "format": DIAGNOSIS_FORMAT,
        "metric_version": DIAGNOSIS_METRIC_VERSION,
        "purpose": (
            "Derived train/dev memory acceptance diagnosis: explicit address "
            "and value correctness, abstention, multi-layer aggregation, "
            "control separation and declared gradient scope.  No native "
            "execution, model load, gradient, optimizer update, checkpoint "
            "write, relabeling or held-out scoring."),
        "source_receipts": {
            "primary": primary_record,
            "supporting": supporting,
            "supporting_metrics_used": False,
            "supporting_note": (
                "Historical v4 receipts are hashed for provenance only.  "
                "Their metric blocks may include heldout rows and are never "
                "parsed or used by this diagnosis."),
        },
        "source_pins": {
            "rankings": dict(receipt["source"].get("rankings") or {}),
            "inventory": dict(receipt["source"].get("inventory") or {}),
            "config": dict(receipt.get("config") or {}),
        },
        "scope": {
            **bottleneck["train_dev_scope"],
            "scored_rows": receipt["source"]["scope"].get("scored_rows"),
            "heldout_rows_present": receipt["source"]["scope"].get(
                "heldout_rows_present"),
            "fact_query_groups": value["denominators"]["fact_query_groups"],
            "fact_layer_readouts": value["denominators"][
                "fact_layer_readouts"],
            "layers": layers,
        },
        "metric_semantics": {
            "accepted": "any_layer_accepted",
            "accepted_correct": "all_layers_accepted_and_correct",
            "accepted_wrong": "accepted_and_not_all_layers_correct",
            "query_view": (
                "one row per fact query group; acceptance is any-layer and "
                "correctness is all-layer"),
            "layer_view": (
                "one row per readout layer; layers are not independent "
                "samples and are never treated as extra queries"),
            "strict_group": "all_layers_accepted_and_correct",
            "partial_group": (
                "accepted but not all layers accepted and correct, with no "
                "accepted wrong layer; for the two-layer receipts this means "
                "exactly one accepted correct layer and one abstention"),
            "strict_denominator": "fact query groups",
            "layer_denominator": "fact readout layers",
            "historical_metrics_relabelled": False,
            "gate_semantics_preserved": True,
            "unknown_and_revoked_separate_from_facts": True,
        },
        "address": address,
        "value": value,
        "oracle": oracle_diag,
        "abstention": {
            "candidate_value": value["abstention"],
            "address": address["abstention"],
            "oracle": oracle_diag["abstention"],
        },
        "multi_layer_aggregation": {
            "layers": layers,
            "layers_per_group": len(layers),
            "layers_are_independent_samples": False,
            "address": address["decomposition"],
            "value": value["decomposition"],
            "oracle": oracle_diag["decomposition"],
            "note": (
                "The strict query view is not the sum of layer views.  "
                f"Address {address['layer_precision_headline']} is a layer "
                "precision over accepted layers; "
                f"{address['strict_group_headline']} is a strict group count "
                "over fact query groups."),
        },
        "by_role": by_role,
        "controls": controls,
        "bottleneck": bottleneck,
        "gradient_scope": GRADIENT_SCOPE,
        "claims": {
            "address_layer_readout_correct_when_accepted": True,
            "address_layer_precision_is_not_strict_group_coverage": True,
            "address_readout_repaired": False,
            "address_acceptance_is_not_the_selected_next_bottleneck": True,
            "value_readout_and_abstention_are_next_bottleneck": True,
            "calibration_is_a_solution": False,
            "historical_metrics_relabelled": False,
            "gate_semantics_preserved": True,
            "train_dev_rows_only": True,
            "heldout_rows_scored": False,
            "heldout_metrics_reported": False,
            "heldout_metric_blocks_parsed": False,
            "supporting_receipt_metrics_used": False,
            "native_execution_performed": False,
            "model_loads": 0,
            "gradients": 0,
            "optimizer_updates": 0,
            "checkpoint_writes": 0,
            "new_training_arm": False,
            "one_step_per_event_contract_unchanged": True,
            "gradient_scope_declared_not_executed": True,
            "receipt_is_derived_not_authority": True,
        },
        "non_claims": [
            "No learned-memory, learned-selector, generated-answer or "
            "calibrated-confidence claim.",
            "Address acceptance is not repaired; perfect accepted-layer "
            "precision does not establish strict group coverage.",
            "Calibration is not a solution; zero frontiers pass the declared "
            "value gates.",
            "The oracle is a gold-target diagnostic, not a legal policy or "
            "an achievable upper bound.",
            "No new training arm and no change to the one-step-per-event "
            "contract.",
            "Prepared04 heldout remains spent and was not reused.",
        ],
        "limits": {
            "derived_from_receipt_fields_only": True,
            "raw_rankings_read": False,
            "supporting_receipt_metrics_parsed": False,
            "native_execution_performed": False,
            "model_loads": 0,
            "tokenizer_loads": 0,
            "gradients": 0,
            "optimizer_updates": 0,
            "checkpoint_writes": 0,
            "new_generation": False,
            "heldout_rows_scored": 0,
        },
        "determinism": {
            "wall_clock_recorded": False,
            "ordering": "sorted keys; fixed supporting-receipt order",
            "content_sha256_covers_receipt_without_this_field": True,
        },
    }
    return dict(body, content_sha256=_sha_json(body))


def write_diagnosis_receipt(receipt, path):
    """Write a derived diagnosis atomically, refusing source overwrites."""
    _require(receipt.get("format") == DIAGNOSIS_FORMAT,
             "Refusing to write a non-diagnosis receipt")
    _require(receipt.get("metric_version") == DIAGNOSIS_METRIC_VERSION,
             "Refusing to write an unexpected diagnosis metric version")
    path = Path(path)
    sources = []
    primary = receipt["source_receipts"]["primary"].get("path")
    if primary:
        sources.append(primary)
    for item in receipt["source_receipts"]["supporting"]:
        if item.get("path"):
            sources.append(item["path"])
    for source in sources:
        if Path(source).resolve() == path.resolve():
            raise ValueError("Refusing to overwrite a source receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)
    return path


def analyze_calibration(config, *, root=ROOT, config_path=None):
    """Read the pinned sources and return a deterministic derived receipt."""
    validate_calibration_config(config)
    source = config["source"]
    scope = config["scope"]
    limits = config["limits"]
    root = Path(root)

    rankings_path = _resolve(source["rankings"], root)
    inventory_path = _resolve(source["inventory"], root)
    _require(rankings_path.is_file(),
             f"Rankings source is missing: {rankings_path}")
    _require(inventory_path.is_file(),
             f"Inventory source is missing: {inventory_path}")
    _require("heldout" not in rankings_path.name.lower(),
             "Rankings source name must not identify a heldout artifact")
    _require("prepared04" not in str(inventory_path),
             "Calibration must not reuse the prepared04 heldout inventory")

    rankings_bytes, rankings = _load_rankings(rankings_path)
    inventory_bytes, inventory = _load_inventory(inventory_path)
    rankings_sha = hashlib.sha256(rankings_bytes).hexdigest()
    inventory_sha = hashlib.sha256(inventory_bytes).hexdigest()
    _require(rankings_sha == source["rankings_sha256"].lower(),
             "Rankings bytes differ from the declared sha256 pin")
    _require(inventory_sha == source["inventory_sha256"].lower(),
             "Inventory bytes differ from the declared sha256 pin")
    _require(len(rankings) == source["expected_rows"],
             f"Expected {source['expected_rows']} rows, found {len(rankings)}")
    _require(len(rankings) <= limits["max_source_rows"],
             "Source row count exceeds the declared limit")

    v4_config = _v4_config(config, inventory)
    validate_v4_config(v4_config, require_inventory=True)
    validation = validate_ranking_inventory(
        rankings, v4_config, inventory, phases=None)
    _require(validation["scope"]["full"] is True,
             "Inventory validation did not cover every declared phase")

    selected_splits = set(scope["splits"])
    scored = [row for row in rankings if row["split"] in selected_splits]
    out_of_scope = [row for row in rankings
                    if row["split"] not in selected_splits]
    _require(scored, "Scored split scope contains no ranking rows")
    scope_audit = _scope_audit(config, inventory, rankings, scored,
                               out_of_scope)
    _require(scope_audit["heldout_rows_scored"] == 0,
             "Heldout rows must not be scored")

    bases = _base_rows(scored)
    grouped = _groups(bases, lambda row: (
        row["arm"], row["variant"], row["phase"], row["episode_id"],
        row["state"], row["case_id"]))
    group_index = {}
    for key, rows in grouped.items():
        group_index.setdefault((key[0], key[1]), {})[key[2:]] = sorted(
            rows, key=lambda row: row["layer"])
    # Query groups also drive the per-layer view; _layer_rows materializes
    # readout rows for each threshold without treating layers as independent.

    arms = _native_metrics(config, group_index)
    native_arm_summary = _native_summary(arms, config)
    arm_separation = _arm_separation(arms, config)
    controls = _control_comparison(arms, config)

    candidate = config["controls"]["candidate_variant"]
    oracle = config["controls"]["oracle_variant"]
    oracle_quality = arms[PRIMARY_QUALITY_ARM]["variants"][oracle]["query"]
    value_frontier = _value_frontier(
        config, group_index, oracle_quality)
    address_frontier = _address_frontier(config, group_index)
    frontier_summary = _frontier_summary(value_frontier, address_frontier, config)
    gates = _gate_classification(config, arms, value_frontier)

    runtime_comparison = _artifact_acceptance_audit(bases, config)
    native_arm_summary["query_key_value"]["artifact_acceptance_audit"] = (
        runtime_comparison)

    # Independent view of zero/tie, policy blocking and categories for the
    # candidate variant and every arm.
    zero_tie = {}
    policy = {}
    categories = {}
    runtime_selection = {}
    unknown_revoked = {}
    for arm in scope["arms"]:
        entry = arms[arm]["variants"][candidate]
        query = entry["query"]
        zero_tie[arm] = {
            "query_zero_or_tied": query["zero_or_tied"],
            "query_zero_or_tied_rate": query["zero_or_tied_rate"],
            "query_observed_top1_excluded": query[
                "zero_or_tied_observed_top1"],
            "query_recoverability_credit": query[
                "zero_or_tied_recoverability_credit"],
            "query_acceptance_credit": query[
                "zero_or_tied_acceptance_credit"],
            "readout_zero_or_tied": entry["readout"]["zero_or_tied"],
        }
        floors = (config["native_thresholds"]["address"]
                  if arm == ADDRESS_ARM
                  else config["native_thresholds"]["value"])
        query_rows = _replay_query_rows(group_index[(arm, candidate)], floors)
        layer_rows = _layer_rows(group_index[(arm, candidate)], floors)
        policy[arm] = {
            "query_policy_blocked": query["policy_blocked"],
            "query_policy_blocked_acceptance": query[
                "policy_blocked_acceptance"],
            "query_policy_blocked_would_accept": sum(
                row["policy_blocked_would_accept"] for row in query_rows),
            "blocked_rows_never_accepted": True,
        }
        categories[arm] = {
            "query_view": _categories(query_rows),
            "layer_view": _layer_categories(layer_rows),
        }
        runtime_selection[arm] = {
            "runtime_selection_available": query[
                "runtime_selection_available"],
            "runtime_rows": query["runtime_rows"],
            "runtime_accepted": query["runtime_accepted"],
            "runtime_accepted_rate": query["runtime_accepted_rate"],
            "runtime_correct": query["runtime_correct"],
            "selected_other_live": query["selected_other_live"],
            "source": "artifact_runtime_fields_not_threshold_replay",
        }
        unknown_revoked[arm] = {
            "unknown_n": entry["unknown"]["unknown_n"],
            "unknown_accepted": entry["unknown"]["accepted"],
            "unknown_abstention_rate": entry["unknown"][
                "unknown_abstention_rate"],
            "unknown_false_recall_rate": entry["unknown"][
                "unknown_false_recall_rate"],
            "revoked_n": entry["revoked"]["revoked_n"],
            "revoked_accepted": entry["revoked"]["accepted"],
            "revoked_abstention_rate": entry["revoked"][
                "revoked_abstention_rate"],
            "revoked_false_recall_rate": entry["revoked"][
                "revoked_false_recall_rate"],
            "revoked_target_other_live": entry["revoked"][
                "revoked_target_other_live"],
        }

    config_record = {
        "format": config["format"],
        "metric_version": config["metric_version"],
        "declared_at_utc": config["declared_at_utc"],
        "sha256": _sha_json(config),
        "path": str(config_path) if config_path is not None else None,
        "file_sha256": (_sha(config_path)
                        if config_path is not None else None),
    }
    _require(_sha(rankings_path) == rankings_sha,
             "Raw rankings changed during analysis")
    _require(_sha(inventory_path) == inventory_sha,
             "Pinned inventory changed during analysis")
    body = {
        "format": RECEIPT_FORMAT,
        "metric_version": METRIC_VERSION,
        "config": config_record,
        "source": {
            "rankings": {
                "path": str(rankings_path),
                "sha256": rankings_sha,
                "sha256_matches": True,
                "expected_rows": source["expected_rows"],
                "rows": len(rankings),
                "rows_match": True,
            },
            "inventory": {
                "path": str(inventory_path),
                "sha256": inventory_sha,
                "sha256_matches": True,
                "rows": len(inventory["rows"]),
                "units": inventory.get("units") or {},
                "units_reconciled": True,
            },
            "scope": scope_audit,
            "raw_rankings_modified": False,
            "raw_labels_modified": False,
            "original_bytes_unchanged": (
                _sha(rankings_path) == rankings_sha
                and _sha(inventory_path) == inventory_sha),
            "inventory_bytes_unchanged": _sha(inventory_path) == inventory_sha,
        },
        "audit": {
            "config_validated": True,
            "source_hash_verified": True,
            "inventory_hash_verified": True,
            "expected_rows_match": True,
            "inventory_units_reconciled": True,
            "ranking_inventory_validated": True,
            "scope_train_dev_only": True,
            "heldout_rows_present": scope_audit["heldout_rows_present"],
            "heldout_rows_scored": 0,
            "heldout_metrics_reported": False,
            "prepared04_heldout_reused": False,
            "split_scope": list(scope["splits"]),
            "phase_scope": list(scope["phases"]),
            "scored_rows_by_split": scope_audit["scored_rows_by_split"],
            "scored_rows_by_phase": scope_audit["scored_rows_by_phase"],
            "scored_rows_by_arm": {
                arm: sum(row["arm"] == arm for row in scored)
                for arm in ALL_ARMS},
            "scored_rows_by_variant": {
                variant: sum(row["variant"] == variant for row in scored)
                for variant in scope["variants"]},
            "scored_rows_by_layer": {
                str(layer): sum(row["layer"] == layer for row in scored)
                for layer in scope["layers"]},
            "row_field_audit": {
                "top_cosine_numeric": sum(
                    _is_number(row.get("top_cosine")) for row in scored),
                "second_cosine_null": sum(
                    row.get("second_cosine") is None for row in scored),
                "zero_norm": sum(row["readout_norm"] == 0 for row in scored),
                "tied": sum(
                    row.get("second_cosine") is not None
                    and row["top_cosine"] == row["second_cosine"]
                    for row in scored),
                "policy_blocked": sum(
                    row.get("policy_blocked") is not None
                    or row.get("stale_blocked")
                    or row.get("revoked_blocked") for row in scored),
                "runtime_fields_present": sum(
                    "runtime_accepted" in row for row in scored),
                "gold_labels_preserved_for_scoring": True,
            },
            "native_rule_reproduces_artifact_acceptance": runtime_comparison[
                "reproduced"],
            "native_rule_artifact_mismatches": runtime_comparison["mismatches"],
        },
        "runtime_decision": {
            "applies_to": "existing readout rows only",
            "value_floors": dict(config["native_thresholds"]["value"]),
            "address_floors": dict(config["native_thresholds"]["address"]),
            "decision_inputs": [
                "readout_norm", "top_cosine", "second_cosine",
                "policy_blocked/stale_blocked/revoked_blocked", "arm"],
            "zero_or_tied_rule": (
                "readout_norm > 0 and top_cosine != second_cosine; otherwise "
                "the row receives no recoverability or acceptance credit"),
            "acceptance_rule": (
                "usable and not policy-blocked and top_cosine >= cosine_floor "
                "and top_cosine - second_cosine >= margin_floor"),
            "gold_inputs_used_for_threshold_decision": [],
            "scoring_only_inputs": [
                "top1", "target_entry_id", "false_recall_kind",
                "cross_world_selected", "target_role", "kind"],
            "artifact_accepted_used_for_decision": False,
            "artifact_acceptance_comparison": runtime_comparison,
        },
        "native": {
            "thresholds": {
                "value": dict(config["native_thresholds"]["value"]),
                "address": dict(config["native_thresholds"]["address"]),
            },
            "arms": arms,
            "arm_summary_candidate_variant": native_arm_summary,
            "arm_separation": arm_separation,
            "control_comparison": controls,
            "oracle_diagnostic": {
                "variant": oracle,
                "arm": PRIMARY_QUALITY_ARM,
                "fact_n": oracle_quality["fact_n"],
                "fact_recovered": oracle_quality["fact_recovered"],
                "answerable_fact_recoverability_rate": oracle_quality[
                    "recoverability_rate"],
                "total_group_raw_top1_rate": oracle_quality["raw_top1_rate"],
                "is_legal_policy": False,
                "excluded_kinds": ["unknown", "revoked"],
            },
            "exact_key_value_diagnostic": controls[
                "exact_key_value_diagnostic"],
            "wrong_address_categories": {
                "declared": list(WRONG_ADDRESS_CATEGORIES),
                "by_arm_candidate_variant": categories,
                "note": ("Categories are assigned after runtime acceptance "
                         "from kind/cross-world scoring fields and are never "
                         "threshold inputs."),
            },
            "unknown_revoked": {
                "separated": True,
                "by_arm_candidate_variant": unknown_revoked,
            },
            "zero_and_tie": {
                "credit": "none",
                "by_arm_candidate_variant": zero_tie,
            },
            "policy_blocking": {
                "blocked_rows_never_accepted": True,
                "by_arm_candidate_variant": policy,
            },
            "runtime_selection": {
                "by_arm_candidate_variant": runtime_selection,
                "note": ("Runtime selection metrics are artifact runtime "
                         "fields where available; they are not recomputed by "
                         "threshold replay."),
            },
            "views": {
                "query_view": {
                    "combination": dict(LAYER_SEMANTICS),
                    "layers_are_independent_samples": False,
                    "description": ("Query metrics combine both layers: any "
                                    "accepted layer, all layers accepted and "
                                    "correct, all layers zero or tied."),
                },
                "layer_view": {
                    "description": ("Per-layer metrics are readout rows.  "
                                   "The same query appears in every layer and "
                                   "layers are not independent samples."),
                    "layers": list(scope["layers"]),
                },
            },
        },
        "value_frontier": value_frontier,
        "address_frontier": address_frontier,
        "frontier_summary": frontier_summary,
        "gate_classification": gates,
        "claims": {
            "runtime_decisions_use_no_gold_inputs": True,
            "gold_labels_used_only_after_runtime_decision": True,
            "zero_and_tied_receive_no_credit": True,
            "policy_blocking_is_preserved": True,
            "exact_key_value_is_not_blended_into_candidate_quality": True,
            "exact_key_value_is_not_an_upper_bound": True,
            "oracle_is_a_diagnostic_not_a_legal_policy": True,
            "address_frontier_is_address_arm_only": True,
            "address_frontier_does_not_reselect_candidates": True,
            "address_frontier_does_not_recompute_downstream_refresh": True,
            "value_frontier_recomputes_existing_readout_acceptance_only": True,
            "heldout_rows_not_scored": True,
            "heldout_metrics_not_reported": True,
            "prepared04_heldout_not_reused": True,
            "raw_rankings_not_modified": True,
            "raw_labels_not_modified": True,
            "receipt_is_derived_not_authority": True,
            "native_execution_performed": False,
            "model_loads": 0,
            "tokenizer_loads": 0,
            "gradients": 0,
            "optimizer_updates": 0,
            "checkpoint_writes": 0,
            "new_generation": False,
        },
        "non_claims": list(config["non_claims"]),
        "determinism": {
            "wall_clock_recorded": False,
            "ordering": ("declared protocol order for grid points; sorted "
                         "keys for maps"),
            "content_sha256_covers_receipt_without_this_field": True,
        },
        "limits": {
            "declared": dict(limits),
            "native_execution_performed": False,
            "bounds_checked": True,
        },
    }
    digest = _sha_json(body)
    receipt = dict(body, content_sha256=digest)
    serialized = json.dumps(receipt, indent=2, sort_keys=True,
                            allow_nan=False) + "\n"
    _require(len(serialized.encode("utf-8")) <= limits["max_output_bytes"],
             "Calibration receipt exceeds the declared output limit")
    return receipt


def write_receipt(receipt, path):
    """Write a derived receipt atomically, refusing to overwrite raw inputs."""
    path = Path(path)
    source = receipt["source"]
    for key in ("rankings", "inventory"):
        raw = Path(source[key]["path"]).resolve()
        if path.resolve() == raw:
            raise ValueError(f"Refusing to overwrite the raw {key} artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/memory_acceptance_calibration_v1.json")
    parser.add_argument("--output", type=Path,
                        help="override the configured derived-receipt path")
    parser.add_argument(
        "--diagnosis", type=Path,
        help=("emit the remediation memory diagnosis for an existing "
              "calibration receipt instead of replaying a config"))
    parser.add_argument(
        "--diagnosis-output", type=Path,
        default=ROOT / DIAGNOSIS_OUTPUT,
        help="override the remediation diagnosis receipt path")
    parser.add_argument("--quiet", action="store_true",
                        help="write the receipt without a stdout summary")
    args = parser.parse_args(argv)

    if args.diagnosis is not None:
        diagnosis_path = _resolve(args.diagnosis, ROOT)
        try:
            receipt = json.loads(diagnosis_path.read_text())
            diagnosis = analyze_memory_diagnosis(
                receipt, receipt_path=diagnosis_path, root=ROOT)
            output = _resolve(args.diagnosis_output, ROOT)
            write_diagnosis_receipt(diagnosis, output)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
            return 2
        if not args.quiet:
            print(json.dumps({
                "output": str(output),
                "output_bytes": output.stat().st_size,
                "content_sha256": diagnosis["content_sha256"],
                "address_headline": diagnosis["address"]["headline"],
                "address_layer_result": diagnosis["address"]["comparison"][
                    "accepted_layer_result"],
                "address_strict_group_result": diagnosis["address"][
                    "comparison"]["strict_group_result"],
                "address_wrong_layers": diagnosis["address"][
                    "layer_correctness"]["fact_wrong_layers"],
                "value_headline": diagnosis["value"]["headline"],
                "bottleneck": diagnosis["bottleneck"]["selected"],
                "address_readout_repaired": diagnosis["claims"][
                    "address_readout_repaired"],
                "calibration_is_a_solution": diagnosis["claims"][
                    "calibration_is_a_solution"],
                "heldout_rows_scored": diagnosis["claims"][
                    "heldout_rows_scored"],
            }, sort_keys=True))
        return 0

    config_path = _resolve(args.config, ROOT)
    try:
        config = json.loads(config_path.read_text())
        receipt = analyze_calibration(config, root=ROOT,
                                      config_path=config_path)
        output = _resolve(args.output or config["output"]["receipt"], ROOT)
        write_receipt(receipt, output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2
    if not args.quiet:
        gates = receipt["gate_classification"]
        native = gates["native_point"]
        print(json.dumps({
            "output": str(output),
            "output_bytes": output.stat().st_size,
            "content_sha256": receipt["content_sha256"],
            "source_rows": receipt["source"]["rankings"]["rows"],
            "scored_rows": receipt["source"]["scope"]["scored_rows"],
            "heldout_rows_scored": receipt["source"]["scope"][
                "heldout_rows_scored"],
            "value_frontier_points": receipt["value_frontier"]["point_count"],
            "address_frontier_points": receipt["address_frontier"]["point_count"],
            "candidate_variant": native["candidate_variant"],
            "candidate_meets_value_gates": native[
                "candidate_meets_value_gates"],
            "oracle_meets_gates": native["oracle_meets_gates"],
            "decision": native["decision"],
            "selected_rule": gates["disposition"]["selected_rule"],
            "oracle_answerable_fact_recoverability": native["observed"][
                "oracle_answerable_fact_recoverability"],
            "accepted_correct_rate": native["observed"][
                "accepted_correct_rate"],
            "accepted_precision": native["observed"]["accepted_precision"],
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
