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
    parser.add_argument("--quiet", action="store_true",
                        help="write the receipt without a stdout summary")
    args = parser.parse_args(argv)

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
