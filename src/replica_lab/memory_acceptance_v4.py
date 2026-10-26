"""Versioned acceptance metrics v4 with an answerable-fact oracle gate.

v4 keeps the v3 separation of candidate quality from diagnostics and repairs
two things that v3 measured ambiguously:

* The oracle gate is answerable-fact recoverability: the numerator is a raw
  top-1 readout that is valid (nonzero, untied) on a ``kind == "fact"`` case
  and the denominator is ``fact_n``.  Unknown and revoked cases have their own
  abstention and false-recall rates and never enter that denominator.  The
  total-group raw top-1 rate stays in the receipt as a descriptive number and
  is never used as the oracle gate.
* Zero and ambiguous tied readouts receive no recoverability or acceptance
  credit.  The observed top-1 flag is preserved separately so the exclusion is
  visible instead of hidden.

Layer grouping is explicit and machine readable in ``layer_combination``:
``raw_top1`` is any valid layer, ``raw_top1_all`` is all layers,
``accepted`` is any layer, ``accepted_correct`` is all layers accepted and
correct, ``accepted_wrong`` is accepted but not all correct, and
``zero_or_tied`` is all layers zero or tied.  Uncertainty stays world
clustered; layers are not independent samples.

Replay is full by default and may be restricted to declared phases.  Phase
scoping validates exact subset coverage, recomputes the subset inventory units
and keeps the pinned parent inventory hash intact.  Missing, duplicate, extra
or out-of-declared-scope rows fail closed.  Revocation is required only when
the selected scope actually contains the revocation phase.

v3 modules, receipts and manifests are never modified and are never accepted
as v4.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from .memory_acceptance_v3 import (
    ALL_ARMS, ARM_CASE_COUNTS, DIAGNOSTIC_ARMS, IDENTITY_KEYS,
    PRIMARY_QUALITY_ARM, _declared_query_rows, _expected_identity, _groups,
    _normalized_inventory, _rate, _sha_json,
)
from .vision_curriculum import clustered_bootstrap

VERSION = "memory.acceptance.v4"
RECEIPT_FORMAT = "replica.memory_acceptance.metrics.v4"
CONFIG_FORMAT = "replica.memory_acceptance.config.v4"
MANIFEST_FORMAT = "replica.memory_acceptance.receipt_manifest.v4"
CURRENT_STATUS = "current"
SUPERSEDED_STATUS = "superseded"
HISTORICAL_STATUS = "historical"

BOOTSTRAP_KEYS = ("seed", "resamples", "confidence")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

LAYER_COMBINATION = {
    "raw_top1": "any_layer_valid_top1",
    "raw_top1_all": "all_layers_valid_top1",
    "observed_top1": "any_layer_observed_top1",
    "accepted": "any_layer_accepted",
    "accepted_correct": "all_layers_accepted_and_correct",
    "accepted_wrong": "accepted_and_not_all_correct",
    "zero_or_tied": "all_layers_zero_or_tied",
    "zero_or_tied_recoverability_credit": "none",
    "zero_or_tied_acceptance_credit": "none",
    "layers_are_independent_samples": False,
    "uncertainty_unit": "world",
}

ORACLE_POPULATION = {
    "gate_population": "answerable_fact",
    "numerator": "raw_valid_top1",
    "denominator": "fact_n",
    "excluded_kinds": ["unknown", "revoked"],
    "total_group_rate_is_descriptive": True,
    "is_legal_policy": False,
}

REQUIRED_GATES = (
    "accepted_correct_rate_min",
    "accepted_precision_min",
    "correction_accepted_correct_rate_min",
    "unaffected_accepted_correct_rate_min",
    "unknown_false_recall_max",
    "oracle_raw_top1_rate_min",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_keys(value, keys, *, label):
    _require(isinstance(value, Mapping), f"{label} must be an object")
    for key in keys:
        _require(key in value, f"{label} is missing: {key}")


def _at_least(value, minimum):
    return value is not None and value >= minimum


def _at_most(value, maximum):
    return value is not None and value <= maximum


def receipt_digest(receipt):
    return _sha_json(receipt)


def validate_config(config, *, require_inventory=True):
    """Validate a v4 acceptance config, optionally requiring its inventory."""
    _require(isinstance(config, Mapping), "Acceptance config must be an object")
    _require(config.get("format") == CONFIG_FORMAT,
             "Unsupported acceptance config format; v1/v2/v3 are not v4")
    _require(config.get("metric_version") == VERSION,
             "Acceptance consumer must declare memory.acceptance.v4")
    for key in ("primary_quality_arm", "diagnostic_arms", "candidate_variant",
                "oracle_variant", "gates", "bootstrap", "oracle_population",
                "layer_combination", "phase_scoping"):
        _require(key in config, f"Acceptance config is missing: {key}")
    _require(config["primary_quality_arm"] == PRIMARY_QUALITY_ARM,
             "v4 candidate quality must use query_key_value")
    _require(tuple(config["diagnostic_arms"]) == DIAGNOSTIC_ARMS,
             "v4 diagnostic arm inventory differs")
    _require(isinstance(config["gates"], Mapping),
             "Acceptance gates must be an object")
    for gate in REQUIRED_GATES:
        _require(gate in config["gates"], f"Acceptance gate is missing: {gate}")
    for gate, value in config["gates"].items():
        _require(isinstance(value, (int, float)) and not isinstance(value, bool)
                 and 0 <= value <= 1,
                 f"Acceptance gate {gate} is not a probability")
    _require(dict(config["layer_combination"]) == LAYER_COMBINATION,
             "v4 layer_combination semantics differ")
    _require(dict(config["oracle_population"]) == ORACLE_POPULATION,
             "v4 oracle population semantics differ")
    scoping = config["phase_scoping"]
    _require(isinstance(scoping, Mapping),
             "Acceptance phase_scoping must be an object")
    for key in ("subset_recomputes_units", "parent_inventory_pin_preserved",
                "revocation_required_only_when_selected"):
        _require(scoping.get(key) is True,
                 f"Acceptance phase_scoping must declare {key} true")
    _require(scoping.get("default") == "all",
             "Acceptance phase scoping default must be all phases")
    bootstrap = config["bootstrap"]
    _require_keys(bootstrap, BOOTSTRAP_KEYS, label="Acceptance bootstrap")
    _require(type(bootstrap["resamples"]) is int and bootstrap["resamples"] > 0,
             "Acceptance bootstrap resamples are invalid")
    _require(isinstance(bootstrap["confidence"], (int, float))
             and 0 < bootstrap["confidence"] < 1,
             "Acceptance bootstrap confidence is invalid")
    if require_inventory:
        expected = config.get("expected_inventory")
        _require_keys(expected, ("variants", "layers", "arms", "phases"),
                      label="Expected inventory")
        _require(isinstance(expected["variants"], list) and expected["variants"],
                 "Expected inventory variants are missing")
        _require(isinstance(expected["layers"], list) and expected["layers"],
                 "Expected inventory layers are missing")
        _require(all(type(layer) is int for layer in expected["layers"]),
                 "Expected inventory layers must be integers")
        _require(tuple(expected["arms"]) == ALL_ARMS,
                 "Expected inventory arms differ")
        _require(isinstance(expected["phases"], list) and expected["phases"],
                 "Expected inventory phases are missing")
        _require(all(isinstance(phase, str) and phase
                     for phase in expected["phases"]),
                 "Expected inventory phases must be nonempty strings")
        _require(("path" in expected) != ("rows" in expected),
                 "Expected inventory must name exactly one source or inline rows")
        if "path" in expected:
            _require(isinstance(expected["path"], str) and expected["path"],
                     "Expected inventory path is missing")
            _require(isinstance(expected.get("sha256"), str)
                     and len(expected["sha256"]) == 64,
                     "Expected inventory path requires a sha256 pin")
    return config


def validate_phase_scope(config, phases):
    """Return the canonical selected phases, defaulting to every phase."""
    declared = list(config["expected_inventory"]["phases"])
    if phases is None:
        return declared
    _require(not isinstance(phases, (str, bytes)),
             "Phase scope must be a sequence, not a string")
    requested = list(phases)
    _require(requested, "Phase scope is empty")
    for phase in requested:
        _require(isinstance(phase, str) and phase,
                 "Phase scope entries must be nonempty strings")
        _require(phase in declared,
                 f"Phase scope names an undeclared phase: {phase}")
    _require(len(requested) == len(set(requested)),
             "Phase scope contains duplicate phases")
    selected = set(requested)
    return [phase for phase in declared if phase in selected]


def load_expected_inventory(config):
    """Load the pinned expected inventory without deriving it from rankings."""
    config = validate_config(config)
    expected = config["expected_inventory"]
    if "rows" in expected:
        rows = expected["rows"]
        units = expected.get("units") or {}
    else:
        path = Path(expected["path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        _require(path.is_file(), f"Expected inventory is missing: {path}")
        _require(_sha(path) == expected["sha256"],
                 "Expected inventory bytes differ from the config pin")
        document = json.loads(path.read_text())
        _require(isinstance(document, Mapping)
                 and isinstance(document.get("rows"), list),
                 "Expected inventory document is malformed")
        rows = document["rows"]
        units = document.get("units") or {}
        expected = dict(expected)
        expected["resolved_path"] = str(path)
        expected["resolved_sha256"] = _sha(path)
    _require(isinstance(rows, list) and rows,
             "Expected inventory rows are missing")
    normalized = []
    seen = set()
    for row in rows:
        _require(isinstance(row, list) and len(row) == 8,
                 "Expected inventory rows must have eight fields")
        episode_id, split, state, phase, case_id, cohort, kind, target_role = row
        for value in (episode_id, split, state, phase, case_id, cohort, kind,
                      target_role):
            _require(isinstance(value, str) and value,
                     "Expected inventory fields must be nonempty strings")
        key = (episode_id, state, phase, case_id)
        _require(key not in seen, "Expected inventory contains duplicate cases")
        seen.add(key)
        _require(kind in ARM_CASE_COUNTS, f"Unknown inventory target kind: {kind}")
        normalized.append({
            "episode_id": episode_id, "split": split, "state": state,
            "phase": phase, "case_id": case_id, "cohort": cohort,
            "kind": kind, "target_role": target_role,
        })
    return {"rows": normalized, "units": units, "source": expected}


def _parent_hash(config, inventory):
    expected = config["expected_inventory"]
    source = inventory.get("source") or {}
    if "path" in expected:
        resolved = source.get("resolved_sha256")
        _require(resolved == expected["sha256"],
                 "Parent inventory hash differs from the config pin")
        return resolved
    digest = _sha_json(inventory["rows"])
    _require(len(digest) == 64, "Parent inventory hash is malformed")
    return digest


def _units_from_rows(rows, variants, layers, arms):
    expected_rows = _expected_identity(rows, variants, layers, arms)
    return {
        "declared_query_rows": _declared_query_rows(rows),
        "inventory_case_rows": len(rows),
        "arm_expanded_metric_rows": len(expected_rows),
    }, expected_rows


def _reconcile_pinned_units(pinned, actual, layers):
    _require(isinstance(pinned, Mapping), "Inventory units must be an object")
    for key, value in actual.items():
        declared = pinned.get(key)
        if key == "arm_expanded_metric_rows" and declared is None:
            per_layer = pinned.get("per_layer_arm_expanded_metric_rows")
            if per_layer is not None:
                declared = per_layer * len(layers)
        if declared is not None:
            _require(declared == value,
                     f"Inventory unit {key} does not reconcile")


def validate_ranking_inventory(rankings, config, inventory, *, phases=None):
    """Fail closed unless rankings exactly cover the selected phase subset.

    The pinned parent inventory is always reconciled in full.  Phase scoping
    then recomputes subset units from the selected inventory rows; it never
    replaces or weakens the parent inventory hash.
    """
    _require(isinstance(rankings, list) and rankings,
             "Ranking artifact is empty")
    inventory = _normalized_inventory(inventory)
    expected = config["expected_inventory"]
    variants = list(expected["variants"])
    layers = list(expected["layers"])
    arms = list(expected["arms"])
    declared_phases = list(expected["phases"])
    selected = validate_phase_scope(config, phases)
    _require(variants == list(dict.fromkeys(variants)),
             "Expected variants are duplicated")
    _require(layers == list(dict.fromkeys(layers)),
             "Expected layers are duplicated")

    parent_units, parent_expected = _units_from_rows(
        inventory["rows"], variants, layers, arms)
    _reconcile_pinned_units(inventory.get("units") or {}, parent_units, layers)
    parent_hash = _parent_hash(config, inventory)

    selected_set = set(selected)
    subset_rows = [row for row in inventory["rows"]
                   if row["phase"] in selected_set]
    _require(subset_rows,
             "Selected phase scope contains no expected inventory rows")
    subset_units, expected_rows = _units_from_rows(
        subset_rows, variants, layers, arms)

    actual = {}
    seen = {}
    out_of_scope = 0
    malformed = 0
    for index, row in enumerate(rankings):
        _require(isinstance(row, Mapping), f"Ranking row {index} is not an object")
        _require("phase" in row, f"Ranking row {index} is missing: phase")
        _require(row["phase"] in declared_phases,
                 f"Ranking row {index} has an undeclared phase")
        in_scope = row["phase"] in selected_set
        for key in IDENTITY_KEYS + ("world_id", "split", "kind",
                                    "target_role", "readout_norm", "accepted",
                                    "top1"):
            _require(key in row, f"Ranking row {index} is missing: {key}")
        _require(row["variant"] in variants,
                 f"Ranking row {index} has an undeclared variant")
        _require(row["arm"] in arms,
                 f"Ranking row {index} has an undeclared arm")
        _require(type(row["layer"]) is int,
                 f"Ranking row {index} has a malformed layer")
        _require(row["layer"] in layers,
                 f"Ranking row {index} has an undeclared layer")
        _require(row["kind"] in ARM_CASE_COUNTS,
                 f"Ranking row {index} has an unknown kind")
        _require(isinstance(row["accepted"], bool)
                 and isinstance(row["top1"], bool),
                 f"Ranking row {index} has malformed acceptance fields")
        _require(type(row["readout_norm"]) in (int, float)
                 and row["readout_norm"] >= 0,
                 f"Ranking row {index} has a malformed readout norm")
        if row["arm"] == "exact_key_value":
            _require(row["kind"] == "fact",
                     "exact_key_value rows exist only for fact targets")
        if row.get("policy_blocked") not in (None, "revoked", "stale"):
            malformed += 1
            raise ValueError(
                f"Ranking row {index} has an unsupported policy block")
        key = tuple(row[name] for name in IDENTITY_KEYS)
        row_inventory = expected_rows if in_scope else parent_expected
        _require(key in row_inventory,
                 f"Ranking row {index} is not in the expected inventory")
        _require(key not in seen, f"Duplicate ranking identity at row {index}")
        seen[key] = row
        case = row_inventory[key]
        for name in ("split", "cohort", "kind", "target_role"):
            _require(row[name] == case[name],
                     f"Ranking row {index} disagrees on {name}")
        if in_scope:
            actual[key] = row
        else:
            out_of_scope += 1

    missing = sorted(set(expected_rows) - set(actual))
    _require(not missing,
             f"Expected ranking rows are missing: {len(missing)} "
             f"(first {missing[:3]})")
    extra = sorted(set(actual) - set(expected_rows))
    _require(not extra,
             f"Unexpected ranking rows are present: {len(extra)}")
    scored_rows = len(rankings) - out_of_scope
    _require(scored_rows == len(expected_rows),
             "Ranking rows do not match the arm-expanded inventory")
    _require(subset_units["arm_expanded_metric_rows"] == sum(
        ARM_CASE_COUNTS[row["kind"]] * len(variants) * len(layers)
        for row in subset_rows),
        "Arm-expanded inventory does not reconcile")

    full = set(selected) == set(declared_phases)
    units = dict(subset_units, ranking_rows=scored_rows)
    parent_units = dict(parent_units,
                        ranking_rows=parent_units["arm_expanded_metric_rows"])
    scope = {
        "phases": list(selected),
        "full": full,
        "phase_scoped": not full,
        "parent_inventory_sha256": parent_hash,
        "parent_inventory_pin_verified": True,
        "parent_units_reconciled": True,
        "subset_units_recomputed": not full,
        "subset_validated": True,
        "revocation_in_scope": "revocation" in selected_set,
        "missing": 0,
        "duplicate": 0,
        "extra": 0,
        "malformed": malformed,
        "out_of_scope_rows": out_of_scope,
        "scored_rows": scored_rows,
    }
    return {
        "valid": True,
        "units": units,
        "parent_units": parent_units,
        "units_note": (
            "parent inventory units are the full pinned inventory; scored "
            "units are recomputed from the selected phase subset.  Declared "
            "query rows are not inventory case rows; inventory case rows "
            "become arm-expanded metric rows by variants x layers x "
            "kind-eligible arms"),
        "variants": len(variants), "layers": len(layers), "arms": len(arms),
        "inventory_source": inventory.get("source", {}),
        "scope": scope,
        "missing": 0, "duplicate": 0, "extra": 0, "malformed": malformed,
    }


def _classify(row):
    norm = row["readout_norm"]
    _require(isinstance(norm, (int, float)) and norm >= 0,
             "Readout norm must be nonnegative")
    zero = norm == 0
    top = row.get("top_cosine")
    second = row.get("second_cosine")
    tied = second is not None and top == second
    usable = not zero and not tied
    policy_blocked = row.get("policy_blocked")
    if policy_blocked is None:
        if row.get("stale_blocked"):
            policy_blocked = "stale"
        elif row.get("revoked_blocked"):
            policy_blocked = "revoked"
    observed_top1 = bool(row["top1"])
    raw_top1 = observed_top1 and usable
    accepted = bool(row["accepted"]) and usable and policy_blocked is None
    accepted_correct = accepted and raw_top1
    cross_world = bool(row.get("cross_world_selected", False))
    false_recall = accepted and not accepted_correct
    kind = row["kind"]
    if false_recall:
        kind_label = row.get("false_recall_kind")
        if kind == "revoked":
            kind_label = "revoked_target_other_live"
        elif kind == "unknown":
            kind_label = "unknown_address"
        elif not kind_label and cross_world:
            kind_label = "cross_world"
        elif not kind_label:
            kind_label = "other_source"
    else:
        kind_label = None
    runtime_available = "runtime_accepted" in row
    runtime_accepted = (bool(row.get("runtime_accepted"))
                        if runtime_available else None)
    runtime_selected = row.get("runtime_selected_record_id")
    target = row.get("target_entry_id")
    return {
        "variant": row["variant"],
        "arm": row["arm"],
        "phase": row["phase"],
        "episode_id": row["episode_id"],
        "world_id": row["world_id"],
        "split": row["split"],
        "state": row["state"],
        "case_id": row["case_id"],
        "cohort": row.get("cohort"),
        "layer": row["layer"],
        "kind": kind,
        "target_role": row.get("target_role"),
        "world": row["world_id"],
        "observed_top1": observed_top1,
        "raw_top1": raw_top1,
        "raw_top1_all": raw_top1,
        "accepted": accepted,
        "accepted_correct": accepted_correct,
        "accepted_wrong": accepted and not accepted_correct,
        "policy_blocked": policy_blocked,
        "policy_blocked_acceptance": (
            bool(row.get("policy_blocked_acceptance", False))
            or (bool(row.get("readout_accepted", row["accepted"]))
                and policy_blocked is not None)),
        "false_recall": false_recall,
        "false_recall_kind": kind_label,
        "cross_world_selected": cross_world,
        "zero_or_tied": not usable,
        "runtime_available": runtime_available,
        "runtime_accepted": runtime_accepted,
        "runtime_correct": (runtime_accepted is True
                            and runtime_selected is not None
                            and runtime_selected == target),
        "selection_correct": (runtime_selected is not None
                              and runtime_selected == target),
        "selected_other_live": (runtime_accepted is True
                                and runtime_selected is not None
                                and runtime_selected != target),
        "refresh_applied": bool(row.get("refresh_applied")),
        "weights_source": row.get("weights_source"),
    }


def _query_row(rows):
    rows = list(rows)
    first = rows[0]
    observed_top1 = any(row["observed_top1"] for row in rows)
    raw_top1 = any(row["raw_top1"] for row in rows)
    accepted = any(row["accepted"] for row in rows)
    accepted_correct = all(row["accepted_correct"] for row in rows)
    zero_or_tied = all(row["zero_or_tied"] for row in rows)
    return {
        "observed_top1": observed_top1,
        "observed_top1_all": all(row["observed_top1"] for row in rows),
        "raw_top1": raw_top1,
        "raw_top1_all": all(row["raw_top1"] for row in rows),
        "accepted": accepted,
        "accepted_correct": accepted_correct,
        "accepted_wrong": accepted and not accepted_correct,
        "policy_blocked": any(row["policy_blocked"] for row in rows),
        "policy_blocked_acceptance": any(
            row["policy_blocked_acceptance"] for row in rows),
        "false_recall": accepted and not accepted_correct,
        "false_recall_kind": next(
            (row["false_recall_kind"] for row in rows
             if row["false_recall_kind"]), None),
        "cross_world_selected": any(
            row["cross_world_selected"] for row in rows),
        "zero_or_tied": zero_or_tied,
        "zero_or_tied_observed_top1": zero_or_tied and observed_top1,
        "zero_or_tied_recoverability_credit": zero_or_tied and raw_top1,
        "zero_or_tied_acceptance_credit": zero_or_tied and accepted,
        "kind": first["kind"],
        "target_role": first["target_role"],
        "variant": first["variant"],
        "arm": first["arm"],
        "phase": first["phase"],
        "episode_id": first["episode_id"],
        "world_id": first["world_id"],
        "state": first["state"],
        "case_id": first["case_id"],
        "cohort": first["cohort"],
        "split": first["split"],
        "layers": tuple(sorted(row["layer"] for row in rows)),
        "runtime_available": any(row["runtime_available"] for row in rows),
        "runtime_accepted": any(
            row["runtime_accepted"] is True for row in rows),
        "runtime_correct": all(
            row["runtime_correct"] for row in rows
            if row["runtime_available"]),
        "selection_correct": any(row["selection_correct"] for row in rows),
        "selected_other_live": any(row["selected_other_live"] for row in rows),
        "refresh_applied": any(row["refresh_applied"] for row in rows),
        "weights_source": first["weights_source"],
    }


def _block(rows):
    rows = list(rows)
    n = len(rows)
    facts = [row for row in rows if row["kind"] == "fact"]
    unknown = [row for row in rows if row["kind"] == "unknown"]
    revoked = [row for row in rows if row["kind"] == "revoked"]
    fact_n = len(facts)

    def total(items, key):
        return sum(bool(row[key]) for row in items)

    observed_top1 = total(rows, "observed_top1")
    raw_top1 = total(rows, "raw_top1")
    raw_top1_all = total(rows, "raw_top1_all")
    accepted = total(rows, "accepted")
    accepted_correct = total(rows, "accepted_correct")
    accepted_correct_fact = total(facts, "accepted_correct")
    accepted_wrong = total(rows, "accepted_wrong")
    false_recall = total(rows, "false_recall")
    fact_recovered = total(facts, "raw_top1")
    fact_recovered_all = total(facts, "raw_top1_all")
    fact_observed = total(facts, "observed_top1")
    zero_or_tied = total(rows, "zero_or_tied")
    zero_or_tied_observed = sum(
        bool(row["zero_or_tied"]) and bool(row["observed_top1"])
        for row in rows)
    zero_or_tied_recovery_credit = sum(
        bool(row["zero_or_tied"]) and bool(row["raw_top1"]) for row in rows)
    zero_or_tied_acceptance_credit = sum(
        bool(row["zero_or_tied"]) and bool(row["accepted"]) for row in rows)
    unknown_accepted = total(unknown, "accepted")
    unknown_false_recall = total(unknown, "false_recall")
    revoked_accepted = total(revoked, "accepted")
    revoked_false_recall = total(revoked, "false_recall")
    runtime_rows = [row for row in rows if row["runtime_available"]]
    runtime_accepted = sum(row["runtime_accepted"] is True
                           for row in runtime_rows)
    return {
        "n": n,
        "fact_n": fact_n,
        "unknown_n": len(unknown),
        "revoked_n": len(revoked),
        "observed_top1": observed_top1,
        "observed_top1_rate": _rate(observed_top1, n),
        "raw_top1": raw_top1,
        "raw_top1_rate": _rate(raw_top1, n),
        "raw_top1_all": raw_top1_all,
        "raw_top1_all_rate": _rate(raw_top1_all, n),
        "fact_observed_top1": fact_observed,
        "fact_observed_top1_rate": _rate(fact_observed, fact_n),
        "fact_recovered": fact_recovered,
        "fact_recovered_all": fact_recovered_all,
        "recoverability_rate": _rate(fact_recovered, fact_n),
        "recoverability_all_layers_rate": _rate(fact_recovered_all, fact_n),
        "accepted": accepted,
        "acceptance_coverage": _rate(accepted, n),
        "abstention_rate": _rate(n - accepted, n),
        "accepted_correct": accepted_correct,
        "accepted_correct_fact": accepted_correct_fact,
        "accepted_correct_rate": _rate(accepted_correct_fact, fact_n),
        "accepted_precision": _rate(accepted_correct_fact, accepted),
        "accepted_wrong": accepted_wrong,
        "accepted_wrong_rate": _rate(accepted_wrong, n),
        "false_recall": false_recall,
        "false_recall_rate": _rate(false_recall, n),
        "false_recall_kinds": {
            kind: sum(row["false_recall_kind"] == kind for row in rows)
            for kind in sorted({row["false_recall_kind"] for row in rows
                                if row["false_recall_kind"]})},
        "cross_world_false_recall": sum(
            bool(row["cross_world_selected"]) and bool(row["false_recall"])
            for row in rows),
        "zero_or_tied": zero_or_tied,
        "zero_or_tied_rate": _rate(zero_or_tied, n),
        "zero_or_tied_observed_top1": zero_or_tied_observed,
        "zero_or_tied_recoverability_credit": zero_or_tied_recovery_credit,
        "zero_or_tied_acceptance_credit": zero_or_tied_acceptance_credit,
        "policy_blocked": total(rows, "policy_blocked"),
        "policy_blocked_acceptance": total(rows, "policy_blocked_acceptance"),
        "policy_blocked_raw_top1": sum(
            bool(row["policy_blocked"]) and bool(row["raw_top1"])
            for row in rows),
        "policy_blocked_observed_top1": sum(
            bool(row["policy_blocked"]) and bool(row["observed_top1"])
            for row in rows),
        "unknown_abstention_rate": _rate(
            len(unknown) - unknown_accepted, len(unknown)),
        "unknown_false_recall": unknown_false_recall,
        "unknown_false_recall_rate": _rate(
            unknown_false_recall, len(unknown)),
        "revoked_abstention_rate": _rate(
            len(revoked) - revoked_accepted, len(revoked)),
        "revoked_false_recall": revoked_false_recall,
        "revoked_false_recall_rate": _rate(
            revoked_false_recall, len(revoked)),
        "revoked_target_other_live": sum(
            row["false_recall_kind"] == "revoked_target_other_live"
            for row in revoked),
        "runtime_selection_available": bool(runtime_rows),
        "runtime_rows": len(runtime_rows),
        "runtime_accepted": runtime_accepted,
        "runtime_accepted_rate": _rate(runtime_accepted, len(runtime_rows)),
        "runtime_correct": sum(row["runtime_correct"]
                               for row in runtime_rows),
        "selected_other_live": sum(row["selected_other_live"]
                                   for row in runtime_rows),
        "refresh_applied_rows": sum(row["refresh_applied"] for row in rows),
    }


def _uncertainty(query_rows, *, seed, resamples, confidence):
    by_world = _groups(query_rows, lambda row: row["world_id"])
    result = {}
    for metric in ("accepted_correct_rate", "accepted_precision",
                   "acceptance_coverage", "false_recall_rate",
                   "recoverability_rate"):
        values = {}
        for world, rows in by_world.items():
            value = _block(rows)[metric]
            if value is not None:
                values[world] = [value]
        result[metric] = (clustered_bootstrap(
            values, resamples=resamples, confidence=confidence, seed=seed)
            if values else None)
    result["unit"] = "world"
    result["worlds"] = len(by_world)
    result["layer_readouts_are_not_independent"] = True
    return result


def _variant_metrics(rows):
    layer_rows = rows
    query_rows = [_query_row(group) for group in _groups(
        rows, lambda row: (row["variant"], row["episode_id"], row["state"],
                           row["case_id"], row["phase"], row["arm"])).values()]
    primary_queries = [row for row in query_rows
                       if row["arm"] == PRIMARY_QUALITY_ARM]
    exact_queries = [row for row in query_rows
                     if row["arm"] == "exact_key_value"]
    return {
        "quality": _block(primary_queries),
        "readout": _block(layer_rows),
        "correction": _block(row for row in primary_queries
                             if row["target_role"] == "correction_target"),
        "unaffected": _block(row for row in primary_queries
                             if row["target_role"] == "retained"),
        "unknown": _block(row for row in primary_queries
                          if row["kind"] == "unknown"),
        "revoked": _block(row for row in primary_queries
                          if row["kind"] == "revoked"),
        "exact_key_value": _block(exact_queries),
        "by_layer": {
            str(layer): _block(row for row in layer_rows if row["layer"] == layer)
            for layer in sorted({row["layer"] for row in layer_rows})},
        "by_episode": {
            episode: _block(row for row in primary_queries
                            if row["episode_id"] == episode)
            for episode in sorted({row["episode_id"]
                                   for row in primary_queries})},
        "by_world": {
            world: _block(row for row in primary_queries
                          if row["world_id"] == world)
            for world in sorted({row["world_id"] for row in primary_queries})},
        "counts": {
            "readout_rows": len(layer_rows),
            "query_groups": len(query_rows),
            "primary_query_groups": len(primary_queries),
            "exact_query_groups": len(exact_queries),
            "episodes": len({row["episode_id"] for row in primary_queries}),
            "states": len({(row["episode_id"], row["state"])
                           for row in primary_queries}),
            "phases": len({row["phase"] for row in primary_queries}),
            "independent_worlds": len({row["world_id"]
                                       for row in primary_queries}),
        },
    }


def _oracle_population(oracle_quality, oracle_variant):
    if oracle_quality is None:
        return dict(ORACLE_POPULATION, variant=oracle_variant,
                    available=False)
    return {
        **ORACLE_POPULATION,
        "variant": oracle_variant,
        "available": True,
        "fact_n": oracle_quality["fact_n"],
        "fact_recovered": oracle_quality["fact_recovered"],
        "answerable_fact_recoverability_rate": oracle_quality[
            "recoverability_rate"],
        "answerable_fact_recoverability_all_layers_rate": oracle_quality[
            "recoverability_all_layers_rate"],
        "observed_top1_rate": oracle_quality["observed_top1_rate"],
        "fact_observed_top1_rate": oracle_quality["fact_observed_top1_rate"],
        "total_group_raw_top1_rate": oracle_quality["raw_top1_rate"],
        "total_group_n": oracle_quality["n"],
        "excluded_non_answerable_n": (oracle_quality["unknown_n"]
                                      + oracle_quality["revoked_n"]),
        "zero_or_tied_observed_top1": oracle_quality[
            "zero_or_tied_observed_top1"],
        "zero_or_tied_recoverability_credit": oracle_quality[
            "zero_or_tied_recoverability_credit"],
    }


def acceptance_metrics(rankings, config):
    """Compute v4 metrics with candidate quality isolated from diagnostics."""
    config = validate_config(config, require_inventory=False)
    classified = [_classify(row) for row in rankings]
    keys = [tuple(row[name] for name in IDENTITY_KEYS) for row in classified]
    _require(len(keys) == len(set(keys)), "Duplicate ranking row identity")
    bootstrap = config["bootstrap"]
    variants = sorted({row["variant"] for row in classified})
    metrics = {}
    for variant in variants:
        variant_rows = [row for row in classified if row["variant"] == variant]
        variant_metrics = _variant_metrics(variant_rows)
        query_rows = [_query_row(group) for group in _groups(
            [row for row in variant_rows
             if row["arm"] == PRIMARY_QUALITY_ARM],
            lambda row: (row["episode_id"], row["state"], row["case_id"])).values()]
        variant_metrics["uncertainty"] = _uncertainty(
            query_rows, seed=bootstrap["seed"],
            resamples=bootstrap["resamples"],
            confidence=bootstrap["confidence"])
        metrics[variant] = variant_metrics
    exact_rows = [row for row in classified
                  if row["arm"] == "exact_key_value"]
    candidate = config["candidate_variant"]
    oracle = config["oracle_variant"]
    refresh_rows = sum(
        block["quality"]["refresh_applied_rows"]
        for block in metrics.values())
    runtime_available = any(
        block["quality"]["runtime_selection_available"]
        for block in metrics.values())
    candidate_quality = metrics.get(candidate, {}).get("quality")
    full_quality = metrics.get("shared_mean/full", {}).get("quality")
    oracle_quality = metrics.get(oracle, {}).get("quality")
    identical = (lambda left, right: left is not None
                 and right is not None and left == right)
    diagnostics = {
        "query_local_refresh_evidence": bool(refresh_rows),
        "query_local_refresh_applied_rows": refresh_rows,
        "runtime_selection_available": runtime_available,
        "candidate_equals_full_quality": identical(
            candidate_quality, full_quality),
        "oracle_equals_candidate_quality": identical(
            oracle_quality, candidate_quality),
        "oracle_total_group_rate_is_descriptive": True,
        "observed_top1_is_preserved": True,
        "zero_or_tied_rows_receive_no_recoverability_credit": True,
        "zero_or_tied_rows_receive_no_acceptance_credit": True,
        "fallback_is_explicit": True,
        "note": (
            "The oracle gate uses answerable-fact recoverability; unknown and "
            "revoked cases are excluded from that denominator.  The total-group "
            "raw top-1 rate is descriptive only.  Zero and tied readouts keep "
            "their observed top-1 flag for audit but never receive "
            "recoverability or acceptance credit.  When refresh evidence is "
            "absent, causal/oracle blocks are explicit full-state or mixture "
            "fallbacks; equality is reported, not hidden.  Runtime selection "
            "metrics are unavailable when the raw artifact predates decision "
            "rows."),
    }
    return {
        "metric_version": VERSION,
        "primary_quality_arm": PRIMARY_QUALITY_ARM,
        "layer_combination": dict(LAYER_COMBINATION),
        "oracle_population": _oracle_population(oracle_quality, oracle),
        "variants": metrics,
        "populations": {
            "candidate_quality": {
                "variant": candidate,
                "arm": PRIMARY_QUALITY_ARM,
                "blended_with_diagnostics": False,
            },
            "exact_key_value_diagnostic": {
                "arm": "exact_key_value",
                "is_upper_bound": False,
                "note": ("mixture diagnostic over a gold key readout; exact-key "
                         "rows exist only for fact targets and are never added "
                         "to candidate quality"),
                "rows": len(exact_rows),
            },
            "oracle_diagnostic": {
                **ORACLE_POPULATION,
                "variant": oracle,
                "note": ("gold-target diagnostic population gated as "
                         "answerable-fact recoverability, not an upper bound and "
                         "never a legal policy"),
            },
        },
        "diagnostics": diagnostics,
        "units": {
            "metric_rows": len(classified),
            "query_groups": sum(block["counts"]["query_groups"]
                                for block in metrics.values()),
            "primary_query_groups": sum(
                block["counts"]["primary_query_groups"]
                for block in metrics.values()),
        },
        "claims": {
            "exact_key_value_is_not_an_upper_bound": True,
            "oracle_is_diagnostic_only": True,
            "oracle_gate_uses_answerable_fact_recoverability": True,
            "unknown_and_revoked_excluded_from_oracle_denominator": True,
            "zero_and_tied_readouts_receive_no_credit": True,
            "policy_blocking_is_separate_from_raw_correctness": True,
            "revoked_target_other_live_is_wrong_retrieval": True,
            "missing_rows_fail_closed": True,
            "layers_are_not_independent_for_uncertainty": True,
            "observed_top1_is_preserved": True,
        },
    }


def evaluate_acceptance(metrics, config, *, require_revoked=True):
    """Apply v4 gates; missing coverage fails closed instead of abstaining.

    ``require_revoked`` is false only when the scored phase scope genuinely has
    no revocation phase (for example a first-process-only bundle).  It never
    excuses a missing fact, correction, unaffected, unknown or oracle block.
    """
    config = validate_config(config, require_inventory=False)
    candidate = config["candidate_variant"]
    oracle = config["oracle_variant"]
    variants = metrics.get("variants")
    _require(isinstance(variants, Mapping), "Acceptance metrics are malformed")
    _require(candidate in variants, "Candidate metrics are missing")
    _require(oracle in variants, "Oracle metrics are missing")
    quality = variants[candidate]["quality"]
    correction = variants[candidate]["correction"]
    unaffected = variants[candidate]["unaffected"]
    unknown = variants[candidate]["unknown"]
    revoked = variants[candidate]["revoked"]
    oracle_quality = variants[oracle]["quality"]
    coverage = {
        "candidate": quality["n"] > 0 and quality["fact_n"] > 0,
        "correction": correction["n"] > 0,
        "unaffected": unaffected["n"] > 0,
        "unknown": unknown["n"] > 0,
        "oracle": oracle_quality["n"] > 0 and oracle_quality["fact_n"] > 0,
    }
    if require_revoked:
        coverage["revoked"] = revoked["n"] > 0
    missing = [name for name, ok in coverage.items() if not ok]
    _require(not missing,
             f"Acceptance coverage is missing: {', '.join(missing)}")
    thresholds = config["gates"]
    zero_or_tied_excluded = all(
        block["zero_or_tied_recoverability_credit"] == 0
        and block["zero_or_tied_acceptance_credit"] == 0
        for block in (quality, oracle_quality, variants[candidate]["readout"]))
    oracle_rate = oracle_quality["recoverability_rate"]
    checks = {
        "accepted_correct_rate": _at_least(
            quality["accepted_correct_rate"],
            thresholds["accepted_correct_rate_min"]),
        "accepted_precision": _at_least(
            quality["accepted_precision"],
            thresholds["accepted_precision_min"]),
        "correction": _at_least(
            correction["accepted_correct_rate"],
            thresholds["correction_accepted_correct_rate_min"]),
        "unaffected": _at_least(
            unaffected["accepted_correct_rate"],
            thresholds["unaffected_accepted_correct_rate_min"]),
        "unknown_false_recall": _at_most(
            unknown["unknown_false_recall_rate"],
            thresholds["unknown_false_recall_max"]),
        "oracle_answerable_fact_recoverability": _at_least(
            oracle_rate, thresholds["oracle_raw_top1_rate_min"]),
        "zero_or_tied_credit_excluded": zero_or_tied_excluded,
    }
    if require_revoked and "revoked_false_recall_max" in thresholds:
        checks["revoked_false_recall"] = _at_most(
            revoked["revoked_false_recall_rate"],
            thresholds["revoked_false_recall_max"])
    return {
        "decision": "accepted" if all(checks.values()) else "rejected",
        "checks": checks,
        "coverage": coverage,
        "coverage_valid": True,
        "fail_closed_on_missing_coverage": True,
        "require_revoked": require_revoked,
        "accepted_correct_rate_min": thresholds["accepted_correct_rate_min"],
        "accepted_precision_min": thresholds["accepted_precision_min"],
        "correction_accepted_correct_rate_min": thresholds[
            "correction_accepted_correct_rate_min"],
        "unaffected_accepted_correct_rate_min": thresholds[
            "unaffected_accepted_correct_rate_min"],
        "unknown_false_recall_max": thresholds["unknown_false_recall_max"],
        "oracle_raw_top1_rate_min": thresholds["oracle_raw_top1_rate_min"],
        "oracle_gate_population": "answerable_fact",
        "oracle_numerator": "raw_valid_top1",
        "oracle_denominator": "fact_n",
        "oracle_answerable_fact_recoverability": oracle_rate,
        "oracle_fact_n": oracle_quality["fact_n"],
        "oracle_fact_recovered": oracle_quality["fact_recovered"],
        "oracle_total_group_raw_top1_rate": oracle_quality["raw_top1_rate"],
        "oracle_observed_top1_rate": oracle_quality["observed_top1_rate"],
        "oracle_non_answerable_excluded_n": (
            oracle_quality["unknown_n"] + oracle_quality["revoked_n"]),
        "oracle_is_legal_policy": False,
        "zero_or_tied_credit_excluded": zero_or_tied_excluded,
        "acceptance_coverage_policy": "report_only; denominators are explicit",
        "candidate_quality_arm": PRIMARY_QUALITY_ARM,
        "exact_key_value_blended": False,
        "layer_combination": dict(LAYER_COMBINATION),
    }


def _generation(generation_path):
    if generation_path is None:
        return {"status": "not_measured", "accuracy": None}
    generation_path = Path(generation_path)
    answers = [json.loads(line) for line in
               generation_path.read_text().splitlines() if line]
    by_arm = {}
    for arm in sorted({row["arm"] for row in answers}):
        rows = [row for row in answers if row["arm"] == arm]
        by_arm[arm] = {"n": len(rows),
                       "correct": sum(row["correct"] for row in rows),
                       "accuracy": _rate(sum(row["correct"] for row in rows),
                                         len(rows))}
    return {"status": "historical_reference_only",
            "source_sha256": _sha(generation_path), "by_arm": by_arm,
            "accuracy": None,
            "note": ("Generation is not vector-readout evidence and is excluded "
                     "from v4 acceptance gates")}


def rescore_ranking_file(path, config, *, generation_path=None, inventory=None,
                         phases=None):
    """Rescore an immutable raw ranking artifact into a fresh v4 receipt.

    ``phases`` selects a declared phase subset; the default scores every phase
    and therefore requires the exact full inventory.  Immutable source bytes
    are never rewritten.
    """
    config = validate_config(config)
    if inventory is None:
        inventory = load_expected_inventory(config)
    path = Path(path)
    original = path.read_bytes()
    rankings = [json.loads(line) for line in original.decode().splitlines()
                if line]
    _require(rankings, "Ranking artifact is empty")
    validation = validate_ranking_inventory(
        rankings, config, inventory, phases=phases)
    selected = set(validation["scope"]["phases"])
    scored = [row for row in rankings if row.get("phase") in selected]
    _require(scored, "Phase scope contains no ranking rows")
    metrics = acceptance_metrics(scored, config)
    require_revoked = "revocation" in selected
    verdict = evaluate_acceptance(
        metrics, config, require_revoked=require_revoked)
    expected = config["expected_inventory"]
    scope = validation["scope"]
    receipt = {
        "format": RECEIPT_FORMAT,
        "metric_version": VERSION,
        "source": {
            "path": str(path), "sha256": _sha(path), "rows": len(rankings),
            "scored_rows": len(scored),
            "out_of_scope_rows": scope["out_of_scope_rows"],
            "phases": list(scope["phases"]),
            "phase_scoped": not scope["full"],
            "original_bytes_unchanged": original == path.read_bytes(),
        },
        "config": {
            "format": config["format"],
            "metric_version": VERSION,
            "sha256": _sha_json(config),
        },
        "candidate_variant": config["candidate_variant"],
        "oracle_variant": config["oracle_variant"],
        "primary_quality_arm": PRIMARY_QUALITY_ARM,
        "layer_combination": dict(LAYER_COMBINATION),
        "oracle_population": dict(metrics["oracle_population"]),
        "inventory": {
            "sha256": scope["parent_inventory_sha256"],
            "path": validation["inventory_source"].get("resolved_path"),
            "units": validation["units"],
            "units_note": validation["units_note"],
            "parent_units": validation["parent_units"],
            "parent_units_reconciled": True,
            "validated": True,
            "scope": scope,
            "expected": {
                "variants": expected["variants"],
                "layers": expected["layers"],
                "arms": expected["arms"],
                "phases": expected["phases"],
            },
        },
        "metrics": metrics,
        "verdict": verdict,
        "generation": _generation(generation_path),
        "claims": {
            "raw_rankings_are_not_accepted_retrievals": True,
            "candidate_quality_is_query_key_value": True,
            "exact_key_value_is_not_an_upper_bound": True,
            "oracle_is_a_separate_diagnostic_population": True,
            "oracle_gate_uses_answerable_fact_recoverability": True,
            "unknown_and_revoked_excluded_from_oracle_denominator": True,
            "zero_and_tied_readouts_receive_no_credit": True,
            "policy_blocking_is_separate_from_raw_correctness": True,
            "revoked_target_other_live_is_wrong_retrieval": True,
            "missing_or_duplicate_rows_fail_closed": True,
            "phase_scope_preserves_parent_inventory_pin": True,
            "subset_units_are_recomputed": True,
            "refresh_fallback_is_explicit": True,
            "runtime_selection_metrics_require_decision_rows": True,
            "layers_are_not_independent_for_uncertainty": True,
            "observed_top1_is_preserved": True,
            "generated_answer_claims_are_separate": True,
        },
    }
    return receipt


def validate_receipt(receipt, *, config=None, superseded=(),
                     require_current=False):
    """Validate a strictly v4 receipt; v1/v2/v3 formats are rejected."""
    _require(isinstance(receipt, Mapping), "Acceptance receipt must be an object")
    _require(receipt.get("format") == RECEIPT_FORMAT,
             "Unsupported acceptance receipt format; v1/v2/v3 are not v4")
    _require(receipt.get("metric_version") == VERSION,
             "Unsupported acceptance metric version; v3 is not v4")
    for key in ("source", "config", "candidate_variant", "oracle_variant",
                "primary_quality_arm", "layer_combination", "oracle_population",
                "inventory", "metrics", "verdict", "generation", "claims"):
        _require(key in receipt, f"Acceptance receipt is missing: {key}")
    _require(receipt["primary_quality_arm"] == PRIMARY_QUALITY_ARM,
             "Receipt candidate quality arm differs")
    _require(dict(receipt["layer_combination"]) == LAYER_COMBINATION,
             "Receipt layer_combination semantics differ")
    _require(isinstance(receipt["oracle_population"], Mapping),
             "Receipt oracle population is malformed")
    verified_oracle = {
        key: receipt["oracle_population"].get(key)
        for key in ("gate_population", "numerator", "denominator",
                    "excluded_kinds", "total_group_rate_is_descriptive",
                    "is_legal_policy")}
    verified_oracle["excluded_kinds"] = list(
        verified_oracle["excluded_kinds"] or [])
    _require(verified_oracle == ORACLE_POPULATION,
             "Receipt oracle population semantics differ")
    source = receipt["source"]
    _require(isinstance(source, Mapping)
             and isinstance(source.get("sha256"), str)
             and len(source["sha256"]) == 64,
             "Acceptance receipt source hash is malformed")
    _require(source.get("original_bytes_unchanged") is True,
             "Acceptance receipt source bytes changed")
    _require(type(source.get("rows")) is int
             and type(source.get("scored_rows")) is int
             and type(source.get("out_of_scope_rows")) is int,
             "Acceptance receipt source row accounting is malformed")
    _require(source["rows"] == source["scored_rows"]
             + source["out_of_scope_rows"],
             "Acceptance receipt source rows do not reconcile")
    _require(isinstance(source.get("phases"), list) and source["phases"],
             "Acceptance receipt source phases are missing")
    inventory = receipt["inventory"]
    _require(isinstance(inventory, Mapping)
             and inventory.get("validated") is True,
             "Acceptance receipt has no validated inventory")
    _require(isinstance(inventory.get("sha256"), str)
             and len(inventory["sha256"]) == 64,
             "Acceptance receipt inventory hash is malformed")
    _require(isinstance(inventory.get("units"), Mapping)
             and isinstance(inventory.get("parent_units"), Mapping),
             "Acceptance receipt inventory units are malformed")
    _require(inventory.get("parent_units_reconciled") is True,
             "Acceptance receipt parent inventory units are not reconciled")
    scope = inventory.get("scope")
    _require(isinstance(scope, Mapping),
             "Acceptance receipt has no phase scope")
    _require(scope.get("parent_inventory_sha256") == inventory["sha256"],
             "Phase scope weakened the parent inventory hash")
    _require(scope.get("parent_inventory_pin_verified") is True,
             "Phase scope did not verify the parent inventory pin")
    _require(scope.get("subset_validated") is True,
             "Phase scope did not validate the scored subset")
    _require(scope.get("phase_scoped") == (scope.get("full") is False),
             "Phase scope flags do not reconcile")
    _require(isinstance(scope.get("phases"), list) and scope["phases"],
             "Phase scope phases are missing")
    _require(scope.get("scored_rows") == source["scored_rows"],
             "Phase scope scored rows do not match the source")
    _require(inventory["units"].get("ranking_rows") == source["scored_rows"],
             "Inventory units do not match the scored source rows")
    _require(list(scope["phases"]) == list(source["phases"]),
             "Phase scope phases do not match the source phases")
    _require(source.get("phase_scoped") == (scope.get("full") is False),
             "Source phase scoping flags do not reconcile")
    verdict = receipt["verdict"]
    _require(isinstance(verdict, Mapping)
             and verdict.get("decision") in {"accepted", "rejected"},
             "Acceptance receipt verdict is malformed")
    _require(verdict.get("coverage_valid") is True,
             "Acceptance receipt coverage is invalid")
    _require(verdict.get("fail_closed_on_missing_coverage") is True,
             "Acceptance receipt does not declare fail-closed coverage")
    _require(verdict.get("oracle_gate_population") == "answerable_fact"
             and verdict.get("oracle_denominator") == "fact_n"
             and verdict.get("oracle_numerator") == "raw_valid_top1",
             "Acceptance receipt oracle gate differs")
    _require(verdict.get("oracle_is_legal_policy") is False,
             "Acceptance receipt treats the oracle as a legal policy")
    _require(isinstance(verdict.get("checks"), Mapping),
             "Acceptance receipt verdict checks are malformed")
    _require("oracle_answerable_fact_recoverability" in verdict["checks"],
             "Acceptance receipt lacks the answerable-fact oracle check")
    _require(isinstance(receipt["metrics"], Mapping) and receipt["metrics"],
             "Acceptance receipt metrics are malformed")
    _require(receipt["metrics"].get("metric_version") == VERSION,
             "Acceptance receipt metrics are not v4")
    _require(dict(receipt["metrics"].get("layer_combination", {}))
             == LAYER_COMBINATION,
             "Acceptance receipt metrics layer_combination differ")
    _require(isinstance(receipt["metrics"].get("oracle_population"), Mapping),
             "Acceptance receipt metrics oracle population is malformed")
    _require(isinstance(receipt["generation"], Mapping),
             "Acceptance receipt generation is malformed")
    _require(isinstance(receipt["claims"], Mapping),
             "Acceptance receipt claims are malformed")
    for claim in ("exact_key_value_is_not_an_upper_bound",
                  "oracle_is_a_separate_diagnostic_population",
                  "oracle_gate_uses_answerable_fact_recoverability",
                  "unknown_and_revoked_excluded_from_oracle_denominator",
                  "zero_and_tied_readouts_receive_no_credit",
                  "policy_blocking_is_separate_from_raw_correctness",
                  "missing_or_duplicate_rows_fail_closed",
                  "phase_scope_preserves_parent_inventory_pin",
                  "subset_units_are_recomputed",
                  "layers_are_not_independent_for_uncertainty",
                  "observed_top1_is_preserved"):
        _require(receipt["claims"].get(claim) is True,
                 f"Acceptance receipt claim is missing: {claim}")
    if config is not None:
        config = validate_config(config, require_inventory=False)
        _require(config["candidate_variant"] == receipt["candidate_variant"],
                 "Receipt candidate variant does not match config")
        _require(config["oracle_variant"] == receipt["oracle_variant"],
                 "Receipt oracle variant does not match config")
        if "expected_inventory" in config:
            _require(tuple(scope["phases"]) == tuple(
                validate_phase_scope(config, scope["phases"])),
                "Receipt phase scope differs from the config declaration")
    if require_current:
        _require(receipt_digest(receipt) not in set(superseded),
                 "Acceptance receipt is superseded or historical")
    return receipt


def load_manifest(path):
    """Load the v4 manifest that names the current receipt."""
    manifest = json.loads(Path(path).read_text())
    _require(isinstance(manifest, Mapping), "Receipt manifest must be an object")
    _require(manifest.get("format") == MANIFEST_FORMAT,
             "Unsupported receipt manifest format; v1/v2/v3 are not v4")
    _require(manifest.get("metric_version") == VERSION,
             "Unsupported receipt manifest metric version; v3 is not v4")
    authoritative = manifest.get("authoritative")
    _require(isinstance(authoritative, Mapping),
             "Receipt manifest is missing the authoritative receipt")
    _require(authoritative.get("status") == CURRENT_STATUS,
             "Receipt manifest has no current authoritative receipt")
    _require(authoritative.get("receipt_format") == RECEIPT_FORMAT
             and authoritative.get("metric_version") == VERSION,
             "Authoritative receipt is not v4")
    receipts = manifest.get("receipts")
    _require(isinstance(receipts, list), "Receipt manifest lacks its receipt list")
    statuses = set()
    for entry in receipts:
        _require(isinstance(entry, Mapping), "Malformed receipt manifest entry")
        status = entry.get("status")
        _require(status in {CURRENT_STATUS, SUPERSEDED_STATUS,
                            HISTORICAL_STATUS},
                 "Unknown receipt manifest status")
        _require(isinstance(entry.get("file_sha256"), str)
                 and len(entry["file_sha256"]) == 64,
                 "Manifest entry lacks a file hash")
        _require(isinstance(entry.get("receipt_digest"), str)
                 and len(entry["receipt_digest"]) == 64,
                 "Manifest entry lacks a receipt digest")
        statuses.add(status)
    _require(CURRENT_STATUS in statuses,
             "Receipt manifest lacks a current entry")
    _require(sum(entry["status"] == CURRENT_STATUS for entry in receipts) == 1,
             "Receipt manifest must name exactly one current entry")
    current = next(entry for entry in receipts
                   if entry["status"] == CURRENT_STATUS)
    _require(current.get("path") == authoritative.get("path")
             and current.get("file_sha256")
             == authoritative.get("file_sha256")
             and current.get("receipt_digest")
             == authoritative.get("receipt_digest"),
             "Manifest current entry disagrees with the authoritative receipt")
    return manifest


def _non_current_digests(manifest):
    digests = set()
    for entry in manifest["receipts"]:
        if entry["status"] == CURRENT_STATUS:
            continue
        digests.add(entry["receipt_digest"])
    _require(manifest["authoritative"].get("receipt_digest") not in digests,
             "The authoritative receipt is marked non-current")
    return digests


def load_current_receipt(manifest_path, *, config=None):
    """Load and validate the sole current v4 receipt named by a v4 manifest."""
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    entry = manifest["authoritative"]
    for key in ("path", "file_sha256", "receipt_digest"):
        _require(isinstance(entry.get(key), str) and entry[key],
                 f"Authoritative receipt is missing: {key}")
    path = manifest_path.parent / entry["path"]
    _require(path.is_file(), "Authoritative receipt path is missing")
    _require(_sha(path) == entry["file_sha256"],
             "Authoritative receipt bytes do not match the manifest")
    receipt = json.loads(path.read_text())
    validate_receipt(receipt, config=config,
                     superseded=_non_current_digests(manifest),
                     require_current=True)
    _require(receipt_digest(receipt) == entry["receipt_digest"],
             "Authoritative receipt digest does not match the manifest")
    return receipt


def load_receipt(path, *, config=None, manifest_path=None,
                 require_current=False):
    if require_current and manifest_path is None:
        raise ValueError("Current receipt selection requires a v4 manifest")
    superseded = ()
    if manifest_path is not None:
        superseded = _non_current_digests(load_manifest(manifest_path))
    receipt = json.loads(Path(path).read_text())
    return validate_receipt(receipt, config=config, superseded=superseded,
                            require_current=require_current)


def build_manifest(*, receipt_path, receipt, historical=(), superseded=(),
                   updated=None):
    """Build a v4 manifest entry list with explicit non-current history."""
    receipt_path = Path(receipt_path)
    _require(receipt.get("format") == RECEIPT_FORMAT
             and receipt.get("metric_version") == VERSION,
             "Only a v4 receipt can be the current manifest entry")
    digest = receipt_digest(receipt)
    entries = [{
        "path": receipt_path.name,
        "status": CURRENT_STATUS,
        "role": "sole current authoritative v4 receipt",
        "receipt_format": receipt["format"],
        "metric_version": receipt["metric_version"],
        "file_sha256": _sha(receipt_path),
        "receipt_digest": digest,
        "config_sha256": receipt.get("config", {}).get("sha256"),
        "config_file_sha256": receipt.get("config", {}).get("file_sha256"),
        "source_sha256": receipt["source"]["sha256"],
        "scored_rows": receipt["source"]["scored_rows"],
        "phases": receipt["source"]["phases"],
    }]
    for item in historical:
        path = Path(item["path"])
        document = json.loads(path.read_text())
        entries.append({
            "path": path.name,
            "status": HISTORICAL_STATUS,
            "reason": item["reason"],
            "receipt_format": document.get("format"),
            "metric_version": document.get("metric_version"),
            "file_sha256": _sha(path),
            "receipt_digest": receipt_digest(document),
            "not_v4": document.get("metric_version") != VERSION,
        })
    for item in superseded:
        path = Path(item["path"])
        document = json.loads(path.read_text())
        entries.append({
            "path": path.name,
            "status": SUPERSEDED_STATUS,
            "reason": item["reason"],
            "superseded_by": receipt_path.name,
            "receipt_format": document.get("format"),
            "metric_version": document.get("metric_version"),
            "file_sha256": _sha(path),
            "receipt_digest": receipt_digest(document),
            "not_v4": document.get("metric_version") != VERSION,
        })
    return {
        "format": MANIFEST_FORMAT,
        "metric_version": VERSION,
        "updated": updated,
        "authoritative": dict(entries[0]),
        "receipts": entries,
    }
