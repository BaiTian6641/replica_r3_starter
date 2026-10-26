"""Versioned acceptance metrics v3 with strict inventory and separated populations.

Candidate quality is the actual ``query_key_value`` readout.  The exact-key
readout is a mixture diagnostic, never an upper bound and never blended into
candidate quality.  Oracle rows form a separate gold-target diagnostic
population.  Raw rankings stay immutable; receipts are written beside them.

Inventory is validated in three explicit units:

* ``declared_query_rows``: unique episode/phase/query rows before state
  duplication;
* ``inventory_case_rows``: episode/state/phase/query cases;
* ``arm_expanded_metric_rows``: case rows multiplied by variants, layers and
  the arms that can exist for the target kind (exact-key only for facts).

Any missing, extra, malformed or duplicate ranking row fails closed.  A v1 or
v2 receipt is never silently accepted as v3.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .vision_curriculum import clustered_bootstrap

VERSION = "memory.acceptance.v3"
RECEIPT_FORMAT = "replica.memory_acceptance.metrics.v3"
CONFIG_FORMAT = "replica.memory_acceptance.config.v3"
MANIFEST_FORMAT = "replica.memory_acceptance.receipt_manifest.v3"
CURRENT_STATUS = "current"
SUPERSEDED_STATUS = "superseded"
HISTORICAL_STATUS = "historical"

PRIMARY_QUALITY_ARM = "query_key_value"
DIAGNOSTIC_ARMS = ("exact_key_value", "query_key_address", "context_removed")
VALUE_ARMS = frozenset({PRIMARY_QUALITY_ARM, "exact_key_value"})
ALL_ARMS = (PRIMARY_QUALITY_ARM,) + DIAGNOSTIC_ARMS
ARM_CASE_COUNTS = {"fact": 4, "unknown": 3, "revoked": 3}
BOOTSTRAP_KEYS = ("seed", "resamples", "confidence")
IDENTITY_KEYS = ("variant", "arm", "phase", "episode_id", "state", "case_id",
                 "layer")
INVENTORY_IDENTITY_KEYS = ("episode_id", "state", "phase", "case_id")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha_json(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def receipt_digest(receipt):
    return _sha_json(receipt)


def _require_keys(value, keys, *, label):
    _require(isinstance(value, Mapping), f"{label} must be an object")
    for key in keys:
        _require(key in value, f"{label} is missing: {key}")


def validate_config(config, *, require_inventory=True):
    """Validate a v3 acceptance config, optionally requiring its inventory."""
    _require(isinstance(config, Mapping), "Acceptance config must be an object")
    _require(config.get("format") == CONFIG_FORMAT,
             "Unsupported acceptance config format")
    _require(config.get("metric_version") == VERSION,
             "Acceptance consumer must declare memory.acceptance.v3")
    for key in ("primary_quality_arm", "diagnostic_arms",
                "candidate_variant", "oracle_variant", "gates", "bootstrap"):
        _require(key in config, f"Acceptance config is missing: {key}")
    _require(config["primary_quality_arm"] == PRIMARY_QUALITY_ARM,
             "v3 candidate quality must use query_key_value")
    _require(tuple(config["diagnostic_arms"]) == DIAGNOSTIC_ARMS,
             "v3 diagnostic arm inventory differs")
    _require(isinstance(config["gates"], Mapping),
             "Acceptance gates must be an object")
    for gate in ("accepted_correct_rate_min", "accepted_precision_min",
                 "correction_accepted_correct_rate_min",
                 "unaffected_accepted_correct_rate_min",
                 "unknown_false_recall_max", "oracle_raw_top1_rate_min"):
        _require(gate in config["gates"], f"Acceptance gate is missing: {gate}")
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
        _require(tuple(expected["arms"]) == ALL_ARMS,
                 "Expected inventory arms differ")
        _require(isinstance(expected["phases"], list) and expected["phases"],
                 "Expected inventory phases are missing")
        _require(("path" in expected) != ("rows" in expected),
                 "Expected inventory must name exactly one source or inline rows")
        if "path" in expected:
            _require(isinstance(expected["path"], str) and expected["path"],
                     "Expected inventory path is missing")
            _require(isinstance(expected.get("sha256"), str)
                     and len(expected["sha256"]) == 64,
                     "Expected inventory path requires a sha256 pin")
    return config


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


def _expected_identity(rows, variants, layers, arms):
    expected = {}
    for row in rows:
        arm_names = [arm for arm in arms
                     if arm != "exact_key_value" or row["kind"] == "fact"]
        for variant in variants:
            for layer in layers:
                for arm in arm_names:
                    key = (variant, arm, row["phase"], row["episode_id"],
                           row["state"], row["case_id"], layer)
                    expected[key] = {
                        "split": row["split"], "cohort": row["cohort"],
                        "kind": row["kind"],
                        "target_role": row["target_role"],
                    }
    return expected


def _declared_query_rows(rows):
    return len({(row["episode_id"], row["phase"], row["case_id"])
                for row in rows})


def _normalized_inventory(inventory):
    _require(isinstance(inventory, Mapping)
             and isinstance(inventory.get("rows"), list),
             "Expected inventory is malformed")
    rows = []
    for row in inventory["rows"]:
        if isinstance(row, Mapping):
            rows.append(dict(row))
            continue
        _require(isinstance(row, list) and len(row) == 8,
                 "Expected inventory rows must have eight fields")
        keys = ("episode_id", "split", "state", "phase", "case_id",
                "cohort", "kind", "target_role")
        rows.append(dict(zip(keys, row)))
    return dict(inventory, rows=rows)


def validate_ranking_inventory(rankings, config, inventory):
    """Fail closed unless rankings exactly cover the pinned inventory."""
    _require(isinstance(rankings, list) and rankings,
             "Ranking artifact is empty")
    inventory = _normalized_inventory(inventory)
    expected = config["expected_inventory"]
    variants = list(expected["variants"])
    layers = list(expected["layers"])
    arms = list(expected["arms"])
    phases = set(expected["phases"])
    _require(variants == list(dict.fromkeys(variants)),
             "Expected variants are duplicated")
    _require(layers == list(dict.fromkeys(layers)),
             "Expected layers are duplicated")
    expected_rows = _expected_identity(
        inventory["rows"], variants, layers, arms)
    actual = {}
    malformed = 0
    for index, row in enumerate(rankings):
        _require(isinstance(row, Mapping), f"Ranking row {index} is not an object")
        for key in IDENTITY_KEYS + ("world_id", "split", "kind",
                                    "target_role", "readout_norm", "accepted",
                                    "top1"):
            _require(key in row, f"Ranking row {index} is missing: {key}")
        _require(row["variant"] in variants,
                 f"Ranking row {index} has an undeclared variant")
        _require(row["arm"] in arms,
                 f"Ranking row {index} has an undeclared arm")
        _require(row["phase"] in phases,
                 f"Ranking row {index} has an undeclared phase")
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
        _require(key in expected_rows,
                 f"Ranking row {index} is not in the expected inventory")
        _require(key not in actual, f"Duplicate ranking identity at row {index}")
        case = expected_rows[key]
        for name in ("split", "cohort", "kind", "target_role"):
            _require(row[name] == case[name],
                     f"Ranking row {index} disagrees on {name}")
        actual[key] = row
    missing = sorted(set(expected_rows) - set(actual))
    _require(not missing,
             f"Expected ranking rows are missing: {len(missing)} "
             f"(first {missing[:3]})")
    extra = sorted(set(actual) - set(expected_rows))
    _require(not extra,
             f"Unexpected ranking rows are present: {len(extra)}")
    declared = _declared_query_rows(inventory["rows"])
    inventory_cases = len(inventory["rows"])
    arm_expanded = len(expected_rows)
    expected_units = {
        "declared_query_rows": declared,
        "inventory_case_rows": inventory_cases,
        "arm_expanded_metric_rows": arm_expanded,
        "ranking_rows": len(rankings),
    }
    for key, value in expected_units.items():
        pinned = inventory.get("units", {}).get(key)
        if key == "arm_expanded_metric_rows" and pinned is None:
            per_layer = inventory.get("units", {}).get(
                "per_layer_arm_expanded_metric_rows")
            if per_layer is not None:
                pinned = per_layer * len(layers)
        if pinned is not None:
            _require(pinned == value,
                     f"Inventory unit {key} does not reconcile")
    _require(arm_expanded == sum(
        ARM_CASE_COUNTS[row["kind"]] * len(variants) * len(layers)
        for row in inventory["rows"]),
        "Arm-expanded inventory does not reconcile")
    _require(len(rankings) == arm_expanded,
             "Ranking rows do not match the arm-expanded inventory")
    return {
        "valid": True,
        "units": expected_units,
        "units_note": (
            "declared query rows are not inventory case rows; inventory case "
            "rows become arm-expanded metric rows by variants x layers x "
            "kind-eligible arms"),
        "variants": len(variants), "layers": len(layers), "arms": len(arms),
        "inventory_source": inventory.get("source", {}),
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
    accepted = bool(row["accepted"]) and usable and policy_blocked is None
    top1 = bool(row["top1"])
    false_recall = accepted and not top1
    kind = row["kind"]
    if false_recall:
        kind_label = row.get("false_recall_kind")
        if kind == "revoked":
            kind_label = "revoked_target_other_live"
        elif kind == "unknown":
            kind_label = "unknown_address"
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
        "raw_top1": top1,
        "accepted": accepted,
        "accepted_correct": accepted and top1,
        "accepted_wrong": accepted and not top1,
        "policy_blocked": policy_blocked,
        "policy_blocked_acceptance": (
            bool(row.get("policy_blocked_acceptance", False))
            or (bool(row.get("readout_accepted", row["accepted"]))
                and policy_blocked is not None)),
        "false_recall": false_recall,
        "false_recall_kind": kind_label,
        "zero_or_tied": zero or tied,
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
    accepted_correct = all(row["accepted_correct"] for row in rows)
    accepted = any(row["accepted"] for row in rows)
    first = rows[0]
    return {
        "raw_top1": any(row["raw_top1"] for row in rows),
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
        "zero_or_tied": all(row["zero_or_tied"] for row in rows),
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
    fact_n = sum(row["kind"] == "fact" for row in rows)
    accepted = sum(row["accepted"] for row in rows)
    correct = sum(row["accepted_correct"] for row in rows)
    wrong = sum(row["accepted_wrong"] for row in rows)
    unknown = [row for row in rows if row["kind"] == "unknown"]
    revoked = [row for row in rows if row["kind"] == "revoked"]
    runtime_rows = [row for row in rows if row["runtime_available"]]
    runtime_accepted = sum(row["runtime_accepted"] is True
                           for row in runtime_rows)
    return {
        "n": n,
        "fact_n": fact_n,
        "unknown_n": len(unknown),
        "revoked_n": len(revoked),
        "raw_top1": sum(row["raw_top1"] for row in rows),
        "raw_top1_rate": _rate(sum(row["raw_top1"] for row in rows), n),
        "raw_top1_all": sum(row.get("raw_top1_all", row["raw_top1"])
                            for row in rows),
        "raw_top1_all_rate": _rate(
            sum(row.get("raw_top1_all", row["raw_top1"]) for row in rows), n),
        "accepted": accepted,
        "acceptance_coverage": _rate(accepted, n),
        "abstention_rate": _rate(n - accepted, n),
        "accepted_correct": correct,
        "accepted_correct_rate": _rate(correct, fact_n),
        "accepted_precision": _rate(correct, accepted),
        "accepted_wrong": wrong,
        "false_recall": sum(row["false_recall"] for row in rows),
        "false_recall_rate": _rate(
            sum(row["false_recall"] for row in rows), n),
        "false_recall_kinds": {
            kind: sum(row["false_recall_kind"] == kind for row in rows)
            for kind in sorted({row["false_recall_kind"] for row in rows
                                if row["false_recall_kind"]})},
        "policy_blocked": sum(bool(row["policy_blocked"]) for row in rows),
        "policy_blocked_acceptance": sum(
            row["policy_blocked_acceptance"] for row in rows),
        "zero_or_tied_top1": sum(
            row["zero_or_tied"] and row["raw_top1"] for row in rows),
        "unknown_false_recall": sum(row["false_recall"] for row in unknown),
        "unknown_false_recall_rate": _rate(
            sum(row["false_recall"] for row in unknown), len(unknown)),
        "revoked_abstention_rate": _rate(
            sum(not row["accepted"] for row in revoked), len(revoked)),
        "revoked_false_recall": sum(row["false_recall"] for row in revoked),
        "revoked_false_recall_rate": _rate(
            sum(row["false_recall"] for row in revoked), len(revoked)),
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


def _groups(rows, key):
    grouped = {}
    for row in rows:
        grouped.setdefault(key(row), []).append(row)
    return grouped


def _uncertainty(query_rows, *, seed, resamples, confidence):
    by_world = _groups(query_rows, lambda row: row["world_id"])
    result = {}
    for metric in ("accepted_correct_rate", "accepted_precision",
                   "acceptance_coverage", "false_recall_rate"):
        values = {}
        for world, rows in by_world.items():
            value = _block(rows)[metric]
            if value is not None:
                values[world] = [value]
        result[metric] = (clustered_bootstrap(
            values, resamples=resamples, confidence=confidence, seed=seed)
            if values else None)
    result["unit"] = "world"
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


def acceptance_metrics(rankings, config):
    """Compute v3 metrics with candidate quality isolated from diagnostics."""
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
        "fallback_is_explicit": True,
        "note": (
            "When refresh evidence is absent, causal/oracle blocks are explicit "
            "full-state or mixture fallbacks; equality is reported, not hidden. "
            "Runtime selection metrics are unavailable when the raw artifact "
            "predates decision rows."),
    }
    return {
        "metric_version": VERSION,
        "primary_quality_arm": PRIMARY_QUALITY_ARM,
        "variants": metrics,
        "populations": {
            "candidate_quality": {
                "variant": config["candidate_variant"],
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
                "variant": config["oracle_variant"],
                "is_legal_policy": False,
                "note": "gold-target diagnostic population, not an upper bound",
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
            "revoked_target_other_live_is_wrong_retrieval": True,
            "policy_blocking_is_separate_from_evaluator_abstention": True,
            "missing_rows_fail_closed": True,
            "layers_are_not_independent_for_uncertainty": True,
        },
    }


def evaluate_acceptance(metrics, config, *, require_revoked=True):
    """Apply v3 gates; missing coverage fails closed instead of abstaining.

    ``require_revoked`` is false only when the scored population genuinely has
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
    checks = {
        "accepted_correct_rate": (
            quality["accepted_correct_rate"]
            >= thresholds["accepted_correct_rate_min"]),
        "accepted_precision": (
            quality["accepted_precision"] is not None
            and quality["accepted_precision"]
            >= thresholds["accepted_precision_min"]),
        "correction": (
            correction["accepted_correct_rate"]
            >= thresholds["correction_accepted_correct_rate_min"]),
        "unaffected": (
            unaffected["accepted_correct_rate"]
            >= thresholds["unaffected_accepted_correct_rate_min"]),
        "unknown_false_recall": (
            unknown["unknown_false_recall_rate"]
            <= thresholds["unknown_false_recall_max"]),
        "oracle_raw_bound": (
            oracle_quality["raw_top1_rate"] is not None
            and oracle_quality["raw_top1_rate"]
            >= thresholds["oracle_raw_top1_rate_min"]),
    }
    if require_revoked and "revoked_false_recall_max" in thresholds:
        checks["revoked_false_recall"] = (
            revoked["revoked_false_recall_rate"] is not None
            and revoked["revoked_false_recall_rate"]
            <= thresholds["revoked_false_recall_max"])
    return {
        "decision": "accepted" if all(checks.values()) else "rejected",
        "checks": checks,
        "coverage": coverage,
        "coverage_valid": True,
        "fail_closed_on_missing_coverage": True,
        "accepted_correct_rate_min": thresholds["accepted_correct_rate_min"],
        "accepted_precision_min": thresholds["accepted_precision_min"],
        "correction_accepted_correct_rate_min": thresholds[
            "correction_accepted_correct_rate_min"],
        "unaffected_accepted_correct_rate_min": thresholds[
            "unaffected_accepted_correct_rate_min"],
        "unknown_false_recall_max": thresholds["unknown_false_recall_max"],
        "oracle_raw_top1_rate_min": thresholds["oracle_raw_top1_rate_min"],
        "acceptance_coverage_policy": "report_only; denominators are explicit",
        "candidate_quality_arm": PRIMARY_QUALITY_ARM,
        "exact_key_value_blended": False,
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
                     "from v3 acceptance gates")}


def rescore_ranking_file(path, config, *, generation_path=None,
                         inventory=None):
    """Rescore an immutable raw ranking artifact into a fresh v3 receipt."""
    config = validate_config(config)
    if inventory is None:
        inventory = load_expected_inventory(config)
    path = Path(path)
    original = path.read_bytes()
    rankings = [json.loads(line) for line in original.decode().splitlines()
                if line]
    _require(rankings, "Ranking artifact is empty")
    validation = validate_ranking_inventory(rankings, config, inventory)
    metrics = acceptance_metrics(rankings, config)
    verdict = evaluate_acceptance(metrics, config)
    expected = config["expected_inventory"]
    receipt = {
        "format": RECEIPT_FORMAT,
        "metric_version": VERSION,
        "source": {
            "path": str(path), "sha256": _sha(path), "rows": len(rankings),
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
        "inventory": {
            "sha256": validation["inventory_source"].get("resolved_sha256")
            or _sha_json(inventory["rows"]),
            "path": validation["inventory_source"].get("resolved_path"),
            "units": validation["units"],
            "units_note": validation["units_note"],
            "validated": True,
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
            "revoked_target_other_live_is_wrong_retrieval": True,
            "missing_or_duplicate_rows_fail_closed": True,
            "refresh_fallback_is_explicit": True,
            "runtime_selection_metrics_require_decision_rows": True,
            "zero_and_ambiguous_tied_readouts_receive_no_credit": True,
            "layers_are_not_independent_for_uncertainty": True,
            "generated_answer_claims_are_separate": True,
        },
    }
    return receipt


def validate_receipt(receipt, *, config=None, superseded=(),
                     require_current=False):
    """Validate a strictly v3 receipt; v1/v2 formats are rejected."""
    _require(isinstance(receipt, Mapping), "Acceptance receipt must be an object")
    _require(receipt.get("format") == RECEIPT_FORMAT,
             "Unsupported acceptance receipt format; v1/v2 are not v3")
    _require(receipt.get("metric_version") == VERSION,
             "Unsupported acceptance metric version")
    for key in ("source", "config", "candidate_variant", "oracle_variant",
                "primary_quality_arm", "inventory", "metrics", "verdict",
                "generation", "claims"):
        _require(key in receipt, f"Acceptance receipt is missing: {key}")
    _require(receipt["primary_quality_arm"] == PRIMARY_QUALITY_ARM,
             "Receipt candidate quality arm differs")
    source = receipt["source"]
    _require(isinstance(source, Mapping)
             and isinstance(source.get("sha256"), str)
             and len(source["sha256"]) == 64,
             "Acceptance receipt source hash is malformed")
    _require(source.get("original_bytes_unchanged") is True,
             "Acceptance receipt source bytes changed")
    inventory = receipt["inventory"]
    _require(isinstance(inventory, Mapping)
             and inventory.get("validated") is True,
             "Acceptance receipt has no validated inventory")
    verdict = receipt["verdict"]
    _require(isinstance(verdict, Mapping)
             and verdict.get("decision") in {"accepted", "rejected"},
             "Acceptance receipt verdict is malformed")
    _require(verdict.get("coverage_valid") is True,
             "Acceptance receipt coverage is invalid")
    _require(verdict.get("fail_closed_on_missing_coverage") is True,
             "Acceptance receipt does not declare fail-closed coverage")
    _require(isinstance(receipt["metrics"], Mapping) and receipt["metrics"],
             "Acceptance receipt metrics are malformed")
    _require(isinstance(receipt["generation"], Mapping),
             "Acceptance receipt generation is malformed")
    _require(isinstance(receipt["claims"], Mapping),
             "Acceptance receipt claims are malformed")
    _require(receipt["claims"].get("exact_key_value_is_not_an_upper_bound")
             is True,
             "Acceptance receipt blends or mislabels exact-key diagnostics")
    if config is not None:
        config = validate_config(config, require_inventory=False)
        _require(config["candidate_variant"] == receipt["candidate_variant"],
                 "Receipt candidate variant does not match config")
        _require(config["oracle_variant"] == receipt["oracle_variant"],
                 "Receipt oracle variant does not match config")
    if require_current:
        _require(receipt_digest(receipt) not in set(superseded),
                 "Acceptance receipt is superseded or historical")
    return receipt


def load_manifest(path):
    """Load the v3 manifest that names the current receipt."""
    manifest = json.loads(Path(path).read_text())
    _require(isinstance(manifest, Mapping), "Receipt manifest must be an object")
    _require(manifest.get("format") == MANIFEST_FORMAT,
             "Unsupported receipt manifest format; v1/v2 are not v3")
    _require(manifest.get("metric_version") == VERSION,
             "Unsupported receipt manifest metric version")
    authoritative = manifest.get("authoritative")
    _require(isinstance(authoritative, Mapping),
             "Receipt manifest is missing the authoritative receipt")
    _require(authoritative.get("status") == CURRENT_STATUS,
             "Receipt manifest has no current authoritative receipt")
    _require(authoritative.get("receipt_format") == RECEIPT_FORMAT
             and authoritative.get("metric_version") == VERSION,
             "Authoritative receipt is not v3")
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
    """Load and validate the sole current v3 receipt named by a v3 manifest."""
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
        raise ValueError("Current receipt selection requires a v3 manifest")
    superseded = ()
    if manifest_path is not None:
        superseded = _non_current_digests(load_manifest(manifest_path))
    receipt = json.loads(Path(path).read_text())
    return validate_receipt(receipt, config=config, superseded=superseded,
                            require_current=require_current)


def build_manifest(*, receipt_path, receipt, historical=(), superseded=(),
                   updated=None):
    """Build a v3 manifest entry list with explicit non-current history."""
    receipt_path = Path(receipt_path)
    digest = receipt_digest(receipt)
    entries = [{
        "path": receipt_path.name,
        "status": CURRENT_STATUS,
        "role": "sole current authoritative v3 receipt",
        "receipt_format": receipt["format"],
        "metric_version": receipt["metric_version"],
        "file_sha256": _sha(receipt_path),
        "receipt_digest": digest,
        "config_sha256": receipt.get("config", {}).get("sha256"),
        "config_file_sha256": receipt.get("config", {}).get("file_sha256"),
        "source_sha256": receipt["source"]["sha256"],
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
            "not_v3": document.get("metric_version") != VERSION,
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
            "not_v3": document.get("metric_version") != VERSION,
        })
    return {
        "format": MANIFEST_FORMAT,
        "metric_version": VERSION,
        "updated": updated,
        "authoritative": dict(entries[0]),
        "receipts": entries,
    }
