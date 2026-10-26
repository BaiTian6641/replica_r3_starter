"""Contracts for memory.acceptance.v4 oracle, credit and phase-scope semantics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from replica_lab.memory_acceptance_v4 import (
    ALL_ARMS, CONFIG_FORMAT, HISTORICAL_STATUS, LAYER_COMBINATION,
    MANIFEST_FORMAT, ORACLE_POPULATION, VERSION, acceptance_metrics,
    build_manifest, evaluate_acceptance, load_current_receipt, load_manifest,
    receipt_digest, rescore_ranking_file, validate_config, validate_phase_scope,
    validate_ranking_inventory, validate_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG = ROOT / "configs/memory_acceptance_v4.json"
REAL_CONFIG_V3 = ROOT / "configs/memory_acceptance_v3.json"
REAL_INVENTORY = (
    ROOT / "data/memory_causal_fresh_20261026/prepared03/inventory.json")
REAL_INVENTORY_PREPARED04 = (
    ROOT / "data/memory_causal_fresh_20261026/prepared04/inventory.json")
REAL_RESTART = ROOT / "runs/memory_causal_fresh_20261026/restart02/rankings.jsonl"
REAL_NATIVE02 = ROOT / "runs/memory_causal_fresh_20261026/native02/rankings.jsonl"
REAL_V3_RECEIPT = (
    ROOT / "reports/memory_acceptance_20261026/"
           "causal_fresh_restart02_acceptance_v3.json")
INVENTORY_SHA256 = (
    "5994de0da6e7f863f4a697a40a25041e760d0d9e8e5c77f565713e1bdf763a93")

VARIANTS = ("candidate", "oracle", "full")
LAYERS = (13, 27)
PHASES = ("first_process", "restart", "revocation")
REAL_GATES = {
    "accepted_correct_rate_min": 0.85,
    "accepted_precision_min": 0.9,
    "correction_accepted_correct_rate_min": 0.6,
    "unaffected_accepted_correct_rate_min": 0.7,
    "unknown_false_recall_max": 0.1,
    "revoked_false_recall_max": 0.25,
    "oracle_raw_top1_rate_min": 0.85,
}
FACT_PLAN = (
    ("first_process", "correction_target", 3),
    ("first_process", "retained", 6),
    ("first_process", "source", 2),
    ("first_process", "distractor", 1),
    ("restart", "correction_target", 2),
    ("restart", "retained", 4),
    ("restart", "source", 1),
    ("restart", "distractor", 1),
)
UNKNOWN_CASES = 5
REVOKED_CASES = 5
WRONG_CASES = {"first_process-source-0", "restart-source-0"}
SPLITS = ("train", "dev", "heldout")


def _cases():
    rows = []
    counter = 0
    role_offset = {"correction_target": 0, "retained": 1, "source": 2,
                   "distractor": 3, "unknown": 4, "revoked": 4}
    for phase, role, count in FACT_PLAN:
        cohort = "distractor" if role == "distractor" else "paraphrase"
        for index in range(count):
            episode = f"ep{(index + role_offset[role]) % 5}"
            rows.append([episode, SPLITS[counter % len(SPLITS)], "initial",
                         phase, f"{phase}-{role}-{index}", cohort, "fact", role])
            counter += 1
    for index in range(UNKNOWN_CASES):
        episode = f"ep{index % 5}"
        rows.append([episode, SPLITS[counter % len(SPLITS)], "initial",
                     "first_process", f"first_process-unknown-{index}",
                     "unknown", "unknown", "unknown"])
        counter += 1
    for index in range(REVOKED_CASES):
        episode = f"ep{index % 5}"
        rows.append([episode, SPLITS[counter % len(SPLITS)], "initial",
                     "revocation", f"revocation-revoked-{index}",
                     "revoked", "revoked", "revoked"])
        counter += 1
    return rows


CASES = _cases()


def _digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


INVENTORY_KEYS = ("episode_id", "split", "state", "phase", "case_id",
                  "cohort", "kind", "target_role")


def _parent_digest(rows):
    normalized = [dict(zip(INVENTORY_KEYS, row)) if isinstance(row, list)
                  else dict(row) for row in rows]
    return _digest(normalized)


def _arm_expanded(rows):
    return sum((4 if row[6] == "fact" else 3) * len(VARIANTS) * len(LAYERS)
               for row in rows)


def _inventory(rows=None, *, units=None):
    rows = rows if rows is not None else CASES
    return {
        "rows": rows,
        "units": units if units is not None else {
            "declared_query_rows": len({(row[0], row[3], row[4])
                                        for row in rows}),
            "inventory_case_rows": len(rows),
            "arm_expanded_metric_rows": _arm_expanded(rows),
        },
    }


def _config(inventory=None):
    inventory = inventory or _inventory()
    return {
        "format": CONFIG_FORMAT,
        "metric_version": VERSION,
        "primary_quality_arm": "query_key_value",
        "diagnostic_arms": ["exact_key_value", "query_key_address",
                            "context_removed"],
        "candidate_variant": "candidate",
        "oracle_variant": "oracle",
        "oracle_population": dict(ORACLE_POPULATION),
        "layer_combination": dict(LAYER_COMBINATION),
        "phase_scoping": {
            "default": "all",
            "subset_recomputes_units": True,
            "parent_inventory_pin_preserved": True,
            "revocation_required_only_when_selected": True,
        },
        "expected_inventory": {
            "rows": inventory["rows"],
            "units": inventory["units"],
            "variants": list(VARIANTS),
            "layers": list(LAYERS),
            "arms": list(ALL_ARMS),
            "phases": list(PHASES),
        },
        "gates": dict(REAL_GATES),
        "bootstrap": {"seed": 17, "resamples": 50, "confidence": 0.9},
    }


def _rows(inventory=None, *, mutate=None):
    inventory = inventory or _inventory()
    rows = []
    for episode, split, state, phase, case, cohort, kind, role in \
            inventory["rows"]:
        for variant in VARIANTS:
            for layer in LAYERS:
                arms = (list(ALL_ARMS) if kind == "fact"
                        else ["query_key_value", "query_key_address",
                              "context_removed"])
                for arm in arms:
                    row = {
                        "variant": variant, "arm": arm, "phase": phase,
                        "episode_id": episode, "world_id": f"world-{episode}",
                        "split": split, "state": state, "case_id": case,
                        "cohort": cohort, "kind": kind,
                        "target_role": role, "layer": layer,
                        "readout_norm": 1.0, "top_cosine": 1.0,
                        "second_cosine": 0.5, "accepted": False,
                        "top1": False,
                    }
                    if mutate is not None:
                        mutate(row)
                    rows.append(row)
    return rows


def _positive_rows(inventory=None):
    rows = _rows(inventory)
    for row in rows:
        if row["arm"] == "exact_key_value":
            # Diagnostic only: accepted on a gold key is never candidate quality.
            row["accepted"] = True
            row["top1"] = False
            continue
        if row["arm"] != "query_key_value":
            continue
        if row["kind"] != "fact":
            continue
        if row["variant"] == "oracle":
            row["top1"] = True
            row["accepted"] = True
        elif row["variant"] == "candidate":
            row["accepted"] = True
            if row["case_id"] in WRONG_CASES:
                row["top1"] = False
                row["cross_world_selected"] = True
            else:
                row["top1"] = True
    return rows


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _subset_rows(rows, phase):
    return [row for row in rows if row["phase"] == phase]


def test_perfect_positive_uses_answerable_fact_oracle_not_total_group():
    config = _config()
    rows = _positive_rows()
    metrics = acceptance_metrics(rows, config)
    candidate = metrics["variants"]["candidate"]["quality"]
    assert candidate["fact_n"] == 20
    assert candidate["accepted_correct_fact"] == 18
    assert candidate["accepted"] == 20
    assert candidate["accepted_correct_rate"] == pytest.approx(18 / 20)
    assert candidate["accepted_precision"] == pytest.approx(18 / 20)
    assert metrics["variants"]["candidate"]["exact_key_value"]["accepted"] == 20
    assert metrics["variants"]["candidate"][
        "exact_key_value"]["accepted_correct"] == 0
    oracle = metrics["variants"]["oracle"]["quality"]
    assert oracle["fact_n"] == 20
    assert oracle["fact_recovered"] == 20
    assert oracle["recoverability_rate"] == 1.0
    assert oracle["raw_top1_rate"] == pytest.approx(20 / 30)
    population = metrics["oracle_population"]
    assert population["gate_population"] == "answerable_fact"
    assert population["denominator"] == "fact_n"
    assert population["excluded_non_answerable_n"] == 10
    assert population["answerable_fact_recoverability_rate"] == 1.0
    assert population["total_group_raw_top1_rate"] == pytest.approx(20 / 30)
    assert population["is_legal_policy"] is False

    verdict = evaluate_acceptance(metrics, config)
    assert verdict["decision"] == "accepted"
    assert verdict["checks"]["oracle_answerable_fact_recoverability"] is True
    assert verdict["checks"]["accepted_correct_rate"] is True
    assert verdict["checks"]["accepted_precision"] is True
    assert verdict["checks"]["zero_or_tied_credit_excluded"] is True
    # The total-group rate is below the threshold; only the fact denominator
    # may satisfy the oracle gate.
    assert verdict["oracle_total_group_raw_top1_rate"] < \
        REAL_GATES["oracle_raw_top1_rate_min"]
    assert verdict["oracle_answerable_fact_recoverability"] == 1.0
    assert verdict["oracle_gate_population"] == "answerable_fact"
    assert verdict["oracle_is_legal_policy"] is False
    assert verdict["oracle_fact_n"] == 20
    assert verdict["oracle_non_answerable_excluded_n"] == 10
    assert metrics["layer_combination"] == LAYER_COMBINATION
    assert metrics["populations"]["exact_key_value_diagnostic"][
        "is_upper_bound"] is False
    assert metrics["populations"]["candidate_quality"][
        "blended_with_diagnostics"] is False
    assert metrics["claims"]["zero_and_tied_readouts_receive_no_credit"] is True


def test_zero_and_tied_readouts_get_no_recoverability_or_acceptance_credit():
    zero_cases = {"first_process-correction_target-0",
                  "first_process-retained-0"}
    tie_cases = {"first_process-retained-1", "restart-retained-0"}

    def mutate(row):
        if row["arm"] != "query_key_value":
            return
        if row["case_id"] in zero_cases:
            row["top1"] = True
            row["accepted"] = True
            row["readout_norm"] = 0.0
        elif row["case_id"] in tie_cases:
            row["top1"] = True
            row["accepted"] = True
            row["top_cosine"] = 1.0
            row["second_cosine"] = 1.0

    config = _config()
    rows = _positive_rows()
    for row in rows:
        mutate(row)
    metrics = acceptance_metrics(rows, config)
    candidate = metrics["variants"]["candidate"]["quality"]
    oracle = metrics["variants"]["oracle"]["quality"]
    assert candidate["accepted_correct_fact"] == 14
    assert candidate["raw_top1"] == 14
    assert candidate["observed_top1"] == 18
    assert candidate["zero_or_tied"] == 4
    assert candidate["zero_or_tied_observed_top1"] == 4
    assert candidate["zero_or_tied_recoverability_credit"] == 0
    assert candidate["zero_or_tied_acceptance_credit"] == 0
    assert candidate["accepted"] == 16
    assert oracle["raw_top1"] == 16
    assert oracle["observed_top1"] == 20
    assert oracle["fact_recovered"] == 16
    assert oracle["recoverability_rate"] == pytest.approx(16 / 20)
    assert oracle["zero_or_tied_recoverability_credit"] == 0
    verdict = evaluate_acceptance(metrics, config)
    assert verdict["decision"] == "rejected"
    assert verdict["checks"]["accepted_correct_rate"] is False
    # The oracle rejects zero/tied readouts although their observed top-1 flag
    # was recorded.
    assert verdict["checks"]["oracle_answerable_fact_recoverability"] is False
    assert verdict["zero_or_tied_credit_excluded"] is True


def test_layer_combination_is_explicit_and_mixed_layers_are_not_full_credit():
    mixed_case = "first_process-correction_target-0"
    config = _config()
    rows = _positive_rows()
    for row in rows:
        if (row["case_id"] == mixed_case
                and row["variant"] == "candidate"
                and row["arm"] == "query_key_value"
                and row["layer"] == 27):
            row["accepted"] = False
            row["top1"] = False
    metrics = acceptance_metrics(rows, config)
    quality = metrics["variants"]["candidate"]["quality"]
    assert quality["raw_top1"] == 18       # any valid layer
    assert quality["raw_top1_all"] == 17   # all layers
    assert quality["accepted"] == 20       # any layer
    assert quality["accepted_correct_fact"] == 17  # all layers accepted+correct
    assert quality["accepted_wrong"] == 3  # accepted but not all correct
    assert quality["zero_or_tied"] == 0
    assert metrics["layer_combination"]["raw_top1"] == "any_layer_valid_top1"
    assert metrics["layer_combination"]["raw_top1_all"] == \
        "all_layers_valid_top1"
    assert metrics["layer_combination"]["accepted_correct"] == \
        "all_layers_accepted_and_correct"
    assert metrics["layer_combination"]["accepted_wrong"] == \
        "accepted_and_not_all_correct"
    assert metrics["layer_combination"]["zero_or_tied"] == \
        "all_layers_zero_or_tied"
    assert metrics["layer_combination"]["layers_are_independent_samples"] is False
    assert metrics["variants"]["candidate"]["uncertainty"][
        "layer_readouts_are_not_independent"] is True


def test_inventory_rejects_missing_duplicate_extra_malformed_and_bad_units():
    config = _config()
    inventory = _inventory()
    rows = _positive_rows()
    validate_ranking_inventory(rows, config, inventory)
    with pytest.raises(ValueError, match="missing"):
        validate_ranking_inventory(rows[1:], config, inventory)
    with pytest.raises(ValueError, match="Duplicate ranking identity"):
        validate_ranking_inventory(rows + [rows[0]], config, inventory)
    malformed = [dict(row) for row in rows]
    malformed[0]["accepted"] = "yes"
    with pytest.raises(ValueError, match="malformed acceptance"):
        validate_ranking_inventory(malformed, config, inventory)
    extra = [dict(row) for row in rows]
    extra.append(dict(extra[0], case_id="extra-case"))
    with pytest.raises(ValueError, match="not in the expected inventory"):
        validate_ranking_inventory(extra, config, inventory)
    unknown_phase = [dict(row) for row in rows]
    unknown_phase.append(dict(unknown_phase[0], phase="sideways"))
    with pytest.raises(ValueError, match="undeclared phase"):
        validate_ranking_inventory(unknown_phase, config, inventory)
    wrong_units = _inventory(units={
        "declared_query_rows": 1,
        "inventory_case_rows": len(CASES),
        "arm_expanded_metric_rows": 1,
    })
    with pytest.raises(ValueError, match="does not reconcile"):
        validate_ranking_inventory(rows, _config(wrong_units), wrong_units)


def test_phase_scope_recomputes_units_and_keeps_parent_pin():
    config = _config()
    inventory = _inventory()
    rows = _positive_rows()
    parent_hash = _parent_digest(inventory["rows"])
    validation = validate_ranking_inventory(
        rows, config, inventory, phases=("first_process",))
    assert validation["scope"]["full"] is False
    assert validation["scope"]["phase_scoped"] is True
    assert validation["scope"]["phases"] == ["first_process"]
    assert validation["scope"]["subset_units_recomputed"] is True
    assert validation["scope"]["parent_inventory_pin_verified"] is True
    assert validation["scope"]["parent_inventory_sha256"] == parent_hash
    assert validation["scope"]["revocation_in_scope"] is False
    assert validation["scope"]["out_of_scope_rows"] == len(rows) - len(
        _subset_rows(rows, "first_process"))
    assert validation["units"] == {
        "declared_query_rows": 17,
        "inventory_case_rows": 17,
        "arm_expanded_metric_rows": 378,
        "ranking_rows": 378,
    }
    assert validation["parent_units"]["arm_expanded_metric_rows"] == 660
    assert validation["parent_units"]["ranking_rows"] == 660
    with pytest.raises(ValueError, match="duplicate phases"):
        validate_phase_scope(config, ("first_process", "first_process"))
    with pytest.raises(ValueError, match="undeclared phase"):
        validate_phase_scope(config, ("sideways",))
    with pytest.raises(ValueError, match="Phase scope is empty"):
        validate_phase_scope(config, ())

    scoped = _subset_rows(rows, "first_process")
    missing = scoped[1:]
    out_of_scope = [row for row in rows if row["phase"] != "first_process"]
    with pytest.raises(ValueError, match="missing"):
        validate_ranking_inventory(
            missing + out_of_scope, config, inventory,
            phases=("first_process",))
    extra = [dict(row) for row in scoped]
    extra.append(dict(extra[0], case_id="extra-scoped"))
    with pytest.raises(ValueError, match="not in the expected inventory"):
        validate_ranking_inventory(
            extra + out_of_scope, config, inventory,
            phases=("first_process",))
    validate_ranking_inventory(
        scoped + out_of_scope, config, inventory, phases=("first_process",))
    bogus_out_of_scope = out_of_scope + [
        dict(out_of_scope[0], case_id="extra-out-of-scope")]
    with pytest.raises(ValueError, match="not in the expected inventory"):
        validate_ranking_inventory(
            scoped + bogus_out_of_scope, config, inventory,
            phases=("first_process",))
    with pytest.raises(ValueError, match="Duplicate ranking identity"):
        validate_ranking_inventory(
            scoped + out_of_scope + [out_of_scope[0]], config, inventory,
            phases=("first_process",))


def test_phase_scoped_receipt_and_full_receipt(tmp_path):
    config = _config()
    inventory = _inventory()
    rows = _positive_rows()
    source = _write_rows(tmp_path / "rankings.jsonl", rows)
    parent_hash = _parent_digest(inventory["rows"])

    scoped = rescore_ranking_file(source, config, phases=("first_process",))
    assert scoped["format"] == "replica.memory_acceptance.metrics.v4"
    assert scoped["inventory"]["sha256"] == parent_hash
    assert scoped["inventory"]["units"] == {
        "declared_query_rows": 17,
        "inventory_case_rows": 17,
        "arm_expanded_metric_rows": 378,
        "ranking_rows": 378,
    }
    assert scoped["inventory"]["parent_units"]["arm_expanded_metric_rows"] == 660
    assert scoped["source"]["rows"] == len(rows)
    assert scoped["source"]["scored_rows"] == 378
    assert scoped["source"]["out_of_scope_rows"] == len(rows) - 378
    assert scoped["source"]["phases"] == ["first_process"]
    assert scoped["source"]["phase_scoped"] is True
    assert scoped["inventory"]["scope"]["subset_validated"] is True
    assert scoped["verdict"]["require_revoked"] is False
    assert "revoked" not in scoped["verdict"]["coverage"]
    assert scoped["verdict"]["decision"] == "accepted"
    assert scoped["verdict"]["oracle_answerable_fact_recoverability"] == 1.0
    assert scoped["verdict"]["oracle_fact_n"] == 12
    assert scoped["metrics"]["variants"]["candidate"]["quality"][
        "revoked_n"] == 0
    validate_receipt(scoped, config=config)

    incomplete = _subset_rows(rows, "first_process")[1:]
    with pytest.raises(ValueError, match="missing"):
        rescore_ranking_file(
            _write_rows(tmp_path / "missing.jsonl", incomplete),
            config, phases=("first_process",))

    with pytest.raises(ValueError, match="coverage is missing"):
        rescore_ranking_file(
            _write_rows(tmp_path / "revocation.jsonl",
                        _subset_rows(rows, "revocation")),
            config, phases=("revocation",))

    full = rescore_ranking_file(source, config)
    assert full["inventory"]["units"] == {
        "declared_query_rows": 30,
        "inventory_case_rows": 30,
        "arm_expanded_metric_rows": 660,
        "ranking_rows": 660,
    }
    assert full["inventory"]["scope"]["full"] is True
    assert full["source"]["phase_scoped"] is False
    assert full["source"]["out_of_scope_rows"] == 0
    assert full["verdict"]["require_revoked"] is True
    assert full["verdict"]["decision"] == "accepted"
    assert full["verdict"]["oracle_fact_n"] == 20
    validate_receipt(full, config=config)


def test_unknown_and_revoked_have_separate_abstention_and_false_recall():
    config = _config()
    rows = _positive_rows()
    for row in rows:
        if (row["variant"] == "candidate"
                and row["arm"] == "query_key_value"
                and row["kind"] in {"unknown", "revoked"}):
            row["accepted"] = True
            row["top1"] = False
    metrics = acceptance_metrics(rows, config)
    candidate = metrics["variants"]["candidate"]
    quality = candidate["quality"]
    unknown = candidate["unknown"]
    revoked = candidate["revoked"]
    assert quality["fact_n"] == 20
    assert quality["unknown_false_recall_rate"] == 1.0
    assert quality["revoked_false_recall_rate"] == 1.0
    assert unknown["unknown_abstention_rate"] == 0.0
    assert revoked["revoked_abstention_rate"] == 0.0
    assert metrics["oracle_population"]["excluded_non_answerable_n"] == 10
    assert metrics["variants"]["oracle"]["quality"][
        "recoverability_rate"] == 1.0
    verdict = evaluate_acceptance(metrics, config)
    assert verdict["decision"] == "rejected"
    assert verdict["checks"]["unknown_false_recall"] is False
    assert verdict["checks"]["revoked_false_recall"] is False
    without_revocation = evaluate_acceptance(
        metrics, config, require_revoked=False)
    assert "revoked_false_recall" not in without_revocation["checks"]


def test_policy_blocking_is_separate_from_raw_correctness_and_acceptance():
    blocked_fact = "first_process-retained-0"
    blocked_unknown = "first_process-unknown-0"
    config = _config()
    rows = _positive_rows()
    for row in rows:
        if (row["variant"] != "candidate"
                or row["arm"] != "query_key_value"):
            continue
        if row["case_id"] == blocked_fact:
            row["policy_blocked"] = "stale"
            row["accepted"] = True
            row["top1"] = True
        elif row["case_id"] == blocked_unknown:
            row["policy_blocked"] = "revoked"
            row["accepted"] = True
            row["top1"] = False
    metrics = acceptance_metrics(rows, config)
    quality = metrics["variants"]["candidate"]["quality"]
    # The blocked fact keeps raw correctness but loses acceptance.
    assert quality["raw_top1"] == 18
    assert quality["policy_blocked"] == 2
    assert quality["policy_blocked_acceptance"] == 2
    assert quality["policy_blocked_raw_top1"] == 1
    assert quality["policy_blocked_observed_top1"] == 1
    assert quality["accepted"] == 19
    assert quality["accepted_correct_fact"] == 17
    assert metrics["variants"]["candidate"]["unknown"][
        "unknown_false_recall_rate"] == 0.0
    assert metrics["variants"]["candidate"]["unknown"][
        "unknown_abstention_rate"] == 1.0
    assert metrics["variants"]["candidate"]["unknown"][
        "policy_blocked_acceptance"] == 1
    verdict = evaluate_acceptance(metrics, config)
    assert verdict["decision"] == "rejected"
    assert verdict["checks"]["accepted_precision"] is False
    assert verdict["checks"]["unknown_false_recall"] is True
    assert verdict["checks"]["oracle_answerable_fact_recoverability"] is True


def test_wrong_world_acceptance_is_wrong_retrieval_not_credit():
    config = _config()
    rows = _positive_rows()
    metrics = acceptance_metrics(rows, config)
    quality = metrics["variants"]["candidate"]["quality"]
    assert quality["accepted_correct_fact"] == 18
    assert quality["accepted_wrong"] == 2
    assert quality["cross_world_false_recall"] == 2
    assert quality["false_recall_kinds"]["cross_world"] == 2
    assert quality["unknown_false_recall_rate"] == 0.0
    assert quality["revoked_false_recall_rate"] == 0.0
    assert quality["accepted_correct_rate"] == pytest.approx(18 / 20)
    assert quality["raw_top1"] == 18
    assert metrics["variants"]["candidate"]["unknown"]["accepted"] == 0


def test_world_clustered_uncertainty_never_treats_layers_as_samples():
    config = _config()
    rows = _positive_rows()
    metrics = acceptance_metrics(rows, config)
    uncertainty = metrics["variants"]["candidate"]["uncertainty"]
    assert uncertainty["unit"] == "world"
    assert uncertainty["worlds"] == 5
    assert uncertainty["accepted_correct_rate"]["clusters"] == 5
    assert uncertainty["recoverability_rate"]["clusters"] == 5
    assert uncertainty["layer_readouts_are_not_independent"] is True
    for row in rows:
        if (row["variant"] == "candidate"
                and row["arm"] == "query_key_value"
                and row["case_id"] == "first_process-retained-2"):
            row["world_id"] = "world-extra"
    moved = acceptance_metrics(rows, config)
    shifted = moved["variants"]["candidate"]["uncertainty"]
    assert shifted["accepted_correct_rate"]["clusters"] == 6
    assert shifted["layer_readouts_are_not_independent"] is True


def test_receipt_manifest_and_v3_rejection(tmp_path):
    config = _config()
    source = _write_rows(tmp_path / "rankings.jsonl", _positive_rows())
    receipt = rescore_ranking_file(source, config)
    assert receipt["format"] == "replica.memory_acceptance.metrics.v4"
    assert receipt["layer_combination"] == LAYER_COMBINATION
    assert receipt["oracle_population"]["gate_population"] == "answerable_fact"
    assert receipt["claims"]["phase_scope_preserves_parent_inventory_pin"] is True
    assert receipt["source"]["original_bytes_unchanged"] is True
    validate_receipt(receipt, config=config)
    with pytest.raises(ValueError, match="not v4"):
        validate_receipt(dict(
            receipt, format="replica.memory_acceptance.metrics.v3"))
    with pytest.raises(ValueError, match="not v4"):
        validate_receipt(dict(receipt, metric_version="memory.acceptance.v3"))
    weakened = json.loads(json.dumps(receipt))
    weakened["inventory"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="weakened the parent inventory hash"):
        validate_receipt(weakened)

    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    historical = tmp_path / "v3.json"
    historical.write_text(json.dumps({
        "format": "replica.memory_acceptance.metrics.v3",
        "metric_version": "memory.acceptance.v3",
    }) + "\n")
    manifest = build_manifest(
        receipt_path=receipt_path, receipt=receipt,
        historical=[{"path": str(historical), "reason": "v3 history"}],
        updated="2026-10-26")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    loaded = load_manifest(manifest_path)
    assert loaded["format"] == MANIFEST_FORMAT
    assert loaded["receipts"][1]["status"] == HISTORICAL_STATUS
    assert loaded["receipts"][1]["not_v4"] is True
    current = load_current_receipt(manifest_path, config=config)
    assert receipt_digest(current) == manifest["authoritative"]["receipt_digest"]
    assert current["metric_version"] == VERSION
    with pytest.raises(ValueError, match="not v4"):
        load_manifest(_write_json(tmp_path / "manifest_v3.json", {
            "format": "replica.memory_acceptance.receipt_manifest.v3",
            "metric_version": "memory.acceptance.v3",
        }))


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def test_config_pins_v3_thresholds_answerable_fact_and_prepared03_hash():
    config = validate_config(json.loads(REAL_CONFIG.read_text()))
    v3_config = json.loads(REAL_CONFIG_V3.read_text())
    assert config["format"] == "replica.memory_acceptance.config.v4"
    assert config["metric_version"] == VERSION
    assert config["gates"] == v3_config["gates"]
    assert config["gates"]["oracle_raw_top1_rate_min"] == 0.85
    assert config["layer_combination"] == LAYER_COMBINATION
    assert config["oracle_population"] == ORACLE_POPULATION
    assert config["expected_inventory"]["path"] == (
        "data/memory_causal_fresh_20261026/prepared03/inventory.json")
    assert config["expected_inventory"]["sha256"] == INVENTORY_SHA256
    assert hashlib.sha256(REAL_INVENTORY.read_bytes()).hexdigest() == \
        INVENTORY_SHA256
    assert hashlib.sha256(REAL_INVENTORY_PREPARED04.read_bytes()).hexdigest() \
        == INVENTORY_SHA256
    with pytest.raises(ValueError, match="not v4"):
        validate_config(v3_config)


def test_real_v3_receipt_is_never_accepted_as_v4():
    receipt = json.loads(REAL_V3_RECEIPT.read_text())
    assert receipt["metric_version"] == "memory.acceptance.v3"
    with pytest.raises(ValueError, match="not v4"):
        validate_receipt(receipt)


def test_replay_restart02_full_v4():
    if not REAL_RESTART.is_file() or not REAL_INVENTORY.is_file():
        pytest.skip("native restart02 raw ranking artifact is absent")
    config = validate_config(json.loads(REAL_CONFIG.read_text()))
    receipt = rescore_ranking_file(REAL_RESTART, config)
    assert receipt["inventory"]["sha256"] == INVENTORY_SHA256
    assert receipt["inventory"]["units"] == {
        "declared_query_rows": 228,
        "inventory_case_rows": 456,
        "arm_expanded_metric_rows": 17520,
        "ranking_rows": 17520,
    }
    assert receipt["inventory"]["parent_units"] == {
        "declared_query_rows": 228,
        "inventory_case_rows": 456,
        "arm_expanded_metric_rows": 17520,
        "ranking_rows": 17520,
    }
    assert receipt["inventory"]["scope"]["full"] is True
    assert receipt["source"]["rows"] == 17520
    assert receipt["source"]["scored_rows"] == 17520
    assert receipt["source"]["out_of_scope_rows"] == 0
    assert receipt["source"]["sha256"] == hashlib.sha256(
        REAL_RESTART.read_bytes()).hexdigest()
    candidate = receipt["metrics"]["variants"][
        receipt["candidate_variant"]]["quality"]
    assert candidate["fact_n"] == 384
    assert candidate["unknown_n"] == 48
    assert candidate["revoked_n"] == 24
    assert candidate["accepted_correct_fact"] == 21
    assert candidate["accepted_correct_rate"] == pytest.approx(21 / 384)
    assert candidate["accepted"] == 138
    assert candidate["accepted_precision"] == pytest.approx(21 / 138)
    assert candidate["recoverability_rate"] == pytest.approx(176 / 384)
    assert candidate["zero_or_tied_acceptance_credit"] == 0
    oracle = receipt["metrics"]["variants"][
        receipt["oracle_variant"]]["quality"]
    assert oracle["fact_n"] == 384
    assert oracle["fact_recovered"] == 384
    assert oracle["recoverability_rate"] == 1.0
    assert oracle["raw_top1_rate"] == pytest.approx(384 / 456)
    assert oracle["observed_top1_rate"] == pytest.approx(384 / 456)
    verdict = receipt["verdict"]
    assert verdict["decision"] == "rejected"
    assert verdict["checks"]["accepted_correct_rate"] is False
    assert verdict["checks"]["accepted_precision"] is False
    assert verdict["checks"]["oracle_answerable_fact_recoverability"] is True
    assert verdict["oracle_answerable_fact_recoverability"] == 1.0
    assert verdict["oracle_total_group_raw_top1_rate"] == pytest.approx(
        384 / 456)
    assert verdict["oracle_total_group_raw_top1_rate"] < 0.85
    assert verdict["oracle_non_answerable_excluded_n"] == 72
    assert verdict["zero_or_tied_credit_excluded"] is True
    validate_receipt(receipt, config=config)


def test_replay_native02_first_process_is_phase_scoped():
    if not REAL_NATIVE02.is_file() or not REAL_INVENTORY.is_file():
        pytest.skip("native native02 raw ranking artifact is absent")
    config = validate_config(json.loads(REAL_CONFIG.read_text()))
    receipt = rescore_ranking_file(
        REAL_NATIVE02, config, phases=("first_process",))
    assert receipt["inventory"]["sha256"] == INVENTORY_SHA256
    assert receipt["inventory"]["units"] == {
        "declared_query_rows": 108,
        "inventory_case_rows": 216,
        "arm_expanded_metric_rows": 8400,
        "ranking_rows": 8400,
    }
    assert receipt["inventory"]["parent_units"] == {
        "declared_query_rows": 228,
        "inventory_case_rows": 456,
        "arm_expanded_metric_rows": 17520,
        "ranking_rows": 17520,
    }
    scope = receipt["inventory"]["scope"]
    assert scope["full"] is False
    assert scope["phase_scoped"] is True
    assert scope["phases"] == ["first_process"]
    assert scope["subset_units_recomputed"] is True
    assert scope["parent_inventory_pin_verified"] is True
    assert scope["revocation_in_scope"] is False
    assert receipt["source"]["rows"] == 8400
    assert receipt["source"]["scored_rows"] == 8400
    assert receipt["source"]["out_of_scope_rows"] == 0
    assert receipt["source"]["phases"] == ["first_process"]
    candidate = receipt["metrics"]["variants"][
        receipt["candidate_variant"]]["quality"]
    assert candidate["fact_n"] == 192
    assert candidate["unknown_n"] == 24
    assert candidate["revoked_n"] == 0
    assert candidate["accepted_correct_fact"] == 12
    assert candidate["accepted_correct_rate"] == pytest.approx(12 / 192)
    assert candidate["accepted"] == 66
    assert candidate["accepted_precision"] == pytest.approx(12 / 66)
    oracle = receipt["metrics"]["variants"][
        receipt["oracle_variant"]]["quality"]
    assert oracle["fact_n"] == 192
    assert oracle["fact_recovered"] == 192
    assert oracle["recoverability_rate"] == 1.0
    assert oracle["raw_top1_rate"] == pytest.approx(192 / 216)
    verdict = receipt["verdict"]
    assert verdict["require_revoked"] is False
    assert "revoked" not in verdict["coverage"]
    assert verdict["decision"] == "rejected"
    assert verdict["checks"]["oracle_answerable_fact_recoverability"] is True
    assert verdict["oracle_answerable_fact_recoverability"] == 1.0
    validate_receipt(receipt, config=config)
