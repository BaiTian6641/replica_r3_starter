"""Contracts for memory.acceptance.v3 scoring, inventory and replay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from replica_lab.memory_acceptance_v3 import (
    ALL_ARMS, CONFIG_FORMAT, HISTORICAL_STATUS, MANIFEST_FORMAT, VERSION,
    acceptance_metrics, build_manifest, evaluate_acceptance,
    load_current_receipt, load_manifest, receipt_digest, rescore_ranking_file,
    validate_config, validate_ranking_inventory, validate_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
REAL_RAW = ROOT / "runs/memory_causal_fresh_20261026/restart02/rankings.jsonl"
REAL_INVENTORY = (
    ROOT / "data/memory_causal_fresh_20261026/prepared04/inventory.json")
REAL_CONFIG = ROOT / "configs/memory_acceptance_v3.json"
VARIANTS = ("candidate", "oracle", "full")
LAYERS = (0, 1)
PHASES = ("first_process", "revocation")

CASES = [
    ["ep1", "dev", "initial", "first_process", "fact-correction",
     "paraphrase", "fact", "correction_target"],
    ["ep1", "dev", "corrected", "first_process", "fact-retained",
     "paraphrase", "fact", "retained"],
    ["ep1", "dev", "initial", "first_process", "unknown",
     "unknown", "unknown", "unknown"],
    ["ep1", "dev", "initial", "revocation", "revoked",
     "revoked", "revoked", "revoked"],
]


def _inventory(rows=None, *, units=None):
    rows = rows or CASES
    declared = len({(row[0], row[3], row[4]) for row in rows})
    arm_expanded = sum(
        (4 if row[6] == "fact" else 3) * len(VARIANTS) * len(LAYERS)
        for row in rows)
    return {
        "rows": rows,
        "units": units if units is not None else {
            "declared_query_rows": declared,
            "inventory_case_rows": len(rows),
            "arm_expanded_metric_rows": arm_expanded,
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
        "expected_inventory": {
            "rows": inventory["rows"],
            "units": inventory["units"],
            "variants": list(VARIANTS),
            "layers": list(LAYERS),
            "arms": list(ALL_ARMS),
            "phases": list(PHASES),
        },
        "gates": {
            "accepted_correct_rate_min": 0.5,
            "accepted_precision_min": 0.5,
            "correction_accepted_correct_rate_min": 0.5,
            "unaffected_accepted_correct_rate_min": 0.5,
            "unknown_false_recall_max": 0.5,
            "revoked_false_recall_max": 1.0,
            "oracle_raw_top1_rate_min": 0.5,
        },
        "bootstrap": {"seed": 17, "resamples": 20, "confidence": 0.9},
    }


def _rows(inventory=None, *, variant_overrides=None, overrides=None):
    inventory = inventory or _inventory()
    rows = []
    for episode, split, state, phase, case, cohort, kind, role in \
            inventory["rows"]:
        for variant in VARIANTS:
            for layer in LAYERS:
                arms = ["query_key_value", "query_key_address",
                        "context_removed"]
                if kind == "fact":
                    arms.insert(1, "exact_key_value")
                for arm in arms:
                    row = {
                        "variant": variant, "arm": arm, "phase": phase,
                        "episode_id": episode, "world_id": f"world-{episode}",
                        "split": split, "state": state, "case_id": case,
                        "cohort": cohort, "kind": kind,
                        "target_role": role, "layer": layer,
                        "readout_norm": 1.0, "accepted": False,
                        "top1": False, "false_recall": False,
                    }
                    if kind == "fact":
                        row["accepted"] = arm == "query_key_value"
                        row["top1"] = arm == "query_key_value"
                        if variant != "candidate" and arm == "query_key_value":
                            row["accepted"] = True
                            row["top1"] = True
                    if variant == "candidate" and arm == "exact_key_value":
                        row["accepted"] = True
                        row["top1"] = False
                    if overrides:
                        row.update(overrides)
                    if variant_overrides:
                        row.update(variant_overrides.get(variant, {}))
                    rows.append(row)
    return rows


def test_primary_quality_is_isolated_from_exact_key_and_oracle_populations():
    config = _config()
    rows = _rows()
    metrics = acceptance_metrics(rows, config)
    candidate = metrics["variants"]["candidate"]
    assert candidate["quality"]["accepted_correct_rate"] == 1.0
    assert candidate["exact_key_value"]["accepted"] == 2
    assert candidate["exact_key_value"]["accepted_correct"] == 0
    assert candidate["exact_key_value"]["accepted_precision"] == 0.0
    assert metrics["populations"]["exact_key_value_diagnostic"][
        "is_upper_bound"] is False
    assert metrics["populations"]["candidate_quality"][
        "blended_with_diagnostics"] is False
    assert metrics["populations"]["oracle_diagnostic"][
        "is_legal_policy"] is False
    assert metrics["claims"]["exact_key_value_is_not_an_upper_bound"] is True
    verdict = evaluate_acceptance(metrics, config)
    assert verdict["decision"] == "accepted"
    assert verdict["candidate_quality_arm"] == "query_key_value"
    assert verdict["exact_key_value_blended"] is False


def test_inventory_rejects_missing_duplicate_malformed_and_extra_rows():
    config = _config()
    inventory = _inventory()
    rows = _rows()
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
    extra_row = dict(extra[0], case_id="extra-case")
    extra.append(extra_row)
    with pytest.raises(ValueError, match="not in the expected inventory"):
        validate_ranking_inventory(extra, config, inventory)
    wrong_units = _inventory(units={
        "declared_query_rows": 1,
        "inventory_case_rows": len(CASES),
        "arm_expanded_metric_rows": 1,
    })
    with pytest.raises(ValueError, match="does not reconcile"):
        validate_ranking_inventory(rows, _config(wrong_units), wrong_units)


def test_unknown_and_revoked_other_live_are_explicit_false_recalls():
    config = _config()
    rows = _rows(overrides={"accepted": True, "top1": False})
    for row in rows:
        if row["case_id"] in {"unknown", "revoked"}:
            row["false_recall_kind"] = (
                "unknown_address" if row["kind"] == "unknown"
                else "revoked_target_other_live")
    metrics = acceptance_metrics(rows, config)
    quality = metrics["variants"]["candidate"]["quality"]
    unknown = metrics["variants"]["candidate"]["unknown"]
    revoked = metrics["variants"]["candidate"]["revoked"]
    assert quality["false_recall_rate"] == 1.0
    assert unknown["unknown_false_recall_rate"] == 1.0
    assert revoked["revoked_false_recall_rate"] == 1.0
    assert revoked["revoked_target_other_live"] == 1
    # The evaluator label kind revoked did not force evaluator abstention.
    assert revoked["accepted"] == 1
    assert revoked["revoked_abstention_rate"] == 0.0


def test_policy_block_is_not_an_evaluator_forced_abstention():
    config = _config()
    rows = _rows()
    for row in rows:
        if row["case_id"] == "revoked":
            row["policy_blocked"] = "revoked"
            row["accepted"] = True
            row["top1"] = False
        if row["case_id"] == "fact-retained" and row["arm"] == "query_key_value":
            row["policy_blocked"] = "stale"
            row["accepted"] = True
            row["top1"] = True
    metrics = acceptance_metrics(rows, config)
    quality = metrics["variants"]["candidate"]["quality"]
    revoked = metrics["variants"]["candidate"]["revoked"]
    assert quality["policy_blocked_acceptance"] > 0
    assert revoked["policy_blocked"] == 1
    assert revoked["policy_blocked_acceptance"] == 1
    assert revoked["accepted"] == 0
    assert revoked["revoked_abstention_rate"] == 1.0
    assert revoked["revoked_target_other_live"] == 0


def test_missing_coverage_fails_closed_not_inconclusive():
    inventory = _inventory(
        rows=[row for row in CASES if row[7] != "correction_target"])
    config = _config(inventory)
    rows = _rows(inventory)
    metrics = acceptance_metrics(rows, config)
    with pytest.raises(ValueError, match="coverage is missing"):
        evaluate_acceptance(metrics, config)


def test_clustered_uncertainty_uses_estimable_worlds_per_metric():
    inventory = _inventory(rows=CASES)
    config = _config(inventory)
    rows = _rows()
    for row in rows:
        if row["case_id"] == "fact-correction":
            row["world_id"] = "world-other"
    metrics = acceptance_metrics(rows, config)
    uncertainty = metrics["variants"]["candidate"]["uncertainty"]
    assert uncertainty["unit"] == "world"
    assert uncertainty["accepted_correct_rate"]["clusters"] == 2
    assert uncertainty["layer_readouts_are_not_independent"] is True


def test_units_reconcile_declared_cases_and_arm_expanded_rows():
    config = _config()
    inventory = _inventory()
    rows = _rows()
    validation = validate_ranking_inventory(rows, config, inventory)
    assert validation["units"] == {
        "declared_query_rows": 4,
        "inventory_case_rows": 4,
        "arm_expanded_metric_rows": len(rows),
        "ranking_rows": len(rows),
    }
    assert len(rows) == (2 * 4 + 2 * 3) * len(VARIANTS) * len(LAYERS)
    per_layer = _inventory(units={
        "declared_query_rows": 4,
        "inventory_case_rows": 4,
        "per_layer_arm_expanded_metric_rows": len(rows) // len(LAYERS),
    })
    validate_ranking_inventory(rows, _config(per_layer), per_layer)


def test_v3_receipt_manifest_and_v2_rejection(tmp_path):
    config = _config()
    source = tmp_path / "rankings.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in _rows()))
    receipt = rescore_ranking_file(source, config)
    assert receipt["format"] == "replica.memory_acceptance.metrics.v3"
    assert receipt["primary_quality_arm"] == "query_key_value"
    assert receipt["inventory"]["validated"] is True
    assert receipt["verdict"]["decision"] == "accepted"
    assert receipt["source"]["original_bytes_unchanged"] is True
    validate_receipt(receipt, config=config)
    with pytest.raises(ValueError, match="v1/v2 are not v3"):
        validate_receipt(dict(receipt,
                              format="replica.memory_acceptance.metrics.v2"))
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    historical = tmp_path / "v2.json"
    historical.write_text(json.dumps({
        "format": "replica.memory_acceptance.metrics.v2",
        "metric_version": "memory.acceptance.v2",
    }))
    manifest = build_manifest(
        receipt_path=receipt_path, receipt=receipt,
        historical=[{"path": str(historical), "reason": "v2 history"}],
        updated="2026-10-26")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    loaded = load_manifest(manifest_path)
    assert loaded["format"] == MANIFEST_FORMAT
    assert loaded["receipts"][1]["status"] == HISTORICAL_STATUS
    assert loaded["receipts"][1]["not_v3"] is True
    current = load_current_receipt(manifest_path, config=config)
    assert receipt_digest(current) == manifest["authoritative"]["receipt_digest"]
    assert current["metric_version"] == VERSION


def test_replay_existing_raw_artifact_to_fresh_v3_receipt():
    if not REAL_RAW.is_file() or not REAL_INVENTORY.is_file():
        pytest.skip("native raw artifact is absent")
    config = validate_config(json.loads(REAL_CONFIG.read_text()))
    receipt = rescore_ranking_file(REAL_RAW, config)
    validation = receipt["inventory"]["units"]
    assert validation == {
        "declared_query_rows": 228,
        "inventory_case_rows": 456,
        "arm_expanded_metric_rows": 17520,
        "ranking_rows": 17520,
    }
    candidate = receipt["metrics"]["variants"][
        receipt["candidate_variant"]]
    assert candidate["quality"]["n"] == 456
    assert candidate["quality"]["fact_n"] == 384
    assert candidate["quality"]["unknown_n"] == 48
    assert candidate["quality"]["revoked_n"] == 24
    assert candidate["quality"]["accepted_correct_rate"] == pytest.approx(
        21 / 384)
    assert candidate["exact_key_value"]["n"] > 0
    assert receipt["metrics"]["populations"][
        "exact_key_value_diagnostic"]["is_upper_bound"] is False
    assert receipt["verdict"]["decision"] == "rejected"
    assert receipt["source"]["original_bytes_unchanged"] is True
    assert receipt["source"]["sha256"] == hashlib.sha256(
        REAL_RAW.read_bytes()).hexdigest()
    validate_receipt(receipt, config=config)
