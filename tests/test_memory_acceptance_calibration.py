"""CPU-only contracts for the memory acceptance/address calibration protocol.

These tests build synthetic inventories and ranking rows, then replay the
predeclared calibration CLI without any native job, model load or held-out
scoring.  A final integration test is optional and skips with an explicit
provisioning reason when the immutable artifacts are absent.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_memory_acceptance_calibration.py"
SPEC = importlib.util.spec_from_file_location(
    "memory_acceptance_calibration", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

VARIANTS = (
    "shared_mean/candidate",
    "shared_mean/oracle",
    "shared_mean/full",
    "shared_mean/shuffled_values",
    "shared_mean/disabled",
)
LAYERS = (13, 27)
PHASES = ("first_process", "restart", "revocation")
SPLITS = ("train", "dev", "heldout")
CANDIDATE = "shared_mean/candidate"
ORACLE = "shared_mean/oracle"
BASELINES = [
    "shared_mean/full",
    "shared_mean/shuffled_values",
    "shared_mean/disabled",
]
WRONG_ADDRESS_CATEGORIES = [
    "other_source",
    "independent_world",
    "unknown_address",
    "revoked_target_other_live",
]
GATES = {
    "accepted_correct_rate_min": 0.85,
    "accepted_precision_min": 0.9,
    "correction_accepted_correct_rate_min": 0.6,
    "unaffected_accepted_correct_rate_min": 0.7,
    "unknown_false_recall_max": 0.1,
    "revoked_false_recall_max": 0.25,
    "oracle_raw_top1_rate_min": 0.85,
}
CORRECT_CASES = {"c1", "c5", "c6", "c9"}

# (episode, split, phase, case_id, cohort, kind, target_role)
CASES = [
    ("ep1", "train", "first_process", "c1", "paraphrase",
     "fact", "correction_target"),
    ("ep1", "dev", "first_process", "c2", "paraphrase", "fact", "retained"),
    ("ep2", "heldout", "first_process", "c3", "paraphrase", "fact", "source"),
    ("ep2", "train", "first_process", "c4", "distractor",
     "fact", "distractor"),
    ("ep3", "train", "first_process", "u1", "unknown", "unknown", "unknown"),
    ("ep3", "dev", "first_process", "u2", "unknown", "unknown", "unknown"),
    ("ep4", "dev", "restart", "c5", "paraphrase", "fact",
     "correction_target"),
    ("ep4", "train", "restart", "c6", "paraphrase", "fact", "retained"),
    ("ep5", "heldout", "restart", "c7", "paraphrase", "fact", "source"),
    ("ep5", "dev", "restart", "c8", "distractor", "fact", "distractor"),
    ("ep6", "train", "revocation", "c9", "paraphrase", "fact", "retained"),
    ("ep6", "dev", "revocation", "c10", "paraphrase", "fact", "retained"),
    ("ep7", "train", "revocation", "r1", "revoked", "revoked", "revoked"),
    ("ep7", "dev", "revocation", "r2", "revoked", "revoked", "revoked"),
]


def _inventory_rows():
    return [[episode, split, "initial", phase, case, cohort, kind, role]
            for episode, split, phase, case, cohort, kind, role in CASES]


def _arm_expanded(rows):
    return sum((4 if row[6] == "fact" else 3) * len(VARIANTS) * len(LAYERS)
               for row in rows)


def _inventory():
    rows = _inventory_rows()
    return {
        "format": "replica.memory_causal_fresh.inventory.v2",
        "rows": rows,
        "units": {
            "arm_case_counts": {"fact": 4, "revoked": 3, "unknown": 3},
            "declared_query_rows": len({(row[0], row[3], row[4])
                                        for row in rows}),
            "inventory_case_rows": len(rows),
            "arm_expanded_metric_rows": _arm_expanded(rows),
            "per_layer_arm_expanded_metric_rows": _arm_expanded(rows) // 2,
            "variants": len(VARIANTS),
            "layers": len(LAYERS),
        },
    }


def _default_row(row):
    case_id = row["case_id"]
    kind = row["kind"]
    arm = row["arm"]
    variant = row["variant"]
    target = f"target-{case_id}"
    row["target_entry_id"] = target
    row["target_rank"] = 1
    row["runtime_accepted"] = False
    row["runtime_selected_record_id"] = None
    row["cross_world_selected"] = False
    if kind != "fact":
        row["top_cosine"] = 0.1
        row["second_cosine"] = 0.05
        row["top1"] = False
        row["top_entry_id"] = f"other-{case_id}"
        return
    if arm == "context_removed":
        row["readout_norm"] = 0.0
        row["top_cosine"] = 0.0
        row["second_cosine"] = 0.0
        row["top1"] = True
        row["top_entry_id"] = target
        return
    if arm == "query_key_address":
        row["top_cosine"] = 0.9
        row["second_cosine"] = 0.5
        row["top1"] = True
        row["top_entry_id"] = target
        return
    if arm == "exact_key_value":
        row["top_cosine"] = 0.9
        row["second_cosine"] = 0.5
        row["top1"] = False
        row["top_entry_id"] = f"other-{case_id}"
        return
    # query_key_value
    if variant == "shared_mean/disabled":
        row["readout_norm"] = 0.0
        row["top_cosine"] = 0.0
        row["second_cosine"] = 0.0
        row["top1"] = True
        row["top_entry_id"] = target
        return
    row["top_cosine"] = 0.9
    row["second_cosine"] = 0.5
    if variant == "shared_mean/oracle":
        correct = True
    elif variant in ("shared_mean/full", "shared_mean/shuffled_values"):
        correct = False
    else:
        correct = case_id in CORRECT_CASES
    row["top1"] = correct
    row["top_entry_id"] = target if correct else f"other-{case_id}"
    if not correct and case_id == "c8":
        row["cross_world_selected"] = True


def _rows(inventory, *, mutate=None):
    rows = []
    for episode, split, state, phase, case, cohort, kind, role in \
            inventory["rows"]:
        for variant in VARIANTS:
            for layer in LAYERS:
                arms = (list(module.ALL_ARMS) if kind == "fact"
                        else ["query_key_value", "query_key_address",
                              "context_removed"])
                for arm in arms:
                    row = {
                        "variant": variant,
                        "arm": arm,
                        "phase": phase,
                        "episode_id": episode,
                        "world_id": f"world-{episode}",
                        "split": split,
                        "state": state,
                        "case_id": case,
                        "cohort": cohort,
                        "kind": kind,
                        "target_role": role,
                        "layer": layer,
                        "readout_norm": 1.0,
                        "top_cosine": 0.9,
                        "second_cosine": 0.5,
                        "accepted": False,
                        "top1": False,
                        "policy_blocked": None,
                        "stale_blocked": False,
                        "revoked_blocked": False,
                        "readout_accepted": False,
                    }
                    _default_row(row)
                    if mutate is not None:
                        mutate(row)
                    rows.append(row)
    return rows


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _config(tmp_path, *, frontier_cosine=None, frontier_margin=None):
    inventory = _inventory()
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
    rows = _rows(inventory)
    rankings_path = tmp_path / "rankings.jsonl"
    rankings_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    config = {
        "format": module.CONFIG_FORMAT,
        "metric_version": module.METRIC_VERSION,
        "declared_at_utc": "2026-10-26T00:00:00+00:00",
        "purpose": "synthetic development-only calibration",
        "source": {
            "rankings": "rankings.jsonl",
            "rankings_sha256": _sha(rankings_path),
            "inventory": "inventory.json",
            "inventory_sha256": _sha(inventory_path),
            "expected_rows": len(rows),
        },
        "scope": {
            "splits": ["train", "dev"],
            "phases": list(PHASES),
            "variants": list(VARIANTS),
            "arms": list(module.ALL_ARMS),
            "heldout_rows_scored": False,
            "heldout_metrics_reported": False,
            "prepared04_heldout_reused": False,
            "layers": list(LAYERS),
        },
        "native_thresholds": {
            "value": {"cosine_floor": 0.5, "margin_floor": 0.05},
            "address": {"cosine_floor": 0.5, "margin_floor": 0.05},
        },
        "frontier": {
            "value": {
                "cosine_floors": frontier_cosine or [0.0, 0.25, 0.5, 0.75, 0.9],
                "margin_floors": frontier_margin or [0.0, 0.01, 0.05, 0.1, 0.2],
                "max_points": 25,
                "applies_to_arms": list(module.VALUE_ARMS),
                "recomputes_existing_readout_acceptance_only": True,
            },
            "address": {
                "cosine_floors": [0.0, 0.25, 0.5, 0.75, 0.9],
                "margin_floors": [0.0, 0.01, 0.05, 0.1, 0.2],
                "max_points": 25,
                "applies_to_arms": [module.ADDRESS_ARM],
                "address_arm_only": True,
                "downstream_refresh_recomputed": False,
                "candidate_policy_reselected": False,
            },
        },
        "layer_semantics": dict(module.LAYER_SEMANTICS),
        "controls": {
            "candidate_variant": CANDIDATE,
            "oracle_variant": ORACLE,
            "baselines": list(BASELINES),
            "exact_key_value_is_mixture_diagnostic_not_upper_bound": True,
            "wrong_address_categories": list(WRONG_ADDRESS_CATEGORIES),
            "unknown_and_revoked_separate": True,
            "policy_blocking_preserved": True,
            "no_relabeling": True,
            "no_gold_inputs_to_threshold_rule": True,
        },
        "gates": dict(GATES),
        "decision_rules": {
            "oracle_cannot_meet_gates":
                "reject_acceptance_calibration_as_boundary",
            "oracle_passes_candidate_fails":
                "selection_or_refresh_boundary_on_existing_native_path",
            "candidate_passes_development":
                "new_sealed_fixture_required_before_any_native_confirmation",
            "prepared04_heldout":
                "spent_and_not_used_for_calibration_or_confirmation",
        },
        "limits": {
            "max_source_rows": 10000,
            "max_frontier_points_per_arm": 25,
            "max_output_bytes": 64 * 1024 * 1024,
            "native_execution": False,
            "model_loads": 0,
            "tokenizer_loads": 0,
            "gradients": 0,
            "optimizer_updates": 0,
            "checkpoint_writes": 0,
            "new_generation": False,
        },
        "output": {
            "receipt": "receipt.json",
            "raw_rankings_modified": False,
            "raw_labels_modified": False,
            "receipt_is_derived_not_authority": True,
            "heldout_opened": False,
        },
        "non_claims": [
            "No native, held-out or learned-memory claim.",
            "Address results are address-arm diagnostics only.",
        ],
    }
    return config, rows, inventory, inventory_path, rankings_path


def _write_fixture(tmp_path, *, mutate=None, config_mutate=None,
                   rows_override=None, frontier_cosine=None,
                   frontier_margin=None):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    config, rows, inventory, inventory_path, rankings_path = _config(
        tmp_path, frontier_cosine=frontier_cosine,
        frontier_margin=frontier_margin)
    if rows_override is not None:
        rows = rows_override(rows)
        rankings_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows))
        config["source"]["rankings_sha256"] = _sha(rankings_path)
        config["source"]["expected_rows"] = len(rows)
    if mutate is not None:
        for row in rows:
            mutate(row)
        rankings_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows))
        config["source"]["rankings_sha256"] = _sha(rankings_path)
    if config_mutate is not None:
        config = config_mutate(config)
    return config, inventory, inventory_path, rankings_path


def _canonical(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def test_runtime_decision_uses_only_readout_fields():
    floors = {"cosine_floor": 0.5, "margin_floor": 0.05}
    row = {
        "readout_norm": 1.0,
        "top_cosine": 0.9,
        "second_cosine": 0.5,
        "policy_blocked": None,
        "top1": False,
    }
    assert module._runtime_decision(row, floors) is True
    # Gold/evaluator fields never change the runtime decision.
    for change in (
            {"top1": True},
            {"top1": True, "target_entry_id": "x", "false_recall_kind":
             "other_source"},
            {"top1": False, "target_entry_id": "x",
             "cross_world_selected": True},
            {"accepted": True}):
        assert module._runtime_decision(dict(row, **change), floors) is True

    assert module._runtime_decision(dict(row, top_cosine=0.4), floors) is False
    assert module._runtime_decision(
        dict(row, top_cosine=0.9, second_cosine=0.86), floors) is False
    assert module._runtime_decision(dict(row, readout_norm=0.0), floors) is False
    assert module._runtime_decision(
        dict(row, top_cosine=0.9, second_cosine=0.9), floors) is False
    assert module._runtime_decision(
        dict(row, policy_blocked="stale"), floors) is False


def test_acceptance_is_invariant_to_gold_labels(tmp_path):
    base_config, _, _, _ = _write_fixture(tmp_path)
    base = module.analyze_calibration(base_config, root=tmp_path)
    base_qkv = base["native"]["arms"]["query_key_value"]["variants"][
        CANDIDATE]["query"]

    def relabel(row):
        if row["arm"] == "query_key_value":
            row["top1"] = True
            row["target_entry_id"] = "relabelled-target"
            row["cross_world_selected"] = True
            row["false_recall_kind"] = "independent_world"
            row["accepted"] = not row["accepted"]

    relabelled_config, _, _, _ = _write_fixture(
        tmp_path / "relabelled", mutate=relabel)
    relabelled = module.analyze_calibration(relabelled_config,
                                            root=tmp_path / "relabelled")
    relabelled_qkv = relabelled["native"]["arms"]["query_key_value"][
        "variants"][CANDIDATE]["query"]
    assert relabelled_qkv["accepted"] == base_qkv["accepted"]
    assert relabelled_qkv["acceptance_coverage"] == base_qkv[
        "acceptance_coverage"]
    assert relabelled_qkv["accepted_correct_rate"] != base_qkv[
        "accepted_correct_rate"]
    assert relabelled["gate_classification"]["native_point"]["checks"][
        "zero_or_tied_credit_excluded"] is True


def test_zero_tie_and_policy_blocking_receive_no_credit(tmp_path):
    def mutate(row):
        if row["arm"] != "query_key_value" or row["variant"] != CANDIDATE:
            return
        if row["case_id"] == "c1":
            row["readout_norm"] = 0.0
            row["top1"] = True
            row["accepted"] = True
        elif row["case_id"] == "c2":
            row["top_cosine"] = 0.9
            row["second_cosine"] = 0.9
            row["top1"] = True
            row["accepted"] = True
        elif row["case_id"] == "c5":
            row["policy_blocked"] = "stale"
            row["top1"] = True
            row["accepted"] = True

    config, _, _, _ = _write_fixture(tmp_path, mutate=mutate)
    receipt = module.analyze_calibration(config, root=tmp_path)
    qkv = receipt["native"]["arms"]["query_key_value"]["variants"][
        CANDIDATE]["query"]
    summary = receipt["native"]["arm_summary_candidate_variant"][
        "query_key_value"]
    assert qkv["zero_or_tied"] >= 2
    assert qkv["zero_or_tied_observed_top1"] >= 2
    assert qkv["zero_or_tied_recoverability_credit"] == 0
    assert qkv["zero_or_tied_acceptance_credit"] == 0
    assert summary["zero_or_tied"]["query_observed_top1_excluded"] >= 2
    assert summary["zero_or_tied"]["query_recoverability_credit"] == 0
    assert summary["zero_or_tied"]["query_acceptance_credit"] == 0
    policy = receipt["native"]["policy_blocking"][
        "by_arm_candidate_variant"]["query_key_value"]
    assert policy["query_policy_blocked"] >= 1
    assert policy["query_policy_blocked_would_accept"] >= 1
    assert policy["blocked_rows_never_accepted"] is True


def test_arm_separation_and_exact_key_value_isolation(tmp_path):
    config, _, _, _ = _write_fixture(tmp_path)
    receipt = module.analyze_calibration(config, root=tmp_path)
    separation = receipt["native"]["arm_separation"]
    assert separation["arms_are_measurably_distinct"] is True
    assert separation["exact_key_value_blended_into_candidate_quality"] is False
    assert separation["address_arm_is_not_value_quality"] is True
    rates = {arm: value["accepted_correct_rate"]
             for arm, value in separation["by_arm"].items()}
    assert rates["query_key_value"] != rates["exact_key_value"]
    assert rates["context_removed"] == 0.0
    assert rates["query_key_address"] > 0.0
    controls = receipt["native"]["control_comparison"]
    assert controls["exact_key_value_diagnostic"]["arm"] == "exact_key_value"
    assert controls["exact_key_value_diagnostic"][
        "blended_into_candidate_quality"] is False
    assert controls["exact_key_value_diagnostic"]["is_upper_bound"] is False
    native_qkv = receipt["native"]["arms"]["query_key_value"]["variants"][
        CANDIDATE]["query"]
    assert controls["candidate"]["compact"] == {
        key: native_qkv[key] for key in module.COMPACT_KEYS}
    assert controls["candidate_equals_full"] is False
    assert controls["candidate_equals_disabled"] is False
    assert set(controls["baselines"]) == {"full", "shuffled", "disabled"}
    # The address arm is reported with its own diagnostics, not value gates.
    address = receipt["native"]["arms"]["query_key_address"]["variants"][
        CANDIDATE]
    assert address["address_diagnostics"]["address_accepted"] > 0
    assert receipt["gate_classification"]["address_arm_value_gated"] is False


def test_frontiers_have_declared_points_and_address_flags(tmp_path):
    config, _, _, _ = _write_fixture(tmp_path)
    receipt = module.analyze_calibration(config, root=tmp_path)
    value = receipt["value_frontier"]
    address = receipt["address_frontier"]
    assert value["point_count"] == 25
    assert address["point_count"] == 25
    assert len(value["points"]) == 25
    assert len(address["points"]) == 25
    first_pairs = [(point["cosine_floor"], point["margin_floor"])
                   for point in value["points"][:5]]
    assert first_pairs == [(0.0, 0.0), (0.0, 0.01), (0.0, 0.05),
                           (0.0, 0.1), (0.0, 0.2)]
    for point in value["points"]:
        assert set(point["arms"]) == set(module.VALUE_ARMS)
        assert module.ADDRESS_ARM not in point["arms"]
        assert "candidate_checks" in point
        assert "oracle_check" in point
    for point in address["points"]:
        assert set(point["arms"]) == {module.ADDRESS_ARM}
        assert point["address_only"] is True
        assert point["downstream_refresh_recomputed"] is False
        assert point["candidate_policy_reselected"] is False
        assert "candidate_address_diagnostics" in point
    assert address["address_arm_only"] is True
    assert address["downstream_refresh_recomputed"] is False
    assert address["candidate_policy_reselected"] is False
    native = receipt["gate_classification"]
    assert native["native_point_in_value_grid"] is not None
    assert native["native_point_matches_value_grid"] is True
    assert native["disposition"]["stop"] is True
    assert native["disposition"]["native_execution_permitted"] is False
    assert native["disposition"]["selected_rule"] in {
        "oracle_cannot_meet_gates",
        "oracle_passes_candidate_fails",
        "candidate_passes_development",
    }


def test_receipt_is_deterministic_and_raw_sources_are_preserved(tmp_path):
    config, _, inventory_path, rankings_path = _write_fixture(tmp_path)
    before_rankings = rankings_path.read_bytes()
    before_inventory = inventory_path.read_bytes()
    first = module.analyze_calibration(config, root=tmp_path)
    second = module.analyze_calibration(config, root=tmp_path)
    assert first == second
    assert first["content_sha256"] == second["content_sha256"]
    body = {key: value for key, value in first.items()
            if key != "content_sha256"}
    assert _canonical(body) == first["content_sha256"]
    assert first["source"]["raw_rankings_modified"] is False
    assert first["source"]["original_bytes_unchanged"] is True
    assert first["source"]["scope"]["heldout_rows_scored"] == 0
    assert first["source"]["scope"]["heldout_rows_present"] > 0
    assert rankings_path.read_bytes() == before_rankings
    assert inventory_path.read_bytes() == before_inventory


def test_cli_writes_a_receipt_and_refuses_raw_output(tmp_path):
    config, _, inventory_path, rankings_path = _write_fixture(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    absolute = json.loads(config_path.read_text())
    absolute["source"]["rankings"] = str(rankings_path)
    absolute["source"]["inventory"] = str(inventory_path)
    config_path.write_text(json.dumps(absolute, indent=2) + "\n")
    output = tmp_path / "derived" / "receipt.json"
    assert module.main(["--config", str(config_path), "--output",
                        str(output), "--quiet"]) == 0
    receipt = json.loads(output.read_text())
    assert receipt["format"] == module.RECEIPT_FORMAT
    assert receipt["metric_version"] == module.METRIC_VERSION
    assert receipt["source"]["rankings"]["rows"] == len(_rows(_inventory()))
    assert receipt["source"]["scope"]["scored_rows"] > 0
    with pytest.raises(ValueError, match="raw rankings"):
        module.write_receipt(receipt, rankings_path)


def test_fail_closed_on_hash_row_and_scope_tampering(tmp_path):
    config, _, _, _ = _write_fixture(tmp_path / "base")
    base_root = tmp_path / "base"

    bad_hash = json.loads(json.dumps(config))
    bad_hash["source"]["rankings_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256 pin"):
        module.analyze_calibration(bad_hash, root=base_root)

    bad_rows = json.loads(json.dumps(config))
    bad_rows["source"]["expected_rows"] += 1
    with pytest.raises(ValueError, match="Expected .* rows"):
        module.analyze_calibration(bad_rows, root=base_root)

    bad_inventory = json.loads(json.dumps(config))
    bad_inventory["source"]["inventory_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="Inventory bytes"):
        module.analyze_calibration(bad_inventory, root=base_root)

    heldout_scope = json.loads(json.dumps(config))
    heldout_scope["scope"]["splits"] = ["train", "dev", "heldout"]
    with pytest.raises(ValueError, match="heldout"):
        module.analyze_calibration(heldout_scope, root=base_root)

    def duplicate(rows_value):
        return rows_value + [dict(rows_value[0])]

    dup_config, _, _, _ = _write_fixture(
        tmp_path / "duplicate", rows_override=duplicate)
    with pytest.raises(ValueError, match="Duplicate"):
        module.analyze_calibration(dup_config, root=tmp_path / "duplicate")

    def extra(rows_value):
        extra_row = dict(rows_value[0])
        extra_row["case_id"] = "extra-case"
        return rows_value + [extra_row]

    extra_config, _, _, _ = _write_fixture(
        tmp_path / "extra", rows_override=extra)
    with pytest.raises(ValueError, match="unexpected|not in the expected"):
        module.analyze_calibration(extra_config, root=tmp_path / "extra")

    def missing(rows_value):
        return rows_value[1:]

    missing_config, _, _, _ = _write_fixture(
        tmp_path / "missing", rows_override=missing)
    with pytest.raises(ValueError, match="missing"):
        module.analyze_calibration(missing_config, root=tmp_path / "missing")

    def tampered_split(rows_value):
        for row in rows_value:
            if row["split"] == "heldout":
                row["split"] = "train"
        return rows_value

    split_config, _, _, _ = _write_fixture(
        tmp_path / "split", rows_override=tampered_split)
    with pytest.raises(ValueError, match="disagrees on split|missing|unexpected"):
        module.analyze_calibration(split_config, root=tmp_path / "split")

    def malformed(rows_value):
        rows_value[0]["accepted"] = "yes"
        return rows_value

    malformed_config, _, _, _ = _write_fixture(
        tmp_path / "malformed", rows_override=malformed)
    with pytest.raises(ValueError, match="malformed acceptance"):
        module.analyze_calibration(malformed_config,
                                   root=tmp_path / "malformed")

    misaligned_config, _, _, _ = _write_fixture(
        tmp_path / "misaligned", frontier_cosine=[0.0, 0.5])
    with pytest.raises(ValueError, match="frontier points"):
        module.analyze_calibration(misaligned_config,
                                   root=tmp_path / "misaligned")


REAL_CONFIG = ROOT / "configs/memory_acceptance_calibration_v1.json"
REAL_RANKINGS = ROOT / "runs/memory_causal_fresh_20261026/restart02/rankings.jsonl"
REAL_INVENTORY = (
    ROOT / "data/memory_causal_fresh_20261026/prepared03/inventory.json")
REAL_CALIBRATION_RECEIPT = (
    ROOT / "reports/memory_acceptance_20261026/calibration_dev01.json")
REAL_MISSING = [str(path) for path in (REAL_RANKINGS, REAL_INVENTORY)
                if not path.exists()]
REAL_DIAGNOSIS_MISSING = (
    [] if REAL_CALIBRATION_RECEIPT.exists()
    else [str(REAL_CALIBRATION_RECEIPT)])


def _real_diagnosis(*, with_supporting=True):
    receipt = json.loads(REAL_CALIBRATION_RECEIPT.read_text())
    return module.analyze_memory_diagnosis(
        receipt, receipt_path=REAL_CALIBRATION_RECEIPT,
        supporting_receipts=(
            None if with_supporting else ()))


@pytest.mark.skipif(
    bool(REAL_DIAGNOSIS_MISSING),
    reason="missing committed calibration receipt")
def test_real_receipt_address_layer_precision_vs_strict_groups():
    diagnosis = _real_diagnosis()
    address = diagnosis["address"]
    layer = address["layer_correctness"]
    query = address["query_aggregation"]
    assert layer["fact_accepted_layers"] == 62
    assert layer["fact_accepted_correct_layers"] == 62
    assert layer["fact_wrong_layers"] == 0
    assert layer["fact_layer_precision"] == 1.0
    assert query["fact_accepted_groups"] == 58
    assert query["fact_strict_all_layers_groups"] == 4
    assert query["fact_partial_groups"] == 54
    assert query["fact_no_accepted_layer_groups"] == 198
    assert query["fact_accepted_any_layer_correct_groups"] == 58
    assert query["fact_any_layer_correct_groups"] == 256
    assert address["denominators"]["fact_query_groups"] == 256
    assert address["denominators"]["fact_layer_readouts"] == 512
    assert query["fact_strict_group_rate"] == 4 / 256
    assert query["fact_any_layer_acceptance_rate"] == 58 / 256
    reconciliation = address["historical_reconciliation"]
    assert reconciliation["query_accepted_including_non_fact"] == 58
    assert reconciliation["query_accepted_wrong"] == 54
    assert reconciliation["non_fact_accepted_groups"] == 0
    assert reconciliation["fact_accepted_groups"] == 58
    assert reconciliation["readout_accepted_including_non_fact"] == 62
    assert reconciliation["readout_accepted_wrong"] == 0
    assert reconciliation["fact_accepted_layers"] == 62
    assert address["comparison"] == {
        "accepted_layer_result": "62/62",
        "strict_group_result": "4/256",
        "strict_groups": 4,
        "partial_groups": 54,
        "no_accepted_layer_groups": 198,
        "wrong_address_layers": 0,
    }
    assert address["status"] == "not_repaired"
    decomposition = diagnosis["multi_layer_aggregation"]["address"]
    assert decomposition["identity_holds"] is True
    assert decomposition["accepted_identity_holds"] is True
    assert decomposition[
        "accepted_correct_layers_from_strict_groups"] == 8
    assert decomposition[
        "accepted_correct_layers_from_partial_groups"] == 54
    assert decomposition[
        "accepted_correct_layers_from_wrong_layer_groups"] == 0
    assert decomposition["fact_accepted_layers"] == 62
    assert diagnosis["metric_semantics"][
        "historical_metrics_relabelled"] is False
    assert diagnosis["metric_semantics"][
        "gate_semantics_preserved"] is True
    assert diagnosis["claims"]["address_readout_repaired"] is False
    assert diagnosis["abstention"]["address"]["unknown_accepted"] == 0
    assert diagnosis["abstention"]["address"]["revoked_accepted"] == 0
    by_layer = address["by_layer"]
    assert by_layer["layers"]["13"]["fact_accepted_layers"] == 4
    assert by_layer["layers"]["13"]["fact_accepted_correct_layers"] == 4
    assert by_layer["layers"]["27"]["fact_accepted_layers"] == 58
    assert by_layer["layers"]["27"]["fact_accepted_correct_layers"] == 58
    assert by_layer["layers"]["13"]["fact_wrong_layers"] == 0
    assert by_layer["layers"]["27"]["fact_wrong_layers"] == 0
    assert by_layer["fact_accepted_layers"] == 62
    assert by_layer["layer_sums_are_not_query_groups"] is True


@pytest.mark.skipif(
    bool(REAL_DIAGNOSIS_MISSING),
    reason="missing committed calibration receipt")
def test_real_receipt_value_and_oracle_partial_fact_groups():
    diagnosis = _real_diagnosis()
    value = diagnosis["value"]
    oracle = diagnosis["oracle"]
    assert value["denominators"]["fact_query_groups"] == 256
    assert value["query_aggregation"]["fact_accepted_groups"] == 98
    assert value["query_aggregation"]["fact_strict_all_layers_groups"] == 14
    assert value["query_aggregation"]["fact_partial_groups"] == 23
    assert value["query_aggregation"]["fact_wrong_layer_groups"] == 61
    assert value["layer_correctness"]["fact_accepted_layers"] == 156
    assert value["layer_correctness"]["fact_accepted_correct_layers"] == 57
    assert value["layer_correctness"]["fact_wrong_layers"] == 99
    assert value["layer_correctness"]["fact_layer_precision"] == 57 / 156
    assert value["abstention"]["unknown_n"] == 32
    assert value["abstention"]["unknown_accepted"] == 10
    assert value["abstention"]["revoked_n"] == 16
    assert value["abstention"]["revoked_accepted"] == 3
    reconciliation = value["historical_reconciliation"]
    assert reconciliation["query_accepted_including_non_fact"] == 111
    assert reconciliation["query_accepted_wrong"] == 97
    assert reconciliation["non_fact_accepted_groups"] == 13
    assert reconciliation["fact_accepted_groups"] == 98
    assert reconciliation["readout_accepted_including_non_fact"] == 179
    assert reconciliation["readout_accepted_wrong"] == 122
    assert reconciliation["non_fact_accepted_layers"] == 23
    assert reconciliation["fact_accepted_layers"] == 156
    assert oracle["query_aggregation"]["fact_accepted_groups"] == 150
    assert oracle["query_aggregation"][
        "fact_strict_all_layers_groups"] == 102
    assert oracle["query_aggregation"]["fact_partial_groups"] == 48
    assert oracle["query_aggregation"]["fact_wrong_layer_groups"] == 0
    assert oracle["layer_correctness"]["fact_accepted_layers"] == 252
    assert oracle["layer_correctness"][
        "fact_accepted_correct_layers"] == 252
    assert oracle["layer_correctness"]["fact_wrong_layers"] == 0
    assert oracle["layer_correctness"]["fact_layer_precision"] == 1.0
    oracle_reconciliation = oracle["historical_reconciliation"]
    assert oracle_reconciliation["query_accepted_including_non_fact"] == 163
    assert oracle_reconciliation["query_accepted_wrong"] == 61
    assert oracle_reconciliation["fact_accepted_groups"] == 150
    assert oracle_reconciliation["readout_accepted_including_non_fact"] == 275
    assert oracle_reconciliation["readout_accepted_wrong"] == 23
    assert oracle_reconciliation["fact_accepted_layers"] == 252
    assert oracle["is_legal_policy"] is False
    assert oracle["is_diagnostic_only"] is True
    oracle_decomposition = diagnosis["multi_layer_aggregation"]["oracle"]
    assert oracle_decomposition["identity_holds"] is True
    assert oracle_decomposition[
        "accepted_correct_layers_from_partial_groups"] == 48
    assert diagnosis["bottleneck"]["selected"] == (
        "value_readout_and_abstention")
    assert diagnosis["bottleneck"]["address_readout_repaired"] is False
    assert diagnosis["bottleneck"]["calibration_is_a_solution"] is False
    assert diagnosis["claims"][
        "value_readout_and_abstention_are_next_bottleneck"] is True
    assert diagnosis["claims"]["calibration_is_a_solution"] is False
    assert diagnosis["bottleneck"]["calibration_evidence"][
        "value_points_passing_all_gates"] == 0
    assert diagnosis["bottleneck"]["calibration_evidence"][
        "address_points_meeting_hit_and_precision"] == 0
    retention = diagnosis["bottleneck"]["retention_evidence"]
    assert retention["candidate_unaffected_fact_groups"] == 80
    assert retention["candidate_unaffected_accepted_groups"] == 33
    assert retention["candidate_unaffected_strict_groups"] == 1
    assert retention["oracle_unaffected_strict_groups"] == 18
    assert retention["unaffected_rows_current_teaching_gradient"] == 0
    value_layers = value["by_layer"]["layers"]
    assert value_layers["13"]["fact_accepted_layers"] == 67
    assert value_layers["13"]["fact_accepted_correct_layers"] == 16
    assert value_layers["13"]["fact_wrong_layers"] == 51
    assert value_layers["27"]["fact_accepted_layers"] == 89
    assert value_layers["27"]["fact_accepted_correct_layers"] == 41
    assert value_layers["27"]["fact_wrong_layers"] == 48
    oracle_layers = oracle["by_layer"]["layers"]
    assert oracle_layers["13"]["fact_accepted_layers"] == 125
    assert oracle_layers["13"]["fact_accepted_correct_layers"] == 125
    assert oracle_layers["27"]["fact_accepted_layers"] == 127
    assert oracle_layers["27"]["fact_accepted_correct_layers"] == 127


@pytest.mark.skipif(
    bool(REAL_DIAGNOSIS_MISSING),
    reason="missing committed calibration receipt")
def test_real_receipt_controls_are_separated():
    diagnosis = _real_diagnosis()
    controls = diagnosis["controls"]
    candidate = controls["candidate"]["metrics"]
    disabled = controls["disabled_reset"]["metrics"]
    shuffled = controls["shuffled_value"]["metrics"]
    exact = controls["exact_context"]["metrics"]
    assert candidate["query_aggregation"][
        "fact_strict_all_layers_groups"] == 14
    assert disabled["query_aggregation"]["fact_accepted_groups"] == 0
    assert disabled["query_aggregation"][
        "fact_strict_all_layers_groups"] == 0
    assert disabled["query_aggregation"]["fact_wrong_layer_groups"] == 0
    assert shuffled["query_aggregation"][
        "fact_strict_all_layers_groups"] == 12
    assert exact["query_aggregation"]["fact_strict_all_layers_groups"] == 15
    assert controls["exact_context"]["abstention_available"] is False
    assert controls["exact_context"]["is_upper_bound"] is False
    assert controls["exact_context"][
        "blended_into_candidate_quality"] is False
    removed = controls["context_removed"]["metrics"]
    assert removed["query_aggregation"]["fact_accepted_groups"] == 0
    assert removed["query_aggregation"][
        "fact_strict_all_layers_groups"] == 0
    assert removed["query_aggregation"]["fact_wrong_layer_groups"] == 0
    assert removed["layer_correctness"]["fact_accepted_layers"] == 0
    selected = controls["selected_record"]
    assert selected["candidate"]["runtime_accepted"] == 58
    assert selected["candidate"]["runtime_correct"] == 4
    assert selected["oracle"]["runtime_accepted"] == 256
    assert selected["oracle"]["runtime_correct"] == 256
    assert selected["address_candidate"]["runtime_accepted"] == 58
    assert selected["address_candidate"]["runtime_correct"] == 4
    wrong_world = controls["wrong_world"]
    assert wrong_world["candidate"]["query_independent_world_groups"] == 27
    assert wrong_world["candidate"]["layer_independent_world_groups"] == 49
    assert wrong_world["address_candidate"][
        "layer_cross_world_false_recall"] == 0
    separation = controls["separation"]
    assert separation["controls_are_distinct"] is True
    assert separation["candidate_equals_disabled"] is False
    assert separation["candidate_equals_shuffled"] is False
    assert separation[
        "address_arm_invariant_across_value_variants"] is True
    assert set(separation["fact_strict_group_rate"]) == {
        "candidate", "oracle", "full", "shuffled_value",
        "disabled_reset", "exact_context"}


@pytest.mark.skipif(
    bool(REAL_DIAGNOSIS_MISSING),
    reason="missing committed calibration receipt")
def test_real_diagnosis_declares_gradient_scope_and_source_hashes():
    first = _real_diagnosis()
    second = _real_diagnosis()
    assert first == second
    body = {key: item for key, item in first.items()
            if key != "content_sha256"}
    assert _canonical(body) == first["content_sha256"]
    assert first["format"] == module.DIAGNOSIS_FORMAT
    assert first["metric_version"] == module.DIAGNOSIS_METRIC_VERSION
    primary = first["source_receipts"]["primary"]
    receipt = json.loads(REAL_CALIBRATION_RECEIPT.read_text())
    assert primary["sha256"] == _sha(REAL_CALIBRATION_RECEIPT)
    assert primary["content_sha256"] == receipt["content_sha256"]
    assert primary["content_sha256_verified"] is True
    supporting = first["source_receipts"]["supporting"]
    assert len(supporting) == 2
    assert all(item["available"] for item in supporting)
    assert all(item["sha256"] for item in supporting)
    assert first["source_receipts"]["supporting_metrics_used"] is False
    assert all(item["metrics_used"] is False for item in supporting)
    assert all("oracle_fact_diagnostics" not in item
               for item in supporting)
    current = first["gradient_scope"]["current_path"]
    assert current["steps_per_event"] == 1
    assert current["gradient_parameters"] == [
        "expert_<selected>.gate",
        "expert_<selected>.up",
        "expert_<selected>.down",
    ]
    assert current["gradient_parameter_count_per_event"] == 3
    assert current["selected_expert_only"] is True
    for name in ("decoder_receives_gradient",
                 "read_projections_receive_gradient",
                 "writers_receive_gradient",
                 "backbone_receives_gradient",
                 "personality_receives_gradient"):
        assert current[name] is False
    assert current["unselected_experts_receive_gradient"] is False
    assert current["unaffected_rows_receive_teaching_gradient"] is False
    assert current["unaffected_rows_current_teaching_gradient"] == 0
    next_scope = first["gradient_scope"]["next_hypothesis_scope"]
    assert next_scope["new_training_arm"] is False
    assert next_scope["one_step_per_event_contract_unchanged"] is True
    assert next_scope["native_execution_permitted_by_this_receipt"] is False
    assert first["claims"]["heldout_rows_scored"] is False
    assert first["claims"]["heldout_metric_blocks_parsed"] is False
    assert first["claims"]["supporting_receipt_metrics_used"] is False
    assert first["claims"]["native_execution_performed"] is False
    assert first["claims"]["gradients"] == 0
    assert first["claims"]["optimizer_updates"] == 0
    assert first["claims"]["new_training_arm"] is False


@pytest.mark.skipif(
    bool(REAL_DIAGNOSIS_MISSING),
    reason="missing committed calibration receipt")
def test_diagnosis_rejects_tampered_and_undercovered_receipts():
    receipt = json.loads(REAL_CALIBRATION_RECEIPT.read_text())

    tampered = json.loads(json.dumps(receipt))
    native_address_index = next(
        index for index, point in enumerate(
            tampered["address_frontier"]["points"])
        if point["cosine_floor"] == 0.5
        and point["margin_floor"] == 0.05)
    tampered["address_frontier"]["points"][native_address_index][
        "candidate_address_diagnostics"]["address_accepted_hit"] = 3
    with pytest.raises(ValueError, match="content_sha256"):
        module.analyze_memory_diagnosis(tampered, supporting_receipts=())
    body = {key: item for key, item in tampered.items()
            if key != "content_sha256"}
    tampered["content_sha256"] = _canonical(body)
    with pytest.raises(ValueError, match="strict groups disagree"):
        module.analyze_memory_diagnosis(tampered, supporting_receipts=())

    undercovered = json.loads(json.dumps(receipt))
    candidate = undercovered["native"]["control_comparison"]["candidate"][
        "variant"]
    undercovered["native"]["arms"]["query_key_value"]["variants"][
        candidate]["query"]["fact_n"] = 0
    body = {key: item for key, item in undercovered.items()
            if key != "content_sha256"}
    undercovered["content_sha256"] = _canonical(body)
    with pytest.raises(ValueError, match="no fact query groups"):
        module.analyze_memory_diagnosis(
            undercovered, supporting_receipts=())

    heldout = json.loads(json.dumps(receipt))
    heldout["source"]["scope"]["heldout_rows_scored"] = 1
    body = {key: item for key, item in heldout.items()
            if key != "content_sha256"}
    heldout["content_sha256"] = _canonical(body)
    with pytest.raises(ValueError, match="heldout rows were scored"):
        module.analyze_memory_diagnosis(heldout, supporting_receipts=())

    missing = json.loads(json.dumps(receipt))
    del missing["native"]["arms"]["query_key_value"]["variants"][
        candidate]["readout"]
    body = {key: item for key, item in missing.items()
            if key != "content_sha256"}
    missing["content_sha256"] = _canonical(body)
    with pytest.raises(ValueError, match="readout"):
        module.analyze_memory_diagnosis(missing, supporting_receipts=())


def test_synthetic_mixed_layer_aggregation_is_exact(tmp_path):
    def mutate(row):
        if (row["arm"] == module.ADDRESS_ARM
                and row["variant"] == CANDIDATE
                and row["case_id"] == "c1"
                and row["layer"] == 27):
            row["top_cosine"] = 0.2
            row["second_cosine"] = 0.1

    config, _, _, _ = _write_fixture(tmp_path, mutate=mutate)
    receipt = module.analyze_calibration(config, root=tmp_path)
    diagnosis = module.analyze_memory_diagnosis(
        receipt, supporting_receipts=())
    address = diagnosis["address"]
    assert address["denominators"]["fact_query_groups"] == 8
    assert address["query_aggregation"]["fact_accepted_groups"] == 8
    assert address["query_aggregation"][
        "fact_strict_all_layers_groups"] == 7
    assert address["query_aggregation"]["fact_partial_groups"] == 1
    assert address["query_aggregation"]["fact_wrong_layer_groups"] == 0
    assert address["layer_correctness"]["fact_accepted_layers"] == 15
    assert address["layer_correctness"][
        "fact_accepted_correct_layers"] == 15
    assert address["layer_correctness"]["fact_wrong_layers"] == 0
    decomposition = diagnosis["multi_layer_aggregation"]["address"]
    assert decomposition["identity_holds"] is True
    assert decomposition["accepted_identity_holds"] is True
    assert decomposition[
        "accepted_correct_layers_from_strict_groups"] == 14
    assert decomposition[
        "accepted_correct_layers_from_partial_groups"] == 1
    value = diagnosis["value"]
    assert value["query_aggregation"]["fact_accepted_groups"] == 8
    assert value["query_aggregation"]["fact_strict_all_layers_groups"] == 4
    assert value["query_aggregation"]["fact_wrong_layer_groups"] == 4
    assert value["layer_correctness"]["fact_accepted_layers"] == 16
    assert value["layer_correctness"]["fact_accepted_correct_layers"] == 8
    assert value["layer_correctness"]["fact_wrong_layers"] == 8
    assert diagnosis["multi_layer_aggregation"][
        "value"]["identity_holds"] is True


def test_diagnosis_cli_writes_receipt_and_refuses_source(tmp_path):
    config, _, _, _ = _write_fixture(tmp_path)
    receipt = module.analyze_calibration(config, root=tmp_path)
    source = tmp_path / "calibration_receipt.json"
    source.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    output = tmp_path / "derived" / "diagnosis.json"
    assert module.main([
        "--diagnosis", str(source),
        "--diagnosis-output", str(output),
        "--quiet",
    ]) == 0
    written = json.loads(output.read_text())
    assert written["format"] == module.DIAGNOSIS_FORMAT
    assert written["metric_version"] == module.DIAGNOSIS_METRIC_VERSION
    expected = module.analyze_memory_diagnosis(
        receipt, receipt_path=source, root=ROOT)
    assert written["content_sha256"] == expected["content_sha256"]
    with pytest.raises(ValueError, match="source receipt"):
        module.write_diagnosis_receipt(written, source)


@pytest.mark.integration
@pytest.mark.skipif(
    bool(REAL_MISSING),
    reason=("missing local artifact(s) — provision the immutable restart02 "
            "rankings and prepared03 inventory; optional CPU replay only"))
def test_real_protocol_replay(tmp_path):
    config = json.loads(REAL_CONFIG.read_text())
    receipt = module.analyze_calibration(
        config, root=ROOT, config_path=REAL_CONFIG)
    assert receipt["source"]["rankings"]["rows"] == 17520
    assert receipt["source"]["scope"]["scored_rows"] == 11680
    assert receipt["source"]["scope"]["heldout_rows_scored"] == 0
    assert receipt["source"]["scope"]["heldout_rows_present"] == 5840
    assert receipt["value_frontier"]["point_count"] == 25
    assert receipt["address_frontier"]["point_count"] == 25
    assert receipt["frontier_summary"]["address_points_meeting_hit_and_precision"] == 0
    assert receipt["audit"]["native_rule_reproduces_artifact_acceptance"] is True
    native = receipt["gate_classification"]["native_point"]
    assert native["oracle_meets_gates"] is False
    assert native["observed"]["oracle_answerable_fact_recoverability"] == 1.0
    assert not all(native["oracle_value_checks"].values())
    assert receipt["gate_classification"]["disposition"]["selected_rule"] == (
        "oracle_cannot_meet_gates")
    assert receipt["gate_classification"]["disposition"]["stop"] is True
    # The configured receipt is not written by the test; this is read-only.
    output = tmp_path / "real_receipt.json"
    module.write_receipt(receipt, output)
    assert output.is_file()
    assert json.loads(output.read_text())["content_sha256"] == receipt[
        "content_sha256"]
