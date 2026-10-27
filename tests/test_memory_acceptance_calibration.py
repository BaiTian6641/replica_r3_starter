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
REAL_MISSING = [str(path) for path in (REAL_RANKINGS, REAL_INVENTORY)
                if not path.exists()]


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
