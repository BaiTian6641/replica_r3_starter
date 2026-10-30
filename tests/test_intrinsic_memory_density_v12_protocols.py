"""The three v12 density comparisons stay separate, preregistered and non-native."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from replica_lab.intrinsic_memory_density_protocols_v12 import (
    COST_TOLERANCE_RELATIVE, DECLARED_COMMON_UPDATES, DECLARED_UPDATES_CAP,
    FILENAMES, build_protocols, canonical_digest, cost_tolerance,
    fit_fixed_marginal, load_protocols, measured_cost_target,
    reader_parameters_per_site, select_updates, validate_protocol,
    write_protocols,
)

ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_protocols_match_the_versioned_builders_exactly():
    checked = load_protocols(ROOT / "configs")
    built = build_protocols()
    assert checked == built
    assert len(checked) == 3
    for name, protocol in checked.items():
        validate_protocol(protocol, comparison=name)
        assert protocol["native_launch_enabled"] is False
        assert protocol["shared_contract"]["P_updates"] == 0
        assert protocol["shared_contract"]["promotion"] is False
        assert protocol["gates"]["one_exclusive_native_worker"] is True
        assert protocol["gates"]["driver_limit_bytes"] == 50 * 1024**3
        assert protocol["gates"]["sustained_swap_growth_limit_bytes"] == 512 * 1024**2
        assert protocol["gates"]["minimum_host_free_fraction"] == 0.15
        assert protocol["authorization"]["inherited_from_v3_or_v11"] is False
        assert protocol["authorization"]["native_launch_enabled"] is False
        assert protocol["confounds"]


def test_three_comparisons_have_distinct_varied_factors_and_confounds():
    protocols = build_protocols()
    factors = [value["varied_factor"] for value in protocols.values()]
    assert len(set(factors)) == 3
    assert protocols["fixed_per_site"]["varied_factor"] == (
        "active_site_count_and_independent_reader_capacity")
    assert protocols["matched_unique_parameters"]["varied_factor"] == (
        "number_of_site_dispatch_aliases_for_two_canonical_groups")
    assert protocols["preregistered_measured_total_cost"]["varied_factor"] == (
        "training_updates_per_world_whole_pass")
    assert protocols["fixed_per_site"]["shared_contract"]["training_exposure"] == (
        protocols["matched_unique_parameters"]["shared_contract"]["training_exposure"])


def test_fixed_and_matched_parameter_tables_are_exact_and_not_interchangeable():
    fixed = build_protocols()["fixed_per_site"]["parameter_accounting"]
    matched = build_protocols()["matched_unique_parameters"]["parameter_accounting"]
    per_site = reader_parameters_per_site()
    assert fixed["parameters_per_site"] == per_site == 1_447_937
    for arm, sites in (("two", 2), ("full_attention_nine", 9), ("all", 36)):
        row = fixed["arms"][arm]
        assert row["unique_reader_parameters"] == per_site * sites
        assert row["reader_parameter_tensors"] == 11 * sites
        assert matched["site_aliases_by_arm"][arm] == sites
    assert matched["unique_reader_groups"] == 2
    assert matched["unique_reader_parameters"] == 2 * per_site == 2_895_874
    assert matched["unique_reader_parameter_tensors"] == 22
    assert len(matched["site_to_group"]["all"]) == 36
    assert set(matched["site_to_group"]["all"].values()) == {0, 1}


def test_total_cost_protocol_preregisters_cost_only_selection_before_labels():
    protocol = build_protocols()["preregistered_measured_total_cost"]
    pilot, selection = protocol["pilot"], protocol["selection"]
    assert pilot["separate_bounded_packet_required"] is True
    assert pilot["pilot_labels_scored"] is False
    assert pilot["pilot_answers_never_select_by_quality"] is True
    assert pilot["restore_common_initial_weights_before_each_pilot_run"] is True
    assert pilot["charge_all_pilot_cost_to_the_arm_and_comparison"] is True
    assert selection["target_arm"] == "two"
    assert selection["selection_before_scored_labels"] is True
    assert selection["selection_inputs"] == (
        "measured_cost_only_never_development_or_final_answers")
    assert selection["outside_tolerance"] == (
        "arm_is_unmatched_evidence_not_a_causal_arm")
    assert "floor" in selection["rule"]
    assert selection["tolerance"]["relative"] == 0.02
    assert protocol["gates"]["no_budget_change_after_scored_labels"] is True
    assert protocol["shared_contract"]["training_exposure"] == (
        "varies_by_preregistered_measured_total_cost_rule_only_after_pilot")
    assert protocol["shared_contract"] != build_protocols()["fixed_per_site"]["shared_contract"]


@pytest.mark.parametrize("change", ("post_label_budget", "score_selection",
                                    "pilot_scores", "common_exposure"))
def test_weakened_total_cost_or_exposure_contract_is_refused(change):
    protocol = deepcopy(build_protocols()["preregistered_measured_total_cost"])
    if change == "post_label_budget":
        protocol["gates"]["no_budget_change_after_scored_labels"] = False
    elif change == "score_selection":
        protocol["selection"]["selection_inputs"] = "development_answers"
    elif change == "pilot_scores":
        protocol["pilot"]["pilot_labels_scored"] = True
    else:
        protocol["selection"]["declared_updates_cap"] = "chosen_after_pilot"
    with pytest.raises(ValueError):
        validate_protocol(protocol, comparison="preregistered_measured_total_cost")


def test_fixed_and_matched_cannot_change_training_exposure_or_placement():
    for name in ("fixed_per_site", "matched_unique_parameters"):
        protocol = deepcopy(build_protocols()[name])
        protocol["shared_contract"]["training_exposure"] = "per_arm_choice"
        with pytest.raises(ValueError):
            validate_protocol(protocol, comparison=name)
        protocol = deepcopy(build_protocols()[name])
        protocol["shared_contract"]["placement"] = "pre_mlp_fusion"
        with pytest.raises(ValueError):
            validate_protocol(protocol, comparison=name)


def test_measured_cost_fit_selection_and_tolerance_are_deterministic():
    fit = fit_fixed_marginal({0: 100.0, 1: 150.0, 4: 300.0})
    assert fit["fixed_cost"] == 100.0
    assert fit["marginal_cost"] == 50.0
    target = measured_cost_target(fit)
    assert target == 100.0 + 50.0 * DECLARED_COMMON_UPDATES
    assert select_updates(100.0, 50.0, target) == DECLARED_COMMON_UPDATES
    assert select_updates(1000.0, 50.0, target) == 1
    assert select_updates(-100.0, 50.0, 10**9) == DECLARED_UPDATES_CAP
    assert cost_tolerance(target, 50.0) == max(
        target * COST_TOLERANCE_RELATIVE, 50.0)
    with pytest.raises(ValueError):
        fit_fixed_marginal({0: 100.0, 1: 100.0})
    with pytest.raises(ValueError):
        select_updates(100.0, 0.0, target)


def test_protocol_digest_is_canonical_and_write_refuses_overwrite(tmp_path):
    protocols = build_protocols()
    for name, protocol in protocols.items():
        assert canonical_digest(protocol) == canonical_digest(
            json.loads(json.dumps(protocol, sort_keys=True)))
    write_protocols(tmp_path)
    with pytest.raises(FileExistsError):
        write_protocols(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(FILENAMES.values())
