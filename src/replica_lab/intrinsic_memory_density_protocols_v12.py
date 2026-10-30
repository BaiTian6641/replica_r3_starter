"""Three separate preregistered density comparisons for the v12 candidate.

None of these protocols authorizes native execution.  They differ in what is
held constant and therefore in what can be inferred:

``fixed_per_site``
    Reader parameters and independent optimization paths grow with the number of
    access sites.  The primary comparison changes density and reader capacity
    together; it is feasibility evidence, not a pure location effect.

``matched_unique_parameters``
    Two canonical reader groups serve 2, 9 or 36 sites.  Unique parameters,
    optimizer tensors, initialization and checkpoint keys stay fixed.  Reuse and
    gradient interference change with density, so this also is not a pure effect
    of site count.

``preregistered_measured_total_cost``
    A separately bounded pilot measures per-arm fixed and marginal cost before
    any scored label is opened.  A fixed deterministic rule selects whole-pass
    training exposure so measured total cost is approximately matched.  The
    changed exposure is an explicit confound and is reported separately from the
    equal-exposure comparisons.

The protocols deliberately publish no native budget.  Every arm must obtain its
own source, execution, numerical, resource, scoring, data, output and
authorization binding; v3/v11 packets are not inherited.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

FORMAT = "replica.intrinsic_memory.density.comparison.v12"
IMPLEMENTATION = "replica.intrinsic_memory.density.v12"
PLACEMENT = "block_input_query_to_post_block_residual"
NATIVE_HIDDEN = 2560
WIDTH = 256
SLOTS = 8
SLOTS_PER_SPAN = 4
MAX_SPANS = 2
BACKBONE_PARAMETERS = 4_112_079_360
NATIVE_SOURCE = "official_BF16_XHToken/Spark-X2.6-4B@66e3ae7bee54e0a523d692cb8ef841f35409807b"
HISTORICAL_FP8 = "XHToken/Spark-X2.6-4B-FP8@0318d6f6ab070c7b5b43052db571489b2508006c"
ARMS = {
    "two": (3, 7),
    "full_attention_nine": (3, 7, 11, 15, 19, 23, 27, 31, 35),
    "all": tuple(range(36)),
}
MATCHED_MAPPING = "group=(layer//4)%2"
MATCHED_ANCHORS = (3, 7)
MATCHED_GROUPS = 2
READER_PARAMETER_TENSORS_PER_SITE = 11  # Q/out, K/V, three affine norms, gate.
COMPARISONS = ("fixed_per_site", "matched_unique_parameters",
               "preregistered_measured_total_cost")


def reader_parameters_per_site(hidden_size=NATIVE_HIDDEN):
    d, m = int(hidden_size), WIDTH
    return 2 * d * m + 2 * m * m + 2 * d + 4 * m + 1


def reader_parameter_table(hidden_size=NATIVE_HIDDEN):
    per = reader_parameters_per_site(hidden_size)
    rows = {}
    for name, sites in ARMS.items():
        count = per * len(sites)
        rows[name] = {
            "sites": list(sites),
            "active_sites": len(sites),
            "unique_reader_parameters": count,
            "reader_parameter_tensors": READER_PARAMETER_TENSORS_PER_SITE * len(sites),
            "fp32_weight_bytes": count * 4,
            "fp32_weight_gradient_two_moments_bytes": count * 16,
            "share_of_declared_backbone_parameters": count / BACKBONE_PARAMETERS,
        }
    return rows


def matched_parameter_table(hidden_size=NATIVE_HIDDEN):
    per = reader_parameters_per_site(hidden_size)
    count = per * MATCHED_GROUPS
    return {
        "unique_reader_groups": MATCHED_GROUPS,
        "group_parameters_each": per,
        "unique_reader_parameters": count,
        "unique_reader_parameter_tensors": READER_PARAMETER_TENSORS_PER_SITE * MATCHED_GROUPS,
        "site_aliases_by_arm": {name: len(sites) for name, sites in ARMS.items()},
        "site_to_group": {name: {str(site): (site // 4) % 2 for site in sites}
                          for name, sites in ARMS.items()},
        "fp32_weight_bytes": count * 4,
        "fp32_weight_gradient_two_moments_bytes": count * 16,
        "share_of_declared_backbone_parameters": count / BACKBONE_PARAMETERS,
    }


def _shared_contract():
    return {
        "placement": PLACEMENT,
        "decoder_layers": 36,
        "reader_width": WIDTH,
        "selected_slots": SLOTS,
        "slots_per_span": SLOTS_PER_SPAN,
        "maximum_spans": MAX_SPANS,
        "source_producer": "reviewed_v12_frozen_source_features_contextual_compressor",
        "source_input": "canonical_span_text_only_no_question_answer_template_or_spine",
        "writer_behavior": "completed_observation_admission_outside_backbone_forward",
        "writes_per_observation": "at_most_one_explicit_admission",
        "update_frequency": "declared_whole_pass_optimizer_updates_only",
        "training_exposure": "identical_across_arms_in_this_comparison",
        "data_split": "same_evidence_and_independent_world_families_per_arm",
        "template": "identical_prompt_template_and_thinking_mode",
        "scorer": "identical_exact_and_contextual_scorer_version",
        "generation_limits": "greedy_no_sampling_identical_max_new_tokens",
        "retrieval": "outside_backbone_forward_one_shared_snapshot_per_forward",
        "projection_cache": "none",
        "optimizer": "NativeFusedAdamW_one_owner_per_intended_parameter",
        "original_parameters": "frozen_unchanged_full_hashes_required",
        "P_updates": 0,
        "promotion": False,
        "external_spine": "separate_specialist_sensor_actor_responsibilities",
        "precision": "official_BF16_identity_bound_per_arm",
        "native_qualification": False,
        "excluded_followups": {
            "pre_mlp_fusion": "separate protocol required; not a density arm",
            "sites_23_31": "diagnostic only; not a compulsory milestone and not folded into 2/9/36",
            "persistent_P_learning": "gated; zero P updates here",
            "promotion_and_RSI": "gated and out of scope",
        },
    }


def measurement_schema():
    """Every cost/quality field an arm must emit, with units and separation."""
    return {
        "quality": {
            "generated_answer_quality": "per_case_scored_no_pooling_away_failures",
            "false_recall": "count_per_case",
            "exact_text_gap": "cases_exact_correct_minus_contextual_correct",
            "negative_controls": ["wrong", "shuffled", "zero", "absent"],
            "chronological_correction": "post_correction_stale_acceptances",
            "revocation": "post_revocation_stale_acceptances",
            "incomplete_evidence": "reported_incomplete_never_imputed",
        },
        "cost": {
            "total_generated_tokens": "tokens",
            "total_input_prompt_tokens": "tokens",
            "reader_invocations": "site_dispatches",
            "physical_group_invocations": "module_forward_calls",
            "activation_forward_bytes": "deterministic_tensor_payload_bytes",
            "activation_backward_work": "backward_calls_and_processed_positions",
            "source_encodes": "frozen_source_forward_calls",
            "source_input_tokens": "tokens",
            "retrieval_calls": "count",
            "retrieval_wall_time": "nanoseconds_monotonic",
            "read_file_bytes": "bytes",
            "durable_writes": "count_and_bytes",
            "snapshot_copies": "count",
            "snapshot_bytes": "bytes",
            "feature_cache_bytes": "current_and_peak_bytes",
            "teaching_cost": "updates_and_processed_positions",
            "replay_cost": "count_and_wall_time",
            "audit_cost": "hashed_and_serialized_bytes",
            "wall_time": "nanoseconds_monotonic",
        },
        "resources": {
            "driver_memory_peak": "sampled_bytes_never_added_to_rss",
            "process_rss_peak": "sampled_bytes",
            "sustained_swap_growth": "bytes_over_declared_samples",
            "minimum_host_free_fraction": "fraction",
            "artifact_bytes": "bytes",
            "checkpoint_bytes": "bytes",
        },
        "separation": {
            "single_forward_tiny_fixture_timing": "engineering_only_not_native_performance",
            "synchronized_native_timing": "declared_backend_synchronization",
            "parameter_counts": "not_a_cost_or_fit_measurement",
            "operation_counts": "not_physical_memory_or_latency",
        },
    }


def _authorization():
    return {
        "required": True,
        "inherited_from_v3_or_v11": False,
        "per_arm_bindings": [
            "source", "execution", "numerical", "resource", "scoring", "data",
            "output", "authorization",
        ],
        "authorization_receipt_required": True,
        "native_launch_enabled": False,
        "consumed_receipts_reusable": False,
    }


def _gates():
    return {
        "useful_two_site_readout_required_before_native_density": True,
        "matching_two_site_mechanism_must_pass": [
            "frozen_competence", "useful_readout", "integrity", "continuation",
            "resource",
        ],
        "one_exclusive_native_worker": True,
        "driver_limit_bytes": 50 * 1024**3,
        "sustained_swap_growth_limit_bytes": 512 * 1024**2,
        "minimum_host_free_fraction": 0.15,
        "P_updates": 0,
        "promotion": False,
        "native_retries": 0,
        "no_full_backbone_export": True,
        "no_download": True,
    }


def _confound_common():
    return [
        "Reader parameter estimates exclude producer, order, activations, temporary buffers, optimizer scalars and the backbone; they cannot declare viability.",
        "Retrieval and writes are outside backbone forward and are constant across density only if the caller keeps them constant; measure them rather than assuming it.",
        "One shared snapshot does not imply equal activation memory, backward graph length or latency.",
        "Random-fixture CPU evidence is engineering mechanics, not Spark competence or native density qualification.",
    ]


def fixed_site_protocol():
    table = reader_parameter_table()
    return {
        "format": FORMAT,
        "comparison": "fixed_per_site",
        "implementation": IMPLEMENTATION,
        "status": "preregistered_CPU_contract_native_unqualified",
        "native_launch_enabled": False,
        "objective": "Measure 2/9/36 active sites with one independent reader per site while holding producer, evidence, writer, exposure, splits, template, scorer and generation limits fixed.",
        "shared_contract": _shared_contract(),
        "varied_factor": "active_site_count_and_independent_reader_capacity",
        "fixed_factors": [
            "producer weights, release and initialization seed",
            "order parameters and selected slot order",
            "admissible evidence set, selected slots and data split",
            "writer behavior and update frequency",
            "training exposure, template, thinking mode, scorer and generation limits",
        ],
        "parameter_accounting": {
            "hidden_size": NATIVE_HIDDEN,
            "reader_parameter_formula": "per_site = 2*d*m + 2*m*m + 2*d + 4*m + 1",
            "parameters_per_site": reader_parameters_per_site(),
            "parameter_tensors_per_site": READER_PARAMETER_TENSORS_PER_SITE,
            "checkpoint_ownership": "one_independent_reader_copy_per_active_site",
            "arms": table,
            "excluded": [
                "source feature compressor", "order embedding", "backbone",
                "activations", "temporary buffers", "optimizer scalar state",
            ],
            "producer_and_order_counts": "invariant across arms; measure at binding, do not infer viability from reader counts",
        },
        "confounds": [
            "Reader parameter count, optimizer tensors, activation graph length and compute all change with site count; the comparison does not isolate access location.",
            "Independent readers give each site its own optimization path, while matched readers couple sites through shared gradients.",
        ] + _confound_common(),
        "measurement_schema": measurement_schema(),
        "authorization": _authorization(),
        "gates": _gates(),
        "scientific_claim_limit": "Feasibility evidence for access density; broader causal claims require independent world families, seeds and the matched controls.",
    }


def matched_protocol():
    table = matched_parameter_table()
    return {
        "format": FORMAT,
        "comparison": "matched_unique_parameters",
        "implementation": "replica.intrinsic_memory.matched_reader.v12",
        "status": "preregistered_CPU_implemented_matched_v12_native_unqualified",
        "native_launch_enabled": False,
        "implementation_scope": (
            "The separately versioned matched v12 wrapper constructs only the two "
            "anchor groups, serves 2/9/36 sites as dispatch aliases and keeps the "
            "reviewed producer/store/checkpoint path. The existing v5 matched "
            "candidate remains a regression control, not the v12 binding."
        ),
        "producer_binding": "reviewed_v12_direct_matched_wrapper",
        "objective": "Serve 2/9/36 sites from exactly two canonical reader groups with identical unique parameters, order parameters, initialization and checkpoint ownership.",
        "shared_contract": _shared_contract(),
        "varied_factor": "number_of_site_dispatch_aliases_for_two_canonical_groups",
        "fixed_factors": [
            "two canonical reader groups and their unique parameters",
            "group initialization streams anchored at sites 3 and 7",
            "producer weights, release and initialization seed",
            "order parameters and selected slot order",
            "admissible evidence set, selected slots and data split",
            "writer behavior, update frequency and training exposure",
            "template, thinking mode, scorer and generation limits",
        ],
        "parameter_accounting": {
            "hidden_size": NATIVE_HIDDEN,
            "mapping": MATCHED_MAPPING,
            "initialization_anchors": list(MATCHED_ANCHORS),
            "initialization_seed": 17,
            "initialization": "two_canonical_groups_from_the_same_anchor_streams_across_arms",
            "checkpoint_ownership": "one_groups.0_and_groups.1_copy_no_site_alias_state_keys",
            "resident_copies": "one_parameter_copy_per_canonical_group",
            **table,
            "excluded": [
                "source feature compressor", "order embedding", "backbone",
                "activations", "temporary buffers", "optimizer scalar state",
            ],
            "aliases_are_not_owners": "site aliases must not create state keys, parameter copies or extra optimizer entries",
        },
        "confounds": [
            "Sharing changes parameter reuse, gradient interference and optimization structure as well as parameter count; it does not isolate site count.",
            "Equal unique parameters do not imply equal compute, activation memory, latency, teaching cost or utility.",
            "The two-site arm is initialization-identical to the independent two-site control, but higher-density arms are not equivalent to independent-reader arms.",
        ] + _confound_common(),
        "measurement_schema": measurement_schema(),
        "authorization": _authorization(),
        "gates": _gates(),
        "scientific_claim_limit": "Parameter-matched access control; no marginal effect of site count is identified without matched total cost and independent worlds.",
    }


def total_cost_protocol():
    return {
        "format": FORMAT,
        "comparison": "preregistered_measured_total_cost",
        "implementation": IMPLEMENTATION,
        "status": "preregistered_not_run_native_unqualified",
        "native_launch_enabled": False,
        "objective": "Compare 2/9/36 access sites at approximately equal measured total engineering cost by varying only a preregistered whole-pass exposure variable.",
        "reference_architecture": "fixed_per_site",
        "shared_contract": {
            **_shared_contract(),
            "training_exposure": (
                "varies_by_preregistered_measured_total_cost_rule_only_after_pilot"),
        },
        "varied_factor": "training_updates_per_world_whole_pass",
        "fixed_factors": [
            "source producer, evidence set, selected slots and data family",
            "writer behavior, retrieval schedule, template, scorer and generation limits",
            "optimization algorithm, learning rate and batch construction",
            "worlds, case order and scoring population",
            "pilot protocol, cost components and selection rule",
        ],
        "pilot": {
            "separate_bounded_packet_required": True,
            "pilot_inputs": "independent_pilot_worlds_disjoint_from_scored_development_and_final",
            "pilot_labels_scored": False,
            "restore_common_initial_weights_before_each_pilot_run": True,
            "charge_all_pilot_cost_to_the_arm_and_comparison": True,
            "runs_per_arm_update_counts": [0, 1, 4],
            "repeat": 3,
            "aggregate": PILOT_AGGREGATE,
            "cost_metric": COST_METRIC,
            "cost_unit": COST_UNIT,
            "workload": "one_durable_admission_and_read_plus_declared_training_updates",
            "fit": PILOT_FIT,
            "measured_components": [
                "source_encodes", "retrieval", "durable_publication", "reader_invocations",
                "activation_forward_bytes", "backward_work", "wall_time",
                "driver_memory_peak", "rss_peak", "cache_bytes", "artifact_bytes",
            ],
            "timing_rule": TIMING_RULE,
            "pilot_answers_never_select_by_quality": True,
        },
        "selection": {
            "target_arm": "two",
            "declared_common_updates": DECLARED_COMMON_UPDATES,
            "declared_updates_cap": DECLARED_UPDATES_CAP,
            "epsilon": SELECTION_EPSILON,
            "target": "measured_total_cost_of_the_two_site_arm_at_the_declared_common_exposure",
            "target_formula": TARGET_FORMULA,
            "rule": SELECTION_RULE,
            "rounding": ROUNDING_RULE,
            "tolerance": {
                "relative": COST_TOLERANCE_RELATIVE,
                "absolute_floor": TOLERANCE_ABSOLUTE,
                "rule": TOLERANCE_RULE,
            },
            "outside_tolerance": "arm_is_unmatched_evidence_not_a_causal_arm",
            "selection_before_scored_labels": True,
            "selection_inputs": "measured_cost_only_never_development_or_final_answers",
            "declared_updates_cap_fixed_before_pilot": True,
        },
        "cost_components": [
            "source_feature_encoding", "tokenization", "retrieval", "durable_writes",
            "reader_dispatch_compute", "shared_snapshot_copy", "activation_forward",
            "backward_and_optimizer", "generation", "audit_serialization", "replay",
        ],
        "confounds": [
            "The selected training exposure differs by arm; optimization exposure is not held constant.",
            "Equal measured total engineering cost does not equalize quality, teaching semantics, latency distribution or peak memory.",
            "Cost is environment, data and implementation specific; an operation count is not physical cost.",
            "Pilot measurements include noise and are charged; an arm outside tolerance is unmatched evidence.",
        ] + _confound_common(),
        "measurement_schema": measurement_schema(),
        "scientific_comparison": "Effect of access density at approximately equal measured total cost; exposure change is a declared confound and must be reported separately from equal-exposure comparisons.",
        "authorization": _authorization(),
        "gates": _gates() | {
            "pilot_packet_separate": True,
            "scored_packet_separate": True,
            "no_budget_change_after_scored_labels": True,
        },
        "scientific_claim_limit": "Cost-matched feasibility evidence only; not a pure density or persistent-learning result.",
    }


BUILDERS = {
    "fixed_per_site": fixed_site_protocol,
    "matched_unique_parameters": matched_protocol,
    "preregistered_measured_total_cost": total_cost_protocol,
}
FILENAMES = {
    "fixed_per_site": "fixed_per_site_v12.json",
    "matched_unique_parameters": "matched_unique_parameters_v12.json",
    "preregistered_measured_total_cost": "measured_total_cost_v12.json",
}


PILOT_UPDATE_COUNTS = (0, 1, 4)
DECLARED_COMMON_UPDATES = 2
DECLARED_UPDATES_CAP = 8
COST_TOLERANCE_RELATIVE = 0.02
SELECTION_EPSILON = 1e-9
PILOT_AGGREGATE = "median_of_repeats"
COST_METRIC = "measured_synchronized_wall_nanoseconds_for_one_complete_declared_pass"
COST_UNIT = "nanoseconds"
PILOT_FIT = "affine_fixed_cost_plus_marginal_cost_per_update_from_measured_runs"
TIMING_RULE = "synchronize_declared_backend_before_each_timed_boundary"
SELECTION_RULE = (
    "updates_arm = clamp(floor((target - fixed_cost_arm) / marginal_cost_arm "
    "+ 1e-9 binary_float_guard), 1, declared_updates_cap)")
TARGET_FORMULA = (
    "fixed_cost_two + marginal_cost_per_update_two * declared_common_updates")
ROUNDING_RULE = "floor_no_residual_reallocation_no_post_hoc_budget"
TOLERANCE_ABSOLUTE = "one_measured_marginal_update_cost_of_the_arm"
TOLERANCE_RULE = "max(target*0.02, one_marginal_update_cost_arm)"


def fit_fixed_marginal(measurements):
    """Least-squares affine fit over measured total costs by update count."""
    points = sorted((int(count), float(cost)) for count, cost in measurements.items())
    if len(points) < 2 or any(cost < 0 or not math.isfinite(cost) for _, cost in points):
        raise ValueError("at least two finite nonnegative cost measurements required")
    xs = [count for count, _ in points]
    ys = [cost for _, cost in points]
    xbar, ybar = sum(xs) / len(xs), sum(ys) / len(ys)
    denominator = sum((x - xbar) ** 2 for x in xs)
    if denominator <= 0:
        raise ValueError("distinct pilot update counts required")
    marginal = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denominator
    fixed = ybar - marginal * xbar
    if not math.isfinite(fixed) or not math.isfinite(marginal) or marginal <= 0:
        raise ValueError("measured marginal cost is not positive and finite")
    return {"fixed_cost": fixed, "marginal_cost": marginal,
            "measurements": {str(count): cost for count, cost in points}}


def measured_cost_target(fit, common_updates=DECLARED_COMMON_UPDATES):
    if type(common_updates) is not int or common_updates < 1:
        raise ValueError("positive declared common updates required")
    target = fit["fixed_cost"] + fit["marginal_cost"] * common_updates
    if not math.isfinite(target) or target <= 0:
        raise ValueError("invalid measured cost target")
    return target


def select_updates(fixed_cost, marginal_cost, target_cost, *,
                   updates_cap=DECLARED_UPDATES_CAP,
                   selection_epsilon=SELECTION_EPSILON):
    """Frozen rule: floor the affordable whole passes, clamp to 1..cap."""
    if (not all(math.isfinite(value) for value in
                (fixed_cost, marginal_cost, target_cost))
            or marginal_cost <= 0 or target_cost <= 0
            or type(updates_cap) is not int or updates_cap < 1
            or not math.isfinite(selection_epsilon) or selection_epsilon < 0):
        raise ValueError("invalid measured cost selection inputs")
    # The epsilon only absorbs binary-float representation of an exact affine
    # target; it does not add a free update beyond the measured fit.
    whole = (target_cost - fixed_cost) / marginal_cost
    return max(1, min(updates_cap, math.floor(whole + selection_epsilon)))


def cost_tolerance(target_cost, marginal_cost, *, relative=COST_TOLERANCE_RELATIVE):
    if (not math.isfinite(target_cost) or not math.isfinite(marginal_cost)
            or target_cost <= 0 or marginal_cost <= 0
            or not math.isfinite(relative) or relative < 0):
        raise ValueError("invalid cost tolerance inputs")
    return max(target_cost * relative, marginal_cost)


def canonical_digest(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def validate_protocol(protocol, *, comparison=None):
    """Independent schema/scientific-invariant check; returns a small receipt."""
    if not isinstance(protocol, dict) or protocol.get("format") != FORMAT:
        raise ValueError("unsupported density comparison protocol")
    name = protocol.get("comparison")
    if name not in COMPARISONS or (comparison is not None and name != comparison):
        raise ValueError("undeclared density comparison")
    if protocol.get("native_launch_enabled") is not False:
        raise ValueError("CPU density protocol cannot enable native launch")
    if protocol.get("authorization", {}).get("native_launch_enabled") is not False:
        raise ValueError("authorization section cannot claim native admission")
    shared = protocol.get("shared_contract", {})
    expected = _shared_contract()
    for key in ("placement", "reader_width", "selected_slots", "slots_per_span",
                "maximum_spans", "P_updates", "promotion", "projection_cache",
                "optimizer", "original_parameters"):
        if shared.get(key) != expected[key]:
            raise ValueError("shared density contract differs: " + key)
    gates = protocol.get("gates", {})
    for key in ("driver_limit_bytes", "sustained_swap_growth_limit_bytes",
                "minimum_host_free_fraction", "one_exclusive_native_worker",
                "P_updates", "promotion", "no_download"):
        if key not in gates:
            raise ValueError("missing density guard: " + key)
    if "varied_factor" not in protocol or "confounds" not in protocol:
        raise ValueError("comparison must declare its varied factor and confounds")
    followups = shared.get("excluded_followups", {})
    if (followups.get("pre_mlp_fusion") != "separate protocol required; not a density arm"
            or followups.get("persistent_P_learning") != "gated; zero P updates here"):
        raise ValueError("pre-MLP and persistent-P followups must remain separate")
    if not isinstance(protocol.get("measurement_schema"), dict):
        raise ValueError("measurement schema is required")

    if name == "fixed_per_site":
        accounting = protocol["parameter_accounting"]
        if accounting.get("parameters_per_site") != reader_parameters_per_site():
            raise ValueError("fixed per-site reader count differs")
        if accounting.get("parameter_tensors_per_site") != READER_PARAMETER_TENSORS_PER_SITE:
            raise ValueError("fixed per-site tensor count differs")
        if accounting.get("arms") != reader_parameter_table():
            raise ValueError("fixed per-site arm counts differ")
        if shared.get("training_exposure") != expected["training_exposure"]:
            raise ValueError("fixed comparison cannot change training exposure")
    elif name == "matched_unique_parameters":
        accounting = protocol["parameter_accounting"]
        table = matched_parameter_table()
        for key, value in table.items():
            if accounting.get(key) != value:
                raise ValueError("matched parameter accounting differs: " + key)
        if accounting.get("mapping") != MATCHED_MAPPING:
            raise ValueError("matched mapping differs")
        if shared.get("training_exposure") != expected["training_exposure"]:
            raise ValueError("matched comparison cannot change training exposure")
    else:
        if shared.get("training_exposure") != (
                "varies_by_preregistered_measured_total_cost_rule_only_after_pilot"):
            raise ValueError("total-cost comparison must declare its exposure change")
        pilot = protocol.get("pilot", {})
        selection = protocol.get("selection", {})
        if pilot.get("separate_bounded_packet_required") is not True:
            raise ValueError("total-cost pilot requires its own bounded packet")
        if pilot.get("pilot_labels_scored") is not False:
            raise ValueError("pilot cannot score labels")
        if pilot.get("restore_common_initial_weights_before_each_pilot_run") is not True:
            raise ValueError("pilot must restore common initial weights")
        if pilot.get("charge_all_pilot_cost_to_the_arm_and_comparison") is not True:
            raise ValueError("pilot cost must be charged")
        if (pilot.get("cost_unit") != COST_UNIT
                or pilot.get("aggregate") != PILOT_AGGREGATE
                or pilot.get("cost_metric") != COST_METRIC
                or pilot.get("fit") != PILOT_FIT
                or pilot.get("timing_rule") != TIMING_RULE
                or type(pilot.get("repeat")) is not int or pilot["repeat"] != 3
                or pilot.get("runs_per_arm_update_counts") != [0, 1, 4]):
            raise ValueError("pilot measurement contract differs")
        if selection.get("selection_before_scored_labels") is not True:
            raise ValueError("total-cost selection must precede scored labels")
        if selection.get("target_arm") != "two":
            raise ValueError("measured-cost target must remain the two-site arm")
        common = selection.get("declared_common_updates")
        cap = selection.get("declared_updates_cap")
        epsilon = selection.get("epsilon")
        if (type(common) is not int or common != DECLARED_COMMON_UPDATES
                or type(cap) is not int or cap != DECLARED_UPDATES_CAP
                or type(epsilon) is not float or epsilon != SELECTION_EPSILON):
            raise ValueError(
                "exposure target and cap must be exact frozen ints (not float/bool)")
        if selection.get("outside_tolerance") != "arm_is_unmatched_evidence_not_a_causal_arm":
            raise ValueError("outside-tolerance arms must remain unmatched")
        if selection.get("selection_inputs") != "measured_cost_only_never_development_or_final_answers":
            raise ValueError("selection cannot inspect scored answers")
        if protocol.get("varied_factor") != "training_updates_per_world_whole_pass":
            raise ValueError("only whole-pass exposure may vary")
        expected_tolerance = {
            "relative": COST_TOLERANCE_RELATIVE,
            "absolute_floor": TOLERANCE_ABSOLUTE,
            "rule": TOLERANCE_RULE,
        }
        if (selection.get("rule") != SELECTION_RULE
                or selection.get("rounding") != ROUNDING_RULE
                or selection.get("target_formula") != TARGET_FORMULA
                or selection.get("tolerance") != expected_tolerance):
            raise ValueError("selection/tolerance/rounding values differ from the frozen contract")
        if not isinstance(selection.get("tolerance", {}).get("relative"), (int, float)):
            raise ValueError("measured-cost tolerance required")
        if (protocol.get("gates", {}).get("no_budget_change_after_scored_labels") is not True
                or protocol.get("gates", {}).get("pilot_packet_separate") is not True
                or protocol.get("gates", {}).get("scored_packet_separate") is not True):
            raise ValueError("total-cost packets and post-label budget must stay fixed")
    return {
        "comparison": name,
        "status": "protocol_consistent_native_not_authorized",
        "sha256": canonical_digest(protocol),
        "native_launch_enabled": False,
        "P_updates": 0,
        "varied_factor": protocol["varied_factor"],
    }


def build_protocols():
    return {name: builder() for name, builder in BUILDERS.items()}


def write_protocols(directory):
    """Write the three separately named contracts; refuse to overwrite."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for name, protocol in build_protocols().items():
        validate_protocol(protocol, comparison=name)
        path = directory / FILENAMES[name]
        if path.exists():
            raise FileExistsError(f"refusing to overwrite protocol: {path}")
        path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
        receipts[name] = {"path": path.name, "sha256": canonical_digest(protocol)}
    return receipts


def load_protocols(directory):
    directory = Path(directory)
    result = {}
    for name, filename in FILENAMES.items():
        protocol = json.loads((directory / filename).read_text())
        validate_protocol(protocol, comparison=name)
        result[name] = protocol
    return result


def summary():
    fixed = fixed_site_protocol()
    matched = matched_protocol()
    total = total_cost_protocol()
    return {
        "format": FORMAT,
        "implementation": IMPLEMENTATION,
        "hidden_size": NATIVE_HIDDEN,
        "reader_parameters_per_site": reader_parameters_per_site(),
        "fixed_per_site": fixed["parameter_accounting"]["arms"],
        "matched_unique_parameters": matched["parameter_accounting"],
        "comparisons": {
            name: {"varied_factor": value["varied_factor"],
                   "confound_count": len(value["confounds"]),
                   "sha256": canonical_digest(value)}
            for name, value in (("fixed_per_site", fixed),
                                ("matched_unique_parameters", matched),
                                ("preregistered_measured_total_cost", total))
        },
        "native_launch_enabled": False,
        "P_updates": 0,
        "promotion": False,
    }
