"""CPU engineering evidence for the contextual fixture and P1 direct CE.

These tests do not run a native Spark model, do not open sealed evaluation and
do not claim useful learning.  They verify mechanics: contextual token mixing,
arm/control separation, direct response-CE gradients reaching P, the single P
optimizer owner, actual deltas/moments/clocks, detach boundaries, and the
fixture-only status of the 23/31 site comparison.
"""
from __future__ import annotations

import torch

from contextual_spark_fixture import (
    LATE_SITE_MIN_LAYERS, NATIVE_SEALED_LAYERS, build_contextual_path,
    build_contextual_adapter, ids_and_mask, labels_from_ids, open_memory_ports,
    populate_contextual_evidence, token_mixing_probe,
)
from replica_lab.p1_response_learning import (
    DETACH_BOUNDARIES, audit_p_only_claim, build_port_optimizer,
    capture_p_state, direct_response_ce_step, evaluate_response_ce,
    port_output_parameter_names, prepare_arm_query, restore_p_state,
    selected_parameter_names, site_accounting,
)
from test_spark2_5_adapter import build as build_pointwise


RECORDS = (
    ("answer", "cobalt alpha is the verified response token"),
    ("near", "cobalt beta is a nearby distractor"),
    ("other", "nickel gamma is unrelated"),
)


def _p_query(path, *, arm="p_only", question="which token follows the prefix",
             input_ids=None, attention_mask=None, target_record_id=None,
             wrong_record_id=None, expert_id=0):
    ids = ids_and_mask()[0] if input_ids is None else input_ids
    mask = torch.ones_like(ids) if attention_mask is None else attention_mask
    return prepare_arm_query(
        path, arm=arm, question=question, input_ids=ids,
        attention_mask=mask, target_record_id=target_record_id,
        wrong_record_id=wrong_record_id, expert_id=expert_id)


def test_contextual_fixture_has_real_token_mixing_and_old_pointwise_fixture_is_separate():
    adapter, _, _, _ = build_contextual_adapter(
        seed=7, layers=32, spine_sites=(3, 7))
    probe = token_mixing_probe(adapter)
    assert probe["same_last_token"] is True
    assert probe["different_prefix"] is True
    assert probe["last_position_equal"] is False
    assert probe["last_position_max_abs_delta"] > 0.0
    assert probe["tied_output"] is True
    assert probe["mlp_activation"] == "gelu"

    pointwise, _, _, _ = build_pointwise(seed=7, spine_sites=(3, 7))
    left = torch.tensor([[3, 5, 7, 9, 11, 13]], dtype=torch.long)
    right = torch.tensor([[17, 19, 21, 23, 11, 13]], dtype=torch.long)
    mask = torch.ones_like(left)
    with torch.no_grad():
        left_logits = pointwise.forward_text(
            left, mask, use_projector=False, use_ports=False, use_spine=False)
        right_logits = pointwise.forward_text(
            right, mask, use_projector=False, use_ports=False, use_spine=False)
    # The historical engineering fixture is preserved and remains pointwise;
    # that property is why its 0/12 result is not a Spark diagnosis.
    assert torch.equal(left_logits[:, -1], right_logits[:, -1])


def test_direct_response_ce_updates_only_selected_p_through_single_owner():
    path = build_contextual_path(
        seed=7, layers=32, spine_sites=(3, 7), read_records=3,
        bank_experts=2, bank_hidden_width=8, bank_lr=1e-2)
    open_memory_ports(path)
    populate_contextual_evidence(path, RECORDS)
    ids, mask = ids_and_mask()
    labels = labels_from_ids(ids)
    query = _p_query(path, input_ids=ids, attention_mask=mask)
    expert_names = selected_parameter_names(path, 0)
    assert set(expert_names) == {"gate", "up", "down"}
    before_moments = path.experiment.bank.optimizer_state_for_expert(0)
    before_steps = tuple(path.experiment.bank.expert_steps)
    before_clock = path.experiment.controller.clock.eligible_events
    before_revision = path.experiment.controller.revision
    before_inactive = {
        name: parameter.detach().clone()
        for name, parameter in path.experiment.bank.expert(1).named_parameters()}

    result = direct_response_ce_step(
        path, query, labels, event_id="direct-ce-0", expert_id=0,
        post_query_factory=lambda: _p_query(
            path, input_ids=ids, attention_mask=mask))

    assert result["schema"] == "replica.p1_direct_response_ce.v1"
    assert result["engineering_evidence"] is True
    assert result["native_capability_claim"] is False
    assert result["sealed_evaluation"] == "closed"
    assert result["objective"] == "direct_response_cross_entropy"
    assert result["selected_expert_id"] == 0
    assert set(result["selected_parameter_names"]) == set(expert_names)
    assert set(result["changed_parameter_names"]) == set(expert_names)
    assert result["actual_parameter_update"] is True
    for name in expert_names:
        assert result["parameter_deltas"][name]["l2"] > 0.0
        assert result["parameter_deltas"][name]["changed"] is True
        assert result["parameter_deltas"][name]["before_sha256"] != \
            result["parameter_deltas"][name]["after_sha256"]
    assert result["gradients"]["sum_norm"] > 0.0
    assert result["gradients"]["spine_gradients_discarded"] is True
    assert result["moments"]["changed_keys"]
    assert all(key.startswith("0.") for key in result["moments"]["changed_keys"])
    after_moments = path.experiment.bank.optimizer_state_for_expert(0)
    for name, current in after_moments.items():
        assert current is not None
        assert int(current["step"]) == 1
    assert result["clock"]["eligible_events_advanced"] is True
    assert result["clock"]["revision_advanced"] is True
    assert result["clock"]["before"] == before_clock
    assert result["clock"]["after"] == before_clock + 1
    assert result["clock"]["revision_after"] == before_revision + 1
    assert tuple(path.experiment.bank.expert_steps) == (before_steps[0] + 1, 0)
    assert path.experiment.bank.expert_steps[0] == 1
    assert path.experiment.bank.expert_steps[1] == 0
    assert path.experiment.controller.clock.eligible_events == before_clock + 1

    ownership = result["optimizer_ownership"]
    assert ownership["update_owner"] == "expert_bank.optimizer"
    assert ownership["response_optimizer_stepped"] is False
    assert ownership["response_optimizer_state_changed"] is False
    assert ownership["response_optimizer_is_distinct"] is True
    assert ownership["response_steps_before"] == ownership["response_steps_after"]
    assert ownership["physical_optimizer_steps"] == 1
    assert path.response_optimizer is not path.experiment.bank.optimizer
    assert path.response_steps == 0
    assert result["response_spine_parameter_changes"] == []
    assert result["inactive_expert_changes"] == {1: []}
    for name, parameter in path.experiment.bank.expert(1).named_parameters():
        torch.testing.assert_close(parameter, before_inactive[name],
                                   rtol=0.0, atol=0.0)

    assert result["loss"]["before"] > 0.0
    assert result["loss"]["after_same_batch"] is not None
    assert result["loss"]["train_fit"] == "same_batch_ce_measured_only"
    assert result["transfer"] == "unevaluated"
    assert result["retention"] == "unevaluated"
    assert result["useful_learning"] == "unevaluated"
    assert result["learning_success"] is None
    assert result["site_accounting"]["native_comparison"] == \
        "eligible_geometry_native_backend_absent"
    assert result["fixture_only_late_sites"] is False
    assert result["native_execution"] == "not_run"


def test_zero_init_port_owner_first_port_update_then_p_then_retry_skip():
    path = build_contextual_path(
        seed=7, layers=32, spine_sites=(3, 7), read_records=3,
        bank_experts=1, bank_hidden_width=8, bank_lr=1e-2)
    # v2 integration: the path's response optimizer is explicitly the
    # port-only owner, and the same object is passed to the direct step.
    port_optimizer = build_port_optimizer(path, lr=1e-2)
    path.response_optimizer = port_optimizer
    assert path.response_optimizer is port_optimizer
    port_names = port_output_parameter_names(path)
    assert all(name.endswith("output_projection.weight") for name in port_names)
    assert all(".ports." in name for name in port_names)
    assert port_optimizer.audit()["groups"][0]["param_names"] == list(port_names)
    memory_port_names = {
        name for name in port_names if ".ports.memory." in name}
    assert len(memory_port_names) == 2
    ids, mask = ids_and_mask()
    labels = labels_from_ids(ids)

    query = _p_query(path, input_ids=ids, attention_mask=mask)
    initial = evaluate_response_ce(
        path, query, labels=labels, extra_passes=("null",))
    # Zero-initialized port outputs leave the response exactly null before any
    # training, even though P slots and the spine context are present.
    torch.testing.assert_close(
        initial["logits"], initial["null_logits"]["null"], rtol=0.0, atol=0.0)

    first = direct_response_ce_step(
        path, query, labels, event_id="zero-init-port-1",
        port_optimizer=port_optimizer)
    assert first["actual_parameter_update"] is False
    assert first["changed_parameter_names"] == []
    assert first["port_optimizer"]["provided"] is True
    assert first["port_optimizer"]["stepped"] is True
    assert first["port_optimizer"]["physical_steps"] == 1
    assert first["port_optimizer"]["skipped_on_retry"] is False
    assert set(first["port_optimizer"]["changed_parameter_names"]) == \
        memory_port_names
    assert set(first["port_optimizer"]["tensor_steps_advanced"]) == \
        memory_port_names
    for name in memory_port_names:
        assert first["port_parameter_deltas"][name]["l2"] > 0.0
        assert first["port_parameter_deltas"][name]["changed"] is True
        assert first["port_optimizer"]["step_counts_after"][name] == 1
        assert first["port_optimizer"]["state_after"][name]["step"] == 1
    for name in set(port_names) - memory_port_names:
        assert first["port_optimizer"]["step_counts_after"][name] == 0
        assert first["port_optimizer"]["state_after"][name] is None
    assert first["non_port_spine_parameter_changes"] == []
    assert first["response_spine_parameter_changes"] == []
    ownership = first["optimizer_ownership"]
    assert ownership["response_optimizer_is_declared_port_owner"] is True
    assert ownership["port_optimizer_reuses_response_optimizer"] is True
    assert ownership["response_optimizer_scope"] == "port_output_only"
    assert ownership["response_optimizer_stepped"] is True
    assert ownership["response_optimizer_state_changed"] is True
    assert ownership["spine_response_optimizer_stepped"] is False
    assert first["compute_charges"]["port_optimizer_steps"] == 1
    assert first["compute_charges"]["port_parameter_updates"] == \
        len(memory_port_names)
    assert path.response_steps == 0

    first_influence = evaluate_response_ce(
        path, _p_query(path, input_ids=ids, attention_mask=mask),
        labels=labels, extra_passes=("null",))
    assert not torch.equal(
        first_influence["logits"], first_influence["null_logits"]["null"])
    assert not torch.equal(first_influence["logits"], initial["logits"])

    second = direct_response_ce_step(
        path, _p_query(path, input_ids=ids, attention_mask=mask), labels,
        event_id="zero-init-port-2", port_optimizer=port_optimizer)
    assert second["actual_parameter_update"] is True
    assert second["changed_parameter_names"]
    assert set(second["changed_parameter_names"]) <= set(
        selected_parameter_names(path, 0))
    assert second["port_optimizer"]["stepped"] is True
    assert set(second["port_optimizer"]["changed_parameter_names"]) <= \
        memory_port_names
    assert second["port_optimizer"]["changed_parameter_names"]
    assert second["non_port_spine_parameter_changes"] == []
    assert second["optimizer_ownership"]["response_optimizer_stepped"] is True
    assert second["compute_charges"]["port_optimizer_steps"] == 1

    second_influence = evaluate_response_ce(
        path, _p_query(path, input_ids=ids, attention_mask=mask),
        labels=labels, extra_passes=("null",))
    assert not torch.equal(
        second_influence["logits"], second_influence["null_logits"]["null"])
    assert second["actual_parameter_update"] is True

    p_before_retry = capture_p_state(path, 0)
    port_before_retry = {
        name: parameter.detach().clone()
        for name, parameter in path.adapter.new_named_parameters()
        if name in set(port_names)}
    steps_before_retry = second["port_optimizer"]["step_counts_after"]
    clock_before_retry = path.experiment.controller.clock.eligible_events
    revision_before_retry = path.experiment.controller.revision
    retry = direct_response_ce_step(
        path, _p_query(path, input_ids=ids, attention_mask=mask), labels,
        event_id="zero-init-port-2", port_optimizer=port_optimizer)
    assert retry["replay"]["retried"] is True
    assert retry["actual_parameter_update"] is False
    assert retry["changed_parameter_names"] == []
    assert retry["port_optimizer"]["stepped"] is False
    assert retry["port_optimizer"]["skipped_on_retry"] is True
    assert retry["port_optimizer"]["changed_parameter_names"] == []
    assert retry["port_optimizer"]["changed_state_keys"] == []
    assert retry["port_optimizer"]["step_counts_after"] == steps_before_retry
    assert retry["optimizer_ownership"]["response_optimizer_stepped"] is False
    assert retry["compute_charges"]["port_optimizer_steps"] == 0
    assert retry["compute_charges"]["port_optimizer_retry_skipped"] is True
    assert path.experiment.controller.clock.eligible_events == clock_before_retry
    assert path.experiment.controller.revision == revision_before_retry
    for name, parameter in path.experiment.bank.expert(0).named_parameters():
        torch.testing.assert_close(parameter, p_before_retry.parameters[name],
                                   rtol=0.0, atol=0.0)
    for name, parameter in path.adapter.new_named_parameters():
        if name in port_before_retry:
            torch.testing.assert_close(parameter, port_before_retry[name],
                                       rtol=0.0, atol=0.0)


def test_arms_controls_disabled_parity_and_shuffled_memory():
    path = build_contextual_path(
        seed=7, layers=32, spine_sites=(3, 7), read_records=3,
        bank_experts=1, bank_hidden_width=8)
    open_memory_ports(path)
    populate_contextual_evidence(path, RECORDS)
    ids, mask = ids_and_mask()
    labels = labels_from_ids(ids)

    arms = {}
    for arm in ("full_eligible", "selected_exact", "e_only", "p_only",
                "e_plus_p", "wrong_memory", "absent_memory"):
        query = _p_query(
            path, arm=arm, input_ids=ids, attention_mask=mask,
            target_record_id="answer", wrong_record_id="other")
        arms[arm] = evaluate_response_ce(path, query, labels=labels)

    assert len(arms["full_eligible"]["selected_record_ids"]) == 3
    assert arms["selected_exact"]["selected_record_ids"] == ["answer"]
    assert arms["wrong_memory"]["selected_record_ids"] == ["other"]
    assert arms["absent_memory"]["selected_record_ids"] == []
    assert arms["p_only"]["selected_record_ids"] == []
    assert arms["absent_memory"]["charges"]["memory_read_slots"] == 0
    assert arms["p_only"]["charges"]["p_memory_read_slots"] == 1
    assert arms["e_plus_p"]["charges"]["e_memory_read_slots"] == 3
    assert arms["e_plus_p"]["charges"]["p_memory_read_slots"] == 1

    def differ(left, right):
        return not torch.equal(arms[left]["logits"], arms[right]["logits"])

    assert differ("full_eligible", "selected_exact")
    assert differ("selected_exact", "wrong_memory")
    assert differ("full_eligible", "absent_memory")
    assert differ("absent_memory", "p_only")
    assert differ("e_only", "e_plus_p")

    p_with_null = evaluate_response_ce(
        path, _p_query(path, input_ids=ids, attention_mask=mask),
        labels=labels, extra_passes=("null",))
    with torch.no_grad():
        disabled = path.adapter.forward_text(
            ids, mask, use_projector=False, use_ports=False, use_spine=False)
    torch.testing.assert_close(
        p_with_null["null_logits"]["null"], disabled, rtol=0.0, atol=0.0)
    assert not torch.equal(p_with_null["null_logits"]["null"],
                           p_with_null["logits"])

    shuffled = build_contextual_path(
        seed=7, layers=32, spine_sites=(3, 7), read_records=3,
        bank_experts=1, bank_hidden_width=8)
    open_memory_ports(shuffled)
    shuffled_records = (
        ("answer", "nickel gamma is unrelated"),
        ("near", "cobalt alpha is the verified response token"),
        ("other", "cobalt beta is a nearby distractor"),
    )
    populate_contextual_evidence(shuffled, shuffled_records)
    shuffled_query = _p_query(
        shuffled, arm="selected_exact", input_ids=ids, attention_mask=mask,
        target_record_id="answer")
    shuffled_result = evaluate_response_ce(shuffled, shuffled_query, labels=labels)
    assert shuffled_result["selected_record_ids"] == ["answer"]
    assert not torch.equal(arms["selected_exact"]["logits"],
                           shuffled_result["logits"])


def test_p_only_leakage_audit_and_disabled_parity():
    path = build_contextual_path(
        seed=7, layers=32, spine_sites=(3, 7), read_records=3,
        bank_experts=1, bank_hidden_width=8)
    open_memory_ports(path)
    populate_contextual_evidence(path, RECORDS)
    path.add_generated_explanation(
        "candidate", "cobalt may be the answer", ("answer",))
    path.open_task("task", (("answer", "say cobalt"),))
    path.clear_for_p_only()

    ids, mask = ids_and_mask()
    labels = labels_from_ids(ids)
    query = _p_query(path, input_ids=ids, attention_mask=mask)
    audit = audit_p_only_claim(
        path, query, answer_text="cobalt", answer_token_ids=(17, 19),
        labels=labels)
    assert audit["memory_source"] == "P"
    assert audit["selected_record_ids"] == []
    assert audit["packet_slot_count"] == 0
    assert audit["lineage_evidence_ids"] == []
    assert audit["source_counts"] == {
        "exact_text_records": 0, "procedures": 0, "task_workspaces": 0,
        "candidate_explanations": 0, "evidence_records": 0}
    assert audit["source_state_present"] is False
    assert audit["workspace_answer_leak"] is False
    assert audit["prompt_contains_answer_text"] is False
    assert audit["prompt_contains_answer_token_ids"] is False
    assert audit["teacher_forced_answer_positions"] > 0
    assert audit["teacher_forced_labels_declared"] is True
    assert audit["evidence_leak"] is False
    assert audit["identity_leak"] is False
    assert audit["p_only_claim"] == "source_free_engineering"
    assert audit["sealed_evaluation"] == "closed"
    assert audit["native_capability_claim"] is False

    p_null = evaluate_response_ce(path, query, labels=labels,
                                  extra_passes=("null",))
    with torch.no_grad():
        disabled = path.adapter.forward_text(
            ids, mask, use_projector=False, use_ports=False, use_spine=False)
    torch.testing.assert_close(p_null["null_logits"]["null"], disabled,
                               rtol=0.0, atol=0.0)
    # The retained P state still changes the response while the null branch is
    # exactly the frozen base; this is mechanics, not a learned-capability claim.
    assert not torch.equal(p_null["logits"], p_null["null_logits"]["null"])


def test_early_and_late_sites_charge_work_and_declare_detach_boundaries():
    early = build_contextual_path(
        seed=7, layers=32, spine_sites=(3, 7), read_records=3,
        bank_experts=1, bank_hidden_width=8, bank_lr=1e-2)
    late = build_contextual_path(
        seed=7, layers=32, spine_sites=(23, 31), read_records=3,
        bank_experts=1, bank_hidden_width=8, bank_lr=1e-2)
    ids, mask = ids_and_mask()
    labels = labels_from_ids(ids)
    results = {}
    for name, path in (("early", early), ("late", late)):
        open_memory_ports(path)
        populate_contextual_evidence(path, RECORDS)
        query = _p_query(path, input_ids=ids, attention_mask=mask)
        results[name] = direct_response_ce_step(
            path, query, labels, event_id=f"{name}-direct-ce", expert_id=0)

    early_meta = results["early"]["site_accounting"]
    late_meta = results["late"]["site_accounting"]
    assert early_meta["sites"] == [3, 7]
    assert late_meta["sites"] == [23, 31]
    assert early_meta["num_layers"] == late_meta["num_layers"] == 32
    assert early_meta["p_backward_suffix_layers_upper_bound"] == 29
    assert late_meta["p_backward_suffix_layers_upper_bound"] == 9
    assert early_meta["native_sites_valid"] is True
    assert late_meta["native_sites_valid"] is True
    assert late_meta["native_comparison"] == \
        "eligible_geometry_native_backend_absent"
    assert late_meta["late_site_native_status"] == \
        "geometry_valid_native_backend_absent"
    assert late_meta["native_sealed_layers"] == NATIVE_SEALED_LAYERS == 36
    assert late_meta["requires_at_least_layers"] == LATE_SITE_MIN_LAYERS
    assert early_meta["native_comparison"] != "native_compared"
    assert late_meta["native_execution"] == "not_run"
    assert results["late"]["fixture_only_late_sites"] is True
    assert results["early"]["fixture_only_late_sites"] is False
    assert late_meta["engineering_fixture_only"] is True

    for name in ("early", "late"):
        charges = results[name]["compute_charges"]
        assert charges["sensing_forward_calls"] == 1
        assert charges["decoder_forwards_objective"] == 1
        assert charges["backward_calls"] == 1
        assert charges["backward_positions"] == 4
        assert charges["optimizer_steps"] == 1
        assert charges["response_optimizer_steps"] == 0
        assert charges["sensing_forward_layers"] == 32
        assert charges["port_injections_per_forward"] == 2
        assert results[name]["actual_parameter_update"] is True
        assert results[name]["moments"]["changed_keys"]
        assert results[name]["clock"]["eligible_events_advanced"] is True
    assert results["early"]["compute_charges"]["p_backward_suffix_layers_upper_bound"] == 29
    assert results["late"]["compute_charges"]["p_backward_suffix_layers_upper_bound"] == 9
    assert results["early"]["compute_charges"]["backward_positions"] == \
        results["late"]["compute_charges"]["backward_positions"]
    assert results["early"]["compute_charges"]["sensing_forward_layers"] == \
        results["late"]["compute_charges"]["sensing_forward_layers"]
    assert any("native_hidden" in row for row in DETACH_BOUNDARIES)
    assert any("labels" in row for row in DETACH_BOUNDARIES)
    assert result_detach_declared(results["late"])


def result_detach_declared(result):
    boundaries = result["detach_boundaries"]
    assert boundaries == list(DETACH_BOUNDARIES)
    assert any("spine response gradients are discarded" in row
               for row in boundaries)
    return True


def test_matched_nonlearning_replay_reset_and_wrong_state_controls():
    path = build_contextual_path(
        seed=7, layers=32, spine_sites=(3, 7), read_records=3,
        bank_experts=1, bank_hidden_width=8, bank_lr=1e-2)
    open_memory_ports(path)
    populate_contextual_evidence(path, RECORDS)
    ids, mask = ids_and_mask()
    labels = labels_from_ids(ids)
    initial = capture_p_state(path, 0)
    query = _p_query(path, input_ids=ids, attention_mask=mask)

    nonlearning = direct_response_ce_step(
        path, query, labels, event_id="matched-no-update", apply=False)
    assert nonlearning["actual_parameter_update"] is False
    assert nonlearning["changed_parameter_names"] == []
    assert nonlearning["moments"]["changed_keys"] == []
    assert nonlearning["clock"]["eligible_events_advanced"] is False
    assert nonlearning["clock"]["revision_advanced"] is False
    assert nonlearning["optimizer_ownership"]["physical_optimizer_steps"] == 0
    assert nonlearning["nonlearning_control"]["optimizer_skipped"] is True
    assert nonlearning["nonlearning_control"]["matched_forward_backward_work"] is True
    assert nonlearning["compute_charges"]["backward_calls"] == 1
    assert nonlearning["compute_charges"]["optimizer_steps"] == 0
    for name, parameter in path.experiment.bank.expert(0).named_parameters():
        torch.testing.assert_close(parameter, initial.parameters[name],
                                   rtol=0.0, atol=0.0)

    committed = direct_response_ce_step(
        path, _p_query(path, input_ids=ids, attention_mask=mask), labels,
        event_id="p1-update", expert_id=0)
    assert committed["actual_parameter_update"] is True
    updated = capture_p_state(path, 0)
    clock_after_update = path.experiment.controller.clock.eligible_events
    revision_after_update = path.experiment.controller.revision

    replay = direct_response_ce_step(
        path, _p_query(path, input_ids=ids, attention_mask=mask), labels,
        event_id="p1-update", expert_id=0)
    assert replay["replay"]["retried"] is True
    assert replay["actual_parameter_update"] is False
    assert replay["changed_parameter_names"] == []
    assert replay["moments"]["changed_keys"] == []
    assert replay["clock"]["eligible_events_advanced"] is False
    assert path.experiment.controller.clock.eligible_events == clock_after_update
    assert path.experiment.controller.revision == revision_after_update
    for name, parameter in path.experiment.bank.expert(0).named_parameters():
        torch.testing.assert_close(parameter, updated.parameters[name],
                                   rtol=0.0, atol=0.0)

    reset = restore_p_state(path, initial)
    assert reset["restored"] is True
    assert reset["owner_identity_unchanged"] is True
    reset_result = evaluate_response_ce(
        path, _p_query(path, input_ids=ids, attention_mask=mask), labels=labels)
    assert abs(reset_result["loss"] - nonlearning["loss"]["before"]) < 1e-4
    for name, parameter in path.experiment.bank.expert(0).named_parameters():
        torch.testing.assert_close(parameter, initial.parameters[name],
                                   rtol=0.0, atol=0.0)

    wrong_path = build_contextual_path(
        seed=11, layers=32, spine_sites=(3, 7), read_records=3,
        bank_experts=1, bank_hidden_width=8, bank_lr=1e-2)
    wrong_state = capture_p_state(wrong_path, 0)
    clock_before_wrong = path.experiment.controller.clock.eligible_events
    revision_before_wrong = path.experiment.controller.revision
    wrong = restore_p_state(
        path, wrong_state, restore_clock=False, restore_revision=False,
        restore_expert_steps=False)
    assert wrong["restored"] is True
    assert path.experiment.controller.clock.eligible_events == clock_before_wrong
    assert path.experiment.controller.revision == revision_before_wrong
    wrong_result = evaluate_response_ce(
        path, _p_query(path, input_ids=ids, attention_mask=mask), labels=labels)
    assert not torch.equal(reset_result["logits"], wrong_result["logits"])
    assert committed["useful_learning"] == "unevaluated"
    assert replay["useful_learning"] == "unevaluated"
    assert committed["sealed_evaluation"] == "closed"


def test_site_accounting_keeps_late_site_execution_fixture_only():
    early = site_accounting((3, 7), 32)
    late = site_accounting((23, 31), 32)
    tiny = site_accounting((3, 7), 8)
    assert early["native_comparison"] == \
        "eligible_geometry_native_backend_absent"
    assert early["late_site_native_status"] == "not_a_late_site"
    assert late["native_comparison"] == \
        "eligible_geometry_native_backend_absent"
    assert late["late_site_native_status"] == \
        "geometry_valid_native_backend_absent"
    assert late["native_sites_valid"] is True
    assert late["native_sealed_layers"] == 36
    assert late["engineering_fixture_only"] is True
    assert late["native_execution"] == "not_run"
    assert tiny["native_sites_valid"] is True
    assert tiny["num_layers"] == 8
    assert early["p_backward_suffix_layers_upper_bound"] == 29
    assert late["p_backward_suffix_layers_upper_bound"] == 9
