"""Regression tests for reader-state isolation and control semantics."""
from __future__ import annotations

import copy

import pytest
import torch

from replica_lab.persistent_learner_runner import (
    ARMS, ArmView, FitState, Protocol, PreparedDataset, Refused, WorkLedger,
    _apply_dynamic, _dynamic_payload, _fit_query, _new_path, _observe_rows,
    _port_output_snapshot, _prepare_query, _restore_port_output,
    build_arm_views, build_cpu_backend, _open_tiered, _tiered_admit_rows,
    _tiered_sync_procedures,
)
from replica_lab.p1_response_learning import direct_response_ce_step

CONFIG = "configs/spark_persistent_learner_protocol_v2.json"
PREPARED = "data/spark_persistent_learner_v2"


def _protocol_dataset_backend():
    protocol = Protocol.load(CONFIG)
    dataset = PreparedDataset.load(
        PREPARED, expected_sha256=protocol.config["dataset"]["manifest_sha256"],
        protocol_sha256=protocol.sha256)
    backend = build_cpu_backend(protocol, dataset, spine_sites=(3, 7))
    return protocol, dataset, backend


def _trained_path_and_fit(protocol, dataset, backend):
    path = _new_path(protocol, backend.adapter)
    nonlearning = _new_path(protocol, backend.adapter)
    replay = _new_path(protocol, backend.adapter)
    pre_port = _port_output_snapshot(path)
    pre_plastic = copy.deepcopy(path.experiment.controller.checkpoint_payload())
    observations = {str(row["record_id"]): row for row in dataset.observations()}
    event = dataset.fit_events()[0]
    _observe_rows(path, [observations[str(record_id)]
                         for record_id in event["observation_record_ids"]])
    query, labels, _ = _fit_query(
        path, backend, event, observations,
        WorkLedger(protocol, context_tokens=256, max_new_tokens=2))
    result = direct_response_ce_step(
        path, query, labels, event_id="control-test-fit", expert_id=0,
        port_optimizer=path.response_optimizer)
    trained_port = _port_output_snapshot(path)
    fit = FitState(
        path=path, nonlearning_path=nonlearning, replay_path=replay,
        pre_plastic=pre_plastic,
        trained_plastic=path.experiment.controller.checkpoint_payload(),
        receipt={}, run_contract={}, pre_port_state=pre_port,
        trained_port_state=trained_port)
    return fit, result, pre_port, trained_port


def test_nonlearning_dynamic_restore_does_not_inherit_trained_reader():
    protocol, dataset, backend = _protocol_dataset_backend()
    fit, result, pre_port, trained_port = _trained_path_and_fit(
        protocol, dataset, backend)
    assert result["optimizer_ownership"]["port_optimizer_stepped"] is True
    assert any(not torch.equal(pre_port[name], trained_port[name])
               for name in pre_port)
    _restore_port_output(fit.nonlearning_path.adapter, pre_port)
    payload = _dynamic_payload(fit.nonlearning_path)
    assert payload["format"].endswith("dynamic_state.v3")
    assert set(payload["port_state"]) == set(pre_port)
    _restore_port_output(fit.path.adapter, trained_port)
    _apply_dynamic(fit.nonlearning_path, payload)
    restored = _port_output_snapshot(fit.nonlearning_path)
    assert all(torch.equal(restored[name], pre_port[name]) for name in pre_port)
    _restore_port_output(fit.path.adapter, trained_port)
    assert all(torch.equal(_port_output_snapshot(fit.path)[name], trained_port[name])
               for name in trained_port)


def test_partial_dynamic_port_authority_is_rejected_before_mutation():
    protocol, dataset, backend = _protocol_dataset_backend()
    path = _new_path(protocol, backend.adapter)
    payload = _dynamic_payload(path)
    before = _port_output_snapshot(path)
    payload["port_state"] = dict(payload["port_state"])
    payload["port_state"].pop(next(iter(payload["port_state"])))
    with pytest.raises(Refused, match="identities differ"):
        _apply_dynamic(path, payload)
    after = _port_output_snapshot(path)
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_all_arm_views_restore_declared_state_and_untouched_parity(tmp_path):
    protocol, dataset, backend = _protocol_dataset_backend()
    fit, _, pre_port, trained_port = _trained_path_and_fit(protocol, dataset, backend)
    tiered = _open_tiered(protocol, tmp_path)
    _tiered_admit_rows(tiered, dataset.observations())
    _tiered_sync_procedures(tiered, dataset.procedures())
    views = build_arm_views(protocol, fit, dataset, tiered=tiered)
    case = dataset._rows("development.jsonl")[0]
    for arm, view in views.items():
        prompt, _ = _prompt_for_view(view, case, dataset)
        query, _ = _prepare_query(view, backend, prompt, case,
                                  WorkLedger(protocol, context_tokens=256, max_new_tokens=2))
        actual = _port_output_snapshot(view.path)
        assert all(torch.equal(actual[name], view.port_state[name]) for name in actual), arm
        if view.untouched_baseline:
            response = view.path.respond(query, extra_passes=("null",))
            assert torch.equal(response.logits, response.null_logits["null"])
            assert query.context.enabled_branches == ()
    assert views["full_eligible_context"].untouched_baseline is True
    assert views["reader_trained_E_only"].frozen_trained_reader is True
    assert views["no_update"].frozen_trained_reader is True
    assert views["reset_P"].frozen_trained_reader is True
    assert views["matched_replay"].frozen_trained_reader is False
    for left, right in (("source_free_P_only", "no_update"),
                        ("E_plus_P", "reset_P")):
        assert all(torch.equal(views[left].port_state[name], views[right].port_state[name])
                   for name in views[left].port_state)
    assert any(not torch.equal(views["reader_trained_E_only"].port_state[name],
                               views["spine_E_only"].port_state[name])
               for name in trained_port)
    tiered.close()
def _prompt_for_view(view, case, dataset):
    # Keep the test's prompt construction identical to production.
    from replica_lab.persistent_learner_runner import _prompt_for
    return _prompt_for(view, case, dataset)


def test_learning_metrics_separate_reader_p_and_retrieval_benefit():
    rows = []
    for arm in ARMS:
        rows.append({"arm_id": arm, "expected": "ready", "streams": ["new_information"],
                     "older_rule_retention": False,
                     "score": {"matched": arm in {"reader_trained_E_only", "E_plus_P"}}})
    metrics = __import__("replica_lab.persistent_learner_runner", fromlist=["_learning_metrics"])._learning_metrics(
        rows, {"optimizer_steps": 8, "selected_P_optimizer_steps": 4,
               "response_port_optimizer_steps": 4}, backend=type("B", (), {"label": "cpu_contextual_fixture"})())
    assert metrics["reader_learning"]["accuracy_delta"] == 1.0
    assert metrics["P_learning"]["source_free_delta"] == 0.0
    assert metrics["P_learning"]["with_evidence_delta"] == 1.0
    assert metrics["retrieval_benefit"]["accuracy_delta"] == 1.0
    assert metrics["matched_replay"]["replay_training"] is False
    assert metrics["control_semantics"]["untouched_baseline"] == "full_eligible_context"
