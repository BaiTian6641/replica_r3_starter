"""Deterministic tiny-CPU contracts for the Spark text memory path.

All decoder tensors come from the local Spark-shaped fixture.  No tokenizer,
model download, native MPS work, or external evidence source is involved.
"""
from __future__ import annotations

import hashlib
import torch

from replica_lab.architecture.spine_memory import SpineMemoryExperiment
from replica_lab.architecture.spine_text import (
    TEXT_PRODUCER_REVISION, SparkTextMemoryPath, VersionedTextProducer,
)
from replica_lab.optimizers import NativeFusedAdamW
from test_spark2_5_adapter import build, ids_and_mask


def _path(seed: int = 7, *, response_lr: float = 1e-3) -> SparkTextMemoryPath:
    adapter, _, _, _ = build(seed=seed, spine_sites=(3, 7), spine_slots=3)
    experiment = SpineMemoryExperiment(
        spec=adapter.spine_spec, core=adapter.extension.spine,
        bank_experts=2, bank_hidden_width=8, seed=seed, read_records=3)
    return SparkTextMemoryPath(adapter, experiment, response_lr=response_lr)


def _open_memory_ports(path: SparkTextMemoryPath) -> None:
    with torch.no_grad():
        for port in path.experiment.core.ports["memory"].values():
            port.output_projection.weight.normal_(0.0, 0.02)


def _labels(ids: torch.Tensor) -> torch.Tensor:
    labels = ids.clone()
    labels[:, :2] = -100
    return labels


def test_text_provenance_versions_and_host_eligibility_precede_semantics():
    path = _path()
    stored = path.observe_text(
        "Alpha appears in the exact source.", record_id="alpha",
        source_lineage={"source": "tiny-fixture", "line": 4})
    assert stored.canonical_text == "Alpha appears in the exact source."
    assert stored.producer_revision == TEXT_PRODUCER_REVISION
    assert stored.provenance["source"] == "tiny-fixture"
    assert stored.provenance["line"] == 4
    assert stored.text_sha256 == stored.provenance["text_sha256"]
    assert path.texts["alpha"].spans[0].token == "Alpha"
    torch.testing.assert_close(
        path.producer.vector(stored.canonical_text),
        VersionedTextProducer(
            representation_version=path.experiment.evidence.representation_version,
            codec_version=path.experiment.evidence.codec_version).vector(stored.canonical_text),
        rtol=0.0, atol=0.0)

    durable = path.observe_text(
        "Beta is rebuildable.", record_id="beta", retain_canonical_text=False,
        durable_ref={
            "locator": "fixture://beta",
            "text_sha256": hashlib.sha256(b"Beta is rebuildable.").hexdigest()})
    assert durable.canonical_text is None
    assert durable.durable_ref["locator"] == "fixture://beta"

    ids, mask = ids_and_mask()
    host_only = path.prepare_query(
        "Alpha appears in the exact source.", ids, mask,
        host_eligible_record_ids=("beta",), semantic_top_k=1)
    assert host_only.eligible_record_ids == ("beta",)
    assert host_only.selected_record_ids == ("beta",)
    semantic = path.prepare_query(
        "Alpha appears in the exact source.", ids, mask, semantic_top_k=1)
    assert semantic.eligible_record_ids == ("alpha", "beta")
    assert semantic.selected_record_ids == ("alpha",)
    assert semantic.semantic_selection.startswith("cosine_top_1_within_host_eligible")
    selected = semantic.context.packet.source_lineage["selected_records"]
    assert selected[0]["producer_revision"] == TEXT_PRODUCER_REVISION
    assert selected[0]["source_epoch"] == 0
    assert selected[0]["representation_version"]
    assert selected[0]["codec_version"]
    assert semantic.context.charges["port_calls"] == 0


def test_default_read_cap_is_query_dependent_not_chronological_truncation():
    path = _path()
    for index, text in enumerate((
            "unrelated zero", "unrelated one", "unrelated two",
            "the exact requested answer")):
        path.observe_text(text, record_id=f"record-{index}", source_epoch=index)
    ids, mask = ids_and_mask()
    query = path.prepare_query("the exact requested answer", ids, mask)
    assert len(query.selected_record_ids) == 3
    assert "record-3" in query.selected_record_ids
    assert query.semantic_selection == "cosine_top_3_within_host_eligible"
    selected = query.context.packet.source_lineage["selected_records"]
    assert {row["record_id"] for row in selected} == set(query.selected_record_ids)


def test_current_question_packet_masks_late_evidence_before_injection():
    path = _path()
    path.observe_text("early evidence", record_id="early", available_at=0)
    path.observe_text("late evidence", record_id="late", available_at=1)
    _open_memory_ports(path)
    ids, mask = ids_and_mask()
    times = (0, 1, 1, 1, 1, 1)
    early_only = path.prepare_query(
        "evidence", ids, mask, token_available_at=times, query_time=1,
        host_eligible_record_ids=("early",))
    both = path.prepare_query(
        "evidence", ids, mask, token_available_at=times, query_time=1)
    late_slot = both.selected_record_ids.index("late")
    assert not bool(both.context.packet.availability[0, 0, late_slot])
    assert bool(both.context.packet.availability[0, 1, late_slot])
    hidden = torch.randn(1, ids.shape[1], 8)
    with torch.no_grad():
        early_delta = early_only.context.delta("memory", 3, hidden)
        both_delta = both.context.delta("memory", 3, hidden)
    torch.testing.assert_close(
        early_delta[:, 0], both_delta[:, 0], rtol=0.0, atol=0.0)
    assert not torch.equal(early_delta[:, 1], both_delta[:, 1])
    # Native query features and codec reads completed before either port call.
    assert both.native_query_passes == 1
    assert both.codec_calls_before_injection == ids.shape[1]


def test_procedure_task_candidates_and_plastic_clock_have_distinct_owners():
    path = _path()
    path.observe_text("independent evaluator exited zero", record_id="receipt")
    procedure = path.add_procedure(
        "repair", applicability=("matching failure",),
        prerequisites=("source is readable",), exceptions=("missing receipt",),
        outcome="repair succeeded", outcome_independently_checked=True,
        outcome_checker="independent-evaluator", source_record_ids=("receipt",))
    assert procedure.outcome_independently_checked is True
    task = path.open_task("task-0", (("repair", "apply repair"),
                                    ("verify", "observe verification")))
    task = path.observe_subgoal_completion(
        task.task_id, "verify", observation="exit code zero", observed_at=3,
        checker="independent-evaluator", evidence_record_ids=("receipt",))
    assert task.observed_completions[0].subgoal_id == "verify"
    assert not task.observed_completions[0].evidence_record_ids == ()
    explanation = path.add_generated_explanation(
        "explanation-0", "perhaps the repair worked", ("receipt",))
    assert explanation.status == "candidate" and explanation.verified is False

    state = path.state_inventory()
    assert state["exact_text_records"] == 1
    assert state["evidence_records"] == 1
    assert state["procedures"] == 1
    assert state["task_workspaces"] == 1
    assert state["candidate_explanations"] == 1
    assert state["plastic_clock_events"] == 0
    assert state["plastic_expert_steps"] == [0, 0]
    assert state["p_optimizer_is_distinct"] is True
    assert set(path.procedures).isdisjoint(path.tasks)


def test_native_null_parity_influence_gradients_and_fused_spine_update():
    path = _path()
    path.observe_text("response memory", record_id="memory")
    path.observe_text("different memory", record_id="other")
    ids, mask = ids_and_mask()
    labels = _labels(ids)
    initial_query = path.prepare_query("response memory", ids, mask)
    initial = path.respond(initial_query, labels=labels, extra_passes=("null",))
    torch.testing.assert_close(
        initial.logits, initial.null_logits["null"], rtol=0.0, atol=0.0)
    assert initial.charges["native_query_passes"] == 1
    assert initial.charges["decoder_passes"] == 2
    assert initial.charges["charged_extra_passes"] == 1

    _open_memory_ports(path)
    influenced_query = path.prepare_query("response memory", ids, mask)
    influenced = path.respond(influenced_query, extra_passes=("null",))
    assert not torch.equal(influenced.logits, influenced.null_logits["null"])
    result = path.response_step(influenced_query, labels=labels)
    assert result["base_unchanged"] is True
    assert result["changed_trainable_names"]
    assert any("memory" in name for name in result["changed_trainable_names"])
    assert result["gradient_audit"]["new_any_nonzero"] is True
    assert result["gradient_audit"]["new_all_finite"] is True
    assert result["gradient_audit"]["original"]["gradient_tensors"] == 0
    component_gradients = {
        "codec": path.experiment.core.codec.output_projection.weight.grad,
        "memory": path.experiment.core.memory_projection.weight.grad,
        "query": path.experiment.core.ports["memory"]["3"].query_projection.weight.grad,
        "port": path.experiment.core.ports["memory"]["3"].output_projection.weight.grad,
    }
    assert all(gradient is not None and bool(torch.count_nonzero(gradient))
               for gradient in component_gradients.values())
    assert result["plastic_clock_events"] == 0
    assert result["plastic_optimizer_steps"] == [0, 0]
    assert isinstance(path.response_optimizer, NativeFusedAdamW)
    assert path.response_optimizer.audit()["fused"] is True
    assert path.response_steps == 1

    assert result["execution_complete"] is True
    assert result["physical_optimizer_steps"] == 1
    assert result["actual_parameter_update"] is True
    assert result["learning_evidence"]["useful_learning"] == "unevaluated"
    assert result["learning_evidence"]["train_fit"] == "unevaluated"
    assert result["changed_expert_names"] == []

def test_query_driven_p_survives_source_removal_without_sharing_response_updates():
    path = _path(seed=11)
    _open_memory_ports(path)
    ids, mask = ids_and_mask()
    labels = _labels(ids)
    teaching = path.prepare_query(
        "learn this question", ids, mask, memory_source="P", plastic_expert_id=0)
    features = path.plastic_features(teaching)
    assert features.shape == (1, ids.shape[1], 256)
    target = torch.linspace(-0.2, 0.2, features.numel()).reshape(features.shape)
    before_teaching = path.experiment.bank.expert(0).down.detach().clone()
    taught = path.teach_query(
        teaching, event_id="query-teaching-0", target=target, expert_id=0)
    assert taught["learning_evidence"]["optimizer_steps"] == 1
    assert not torch.equal(before_teaching, path.experiment.bank.expert(0).down)
    assert path.experiment.controller.clock.eligible_events == 1

    path.observe_text("source which must be removed", record_id="source")
    path.add_generated_explanation("candidate", "unverified source claim", ("source",))
    path.clear_for_p_only()
    assert path.experiment.evidence.pending == 0
    assert not path.texts and not path.explanations
    assert path.experiment.controller.clock.eligible_events == 1
    p_only = path.prepare_query(
        "learn this question", ids, mask, memory_source="P", plastic_expert_id=0)
    assert p_only.selected_record_ids == ()
    assert p_only.context.memory_source == "P"
    response = path.respond(p_only, extra_passes=("null",))
    assert not torch.equal(response.logits, response.null_logits["null"])

    plastic_before_response = {
        name: parameter.detach().clone()
        for name, parameter in path.experiment.bank.named_parameters()}
    result = path.response_step(p_only, labels=labels)
    assert result["plastic_nonzero_gradient_names"]
    assert result["plastic_clock_events"] == 1
    assert result["plastic_optimizer_steps"] == [1, 0]
    for name, parameter in path.experiment.bank.named_parameters():
        torch.testing.assert_close(
            plastic_before_response[name], parameter.detach(), rtol=0.0, atol=0.0)
    assert result["changed_trainable_names"]
    assert path.state_inventory()["exact_text_records"] == 0


def test_e_plus_p_preserves_e_and_adds_differentiable_p_after_source_removal():
    path = _path(seed=13)
    _open_memory_ports(path)
    path.observe_text("combined exact evidence", record_id="evidence", available_at=1)
    ids, mask = ids_and_mask()
    labels = _labels(ids)
    times = (0, 1, 1, 1, 1, 1)
    teaching = path.prepare_query(
        "combined question", ids, mask, token_available_at=times, query_time=1,
        memory_source="E_plus_P", plastic_expert_id=0)
    features = path.plastic_features(teaching)
    target = torch.linspace(-0.1, 0.1, features.numel()).reshape(features.shape)
    path.teach_query(teaching, event_id="combined-teaching", target=target)

    combined = path.prepare_query(
        "combined question", ids, mask, token_available_at=times, query_time=1,
        memory_source="E_plus_P", plastic_expert_id=0)
    e_only = path.prepare_query(
        "combined question", ids, mask, token_available_at=times, query_time=1,
        memory_source="E")
    p_only = path.prepare_query(
        "combined question", ids, mask, token_available_at=times, query_time=1,
        memory_source="P", plastic_expert_id=0)
    assert combined.context.memory_source == "E_plus_P"
    assert combined.selected_record_ids == ("evidence",)
    assert combined.context.slots["memory"].shape[2] == 1
    assert not bool(combined.context.packet.availability[0, 0, 0])
    assert bool(combined.context.packet.availability[0, 1, 0])
    assert combined.context.packet.source_lineage["selected_records"][0][
        "record_id"] == "evidence"
    assert combined.e_codec_calls_before_injection == ids.shape[1]
    assert combined.p_codec_calls_before_injection == ids.shape[1]
    assert combined.p_read_calls_before_injection == 1
    assert combined.codec_calls_before_injection == 2 * ids.shape[1]

    combined_response = path.respond(combined)
    e_response = path.respond(e_only)
    p_response = path.respond(p_only)
    assert not torch.equal(combined_response.logits, e_response.logits)
    assert not torch.equal(combined_response.logits, p_response.logits)
    assert combined_response.charges["memory_read_slots"] == 2
    assert combined_response.charges["e_memory_read_slots"] == 1
    assert combined_response.charges["p_memory_read_slots"] == 1
    assert combined_response.charges["port_calls"] == 4
    assert combined_response.charges["decoder_passes"] == 1
    result = path.response_step(combined, labels=labels)
    assert result["plastic_nonzero_gradient_names"]
    assert result["plastic_clock_events"] == 1
    assert result["plastic_optimizer_steps"] == [1, 0]
    codec_gradient = path.experiment.core.codec.output_projection.weight.grad
    assert codec_gradient is not None and bool(torch.count_nonzero(codec_gradient))

    path.clear_for_p_only()
    removed = path.prepare_query(
        "combined question", ids, mask, memory_source="E_plus_P",
        plastic_expert_id=0)
    assert removed.selected_record_ids == ()
    assert removed.context.memory_source == "E_plus_P"
    assert removed.context.slots["memory"].shape[2] == 0
    removed_response = path.respond(removed, extra_passes=("null",))
    assert not torch.equal(
        removed_response.logits, removed_response.null_logits["null"])
    assert removed_response.charges["e_memory_read_slots"] == 0
    assert removed_response.charges["p_memory_read_slots"] == 1
    assert path.experiment.controller.clock.eligible_events == 1


def test_zero_lr_is_physical_execution_without_parameter_or_learning_claim():
    path = _path(response_lr=0.0)
    _open_memory_ports(path)
    path.observe_text("no-op response", record_id="memory")
    ids, mask = ids_and_mask()
    query = path.prepare_query("no-op response", ids, mask)
    result = path.response_step(query, labels=_labels(ids))
    assert result["execution_complete"] is True
    assert result["physical_optimizer_steps"] == 1
    assert result["actual_parameter_update"] is False
    assert result["changed_trainable_names"] == []
    assert result["changed_expert_names"] == []
    assert result["useful_learning"] == "unevaluated"
    assert result["learning_evidence"]["learning_success"] is None
    assert path.response_steps == 1
    assert path.response_parameter_updates == 0
    # No parameter update means the prepared query is still usable.
    path.respond(query)


def _contract() -> dict:
    return {
        "schema": "replica.spark.spine.v1",
        "spine_schema": "replica.spark.spine.v1",
        "model_identity": "a" * 64,
        "codec_version": "codec-1",
        "representation_version": "repr-1",
    }


def test_durable_checkpoint_restores_text_owners_and_experiment_without_caches(tmp_path):
    path = _path()
    path.observe_text("Alpha durable evidence", record_id="alpha")
    path.observe_text(
        "Beta referenced evidence", record_id="beta", retain_canonical_text=False,
        durable_ref={
            "locator": "fixture://beta",
            "text_sha256": hashlib.sha256(b"Beta referenced evidence").hexdigest()})
    path.add_procedure(
        "procedure", applicability=("Alpha applies",), prerequisites=(),
        exceptions=(), outcome="checked outcome", outcome_independently_checked=True,
        outcome_checker="independent", source_record_ids=("alpha",))
    path.open_task("task", (("observe", "observe Alpha"),))
    path.observe_subgoal_completion(
        "task", "observe", observation="Alpha observed", observed_at=1,
        checker="independent", evidence_record_ids=("alpha",))
    path.add_generated_explanation("candidate", "unverified explanation", ("alpha",))
    ids, mask = ids_and_mask()
    teaching = path.prepare_query(
        "checkpoint P", ids, mask, memory_source="P", plastic_expert_id=0)
    target = torch.ones_like(path.plastic_features(teaching)) * 0.05
    path.teach_query(teaching, event_id="checkpoint-teaching", target=target)
    receipt = path.checkpoint(tmp_path, "state.pt", run_contract=_contract())
    assert receipt["derived_caches_saved"] is False

    fresh = _path()
    restored = fresh.restore(tmp_path, "state.pt", run_contract=_contract())
    assert restored["derived_caches_restored"] is False
    assert restored["text_records"] == 2
    assert fresh.texts == path.texts
    assert fresh.procedures == path.procedures
    assert fresh.tasks == path.tasks
    assert fresh.explanations == path.explanations
    assert fresh.experiment.evidence.pending == 2
    assert fresh.experiment.controller.clock.eligible_events == 1
    assert fresh.experiment.accounting()["transient"]["live_contexts"] == 0
    for name, parameter in path.experiment.bank.named_parameters():
        torch.testing.assert_close(
            dict(fresh.experiment.bank.named_parameters())[name], parameter,
            rtol=0.0, atol=0.0)
    query = fresh.prepare_query("Alpha durable evidence", ids, mask, semantic_top_k=1)
    assert query.selected_record_ids == ("alpha",)
    assert fresh.experiment.accounting()["transient"]["live_contexts"] == 0
    assert query.context.packet.source_lineage["selected_records"][0]["source_epoch"] == 0
