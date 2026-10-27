"""CPU contracts for the nested-text one-clock plastic baseline.

These tests build tiny deterministic adapters/banks.  They do not load
MiniCPM, run MPS, train a capability, or produce native evidence.  They cover
the mechanical properties the native runner relies on: exactly-once
transaction/chunk behavior, null-then-influence read parity, source-free P-only
export/load, restart equality, personality/spine parity, access order and
fail-closed stale bindings.
"""

from __future__ import annotations

import hashlib
import copy
import functools
import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from replica_lab.architecture.contracts import (DifferentiablePlasticRead,
                                                EvidencePin, MemoryRead,
                                                PlasticRead, PlasticSpec,
                                                SenseFrame, DecisionFrame)
from replica_lab.architecture.plastic import (PlasticKnowledgePort,
                                              fixed_projection,
                                              projection_digest)
from replica_lab.nested_learning import (DuplicateObservation, ExpertBank,
                                         NestedLearningError, ObservationCoordinator,
                                         Observer, PersonalityIdentity,
                                         PlasticLearningController, RouteControl)
from replica_lab.nested_text import (NestedTextError, NestedTextLearner,
                                     P_ONLY_FORMAT, TextAccessOrder,
                                     TextProtocol, TextTeachingEvent,
                                     evaluate_dev_gate,
                                     initialize_null_plastic, prompt_route_key,
                                     validate_p_only_payload)
from replica_lab.nested_continuation import canonical_state

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 4
HIDDEN_WIDTH = 5
HIDDEN_SIZE = 8
EXPERTS = 3


class FakeAdapter:
    """Deterministic site hidden states; no native model or parameters."""

    hidden_size = HIDDEN_SIZE

    def encode_sites(self, input_ids, valid, sites=None):
        ids = torch.as_tensor(input_ids).long()
        mask = torch.as_tensor(valid).bool()
        result = {}
        for site in tuple(sites):
            values = torch.zeros(1, ids.shape[1], HIDDEN_SIZE, dtype=torch.float32)
            for position in range(ids.shape[1]):
                token = int(ids[0, position])
                base = torch.arange(HIDDEN_SIZE, dtype=torch.float32) * 0.01
                values[0, position] = torch.sin(base * (token + 1 + site)) + 0.1 * site
            values = values.masked_fill(~mask.unsqueeze(-1), 0)
            result[int(site)] = values
        return result


def projection_for(seed: int = 1) -> torch.Tensor:
    return fixed_projection(HIDDEN_SIZE, WIDTH, seed)


def protocol(*, world: str = "world-a", scope: str = "world-a/text",
             seed: int = 1, route_seed: int = 0,
             theta_revision: str = "rev",
             parameter_stamp: str = "", hidden_size: int = HIDDEN_SIZE,
             feature_sites: tuple[int, ...] = (1, 2)) -> TextProtocol:
    projection = fixed_projection(hidden_size, WIDTH, seed)
    digest = projection_digest(projection)
    return TextProtocol(
        world_uid=world, scope=scope, expert_count=EXPERTS, width=WIDTH,
        hidden_width=HIDDEN_WIDTH, hidden_size=hidden_size,
        feature_sites=feature_sites, target_projection_seed=seed,
        target_projection_sha256=digest, read_gain=0.5,
        read_path_sha256=digest, route_mode="fixed", route_seed=route_seed,
        theta_revision=theta_revision, theta_lock_sha256="",
        parameter_stamp_sha256=parameter_stamp, tokenizer_sha256="")


def bank(seed: int = 7, *, lr: float = 0.05) -> ExpertBank:
    return ExpertBank(expert_count=EXPERTS, width=WIDTH,
                      hidden_width=HIDDEN_WIDTH, seed=seed, lr=lr)


def owners(protocol_value: TextProtocol | None = None, *, observer=None,
           spine=None, personality=None):
    protocol_value = protocol_value or protocol()
    personality = personality or PersonalityIdentity.default()
    expert_bank = bank()
    coordinator = ObservationCoordinator(
        scope=protocol_value.scope, width=WIDTH, seed=0,
        personality=personality, observer=observer, spine=spine)
    controller = PlasticLearningController(
        expert_bank, scope=protocol_value.scope, personality=personality,
        coordinator=coordinator, max_steps=2, total_step_cap=4,
        clock_period=protocol_value.clock_period)
    learner = NestedTextLearner(FakeAdapter(), bank=expert_bank,
                                protocol=protocol_value,
                                controller=controller,
                                coordinator=coordinator,
                                personality=personality,
                                tokenizer_sha256="")
    return learner, expert_bank, controller, coordinator


def prompt(*ids: int) -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.tensor([list(ids)], dtype=torch.long)
    return value, torch.ones_like(value, dtype=torch.bool)


def event(event_id: str, protocol_value: TextProtocol, prompt_ids: torch.Tensor,
          *, eligible: bool = True, source: str = "feedback",
          available: float = 1.0, text: str = "prompt",
          role: str | None = None, response: str = "answer") -> TextTeachingEvent:
    import hashlib
    return TextTeachingEvent(
        event_id=event_id, world_uid=protocol_value.world_uid,
        scope=protocol_value.scope,
        split_role=role or ("teaching" if eligible else "development"),
        teaching_or_query="teaching" if eligible else "future_query",
        prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
        response_sha256=hashlib.sha256(response.encode()).hexdigest(),
        target_available_at=available, causal_cutoff=available + 1,
        eligible=eligible, source=source, verified=True,
        route_key=tuple(float(item) for item in prompt_route_key(prompt_ids)),
        concept_family="test", contamination_group="test")


def assert_state_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.equal(torch.as_tensor(left), torch.as_tensor(right))
    elif isinstance(left, dict) and isinstance(right, dict):
        assert set(left) == set(right)
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_state_equal(first, second)
    else:
        assert left == right


def teach(learner: NestedTextLearner, value: TextTeachingEvent,
          prompt_ids: torch.Tensor, prompt_mask: torch.Tensor) -> None:
    full = torch.cat((prompt_ids, torch.tensor([[9, 10]], dtype=torch.long)), dim=1)
    mask = torch.ones_like(full, dtype=torch.bool)
    response = torch.zeros_like(mask)
    response[:, -2:] = True
    learner.learn_event(value, prompt_ids=prompt_ids, prompt_mask=prompt_mask,
                        input_ids=full, attention_mask=mask,
                        response_mask=response)


def test_protocol_rejects_projection_mismatch():
    with pytest.raises(NestedTextError, match="target projection"):
        TextProtocol(world_uid="w", scope="w/s", expert_count=EXPERTS,
                     width=WIDTH, hidden_width=HIDDEN_WIDTH,
                     hidden_size=HIDDEN_SIZE, feature_sites=(1, 2),
                     target_projection_seed=1,
                     target_projection_sha256="0" * 64,
                     read_path_sha256="0" * 64)


def test_plastic_port_exact_null_and_later_influence():
    decoder = projection_for()
    port = PlasticKnowledgePort(HIDDEN_SIZE, WIDTH, decoder=decoder, gain=0.5)
    hidden = torch.ones(1, 3, HIDDEN_SIZE)
    assert torch.equal(port(hidden, torch.zeros(1, WIDTH)), torch.zeros_like(hidden))
    slot = torch.ones(1, WIDTH)
    delta = port(hidden, slot)
    assert bool(torch.isfinite(delta).all()) and bool(delta.abs().sum() > 0)
    assert torch.equal(delta[0, 0], delta[0, 1])


def test_initial_plastic_read_is_null_then_nonzero():
    learner, expert_bank, controller, _ = owners()
    ids, mask = prompt(5, 6, 7)
    read = learner.plastic_read(ids, mask)
    assert torch.equal(read.slot, torch.zeros(1, WIDTH))
    features = learner.prompt_features(ids, mask)
    stacked = torch.stack([features[site] for site in learner.protocol.feature_sites])
    value = event("e1", learner.protocol, ids, available=1.0)
    teach(learner, value, ids, mask)
    assert controller.clock.eligible_events == 1
    after = learner.plastic_read(ids, mask)
    assert after.expert_id == read.expert_id
    assert bool(after.slot.abs().sum() > 0), "a committed transaction must influence P"
    assert stacked.shape == (2, WIDTH)


def test_learn_event_exactly_once_and_retry():
    learner, expert_bank, controller, coordinator = owners()
    ids, mask = prompt(1, 2, 3)
    value = event("e1", learner.protocol, ids, available=1.0)
    teach(learner, value, ids, mask)
    assert controller.clock.eligible_events == 1
    assert controller.revision == 1
    assert coordinator.transition_count == 1
    assert len(coordinator.checkpoint_payload()["observations"]) == 1
    parameters = expert_bank.parameter_snapshot(range(EXPERTS))
    moments = expert_bank.optimizer_state_snapshot(range(EXPERTS))
    retry = learner.learn_event(
        value, prompt_ids=ids, prompt_mask=mask,
        input_ids=torch.cat((ids, torch.tensor([[9, 10]])), dim=1),
        attention_mask=torch.ones(1, 5, dtype=torch.bool),
        response_mask=torch.tensor([[False, False, False, True, True]]))
    assert retry.transaction is not None and retry.transaction.retried
    assert retry.chunk.retried and retry.retried
    assert controller.clock.eligible_events == 1 and controller.revision == 1
    assert coordinator.transition_count == 1
    for expert_id in range(EXPERTS):
        for name, parameter in expert_bank.expert(expert_id).named_parameters():
            assert torch.equal(parameter, parameters[expert_id][name])
    assert expert_bank.optimizer_state_snapshot(range(EXPERTS)).keys() == moments.keys()


def test_changed_duplicate_fails_closed():
    learner, _, controller, _ = owners()
    ids, mask = prompt(1, 2, 3)
    teach(learner, event("e1", learner.protocol, ids, available=1.0), ids, mask)
    with pytest.raises(DuplicateObservation):
        teach(learner, event("e1", learner.protocol, ids, available=1.0,
                             text="different prompt"), ids, mask)
    assert controller.clock.eligible_events == 1


def test_ineligible_event_has_zero_transactions_and_transitions():
    learner, _, controller, coordinator = owners()
    ids, mask = prompt(4, 5)
    outcome = learner.learn_event(
        event("q1", learner.protocol, ids, eligible=False, source="observation"),
        prompt_ids=ids, prompt_mask=mask)
    assert outcome.transaction is None
    assert outcome.chunk.transition_count == 0
    assert controller.clock.eligible_events == 0 and controller.revision == 0
    assert coordinator.transition_count == 0
    assert learner.accounting["ineligible_events"] == 1


def test_reader_count_does_not_change_transition():
    def run(readers):
        protocol_value = protocol()
        personality = PersonalityIdentity.default()
        coordinator = ObservationCoordinator(scope=protocol_value.scope,
                                             width=WIDTH, seed=0,
                                             personality=personality)
        from replica_lab.nested_learning import Observation
        observation = Observation(
            event_id="e1", scope=protocol_value.scope,
            key=tuple([1.0, 0.0, 0.0, 0.0]),
            value=tuple([0.5, 0.0, 0.0, 0.0]),
            available_at=1.0, eligible=True, source="feedback")
        return coordinator.seal_observation_chunk(
            chunk_id="c1", observations=(observation,),
            reader_ids=tuple(readers), sealed_at=1.0)
    one = run(("13",))
    many = run(("13", "27", "41"))
    assert one.transition_count == many.transition_count == 1
    assert one.aggregate_key == many.aggregate_key
    assert one.aggregate_value == many.aggregate_value


def test_p_only_export_has_no_learning_state_and_round_trips():
    learner, _, _, _ = owners()
    ids, mask = prompt(1, 2, 3)
    teach(learner, event("e1", learner.protocol, ids, available=1.0), ids, mask)
    payload = learner.p_only_export()
    assert payload["format"] == P_ONLY_FORMAT
    assert set(payload) == {
        "format", "scope", "theta", "protocol_sha256", "bank", "route",
        "feature_producer", "read_path", "expert_steps", "parameters"}
    for forbidden in ("optimizer", "dedup", "clock", "revision",
                      "coordinator", "observations", "sources", "moments"):
        assert forbidden not in payload
    primary_read = learner.plastic_read(ids, mask)
    fresh_bank = bank()
    fresh = NestedTextLearner(FakeAdapter(), bank=fresh_bank,
                              protocol=learner.protocol)
    assert torch.equal(
        plastic_read_slot(fresh, ids, mask), torch.zeros(1, WIDTH))
    fresh.load_p_only(payload)
    assert torch.equal(fresh.plastic_read(ids, mask).slot, primary_read.slot)
    assert fresh.plastic_read(ids, mask).expert_id == primary_read.expert_id
    assert fresh.controller is None and fresh.coordinator is None


def plastic_read_slot(learner, ids, mask):
    return learner.plastic_read(ids, mask).slot


@pytest.mark.parametrize("mutate,message", [
    (lambda p: p.update(scope="other/scope"), "scope"),
    (lambda p: p["theta"].update(stamp_sha256="0" * 64), "theta"),
    (lambda p: p.update(protocol_sha256="0" * 64), "protocol"),
    (lambda p: p["route"].update(seed=9), "route"),
    (lambda p: p["feature_producer"].update(target_projection_sha256="0" * 64),
     "feature"),
    (lambda p: p["read_path"].update(gain=0.9), "read path"),
    (lambda p: p.update(dedup={"records": []}), "forbidden"),
    (lambda p: p.update(optimizer={"state": {}}), "forbidden"),
])
def test_p_only_load_rejects_tampering(mutate, message):
    learner, _, _, _ = owners()
    payload = copy.deepcopy(learner.p_only_export())
    mutate(payload)
    with pytest.raises(NestedTextError, match=message):
        validate_p_only_payload(payload, protocol=learner.protocol)


def test_p_only_load_is_forbidden_on_learning_learner():
    learner, _, _, _ = owners()
    with pytest.raises(NestedTextError, match="P-only load is forbidden"):
        learner.load_p_only(learner.p_only_export())


def test_partial_learning_state_is_not_exported():
    learner, expert_bank, controller, _ = owners()
    ids, mask = prompt(1, 2, 3)
    teach(learner, event("e1", learner.protocol, ids, available=1.0), ids, mask)
    payload = learner.p_only_export()
    assert payload["expert_steps"] == expert_bank.expert_steps
    assert "parameters" in payload and len(payload["parameters"]) == EXPERTS * 3
    checkpoint = learner.checkpoint_payload()
    assert "optimizer" in checkpoint["controller"]["bank"]
    assert "dedup" in checkpoint["controller"]
    assert "observation" in checkpoint["controller"]


def test_full_checkpoint_restart_equals_uninterrupted(tmp_path):
    learner, expert_bank, controller, coordinator = owners()
    ids1, mask1 = prompt(1, 2, 3)
    ids2, mask2 = prompt(4, 5, 6)
    ids3, mask3 = prompt(7, 8, 9)
    teach(learner, event("e1", learner.protocol, ids1, available=1.0), ids1, mask1)
    teach(learner, event("e2", learner.protocol, ids2, available=2.0), ids2, mask2)
    path = learner.save_checkpoint(tmp_path / "full.pt")
    teach(learner, event("e3", learner.protocol, ids3, available=3.0), ids3, mask3)

    restored_bank = bank(lr=0.05)
    restored_coordinator = ObservationCoordinator(
        scope=learner.protocol.scope, width=WIDTH, personality=learner.personality)
    restored_controller = PlasticLearningController(
        restored_bank, scope=learner.protocol.scope,
        personality=learner.personality, coordinator=restored_coordinator,
        max_steps=2, total_step_cap=4,
        clock_period=learner.protocol.clock_period)
    restored = NestedTextLearner(FakeAdapter(), bank=restored_bank,
                                 protocol=learner.protocol,
                                 controller=restored_controller,
                                 coordinator=restored_coordinator,
                                 personality=learner.personality,
                                 tokenizer_sha256="")
    restored.load_checkpoint_file(path)
    assert restored.controller.clock.eligible_events == 2
    assert restored.controller.revision == 2
    teach(restored, event("e3", learner.protocol, ids3, available=3.0), ids3, mask3)
    assert restored.controller.clock == controller.clock
    assert restored.controller.dedup.event_ids == controller.dedup.event_ids
    assert restored.controller.revision == controller.revision
    for expert_id in range(EXPERTS):
        for name, parameter in expert_bank.expert(expert_id).named_parameters():
            assert torch.equal(parameter,
                               restored_bank.expert(expert_id).get_parameter(name))
        assert (expert_bank.optimizer_state_for_expert(expert_id).keys()
                == restored_bank.optimizer_state_for_expert(expert_id).keys())
        assert_state_equal(expert_bank.optimizer_state_for_expert(expert_id),
                           restored_bank.optimizer_state_for_expert(expert_id))
    assert torch.equal(coordinator.fast_state, restored_coordinator.fast_state)
    assert coordinator.transition_count == restored_coordinator.transition_count
    assert expert_bank.expert_steps == restored_bank.expert_steps


def test_checkpoint_rejects_stale_protocol_and_personality(tmp_path):
    learner, _, _, _ = owners()
    ids, mask = prompt(1, 2, 3)
    teach(learner, event("e1", learner.protocol, ids, available=1.0), ids, mask)
    path = learner.save_checkpoint(tmp_path / "full.pt")
    other_protocol = protocol(theta_revision="other")
    _, _, _, _ = owners(other_protocol)
    with pytest.raises(NestedTextError, match="protocol"):
        other = NestedTextLearner(FakeAdapter(), bank=bank(),
                                  protocol=other_protocol)
        other.controller = learner.controller
        other.load_checkpoint_file(path)
    with pytest.raises(NestedTextError, match="personality"):
        personality = PersonalityIdentity.default(snapshot="different")
        coordinator = ObservationCoordinator(
            scope=learner.protocol.scope, width=WIDTH, personality=personality)
        controller = PlasticLearningController(
            bank(), scope=learner.protocol.scope, personality=personality,
            coordinator=coordinator, max_steps=2, total_step_cap=4,
            clock_period=learner.protocol.clock_period)
        other = NestedTextLearner(FakeAdapter(), bank=controller.bank,
                                  protocol=learner.protocol,
                                  controller=controller,
                                  coordinator=coordinator,
                                  personality=personality,
                                  tokenizer_sha256="")
        other.load_checkpoint_file(path)


def test_personality_spine_parity_and_read_only_observer():
    observer = Observer()
    spine = Observer()
    learner, _, controller, coordinator = owners(observer=observer, spine=spine)
    identity_before = learner.personality.key()
    ids, mask = prompt(2, 3, 4)
    teach(learner, event("e1", learner.protocol, ids, available=1.0), ids, mask)
    assert learner.personality.key() == identity_before
    assert controller.personality.key() == identity_before
    assert len(observer.reports) == len(spine.reports) == 1
    assert observer.reports[0]["view_id"] == spine.reports[0]["view_id"]
    assert observer.reports[0]["readonly"] is True
    assert not any(isinstance(report, dict) and report.get("write")
                   for report in observer.reports)


def test_access_order_blocks_failed_development_and_double_heldout():
    order = TextAccessOrder()
    order.access("teaching")
    order.record_dev_gate({"all_declared_gates_passed": False})
    with pytest.raises(NestedTextError, match="development gate"):
        order.open_heldout()
    with pytest.raises(NestedTextError, match="sealed after failure"):
        order.access("sealed_final")
    success = TextAccessOrder()
    success.access("teaching")
    success.record_dev_gate({"all_declared_gates_passed": True})
    success.open_heldout()
    with pytest.raises(NestedTextError, match="already open"):
        success.access("sealed_final")
    with pytest.raises(NestedTextError, match="already open"):
        success.open_heldout()


def test_route_key_is_prompt_only_and_deterministic():
    ids = torch.tensor([[5, 6, 7]])
    assert torch.equal(prompt_route_key(ids), prompt_route_key(ids.clone()))
    assert not torch.equal(prompt_route_key(ids),
                           prompt_route_key(torch.tensor([[5, 6, 8]])))
    control = RouteControl("fixed", EXPERTS, 0)
    assert control.route(prompt_route_key(ids)) < EXPERTS
    first = event("e1", protocol(world="a"), ids, available=1.0)
    second = event("e1", protocol(world="b", scope="b/text"), ids, available=1.0)
    assert first.route_key == second.route_key


def test_event_rejects_unverified_or_nonteaching_eligibility():
    ids = torch.tensor([[1, 2]])
    base = dict(
        event_id="e1", world_uid="w", scope="w/s", split_role="teaching",
        teaching_or_query="teaching",
        prompt_sha256="0" * 64, response_sha256="1" * 64,
        target_available_at=1.0, causal_cutoff=2.0, eligible=True,
        source="feedback", verified=False,
        route_key=tuple(float(item) for item in prompt_route_key(ids)))
    with pytest.raises(NestedTextError, match="verified"):
        TextTeachingEvent(**base)
    with pytest.raises(NestedTextError, match="teaching-role"):
        TextTeachingEvent(**{**base, "verified": True, "split_role": "sealed_final"})


def test_decision_frame_plastic_branch_requires_read():
    read = PlasticRead(
        slot=torch.zeros(1, WIDTH), expert_id=0, producer="p", revision=0,
        route_mode="fixed", route_permutation=(0, 1, 2),
        read_path_sha256="0" * 64)
    evidence = EvidencePin("ledger", 0, "subject", "branch", "purpose", 0,
                           "live", None, ())
    memory = MemoryRead((), torch.zeros(1, 0, 4, WIDTH),
                        torch.zeros(1, 0, 4, dtype=torch.bool), "m")
    ids = torch.tensor([[1, 2]])
    sense = SenseFrame((), ids, torch.ones_like(ids, dtype=torch.bool), "s")
    frame = DecisionFrame("d", evidence, memory, sense, "manifest", (),
                          (), enabled_branches=("plastic",), plastic=read)
    assert frame.plastic is read
    with pytest.raises(ValueError, match="Plastic branch"):
        DecisionFrame("d", evidence, memory, sense, "manifest", (),
                      (), enabled_branches=("plastic",))
    with pytest.raises(ValueError, match="detached"):
        PlasticRead(torch.zeros(1, WIDTH, requires_grad=True), 0, "p", 0,
                    "fixed", (0, 1, 2), "0" * 64)


def test_null_initialization_requires_untaught_bank():
    expert_bank = bank()
    initialize_null_plastic(expert_bank)
    assert not bool(expert_bank.expert(0).down.any())
    expert_bank.expert_steps[0] = 1
    with pytest.raises(NestedLearningError, match="zero expert steps"):
        initialize_null_plastic(expert_bank)


def test_real_adapter_plastic_branch_is_null_then_influences_logits():
    from architecture_helpers import tiny_adapter
    adapter = tiny_adapter().eval()
    adapter.install_plastic(PlasticSpec(experts=EXPERTS, width=WIDTH,
                                        hidden_width=HIDDEN_WIDTH,
                                        sites=(0, 2), route_mode="fixed",
                                        route_seed=0, projection_seed=1,
                                        read_gain=0.5))
    protocol_value = protocol(seed=1, hidden_size=16, feature_sites=(0, 2))
    expert_bank = bank()
    coordinator = ObservationCoordinator(scope=protocol_value.scope,
                                         width=WIDTH)
    controller = PlasticLearningController(
        expert_bank, scope=protocol_value.scope, coordinator=coordinator,
        max_steps=2, total_step_cap=4,
        clock_period=protocol_value.clock_period)
    learner = NestedTextLearner(adapter, bank=expert_bank,
                                protocol=protocol_value,
                                controller=controller,
                                coordinator=coordinator,
                                tokenizer_sha256="")
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    read = learner.plastic_read(ids, mask)
    assert torch.equal(read.slot, torch.zeros(1, WIDTH))

    def frame(plastic):
        memory = MemoryRead(
            (), adapter.base.get_input_embeddings().weight.new_zeros(
                1, 0, adapter.ports.slots_per_unit, adapter.ports.memory_width),
            torch.zeros(1, 0, adapter.ports.slots_per_unit, dtype=torch.bool),
            adapter.memory_producer_key())
        evidence = EvidencePin("ledger", 0, "subject", "branch", "purpose", 0,
                               "live", None, ())
        sense = adapter.sense(ids, mask)
        if plastic is None:
            return adapter.make_frame("d", evidence, memory, sense,
                                      enabled_ports=(), plastic=None,
                                      enabled_branches=("memory",))
        return adapter.make_frame("d", evidence, memory, sense,
                                  enabled_ports=(), plastic=plastic,
                                  enabled_branches=("plastic",))

    with torch.no_grad():
        null_logits = adapter.forward_with_frame(ids, frame(read)).logits
        baseline = adapter.forward_with_frame(ids, frame(None)).logits
    torch.testing.assert_close(null_logits, baseline, atol=1e-6, rtol=0)
    teach(learner, event("e1", protocol_value, ids, available=1.0), ids, mask)
    assert learner.controller.clock.eligible_events == 1
    influenced = learner.plastic_read(ids, mask)
    assert bool(influenced.slot.abs().sum() > 0)
    with torch.no_grad():
        changed = adapter.forward_with_frame(ids, frame(influenced)).logits
    assert not torch.equal(changed, baseline)


def test_session_p_only_request_has_no_source_state(tmp_path):
    from architecture_helpers import tiny_adapter
    from replica_lab.architecture.contracts import TaskGraph
    from replica_lab.memory_system.ledger import EligibilityScope, RecordLedger
    from replica_lab.memory_system.session import MemorySession

    adapter = tiny_adapter().eval()
    adapter.install_plastic(PlasticSpec(experts=EXPERTS, width=WIDTH,
                                        hidden_width=HIDDEN_WIDTH,
                                        sites=(0, 2), route_mode="fixed",
                                        route_seed=0, projection_seed=1,
                                        read_gain=0.5))
    protocol_value = protocol(seed=1, hidden_size=16, feature_sites=(0, 2))
    learner = NestedTextLearner(adapter, bank=bank(),
                                protocol=protocol_value)
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    ledger = RecordLedger(tmp_path / "ledger.sqlite")
    session = MemorySession(adapter, None, ledger)
    scope = EligibilityScope(subject_id="subject", branch_id="branch",
                             purpose="purpose", as_of=10)
    read = learner.plastic_read(ids, mask)
    task = TaskGraph(task_id="task", goal="query", revision=0)
    request = session.prepare(
        prompt_ids=ids, prompt_mask=mask, query="query", scope=scope,
        task=task, training=False, enabled_ports=(), core_impl="none",
        enabled_branches=("plastic",), plastic=read,
        identity=None, identity_enabled=False)
    assert request.frame.plastic is read
    assert request.frame.branch_enabled("plastic")
    assert not request.frame.branch_enabled("memory")
    assert request.frame.memory.spans == ()
    assert request.selection.records == ()
    assert ledger.snapshot(scope).records == ()
    assert learner.controller is None and learner.coordinator is None
    with torch.no_grad():
        ids_out = session.generate(request, max_new_tokens=1)
    assert ids_out.shape[1] >= ids.shape[1]




def test_response_loss_reaches_prompt_sites_and_selected_expert(tmp_path):
    from architecture_helpers import tiny_adapter
    from replica_lab.architecture.contracts import TaskGraph
    from replica_lab.memory_system.ledger import EligibilityScope, RecordLedger
    from replica_lab.memory_system.session import MemorySession

    adapter = tiny_adapter().eval()
    adapter.install_plastic(PlasticSpec(experts=EXPERTS, width=WIDTH,
                                        hidden_width=HIDDEN_WIDTH,
                                        sites=(0, 2), route_mode="fixed",
                                        route_seed=0, projection_seed=1,
                                        read_gain=0.5))
    protocol_value = protocol(seed=1, hidden_size=16, feature_sites=(0, 2))
    expert_bank = bank()
    coordinator = ObservationCoordinator(scope=protocol_value.scope, width=WIDTH)
    controller = PlasticLearningController(
        expert_bank, scope=protocol_value.scope, coordinator=coordinator,
        max_steps=2, total_step_cap=4,
        clock_period=protocol_value.clock_period)
    learner = NestedTextLearner(adapter, bank=expert_bank,
                                protocol=protocol_value,
                                controller=controller,
                                coordinator=coordinator)
    for expert in learner.bank.experts:
        expert.down.data.normal_(0.0, 0.05)
    captured = {}
    original_encode_sites = adapter.encode_sites
    sensor = torch.nn.Parameter(torch.tensor(0.1))

    def retain_sites(ids, valid, sites=None):
        result = original_encode_sites(ids, valid, sites)
        for site in result:
            result[site] = result[site] + sensor
            if result[site].requires_grad:
                result[site].retain_grad()
            captured[site] = result[site]
        return result

    adapter.encode_sites = retain_sites
    prompt_ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(prompt_ids, dtype=torch.bool)
    read = learner.differentiable_plastic_read(prompt_ids, mask)
    assert isinstance(read, DifferentiablePlasticRead)
    assert read.slot.requires_grad is True
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    labels = torch.full((1, 5), -100, dtype=torch.long)
    labels[:, 3:] = input_ids[:, 3:]
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    response_mask[:, 3:] = True
    ledger = RecordLedger(tmp_path / "ledger.sqlite")
    session = MemorySession(adapter, None, ledger)
    scope = EligibilityScope(subject_id="subject", branch_id="branch",
                             purpose="purpose", as_of=10)
    task = TaskGraph(task_id="task", goal="query", revision=0)
    with session.training_request(
            prompt_ids=prompt_ids, prompt_mask=prompt_ids.bool(), query="query",
            scope=scope, task=task, enabled_ports=(), core_impl="none",
            enabled_branches=("plastic",), plastic=read,
            identity=None, identity_enabled=False) as request:
        output = session.forward(request, input_ids, labels=labels,
                                 attention_mask=mask)
        assert output.loss is not None and bool(torch.isfinite(output.loss))
        assert output.loss.requires_grad is True
        session.backward(request, output.loss)
    for site, value in captured.items():
        assert value.grad is not None and bool(value.grad.abs().sum() > 0), site
    for index, expert in enumerate(learner.bank.experts):
        for parameter in expert.parameters():
            if index == read.expert_id:
                assert parameter.grad is not None
                assert bool(parameter.grad.abs().sum() > 0)
            else:
                assert parameter.grad is None
    assert sensor.grad is not None and bool(sensor.grad.abs().sum() > 0)
    assert all(parameter.grad is None for parameter in adapter.base.parameters())
    assert request.state == "released"
    outcome = learner.learn_event_response_ce(
        session, event("response-ce", protocol_value, prompt_ids, available=1.0),
        scope=scope, prompt_ids=prompt_ids, prompt_mask=prompt_ids.bool(),
        input_ids=input_ids, attention_mask=mask, response_mask=response_mask)
    assert outcome.status == "committed"
    assert outcome.transaction.status == "committed"
    assert outcome.transaction.revision == 1
    assert outcome.chunk.transition_count == 1
    assert outcome.chunk.post_revision == 1
    assert learner.controller.clock.eligible_events == 1
    assert learner.bank.expert_steps[read.expert_id] == 1

    with pytest.raises(ValueError, match="requires a training request"):
        session.prepare(
            prompt_ids=prompt_ids, prompt_mask=prompt_ids.bool(), query="query",
            scope=scope, task=task, training=False, enabled_ports=(),
            core_impl="none", enabled_branches=("plastic",), plastic=read,
            identity=None, identity_enabled=False)
    detached = learner.plastic_read(prompt_ids, prompt_ids.bool())
    with pytest.raises(ValueError, match="requires a differentiable plastic input"):
        with session.training_request(
                prompt_ids=prompt_ids, prompt_mask=prompt_ids.bool(), query="query",
                scope=scope, task=task, enabled_ports=(), core_impl="none",
                enabled_branches=("plastic",), plastic=detached,
                identity=None, identity_enabled=False):
            pass

def test_response_diagnostic_is_prompt_only_aligned_and_no_grad(tmp_path):
    from architecture_helpers import tiny_adapter
    from replica_lab.memory_system.ledger import RecordLedger
    from replica_lab.memory_system.session import MemorySession

    diagnostic = diagnostic_module()
    runner = runner_module()
    adapter = tiny_adapter().eval()
    adapter.install_plastic(PlasticSpec(
        experts=EXPERTS, width=WIDTH, hidden_width=HIDDEN_WIDTH,
        sites=(0, 2), route_mode="fixed", route_seed=0,
        projection_seed=1, read_gain=0.5))
    protocol_value = protocol(seed=1, hidden_size=16, feature_sites=(0, 2))
    expert_bank = ExpertBank(expert_count=EXPERTS, width=WIDTH,
                             hidden_width=HIDDEN_WIDTH, seed=7, lr=0.05)
    learner = NestedTextLearner(adapter, bank=expert_bank,
                                protocol=protocol_value)
    prompt_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    attention = torch.ones_like(input_ids, dtype=torch.bool)
    response = torch.zeros_like(attention)
    response[:, 3:] = True
    episode = runner.Episode(
        row={"id": "diagnostic-smoke", "expected_response": "4 5",
             "prompt_sha256": "smoke", "prompt": "smoke"},
        event=None, prompt_ids=prompt_ids, prompt_mask=prompt_ids.bool(),
        input_ids=input_ids, attention_mask=attention, response_mask=response)
    tokenizer = SimpleNamespace(
        decode=lambda ids, skip_special_tokens=True: " ".join(map(str, ids)))
    session = MemorySession(adapter, tokenizer,
                            RecordLedger(tmp_path / "diagnostic.sqlite"))
    result = diagnostic.teacher_forced(learner, session, [episode], label="smoke")
    assert result["successful"] == 1 and result["failures"] == 0
    assert result["loss_alignment_max_abs"] == 0.0
    assert result["results"][0]["target_ids"] == [4, 5]
    assert result["work"]["forward_calls"]["total"] == 3
    assert result["work"]["backward_calls"] == 0
    assert result["results"][0]["request_audit"]["sources_present"] is False
    assert all(parameter.grad is None for parameter in expert_bank.parameters())

def test_dev_gate_fails_closed_before_heldout():
    config = json.loads((ROOT / "configs" / "nested_text_nl1_v1.json").read_text())
    gates = config["gates"]["dev"]
    limits = config["gates"]["cost"]
    good_metrics = {
        "new_instance_accuracy": 0.8, "preteach_accuracy": 0.2,
        "correction_accuracy": 0.9, "unaffected_accuracy": 0.9,
        "unknown_false_recall": 0.0, "revoked_false_recall": 0.0,
        "retention_decline": 0.0, "shuffled_accuracy": 0.2,
        "frozen_accuracy": 0.2, "reset_P_accuracy": 0.2,
        "wrong_world_available": True,
        "wrong_world_on_primary_accuracy": 0.2,
        "spine_absent_parity": True,
    }
    good_cost = {"peak_driver_bytes": 1, "warm_p95_overhead_ms": 1.0,
                 "duty_cycle": 0.0, "checkpoint_bytes": 1, "p_only_bytes": 1}
    assert evaluate_dev_gate(metrics=good_metrics, cost=good_cost,
                             gates=gates, cost_limits=limits) == []
    failures = evaluate_dev_gate(
        metrics={**good_metrics, "new_instance_accuracy": 0.2,
                 "wrong_world_available": False},
        cost={**good_cost, "duty_cycle": None}, gates=gates,
        cost_limits=limits)
    assert "min_gain_over_preteach" in failures
    assert "wrong_world_control_unavailable" in failures
    assert "duty_cycle_unavailable" in failures
    assert evaluate_dev_gate(
        metrics={**good_metrics, "shuffled_accuracy": 0.95},
        cost=good_cost, gates=gates, cost_limits=limits
    ) == ["shuffled_route_not_better"]
    assert evaluate_dev_gate(
        metrics=good_metrics,
        cost={**good_cost,
              "checkpoint_bytes": limits["max_checkpoint_bytes"] + 1},
        gates=gates, cost_limits=limits) == ["max_checkpoint_bytes"]
    missing_checkpoint = dict(good_cost)
    del missing_checkpoint["checkpoint_bytes"]
    assert evaluate_dev_gate(
        metrics=good_metrics, cost=missing_checkpoint,
        gates=gates, cost_limits=limits) == ["max_checkpoint_bytes"]


def test_declared_configs_are_design_only_and_consistent():
    nl1 = json.loads((ROOT / "configs" / "nested_text_nl1_v1.json").read_text())
    assert nl1["status"] == "design_only_unrun"
    assert nl1["one_clock_control_only"] is True
    assert nl1["adaptive_router"] is False and nl1["rsi_enabled"] is False
    assert nl1["replay_bytes"] == 0
    assert nl1["claims"]["measured"] == []
    assert set(nl1["controls"]["p_only"]) == {
        "preteach", "reset_P", "wrong_world_P", "frozen",
        "shuffled_route", "spine_absent"}
    assert nl1["gates"]["promotion"]["min_independent_seeds"] == 3
    assert nl1["transaction"]["external_gradient_transaction"] is False
    assert nl1["transaction"]["second_controller"] is False
    episodes = json.loads((ROOT / "configs" / "nested_text_episodes_v1.json")
                          .read_text())
    assert episodes["routing"]["labels_used_for_routing"] is False
    assert episodes["routing"]["world_used_for_routing"] is False
    assert episodes["role_files"]["sealed_final"] == "sealed_final.jsonl"


def test_synthetic_episode_bundle_is_deterministic_and_label_free(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "prepare_nested_text_episodes",
        ROOT / "scripts" / "prepare_nested_text_episodes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads((ROOT / "configs" / "nested_text_episodes_v1.json")
                        .read_text())
    first = module.build_episode_bundle(config)
    second = module.build_episode_bundle(config)
    assert first == second
    required = {"teaching", "development", "correction", "unaffected",
                "unknown", "revoked", "wrong_world", "wrong_world_teaching",
                "legacy_retention", "sealed_final"}
    assert required <= set(first)
    for rows in first.values():
        for row in rows:
            assert row["labels_used_for_routing"] is False
            assert row["world_used_for_routing"] is False
            assert "route_key" not in row
            assert row["answer_label_used_for_observation"] is False
            assert row["prompt_sha256"] and row["expected_response"]
    assert first["teaching"] and all(
        row["verified"] and row["eligible"] and row["split_role"] == "teaching"
        for row in first["teaching"])
    assert all(row["split_role"] == "sealed_final"
               for row in first["sealed_final"])
    output = tmp_path / "episodes"
    manifest = module.write_bundle(first, config, output)
    assert manifest["files_sha256"]
    assert (output / "manifest.json").exists()
    with pytest.raises(ValueError):
        module.write_bundle(first, config, output)


def test_v2_episode_correction_is_minimal_and_v1_bytes_are_unchanged(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "prepare_nested_text_episodes_v2_test",
        ROOT / "scripts" / "prepare_nested_text_episodes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config_v1 = json.loads((ROOT / "configs" / "nested_text_episodes_v1.json")
                           .read_text())
    config_v2 = json.loads((ROOT / "configs" / "nested_text_episodes_v2.json")
                           .read_text())
    first = module.build_episode_bundle(config_v1)
    second = module.build_episode_bundle(config_v2)
    differences = []
    for role, rows in first.items():
        assert len(rows) == len(second[role])
        for old, new in zip(rows, second[role]):
            changed = sorted(key for key in set(old) | set(new)
                             if old.get(key) != new.get(key))
            if changed:
                differences.append((role, old["id"], changed))
    assert differences == [
        ("legacy_retention", "legacy-0002", ["prompt", "prompt_sha256"])]
    corrected = second["legacy_retention"][2]
    assert corrected["prompt"] == "Complete the sentence: The library ____ at dusk."
    assert corrected["expected_response"] == "closes"
    assert "closes" not in corrected["prompt"]
    output_v1 = tmp_path / "v1"
    module.write_bundle(first, config_v1, output_v1)
    assert ((output_v1 / "manifest.json").read_bytes()
            == (ROOT / "data" / "nested_text_nl1_20261026" / "manifest.json").read_bytes())
    output_v2 = tmp_path / "v2"
    manifest = module.write_bundle(second, config_v2, output_v2)
    assert manifest["schema_version"] == 2
    assert manifest["parent"]["manifest_sha256"] == config_v2["parent"]["manifest_sha256"]
    assert manifest["corrections"] == config_v2["corrections"]
    for name, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((output_v2 / name).read_bytes()).hexdigest() == digest


def test_v2_protocol_declares_corrected_benchmark_and_no_training():
    config = json.loads((ROOT / "configs" / "nested_text_nl1_v2.json").read_text())
    assert config["format"] == "replica.nested_text.nl1.v2"
    assert config["bank"]["experts"] == 1
    assert config["routing"]["constant_single_expert"] is True
    assert config["routing"]["labels_used"] is False
    assert config["routing"]["world_uid_used"] is False
    assert config["routing"]["gold_record_ids_used"] is False
    assert config["training_admitted"] is False
    assert config["allowed_native_modes"] == ["init", "p-only-eval"]
    generation = config["generation"]
    assert generation["max_new_tokens"] == 24
    assert generation["stop_tokens"] == [1, 130073]
    assert generation["scoring_version"] == "replica.nested_text.scoring.v2.1"
    assert generation["token_budget_credit"] is False
    workload = config["workload"]
    assert generation["sentence_stop"]["enabled"] is True
    assert generation["sentence_stop"]["answer_selection_allowed"] is False
    assert sum(workload["attempts"].values()) == workload["scoring_attempts"] == 196
    assert "teaching_end_to_end_seconds" in workload["duty_cycle"]
    assert "evaluation_end_to_end_seconds" in workload["duty_cycle"]
    assert workload["teaching_events_for_this_baseline"] == 0
    assert config["gates"]["cost"]["max_duty_cycle"] == 0.1
    assert config["gates"]["cost"]["max_checkpoint_bytes"] == 268435456
    manifest = ROOT / "data" / "nested_text_nl1_v2_20261026" / "manifest.json"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == (
        config["episodes"]["manifest_sha256"])
    assert config["transaction"]["objective_admitted_for_this_protocol"] is False


# ---------------------------------------------------------------------------
# NL1 v2 runner helpers (CPU only: no model load, no MPS, no native job)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def runner_module():
    spec = importlib.util.spec_from_file_location(
        "run_nested_text_v2_contract",
        ROOT / "scripts" / "run_nested_text.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

@functools.lru_cache(maxsize=1)
def diagnostic_module():
    spec = importlib.util.spec_from_file_location(
        "diagnose_nested_response_contract",
        ROOT / "scripts" / "diagnose_nested_response.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module



def v2_directory() -> Path:
    return ROOT / "data" / "nested_text_nl1_v2_20261026"


def test_v2_runner_accepts_both_episode_manifest_versions():
    module = runner_module()
    v1 = module.load_episode_manifest(ROOT / "data" / "nested_text_nl1_20261026")
    v2 = module.load_episode_manifest(v2_directory())
    assert v1["format"] == "replica.nested_text.episode_manifest.v1"
    assert v2["format"] == "replica.nested_text.episode_manifest.v2"
    assert v2["schema_version"] == 2
    assert set(v2["role_files"]) >= {"development", "sealed_final", "wrong_world"}


def test_v2_mode_dispatch_enforces_allowed_native_modes():
    module = runner_module()
    v1 = json.loads((ROOT / "configs" / "nested_text_nl1_v1.json").read_text())
    v2 = json.loads((ROOT / "configs" / "nested_text_nl1_v2.json").read_text())
    module.validate_run_mode(v1, "p-only-eval")
    module.validate_run_mode(v1, "teach")
    with pytest.raises(module.RunnerError, match="init"):
        module.validate_run_mode(v1, "init")
    module.validate_run_mode(v2, "init")
    module.validate_run_mode(v2, "p-only-eval")
    for mode in ("teach", "resume"):
        with pytest.raises(module.RunnerError, match="not admitted"):
            module.validate_run_mode(v2, mode)
    module.validate_v2_config(v2)
    with pytest.raises(module.RunnerError, match="one shared expert"):
        module.validate_v2_config({**v2, "bank": {**v2["bank"], "experts": 2}})
    with pytest.raises(module.RunnerError, match="stop tokens"):
        module.validate_v2_config(
            {**v2, "generation": {**v2["generation"], "stop_tokens": [1]}})
    with pytest.raises(module.RunnerError, match="v2 protocol"):
        module.validate_v2_config(v1)


def test_behavioral_reference_admits_response_ce_and_fit_gate():
    module = runner_module()
    config = json.loads((ROOT / "configs" /
                         "nested_text_behavioral_reference_v1.json").read_text())
    module.validate_run_mode(config, "teach")
    module.validate_run_mode(config, "fit-eval")
    module.validate_v2_config(config)
    assert config["transaction"]["response_only_labels"] is True
    assert config["workload"]["fit_eval_attempts"] == 28
    bad_objective = {**config, "transaction": {
        **config["transaction"], "objective": "detached_local_feature_alignment"}}
    with pytest.raises(module.RunnerError, match="response objective"):
        module.validate_v2_config(bad_objective)
    bad_fit = {**config, "training": {
        **config["training"], "fit_eval_required": False}}
    with pytest.raises(module.RunnerError, match="fit evaluation"):
        module.validate_v2_config(bad_fit)

def test_behavioral_budget02_is_rejected_by_one_clock_invariant():
    module = runner_module()
    config = json.loads((ROOT / "configs" /
                         "nested_text_behavioral_budget02_v1.json").read_text())
    assert config["transaction"]["steps_per_event"] == 2
    assert config["training"]["max_updates"] == 56
    with pytest.raises(module.RunnerError, match="one step per event"):
        module.validate_v2_config(config)




def test_v2_protocol_binds_constant_single_expert():
    module = runner_module()
    config = json.loads((ROOT / "configs" / "nested_text_nl1_v2.json").read_text())
    lock = {"snapshot_files_sha256": {
        "model.safetensors": "a" * 64,
        "tokenizer.json": "b" * 64,
        "chat_template.jinja": "c" * 64,
    }}
    protocol_value = module.protocol_for_world(
        config, lock, hidden_size=32, world_uid="world_alpha")
    assert protocol_value.expert_count == 1
    assert protocol_value.expert_count == config["bank"]["experts"]
    assert protocol_value.route_mode == "fixed"
    assert protocol_value.route_seed == 0
    assert protocol_value.scope == "world_alpha/text"


def test_v2_workload_duty_amortizes_teaching_and_evaluation():
    module = runner_module()
    assert module.workload_duty_cycle(
        learning_update_seconds=2.0, teaching_end_to_end_seconds=6.0,
        evaluation_end_to_end_seconds=2.0) == pytest.approx(0.25)
    assert module.workload_duty_cycle(
        learning_update_seconds=0.0, teaching_end_to_end_seconds=0.0,
        evaluation_end_to_end_seconds=3.0) == 0.0
    for kwargs in ({"learning_update_seconds": None,
                    "teaching_end_to_end_seconds": 1.0,
                    "evaluation_end_to_end_seconds": 1.0},
                   {"learning_update_seconds": 1.0,
                    "teaching_end_to_end_seconds": -1.0,
                    "evaluation_end_to_end_seconds": 1.0},
                   {"learning_update_seconds": 1.0,
                    "teaching_end_to_end_seconds": 0.0,
                    "evaluation_end_to_end_seconds": 0.0}):
        with pytest.raises(module.RunnerError):
            module.workload_duty_cycle(**kwargs)


def test_v2_phase_work_counts_measured_calls_and_positions():
    module = runner_module()
    records = [("plastic_read", 7), ("session_prepare", 7),
               ("generation", 7), ("generation", 8), ("generation", 9)]
    work = module.phase_work_from_records(
        records, generated_ids=3, stop_reason="stop_token", stop_token=1)
    assert work["plastic_feature_forward_calls"] == 1
    assert work["plastic_feature_prompt_positions"] == 7
    assert work["session_sense_forward_calls"] == 1
    assert work["session_sense_prompt_positions"] == 7
    assert work["generation_prefill_forward_calls"] == 1
    assert work["generation_prefill_positions"] == 7
    assert work["generation_decode_forward_calls"] == 2
    assert work["decode_positions"] == 17
    assert work["generated_ids"] == 3
    assert work["backward_calls"] == 0 and work["replay_calls"] == 0
    assert work["response_supervised_tokens"] == 0
    assert work["truncated"] is False
    budget = module.phase_work_from_records(
        records[:3], generated_ids=1, stop_reason="token_budget",
        stop_token=None)
    assert budget["decode_positions"] == 0
    assert budget["truncated"] is True
    missing = module.phase_work_from_records(
        [], generated_ids=1, stop_reason="missing_stop", stop_token=None)
    assert missing["generation_prefill_positions"] == 0
    assert missing["generation_prefill_forward_calls"] == 0
    with pytest.raises(module.RunnerError, match="unknown measured phase"):
        module.phase_work_from_records(
            [("unattributed", 1)], generated_ids=1,
            stop_reason="stop_token", stop_token=1)


def test_v2_work_totals_sum_positions_and_forward_calls():
    module = runner_module()
    first = module.phase_work_from_records(
        [("plastic_read", 4), ("session_prepare", 4), ("generation", 4),
         ("generation", 5)],
        generated_ids=2, stop_reason="stop_token", stop_token=1)
    second = module.phase_work_from_records(
        [("plastic_read", 3), ("session_prepare", 3), ("generation", 3)],
        generated_ids=1, stop_reason="token_budget", stop_token=None)
    totals = module.sum_phase_work([first, second])
    assert totals["requests"] == 2
    assert totals["generated_ids"] == 3
    assert totals["plastic_feature_prompt_positions"] == 7
    assert totals["session_sense_prompt_positions"] == 7
    assert totals["generation_prefill_positions"] == 7
    assert totals["decode_positions"] == 5
    assert totals["forward_calls"]["total"] == 7
    assert totals["stop_reasons"] == {"stop_token": 1, "token_budget": 1}
    assert totals["truncated_attempts"] == 1
    assert totals["backward_calls"] == 0 and totals["replay_calls"] == 0



def test_response_teaching_work_counts_targets_and_each_optimizer_step():
    module = runner_module()
    work = module.response_teaching_work(
        prompt_positions=7, full_positions=11, response_tokens=5,
        events=2, steps_per_event=2)
    assert work["processed_positions"] == 68
    assert work["response_target_positions"] == 11
    assert work["response_ce_positions"] == 22
    assert work["response_supervised_tokens"] == 10
    assert work["differentiable_read_prompt_positions"] == 14
    assert work["session_sense_prompt_positions"] == 14
    assert work["forward_calls"] == {
        "prompt_features": 4, "differentiable_read": 4,
        "session_sense": 4, "response_ce": 4, "total": 16}
    assert work["backward_calls"] == 4
    assert work["response_ce_backward_calls"] == 4
    single = module.response_teaching_work(
        prompt_positions=637, full_positions=759, response_tokens=122,
        events=28, steps_per_event=1)
    assert single["processed_positions"] == 3429
    assert single["forward_calls"]["total"] == 140

def test_v2_latency_summary_reports_p50_and_p95_for_each_phase():
    module = runner_module()
    rows = [{field: index / 1000.0 for field in module.LATENCY_FIELDS}
            for index in range(1, 21)]
    summary = module.summarize_latency(rows)
    assert set(summary) == set(module.LATENCY_FIELDS)
    values = [index / 1000.0 for index in range(1, 21)]
    for field in module.LATENCY_FIELDS:
        entry = summary[field]
        assert entry["count"] == 20
        assert entry["p50_ms"] == pytest.approx(module.p50(values) * 1000.0)
        assert entry["p95_ms"] >= entry["p50_ms"]
    empty = module.summarize_latency([])
    assert empty["total_seconds"] == {"count": 0, "p50_ms": 0.0,
                                      "p95_ms": 0.0}


def test_v2_phase_work_merges_per_request_latency():
    module = runner_module()
    work = [{"id": "dev-0000", "label": "taught_development",
             "generated_ids": 2}]
    latency = [{"id": "dev-0000", "label": "taught_development",
                **{field: 0.01 for field in module.LATENCY_FIELDS}}]
    merged = module._merge_request_rows(work, latency)
    assert merged[0]["generated_ids"] == 2
    assert merged[0]["total_seconds"] == 0.01
    assert merged[0]["plastic_read_seconds"] == 0.01


def test_v2_scoring_uses_stop_metadata_and_rejects_empty_truncated():
    module = runner_module()
    stopped = module.finalize_generation(
        [7, 1], decode=lambda ids: "Tevufa", max_new_tokens=24)
    assert stopped.stop_reason == "stop_token" and stopped.truncated is False
    assert module.score_generation("Tevufa", stopped).correct is True
    budget = module.finalize_generation(
        [7, 8], decode=lambda ids: "Tevu", max_new_tokens=2)
    assert budget.stop_reason == "token_budget" and budget.truncated is True
    assert module.score_generation("Tevufa", budget).correct is False
    empty = module.finalize_generation(
        [1], decode=lambda ids: "", max_new_tokens=24)
    assert empty.truncated is False
    assert module.score_generation("Tevufa", empty).correct is False
    conflict = module.finalize_generation(
        [9, 1], decode=lambda ids: "Tevufa Mipezu", max_new_tokens=24)
    assert module.score_generation(
        "Tevufa", conflict, candidate_answers=("Mipezu",)).correct is False
    exact = module.finalize_generation(
        [9, 1], decode=lambda ids: "unknown", max_new_tokens=24)
    assert module.score_generation("unknown", exact).correct is True
    loose = module.finalize_generation(
        [9, 1], decode=lambda ids: "the answer is unknown", max_new_tokens=24)
    assert module.score_generation("unknown", loose).correct is False


def test_v2_native_forward_counter_tracks_actual_calls():
    module = runner_module()

    class CountingAdapter:
        def __init__(self):
            self.seen = []

        def _native(self, ids, valid=None, **kwargs):
            self.seen.append(int(ids.shape[-1]))
            return ids

    adapter = CountingAdapter()
    with module.counted_native_forwards(adapter) as counter:
        counter.phase = "plastic_read"
        adapter._native(torch.ones(1, 4, dtype=torch.long))
        counter.phase = "generation"
        adapter._native(torch.ones(1, 5, dtype=torch.long))
        adapter._native(torch.ones(1, 6, dtype=torch.long))
    assert counter.records == [("plastic_read", 4), ("generation", 5),
                               ("generation", 6)]
    assert adapter.seen == [4, 5, 6]
    with pytest.raises(module.RunnerError, match="accounting phase"):
        with module.counted_native_forwards(adapter) as unphased:
            adapter._native(torch.ones(1, 1, dtype=torch.long))
    adapter._native(torch.ones(1, 2, dtype=torch.long))
    assert adapter.seen == [4, 5, 6, 2]
    assert counter.records == [("plastic_read", 4), ("generation", 5),
                               ("generation", 6)]
    assert unphased.records == []


def test_v2_candidate_answers_are_fit_scoped_and_dev_blocked():
    module = runner_module()
    directory = v2_directory()
    manifest = module.load_episode_manifest(directory)
    module.reset_role_access()
    world = manifest["primary_world"]
    scope = f"{world}/text"
    fit_answers = module.fit_candidate_answers_for(
        directory, manifest, world_uid=world, scope=scope)
    fit_expected = {row["expected_response"]
                    for role in module.FIT_CANDIDATE_ROLES
                    for row in module.load_rows(directory, manifest, role)
                    if row["eligible"] and row["world_uid"] == world
                    and row["scope_uid"] == scope}
    assert set(fit_answers) == fit_expected
    assert list(fit_answers) == sorted(fit_expected)
    with pytest.raises(module.RunnerError,
                       match="cannot be parsed before a validated fit receipt"):
        module.development_candidate_answers_for(directory, manifest)
    with pytest.raises(module.RunnerError,
                       match="cannot be parsed before a validated fit receipt"):
        module.load_rows(directory, manifest, "sealed_final")
    baseline = json.loads((ROOT / "configs" /
                           "nested_text_nl1_v2.json").read_text())
    module.ROLE_ACCESS.admit_without_fit_receipt(baseline)
    try:
        answers = module.development_candidate_answers_for(directory, manifest)
        expected = set()
        for role in module.DEVELOPMENT_CANDIDATE_ROLES:
            expected.update(row["expected_response"]
                            for row in module.load_rows(directory, manifest, role))
        assert set(answers) == expected
    finally:
        module.reset_role_access()


def test_v2_episode_binding_rejects_alternate_root_and_manifest_hash():
    module = runner_module()
    config = json.loads((ROOT / "configs" / "nested_text_nl1_v2.json").read_text())
    args = SimpleNamespace(episodes=ROOT / "data" / "nested_text_nl1_v2_20261026")
    manifest = json.loads((ROOT / "data" / "nested_text_nl1_v2_20261026" / "manifest.json").read_text())
    module.validate_episode_binding(args, config, manifest)
    with pytest.raises(module.RunnerError, match="pinned root"):
        args.episodes = ROOT / "configs"
        module.validate_episode_binding(args, config, manifest)
    with pytest.raises(module.RunnerError, match="digest differs"):
        drifted = copy.deepcopy(config)
        drifted["episodes"]["manifest_sha256"] = "0" * 64
        args.episodes = ROOT / "data" / "nested_text_nl1_v2_20261026"
        module.validate_episode_binding(args, drifted, manifest)


def test_v2_duty_cycle_includes_wrong_world_initialization():
    module = runner_module()
    duty = module.workload_duty_cycle(
        learning_update_seconds=2.0, teaching_end_to_end_seconds=4.0,
        evaluation_end_to_end_seconds=6.0,
        wrong_world_learning_update_seconds=1.0,
        wrong_world_teaching_end_to_end_seconds=2.0)
    assert duty == 0.25
    with pytest.raises(module.RunnerError, match="nonnegative"):
        module.workload_duty_cycle(
            learning_update_seconds=0.0, teaching_end_to_end_seconds=1.0,
            evaluation_end_to_end_seconds=1.0,
            wrong_world_teaching_end_to_end_seconds=-1.0)


def test_v2_amended_scoring_and_stop_policy_are_explicit():
    module = runner_module()
    config = json.loads((ROOT / "configs" / "nested_text_nl1_v2.json").read_text())
    assert config["generation"]["scoring_version"] == "replica.nested_text.scoring.v2.1"
    assert config["generation"]["sentence_stop"]["version"] == (
        "replica.nested_text.sentence_stop.v1")
    assert config["workload"]["id"] == "nl1_v2_development_196_attempts_v2_1"
    assert config["workload"]["wrong_world_initialization_cost_required"] is True
    assert "after all corrected development gates pass" in config["workload"]["sealed_final_access"]
    with pytest.raises(module.RunnerError, match="stop-policy version"):
        drifted = copy.deepcopy(config)
        drifted["generation"]["sentence_stop"]["version"] = "other"
        module.validate_v2_config(drifted)


# ---------------------------------------------------------------------------
# Fit receipt: dependent-development admission (CPU only, no model/worker)
# ---------------------------------------------------------------------------

BEHAVIORAL_CONFIG = (ROOT / "configs" /
                     "nested_text_behavioral_reference_v1.json")
FIT_UPDATES = 28


def _write_json_file(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _protocol_lock() -> dict:
    return {"snapshot_files_sha256": {
        "model.safetensors": "a" * 64,
        "tokenizer.json": "b" * 64,
        "chat_template.jinja": "c" * 64,
    }}


def fit_bundle(tmp_path: Path, *, failing_attempts: int = 0):
    """A complete CPU-only fit/teaching fixture with no native execution."""
    module = runner_module()
    config = json.loads(BEHAVIORAL_CONFIG.read_text(encoding="utf-8"))
    config_path = _write_json_file(tmp_path / "config.json", config)
    episodes = v2_directory()
    manifest_path = episodes / "manifest.json"
    manifest = module.load_episode_manifest(episodes)
    lock = _protocol_lock()
    lock_path = _write_json_file(tmp_path / "model_lock.json", lock)
    protocol = module.protocol_for_world(
        config, lock, hidden_size=HIDDEN_SIZE,
        world_uid=manifest["primary_world"])
    expert_bank = ExpertBank(expert_count=1, width=256, hidden_width=1024,
                             seed=0, lr=1e-4)
    personality = PersonalityIdentity.default()
    coordinator = ObservationCoordinator(scope=protocol.scope, width=256,
                                         seed=0, personality=personality)
    controller = PlasticLearningController(
        expert_bank, scope=protocol.scope, personality=personality,
        coordinator=coordinator, max_steps=2, total_step_cap=4,
        clock_period=protocol.clock_period)
    learner = NestedTextLearner(FakeAdapter(), bank=expert_bank,
                                protocol=protocol, controller=controller,
                                coordinator=coordinator,
                                personality=personality,
                                tokenizer_sha256=protocol.tokenizer_sha256)
    expert_bank.expert_steps[:] = [FIT_UPDATES]
    teach = tmp_path / "teach_primary"
    (teach / "p_only").mkdir(parents=True)
    state_path = learner.save_p_only(teach / "p_only" / "state.pt")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    checkpoint = teach / "full_checkpoint.pt"
    checkpoint.write_bytes(b"bounded-checkpoint-placeholder")
    final_state = canonical_state(learner)
    _write_json_file(teach / "final_state.json", final_state)
    _write_json_file(teach / "p_only" / "manifest.json", {
        "format": P_ONLY_FORMAT, "bytes": state_path.stat().st_size,
        "sha256": module._sha256_file(state_path), "keys": sorted(state),
        "forbidden_absent": True, "source_inputs": 0})
    wall = 4.0
    update_seconds = 2.0
    cost = {
        "events": FIT_UPDATES, "optimizer_steps": FIT_UPDATES,
        "backward_calls": FIT_UPDATES, "wall_seconds": wall,
        "update_seconds": update_seconds,
        "learning_update_seconds": update_seconds,
        "teaching_end_to_end_seconds": wall, "end_to_end_seconds": wall,
        "duty_cycle": update_seconds / wall,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "p_only_bytes": state_path.stat().st_size,
        "protocol_sha256": protocol.protocol_sha256,
        "config_sha256": module._sha256_file(config_path),
        "scoring_version": module.SCORING_VERSION,
    }
    cost_path = _write_json_file(teach / "cost.json", cost)
    run = {
        "format": module.PROMPT_FORM, "mode": "teach", "status": "completed",
        "world_uid": protocol.world_uid,
        "protocol_sha256": protocol.protocol_sha256,
        "config_sha256": module._sha256_file(config_path),
        "episodes_manifest_sha256": module._sha256_file(manifest_path),
        "implementation_sha256": {},
        "theta_stamp_unchanged": True,
        "updates": FIT_UPDATES, "learning_events": FIT_UPDATES,
        "expert_steps": [FIT_UPDATES],
        "p_only_sha256": module._sha256_file(state_path),
        "p_only_manifest_sha256": module._sha256_file(
            teach / "p_only" / "manifest.json"),
        "checkpoint_sha256": module._sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "canonical_state_sha256": final_state["sha256"],
        "cost_sha256": module._sha256_file(cost_path),
        "checkpoint": str(checkpoint), "p_only": str(state_path),
    }
    _write_json_file(teach / "run.json", run)
    rows = module.teaching_rows(
        episodes, manifest, manifest["primary_world"],
        f"{manifest['primary_world']}/text",
        limit=int(config["workload"]["fit_eval_attempts"]))
    audit = {"sources_present": False, "memory_spans": 0,
             "selection_records": 0, "ledger_records": 0,
             "controller_constructed": False, "coordinator_constructed": False,
             "updates_enabled": False}
    results = []
    for index, row in enumerate(rows):
        record = {"id": row["id"], "label": "fit",
                  "expected": row["expected_response"],
                  "prompt_sha256": row["prompt_sha256"],
                  "scoring_version": module.SCORING_VERSION,
                  "answer_grammar_version": module.ANSWER_GRAMMAR_VERSION,
                  "generated_ids": [1], "content_ids": [],
                  "stop_reason": "stop_token", "stop_token": 1,
                  "max_new_tokens": 24, "truncated": False}
        if index < failing_attempts:
            for field in ("generated_ids", "content_ids", "stop_reason",
                          "stop_token", "max_new_tokens", "truncated"):
                record.pop(field)
            record.update({"correct": False, "reason": "generation_failure",
                           "truncated": None,
                           "error": "RuntimeError: dropped request"})
        else:
            record.update({"output": row["expected_response"],
                           "correct": True, "reason": "correct",
                           "request_audit": dict(audit)})
        results.append(record)
    correct = sum(1 for item in results if item["correct"] is True)
    scores = {"label": "fit", "count": len(results), "correct": correct,
              "failures": failing_attempts,
              "accuracy": correct / len(results), "results": results,
              "scoring_version": module.SCORING_VERSION}
    gate_failures = []
    if correct / len(results) < 0.5:
        gate_failures.append("min_fit_accuracy")
    if failing_attempts:
        gate_failures.extend(["generation_failures",
                              "incomplete_request_audit"])
    gate = {
        "label": "fit", "fit_accuracy": correct / len(results),
        "all_declared_gates_passed": not gate_failures,
        "gate_failures": gate_failures, "metrics": {"cost": cost},
        "scores": scores, "scoring_version": module.SCORING_VERSION,
        "config_sha256": module._sha256_file(config_path),
        "stop_policy_version": module.STOP_POLICY_VERSION,
        "answer_grammar_version": module.ANSWER_GRAMMAR_VERSION,
    }
    gate_path = _write_json_file(tmp_path / "gates" / "fit.json", gate)
    receipt = module.build_fit_receipt(
        config=config, config_path=config_path, manifest=manifest,
        manifest_path=manifest_path, episodes_root=episodes,
        state_path=state_path, state=state, protocol=protocol,
        teach_dir=teach, teach_expected_updates=FIT_UPDATES,
        fit_gate_path=gate_path)
    receipt_path = _write_json_file(tmp_path / "fit_receipt.json", receipt)
    return SimpleNamespace(
        module=module, config=config, config_path=config_path,
        manifest=manifest, manifest_path=manifest_path, episodes=episodes,
        lock=lock, lock_path=lock_path, protocol=protocol,
        state_path=state_path, state=state, teach=teach, gate=gate,
        gate_path=gate_path, receipt=receipt, receipt_path=receipt_path,
        rows=rows)


def validate_bundle(bundle, receipt=None, *, state="default", state_path=None):
    bundle.module.validate_fit_receipt(
        receipt if receipt is not None else bundle.receipt,
        config=bundle.config, config_path=bundle.config_path,
        manifest=bundle.manifest, manifest_path=bundle.manifest_path,
        episodes_root=bundle.episodes,
        state_path=state_path or bundle.state_path, lock=bundle.lock,
        state=bundle.state if state == "default" else state)


def resealed(bundle, mutate) -> dict:
    receipt = copy.deepcopy(bundle.receipt)
    mutate(receipt)
    return bundle.module.seal_fit_receipt(receipt)


def _rebuild_with_gate(bundle, mutate):
    """Rewrite/rebuild the preserved fit gate, then reseal its admission."""
    gate = copy.deepcopy(bundle.gate)
    mutate(gate)
    gate_path = _write_json_file(bundle.gate_path, gate)
    return bundle.module.build_fit_receipt(
        config=bundle.config, config_path=bundle.config_path,
        manifest=bundle.manifest, manifest_path=bundle.manifest_path,
        episodes_root=bundle.episodes, state_path=bundle.state_path,
        state=bundle.state, protocol=bundle.protocol, teach_dir=bundle.teach,
        teach_expected_updates=FIT_UPDATES, fit_gate_path=gate_path)


def _receipt_for_gate(bundle, mutate) -> dict:
    """Replace the gate file and bind the existing receipt to it."""
    gate = copy.deepcopy(bundle.gate)
    mutate(gate)
    gate_path = _write_json_file(bundle.gate_path, gate)
    receipt = copy.deepcopy(bundle.receipt)
    receipt["gate_report"]["sha256"] = bundle.module._sha256_file(gate_path)
    return bundle.module.seal_fit_receipt(receipt)


def _set(path, value):
    def mutate(receipt):
        target = receipt
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return mutate


def test_fit_receipt_round_trip_binds_every_admission_field(tmp_path):
    bundle = fit_bundle(tmp_path)
    module = bundle.module
    receipt = module.load_fit_receipt(bundle.receipt_path)
    assert receipt == bundle.receipt
    assert receipt["format"] == "replica.nested_text.fit_receipt.v1"
    assert receipt["schema_version"] == 1
    assert receipt["status"] == "passed"
    assert set(receipt) == module.FIT_RECEIPT_KEYS
    assert receipt["receipt_sha256"] == module.seal_fit_receipt(
        receipt)["receipt_sha256"]
    assert receipt["p_state"]["sha256"] == module._sha256_file(bundle.state_path)
    assert receipt["p_state"]["bytes"] == bundle.state_path.stat().st_size
    assert receipt["p_state"]["binding_sha256"] == (
        module.p_only_binding_digest(bundle.state))
    assert receipt["p_state"]["theta"] == receipt["protocol"]["theta"]
    expected_ids = [row["id"] for row in bundle.rows]
    assert receipt["coverage"]["ids"] == expected_ids
    assert receipt["coverage"]["measured_attempts"] == FIT_UPDATES
    assert receipt["coverage"]["declared_attempts"] == FIT_UPDATES
    assert receipt["coverage"]["failed_attempts"] == 0
    assert receipt["coverage"]["source_free"] is True
    assert receipt["coverage"]["audit_complete"] is True
    assert receipt["attempts"]["correct"] == FIT_UPDATES
    assert receipt["attempts"]["accuracy"] == 1.0
    assert receipt["attempts"]["all_declared_gates_passed"] is True
    assert receipt["gate_report"]["all_declared_gates_passed"] is True
    assert receipt["heldout_opened"] is False
    assert receipt["scoring"]["scoring_version"] == module.SCORING_VERSION
    assert receipt["scoring"]["answer_grammar_version"] == (
        module.ANSWER_GRAMMAR_VERSION)
    assert receipt["scoring"]["candidate_answers_sha256"] == (
        module._sha256_json(list(module.fit_candidate_answers_for(
            bundle.episodes, bundle.manifest,
            world_uid=bundle.manifest["primary_world"],
            scope=f"{bundle.manifest['primary_world']}/text"))))
    primary = receipt["teaching"]["primary"]
    assert primary["p_only_sha256"] == receipt["p_state"]["sha256"]
    assert primary["canonical_state_sha256"] == json.loads(
        (bundle.teach / "final_state.json").read_text())["sha256"]
    assert primary["updates"] == FIT_UPDATES
    assert primary["run_sha256"] == module._sha256_file(
        bundle.teach / "run.json")
    validate_bundle(bundle)
    tampered = copy.deepcopy(receipt)
    tampered["attempts"]["correct"] = FIT_UPDATES - 1
    with pytest.raises(module.RunnerError, match="digest differs"):
        validate_bundle(bundle, tampered, state=None)


@pytest.mark.parametrize("label, mutate, message", [
    ("p_state sha256", _set(("p_state", "sha256"), "0" * 64), "P-state digest"),
    ("p_state bytes", _set(("p_state", "bytes"), 1), "byte count"),
    ("p_state binding",
     _set(("p_state", "binding_sha256"), "0" * 64), "binding digest"),
    ("p_state protocol",
     _set(("p_state", "protocol_sha256"), "0" * 64),
     "binding differs from its protocol"),
    ("p_state theta",
     _set(("p_state", "theta", "lock_sha256"), "0" * 64), "binding digest"),
    ("p_state bank", _set(("p_state", "bank", "width"), 8), "binding digest"),
    ("p_state route", _set(("p_state", "route", "seed"), 7), "binding digest"),
    ("p_state features",
     _set(("p_state", "feature_producer", "hidden_size"), 4), "binding digest"),
    ("p_state read", _set(("p_state", "read_path", "gain"), 0.9),
     "binding digest"),
    ("config sha256", _set(("config", "sha256"), "0" * 64), "config digest"),
    ("config objective",
     _set(("config", "objective"), "other"), "objective"),
    ("config steps", _set(("config", "steps_per_event"), 2), "step policy"),
    ("protocol sha256",
     _set(("protocol", "protocol_sha256"), "0" * 64), "protocol"),
    ("protocol gain", _set(("protocol", "read_gain"), 0.9), "protocol"),
    ("protocol theta",
     _set(("protocol", "theta", "stamp_sha256"), "0" * 64), "protocol"),
    ("world", _set(("world_uid",), "world_beta"), "world/scope"),
    ("manifest sha256",
     _set(("episodes", "manifest_sha256"), "0" * 64), "manifest digest"),
    ("manifest format",
     _set(("episodes", "schema_version"), 1), "manifest format"),
    ("manifest files",
     _set(("episodes", "files_sha256_digest"), "0" * 64), "file digests"),
    ("manifest roles",
     _set(("episodes", "role_files_digest"), "0" * 64), "file digests"),
    ("manifest world",
     _set(("episodes", "primary_world"), "world_beta"), "world pair"),
])
def test_fit_receipt_rejects_state_config_protocol_and_manifest_drift(
        tmp_path, label, mutate, message):
    bundle = fit_bundle(tmp_path)
    receipt = resealed(bundle, mutate)
    with pytest.raises(bundle.module.RunnerError, match=message):
        validate_bundle(bundle, receipt, state=None)


def forged_p_state(bundle, mutate) -> dict:
    """Mutate the P-state block and re-forge a self-consistent binding digest."""
    receipt = copy.deepcopy(bundle.receipt)
    mutate(receipt)
    block = receipt["p_state"]
    block["binding_sha256"] = bundle.module._sha256_json(
        {key: block[key] for key in bundle.module.P_ONLY_KEYS
         if key != "parameters"})
    return bundle.module.seal_fit_receipt(receipt)


@pytest.mark.parametrize("label, mutate, message", [
    ("protocol", _set(("p_state", "protocol_sha256"), "0" * 64),
     "binding differs from its protocol"),
    ("scope", _set(("p_state", "scope"), "world_beta/text"),
     "binding differs from its protocol"),
    ("theta", _set(("p_state", "theta", "stamp_sha256"), "0" * 64),
     "theta binding differs"),
    ("bank", _set(("p_state", "bank", "width"), 8), "bank binding differs"),
    ("route", _set(("p_state", "route", "seed"), 7), "route binding differs"),
    ("features", _set(("p_state", "feature_producer", "hidden_size"), 4),
     "feature producer differs"),
    ("read", _set(("p_state", "read_path", "gain"), 0.9),
     "read path differs"),
    ("counters", _set(("p_state", "expert_steps"), [1]),
     "P-state counters differ"),
])
def test_fit_receipt_rejects_forged_p_state_binding_fields(
        tmp_path, label, mutate, message):
    bundle = fit_bundle(tmp_path)
    receipt = forged_p_state(bundle, mutate)
    with pytest.raises(bundle.module.RunnerError, match=message):
        validate_bundle(bundle, receipt, state=None)


@pytest.mark.parametrize("label, mutate, message", [
    ("scorer", _set(("scoring", "scoring_version"), "other"), "scorer version"),
    ("grammar", _set(("scoring", "answer_grammar_version"), "other"),
     "answer grammar"),
    ("stop policy", _set(("scoring", "stop_policy_version"), "other"),
     "stop policy"),
    ("tokens", _set(("scoring", "max_new_tokens"), 8), "generation budget"),
    ("stop tokens", _set(("scoring", "stop_tokens"), [1]), "stop tokens"),
    ("sentence stop",
     _set(("scoring", "sentence_stop", "enabled"), False),
     "sentence-stop policy"),
    ("empty credit", _set(("scoring", "empty_generation_credit"), True),
     "credits empty"),
    ("candidates", _set(("scoring", "candidate_answers_sha256"), "0" * 64),
     "candidate answers"),
])
def test_fit_receipt_rejects_scorer_and_stopping_policy_drift(
        tmp_path, label, mutate, message):
    bundle = fit_bundle(tmp_path)
    receipt = resealed(bundle, mutate)
    with pytest.raises(bundle.module.RunnerError, match=message):
        validate_bundle(bundle, receipt, state=None)


def test_fit_receipt_rejects_low_accuracy_despite_upstream_pass_flag(tmp_path):
    bundle = fit_bundle(tmp_path)

    def mutate(gate):
        results = gate["scores"]["results"]
        for item in results[1:]:
            item["output"] = "nothing relevant"
            item["correct"] = False
            item["reason"] = "missing_answer"
        gate["scores"]["correct"] = 1
        gate["scores"]["accuracy"] = 1 / len(results)
        gate["fit_accuracy"] = 1 / len(results)
        gate["all_declared_gates_passed"] = True
        gate["gate_failures"] = []

    with pytest.raises(bundle.module.RunnerError,
                       match="independently recomputed|fit gate"):
        _rebuild_with_gate(bundle, mutate)


def test_fit_receipt_rejects_incorrect_text_marked_correct(tmp_path):
    bundle = fit_bundle(tmp_path)
    with pytest.raises(bundle.module.RunnerError, match="contradicts"):
        _rebuild_with_gate(
            bundle,
            lambda gate: gate["scores"]["results"][-1].__setitem__(
                "output", "Unrelated sentence."))


@pytest.mark.parametrize("mutate", [
    lambda gate: gate.__setitem__("answer_grammar_version", "stale"),
    lambda gate: gate.pop("answer_grammar_version"),
    lambda gate: gate["scores"]["results"][0].__setitem__(
        "answer_grammar_version", "stale"),
    lambda gate: gate["scores"]["results"][0].pop("answer_grammar_version"),
])
def test_fit_receipt_rejects_stale_or_missing_upstream_grammar(tmp_path, mutate):
    bundle = fit_bundle(tmp_path)
    with pytest.raises(bundle.module.RunnerError, match="grammar"):
        _rebuild_with_gate(bundle, mutate)


def test_fit_receipt_validate_rejects_resealed_stale_gate_evidence(tmp_path):
    bundle = fit_bundle(tmp_path)
    receipt = _receipt_for_gate(
        bundle, lambda gate: gate.pop("answer_grammar_version"))
    with pytest.raises(bundle.module.RunnerError, match="grammar"):
        validate_bundle(bundle, receipt, state=None)
    forged = _receipt_for_gate(bundle, lambda gate: gate["scores"][
        "results"][-1].__setitem__("output", "Unrelated sentence."))
    with pytest.raises(bundle.module.RunnerError, match="contradicts"):
        validate_bundle(bundle, forged, state=None)


@pytest.mark.parametrize("label, mutate, message", [
    ("truncated flag", lambda r: r.__setitem__("truncated", True),
     "truncation"),
    ("stop token", lambda r: r.__setitem__("stop_token", 2), "stop token"),
    ("missing content", lambda r: r.pop("content_ids"), "content_ids"),
    ("content prefix", lambda r: r.__setitem__("content_ids", [9]), "content"),
    ("budget stop", lambda r: r.update(
        {"stop_reason": "token_budget", "stop_token": None,
         "content_ids": [1]}), "token-budget"),
    ("budget value", lambda r: r.__setitem__("max_new_tokens", 8),
     "generation budget"),
    ("empty ids", lambda r: r.__setitem__("generated_ids", []),
     "token budget"),
    ("missing stop token", lambda r: r.pop("stop_token"), "stop token"),
    ("sentence stop", lambda r: r.update(
        {"stop_reason": "sentence_complete", "stop_token": None,
         "content_ids": [1]}), "sentence"),
    ("missing generated", lambda r: r.pop("generated_ids"),
     "generated_ids"),
])
def test_fit_receipt_rejects_inconsistent_generation_stop_evidence(
        tmp_path, label, mutate, message):
    bundle = fit_bundle(tmp_path)
    with pytest.raises(bundle.module.RunnerError, match=message):
        _rebuild_with_gate(
            bundle,
            lambda gate: mutate(gate["scores"]["results"][0]))


@pytest.mark.parametrize("label, mutate", [
    ("missing case", lambda gate: gate["scores"]["results"].pop()),
    ("duplicate case", lambda gate: gate["scores"]["results"].append(
        copy.deepcopy(gate["scores"]["results"][0]))),
    ("unexpected case", lambda gate: gate["scores"]["results"][-1].__setitem__(
        "id", "dev-9999")),
])
def test_fit_receipt_rejects_incomplete_or_duplicated_gate_cases(
        tmp_path, label, mutate):
    bundle = fit_bundle(tmp_path)
    with pytest.raises(bundle.module.RunnerError, match="coverage"):
        _rebuild_with_gate(bundle, mutate)


def test_fit_receipt_validate_rejects_resealed_bad_stop_evidence(tmp_path):
    bundle = fit_bundle(tmp_path)
    receipt = _receipt_for_gate(
        bundle,
        lambda gate: gate["scores"]["results"][0].__setitem__(
            "truncated", True))
    with pytest.raises(bundle.module.RunnerError, match="truncation"):
        validate_bundle(bundle, receipt, state=None)


def test_fit_receipt_validate_rejects_resealed_low_accuracy_gate(tmp_path):
    bundle = fit_bundle(tmp_path)

    def mutate(gate):
        results = gate["scores"]["results"]
        for item in results[1:]:
            item["output"] = "nothing relevant"
            item["correct"] = False
            item["reason"] = "missing_answer"
        gate["fit_accuracy"] = 1 / len(results)
        gate["all_declared_gates_passed"] = True
        gate["gate_failures"] = []

    receipt = _receipt_for_gate(bundle, mutate)
    with pytest.raises(bundle.module.RunnerError,
                       match="aggregates|attempt accounting|independently"):
        validate_bundle(bundle, receipt, state=None)


@pytest.mark.parametrize("label, mutate", [
    ("missing id", lambda r: r["coverage"]["ids"].pop()),
    ("duplicate id",
     lambda r: r["coverage"]["ids"].__setitem__(1, r["coverage"]["ids"][0])),
    ("unexpected id", lambda r: r["coverage"]["ids"].append("dev-9999")),
    ("cases digest", _set(("coverage", "cases_sha256"), "0" * 64)),
    ("measured", _set(("coverage", "measured_attempts"), FIT_UPDATES - 1)),
    ("unique", _set(("coverage", "unique_ids"), FIT_UPDATES - 1)),
    ("failed", _set(("coverage", "failed_attempts"), 1)),
    ("source free", _set(("coverage", "source_free"), False)),
    ("audit", _set(("coverage", "audit_complete"), False)),
    ("role", _set(("coverage", "role"), "development")),
    ("declared", _set(("coverage", "declared_attempts"), 27)),
])
def test_fit_receipt_rejects_incomplete_or_duplicate_case_coverage(
        tmp_path, label, mutate):
    bundle = fit_bundle(tmp_path)
    receipt = resealed(bundle, mutate)
    with pytest.raises(bundle.module.RunnerError, match="case coverage"):
        validate_bundle(bundle, receipt, state=None)


@pytest.mark.parametrize("label, mutate, message", [
    ("status", _set(("status",), "failed"), "status is 'failed'"),
    ("gate", _set(("gate_report", "all_declared_gates_passed"), False),
     "failed fit gate"),
    ("gate failures", _set(("gate_report", "gate_failures"), ["min_fit_accuracy"]),
     "fit gate failures"),
    ("gate digest", _set(("gate_report", "sha256"), "0" * 64),
     "gate report digest"),
    ("accuracy", _set(("attempts", "accuracy"), 0.25), "attempt accounting"),
    ("correct", _set(("attempts", "correct"), FIT_UPDATES - 1),
     "attempt accounting"),
    ("failures", _set(("attempts", "failures"), 1), "attempt accounting"),
    ("min accuracy", _set(("attempts", "min_accuracy"), 0.9),
     "attempt accounting"),
    ("heldout", _set(("heldout_opened",), True), "heldout"),
    ("mode", _set(("mode",), "teach"), "mode differs"),
    ("schema", _set(("schema_version",), 2), "schema version"),
    ("extra", _set(("extra",), 1), "declared schema"),
])
def test_fit_receipt_rejects_failed_gate_and_attempt_mismatch(
        tmp_path, label, mutate, message):
    bundle = fit_bundle(tmp_path)
    receipt = resealed(bundle, mutate)
    with pytest.raises(bundle.module.RunnerError, match=message):
        validate_bundle(bundle, receipt, state=None)


@pytest.mark.parametrize("label, mutate, message", [
    ("run hash", _set(("teaching", "primary", "run_sha256"), "0" * 64),
     "hashes are stale"),
    ("cost hash", _set(("teaching", "primary", "cost_sha256"), "0" * 64),
     "hashes are stale"),
    ("p_only hash", _set(("teaching", "primary", "p_only_sha256"), "0" * 64),
     "hashes are stale"),
    ("checkpoint hash",
     _set(("teaching", "primary", "checkpoint_sha256"), "0" * 64),
     "hashes are stale"),
    ("canonical state",
     _set(("teaching", "primary", "canonical_state_sha256"), "0" * 64),
     "hashes are stale"),
    ("updates", _set(("teaching", "primary", "updates"), FIT_UPDATES - 1),
     "hashes are stale"),
    ("optimizer", _set(("teaching", "primary", "optimizer_steps"),
                       FIT_UPDATES - 1), "hashes are stale"),
    ("duty cycle", _set(("teaching", "primary", "duty_cycle"), 0.25),
     "hashes are stale"),
    ("theta stamp",
     _set(("teaching", "primary", "theta_stamp_unchanged"), False),
     "hashes are stale"),
    ("expert steps", _set(("teaching", "primary", "expert_steps"), [1]),
     "hashes are stale"),
])
def test_fit_receipt_rejects_stale_teaching_run_cost_or_state(
        tmp_path, label, mutate, message):
    bundle = fit_bundle(tmp_path)
    receipt = resealed(bundle, mutate)
    with pytest.raises(bundle.module.RunnerError, match=message):
        validate_bundle(bundle, receipt, state=None)


def test_teaching_bundle_rejects_a_run_without_state_hashes(tmp_path):
    bundle = fit_bundle(tmp_path)
    run = json.loads((bundle.teach / "run.json").read_text())
    for key in ("p_only_sha256", "checkpoint_sha256",
                "canonical_state_sha256"):
        run.pop(key)
    _write_json_file(bundle.teach / "run.json", run)
    with pytest.raises(bundle.module.RunnerError, match="lacks p_only_sha256"):
        bundle.module.teaching_bundle(bundle.teach,
                                      expected_updates=FIT_UPDATES)


def test_teaching_bundle_rejects_a_cost_report_from_another_run(tmp_path):
    bundle = fit_bundle(tmp_path)
    cost = json.loads((bundle.teach / "cost.json").read_text())
    cost["wall_seconds"] = 9.0
    cost["teaching_end_to_end_seconds"] = 9.0
    cost["duty_cycle"] = cost["learning_update_seconds"] / 9.0
    _write_json_file(bundle.teach / "cost.json", cost)
    with pytest.raises(bundle.module.RunnerError,
                       match="cost_sha256 differs"):
        bundle.module.teaching_bundle(bundle.teach,
                                      expected_updates=FIT_UPDATES)


def test_generation_failure_seals_a_failed_receipt_that_cannot_admit_dev(tmp_path):
    bundle = fit_bundle(tmp_path, failing_attempts=2)
    module = bundle.module
    receipt = module.load_fit_receipt(bundle.receipt_path)
    assert receipt["status"] == "failed"
    assert receipt["attempts"]["failures"] == 2
    assert receipt["attempts"]["correct"] == FIT_UPDATES - 2
    assert receipt["coverage"]["failed_attempts"] == 2
    assert receipt["coverage"]["audit_complete"] is False
    assert receipt["coverage"]["source_free"] is False
    with pytest.raises(module.RunnerError, match="status is 'failed'"):
        validate_bundle(bundle)
    counts = module.attempt_counts(scoring_attempts=FIT_UPDATES, failures=2)
    assert counts == {"attempts": FIT_UPDATES - 2, "failed_attempts": 2,
                      "scoring_attempts": FIT_UPDATES}
    with pytest.raises(module.RunnerError, match="failed attempts"):
        module.attempt_counts(scoring_attempts=FIT_UPDATES,
                              failures=FIT_UPDATES + 1)


def test_non_training_v2_never_requires_a_fit_receipt():
    module = runner_module()
    baseline = json.loads((ROOT / "configs" / "nested_text_nl1_v2.json").read_text())
    behavioral = json.loads(BEHAVIORAL_CONFIG.read_text())
    assert module.fit_receipt_required(baseline) is False
    assert module.fit_receipt_required(behavioral) is True
    args = SimpleNamespace(fit_receipt=None)
    assert module.require_fit_receipt(
        args, baseline, manifest={}, manifest_path=ROOT / "unused",
        episodes_root=ROOT) is None
    assert module.ROLE_ACCESS.admission == "protocol_without_fit_receipt"
    module.reset_role_access()
    with pytest.raises(module.RunnerError, match="fit receipt"):
        module.require_fit_receipt(
            args, behavioral, manifest={}, manifest_path=ROOT / "unused",
            episodes_root=ROOT)
    assert module.ROLE_ACCESS.admission is None
    module.reset_role_access()


def test_development_labels_stay_unreadable_until_full_receipt_admission(tmp_path):
    bundle = fit_bundle(tmp_path)
    module = bundle.module
    module.reset_role_access()
    failed = module.seal_fit_receipt(
        {**copy.deepcopy(bundle.receipt), "status": "failed"})
    args = _p_only_args(
        tmp_path,
        receipt_path=_write_json_file(tmp_path / "failed_receipt.json", failed),
        state_path=bundle.state_path, config_path=bundle.config_path)
    with pytest.raises(module.RunnerError, match="status is 'failed'"):
        module.require_fit_receipt(
            args, bundle.config, manifest=bundle.manifest,
            manifest_path=bundle.manifest_path,
            episodes_root=bundle.episodes)
    assert module.ROLE_ACCESS.admission is None
    with pytest.raises(module.RunnerError,
                       match="cannot be parsed before a validated fit receipt"):
        module.development_candidate_answers_for(bundle.episodes,
                                                 bundle.manifest)
    validate_bundle(bundle)
    assert module.ROLE_ACCESS.admission is None, (
        "recording validation must not itself open label parsing")
    module.ROLE_ACCESS.admit_fit_receipt(bundle.receipt)
    assert module.ROLE_ACCESS.admission == (
        f"fit_receipt:{bundle.receipt['receipt_sha256']}")
    try:
        assert module.development_candidate_answers_for(
            bundle.episodes, bundle.manifest)
    finally:
        module.reset_role_access()


def _p_only_args(tmp_path, *, receipt_path, state_path, config_path=None):
    return SimpleNamespace(
        protocol=config_path or BEHAVIORAL_CONFIG, episodes=v2_directory(),
        output=tmp_path / "eval", model_path=tmp_path / "model",
        model_lock=tmp_path / "model_lock.json", p_state=state_path,
        wrong_world_p=None, teach_cost=None, wrong_world_teach_cost=None,
        fit_receipt=receipt_path, max_prompt_tokens=256,
        max_response_tokens=64, max_dev=32, max_heldout=32,
        max_new_tokens=8, no_heldout=True, device="mps", world_role="primary")


@pytest.mark.parametrize("scenario, message", [
    ("missing", "fit receipt"),
    ("failed", "status is 'failed'"),
    ("other-state", "another P-state file"),
    ("other-config", "config"),
])
def test_behavioral_p_only_rejects_invalid_fit_evidence_before_native_load(
        tmp_path, monkeypatch, scenario, message):
    bundle = fit_bundle(tmp_path)
    module = bundle.module
    calls: list[str] = []

    def forbidden(name):
        def _raise(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} must not run without a valid receipt")
        return _raise

    monkeypatch.setattr(module.torch, "load", forbidden("torch.load"))
    monkeypatch.setattr(module, "exclusive_mps_worker", forbidden("worker"))
    monkeypatch.setattr(module, "load_local_mps", forbidden("model"))
    monkeypatch.setattr(module.TextAccessOrder, "access",
                        forbidden("development-access"))
    monkeypatch.setattr(module.TextAccessOrder, "advance",
                        forbidden("development-advance"))
    parsed_roles: list[str] = []
    original_load_rows = module.load_rows

    def watched_load_rows(directory, manifest, role):
        if role not in module.FIT_TARGET_ROLES:
            parsed_roles.append(role)
        return original_load_rows(directory, manifest, role)

    monkeypatch.setattr(module, "load_rows", watched_load_rows)
    receipt_path = bundle.receipt_path
    state_path = bundle.state_path
    config_path = bundle.config_path
    args = _p_only_args(tmp_path, receipt_path=receipt_path,
                        state_path=state_path, config_path=config_path)
    if scenario == "missing":
        args.fit_receipt = None
    elif scenario == "failed":
        failed = bundle.module.seal_fit_receipt(
            {**copy.deepcopy(bundle.receipt), "status": "failed"})
        args.fit_receipt = _write_json_file(tmp_path / "failed_receipt.json",
                                            failed)
    elif scenario == "other-state":
        other = tmp_path / "other_state.pt"
        other.write_bytes(bundle.state_path.read_bytes())
        args.p_state = other
    elif scenario == "other-config":
        other_config = _write_json_file(
            tmp_path / "other_config.json",
            {**bundle.config, "status": "other"})
        args.protocol = other_config
    with pytest.raises(module.RunnerError, match=message):
        module.run_p_only_v2(args, bundle.config)
    assert calls == [], "receipt admission must precede native/development access"
    assert parsed_roles == [], "development labels were parsed before admission"
    assert not (tmp_path / "eval" / "p_only_input.json").exists()
    bundle.module.reset_role_access()


def test_behavioral_p_only_accepts_the_bound_receipt_before_native_load(tmp_path,
                                                                        monkeypatch):
    """A valid receipt passes admission; the very next step is torch.load."""
    bundle = fit_bundle(tmp_path)
    module = bundle.module
    calls: list[str] = []

    def stop_at_load(*args, **kwargs):
        assert module.ROLE_ACCESS.admission == (
            f"fit_receipt:{bundle.receipt['receipt_sha256']}")
        calls.append("torch.load")
        raise AssertionError("admission reached torch.load")

    monkeypatch.setattr(module.torch, "load", stop_at_load)
    monkeypatch.setattr(module, "exclusive_mps_worker",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("worker must not run")))
    monkeypatch.setattr(module, "load_local_mps",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("model must not load")))
    monkeypatch.setattr(module.TextAccessOrder, "access",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("development must not be accessed")))
    args = _p_only_args(tmp_path, receipt_path=bundle.receipt_path,
                        state_path=bundle.state_path,
                        config_path=bundle.config_path)
    try:
        with pytest.raises(AssertionError, match="admission reached torch.load"):
            module.run_p_only_v2(args, bundle.config)
        assert calls == ["torch.load"]
    finally:
        module.reset_role_access()
