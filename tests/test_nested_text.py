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
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from replica_lab.architecture.contracts import (EvidencePin, MemoryRead,
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
    assert generation["scoring_version"] == "replica.nested_text.scoring.v2"
    assert generation["token_budget_credit"] is False
    workload = config["workload"]
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
