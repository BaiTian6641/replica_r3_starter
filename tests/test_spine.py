"""Mechanical internal-spine contracts; no native model or capability claims."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from replica_lab.architecture.spark2_5 import (  # noqa: E402
    build_extension_export, build_extensions, validate_extension_export,
)
from replica_lab.architecture.spine import (  # noqa: E402
    BRANCHES, PACKET_SCHEMA, SPINE_SCHEMA, DecisionSnapshot, Packet,
    PacketBuffer, SpineCore, SpineError, SpineForwardContext, SpineSpec,
)
from replica_lab.architecture.spine_checkpoint import (  # noqa: E402
    SPINE_CHECKPOINT_VERSION, load_spine_checkpoint, save_spine_checkpoint,
)
from replica_lab.architecture.spine_memory import (  # noqa: E402
    EVIDENCE_CHECKPOINT_FORMAT, EvidenceBank, EvidenceRecord,
    SpineMemoryExperiment,
)
from replica_lab.optimizers import NativeFusedAdamW  # noqa: E402
from replica_lab.architecture.spine_checkpoint import SPINE_MEMORY_CHECKPOINT_FORMAT  # noqa: E402
from test_spark2_5_adapter import build, ids_and_mask  # noqa: E402


def _packet(availability, *, payload=None, sequence=0, event_id="event-0",
            validity_end=None, deadline=None) -> Packet:
    if payload is None:
        slots = len(availability[0][0])
        payload = (torch.arange(slots * 512, dtype=torch.float32)
                   .reshape(1, slots, 512) / 1000.0)
    return Packet(
        schema=PACKET_SCHEMA,
        producer_revision="producer-1",
        representation_version="repr-1",
        codec_version="codec-1",
        scope="task",
        stream_id="stream",
        event_id=event_id,
        sequence=sequence,
        observation_start=0,
        observation_end=0,
        available_at=0,
        state_revision=7,
        payload_type="observation",
        source_lineage={"source": "cpu-fixture", "event": event_id},
        payload=payload,
        availability=torch.tensor(availability, dtype=torch.bool),
        validity_end=validity_end,
        deadline=deadline,
    )


def _snapshot() -> DecisionSnapshot:
    return DecisionSnapshot(
        decision_id="decision-1", scope="task", stream_id="stream",
        state_revision=7, arm="arm-0", event="event-0", phase="teaching")


def _open_ports(core: SpineCore) -> None:
    with torch.no_grad():
        for name, parameter in core.named_parameters():
            if name.endswith("output_projection.weight"):
                parameter.normal_(0.0, 0.02)


def _state(core: SpineCore) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in core.state_dict().items()}


def test_causal_masks_and_live_gradient_after_gates_open():
    core = SpineCore(SpineSpec(hidden_size=8, slots=3, sites=(3, 7)))
    availability = [[[True, True, False], [False, False, True]]]
    packet = _packet(availability)
    context = SpineForwardContext(core, packet, _snapshot())
    hidden = torch.randn(1, 2, 8)
    with torch.no_grad():
        initial = context.delta("spine", 3, hidden)
    assert torch.equal(initial, torch.zeros_like(hidden))

    _open_ports(core)
    opened = SpineForwardContext(core, packet, _snapshot())
    changed = opened.delta("spine", 3, hidden)
    assert bool(changed.abs().sum() > 0)
    loss = changed.square().mean()
    loss.backward()
    gradient = core.ports["spine"]["3"].output_projection.weight.grad
    assert gradient is not None and bool(torch.isfinite(gradient).all())
    assert bool(gradient.abs().sum() > 0)


def test_later_packet_cannot_change_earlier_query_or_fully_masked_delta():
    core = SpineCore(SpineSpec(hidden_size=8, slots=3, sites=(3, 7)))
    _open_ports(core)
    availability = [[[True, True, False], [False, False, True]]]
    first = _packet(availability)
    second_payload = first.payload.clone()
    second_payload[0, 2] += 100.0
    second = _packet(availability, payload=second_payload)
    left = SpineForwardContext(core, first, _snapshot())
    right = SpineForwardContext(core, second, _snapshot())
    hidden = torch.randn(1, 2, 8)
    left_delta = left.delta("spine", 3, hidden)
    right_delta = right.delta("spine", 3, hidden)
    torch.testing.assert_close(left_delta[:, 0], right_delta[:, 0],
                               rtol=0.0, atol=0.0)
    assert not torch.equal(left_delta[:, 1], right_delta[:, 1])

    masked = _packet([[[False, False, False], [False, False, False]]])
    masked_context = SpineForwardContext(core, masked, _snapshot())
    assert torch.equal(masked_context.delta("spine", 3, hidden),
                       torch.zeros_like(hidden))
    assert not bool(torch.isnan(masked_context.delta("spine", 3, hidden)).any())


def test_packet_buffer_deduplicates_retries_and_rejects_sequence_or_content_drift():
    buffer = PacketBuffer(max_packets=2, max_payload_bytes=1 << 20,
                          max_pending=2)
    packet = _packet([[[True]]], event_id="event-0")
    assert buffer.append(packet, now=0) is True
    assert buffer.append(packet, now=0) is False
    assert buffer.pending == 1
    changed = _packet([[[True]]], payload=packet.payload + 1.0)
    with pytest.raises(SpineError, match="changed payload"):
        buffer.append(changed, now=0)
    later = _packet([[[True]]], sequence=2, event_id="event-2")
    with pytest.raises(SpineError, match="missing or reordered"):
        buffer.append(later, now=0)
    expired = _packet([[[True]]], sequence=1, event_id="event-1",
                       validity_end=1, deadline=1)
    assert buffer.append(expired, now=0) is True
    assert buffer.expire(now=2) == 1
    assert buffer.pending == 1
    assert buffer.snapshot()[0].event_id == "event-0"


def test_packet_buffer_enforces_expiry_and_payload_limits():
    buffer = PacketBuffer(max_packets=1, max_payload_bytes=1, max_pending=1)
    packet = _packet([[[True]]], validity_end=1, deadline=1)
    with pytest.raises(SpineError, match="payload byte limit"):
        buffer.append(packet, now=0)
    small = PacketBuffer(max_packets=1, max_payload_bytes=1 << 20,
                         max_pending=1)
    assert small.append(packet, now=0) is True
    with pytest.raises(SpineError, match="queue/pending"):
        small.append(_packet([[[True]]], sequence=1, event_id="event-1"), now=0)
    assert small.expire(now=2) == 1
    assert small.pending == 0


def test_spark_adapter_native_spine_parity_influence_ablation_and_gradients():
    adapter, base, _, _ = build(spine_sites=(3, 7), spine_slots=3)
    ids, mask = ids_and_mask()
    context = SpineForwardContext(adapter.extension.spine,
                                  _packet([[[True, True, False],
                                            [False, False, True],
                                            [False, False, True],
                                            [False, False, True],
                                            [False, False, True],
                                            [False, False, True]]]), _snapshot())
    with torch.no_grad():
        baseline = base(input_ids=ids, attention_mask=mask, use_cache=False).logits
        initial = adapter.forward_text(
            ids, mask, use_projector=False, use_ports=False,
            spine=context, use_spine=True)
    torch.testing.assert_close(initial, baseline, rtol=0.0, atol=0.0)

    _open_ports(adapter.extension.spine)
    opened_context = SpineForwardContext(adapter.extension.spine,
                                         _packet([[[True, True, False],
                                                   [False, False, True],
                                                   [False, False, True],
                                                   [False, False, True],
                                                   [False, False, True],
                                                   [False, False, True]]]),
                                         _snapshot())
    changed = adapter.forward_text(
        ids, mask, use_projector=False, use_ports=False,
        spine=opened_context, use_spine=True)
    assert not torch.equal(changed, baseline)
    assert opened_context.charges["port_calls"] == 6
    assert opened_context.charges["codec_calls"] == 6
    assert opened_context.charges["codec_positions"] == 6
    assert opened_context.charges["exchange_rounds"] == 0
    assert opened_context.charges["specialist_calls"] == 0
    assert context.charges["codec_calls"] == 6
    assert context.charges["codec_positions"] == 6
    assert context.charges["exchange_rounds"] == 0
    assert context.charges["specialist_calls"] == 0
    empty = adapter.forward_text(
        ids, mask, use_projector=False, use_ports=False,
        spine=SpineForwardContext(adapter.extension.spine,
                                  _packet([[[True]]]), _snapshot(),
                                  enabled_branches=()),
        use_spine=True)
    torch.testing.assert_close(empty, baseline, rtol=0.0, atol=0.0)

    adapter.zero_grad(set_to_none=True)
    loss = adapter.causal_loss(changed, ids)
    loss.backward()
    audit = adapter.gradient_audit()
    assert audit["new_any_nonzero"] is True
    assert audit["new_all_finite"] is True
    assert audit["original"]["gradient_tensors"] == 0


def test_spine_v2_extension_inventory_and_reload(tmp_path):
    adapter, _, config, precision = build(spine_sites=(3, 7), spine_slots=3)
    metadata = adapter.extension_metadata()
    assert metadata["schema"] == "replica.spark2_5_extension_export.v2"
    assert metadata["extension"]["spine"]["spec"]["packet_width"] == 512
    assert metadata["extension"]["spine"]["spec"]["workspace_width"] == 256
    assert metadata["spine_migration"]["vision_projector_relabelled"] is False
    receipt = validate_extension_export(metadata, config, precision)
    assert receipt["schema"] == metadata["schema"]
    assert receipt["tensor_count"] == len(metadata["tensors"])
    assert receipt["parameter_elements"] > 0

    directory = tmp_path / "extension"
    adapter.save_extension(directory)
    fresh, _, _, _ = build(spine_sites=(3, 7), spine_slots=3)
    loaded = fresh.load_extension(directory)
    assert loaded["extension_only"] is True
    assert loaded["loaded_tensors"] == len(metadata["adapter_tensors"])
    assert fresh.durable_identity() == adapter.durable_identity()

    legacy = build_extension_export(config, precision, build_extensions(config))
    assert validate_extension_export(legacy, config, precision)["schema"] == (
        "replica.spark2_5_extension_export.v1")


def _contract() -> dict:
    return {"schema": SPINE_SCHEMA, "spine_schema": SPINE_SCHEMA,
            "model_identity": "a" * 64, "codec_version": "codec-1",
            "representation_version": "repr-1"}


def test_checkpoint_restores_active_state_and_subsequent_learning(tmp_path):
    spec = SpineSpec(hidden_size=8, slots=3, sites=(3, 7))
    torch.manual_seed(19)
    core = SpineCore(spec)
    optimizer = torch.optim.AdamW(core.parameters(), lr=1e-3)
    packet = _packet([[[True, True, False], [False, False, True]]])
    available = packet.availability.any(dim=1)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = core.codec(packet.payload, available).square().mean()
        loss.backward()
        optimizer.step()
    expected = _state(core)
    evaluations = [{"arm": "arm-0", "event": "event-0", "phase": "fit",
                    "digest": "eval-0"}]
    counters = {"optimizer_steps": 2, "forward_calls": 2, "backward_calls": 2}
    save_spine_checkpoint(
        tmp_path, "spine.pt", core, optimizer, arm="arm-0", event="event-0",
        phase="teaching", completed_evaluations=evaluations,
        counters=counters, run_contract=_contract())

    restored = SpineCore(spec)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    metadata = load_spine_checkpoint(
        tmp_path, "spine.pt", restored, restored_optimizer,
        run_contract=_contract())
    assert metadata["arm"] == "arm-0"
    assert metadata["event"] == "event-0"
    assert metadata["phase"] == "teaching"
    assert metadata["completed_evaluations"] == evaluations
    assert metadata["counters"] == counters
    for name, value in expected.items():
        torch.testing.assert_close(restored.state_dict()[name], value,
                                   rtol=0.0, atol=0.0)

    expected_optimizer = torch.optim.AdamW(core.parameters(), lr=1e-3)
    expected_optimizer.load_state_dict(optimizer.state_dict())
    for _ in range(1):
        expected_optimizer.zero_grad(set_to_none=True)
        expected_loss = core.codec(packet.payload, available).square().mean()
        expected_loss.backward()
        expected_optimizer.step()
        restored_optimizer.zero_grad(set_to_none=True)
        restored_loss = restored.codec(packet.payload, available).square().mean()
        restored_loss.backward()
        restored_optimizer.step()
    for name, value in core.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value,
                                   rtol=0.0, atol=0.0)
    with pytest.raises(SpineError, match="run_contract differs"):
        load_spine_checkpoint(
            tmp_path, "spine.pt", SpineCore(spec),
            torch.optim.AdamW(SpineCore(spec).parameters()),
            run_contract={**_contract(), "model_identity": "b" * 64})


def _evidence(record_id="record-0", **overrides):
    values = {
        "record_id": record_id, "scope": "task",
        "payload": torch.linspace(-1.0, 1.0, 512),
        "source_epoch": 0, "available_at": 0, "valid_from": 0,
        "producer_revision": "producer-1", "representation_version": "repr-1",
        "codec_version": "codec-1", "payload_type": "observation",
        "source_lineage": {"source": "cpu-fixture", "record": record_id},
    }
    values.update(overrides)
    return EvidenceRecord(**values)


def _memory_decision(bank, decision_id="memory-decision"):
    return DecisionSnapshot(
        decision_id=decision_id, scope="task", stream_id="memory",
        state_revision=bank.revision, arm="arm-0", event="event-0", phase="read")


def test_evidence_correction_revocation_and_atomic_capacity():
    bank = EvidenceBank("task", max_records=2, max_payload_bytes=2048)
    source = torch.linspace(-1.0, 1.0, 512)
    original = _evidence(payload=source)
    assert bank.admit(original) is True
    source.add_(100.0)
    assert bank.admit(original) is False
    packet = bank.snapshot(query_time=0, query_count=2,
                           decision=_memory_decision(bank))
    torch.testing.assert_close(packet.payload[0, 0], original.payload)
    stale = _memory_decision(bank, "stale-decision")
    correction = _evidence("record-1", source_epoch=1,
                           supersedes="record-0", payload=original.payload + 1.0)
    assert bank.admit(correction) is True
    assert bank.pending == 1
    assert len(bank.history) == 2
    with pytest.raises(SpineError, match="stale evidence revision"):
        bank.snapshot(query_time=0, query_count=1, decision=stale)
    with pytest.raises(SpineError, match="payload byte capacity"):
        bank.admit(_evidence("record-2", source_epoch=2,
                             supersedes="record-1",
                             payload=original.payload.double() + 2.0))
    assert bank.pending == 1
    assert bank.revision == 2
    current = bank.snapshot(query_time=0, query_count=1,
                            decision=_memory_decision(bank))
    assert current.source_lineage["record_ids"] == ["record-1"]
    with pytest.raises(SpineError, match="already used"):
        bank.admit(original)
    bank.revoke("record-1", reason="test revocation")
    empty = bank.snapshot(query_time=0, query_count=2,
                          decision=_memory_decision(bank))
    assert empty.payload.shape == (1, 0, 512)
    assert empty.availability.shape == (1, 2, 0)
    with pytest.raises(SpineError, match="revoked record"):
        bank.admit(correction)


def test_spine_memory_p_override_has_native_influence_and_gradients():
    torch.manual_seed(7)
    experiment = SpineMemoryExperiment(
        spec=SpineSpec(hidden_size=8, slots=3, sites=(3, 7)))
    experiment.observe(_evidence())
    baseline = experiment.snapshot(query_time=0, query_count=2,
                                   enabled_branches=("memory",))
    plastic = experiment.snapshot(query_time=0, query_count=2,
                                  plastic_expert_id=0, enabled_branches=("memory",))
    assert baseline.memory_source == "E"
    assert plastic.memory_source == "P"
    hidden = torch.randn(1, 2, 8)
    with torch.no_grad():
        assert torch.equal(baseline.delta("memory", 3, hidden),
                           plastic.delta("memory", 3, hidden))
        for port in experiment.core.ports["memory"].values():
            port.output_projection.weight.normal_(0.0, 0.02)
    baseline_delta = baseline.delta("memory", 3, hidden)
    plastic_delta = plastic.delta("memory", 3, hidden)
    assert bool(plastic_delta.abs().sum() > 0)
    assert not torch.equal(baseline_delta, plastic_delta)
    plastic_delta.square().mean().backward()
    expert_gradient = experiment.bank.expert(0).down.grad
    assert expert_gradient is not None and bool(torch.isfinite(expert_gradient).all())
    assert bool(expert_gradient.abs().sum() > 0)
    core_gradient = experiment.core.memory_projection.weight.grad
    assert core_gradient is not None and bool(core_gradient.abs().sum() > 0)
    all_branches = experiment.snapshot(query_time=0, query_count=2,
                                       plastic_expert_id=0)
    assert torch.equal(all_branches.delta("spine", 3, hidden),
                       torch.zeros_like(hidden))
    assert torch.equal(all_branches.delta("workspace", 3, hidden),
                       torch.zeros_like(hidden))
    with pytest.raises(SpineError, match="memory_slots must have shape"):
        SpineForwardContext(experiment.core, baseline.packet, baseline.snapshot,
                            memory_slots=torch.zeros(1, 1, 1, 256))


def test_spine_memory_revisions_invalidate_all_live_contexts():
    experiment = SpineMemoryExperiment(
        spec=SpineSpec(hidden_size=8, slots=3, sites=(3, 7)))
    experiment.observe(_evidence())
    first = experiment.snapshot(query_time=0, query_count=1)
    second = experiment.snapshot(query_time=0, query_count=1)
    experiment.observe(_evidence("record-1", source_epoch=1,
                                 supersedes="record-0"))
    assert first.invalidated and second.invalidated
    hidden = torch.randn(1, 1, 8)
    with pytest.raises(SpineError, match="decision context is stale"):
        first.delta("memory", 3, hidden)
    with pytest.raises(SpineError, match="decision context is stale"):
        second.delta("memory", 3, hidden)

    before_learning = experiment.snapshot(query_time=0, query_count=1)
    features, target = torch.randn(2, 256), torch.randn(2, 256)
    experiment.learn(event_id="event-0", features=features, target=target)
    assert before_learning.invalidated
    after_learning = experiment.snapshot(query_time=0, query_count=1)
    retry = experiment.learn(event_id="event-0", features=features, target=target)
    assert retry["outcome"]["retried"] is True
    assert not after_learning.invalidated
    experiment.clear_for_p_only()
    assert after_learning.invalidated


def test_spine_memory_checkpoint_restart_retry_and_p_only(tmp_path):
    torch.manual_seed(11)
    spec = SpineSpec(hidden_size=8, slots=3, sites=(3, 7))
    experiment = SpineMemoryExperiment(spec=spec)
    experiment.observe(_evidence())
    experiment.snapshot(query_time=0, query_count=1)
    features = torch.randn(2, 256)
    target = torch.randn(2, 256)
    first = experiment.learn(event_id="event-0", features=features, target=target)
    assert first["learning_evidence"]["optimizer_steps"] == 1
    saved = experiment.checkpoint(tmp_path, "memory.pt", run_contract=_contract())
    assert saved["metadata"]["format"] == SPINE_MEMORY_CHECKPOINT_FORMAT

    restored = SpineMemoryExperiment(spec=spec)
    restored.restore(tmp_path, "memory.pt", run_contract=_contract())
    assert restored.evidence.pending == 1
    assert len(restored.evidence.history) == 1
    assert restored.completed_evaluations == experiment.completed_evaluations
    for name, value in experiment.bank.state_dict().items():
        torch.testing.assert_close(restored.bank.state_dict()[name], value,
                                   rtol=0.0, atol=0.0)
    retry = restored.learn(event_id="event-0", features=features, target=target)
    assert retry["outcome"]["retried"] is True
    assert retry["learning_evidence"]["optimizer_steps"] == 0
    assert retry["learning_evidence"]["forward_calls"] == 0
    assert len(restored.completed_evaluations) == 1
    next_features = torch.randn(2, 256)
    next_target = torch.randn(2, 256)
    experiment.learn(event_id="event-1", features=next_features, target=next_target)
    restored.learn(event_id="event-1", features=next_features, target=next_target)
    for name, value in experiment.bank.state_dict().items():
        torch.testing.assert_close(restored.bank.state_dict()[name], value,
                                   rtol=0.0, atol=0.0)

    experiment.clear_for_p_only()
    assert experiment.evidence.pending == 0
    assert len(experiment.evidence.history) == 0
    experiment.checkpoint(tmp_path, "p_only.pt", run_contract=_contract())
    p_only = SpineMemoryExperiment(spec=spec)
    p_only.restore(tmp_path, "p_only.pt", run_contract=_contract())
    assert p_only.evidence.pending == 0
    assert len(p_only.evidence.history) == 0
    for name, value in experiment.bank.state_dict().items():
        torch.testing.assert_close(p_only.bank.state_dict()[name], value,
                                   rtol=0.0, atol=0.0)
    with pytest.raises(SpineError, match="owners differ"):
        load_spine_checkpoint(tmp_path, "memory.pt", SpineCore(spec),
                              torch.optim.AdamW(SpineCore(spec).parameters()),
                              run_contract=_contract())


def test_native_adapter_uses_experiment_owned_spine_and_plastic_read():
    adapter, base, _, _ = build(spine_sites=(3, 7))
    experiment = SpineMemoryExperiment(
        spec=adapter.spine_spec, core=adapter.extension.spine)
    assert experiment.core is adapter.extension.spine
    experiment.observe(_evidence())
    context = experiment.snapshot(query_time=0, query_count=6,
                                  plastic_expert_id=0,
                                  enabled_branches=("memory",))
    ids, mask = ids_and_mask()
    with torch.no_grad():
        baseline = base(input_ids=ids, attention_mask=mask, use_cache=False).logits
        initial = adapter.forward_text(
            ids, mask, use_projector=False, use_ports=False,
            spine=context, use_spine=True, enabled_spine_branches=("memory",))
    torch.testing.assert_close(initial, baseline, rtol=0.0, atol=0.0)
    with torch.no_grad():
        for port in experiment.core.ports["memory"].values():
            port.output_projection.weight.normal_(0.0, 0.02)
    changed = adapter.forward_text(
        ids, mask, use_projector=False, use_ports=False,
        spine=context, use_spine=True, enabled_spine_branches=("memory",))
    assert not torch.equal(changed, baseline)
    adapter.causal_loss(changed, ids).backward()
    assert experiment.bank.expert(0).down.grad is not None
    assert all(parameter.grad is None for parameter in base.parameters())


def test_nine_record_query_read_has_separate_storage_and_read_bounds():
    bank = EvidenceBank("task", max_records=16, max_read_records=4)
    chosen_payload = torch.zeros(512)
    chosen_payload[0] = 1.0
    for index in range(9):
        bank.admit(_evidence(
            f"record-{index}", source_epoch=index,
            payload=chosen_payload if index == 8 else torch.linspace(-1, 1, 512)))
    packet = bank.snapshot(
        query_time=0, query_count=1, decision=_memory_decision(bank),
        query_features=chosen_payload.unsqueeze(0))
    lineage = packet.source_lineage
    assert bank.pending == 9
    assert packet.slot_count == 4
    assert lineage["eligible_record_count"] == 9
    assert lineage["read_record_count"] == 4
    assert lineage["read_budget"] == 4
    assert lineage["record_ids"][0] == "record-8"
    assert lineage["omitted_record_count"] == 5
    assert len(lineage["omitted_records"]) == 5
    assert len(set(lineage["omitted_record_ids"])) == 5
    assert all(row["version"] and row["producer_revision"]
               for row in lineage["omitted_records"])
    assert bank.last_read["omitted_records"] == 5
    experiment = SpineMemoryExperiment(
        spec=SpineSpec(hidden_size=8, slots=4, sites=(3, 7)), read_records=4)
    for record in bank.active:
        experiment.observe(record)
    context = experiment.read(query_features=chosen_payload.unsqueeze(0),
                              enabled_branches=("memory",))
    assert context.packet.slot_count == 4
    assert context.slots["memory"].shape[:3] == (1, 1, 4)
    assert context.packet.source_lineage["omitted_record_count"] == 5


def test_packet_and_evidence_snapshots_own_payload_and_nested_lineage():
    bank = EvidenceBank("task")
    record = _evidence(source_lineage={"nested": {"source": "original"}})
    bank.admit(record)
    record.payload.add_(100)
    decision = _memory_decision(bank)
    first = bank.snapshot(query_time=0, query_count=1, decision=decision)
    torch.testing.assert_close(first.payload[0, 0], _evidence().payload)
    expected = first.payload.clone()
    first.payload.add_(50)
    first.source_lineage["record_ids"].append("foreign")
    second = bank.snapshot(query_time=0, query_count=1, decision=decision)
    torch.testing.assert_close(second.payload, expected)
    assert second.source_lineage["record_ids"] == ["record-0"]
    source = torch.ones(1, 1, 512)
    owned = _packet([[[True]]], payload=source)
    source.zero_()
    assert bool(owned.payload.abs().sum() > 0)


def test_evidence_history_revocation_metadata_and_representation_bounds_are_atomic():
    bounded = EvidenceBank("task", max_records=1, max_read_records=1,
                           max_history_records=1)
    original = _evidence()
    bounded.admit(original)
    with pytest.raises(SpineError, match="history record capacity"):
        bounded.admit(_evidence("record-1", source_epoch=1,
                                supersedes="record-0"))
    assert [row.record_id for row in bounded.active] == ["record-0"]
    assert bounded.revision == 1
    with pytest.raises(SpineError, match="representation/codec"):
        bounded.admit(_evidence("other-codec", codec_version="codec-2"))

    revocations = EvidenceBank("task", max_records=2, max_revocations=1)
    revocations.admit(_evidence())
    revocations.admit(_evidence("record-1", source_epoch=1))
    revocations.revoke("record-0", reason="first")
    with pytest.raises(SpineError, match="revocation capacity"):
        revocations.revoke("record-1", reason="second")
    assert [row.record_id for row in revocations.active] == ["record-1"]

    metadata = EvidenceBank("task", max_record_metadata_bytes=512)
    with pytest.raises(SpineError, match="metadata capacity"):
        metadata.admit(_evidence(source_lineage={"large": "x" * 2048}))
    assert metadata.pending == 0


def test_public_cache_bounds_accounting_and_complete_invalidation():
    experiment = SpineMemoryExperiment(
        spec=SpineSpec(hidden_size=8, slots=3, sites=(3, 7)),
        max_transient_entries=2)
    experiment.observe(_evidence())
    context = experiment.read(query_features=torch.randn(2, 512))
    caches = (experiment.source_cache, experiment.provider_cache,
              experiment.source_kv, experiment.fast_state,
              experiment.workspace_state)
    for cache in caches:
        cache["a"] = torch.ones(2)
        cache["b"] = torch.ones(2)
        cache["c"] = torch.ones(2)
        assert len(cache) == 2
        assert cache.evictions == 1
        owned = cache["b"]
        owned.zero_()
        assert bool(cache["b"].abs().sum() > 0)
    experiment.invalidate(reason="test_parameter_change")
    assert context.invalidated
    assert all(len(cache) == 0 for cache in caches)
    accounting = experiment.accounting()
    assert accounting["transient"]["live_contexts"] == 0
    assert accounting["caches"]["source_cache"]["evictions"] == 1
    assert accounting["journal"]["bound"] == experiment.max_journal_records
    assert accounting["replay"]["bound"] == experiment.max_journal_records
    assert accounting["metadata"]["evidence_record_bound"] > 0

    experiment.observe(_evidence("record-1", source_epoch=1))
    correction_context = experiment.read(query_features=torch.randn(1, 512))
    experiment.observe(_evidence("record-2", source_epoch=2,
                                 supersedes="record-1"))
    assert correction_context.invalidated
    revocation_context = experiment.read(query_features=torch.randn(1, 512))
    experiment.evidence.revoke("record-2", reason="direct public revocation")
    assert revocation_context.invalidated


def test_zero_lr_noop_and_retry_report_actual_work_without_learning_claim():
    experiment = SpineMemoryExperiment(
        spec=SpineSpec(hidden_size=8, slots=3, sites=(3, 7)))
    experiment.observe(_evidence())
    experiment.bank.optimizer.param_groups[0]["lr"] = 0.0
    features, target = torch.randn(2, 256), torch.randn(2, 256)
    before = _state(experiment.bank)
    result = experiment.learn(event_id="zero-lr", features=features, target=target)
    evidence = result["learning_evidence"]
    assert evidence["execution_complete"] is True
    assert evidence["optimizer_steps"] == 1
    assert evidence["parameter_update"] is False
    assert evidence["changed_trainable_names"] == []
    assert evidence["changed_tensor_count"] == 0
    assert evidence["train_fit"]["evaluated"] is True
    assert evidence["train_fit"]["improved"] is False
    assert evidence["transfer"] == {"evaluated": False, "improved": None}
    assert evidence["retention"] == {"evaluated": False, "retained": None}
    assert evidence["useful_learning"] is False
    assert evidence["learning_success"] is False
    assert evidence["status"] == "no_op"
    assert experiment.counters["learning_events"] == 0
    assert experiment.counters["learning_transactions"] == 1
    for name, value in before.items():
        torch.testing.assert_close(experiment.bank.state_dict()[name], value,
                                   rtol=0.0, atol=0.0)

    counters = dict(experiment.counters)
    retry = experiment.learn(event_id="zero-lr", features=features, target=target)
    retried = retry["learning_evidence"]
    assert retried["retried"] is True
    assert retried["optimizer_steps"] == 0
    assert retried["forward_calls"] == 0
    assert retried["backward_calls"] == 0
    assert retried["fit_evaluation_forward_calls"] == 0
    assert retried["train_fit"]["evaluated"] is False
    assert experiment.counters["learning_transactions"] == counters["learning_transactions"]
    assert experiment.counters["optimizer_steps"] == counters["optimizer_steps"]
    assert experiment.counters["forward_calls"] == counters["forward_calls"]
    assert experiment.counters["fit_evaluation_forward_calls"] == counters[
        "fit_evaluation_forward_calls"]
    assert len(experiment.completed_evaluations) == 1
    assert len(experiment.learning_journal) == 1


def test_learning_fields_and_bounded_replay_journal_track_actual_changes():
    experiment = SpineMemoryExperiment(
        spec=SpineSpec(hidden_size=8, slots=3, sites=(3, 7)),
        max_journal_records=1)
    experiment.observe(_evidence())
    for index in range(2):
        result = experiment.learn(
            event_id=f"fit-{index}", features=torch.randn(2, 256),
            target=torch.randn(2, 256))
        evidence = result["learning_evidence"]
        actual = [name for name, parameter in experiment.bank.expert(0).named_parameters()
                  if parameter.grad is not None]
        assert evidence["execution_complete"] is True
        assert evidence["parameter_update"] == bool(evidence["changed_trainable_names"])
        assert evidence["changed_tensor_count"] == len(evidence["changed_trainable_names"])
        assert evidence["train_fit"]["evaluated"] is True
        assert evidence["learning_success"] == evidence["useful_learning"]
        assert evidence["transfer"]["evaluated"] is False
        assert evidence["retention"]["evaluated"] is False
        assert actual
    assert len(experiment.learning_journal) == 1
    assert len(experiment.completed_evaluations) == 1
    assert experiment.counters["dropped_journal_records"] >= 2
    row = experiment.learning_journal[0]
    assert row["event_id"] == "fit-1"
    assert row["evidence_versions"]
    assert row["replay"]["features_sha256"]
    assert row["replay"]["target_sha256"]


def test_p_query_read_survives_e_clear_and_uses_question_and_native_hidden():
    torch.manual_seed(101)
    experiment = SpineMemoryExperiment(
        spec=SpineSpec(hidden_size=8, slots=3, sites=(3, 7)))
    experiment.observe(_evidence())
    experiment.learn(event_id="teach-p", features=torch.randn(2, 256),
                     target=torch.randn(2, 256))
    taught = dict(experiment.taught_evidence)
    stale = experiment.snapshot(query_time=0, query_count=1)
    experiment.clear_for_p_only()
    assert stale.invalidated
    assert experiment.evidence.pending == 0
    assert len(experiment.evidence.history) == 0
    assert dict(experiment.taught_evidence) == taught
    _open_ports(experiment.core)
    with pytest.raises(SpineError, match="requires query_features"):
        experiment.snapshot(query_time=0, query_count=1, plastic_expert_id=0)

    question = torch.randn(1, 2, 512)
    native = torch.randn(1, 2, 8)
    first = experiment.read(query_features=question, native_hidden=native,
                            plastic_expert_id=0)
    other_question = experiment.read(query_features=question + 1.0,
                                     native_hidden=native, plastic_expert_id=0)
    other_native = experiment.read(query_features=question,
                                   native_hidden=native + 1.0,
                                   plastic_expert_id=0)
    assert first.memory_source == "P"
    assert first.packet.slot_count == 0
    assert first.packet.source_lineage["evidence_record_ids"] == []
    assert first.slots["memory"].shape == (1, 2, 1, 256)
    assert first.charges["memory_read_slots"] == 1
    assert not torch.equal(first.slots["memory"], other_question.slots["memory"])
    assert not torch.equal(first.slots["memory"], other_native.slots["memory"])
    delta = first.delta("memory", 3, native)
    assert bool(delta.abs().sum() > 0)
    delta.square().mean().backward()
    assert experiment.bank.expert(0).down.grad is not None


def test_fresh_process_p_rebuild_needs_no_derived_caches_and_invalidates_restore(tmp_path):
    experiment = SpineMemoryExperiment(
        spec=SpineSpec(hidden_size=8, slots=3, sites=(3, 7)))
    experiment.observe(_evidence())
    experiment.learn(event_id="persist-p", features=torch.randn(2, 256),
                     target=torch.randn(2, 256))
    experiment.clear_for_p_only()
    experiment.source_cache["derived"] = torch.ones(4)
    saved = experiment.checkpoint(tmp_path, "rebuild.pt", run_contract=_contract())
    assert saved["metadata"]["version"] == SPINE_CHECKPOINT_VERSION

    restored = SpineMemoryExperiment(
        spec=SpineSpec(hidden_size=8, slots=3, sites=(3, 7)))
    stale = restored.read(query_features=torch.randn(1, 512),
                          plastic_expert_id=0)
    metadata = restored.restore(tmp_path, "rebuild.pt", run_contract=_contract())
    assert metadata["checkpoint_version"] == SPINE_CHECKPOINT_VERSION
    assert stale.invalidated
    assert len(restored.source_cache) == 0
    assert restored.taught_evidence == experiment.taught_evidence
    assert restored.learning_journal == experiment.learning_journal
    context = restored.read(query_features=torch.randn(1, 512),
                            plastic_expert_id=0)
    assert context.memory_source == "P"
    assert context.packet.slot_count == 0
    assert context.slots["memory"].shape == (1, 1, 1, 256)


def test_v2_memory_key_is_required_and_legacy_evidence_migrates_explicitly(tmp_path):
    experiment = SpineMemoryExperiment(
        spec=SpineSpec(hidden_size=8, slots=3, sites=(3, 7)))
    experiment.checkpoint(tmp_path, "strict.pt", run_contract=_contract())
    payload = torch.load(tmp_path / "strict.pt", map_location="cpu", weights_only=True)
    del payload["memory_state"]
    torch.save(payload, tmp_path / "missing-memory.pt")
    with pytest.raises(SpineError, match="keys differ"):
        SpineMemoryExperiment(
            spec=SpineSpec(hidden_size=8, slots=3, sites=(3, 7))).restore(
                tmp_path, "missing-memory.pt", run_contract=_contract())
    missing_counter = torch.load(tmp_path / "strict.pt", map_location="cpu",
                                 weights_only=True)
    del missing_counter["counters"]["read_calls"]
    torch.save(missing_counter, tmp_path / "missing-counter.pt")
    with pytest.raises(SpineError, match="counter inventory"):
        SpineMemoryExperiment(
            spec=SpineSpec(hidden_size=8, slots=3, sites=(3, 7))).restore(
                tmp_path, "missing-counter.pt", run_contract=_contract())

    bank = EvidenceBank("task")
    bank.admit(_evidence())
    current = bank.checkpoint_payload()
    legacy = {key: current[key] for key in (
        "scope", "max_records", "max_payload_bytes", "revision", "active",
        "history", "revocations", "payload_bytes")}
    legacy["format"] = "replica.spark.evidence.v1"
    migrated = EvidenceBank("task")
    migrated.load_checkpoint(legacy)
    assert migrated.pending == 1
    assert migrated.checkpoint_payload()["format"] == EVIDENCE_CHECKPOINT_FORMAT


def test_legacy_plain_adamw_checkpoint_migrates_only_to_native_fused(tmp_path):
    spec = SpineSpec(hidden_size=8, slots=3, sites=(3, 7))
    core = SpineCore(spec)
    plain = torch.optim.AdamW(core.parameters(), lr=1e-3, weight_decay=0.0)
    save_spine_checkpoint(
        tmp_path, "legacy-core.pt", core, plain, arm="arm-0", event="event-0",
        phase="fit", completed_evaluations=[], counters={}, run_contract=_contract())
    payload = torch.load(tmp_path / "legacy-core.pt", map_location="cpu",
                         weights_only=True)
    payload["version"] = 1
    torch.save(payload, tmp_path / "legacy-core.pt")
    restored = SpineCore(spec)
    parameters = list(restored.parameters())
    native = NativeFusedAdamW([{
        "params": parameters, "lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8,
        "weight_decay": 0.0, "algorithm": "adamw_fused", "fused": True,
        "group_name": "spine_core", "scope": "task/spine",
        "param_names": [name for name, _ in restored.named_parameters()],
    }])
    metadata = load_spine_checkpoint(
        tmp_path, "legacy-core.pt", restored, native, run_contract=_contract())
    assert metadata["optimizer_migration"] == "torch_adamw_v1_to_native_fused_v1"
    assert native.audit()["implementation"] == "native_fused"

    incompatible = dict(payload)
    incompatible["optimizer_state"] = {"state": {}, "param_groups": [],
                                       "custom": "muon"}
    torch.save(incompatible, tmp_path / "custom-optimizer.pt")
    with pytest.raises(SpineError, match="plain AdamW"):
        load_spine_checkpoint(tmp_path, "custom-optimizer.pt", SpineCore(spec),
                              native, run_contract=_contract())


def test_e_plus_p_combines_late_e_and_p_with_gradients_and_null_parity():
    torch.manual_seed(131)
    core = SpineCore(SpineSpec(hidden_size=8, slots=3, sites=(3, 7)))
    packet = _packet([[[True, False], [False, False]]])
    p_slots = torch.randn(1, 2, 1, 256, requires_grad=True)
    hidden = torch.randn(1, 2, 8)
    null_context = SpineForwardContext(
        core, packet, _snapshot(), enabled_branches=("memory",),
        memory_slots=p_slots, memory_source="E_plus_P")
    with torch.no_grad():
        assert torch.equal(null_context.delta("memory", 3, hidden),
                           torch.zeros_like(hidden))

    _open_ports(core)
    e_context = SpineForwardContext(
        core, packet, _snapshot(), enabled_branches=("memory",))
    p_context = SpineForwardContext(
        core, packet, _snapshot(), enabled_branches=("memory",),
        memory_slots=p_slots, memory_source="P")
    combined = SpineForwardContext(
        core, packet, _snapshot(), enabled_branches=("memory",),
        memory_slots=p_slots, memory_source="E_plus_P")
    e_delta = e_context.delta("memory", 3, hidden)
    p_delta = p_context.delta("memory", 3, hidden)
    combined_delta = combined.delta("memory", 3, hidden)
    torch.testing.assert_close(combined_delta, e_delta + p_delta,
                               rtol=0.0, atol=0.0)
    assert bool(e_delta[:, 0].abs().sum() > 0)
    assert bool(p_delta[:, 0].abs().sum() > 0)
    assert torch.equal(e_delta[:, 1], torch.zeros_like(e_delta[:, 1]))
    torch.testing.assert_close(combined_delta[:, 1], p_delta[:, 1],
                               rtol=0.0, atol=0.0)
    assert combined.charges["memory_read_slots"] == 3
    assert combined.charges["e_memory_read_slots"] == 2
    assert combined.charges["p_memory_read_slots"] == 1
    assert combined.charges["port_calls"] == 2
    combined_delta.square().mean().backward()
    assert p_slots.grad is not None
    assert bool(torch.isfinite(p_slots.grad).all())
    assert bool(p_slots.grad.abs().sum() > 0)
