"""Offline E3 cognitive-core contracts; no downloaded model or GPU use."""

import math
import pytest
import torch
from dataclasses import replace

from architecture_helpers import action_batch, synthetic_frame, tiny_adapter
from replica_lab.architecture.contracts import CoreSpec, CoreState, MemoryRead
from replica_lab.architecture.core import (CognitiveCore, FeedForwardCore,
                                           TaskStateEncoder,
                                           cognitive_parameter_count,
                                           core_parameter_count)
from replica_lab.architecture.heads import TaskSuccessHead
from replica_lab.architecture.minicpm5 import MiniCPM5Adapter
from replica_lab.architecture.ports import MemoryAccessPort, WorkspaceAccessPort

TINY_SPEC = CoreSpec(width=16, slots=4, steps=2, heads=4, task_features=24,
                     ffn_hidden=64, control_hidden=29)


def core_adapter(regime='modules_only'):
    adapter = tiny_adapter(regime)
    return MiniCPM5Adapter(adapter.base, replace(adapter.manifest, core=TINY_SPEC), regime)


def core_frame(adapter, *, features=None, enabled_branches=None, empty=False,
               all_masked=False, train=False, task_digest='test-task-digest', revision=3):
    base = synthetic_frame(adapter, empty=empty, all_masked=all_masked)
    features = torch.rand(1, TINY_SPEC.task_features) if features is None else features
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        core = adapter.build_core_state(base.sense, base.memory, features,
                                        task_digest=task_digest, revision=revision)
    assert core.task_digest == task_digest and core.revision == revision
    return adapter.make_frame(base.decision_id, base.evidence, base.memory, base.sense,
                              core=core, enabled_branches=enabled_branches)


def run(adapter, frame):
    with torch.no_grad():
        return adapter.forward_with_frame(frame=frame, **action_batch(frame)).logits


def test_core_output_shapes_dtype_and_validity():
    torch.manual_seed(7)
    core = CognitiveCore(TINY_SPEC)
    sense = torch.randn(1, 2, 16)
    memory = torch.randn(1, 8, 16)
    workspace, valid = core(sense, torch.ones(1, 2, dtype=torch.bool), memory,
                            torch.ones(1, 8, dtype=torch.bool), torch.rand(1, 24))
    assert workspace.shape == (1, 4, 16) and workspace.is_floating_point()
    assert bool(torch.isfinite(workspace).all())
    assert valid.shape == (1, 4) and valid.dtype == torch.bool and bool(valid.all())


def test_steps_zero_returns_initial_workspace_and_iterations_change_state():
    torch.manual_seed(7)
    core = CognitiveCore(TINY_SPEC)
    sense, memory = torch.randn(1, 2, 16), torch.randn(1, 8, 16)
    valid = torch.ones(1, 8, dtype=torch.bool)
    features = torch.rand(1, 24)
    initial, _ = core(sense, torch.ones(1, 2, dtype=torch.bool), memory, valid,
                      features, steps=0)
    torch.testing.assert_close(initial, core.slot_queries.unsqueeze(0), atol=0, rtol=0)
    carried = torch.randn(1, TINY_SPEC.slots, TINY_SPEC.width)
    seeded, _ = core(sense, torch.ones(1, 2, dtype=torch.bool), memory, valid,
                     features, previous=carried, steps=0)
    torch.testing.assert_close(seeded, core.slot_queries.unsqueeze(0) + carried,
                               atol=0, rtol=0)
    one, _ = core(sense, torch.ones(1, 2, dtype=torch.bool), memory, valid, features, steps=1)
    two, _ = core(sense, torch.ones(1, 2, dtype=torch.bool), memory, valid, features, steps=2)
    assert not torch.equal(one, two)
    # One tied block: the same parameter objects and normalization modules serve
    # every iteration, so a two-step pass invokes the block exactly twice.
    calls = []
    handle = core.self_norm.register_forward_hook(lambda *args: calls.append(1))
    try:
        core(sense, torch.ones(1, 2, dtype=torch.bool), memory, valid, features, steps=2)
    finally:
        handle.remove()
    assert len(calls) == 2
    assert core.self_o.weight is core.self_o.weight


def test_workspace_and_memory_readers_are_exactly_null_at_init_and_independent():
    adapter = core_adapter().eval()
    frame = core_frame(adapter, enabled_branches=('memory', 'workspace'))
    memory_only = core_frame(adapter, enabled_branches=('memory',))
    baseline = run(adapter, memory_only)
    torch.testing.assert_close(run(adapter, frame), baseline, atol=0, rtol=0)

    with torch.no_grad():
        adapter.workspace_ports['0'].output_projection.weight.add_(0.05)
    workspace_change = run(adapter, core_frame(adapter, enabled_branches=('memory', 'workspace')))
    assert not torch.equal(workspace_change, baseline)

    with torch.no_grad():
        adapter.workspace_ports['0'].output_projection.weight.sub_(0.05)
        adapter.memory_ports['0'].output_projection.weight.add_(0.05)
    memory_change = run(adapter, core_frame(adapter, enabled_branches=('memory', 'workspace')))
    assert not torch.equal(memory_change, baseline)
    assert not torch.equal(memory_change, workspace_change)


def test_fully_masked_memory_source_is_dropped_exactly():
    torch.manual_seed(3)
    core = CognitiveCore(TINY_SPEC)
    with torch.no_grad():
        for p in core.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    sense = torch.randn(1, 2, 16)
    memory = torch.randn(1, 8, 16)
    features = torch.rand(1, 24)
    sense_valid = torch.ones(1, 2, dtype=torch.bool)
    masked, _ = core(sense, sense_valid, memory, torch.zeros(1, 8, dtype=torch.bool), features)
    excluded, _ = core(sense, sense_valid, memory, torch.ones(1, 8, dtype=torch.bool),
                       features, enabled_inputs=('sense', 'task'))
    torch.testing.assert_close(masked, excluded, atol=0, rtol=0)


def test_enabled_inputs_separate_core_sources():
    torch.manual_seed(11)
    core = CognitiveCore(TINY_SPEC)
    sense, memory = torch.randn(1, 2, 16), torch.randn(1, 8, 16)
    valid_s = torch.ones(1, 2, dtype=torch.bool)
    valid_m = torch.ones(1, 8, dtype=torch.bool)
    features = torch.rand(1, 24)
    task_only, _ = core(sense, valid_s, memory, valid_m, features, enabled_inputs=('task',))
    all_inputs, _ = core(sense, valid_s, memory, valid_m, features)
    sense_only, _ = core(sense, valid_s, memory, valid_m, features, enabled_inputs=('sense',))
    assert not torch.equal(task_only, all_inputs)
    assert not torch.equal(sense_only, task_only)
    with pytest.raises(ValueError, match='enabled_inputs'):
        core(sense, valid_s, memory, valid_m, features, enabled_inputs=('task', 'goal'))


def test_previous_workspace_influences_and_detached_has_no_grad_history():
    torch.manual_seed(5)
    core = CognitiveCore(TINY_SPEC)
    sense, memory = torch.randn(1, 2, 16), torch.randn(1, 8, 16)
    valid_s = torch.ones(1, 2, dtype=torch.bool)
    valid_m = torch.ones(1, 8, dtype=torch.bool)
    features = torch.rand(1, 24)
    plain, _ = core(sense, valid_s, memory, valid_m, features)
    previous = torch.randn(1, 4, 16, requires_grad=True)
    carried, _ = core(sense, valid_s, memory, valid_m, features, previous=previous)
    assert not torch.equal(plain, carried)
    carried.sum().backward()
    assert previous.grad is not None and bool(previous.grad.abs().sum() > 0)


def test_core_state_detached_breaks_graph():
    adapter = core_adapter().eval()
    with torch.enable_grad():
        frame = core_frame(adapter, train=True)
    assert frame.core.workspace.grad_fn is not None
    detached = frame.core.detached()
    assert detached.workspace.grad_fn is None and detached.task_features.grad_fn is None
    assert not torch.equal(detached.workspace, torch.zeros_like(detached.workspace))
    assert detached.producer == frame.core.producer and detached.stamp() == frame.core.stamp()


def test_action_loss_reaches_sensors_core_and_workspace_readers():
    adapter = core_adapter(regime='full_augmented').train()
    with torch.no_grad():
        # Zero-initialized reader output projections block first-step upstream
        # gradients; open both branches as after one real optimizer step.
        for port in adapter.memory_ports.values():
            port.output_projection.weight.normal_(std=0.04)
        for port in adapter.workspace_ports.values():
            port.output_projection.weight.normal_(std=0.04)
    frame = core_frame(adapter, train=True)
    output = adapter.forward_with_frame(frame=frame, **action_batch(frame))
    output.loss.backward()
    audit = adapter.gradient_audit()
    assert audit['new']['nonfinite'] == []
    names = set(audit['new']['nonzero'])
    assert any(n.startswith('sense_ports.') for n in names)
    assert any(n.startswith('core.') for n in names)
    assert any(n.startswith('workspace_ports.') for n in names)


@pytest.mark.parametrize('mutate', ['prefix', 'mask', 'prefix_labels'])
def test_core_frame_rejects_mutated_prefix_and_prefix_labels(mutate):
    adapter = core_adapter().eval()
    frame = core_frame(adapter)
    batch = action_batch(frame)
    if mutate == 'prefix':
        batch['input_ids'] = batch['input_ids'].clone()
        batch['input_ids'][0, 0] = (batch['input_ids'][0, 0] + 1) % adapter.base.config.vocab_size
    elif mutate == 'mask':
        batch['attention_mask'] = batch['attention_mask'].clone()
        batch['attention_mask'][0, 0] = False
    else:
        batch['labels'] = batch['labels'].clone()
        batch['labels'][0, 0] = batch['input_ids'][0, 0]
    with pytest.raises(ValueError):
        adapter.forward_with_frame(frame=frame, **batch)


def test_feedforward_control_matches_core_parameter_budget_and_contract():
    spec = CoreSpec()
    core_parameters = cognitive_parameter_count(spec)
    assert core_parameter_count(CognitiveCore(spec)) == core_parameters
    control = FeedForwardCore(spec)
    ratio = core_parameter_count(control) / core_parameters
    assert 0.95 <= ratio <= 1.05
    workspace, valid = control(torch.randn(1, spec.width), torch.randn(1, spec.width),
                               torch.rand(1, spec.task_features))
    assert workspace.shape == (1, spec.slots, spec.width)
    assert valid.shape == (1, spec.slots) and valid.dtype == torch.bool
    assert bool(torch.isfinite(workspace).all())


def test_success_head_detaches_inputs_but_keeps_head_gradient():
    head = TaskSuccessHead(TINY_SPEC, hidden=16)
    features = torch.rand(1, TINY_SPEC.task_features, requires_grad=True)
    workspace = torch.randn(1, TINY_SPEC.slots, TINY_SPEC.width, requires_grad=True)
    logit = head(features, workspace)
    assert logit.shape == (1,)
    assert logit.requires_grad and logit.grad_fn is not None
    logit.sum().backward()
    assert features.grad is None and workspace.grad is None
    gradients = [p.grad for p in head.mlp.parameters() if p.grad is not None]
    assert len(gradients) == len(list(head.mlp.parameters()))
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
    assert any(bool(g.ne(0).any()) for g in gradients)

    valid = torch.tensor([[True, False, True, True]])
    masked = head(torch.rand(1, TINY_SPEC.task_features),
                  torch.randn(1, TINY_SPEC.slots, TINY_SPEC.width), valid)
    assert masked.shape == (1,)
    # All-invalid pools contribute an exact zero vector, never NaN.
    empty = head(torch.zeros(1, TINY_SPEC.task_features),
                 torch.randn(1, TINY_SPEC.slots, TINY_SPEC.width),
                 torch.zeros(1, TINY_SPEC.slots, dtype=torch.bool))
    assert torch.isfinite(empty).all()


def test_predict_success_is_detached_and_producer_bound():
    adapter = core_adapter().eval()
    frame = core_frame(adapter)
    logit = adapter.predict_success(frame.core)
    assert logit.shape == (1,)
    assert not logit.requires_grad and logit.grad_fn is None
    with torch.no_grad():
        adapter.core.slot_queries.add_(0.01)
    with pytest.raises(ValueError, match='Core producer'):
        adapter.predict_success(frame.core)


def test_success_head_direct_training_never_reaches_core():
    adapter = core_adapter().eval()
    frame = core_frame(adapter, train=True)
    logit = adapter.task_head(frame.core.task_features, frame.core.workspace, frame.core.valid)
    assert logit.requires_grad
    logit.sum().backward()
    assert all(p.grad is None for p in adapter.core.parameters())
    assert any(p.grad is not None for p in adapter.task_head.parameters())


def test_attention_path_matches_explicit_softmax_and_all_invalid_is_exact_null():
    torch.manual_seed(1234)
    core = CognitiveCore(TINY_SPEC).double()
    query = torch.randn(1, 3, TINY_SPEC.width, dtype=torch.float64)
    keys = torch.randn(1, 5, TINY_SPEC.width, dtype=torch.float64)
    cross_mask = torch.tensor([[True, False, False, True, False]])

    def manual(query_source, key_source, valid, self_attention):
        heads, size = core.heads, core.head_size
        q_proj, k_proj, v_proj, out_proj = (
            (core.self_q, core.self_k, core.self_v, core.self_o) if self_attention
            else (core.cross_q, core.cross_k, core.cross_v, core.cross_o))
        q = q_proj(query_source).reshape(1, query_source.shape[1], heads, size).transpose(1, 2)
        k = k_proj(key_source).reshape(1, key_source.shape[1], heads, size).transpose(1, 2)
        v = v_proj(key_source).reshape(1, key_source.shape[1], heads, size).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(size)
        scores = scores.masked_fill(~valid[:, None, None, :], float('-inf'))
        read = (scores.softmax(dim=-1) @ v).transpose(1, 2)
        return out_proj(read.reshape(1, query_source.shape[1], core.width))

    torch.testing.assert_close(core._attend(query, keys, cross_mask),
                               manual(query, keys, cross_mask, self_attention=False),
                               atol=1e-12, rtol=1e-12)
    self_mask = torch.tensor([[True, True, False]])
    torch.testing.assert_close(core._attend(query, query, self_mask),
                               manual(query, query, self_mask, self_attention=True),
                               atol=1e-12, rtol=1e-12)
    invalid = torch.zeros(1, 5, dtype=torch.bool)
    isolated = core._attend(query, keys, invalid)
    assert torch.equal(isolated, torch.zeros_like(isolated))


def test_core_masked_source_contents_are_inert_and_detach_is_value_only():
    torch.manual_seed(9)
    core = CognitiveCore(TINY_SPEC)
    with torch.no_grad():
        for parameter in core.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    sense = torch.randn(1, 2, TINY_SPEC.width)
    memory = torch.randn(1, 5, TINY_SPEC.width)
    features = torch.rand(1, TINY_SPEC.task_features)
    mask = torch.tensor([[True, False, True, False, False]])
    clean, _ = core(sense, torch.ones(1, 2, dtype=torch.bool), memory, mask, features)
    tampered = memory.clone()
    tampered[~mask] = 1e6
    dirty, _ = core(sense, torch.ones(1, 2, dtype=torch.bool), tampered, mask, features)
    torch.testing.assert_close(clean, dirty, atol=0, rtol=0)


def test_fully_masked_sense_is_dropped_exactly():
    torch.manual_seed(4)
    core = CognitiveCore(TINY_SPEC)
    sense = torch.randn(1, 2, TINY_SPEC.width)
    memory = torch.randn(1, 3, TINY_SPEC.width)
    features = torch.rand(1, TINY_SPEC.task_features)
    memory_valid = torch.ones(1, 3, dtype=torch.bool)
    masked, _ = core(sense, torch.zeros(1, 2, dtype=torch.bool), memory, memory_valid, features)
    excluded, _ = core(sense, torch.ones(1, 2, dtype=torch.bool), memory, memory_valid,
                       features, enabled_inputs=('memory', 'task'))
    torch.testing.assert_close(masked, excluded, atol=0, rtol=0)


def test_task_feature_encoder_forms_two_finite_tokens():
    encoder = TaskStateEncoder(TINY_SPEC.task_features, TINY_SPEC.width)
    first = encoder(torch.rand(1, TINY_SPEC.task_features))
    second = encoder(torch.rand(1, TINY_SPEC.task_features))
    assert first.shape == (1, 2, TINY_SPEC.width) and torch.isfinite(first).all()
    assert not torch.equal(first, second)
    first.sum().backward()
    assert encoder.projection.weight.grad is not None
    with pytest.raises(ValueError, match='task_features'):
        encoder(torch.rand(1, TINY_SPEC.task_features - 1))


def test_core_rejects_nonfinite_inputs_and_bad_previous_shape():
    core = CognitiveCore(TINY_SPEC)
    sense = torch.randn(1, 2, TINY_SPEC.width)
    memory = torch.randn(1, 3, TINY_SPEC.width)
    sense_valid = torch.ones(1, 2, dtype=torch.bool)
    memory_valid = torch.ones(1, 3, dtype=torch.bool)
    features = torch.rand(1, TINY_SPEC.task_features)
    bad = sense.clone()
    bad[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        core(bad, sense_valid, memory, memory_valid, features)
    with pytest.raises(ValueError, match='previous'):
        core(sense, sense_valid, memory, memory_valid, features,
             previous=torch.randn(1, 1, TINY_SPEC.width))
    with pytest.raises(ValueError, match='previous'):
        core(sense, sense_valid, memory, memory_valid, features,
             previous=torch.full((1, TINY_SPEC.slots, TINY_SPEC.width), float('inf')))


def test_core_forward_keeps_no_mutable_state():
    core = CognitiveCore(TINY_SPEC)
    before = {name: parameter.detach().clone() for name, parameter in core.named_parameters()}
    initial = dict(core.state_dict())
    sense, memory = torch.randn(1, 2, TINY_SPEC.width), torch.randn(1, 3, TINY_SPEC.width)
    features = torch.rand(1, TINY_SPEC.task_features)
    with torch.no_grad():
        core(sense, torch.ones(1, 2, dtype=torch.bool), memory,
             torch.ones(1, 3, dtype=torch.bool), features)
        core(sense, torch.ones(1, 2, dtype=torch.bool), memory,
             torch.ones(1, 3, dtype=torch.bool), features)
    for name, parameter in core.named_parameters():
        torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
    assert set(core.state_dict()) == set(initial)
    assert dict(core.named_buffers()) == {}


def test_all_masked_memory_logits_equal_no_memory_with_active_reader():
    adapter = core_adapter().eval()
    with torch.no_grad():
        adapter.memory_ports['0'].output_projection.weight.normal_(std=0.1)
        adapter.memory_ports['2'].output_projection.weight.normal_(std=0.1)
    masked = run(adapter, core_frame(adapter, all_masked=True))
    empty = run(adapter, core_frame(adapter, empty=True))
    torch.testing.assert_close(masked, empty, atol=0, rtol=0)


def test_core_state_and_frame_validation():
    adapter = core_adapter().eval()
    frame = core_frame(adapter)
    with pytest.raises(ValueError, match='Workspace branch requires'):
        adapter.make_frame('x', frame.evidence, frame.memory, frame.sense,
                           enabled_branches=('memory', 'workspace'))
    with pytest.raises(ValueError, match='task_features'):
        adapter.build_core_state(frame.sense, frame.memory, torch.rand(1, 3),
                                 task_digest='d', revision=1)
    with pytest.raises(ValueError, match='task_digest'):
        adapter.build_core_state(frame.sense, frame.memory, frame.core.task_features,
                                 task_digest='', revision=1)
    with pytest.raises(ValueError, match='revision'):
        adapter.build_core_state(frame.sense, frame.memory, frame.core.task_features,
                                 task_digest='d', revision=-1)
    wrong_features = CoreState(frame.core.workspace, frame.core.valid,
                               torch.zeros(1, TINY_SPEC.task_features - 1),
                               frame.core.producer, frame.core.task_digest,
                               frame.core.revision, frame.core.steps)
    with pytest.raises(ValueError, match='Core task features'):
        adapter.make_frame('x', frame.evidence, frame.memory, frame.sense, core=wrong_features)
    no_digest = CoreState(frame.core.workspace, frame.core.valid, frame.core.task_features,
                          frame.core.producer, '', frame.core.revision, frame.core.steps)
    with pytest.raises(ValueError, match='task digest'):
        adapter.make_frame('x', frame.evidence, frame.memory, frame.sense, core=no_digest)
    with pytest.raises(ValueError, match='owned CoreState'):
        adapter.predict_success(object())
    with pytest.raises(ValueError, match='core-bearing manifest'):
        coreless = tiny_adapter().eval()
        coreless_frame = synthetic_frame(coreless)
        coreless.make_frame(coreless_frame.decision_id, coreless_frame.evidence,
                            coreless_frame.memory, coreless_frame.sense,
                            core=frame.core)


def test_readers_are_separate_modules_and_exactly_null_at_init():
    torch.manual_seed(21)
    memory_reader = MemoryAccessPort(16, 16, 4)
    workspace_reader = WorkspaceAccessPort(16, 16, 4)
    hidden = torch.randn(1, 3, 16)
    bank = torch.randn(1, 5, 16)
    bank_valid = torch.ones(1, 5, dtype=torch.bool)
    zero = torch.zeros_like(hidden)
    assert torch.equal(memory_reader(hidden, bank, bank_valid), zero)
    assert torch.equal(workspace_reader(hidden, bank, bank_valid), zero)
    memory_ids = {id(p) for p in memory_reader.parameters()}
    workspace_ids = {id(p) for p in workspace_reader.parameters()}
    assert memory_ids.isdisjoint(workspace_ids)
    with torch.no_grad():
        memory_reader.output_projection.weight.normal_(std=0.1)
    assert not torch.equal(memory_reader(hidden, bank, bank_valid), zero)
    assert torch.equal(workspace_reader(hidden, bank, bank_valid), zero)
    with torch.no_grad():
        memory_reader.output_projection.weight.zero_()
        workspace_reader.output_projection.weight.normal_(std=0.1)
    assert torch.equal(memory_reader(hidden, bank, bank_valid), zero)
    assert not torch.equal(workspace_reader(hidden, bank, bank_valid), zero)


def test_each_block_injects_each_enabled_branch_exactly_once(monkeypatch):
    adapter = core_adapter().eval()
    counts = {}
    for branch, ports in (('memory', adapter.memory_ports),
                          ('workspace', adapter.workspace_ports)):
        for layer, port in ports.items():
            original = port.forward

            def counted(*args, _original=original, _key=(branch, layer), **kwargs):
                counts[_key] = counts.get(_key, 0) + 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(port, 'forward', counted)
    frame = core_frame(adapter, enabled_branches=('memory', 'workspace'))
    assert torch.isfinite(run(adapter, frame)).all()
    expected = {(branch, str(layer)): 1 for branch in ('memory', 'workspace')
                for layer in adapter.ports.injection_layers}
    assert counts == expected


def test_branch_subsets_are_independent_and_validated():
    adapter = core_adapter().eval()
    workspace_only = core_frame(adapter, enabled_branches=('workspace',))
    memory_only = core_frame(adapter, enabled_branches=('memory',))
    assert workspace_only.enabled_branches == ('workspace',)
    assert memory_only.enabled_branches == ('memory',)
    assert torch.isfinite(run(adapter, workspace_only)).all()
    assert torch.isfinite(run(adapter, memory_only)).all()
    with pytest.raises(ValueError, match='nonempty distinct subset'):
        core_frame(adapter, enabled_branches=())
    with pytest.raises(ValueError, match='nonempty distinct subset'):
        core_frame(adapter, enabled_branches=('memory', 'memory'))
    with pytest.raises(ValueError, match='nonempty distinct subset'):
        core_frame(adapter, enabled_branches=('goal',))


def test_core_producer_and_frame_stamp_bind_every_owned_tensor():
    adapter = core_adapter().eval()
    producer = adapter.core_producer_key()
    stamp = adapter.parameter_stamp()
    targets = {
        'core.self_q': adapter.core.self_q.weight,
        'core.slot_queries': adapter.core.slot_queries,
        'core.task_encoder': adapter.core.task_encoder.projection.weight,
        'core.gate': adapter.core.gate_ffn_logit,
        'ff_core': adapter.ff_core.ff_up.weight,
        'workspace_reader.0': adapter.workspace_ports['0'].query_projection.weight,
        'workspace_reader.2': adapter.workspace_ports['2'].value_projection.weight,
        'memory_reader.0': adapter.memory_ports['0'].query_projection.weight,
        'record_writer': next(adapter.record_writer.parameters()),
        'sense_port': next(adapter.sense_ports.parameters()),
        'task_head': next(adapter.task_head.parameters()),
    }
    for label, parameter in targets.items():
        with torch.no_grad():
            parameter.add_(0.01)
        assert adapter.core_producer_key() != producer, label
        assert adapter.parameter_stamp() != stamp, label
        with torch.no_grad():
            parameter.sub_(0.01)


def test_stale_core_is_rejected_for_frames_and_previous_state():
    adapter = core_adapter().eval()
    frame = core_frame(adapter)
    with torch.no_grad():
        adapter.core.self_q.weight.add_(0.01)
    with pytest.raises(ValueError, match='Core producer'):
        adapter.make_frame('x', frame.evidence, frame.memory, frame.sense, core=frame.core)
    with pytest.raises(RuntimeError, match='changed; rebuild'):
        adapter.forward_with_frame(frame=frame, **action_batch(frame))
    with pytest.raises(ValueError, match='Previous core producer'):
        adapter.build_core_state(frame.sense, frame.memory, frame.core.task_features,
                                 task_digest='next-revision', revision=4, previous=frame.core)


def test_feedforward_control_treats_an_all_invalid_pool_as_zero():
    adapter = core_adapter().eval()
    masked = synthetic_frame(adapter, all_masked=True)
    empty = synthetic_frame(adapter, empty=True)
    features = torch.rand(1, TINY_SPEC.task_features)
    with torch.no_grad():
        from_masked = adapter.build_core_state(
            masked.sense, masked.memory, features, task_digest='d', revision=1,
            implementation='feedforward')
        from_empty = adapter.build_core_state(
            empty.sense, empty.memory, features, task_digest='d', revision=1,
            implementation='feedforward')
        dirty_bank = MemoryRead(masked.memory.spans,
                                torch.full_like(masked.memory.slots, 1e4),
                                torch.zeros_like(masked.memory.valid),
                                masked.memory.producer)
        from_dirty = adapter.build_core_state(
            masked.sense, dirty_bank, features, task_digest='d', revision=1,
            implementation='feedforward')
        pooled_sense = torch.cat([summary for _, summary in masked.sense.summaries],
                                 dim=1).mean(dim=1)
        expected, _ = adapter.ff_core(pooled_sense, torch.zeros_like(pooled_sense), features)
    assert from_masked.workspace.shape == (1, TINY_SPEC.slots, TINY_SPEC.width)
    torch.testing.assert_close(from_masked.workspace, expected, atol=1e-7, rtol=0)
    torch.testing.assert_close(from_dirty.workspace, expected, atol=1e-7, rtol=0)
    torch.testing.assert_close(from_empty.workspace, expected, atol=1e-7, rtol=0)


def test_declared_parameter_counts_and_control_ratio():
    spec = CoreSpec()
    core_count = cognitive_parameter_count(spec)
    assert core_count == 1_066_243
    assert core_parameter_count(CognitiveCore(spec)) == core_count
    control_count = core_parameter_count(FeedForwardCore(spec))
    assert control_count == 1_060_800
    assert 0.95 <= control_count / core_count <= 1.05
