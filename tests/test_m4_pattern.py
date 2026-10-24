"""M4 causal pattern tables, native branch and deterministic routing contracts."""

import pytest
import torch

from architecture_helpers import action_batch, synthetic_frame, tiny_adapter
from replica_lab.architecture.contracts import FastMemorySpec, PatternRead, PatternSpec
from replica_lab.architecture.ports import PatternAccessPort
from replica_lab.runtime.routing import choose_route


SPEC = PatternSpec(injection_layer=0, orders=(2, 3), heads=2, table_rows=97, width=4, sparse=True)


def _port():
    return PatternAccessPort(8, orders=SPEC.orders, heads=SPEC.heads,
                             table_rows=SPEC.table_rows, width=SPEC.width, sparse=SPEC.sparse)


def _pattern_frame(adapter, *, enabled_branches=('memory', 'pattern'), prompt_ids=None):
    base = synthetic_frame(adapter, prompt_ids=prompt_ids)
    read = PatternRead(adapter.pattern_producer_key(), 'test-pattern-v1')
    return adapter.make_frame(base.decision_id, base.evidence, base.memory,
                              base.sense, pattern=read, enabled_branches=enabled_branches)


def test_hash_is_causal_deterministic_multihead_and_order_masked():
    port = _port()
    ids = torch.tensor([[1, 2, 3, 4]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    first = port.hash_indices(ids, valid)
    changed = port.hash_indices(torch.tensor([[1, 2, 3, 99]]), valid)
    for (_, _, initial, available), (_, _, later, _) in zip(first, changed):
        torch.testing.assert_close(initial[:, :3], later[:, :3], atol=0, rtol=0)
    for order, head, _, available in first:
        assert available[0, 0].item() == (order <= 1)
        assert available[0, 1].item() == (order <= 2)
    assert not torch.equal(first[0][2], first[1][2])


def test_pattern_null_influence_and_sparse_gradients():
    port = _port()
    ids = torch.tensor([[1, 2, 3, 4]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    hidden = torch.randn(1, 4, 8)
    assert torch.equal(port(hidden, ids, valid), hidden * 0)
    with torch.no_grad():
        port.output_projection.weight.normal_(0.0, 0.02)
    changed = port(hidden, ids, valid)
    assert not torch.equal(changed, hidden * 0)
    assert torch.equal(port(hidden, ids, valid, enabled=False), hidden * 0)
    loss = changed.square().mean()
    loss.backward()
    table = port.tables['2_0'].weight
    assert table.grad is not None and table.grad.is_sparse
    assert torch.isfinite(port.gate_projection.weight.grad).all()
    assert torch.isfinite(port.output_projection.weight.grad).all()


def test_native_pattern_null_influence_frame_and_extension_roundtrip(tmp_path):
    adapter = tiny_adapter(pattern=SPEC).eval()
    baseline_frame = _pattern_frame(adapter, enabled_branches=('memory',))
    active_frame = _pattern_frame(adapter)
    with torch.no_grad():
        baseline = adapter.forward_with_frame(frame=baseline_frame, **action_batch(baseline_frame)).logits
        initial = adapter.forward_with_frame(frame=active_frame, **action_batch(active_frame)).logits
    torch.testing.assert_close(initial, baseline, atol=0, rtol=0)
    with torch.no_grad():
        adapter.pattern_port.output_projection.weight.normal_(0.0, 0.02)
    active_frame = _pattern_frame(adapter)
    changed = adapter.forward_with_frame(frame=active_frame, **action_batch(active_frame)).logits
    assert not torch.equal(changed, baseline)

    adapter.seal_original_identity()
    destination = tmp_path / 'pattern-extension'
    receipt = adapter.save_extension(destination)
    assert receipt['format'] == 'replica.minicpm5.extensions.v2'
    assert 'pattern_port' in receipt['components']
    restored = tiny_adapter(pattern=SPEC).eval()
    restored.seal_original_identity()
    restored.load_extension(destination)
    restored_frame = _pattern_frame(restored)
    with torch.no_grad():
        actual = restored.forward_with_frame(frame=restored_frame, **action_batch(restored_frame)).logits
    torch.testing.assert_close(actual, changed, atol=0, rtol=0)


def test_pattern_frame_requires_owned_read_and_manifest():
    adapter = tiny_adapter(pattern=SPEC)
    base = synthetic_frame(adapter)
    with pytest.raises(ValueError, match='Pattern branch requires an owned PatternRead'):
        adapter.make_frame(base.decision_id, base.evidence, base.memory, base.sense,
                           enabled_branches=('memory', 'pattern'))
    without = tiny_adapter()
    without_base = synthetic_frame(without)
    with pytest.raises(ValueError, match='pattern-memory manifest'):
        without.make_frame(without_base.decision_id, without_base.evidence,
                           without_base.memory, without_base.sense,
                           pattern=PatternRead(adapter.pattern_producer_key(), 'v1'))


def test_router_priorities_and_uncovered_hash_fallbacks():
    trained = frozenset({(2, 0, 11), (3, 1, 17)})
    all_known = frozenset({(2, 0, 11), (3, 1, 17)})
    assert choose_route(case_role='delayed_audit', query_hashes=all_known,
                        trained_hashes=trained, exact_available=True).kind == 'record'
    assert choose_route(case_role='delayed_audit', query_hashes=all_known,
                        trained_hashes=trained, correction=True).kind == 'fast'
    assert choose_route(case_role='delayed_audit', query_hashes=all_known,
                        trained_hashes=trained, procedure_available=True).kind == 'procedure'
    assert choose_route(case_role='delayed_audit', query_hashes=all_known,
                        trained_hashes=trained).kind == 'pattern'
    assert choose_route(case_role='delayed_audit', query_hashes=frozenset({(2, 0, 99)}),
                        trained_hashes=trained).kind == 'fast'
    assert choose_route(case_role='retention_audit', query_hashes=frozenset({(2, 0, 99)}),
                        trained_hashes=trained).kind == 'pattern'
    assert choose_route(case_role='delayed_audit', query_hashes=frozenset({(2, 0, 99)}),
                        trained_hashes=trained, fast_available=False).kind == 'pattern'
    assert choose_route(case_role='delayed_audit', query_hashes=frozenset(),
                        trained_hashes=trained, fast_available=False,
                        pattern_available=False).kind == 'none'


def test_sparse_adamw_step_densifies_and_muon_rejects_sparse_gradient():
    from replica_lab.optimizers import build_optimizer

    adapter = tiny_adapter(pattern=SPEC).train()
    for name, parameter in adapter.new_named_parameters():
        parameter.requires_grad_(name.startswith('pattern_port.'))
    ids = torch.tensor([[1, 2, 3, 4]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    hidden = torch.randn(1, 4, 16)
    optimizer = build_optimizer(adapter, 'muon8bit', lr_base=0.0, lr_new=0.01,
                                lr_muon_base=0.0, lr_muon_new=0.01, weight_decay=0.0)
    table = adapter.pattern_port.tables['2_0'].weight
    before = table.detach().clone()
    loss = adapter.pattern_port(hidden, ids, valid).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    loss = adapter.pattern_port(hidden, ids, valid).square().mean()
    loss.backward()
    assert table.grad is not None and table.grad.is_sparse and table.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, table.detach())

    rejected = tiny_adapter(pattern=SPEC).train()
    for name, parameter in rejected.new_named_parameters():
        parameter.requires_grad_(name == 'pattern_port.output_projection.weight')
    optimizer = build_optimizer(rejected, 'muon8bit', lr_base=0.0, lr_new=0.01,
                                lr_muon_base=0.0, lr_muon_new=0.01, weight_decay=0.0)
    parameter = rejected.pattern_port.output_projection.weight
    parameter.grad = torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.0]), parameter.shape)
    with pytest.raises(ValueError, match='dense gradients'):
        optimizer.step()


def test_additive_pattern_migration_is_strict(tmp_path):
    fast_spec = FastMemorySpec(heads=2, key_size=4, value_size=4, update_rule='sgdn2')
    parent = tiny_adapter(fast_memory=fast_spec).eval()
    parent.seal_original_identity()
    destination = tmp_path / 'parent-v2'
    parent.save_extension(destination)
    target = tiny_adapter(fast_memory=fast_spec, pattern=SPEC).eval()
    target.seal_original_identity()
    with pytest.raises(ValueError, match='Incompatible or changed'):
        target.load_extension(destination)
    receipt = target.load_extension(destination, accept_parent_pattern=True)
    assert receipt['migration']['accepted_parent_v2_pattern'] is True
    assert receipt['migration']['fresh_init'] == sorted(
        f'pattern_port.{name}' for name, _ in target.pattern_port.named_parameters())
    for name, parameter in target.fast_ports.named_parameters():
        original = dict(parent.fast_ports.named_parameters())[name]
        torch.testing.assert_close(parameter, original, atol=0, rtol=0)

    invalid = tiny_adapter(pattern=SPEC).eval()
    invalid.seal_original_identity()
    with pytest.raises(ValueError, match='additive pattern branch'):
        invalid.load_extension(destination, accept_parent_pattern=True)
