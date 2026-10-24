"""M3 private/shared SGDN-2 ownership, reducer and native-port contracts."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import hashlib
import json

import pytest
import torch

from architecture_helpers import action_batch, synthetic_frame, tiny_adapter
from replica_lab.architecture.contracts import FastMemoryRead, FastMemorySpec
from replica_lab.neural import FastState, SGDN2Memory, SGDN2Write


SPEC = FastMemorySpec(heads=2, key_size=4, value_size=4,
                      update_rule='sgdn2', depth_mode='private')


def _write(chunk, key, value, *, requires_grad=False):
    keys = torch.tensor([[[[key, 1.0 - key]]]], dtype=torch.float32)
    values = torch.tensor([[[[value, -value]]]], dtype=torch.float32)
    erase = torch.full_like(keys, 0.5)
    gate = torch.full_like(values, 0.75)
    decay = torch.full_like(keys, 0.9)
    if requires_grad:
        for tensor in (keys, values, erase, gate, decay):
            tensor.requires_grad_()
    return SGDN2Write(keys, values, erase, gate, decay, chunk)


def _private_read(adapter, first, second, *, requires_grad=False):
    shape = (1, SPEC.heads, SPEC.key_size, SPEC.value_size)
    first_state = torch.full(shape, first)
    second_state = torch.full(shape, second)
    if requires_grad:
        first_state.requires_grad_()
        second_state.requires_grad_()
    return FastMemoryRead(first_state, adapter.fast_producer_key(), 'private-state',
                          {0: first_state, 2: second_state})


def _frame(adapter, fast, *, enabled_ports=None):
    base = synthetic_frame(adapter, enabled_ports=enabled_ports)
    return adapter.make_frame(base.decision_id, base.evidence, base.memory,
                              base.sense, enabled_ports=enabled_ports, fast=fast,
                              enabled_branches=('memory', 'fast'))


def _open_outputs(adapter):
    with torch.no_grad():
        for port in adapter.fast_ports.values():
            port.output_projection.weight.normal_(0.0, 0.02)


def test_private_snapshot_is_immutable_and_owns_exact_layer_tensors():
    adapter = tiny_adapter(fast_memory=SPEC)
    shape = (1, SPEC.heads, SPEC.key_size, SPEC.value_size)
    first, second = torch.zeros(shape), torch.ones(shape)
    source = {0: first, 2: second}
    read = FastMemoryRead(first, adapter.fast_producer_key(), 'private', source)
    source[0] = second
    assert read.for_layer(0) is first and read.for_layer(2) is second
    assert read.tensors() == (first, second)
    with pytest.raises(TypeError):
        read.private_weights[1] = first
    with pytest.raises(ValueError, match='missing port 1'):
        read.for_layer(1)
    read.assert_immutable()
    first.add_(1)
    with pytest.raises(RuntimeError, match='tensor mutated'):
        read.assert_immutable()


def test_private_snapshot_rejects_unowned_or_incompatible_placeholder():
    adapter = tiny_adapter(fast_memory=SPEC)
    shape = (1, SPEC.heads, SPEC.key_size, SPEC.value_size)
    first, second = torch.zeros(shape), torch.ones(shape)
    with pytest.raises(ValueError, match='alias'):
        FastMemoryRead(second.clone(), adapter.fast_producer_key(), 'private',
                       {0: first, 2: second})
    with pytest.raises(ValueError, match='match finite'):
        FastMemoryRead(first, adapter.fast_producer_key(), 'private',
                       {0: first, 2: torch.ones(1, 1, 2, 2)})


def test_shared_reducer_uses_one_start_and_commits_one_convex_mean():
    memory = SGDN2Memory(1, key_size=2, value_size=2)
    start = FastState(torch.tensor([[[[0.2, -0.1], [0.3, 0.4]]]]), chunk_index=4)
    before = start.weights.clone()
    first, second = _write(4, 0.8, 0.5), _write(4, 0.2, -0.25)
    reduced = memory.reduce_shared_chunk(start, {13: (first,), 27: (second,)})
    first_next = memory.update_shared_chunk(start, [first])
    second_next = memory.update_shared_chunk(start, [second])
    torch.testing.assert_close(reduced.weights,
                               (first_next.weights + second_next.weights) / 2)
    assert reduced.chunk_index == 5 and start.chunk_index == 4
    torch.testing.assert_close(start.weights, before, atol=0, rtol=0)

    one_empty = memory.reduce_shared_chunk(start, {13: (first,), 27: ()})
    torch.testing.assert_close(one_empty.weights, (first_next.weights + before) / 2)
    assert one_empty.chunk_index == 5


def test_shared_reducer_gradient_and_detached_first_write_control():
    memory = SGDN2Memory(1, key_size=2, value_size=2)
    start = memory.initial_state(1)
    first, second = _write(0, 0.8, 0.5, requires_grad=True), _write(0, 0.2, -0.25, requires_grad=True)
    reduced = memory.reduce_shared_chunk(start, {13: (first,), 27: (second,)})
    reduced.weights.square().sum().backward()
    assert first.keys.grad.abs().sum() > 0 and second.keys.grad.abs().sum() > 0

    cut = _write(0, 0.8, 0.5, requires_grad=True).detached()
    later = _write(0, 0.2, -0.25, requires_grad=True)
    reduced = memory.reduce_shared_chunk(start, {13: (cut,), 27: (later,)})
    reduced.weights.square().sum().backward()
    assert cut.keys.grad is None and later.keys.grad.abs().sum() > 0


def test_native_private_second_port_changes_output_and_mode_is_strict(tmp_path):
    adapter = tiny_adapter(fast_memory=SPEC).eval()
    _open_outputs(adapter)
    first = _frame(adapter, _private_read(adapter, 0.1, 0.5))
    second = _frame(adapter, _private_read(adapter, 0.1, 0.9))
    with torch.no_grad():
        first_logits = adapter.forward_with_frame(frame=first, **action_batch(first)).logits
        second_logits = adapter.forward_with_frame(frame=second, **action_batch(second)).logits
    assert not torch.equal(first_logits, second_logits)

    shared = tiny_adapter(fast_memory=replace(SPEC, depth_mode='shared')).eval()
    with pytest.raises(ValueError, match='Shared fast frame'):
        _frame(shared, _private_read(shared, 0.1, 0.5))
    shared_state = torch.zeros(1, SPEC.heads, SPEC.key_size, SPEC.value_size)
    with pytest.raises(ValueError, match='Private fast frame'):
        _frame(adapter, FastMemoryRead(shared_state, adapter.fast_producer_key(), 'shared'))

    adapter.seal_original_identity()
    destination = tmp_path / 'private-extension'
    assert adapter.save_extension(destination)['format'] == 'replica.minicpm5.extensions.v2'
    restored = tiny_adapter(fast_memory=SPEC).eval()
    restored.seal_original_identity()
    restored.load_extension(destination)
    shared.seal_original_identity()
    with pytest.raises(ValueError, match='Incompatible or changed'):
        shared.load_extension(destination)


def test_native_private_read_ablation_isolates_each_state_gradient():
    for enabled, active, inactive in (((0,), 0, 2), ((2,), 2, 0)):
        adapter = tiny_adapter(fast_memory=SPEC).train()
        _open_outputs(adapter)
        read = _private_read(adapter, 0.1, 0.5, requires_grad=True)
        frame = _frame(adapter, read, enabled_ports=enabled)
        output = adapter.forward_with_frame(frame=frame, **action_batch(frame))
        output.loss.backward()
        states = dict(read.private_weights)
        assert states[active].grad is not None and states[active].grad.abs().sum() > 0
        assert states[inactive].grad is None


def test_legacy_manifest_and_independent_depth_writer_versioning(tmp_path):
    legacy = replace(SPEC, depth_mode='shared')
    assert legacy.to_dict() == {'heads': 2, 'key_size': 4, 'value_size': 4,
                                'update_rule': 'sgdn2'}
    depth_spec = replace(SPEC, depth_writers=True)
    assert depth_spec.to_dict()['depth_writers'] is True

    class Tokenizer:
        def __call__(self, text, *, add_special_tokens):
            assert text == 'observed page' and add_special_tokens is False
            return {'input_ids': [1, 2, 3]}

    adapter = tiny_adapter(fast_memory=depth_spec)
    first = adapter.encode_fast_observation('observed page', Tokenizer(), layer=0)
    second = adapter.encode_fast_observation('observed page', Tokenizer(), layer=2)
    assert not torch.equal(first.keys, second.keys)
    assert any(name.startswith('fast_depth_write_encoders.2.')
               for name, _ in adapter.new_named_parameters())
    with pytest.raises(ValueError, match='requires a declared injection layer'):
        adapter.encode_fast_observation('observed page', Tokenizer())
    version = adapter.fast_version_key()
    assert version == tiny_adapter(fast_memory=depth_spec).fast_version_key()

    adapter.seal_original_identity()
    destination = tmp_path / 'depth-extension'
    adapter.save_extension(destination)
    restored = tiny_adapter(fast_memory=depth_spec)
    restored.seal_original_identity()
    restored.load_extension(destination)
    assert restored.fast_version_key() == version
    reloaded = restored.encode_fast_observation('observed page', Tokenizer(), layer=2)
    torch.testing.assert_close(reloaded.keys, second.keys, atol=0, rtol=0)


@pytest.mark.parametrize('mode', ['shared', 'private'])
def test_depth_image_reload_binds_tensors_lineage_and_stable_producer(tmp_path, mode):
    from replica_lab.runtime.fast_memory import (
        FastMemoryConfig, FastMemoryScope, load_fast_depth_image, save_fast_depth_image)

    adapter = tiny_adapter(fast_memory=replace(SPEC, depth_mode=mode, depth_writers=True))
    shape = (1, SPEC.heads, SPEC.key_size, SPEC.value_size)
    first, second = torch.full(shape, 0.1), torch.full(shape, 0.5)
    read = FastMemoryRead(first, adapter.fast_producer_key(), 'depth-state')
    if mode == 'private':
        read = FastMemoryRead(first, adapter.fast_producer_key(), 'depth-state',
                              {0: first, 2: second})
    config = FastMemoryConfig(heads=SPEC.heads, key_size=SPEC.key_size,
                              value_size=SPEC.value_size, update_rule='sgdn2')
    scope = FastMemoryScope('fictional', 'branch-a', 'm3-test', 'parallel_replica', 0)
    lineage = (('observation-0', hashlib.sha256(b'observed').hexdigest()),)
    producer = adapter.fast_version_key()
    metadata = save_fast_depth_image(
        tmp_path / 'image', read, config=config, scope=scope, producer=producer,
        chunk_index=1, update_count=1, cursor=3, lineage=lineage)
    loaded, loaded_metadata = load_fast_depth_image(
        tmp_path / 'image', config=config, scope=scope, producer=producer,
        frame_producer=adapter.fast_producer_key())
    assert loaded.state_digest == metadata['state_digest']
    assert loaded_metadata['lineage'] == [list(lineage[0])]
    assert loaded.producer == adapter.fast_producer_key()
    torch.testing.assert_close(loaded.weights, first, atol=0, rtol=0)
    if mode == 'private':
        torch.testing.assert_close(loaded.private_weights[2], second, atol=0, rtol=0)
    with pytest.raises(ValueError, match='incompatible'):
        load_fast_depth_image(tmp_path / 'image', config=config, scope=scope,
                              producer='0' * 64)

    manifest_path = tmp_path / 'image' / 'manifest.json'
    tampered = json.loads(manifest_path.read_text())
    tampered['lineage'][0][0] = 'tampered-observation'
    manifest_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match='metadata or lineage'):
        load_fast_depth_image(tmp_path / 'image', config=config, scope=scope, producer=producer)


def test_runner_private_linear_and_zero_states_make_valid_native_frames():
    module_spec = importlib.util.spec_from_file_location(
        'run_m2_dynamic_test', Path(__file__).resolve().parents[1] / 'scripts/run_m2_dynamic.py')
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)
    make_read, zero_read = runner.make_read, runner.zero_read

    class Tokenizer:
        def __call__(self, text, *, add_special_tokens):
            assert text.startswith('observed') and add_special_tokens is False
            return {'input_ids': [1, 2, 3]}

    adapter = tiny_adapter(fast_memory=replace(SPEC, depth_writers=True)).eval()
    _open_outputs(adapter)
    for linear in (False, True):
        read = make_read(adapter, Tokenizer(), ['observed one', 'observed two'],
                         private_depth=True, linear=linear)
        assert set(read.private_weights) == {0, 2}
        frame = _frame(adapter, read)
        with torch.no_grad():
            logits = adapter.forward_with_frame(frame=frame, **action_batch(frame)).logits
        assert torch.isfinite(logits).all()
    zero = zero_read(adapter, private_depth=True)
    _frame(adapter, zero)

    shared = tiny_adapter(fast_memory=replace(SPEC, depth_mode='shared', depth_writers=True)).eval()
    linear = make_read(shared, Tokenizer(), ['observed one'], linear=True, shared_reducer=True)
    direct = make_read(shared, Tokenizer(), ['observed one'], linear=True, shared_reducer=False)
    torch.testing.assert_close(linear.weights, direct.weights, atol=0, rtol=0)
    assert linear.state_digest == direct.state_digest
    _frame(shared, linear)
