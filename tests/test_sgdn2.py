"""SGDN-2 memory contracts: head writes, stable updates, gradients and snapshots."""
import torch

from replica_lab.architecture.ports import SGDN2WriteEncoder
from replica_lab.neural import SGDN2Memory, SGDN2Write
from replica_lab.runtime.fast_memory import FastMemoryConfig, FastMemoryScope, FastMemorySession


def _write(keys, values, *, erase=0.5, gate=0.75, decay=0.9, chunk=0, mask=None):
    keys = torch.as_tensor(keys, dtype=torch.float32)
    values = torch.as_tensor(values, dtype=torch.float32)
    if keys.ndim == 2:
        keys = keys[:, None, :]
    if values.ndim == 2:
        values = values[:, None, :]
    if keys.ndim == 3:
        keys = keys[None, ...]
    if values.ndim == 3:
        values = values[None, ...]
    erase = torch.full_like(keys, erase)
    gate = torch.full_like(values, gate)
    decay = torch.full_like(keys, decay)
    return SGDN2Write(keys, values, erase, gate, decay, chunk, mask)


def test_headed_writer_produces_independent_heads_and_valid_gates():
    encoder = SGDN2WriteEncoder(8, heads=2, key_size=3, value_size=2)
    embeddings = torch.randn(1, 3, 8)
    mask = torch.tensor([[True, True, False]])
    write = encoder(embeddings, mask)
    assert write.keys.shape == (1, 3, 2, 3)
    assert write.values.shape == (1, 3, 2, 2)
    assert write.erase.shape == write.keys.shape
    assert write.write_gate.shape == write.values.shape
    assert write.decay.shape == write.keys.shape
    assert not torch.equal(write.keys[:, :, 0], write.keys[:, :, 1])
    assert not torch.equal(write.values[:, :, 0], write.values[:, :, 1])
    assert torch.all(write.erase >= 1e-4) and torch.all(write.erase <= 1)
    assert torch.all(write.decay >= 1e-4) and torch.all(write.decay < 1)
    assert write.mask is mask


def test_sgdn2_update_is_head_specific_and_finite():
    memory = SGDN2Memory(2, key_size=2, value_size=2)
    start = memory.initial_state(1)
    write = _write([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
                   [[[0.5, 0.25], [1.0, -0.5]], [[0.5, 0.25], [1.0, -0.5]]])
    state = memory.update_shared_chunk(start, [write])
    assert state.weights.shape == (1, 2, 2, 2)
    assert torch.isfinite(state.weights).all()
    assert not torch.equal(state.weights[:, 0], state.weights[:, 1])
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    read = memory.read(query, state)
    assert read.shape == (1, 2, 2, 2)
    assert torch.isfinite(read).all()


def test_sgdn2_later_read_backpropagates_through_both_chunks():
    memory = SGDN2Memory(1, key_size=2, value_size=2)
    first = _write([[[0.8, 0.2]]], [[[0.4, 0.2]]], chunk=0)
    second = _write([[[0.1, 0.9]]], [[[0.1, 0.3]]], chunk=1)
    for write in (first, second):
        for value in (write.keys, write.values, write.erase, write.write_gate, write.decay):
            value.requires_grad_(True)
    state = memory.update_shared_chunk(memory.initial_state(1), [first])
    state = memory.update_shared_chunk(state, [second])
    loss = memory.read(torch.tensor([[[1.0, 0.0]]]), state).sum()
    loss.backward()
    for write in (first, second):
        assert write.keys.grad is not None
        assert torch.isfinite(write.keys.grad).all()
        assert write.values.grad is not None
        assert torch.isfinite(write.values.grad).all()
    assert first.keys.grad.abs().sum() > 0


def test_sgdn2_masked_tokens_do_not_update_state():
    memory = SGDN2Memory(2, key_size=2, value_size=2)
    write = _write([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
                   [[[0.4, 0.2], [0.1, 0.3]], [[0.4, 0.2], [0.1, 0.3]]],
                   mask=torch.tensor([[True, False]]))
    state = memory.update_shared_chunk(memory.initial_state(1), [write])
    assert torch.isfinite(state.weights).all()
    assert state.weights.abs().sum() > 0


def test_sgdn2_host_commit_and_image_reload(tmp_path):
    scope = FastMemoryScope('fictional', 'branch-a', 'test', 'parallel_replica', 0)
    config = FastMemoryConfig(key_size=2, value_size=2, heads=2, update_rule='sgdn2',
                              max_tokens_per_chunk=8)
    session = FastMemorySession(config, scope, producer='producer-1')
    write = _write([[[1.0, 0.0], [0.0, 1.0]], [[0.4, 0.2], [0.1, 0.3]]],
                   [[[0.4, 0.2], [0.1, 0.3]], [[0.4, 0.2], [0.1, 0.3]]])
    assert session.stage_sgdn2_observation('obs-1', write, source_hash='a' * 64)
    assert session.read(torch.ones(1, 1, 2)).abs().sum() == 0
    receipt = session.commit()
    assert receipt.committed
    image = session.save(tmp_path / 'image')
    loaded = FastMemorySession.load(tmp_path / 'image', config=config, scope=scope,
                                    producer='producer-1')
    assert loaded.state.digest == session.state.digest
    assert loaded.read(torch.ones(1, 1, 2)).shape == (1, 1, 4)
    assert loaded.state.chunk_index == 1
    assert loaded.state.update_count == 1
    assert image['schema'] == 'replica.fast_memory.state.v1'


def test_sgdn2_host_accepts_more_tokens_than_heads(tmp_path):
    scope = FastMemoryScope('fictional', 'branch-a', 'test', 'parallel_replica', 0)
    config = FastMemoryConfig(key_size=2, value_size=2, heads=2, update_rule='sgdn2',
                              max_tokens_per_chunk=8)
    session = FastMemorySession(config, scope, producer='producer-1')
    write = _write(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        [
            [[0.4, 0.2], [0.1, 0.3]],
            [[0.4, 0.2], [0.1, 0.3]],
            [[0.4, 0.2], [0.1, 0.3]],
        ],
    )
    assert session.stage_sgdn2_observation('obs-multi', write, source_hash='b' * 64)
    session.commit()
    image = session.save(tmp_path / 'image')
    loaded = FastMemorySession.load(tmp_path / 'image', config=config, scope=scope,
                                    producer='producer-1')
    assert loaded.state.chunk_index == 1
    assert loaded.read(torch.ones(1, 1, 2)).shape == (1, 1, 4)


def test_sgdn2_port_preserves_null_output_but_allows_output_gradient():
    from replica_lab.architecture.ports import FastMemoryAccessPort
    port = FastMemoryAccessPort(8, heads=1, key_size=2, value_size=2,
                                update_rule='sgdn2')
    hidden = torch.randn(1, 1, 8, requires_grad=True)
    weights = torch.randn(1, 1, 2, 2)
    delta = port(hidden, weights)
    assert delta.abs().sum() == 0
    delta.sum().backward()
    gradient = port.output_projection.weight.grad
    assert gradient is not None
    assert gradient.abs().sum() > 0
