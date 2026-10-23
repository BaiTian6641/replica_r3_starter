"""Reference E4 fast memory: staged writes, one commit, reset and persistence."""

import json
from pathlib import Path
import tempfile

import torch

from replica_lab.runtime.fast_memory import (FastMemoryConfig, FastMemoryScope,
                                             FastMemorySession)


def config(**changes):
    values = dict(key_size=4, value_size=4, heads=2, learning_rate=0.5,
                  decay=0.0, max_update_norm=None, max_chunks=8,
                  max_tokens_per_chunk=8)
    values.update(changes)
    return FastMemoryConfig(**values)


def session(**changes):
    scope = FastMemoryScope('fictional', 'branch-a', 'test', 'parallel_replica', 0)
    return FastMemorySession(config(**changes), scope, producer='producer-1')


def one_hot(index, width=4):
    value = torch.zeros(1, 1, width)
    value[0, 0, index] = 1.0
    return value


def test_staged_write_is_invisible_until_one_commit_and_duplicate_is_idempotent():
    memory = session()
    query = one_hot(0)
    assert torch.equal(memory.read(query), torch.zeros(1, 1, 8))
    assert memory.stage_observation('obs-1', one_hot(0), one_hot(1),
                                    source_hash='a' * 64)
    assert not memory.stage_observation('obs-1', one_hot(0), one_hot(1),
                                        source_hash='a' * 64)
    assert torch.equal(memory.read(query), torch.zeros(1, 1, 8))
    receipt = memory.commit()
    assert receipt.committed and receipt.observation_ids == ('obs-1',)
    assert memory.state.chunk_index == 1 and memory.state.update_count == 1
    assert memory.pending_ids == ()
    assert torch.isfinite(memory.read(query)).all()
    second = memory.commit()
    assert not second.committed and second.state.chunk_index == 1


def test_reset_and_shuffled_state_controls_change_reads():
    memory = session()
    memory.stage_observation('obs-1', one_hot(0), one_hot(1), source_hash='a' * 64)
    memory.commit()
    read = memory.read(one_hot(0))
    memory.reset()
    assert torch.equal(memory.read(one_hot(0)), torch.zeros(1, 1, 8))
    other = session()
    other.stage_observation('obs-2', one_hot(0), one_hot(3), source_hash='b' * 64)
    other.commit()
    assert not torch.equal(read, other.read(one_hot(0)))


def test_bounds_and_invalidation():
    memory = session(max_tokens_per_chunk=1)
    memory.stage_observation('obs-1', one_hot(0), one_hot(1), source_hash='a' * 64)
    try:
        memory.stage_observation('obs-2', one_hot(1), one_hot(2), source_hash='b' * 64)
    except ValueError:
        pass
    else:
        raise AssertionError('chunk token bound must reject')
    memory.commit()
    memory.invalidate('source corrected')
    assert memory.state.generation == 1 and memory.state.chunk_index == 0
    assert torch.equal(memory.read(one_hot(0)), torch.zeros(1, 1, 8))


def test_persistence_is_scope_producer_and_config_bound(tmp_path):
    memory = session()
    memory.stage_observation('obs-1', one_hot(0), one_hot(1), source_hash='a' * 64)
    memory.commit()
    directory = tmp_path / 'image'
    metadata = memory.save(directory)
    assert metadata['format'] == 'replica.fast_memory.image.v1'
    loaded = FastMemorySession.load(directory, config=memory.config,
                                    scope=memory.scope, producer=memory.producer)
    assert loaded.state.digest == memory.state.digest
    assert torch.equal(loaded.read(one_hot(0)), memory.read(one_hot(0)))
    other_scope = FastMemoryScope('fictional', 'branch-b', 'test',
                                  'parallel_replica', 0)
    try:
        FastMemorySession.load(directory, config=memory.config,
                               scope=other_scope, producer=memory.producer)
    except ValueError:
        pass
    else:
        raise AssertionError('cross-branch image must be rejected')
    try:
        FastMemorySession.load(directory, config=memory.config,
                               scope=memory.scope, producer='producer-2')
    except ValueError:
        pass
    else:
        raise AssertionError('producer mismatch must be rejected')
    (directory / 'fast_state.safetensors').write_bytes(b'changed')
    try:
        FastMemorySession.load(directory, config=memory.config,
                               scope=memory.scope, producer=memory.producer)
    except ValueError:
        pass
    else:
        raise AssertionError('changed bytes must be rejected')
