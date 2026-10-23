"""Reference E4 fast associative memory: host-owned, functional and bounded.

This is the connected host service around the pure ``LinearFastMemory``
primitive, not a native port or a trained capability. One observation stages
writes against one chunk-start state; only ``commit`` applies them. All depth
ports in a future native integration must read one snapshot and share one commit.

State is task/branch-owned. A source correction or revocation must discard the
state and rebuild from permitted observations; this module never claims neural
unlearning. Images are compatible only with the exact owner, scope, schema and
producer that wrote them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from ..neural import FastState, FastWrite, LinearFastMemory

FORMAT = 'replica.fast_memory.image.v1'
SCHEMA = 'replica.fast_memory.state.v1'
MAX_DIM = 1024


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(
            c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{name} must be a lowercase SHA256')


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip() or any(c.isspace() for c in value):
        raise ValueError(f'{name} must be an identifier')


@dataclass(frozen=True)
class FastMemoryConfig:
    key_size: int = 64
    value_size: int = 64
    heads: int = 4
    learning_rate: float = 0.1
    decay: float = 0.0
    max_update_norm: float | None = 1.0
    max_chunks: int = 4096
    max_tokens_per_chunk: int = 256
    max_image_bytes: int = 1024 * 1024

    def __post_init__(self):
        for name in ('key_size', 'value_size'):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= MAX_DIM:
                raise ValueError(f'{name} must be a positive bounded integer')
        if type(self.heads) is not int or not 1 <= self.heads <= 64:
            raise ValueError('heads must be in 1..64')
        for name in ('max_chunks', 'max_tokens_per_chunk'):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 4096:
                raise ValueError(f'{name} must be in 1..4096')
        if type(self.max_image_bytes) is not int or not 1 <= self.max_image_bytes <= 8 * 1024 * 1024:
            raise ValueError('max_image_bytes must be in 1..8 MiB')
        if self.max_chunks > 4096 or self.max_tokens_per_chunk > 4096:
            raise ValueError('fast-memory bounds exceed the reference policy')
        if self.max_image_bytes > 8 * 1024 * 1024:
            raise ValueError('fast-memory image quota exceeds 8 MiB')
        if not 0 < self.learning_rate <= 1 or not 0 <= self.decay < 1:
            raise ValueError('learning_rate must be in (0,1] and decay in [0,1)')
        if self.max_update_norm is not None and self.max_update_norm <= 0:
            raise ValueError('max_update_norm must be positive or None')

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class FastMemoryScope:
    subject_id: str
    branch_id: str
    purpose: str
    mode: str
    source_epoch: int

    def __post_init__(self):
        for name in ('subject_id', 'branch_id', 'purpose', 'mode'):
            _identifier(getattr(self, name), name)
        if type(self.source_epoch) is not int or self.source_epoch < 0:
            raise ValueError('source_epoch must be nonnegative')

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class FastMemoryState:
    weights: torch.Tensor  # [batch=1, heads, key, value]
    chunk_index: int
    update_count: int
    cursor: int
    lineage: tuple[tuple[str, str], ...]
    generation: int = 0
    schema: str = SCHEMA

    def __post_init__(self):
        if (self.weights.ndim != 4 or self.weights.shape[0] != 1
                or not self.weights.is_floating_point()
                or not bool(torch.isfinite(self.weights).all())):
            raise ValueError('weights must be finite [1, heads, key, value]')
        for name in ('chunk_index', 'update_count', 'cursor', 'generation'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative')
        if not isinstance(self.lineage, tuple):
            raise ValueError('lineage must be a tuple')
        if self.schema != SCHEMA:
            raise ValueError('unknown fast-memory state schema')

    @property
    def digest(self):
        return digest({'schema': self.schema, 'chunk_index': self.chunk_index,
                       'update_count': self.update_count, 'cursor': self.cursor,
                       'lineage': list(self.lineage), 'generation': self.generation,
                       'weights': [float(v) for v in
                                   self.weights.detach().cpu().flatten().tolist()]})

    def detached(self):
        return FastMemoryState(self.weights.detach().clone(), self.chunk_index,
                               self.update_count, self.cursor, self.lineage,
                               self.generation, self.schema)


@dataclass(frozen=True)
class FastMemoryCommit:
    state: FastMemoryState
    observation_ids: tuple[str, ...]
    chunk_index: int
    state_digest: str
    committed: bool


class FastMemorySession:
    """One branch/task-owned fast memory; no global state and no I/O in compute."""

    def __init__(self, config: FastMemoryConfig, scope: FastMemoryScope, *,
                 producer: str, state: FastMemoryState | None = None):
        self.config = config
        self.scope = scope
        _identifier(producer, 'producer')
        self.producer = producer
        if state is None:
            weights = torch.zeros(1, config.heads, config.key_size, config.value_size,
                                  dtype=torch.float32)
            state = FastMemoryState(weights, 0, 0, 0, (), 0)
        self._check_state(state)
        self._state = state
        self._pending: dict[str, tuple[torch.Tensor, torch.Tensor, str]] = {}

    def _check_state(self, state):
        if not isinstance(state, FastMemoryState):
            raise ValueError('state must be FastMemoryState')
        if tuple(state.weights.shape) != (1, self.config.heads,
                                          self.config.key_size, self.config.value_size):
            raise ValueError('fast-memory state dimensions differ from config')

    @property
    def state(self):
        return self._state

    @property
    def pending_ids(self):
        return tuple(sorted(self._pending))

    def snapshot(self):
        """Detached copy for another depth port; never shares live storage."""
        return self._state.detached()

    def read(self, queries: torch.Tensor) -> torch.Tensor:
        """Return [1, positions, heads*value] from the committed state only."""
        if queries.ndim != 3 or queries.shape[0] != 1 \
                or queries.shape[-1] != self.config.key_size \
                or not queries.is_floating_point():
            raise ValueError('queries must be finite [1, positions, key_size]')
        if not bool(torch.isfinite(queries).all()):
            raise ValueError('queries must be finite')
        values = []
        for head in range(self.config.heads):
            values.append(self._primitive(head).read(
                queries, FastState(self._state.weights[:, head])))
        return torch.cat(values, dim=-1)

    def stage_observation(self, observation_id: str, keys: torch.Tensor,
                          values: torch.Tensor, *, source_hash: str):
        """Stage one observed event against the current chunk-start state."""
        _identifier(observation_id, 'observation_id')
        _sha(source_hash, 'source_hash')
        if self._state.chunk_index >= self.config.max_chunks:
            raise ValueError('fast-memory chunk bound reached; rebuild or reset')
        if (keys.ndim != 3 or values.ndim != 3 or keys.shape[0] != 1
                or values.shape[0] != 1 or keys.shape[:2] != values.shape[:2]
                or keys.shape[-1] != self.config.key_size
                or values.shape[-1] != self.config.value_size
                or not keys.is_floating_point() or not values.is_floating_point()
                or not bool(torch.isfinite(keys).all())
                or not bool(torch.isfinite(values).all())
                or keys.shape[1] == 0):
            raise ValueError('keys/values must be finite matching [1, tokens, dim]')
        total = sum(item[0].shape[1] for item in self._pending.values())
        if total + keys.shape[1] > self.config.max_tokens_per_chunk:
            raise ValueError('fast-memory chunk token budget exceeded')
        existing = self._pending.get(observation_id)
        if existing is not None:
            same = (existing[0].shape == keys.shape and existing[1].shape == values.shape
                    and torch.equal(existing[0], keys)
                    and torch.equal(existing[1], values)
                    and existing[2] == source_hash)
            if same:
                return False
            raise ValueError('observation ID already staged with different bytes')
        self._pending[observation_id] = (keys.detach().clone(),
                                         values.detach().clone(), source_hash)
        return True

    def commit(self) -> FastMemoryCommit:
        """Apply every staged write once against one chunk-start snapshot."""
        if not self._pending:
            return FastMemoryCommit(self._state, (), self._state.chunk_index,
                                    self._state.digest, False)
        if self._state.chunk_index >= self.config.max_chunks:
            raise ValueError('fast-memory chunk bound reached')
        start = self._state
        weights = []
        for head in range(self.config.heads):
            writes = [FastWrite(keys, values, start.chunk_index)
                      for keys, values, _ in self._pending.values()]
            next_state = self._primitive(head).update_shared_chunk(
                FastState(start.weights[:, head]), writes)
            weights.append(next_state.weights)
        combined = torch.stack(weights, dim=1)
        tokens = sum(item[0].shape[1] for item in self._pending.values())
        lineage = start.lineage + tuple((event_id, item[2])
                                        for event_id, item in self._pending.items())
        state = FastMemoryState(combined, start.chunk_index + 1,
                                start.update_count + len(self._pending),
                                start.cursor + tokens, lineage, start.generation)
        self._pending.clear()
        self._state = state
        new_ids = tuple(event_id for event_id, _ in lineage[len(start.lineage):])
        return FastMemoryCommit(state, new_ids, state.chunk_index,
                                state.digest, True)

    def reset(self):
        self._pending.clear()
        weights = torch.zeros_like(self._state.weights)
        self._state = FastMemoryState(weights, 0, 0, 0, (),
                                      self._state.generation + 1)
        return self._state

    def invalidate(self, reason: str):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('invalidation reason required')
        self.reset()
        return self._state

    def _primitive(self, head):
        return LinearFastMemory(self.config.key_size, self.config.value_size,
                                self.config.learning_rate, self.config.decay,
                                self.config.max_update_norm)

    def save(self, directory):
        """Persist one image and metadata; never called during forward/backward."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        weights_path = path / 'fast_state.safetensors'
        save_file({'weights': self._state.weights.detach().cpu().contiguous()},
                  str(weights_path))
        if weights_path.stat().st_size > self.config.max_image_bytes:
            weights_path.unlink()
            raise ValueError('fast-memory image exceeds the declared quota')
        metadata = {
            'format': FORMAT, 'schema': SCHEMA, 'scope': self.scope.to_dict(),
            'producer': self.producer, 'config': self.config.to_dict(),
            'weights_sha256': hashlib.sha256(weights_path.read_bytes()).hexdigest(),
            'state_digest': self._state.digest,
            'chunk_index': self._state.chunk_index,
            'update_count': self._state.update_count,
            'cursor': self._state.cursor,
            'generation': self._state.generation,
            'lineage': [list(item) for item in self._state.lineage],
        }
        (path / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
        return metadata

    @classmethod
    def load(cls, directory, *, config, scope, producer):
        """Load only an image written by this owner, scope, schema and producer."""
        path = Path(directory)
        metadata = json.loads((path / 'manifest.json').read_text())
        _identifier(producer, 'producer')
        if (metadata.get('format') != FORMAT or metadata.get('schema') != SCHEMA
                or metadata.get('producer') != producer
                or metadata.get('scope') != scope.to_dict()
                or metadata.get('config') != config.to_dict()):
            raise ValueError('fast-memory image is incompatible with this session')
        weights_path = path / 'fast_state.safetensors'
        if (hashlib.sha256(weights_path.read_bytes()).hexdigest()
                != metadata.get('weights_sha256')):
            raise ValueError('fast-memory image bytes changed')
        weights = load_file(str(weights_path)).get('weights')
        if weights is None or tuple(weights.shape) != (1, config.heads,
                                                       config.key_size,
                                                       config.value_size):
            raise ValueError('fast-memory tensor inventory differs')
        state = FastMemoryState(weights, int(metadata['chunk_index']),
                                int(metadata['update_count']), int(metadata['cursor']),
                                tuple(tuple(item) for item in metadata['lineage']),
                                int(metadata['generation']))
        if state.digest != metadata.get('state_digest'):
            raise ValueError('fast-memory state digest differs')
        return cls(config, scope, producer=producer, state=state)
