"""Small, explicit PyTorch building blocks for the Replica laboratory.

These are reference components, not implementations of Titans, HOPE, Engram,
or uTTT. They do not load a model, access storage, or own mutable session state.
Support and prompt tensors MUST contain only information available before the
prediction being trained. Passing an entire teacher-forced answer as a prompt
would violate this contract; a tensor module cannot infer that provenance.

All masks use True/1 for valid positions. Modules follow the caller's chosen
device and precision. Fast-state arithmetic follows the state's dtype; FP32 is
the suggested baseline even when decoder activations use a lower precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _hidden(x: Tensor, width: int, name: str) -> None:
    if x.ndim != 3 or x.shape[-1] != width:
        raise ValueError(f"{name} must have shape [batch, positions, {width}]")


def _mask(x: Tensor, valid: Tensor | None) -> Tensor:
    if valid is None:
        return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    if tuple(valid.shape) != tuple(x.shape[:2]):
        raise ValueError("mask must match [batch, positions]")
    return valid.to(device=x.device, dtype=torch.bool)


def _attention_context(values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Avoid all-masked softmax rows, while retaining an exact null indicator."""
    available = valid.any(dim=1)
    # Callers handle a literal zero-length context before entering this helper.
    safe = valid.clone()
    safe[:, 0] = safe[:, 0] | ~available
    clean = values.masked_fill(~valid.unsqueeze(-1), 0)
    return clean, ~safe, available


class RecordEncoder(nn.Module):
    """Compress an already-observed record into a fixed number of latent slots.

    One record is encoded per batch row: [B, source_positions, hidden_size].
    Several records can be encoded as B*R rows and reshaped to [B, R*K, M].
    This explicit encoding does not retrieve records or authorize their use.
    """

    def __init__(self, hidden_size: int, memory_size: int, slots: int = 4,
                 heads: int = 4) -> None:
        super().__init__()
        if slots < 1 or memory_size < 1 or heads < 1 or memory_size % heads:
            raise ValueError("positive slots and head-divisible memory_size required")
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.slots = slots
        self.input_norm = nn.LayerNorm(hidden_size)
        self.input_projection = nn.Linear(hidden_size, memory_size, bias=False)
        self.slot_queries = nn.Parameter(torch.randn(slots, memory_size) * 0.02)
        self.attention = nn.MultiheadAttention(memory_size, heads, batch_first=True,
                                               dropout=0.0)
        self.output_norm = nn.LayerNorm(memory_size)

    def forward(self, support_hidden: Tensor,
                support_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        _hidden(support_hidden, self.hidden_size, "support_hidden")
        valid = _mask(support_hidden, support_mask)
        batch = support_hidden.shape[0]
        if support_hidden.shape[1] == 0:
            return (support_hidden.new_zeros(batch, self.slots, self.memory_size),
                    torch.zeros(batch, self.slots, dtype=torch.bool,
                                device=support_hidden.device))
        # Mask before the projection too, so arbitrary padded values are inert.
        source = support_hidden.masked_fill(~valid.unsqueeze(-1), 0)
        source = self.input_projection(self.input_norm(source))
        source, padding, available = _attention_context(source, valid)
        query = self.slot_queries.unsqueeze(0).expand(batch, -1, -1)
        encoded, _ = self.attention(query, source, source, key_padding_mask=padding,
                                    need_weights=False)
        encoded = self.output_norm(encoded + query)
        encoded = encoded * available[:, None, None].to(encoded.dtype)
        return encoded, available[:, None].expand(-1, self.slots)


class MemoryReader(nn.Module):
    """Independent gated residual delta from permitted memory slots.

    ``forward`` returns a delta, never hidden + delta. An output projection
    initialized to zero preserves the initial backbone function while allowing
    that projection to receive a first-step gradient. Empty/disabled memory
    returns exactly zero, including after training. The branch's gate is
    independent of any workspace or fast-memory gate.
    """

    def __init__(self, hidden_size: int, memory_size: int, heads: int = 4) -> None:
        super().__init__()
        if hidden_size < 1 or memory_size < 1 or heads < 1 or memory_size % heads:
            raise ValueError("positive widths and head-divisible memory_size required")
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.query_norm = nn.LayerNorm(hidden_size)
        self.query_projection = nn.Linear(hidden_size, memory_size, bias=False)
        self.memory_norm = nn.LayerNorm(memory_size)
        self.attention = nn.MultiheadAttention(memory_size, heads, batch_first=True,
                                               dropout=0.0)
        self.output_projection = nn.Linear(memory_size, hidden_size, bias=False)
        nn.init.zeros_(self.output_projection.weight)
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: Tensor, memory_slots: Tensor,
                memory_mask: Tensor | None = None, enabled: bool = True) -> Tensor:
        _hidden(hidden, self.hidden_size, "hidden")
        _hidden(memory_slots, self.memory_size, "memory_slots")
        if hidden.shape[0] != memory_slots.shape[0]:
            raise ValueError("hidden and memory batch sizes must match")
        valid = _mask(memory_slots, memory_mask)
        if not enabled or memory_slots.shape[1] == 0 or hidden.shape[1] == 0:
            return hidden * 0
        memory = memory_slots.masked_fill(~valid.unsqueeze(-1), 0)
        memory = self.memory_norm(memory)
        memory, padding, available = _attention_context(memory, valid)
        query = self.query_projection(self.query_norm(hidden))
        value, _ = self.attention(query, memory, memory, key_padding_mask=padding,
                                  need_weights=False)
        delta = self.gate_logit.sigmoid() * self.output_projection(value)
        delta = delta * available[:, None, None].to(delta.dtype)
        return delta.to(hidden.dtype)


class PromptWorkspace(nn.Module):
    """Decision-boundary workspace built only from the known prompt.

    ``build`` never receives answer/target hidden states. Keep its result fixed
    throughout the next generated answer/action segment. Rebuild it only at a
    new decision boundary after a real observation. All-zero slots represent
    an absent workspace and produce an exact zero read delta.
    """

    def __init__(self, hidden_size: int, workspace_size: int, slots: int = 4,
                 steps: int = 2, heads: int = 4) -> None:
        super().__init__()
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        self.hidden_size = hidden_size
        self.workspace_size = workspace_size
        self.steps = steps
        self.encoder = RecordEncoder(hidden_size, workspace_size, slots, heads)
        self.recurrent = nn.GRUCell(workspace_size, workspace_size)
        self.workspace_norm = nn.LayerNorm(workspace_size)
        self.reader = MemoryReader(hidden_size, workspace_size, heads)

    def build(self, prompt_hidden: Tensor, prompt_mask: Tensor | None = None) -> Tensor:
        _hidden(prompt_hidden, self.hidden_size, "prompt_hidden")
        workspace, valid = self.encoder(prompt_hidden, prompt_mask)
        batch, count, width = workspace.shape
        # Fixed prompt-derived context is distinct from the recurrent state.
        context = workspace.mean(dim=1, keepdim=True).expand(-1, count, -1)
        state = workspace
        for _ in range(self.steps):
            state = self.recurrent(context.reshape(-1, width),
                                   state.reshape(-1, width)).reshape(batch, count, width)
        state = self.workspace_norm(state)
        return state * valid.unsqueeze(-1).to(state.dtype)

    def read(self, hidden: Tensor, workspace: Tensor, enabled: bool = True) -> Tensor:
        _hidden(workspace, self.workspace_size, "workspace")
        valid = workspace.ne(0).any(dim=-1)
        return self.reader(hidden, workspace, valid, enabled=enabled)


@dataclass(frozen=True)
class FastState:
    """Explicit session state; tensor contents must be treated as immutable."""

    weights: Tensor
    chunk_index: int = 0


@dataclass(frozen=True)
class FastWrite:
    keys: Tensor
    values: Tensor
    chunk_index: int
    mask: Tensor | None = None


class LinearFastMemory:
    """Functional linear associative-state baseline with a chunk barrier.

    Read every port from one ``start_state``. Collect FastWrite values for that
    chunk, then call ``update_shared_chunk`` once. It evaluates every write
    against the SAME start-state matrix and creates a new state for the NEXT
    chunk. This is a pure numerical contract; the host owns session isolation
    and transaction ordering. No globals, optimizer slots or in-place writes
    are used. It is not a reproduction of the losses/optimizers in uTTT/Titans.

    The local objective is 1/(2*N) * sum ||normalize(k) @ A - v||^2.
    Updates remain differentiable for optional offline outer-loop training.
    Runtime callers may detach the returned state to bound history.
    """

    def __init__(self, key_size: int, value_size: int, learning_rate: float = 0.1,
                 decay: float = 0.0, max_update_norm: float | None = 1.0) -> None:
        if key_size < 1 or value_size < 1:
            raise ValueError("state dimensions must be positive")
        if not 0 < learning_rate <= 1 or not 0 <= decay < 1:
            raise ValueError("learning_rate must be in (0,1], decay in [0,1)")
        if max_update_norm is not None and max_update_norm <= 0:
            raise ValueError("max_update_norm must be positive or None")
        self.key_size = key_size
        self.value_size = value_size
        self.learning_rate = learning_rate
        self.decay = decay
        self.max_update_norm = max_update_norm

    def initial_state(self, batch_size: int, *, device: torch.device | str | None = None,
                      dtype: torch.dtype = torch.float32) -> FastState:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return FastState(torch.zeros(batch_size, self.key_size, self.value_size,
                                     device=device, dtype=dtype))

    def _check_state(self, state: FastState) -> None:
        if state.weights.ndim != 3 or tuple(state.weights.shape[1:]) != (
                self.key_size, self.value_size):
            raise ValueError("incompatible fast-state dimensions")
        if not state.weights.is_floating_point() or state.chunk_index < 0:
            raise ValueError("floating state and nonnegative chunk index required")

    def read(self, queries: Tensor, start_state: FastState) -> Tensor:
        self._check_state(start_state)
        _hidden(queries, self.key_size, "queries")
        if queries.shape[0] != start_state.weights.shape[0]:
            raise ValueError("query and state batch sizes must match")
        query = F.normalize(queries.to(start_state.weights), dim=-1, eps=1e-6)
        return torch.bmm(query, start_state.weights)

    def update_chunk(self, start_state: FastState, keys: Tensor, values: Tensor,
                     mask: Tensor | None = None) -> FastState:
        write = FastWrite(keys, values, start_state.chunk_index, mask)
        return self.update_shared_chunk(start_state, [write])

    def update_shared_chunk(self, start_state: FastState,
                            writes: Sequence[FastWrite]) -> FastState:
        self._check_state(start_state)
        weights = start_state.weights
        gradient_sum = torch.zeros_like(weights)
        count = weights.new_zeros(weights.shape[0], 1, 1)
        for write in writes:
            if write.chunk_index != start_state.chunk_index:
                raise ValueError("writes must belong to this chunk-start snapshot")
            _hidden(write.keys, self.key_size, "keys")
            _hidden(write.values, self.value_size, "values")
            if write.keys.shape[:2] != write.values.shape[:2]:
                raise ValueError("keys and values must align by batch and position")
            if write.keys.shape[0] != weights.shape[0]:
                raise ValueError("write and state batch sizes must match")
            valid = _mask(write.keys, write.mask).to(weights.device)
            keys = write.keys.to(weights).masked_fill(~valid.unsqueeze(-1), 0)
            values = write.values.to(weights).masked_fill(~valid.unsqueeze(-1), 0)
            keys = F.normalize(keys, dim=-1, eps=1e-6)
            error = (torch.bmm(keys, weights) - values) * valid.unsqueeze(-1)
            gradient_sum = gradient_sum + torch.bmm(keys.transpose(1, 2), error)
            count = count + valid.sum(dim=1).to(weights.dtype)[:, None, None]
        gradient = gradient_sum / count.clamp_min(1)
        update = self.learning_rate * gradient + self.decay * weights
        if self.max_update_norm is not None:
            norm = torch.linalg.vector_norm(update, dim=(1, 2), keepdim=True)
            scale = (self.max_update_norm / norm.clamp_min(1e-12)).clamp_max(1)
            update = update * scale
        next_weights = torch.where(count > 0, weights - update, weights)
        return FastState(next_weights, start_state.chunk_index + 1)


@dataclass(frozen=True)
class SGDN2Write:
    """One headed gated-delta write proposal; no host side effects."""

    keys: Tensor  # [batch, tokens, heads, key]
    values: Tensor  # [batch, tokens, heads, value]
    erase: Tensor  # [batch, tokens, heads, key]
    write_gate: Tensor  # [batch, tokens, heads, value]
    decay: Tensor  # [batch, tokens, heads, key]
    chunk_index: int = 0
    mask: Tensor | None = None  # [batch, tokens]

    def detached(self) -> 'SGDN2Write':
        return SGDN2Write(self.keys.detach().clone(), self.values.detach().clone(),
                          self.erase.detach().clone(), self.write_gate.detach().clone(),
                          self.decay.detach().clone(), self.chunk_index,
                          None if self.mask is None else self.mask.detach().clone())
    def head(self, index: int) -> 'SGDN2Write':
        if type(index) is not int or not 0 <= index < self.keys.shape[2]:
            raise ValueError('head index outside write')
        return SGDN2Write(
            self.keys[:, :, index:index + 1], self.values[:, :, index:index + 1],
            self.erase[:, :, index:index + 1], self.write_gate[:, :, index:index + 1],
            self.decay[:, :, index:index + 1], self.chunk_index, self.mask)


def _sgdn2_check_write(write: SGDN2Write, heads: int, key_size: int, value_size: int,
                       chunk_index: int, batch_size: int) -> None:
    if not isinstance(write, SGDN2Write) or write.chunk_index != chunk_index:
        raise ValueError('SGDN-2 writes must belong to the current chunk-start snapshot')
    expected = (heads, key_size, value_size)
    if write.keys.ndim != 4 or write.values.ndim != 4:
        raise ValueError('SGDN-2 keys/values must be [batch, tokens, heads, dim]')
    if write.keys.shape[:3] != (batch_size, write.keys.shape[1], heads):
        raise ValueError('SGDN-2 key batch/heads differ from the state')
    if write.keys.shape[-1] != key_size or write.values.shape[-1] != value_size:
        raise ValueError('SGDN-2 key/value dimensions differ from the state')
    if write.values.shape[:3] != write.keys.shape[:3]:
        raise ValueError('SGDN-2 key/value positions differ')
    for value, name, last in ((write.erase, 'erase', key_size),
                              (write.decay, 'decay', key_size),
                              (write.write_gate, 'write_gate', value_size)):
        if value.shape != write.keys.shape[:3] + (last,) or not value.is_floating_point():
            raise ValueError(f'SGDN-2 {name} has incompatible dimensions')
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'SGDN-2 {name} must be finite')
    for value, name in ((write.keys, 'keys'), (write.values, 'values')):
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f'SGDN-2 {name} must be finite')
    if write.mask is not None:
        if (write.mask.dtype != torch.bool
                or tuple(write.mask.shape) != tuple(write.keys.shape[:2])):
            raise ValueError('SGDN-2 mask must be boolean [batch, tokens]')


class SGDN2Memory:
    """Functional gated-delta fast state with head-specific writes.

    Each token computes a stable gated delta update against the same state that
    later chunks read. The final residual port supplies zero-initialized output
    projections; this primitive never performs I/O or host commits.
    """

    def __init__(self, heads: int, key_size: int = 64, value_size: int = 64) -> None:
        if min(heads, key_size, value_size) < 1:
            raise ValueError('positive SGDN-2 dimensions required')
        self.heads, self.key_size, self.value_size = heads, key_size, value_size

    def initial_state(self, batch_size: int, *, device=None, dtype=torch.float32) -> FastState:
        if batch_size < 1:
            raise ValueError('batch size must be positive')
        return FastState(torch.zeros(batch_size, self.heads, self.key_size,
                                     self.value_size, device=device, dtype=dtype))

    def _check_state(self, state: FastState) -> None:
        if (state.weights.ndim != 4
                or state.weights.shape[1:] != (self.heads, self.key_size, self.value_size)
                or not state.weights.is_floating_point()
                or not bool(torch.isfinite(state.weights).all())):
            raise ValueError('incompatible SGDN-2 state')
        if state.chunk_index < 0:
            raise ValueError('chunk index must be nonnegative')

    def read(self, queries: Tensor, start_state: FastState) -> Tensor:
        self._check_state(start_state)
        if queries.ndim != 3 or queries.shape[-1] != self.key_size:
            raise ValueError('queries must be [batch, positions, key_size]')
        if queries.shape[0] != start_state.weights.shape[0]:
            raise ValueError('query and state batch sizes must match')
        query = F.normalize(queries.to(start_state.weights), dim=-1, eps=1e-6)
        return torch.einsum('btk,bhkv->bthv', query, start_state.weights)

    def update_shared_chunk(self, start_state: FastState,
                            writes: Sequence[SGDN2Write]) -> FastState:
        self._check_state(start_state)
        state = start_state.weights
        for write in writes:
            if state.device != write.keys.device:
                state = state.to(write.keys.device)
            _sgdn2_check_write(write, self.heads, self.key_size, self.value_size,
                               start_state.chunk_index, state.shape[0])
            mask = (torch.ones(write.keys.shape[:2], dtype=torch.bool,
                               device=write.keys.device)
                    if write.mask is None else write.mask)
            keys = F.normalize(write.keys.to(state), dim=-1, eps=1e-6)
            values = write.values.to(state)
            erase = write.erase.to(state).clamp(1e-4, 1.0)
            gate = write.write_gate.to(state).clamp(0.0, 1.0)
            decay = write.decay.to(state).clamp(1e-4, 1.0 - 1e-4)
            # sqrt(beta) k has norm <= 1, so the erase/update is stable. The
            # token loop avoids materializing [tokens, heads, key, key] tensors.
            scaled_keys = erase.sqrt() * keys
            for index in range(keys.shape[1]):
                key = scaled_keys[:, index]
                value = values[:, index]
                gate_value = gate[:, index]
                forget = decay[:, index]
                decayed = forget.unsqueeze(-1) * state
                read = (key.unsqueeze(-1) * decayed).sum(dim=2)
                candidate = decayed + key.unsqueeze(-1) * (
                    gate_value * value - read).unsqueeze(-2)
                state = torch.where(mask[:, index, None, None, None], candidate, state)
        return FastState(state, start_state.chunk_index + 1)

    def reduce_shared_chunk(self, start_state: FastState,
                            port_writes: Mapping[int, Sequence[SGDN2Write]]) -> FastState:
        """Convex-mean SGDN-2 proposals from one immutable chunk-start state.

        Every port proposal is evaluated independently against ``start_state``;
        no proposal can read another port's current-chunk result. This project
        reducer is not uTTT's aggregated-gradient update.
        """
        self._check_state(start_state)
        if not isinstance(port_writes, Mapping) or not port_writes:
            raise ValueError('shared reduction requires named port proposals')
        if any(type(layer) is not int or layer < 0 for layer in port_writes):
            raise ValueError('shared proposal ports must be nonnegative integers')
        proposals = [self.update_shared_chunk(start_state, port_writes[layer])
                     for layer in sorted(port_writes)]
        device = proposals[0].weights.device
        dtype = proposals[0].weights.dtype
        if any(p.weights.device != device or p.weights.dtype != dtype for p in proposals):
            raise ValueError('shared proposals must use one device and dtype')
        weights = torch.stack([p.weights for p in proposals], dim=0).mean(dim=0)
        return FastState(weights, start_state.chunk_index + 1)

