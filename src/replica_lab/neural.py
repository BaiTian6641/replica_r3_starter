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
from typing import Sequence

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

