"""Pure neural ports for the explicit E1/E2 backbone interface.

The host supplies eligible source slots and known-input masks. These modules
neither retrieve evidence nor own request state. Each memory port has its own
parameters and returns a residual *delta*, to be added once after a complete
native block. Callers keep the supplied memory immutable through backward.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ..neural import SGDN2Write


def _softcap_silu(value: Tensor) -> Tensor:
    return 2.0 * torch.tanh(F.silu(value) / 2.0)


def _hidden(value: Tensor, width: int, name: str) -> None:
    if value.ndim != 3 or value.shape[-1] != width:
        raise ValueError(f"{name} must have shape [batch, positions, {width}]")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating point")


def _valid(value: Tensor, valid: Tensor, name: str) -> Tensor:
    if tuple(valid.shape) != tuple(value.shape[:2]):
        raise ValueError(f"{name} must match [batch, positions]")
    return valid.to(device=value.device, dtype=torch.bool)


class MemoryAccessPort(nn.Module):
    """Masked cross-attention from native tokens into a fixed memory bank.

    Q and O connect native width ``h`` to memory width ``d``; K and V stay at
    memory width. The four bias-free projections therefore contain exactly
    ``2*h*d + 2*d*d`` parameters. There is no second attention projection pair
    hidden inside a MultiheadAttention wrapper. Layer norms and a scalar gate
    are additional parameters.

    Only the final O projection is zero initialized. Its first-step gradient
    opens the branch, after which gradients reach Q/K/V and the source writer.
    Empty, disabled and all-masked rows remain exactly null after training.
    Queries are independent across positions; attention never mixes native
    time positions. Masked memory contents, including NaNs, are inert.
    """

    def __init__(self, hidden_size: int, memory_size: int = 256,
                 heads: int = 4) -> None:
        super().__init__()
        if hidden_size < 1 or memory_size < 1 or heads < 1 or memory_size % heads:
            raise ValueError("positive widths and head-divisible memory_size required")
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.heads = heads
        self.head_size = memory_size // heads
        self.query_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(memory_size)
        self.query_projection = nn.Linear(hidden_size, memory_size, bias=False)
        self.key_projection = nn.Linear(memory_size, memory_size, bias=False)
        self.value_projection = nn.Linear(memory_size, memory_size, bias=False)
        self.output_projection = nn.Linear(memory_size, hidden_size, bias=False)
        nn.init.zeros_(self.output_projection.weight)
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: Tensor, memory: Tensor, valid: Tensor,
                enabled: bool = True, return_attention: bool = False):
        _hidden(hidden, self.hidden_size, "hidden")
        _hidden(memory, self.memory_size, "memory")
        if hidden.shape[0] != memory.shape[0]:
            raise ValueError("hidden and memory batch sizes must match")
        if hidden.device != memory.device:
            raise ValueError("hidden and memory must be on the same device")
        valid = _valid(memory, valid, "valid")
        batch, length, _ = hidden.shape
        count = memory.shape[1]
        if not enabled or batch == 0 or length == 0 or count == 0:
            return torch.zeros_like(hidden)

        available = valid.any(dim=1)
        # Keep softmax well-defined for empty rows, then remove their entire
        # delta. Mask before normalization so arbitrary padded bytes are inert.
        safe = valid.clone()
        safe[:, 0] = safe[:, 0] | ~available
        source = self.memory_norm(memory.masked_fill(~valid.unsqueeze(-1), 0))
        query = self.query_projection(self.query_norm(hidden))
        key = self.key_projection(source)
        value = self.value_projection(source)
        query = query.reshape(batch, length, self.heads, self.head_size).transpose(1, 2)
        key = key.reshape(batch, count, self.heads, self.head_size).transpose(1, 2)
        value = value.reshape(batch, count, self.heads, self.head_size).transpose(1, 2)
        read_heads = F.scaled_dot_product_attention(
            query, key, value, attn_mask=safe[:, None, None, :], dropout_p=0.0,
        )
        read = read_heads.transpose(1, 2).reshape(batch, length, self.memory_size)
        delta = self.gate_logit.sigmoid() * self.output_projection(read)
        delta = delta.masked_fill(~available[:, None, None], 0).to(hidden.dtype)
        if return_attention:
            scores = query @ key.transpose(-2, -1) / self.head_size**0.5
            weights = torch.softmax(scores.masked_fill(~safe[:, None, None, :], float('-inf')), dim=-1)
            weights = weights.masked_fill(~available[:, None, None, None], 0)
            return delta, {'query': query, 'key': key, 'value': value,
                           'weights': weights, 'read': read_heads}
        return delta


class WorkspaceAccessPort(MemoryAccessPort):
    """Masked cross-attention from native tokens into the task-owned workspace.

    Same attention pattern and parameter arithmetic as ``MemoryAccessPort`` but
    with a fully independent parameter set and gate, reading a ``CoreState``
    workspace bank instead of record memory. The final output projection is zero
    initialized, so a disabled port, an empty bank or an all-masked workspace
    contributes an exact null delta.
    """


class SensePort(nn.Module):
    """One bounded summary of the explicitly masked known-input features.

    A caller performs the separate, unaugmented sense pass and supplies only
    already-known tokens. The tensor API cannot infer whether content contains
    an answer or future observation. It does not cache or detach the summary.
    Tanh bounds each summary element to [-1, 1]; empty rows have an exact null
    summary and a false availability mask.
    """

    def __init__(self, hidden_size: int, memory_size: int = 256) -> None:
        super().__init__()
        if hidden_size < 1 or memory_size < 1:
            raise ValueError("positive hidden_size and memory_size required")
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.input_norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, memory_size, bias=False)

    def forward(self, hidden: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        _hidden(hidden, self.hidden_size, "hidden")
        valid = _valid(hidden, valid, "valid")
        batch = hidden.shape[0]
        available = valid.any(dim=1, keepdim=True)
        if batch == 0 or hidden.shape[1] == 0:
            return hidden.new_zeros(batch, 1, self.memory_size), available
        source = self.input_norm(hidden.masked_fill(~valid.unsqueeze(-1), 0))
        source = source.masked_fill(~valid.unsqueeze(-1), 0)
        # FP32 accumulation avoids overflow when the native features are BF16
        # or FP16. It retains gradients through all valid known positions.
        pooled = source.float().sum(dim=1, keepdim=True)
        count = valid.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
        pooled = (pooled / count).to(hidden.dtype)
        summary = self.projection(pooled).tanh()
        return summary.masked_fill(~available.unsqueeze(-1), 0), available


class FastWriteEncoder(nn.Module):
    """Project source-token features into one fast-memory key/value chunk.

    This is a learned global write projection, not a per-branch state owner.
    Inputs are frozen native token embeddings of an *observed* source page;
    generated answers and teacher targets are never accepted here. Keys are
    normalized for the linear associative update. Values are bounded with tanh
    so a single long source page cannot create unbounded state magnitudes.
    """

    def __init__(self, hidden_size: int, key_size: int = 64,
                 value_size: int = 64) -> None:
        super().__init__()
        if min(hidden_size, key_size, value_size) < 1:
            raise ValueError('positive fast-write dimensions required')
        self.hidden_size, self.key_size, self.value_size = (
            hidden_size, key_size, value_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.key_projection = nn.Linear(hidden_size, key_size, bias=False)
        self.value_projection = nn.Linear(hidden_size, value_size, bias=False)

    def forward(self, embeddings: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        _hidden(embeddings, self.hidden_size, 'embeddings')
        if not bool(torch.isfinite(embeddings).all()):
            raise ValueError('embeddings must be finite')
        if mask is not None and tuple(mask.shape) != tuple(embeddings.shape[:2]):
            raise ValueError('fast-write mask must match [batch, positions]')
        source = self.input_norm(embeddings)
        if mask is not None:
            source = source.masked_fill(~mask.to(embeddings.device).bool().unsqueeze(-1), 0)
        keys = F.normalize(self.key_projection(source).float(), dim=-1, eps=1e-6)
        values = torch.tanh(self.value_projection(source)).float()
        return keys.to(embeddings.dtype), values.to(embeddings.dtype)


class SGDN2WriteEncoder(nn.Module):
    """Produce independent headed gated-delta writes from observed features."""

    def __init__(self, hidden_size: int, heads: int = 4,
                 key_size: int = 64, value_size: int = 64) -> None:
        super().__init__()
        if min(hidden_size, heads, key_size, value_size) < 1:
            raise ValueError('positive SGDN-2 dimensions required')
        self.hidden_size, self.heads = hidden_size, heads
        self.key_size, self.value_size = key_size, value_size
        self.input_norm = nn.LayerNorm(hidden_size)
        self.key_projection = nn.Linear(hidden_size, heads * key_size, bias=False)
        self.value_projection = nn.Linear(hidden_size, heads * value_size, bias=False)
        self.erase_projection = nn.Linear(hidden_size, heads * key_size, bias=False)
        self.write_projection = nn.Linear(hidden_size, heads * value_size, bias=False)
        self.forget_projection = nn.Linear(hidden_size, heads * key_size, bias=False)

    def forward(self, embeddings: Tensor, mask: Tensor | None = None) -> SGDN2Write:
        _hidden(embeddings, self.hidden_size, 'embeddings')
        if not bool(torch.isfinite(embeddings).all()):
            raise ValueError('embeddings must be finite')
        if mask is not None and tuple(mask.shape) != tuple(embeddings.shape[:2]):
            raise ValueError('SGDN-2 write mask must match [batch, positions]')
        source = self.input_norm(embeddings)
        if mask is not None:
            source = source.masked_fill(~mask.to(embeddings.device).bool().unsqueeze(-1), 0)
        batch, tokens = source.shape[:2]
        keys = self.key_projection(source).reshape(
            batch, tokens, self.heads, self.key_size)
        values = self.value_projection(source).reshape(
            batch, tokens, self.heads, self.value_size)
        erase = self.erase_projection(source).reshape(
            batch, tokens, self.heads, self.key_size)
        gate = self.write_projection(source).reshape(
            batch, tokens, self.heads, self.value_size)
        decay = self.forget_projection(source).reshape(
            batch, tokens, self.heads, self.key_size)
        keys = F.normalize(keys.float(), dim=-1, eps=1e-6)
        values = torch.tanh(values.float())
        erase = erase.float().sigmoid().clamp(1e-4, 1.0)
        gate = gate.float().sigmoid().clamp(0.0, 1.0)
        decay = decay.float().sigmoid().clamp(1e-4, 1.0 - 1e-4)
        return SGDN2Write(keys, values, erase, gate, decay, 0, mask)


class FastMemoryAccessPort(nn.Module):
    """Independent gated read of one committed fast-associative snapshot.

    ``weights`` is ``[1, heads, key, value]``. Queries are projected from the
    native hidden width into the per-head key width, normalized, multiplied by
    the committed matrices, flattened and projected back. The final projection
    starts at zero; a disabled, empty or all-zero state contributes exact zero.
    """

    def __init__(self, hidden_size: int, heads: int = 4, key_size: int = 64,
                 value_size: int = 64, update_rule: str = 'linear') -> None:
        super().__init__()
        if min(hidden_size, heads, key_size, value_size) < 1:
            raise ValueError('positive fast-port dimensions required')
        if update_rule not in {'linear', 'sgdn2'}:
            raise ValueError('unknown fast-memory update rule')
        self.hidden_size, self.heads = hidden_size, heads
        self.key_size, self.value_size = key_size, value_size
        self.update_rule = update_rule
        self.query_norm = nn.LayerNorm(hidden_size)
        self.query_projection = nn.Linear(hidden_size, heads * key_size, bias=False)
        self.output_projection = nn.Linear(heads * value_size, hidden_size, bias=False)
        nn.init.zeros_(self.output_projection.weight)
        self.gate_logit = nn.Parameter(torch.ones(()) if update_rule == 'sgdn2'
                                       else torch.zeros(()))

    def forward(self, hidden: Tensor, weights: Tensor, enabled: bool = True) -> Tensor:
        _hidden(hidden, self.hidden_size, 'hidden')
        if (weights.ndim != 4 or weights.shape[0] != 1
                or weights.shape[1:] != (self.heads, self.key_size, self.value_size)
                or not weights.is_floating_point()):
            raise ValueError('fast weights must be [1, heads, key, value]')
        if not bool(torch.isfinite(weights).all()):
            raise ValueError('fast weights must be finite')
        if not enabled or not bool(weights.any()):
            return hidden * 0
        query = self.query_projection(self.query_norm(hidden))
        query = query.reshape(hidden.shape[0], hidden.shape[1], self.heads,
                              self.key_size)
        query = F.normalize(query.float(), dim=-1, eps=1e-6)
        read = torch.einsum('bthk,hkv->bthv', query,
                            weights[0].to(query.dtype)).reshape(
            hidden.shape[0], hidden.shape[1], self.heads * self.value_size)
        if self.update_rule == 'sgdn2':
            read = F.rms_norm(read, (self.heads * self.value_size,))
            gate = _softcap_silu(self.gate_logit)
        else:
            gate = self.gate_logit.sigmoid()
        delta = gate * self.output_projection(read)
        return delta.to(hidden.dtype)


class PatternAccessPort(nn.Module):
    """Static multi-head causal n-gram lookup with contextual gated residual."""

    def __init__(self, hidden_size: int, *, orders=(2, 3), heads: int = 4,
                 table_rows: int = 65536, width: int = 64, sparse: bool = True) -> None:
        super().__init__()
        if min(hidden_size, heads, table_rows, width) < 1:
            raise ValueError('positive pattern dimensions required')
        if not orders or any(order not in {2, 3} for order in orders):
            raise ValueError('pattern orders must be causal 2/3-grams')
        self.hidden_size, self.orders, self.heads = hidden_size, tuple(orders), heads
        self.table_rows, self.width = table_rows, width
        self.tables = nn.ModuleDict({
            f'{order}_{head}': nn.Embedding(table_rows, width, sparse=sparse)
            for order in self.orders for head in range(self.heads)})
        self.gate_norm = nn.LayerNorm(hidden_size)
        self.gate_projection = nn.Linear(hidden_size, self.heads)
        self.output_projection = nn.Linear(len(self.orders) * self.heads * width,
                                           hidden_size, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def hash_indices(self, input_ids: Tensor, valid: Tensor):
        """Deterministic [batch, positions] hashes and validity per order/head."""
        if (input_ids.ndim != 2 or input_ids.dtype != torch.long
                or valid.shape != input_ids.shape):
            raise ValueError('pattern IDs must be [batch, positions] int64 with matching mask')
        valid = valid.to(device=input_ids.device, dtype=torch.bool)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        result = []
        for order in self.orders:
            for head in range(self.heads):
                hashed = torch.full_like(input_ids, 17 + head * 1009)
                for offset in range(order):
                    if offset:
                        tokens = torch.cat((torch.zeros_like(input_ids[:, :offset]),
                                            input_ids[:, :-offset]), dim=1)
                    else:
                        tokens = input_ids
                    hashed = (hashed * 1000003 + tokens) % 2147483647
                hashed = (hashed * 1000003 + (head + 1) * 7919) % 2147483647
                available = valid & (positions >= order - 1)
                result.append((order, head, hashed % self.table_rows, available))
        return tuple(result)

    def forward(self, hidden: Tensor, input_ids: Tensor, valid: Tensor,
                enabled: bool = True) -> Tensor:
        _hidden(hidden, self.hidden_size, 'hidden')
        if input_ids.shape != hidden.shape[:2] or valid.shape != input_ids.shape:
            raise ValueError('pattern hidden, IDs and mask shapes differ')
        if not enabled:
            return hidden * 0
        gates = self.gate_projection(self.gate_norm(hidden)).sigmoid()
        reads = []
        for order, head, indices, available in self.hash_indices(input_ids, valid):
            values = self.tables[f'{order}_{head}'](indices)
            values = values * gates[..., head].unsqueeze(-1)
            values = values * available.unsqueeze(-1).to(values.dtype)
            reads.append(values)
        read = torch.cat(reads, dim=-1)
        read = F.rms_norm(read, (len(self.orders) * self.heads * self.width,))
        return self.output_projection(read).to(hidden.dtype)
class RecordCopyHead(nn.Module):
    """Add a learned logit bias for exact tokens present in selected records.

    The host supplies already-eligible record token IDs and their frozen
    embeddings. Queries come from native hidden states; a scalar gate starts at
    zero, so an absent/disabled/untrained copy branch is exactly null. The head
    never chooses a record, mutates source text or declares success; it only
    exposes source-token candidates to the decoder.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError('positive hidden_size required')
        self.hidden_size = hidden_size
        self.query_norm = nn.LayerNorm(hidden_size)
        self.key_norm = nn.LayerNorm(hidden_size)
        self.query_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: Tensor, token_embeddings: Tensor,
                token_ids: Tensor, mask: Tensor | None = None,
                enabled: bool = True) -> Tensor:
        _hidden(hidden, self.hidden_size, 'hidden')
        if token_embeddings.ndim != 3 or token_embeddings.shape[-1] != self.hidden_size \
                or token_embeddings.shape[0] != hidden.shape[0]:
            raise ValueError('token_embeddings must match hidden batch and width')
        if token_ids.ndim != 2 or tuple(token_ids.shape) != tuple(token_embeddings.shape[:2]) \
                or token_ids.dtype != torch.long:
            raise ValueError('token_ids must be [batch, candidates] int64')
        if mask is None:
            mask = torch.ones(token_ids.shape, dtype=torch.bool,
                              device=token_ids.device)
        if mask.dtype != torch.bool or tuple(mask.shape) != tuple(token_ids.shape):
            raise ValueError('copy mask must be boolean [batch, candidates]')
        for value, name in ((hidden, 'hidden'), (token_embeddings, 'token_embeddings')):
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f'{name} must be finite floating point')
        if not enabled or token_ids.shape[1] == 0 or not bool(mask.any()):
            return hidden.new_zeros(*hidden.shape[:2], self._vocab_hint(token_ids))
        available = mask
        query = F.normalize(self.query_projection(self.query_norm(hidden)).float(),
                            dim=-1, eps=1e-6)
        keys = F.normalize(self.key_projection(self.key_norm(token_embeddings)).float(),
                           dim=-1, eps=1e-6)
        scores = torch.einsum('bnd,bcd->bnc', query, keys) / (self.hidden_size ** 0.5)
        scores = scores.masked_fill(~available[:, None, :], -1e9)
        weights = scores.softmax(dim=-1) * available[:, None, :].to(scores.dtype)
        weights = weights * torch.tanh(self.gate_logit)
        # Scatter exact candidate token IDs into the decoder vocabulary. This
        # is source-token copying, not a learned vocabulary classifier.
        vocab = int(token_ids.max().item()) + 1
        # The caller normally supplies the model vocabulary; size the bias from
        # the maximum observed ID and let the adapter pad it to its LM head.
        expanded = token_ids.unsqueeze(1).expand(-1, hidden.shape[1], -1)
        bias = token_ids.new_zeros((hidden.shape[0], hidden.shape[1], vocab),
                                   dtype=weights.dtype)
        bias = bias.scatter_add(2, expanded, weights)
        return bias.to(hidden.dtype)

    @staticmethod
    def _vocab_hint(token_ids: Tensor) -> int:
        return int(token_ids.max().item()) + 1

class RecordPointerHead(nn.Module):
    """Score contiguous exact source-token spans for pointer copying.

    The caller supplies candidate spans, their frozen token embeddings and a
    valid mask. The selected span is decoded directly; this head does not alter
    LM logits and cannot select records or declare success.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError('positive hidden_size required')
        self.hidden_size = hidden_size
        self.query_norm = nn.LayerNorm(hidden_size)
        self.key_norm = nn.LayerNorm(hidden_size)
        self.query_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, query: Tensor, token_embeddings: Tensor,
                token_mask: Tensor, candidate_mask: Tensor) -> Tensor:
        if query.ndim == 2:
            query = query.unsqueeze(1)
        _hidden(query, self.hidden_size, 'query')
        if query.shape[0] != 1 or query.shape[1] != 1:
            raise ValueError('pointer query must be one row and one position')
        if (token_embeddings.ndim != 4 or token_embeddings.shape[0] != 1
                or token_embeddings.shape[-1] != self.hidden_size
                or token_mask.shape != token_embeddings.shape[:3]
                or candidate_mask.shape != token_embeddings.shape[:2]):
            raise ValueError('pointer candidates must be [1, candidates, span, hidden]')
        if token_mask.dtype != torch.bool or candidate_mask.dtype != torch.bool:
            raise ValueError('pointer masks must be boolean')
        for value, name in ((query, 'query'), (token_embeddings, 'token_embeddings')):
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f'{name} must be finite floating point')
        if not bool(candidate_mask.any()):
            return query.new_zeros(1, token_embeddings.shape[1])
        q = F.normalize(self.query_projection(self.query_norm(query)).float(),
                        dim=-1, eps=1e-6)
        k = self.key_projection(self.key_norm(token_embeddings)).float()
        k = k.masked_fill(~token_mask.unsqueeze(-1), 0)
        pooled = k.sum(dim=2)
        count = token_mask.sum(dim=2).clamp_min(1).unsqueeze(-1)
        pooled = F.normalize((pooled / count).squeeze(2), dim=-1, eps=1e-6)
        scores = (q * pooled).sum(dim=-1) / (self.hidden_size ** 0.5)
        return scores.masked_fill(~candidate_mask, -1e9)
