"""Differentiable E2 record compression and candidate scoring.

Input features come from a separately declared, source-only native feature
producer. This module performs no tokenization, source admission, retrieval,
storage, cache lookup or implicit detachment. The host preserves exact source
and span references alongside the returned slots.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from replica_lab.neural import RecordEncoder


class ContextualRecordWriter(nn.Module):
    """Pool each record independently into ``slots`` latent memory vectors.

    Inputs are ``hidden[B,R,T,H]`` and ``valid[B,R,T]``. Flattening only the
    batch/record axes lets the existing RecordEncoder pool one source span per
    row. No attention or pooling crosses records. The output remains
    ``[B,R,K,D]`` until the host explicitly flattens a selected memory bank.

    Empty and all-masked records return zero slots with false masks. Native
    source features remain in the gradient graph; D1 callers may detach a
    pinned frozen producer, but the writer itself must run during training.
    """

    def __init__(self, hidden_size: int, memory_size: int = 256,
                 slots: int = 4, heads: int = 4) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("positive hidden_size required")
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.slots = slots
        self.encoder = RecordEncoder(hidden_size, memory_size, slots, heads)

    def forward(self, hidden: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        if hidden.ndim != 4 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden must have shape [batch, records, positions, {self.hidden_size}]"
            )
        if not hidden.is_floating_point():
            raise ValueError("hidden must be floating point")
        if tuple(valid.shape) != tuple(hidden.shape[:3]):
            raise ValueError("valid must match [batch, records, positions]")
        valid = valid.to(device=hidden.device, dtype=torch.bool)
        batch, records, positions, _ = hidden.shape
        if batch == 0 or records == 0 or positions == 0:
            return (
                hidden.new_zeros(batch, records, self.slots, self.memory_size),
                torch.zeros(batch, records, self.slots, dtype=torch.bool,
                            device=hidden.device),
            )
        encoded, mask = self.encoder(
            hidden.reshape(batch * records, positions, self.hidden_size),
            valid.reshape(batch * records, positions),
        )
        return (
            encoded.reshape(batch, records, self.slots, self.memory_size),
            mask.reshape(batch, records, self.slots),
        )


class TrainableRecordReranker(nn.Module):
    """Learned cosine scoring within a host-authorized candidate set.

    Query and candidate features are normalized, separately projected, and
    compared. Parameters are randomly initialized: these scores are *untrained*
    until a declared ranking objective trains them. They do not imply evidence
    quality, calibrated confidence, or permission to broaden a candidate set.

    Valid scores are bounded cosine values for finite input features. Invalid
    candidates receive ``-inf``; an all-invalid row stays all ``-inf``. The host
    must check its mask before softmax or selection. No empty-set fallback may
    retrieve from another scope. Candidate order is otherwise immaterial.
    """

    def __init__(self, memory_size: int = 256) -> None:
        super().__init__()
        if memory_size < 1:
            raise ValueError("positive memory_size required")
        self.memory_size = memory_size
        self.query_norm = nn.LayerNorm(memory_size)
        self.record_norm = nn.LayerNorm(memory_size)
        self.query_projection = nn.Linear(memory_size, memory_size, bias=False)
        self.record_projection = nn.Linear(memory_size, memory_size, bias=False)

    def forward(self, query: Tensor, candidates: Tensor, valid: Tensor) -> Tensor:
        if query.ndim != 2 or query.shape[-1] != self.memory_size:
            raise ValueError(f"query must have shape [batch, {self.memory_size}]")
        if candidates.ndim != 3 or candidates.shape[-1] != self.memory_size:
            raise ValueError(
                f"candidates must have shape [batch, records, {self.memory_size}]"
            )
        if query.shape[0] != candidates.shape[0]:
            raise ValueError("query and candidate batch sizes must match")
        if tuple(valid.shape) != tuple(candidates.shape[:2]):
            raise ValueError("valid must match [batch, records]")
        if not query.is_floating_point() or not candidates.is_floating_point():
            raise ValueError("query and candidates must be floating point")
        if query.device != candidates.device:
            raise ValueError("query and candidates must be on the same device")
        valid = valid.to(device=candidates.device, dtype=torch.bool)
        clean = candidates.masked_fill(~valid.unsqueeze(-1), 0)
        q = self.query_projection(self.query_norm(query))
        k = self.record_projection(self.record_norm(clean))
        # FP32 normalization also makes low-precision score reductions stable.
        q = F.normalize(q.float(), dim=-1, eps=1e-6)
        k = F.normalize(k.float(), dim=-1, eps=1e-6)
        scores = (q.unsqueeze(1) * k).sum(dim=-1)
        return scores.masked_fill(~valid, float("-inf"))
