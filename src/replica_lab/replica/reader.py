"""Optional independently gated identity reader; disabled path is exact zero.

The host supplies already-approved identity slots; this module never reads the
archive, resolves a snapshot or decides which source is eligible. It is not yet
attached to a native block. Its final output projection starts at zero so an
untrained or null identity path preserves the native function.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class IdentityReader(nn.Module):
    def __init__(self, width: int = 256, heads: int = 4) -> None:
        super().__init__()
        if width < 1 or heads < 1 or width % heads:
            raise ValueError('positive head-divisible width required')
        self.width, self.heads = width, heads
        self.query_norm = nn.LayerNorm(width)
        self.slot_norm = nn.LayerNorm(width)
        self.query_projection = nn.Linear(width, width, bias=False)
        self.slot_projection = nn.Linear(width, width, bias=False)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True,
                                               dropout=0.0)
        self.output_projection = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.output_projection.weight)
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: Tensor, slots: Tensor, valid: Tensor | None = None,
                enabled: bool = True) -> Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.width:
            raise ValueError(f'hidden must be [batch, positions, {self.width}]')
        if slots.ndim != 3 or slots.shape[-1] != self.width or slots.shape[0] != hidden.shape[0]:
            raise ValueError('slots must match the hidden batch and width')
        if not hidden.is_floating_point() or not slots.is_floating_point():
            raise ValueError('identity reader requires floating tensors')
        if valid is None:
            valid = torch.ones(slots.shape[:2], dtype=torch.bool, device=slots.device)
        if valid.dtype != torch.bool or tuple(valid.shape) != tuple(slots.shape[:2]):
            raise ValueError('valid must be a boolean slot mask')
        if not enabled or slots.shape[1] == 0:
            return hidden * 0
        available = valid.any(dim=1)
        if not bool(available.any()):
            return hidden * 0
        safe = valid.clone()
        safe[:, 0] = safe[:, 0] | ~available
        clean = slots.masked_fill(~safe.unsqueeze(-1), 0)
        clean = self.slot_projection(self.slot_norm(clean))
        query = self.query_projection(self.query_norm(hidden))
        value, _ = self.attention(query, clean, clean,
                                  key_padding_mask=~safe, need_weights=False)
        delta = self.gate_logit.sigmoid() * self.output_projection(value)
        return (delta * available[:, None, None].to(delta.dtype)).to(hidden.dtype)
