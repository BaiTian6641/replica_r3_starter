"""Fixed-projection plastic read path for one selected expert slot.

The plastic bank (``P``) itself lives in :mod:`replica_lab.nested_learning`.
This module owns only the *read* projection from the selected expert's bounded
256-wide output into a native residual.  The decoder is a frozen buffer, not a
trained parameter: it is ``R`` where ``R`` is the same fixed map used to build
teaching features and targets.  A zero slot therefore yields an exact null
residual, and no plastic read can create an optimizer-visible parameter or
change the native parameter stamp.

No source records, task labels, moments or clock state enter this module.
"""

from __future__ import annotations

import hashlib

import torch
from torch import Tensor, nn

__all__ = ["PlasticKnowledgePort", "fixed_projection", "projection_digest"]


def projection_digest(projection: Tensor) -> str:
    """Stable SHA-256 over a fixed FP32 projection's exact bytes."""
    values = projection.detach().to("cpu", torch.float32).contiguous()
    return hashlib.sha256(values.numpy().tobytes()).hexdigest()


def fixed_projection(hidden_size: int, width: int, seed: int) -> Tensor:
    """Deterministic fixed ``[width, hidden_size]`` projection (FP32)."""
    if type(hidden_size) is not int or hidden_size < 1:
        raise ValueError("hidden_size must be a positive integer")
    if type(width) is not int or width < 1:
        raise ValueError("width must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scale = 1.0 / hidden_size ** 0.5
    values = torch.randn(width, hidden_size, generator=generator,
                         dtype=torch.float32)
    return values * scale


class PlasticKnowledgePort(nn.Module):
    """Broadcast one selected plastic slot through a frozen decoder.

    ``slot`` is the selected expert output of shape ``[1, width]``.  The delta
    is ``gain * (slot @ decoder)`` broadcast over native positions and added to
    the residual stream.  Disabled calls and zero slots return exact zeros.
    """

    def __init__(self, hidden_size: int, width: int = 256,
                 decoder: Tensor | None = None, gain: float = 0.1) -> None:
        super().__init__()
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        if type(width) is not int or width < 1:
            raise ValueError("width must be a positive integer")
        if not isinstance(gain, (int, float)) or not torch.isfinite(torch.tensor(float(gain))):
            raise ValueError("gain must be finite")
        if float(gain) < 0:
            raise ValueError("gain must be nonnegative")
        if decoder is None:
            decoder = torch.zeros(width, hidden_size, dtype=torch.float32)
        if (not isinstance(decoder, Tensor) or decoder.dtype != torch.float32
                or tuple(decoder.shape) != (width, hidden_size)
                or not bool(torch.isfinite(decoder).all())):
            raise ValueError(
                f"decoder must be finite FP32 [{width}, {hidden_size}]")
        self.hidden_size = hidden_size
        self.width = width
        self.read_gain = float(gain)
        self.register_buffer("decoder", decoder.detach().clone(), persistent=True)
        self.decoder.requires_grad_(False)

    @property
    def decoder_digest(self) -> str:
        return projection_digest(self.decoder)

    def forward(self, hidden: Tensor, slot: Tensor, enabled: bool = True) -> Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden must be [batch, positions, {self.hidden_size}]")
        if not hidden.is_floating_point():
            raise ValueError("hidden must be floating point")
        if slot.ndim != 2 or slot.shape != (1, self.width):
            raise ValueError(f"slot must be [1, {self.width}]")
        if slot.device != hidden.device:
            raise ValueError("slot and hidden must share a device")
        if not bool(torch.isfinite(slot).all()):
            raise ValueError("slot must be finite")
        if not enabled:
            return torch.zeros_like(hidden)
        delta = slot.to(self.decoder.dtype) @ self.decoder
        delta = (delta * self.read_gain).to(hidden.dtype)
        return delta.unsqueeze(1).expand(-1, hidden.shape[1], -1)
