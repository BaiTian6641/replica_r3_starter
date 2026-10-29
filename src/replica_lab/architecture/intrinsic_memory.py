"""Model-owned memory readers, independent of external spine connections.

The residual attention primitive is shared with the legacy spine implementation;
packets, codecs, peripheral registration and SpineCore are not part of this API.
No read mutates memory. Reasoning-time writes need a separately declared causal
commit schedule; growing-prefix replay must never implicitly duplicate writes.
"""
from dataclasses import dataclass

import torch
from torch import nn

from .spine import SpinePort


@dataclass(frozen=True)
class IntrinsicMemorySpec:
    hidden_size: int
    width: int = 256
    slots: int = 8
    sites: tuple[int, ...] = (3, 7)

    @property
    def workspace_width(self):
        # Shape adapter for the reused residual-attention primitive only.
        return self.width


class IntrinsicMemoryReader(nn.Module):
    def __init__(self, spec, *, ports=None):
        super().__init__()
        if spec.width != 256 or spec.slots != 8 or spec.sites != (3, 7):
            raise ValueError("intrinsic v3 requires the declared two-site/eight-slot experiment")
        self.spec = spec
        self.ports = ports if ports is not None else nn.ModuleDict({
            str(site): SpinePort(spec, "memory", site) for site in spec.sites})
        if set(self.ports) != {str(site) for site in spec.sites}:
            raise ValueError("intrinsic reader site inventory differs")
        for port in self.ports.values():
            port.spec = spec
            port.branch = "intrinsic_memory"

    @classmethod
    def install(cls, adapter, spec):
        if any(site >= adapter.config.geometry.num_hidden_layers for site in spec.sites):
            raise ValueError("intrinsic memory site is outside the decoder")
        if hasattr(adapter.extension, "intrinsic_memory"):
            raise ValueError("intrinsic memory already has an owner")
        legacy = getattr(adapter.extension, "spine", None)
        ports = None
        if legacy is not None and "memory" in legacy.ports:
            # Transfer Parameter objects rather than registering aliases or
            # retaining a second memory owner under the external spine.
            old = legacy.ports.pop("memory")
            if tuple(legacy.spec.sites) == spec.sites:
                ports = old
            if hasattr(legacy, "memory_projection"):
                del legacy.memory_projection  # unused packet-codec projection
        reader = cls(spec, ports=ports).to(next(adapter.base.parameters()).device)
        adapter.extension.intrinsic_memory = reader
        return reader


class IntrinsicMemoryContext:
    """One immutable, revalidated read snapshot; not an external packet."""
    def __init__(self, reader, values, availability, *, validate):
        if (values.ndim != 4 or values.shape[-1] != reader.spec.width
                or not 0 < values.shape[2] <= reader.spec.slots
                or availability.shape != values.shape[:3]
                or availability.dtype != torch.bool or not torch.isfinite(values).all()):
            raise ValueError("invalid intrinsic memory snapshot")
        self.reader, self.values, self.availability = reader, values.clone(), availability.clone()
        self.spec = reader.spec
        self.validate = validate

    def delta(self, site, hidden):
        self.validate()
        if site not in self.spec.sites:
            raise ValueError("undeclared intrinsic memory site")
        return self.reader.ports[str(site)](hidden, self.values, self.availability)
