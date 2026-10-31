"""External neural spine: typed packets, trained codecs and model endpoints.

External communication is distinct from intrinsic memory. Composite memory and
workspace branches remain here for historical checkpoint compatibility; intrinsic
v3 explicitly transfers memory ownership to its own decoder path. A spine context
computes packet slots once for its endpoints. Matching tensor widths alone is not
a claim of semantic compatibility with another model. No durable writes occur here.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import json
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from .spark2_5 import fingerprint

SPINE_SCHEMA = "replica.spark.spine.v1"
SPINE_EXTENSION_SCHEMA = "replica.spark2_5_extension_export.v2"
PACKET_SCHEMA = "replica.spark.packet.v1"
PAYLOAD_TYPES = frozenset({"observation", "interpretation", "prediction",
                           "verified_feedback"})
BRANCHES = ("spine", "memory", "workspace")


class SpineError(ValueError):
    """Packet, spine, or native integration contract violation."""


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SpineError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpineError(f"{name} must be a nonempty string")
    return value


def _bool_tensor(value: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise SpineError(f"{name} must be a tensor")
    if value.dtype not in (torch.bool, torch.int8, torch.int16, torch.int32,
                           torch.int64, torch.uint8):
        raise SpineError(f"{name} must be an integer/bool availability mask")
    return value.to(dtype=torch.bool)


def _masked_attention(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Softmax with an exact zero row for every fully masked query."""
    if scores.shape != valid.shape:
        raise SpineError("attention scores and availability mask shapes differ")
    masked = scores.masked_fill(~valid, -1e9)
    weights = torch.softmax(masked, dim=-1)
    weights = weights * valid.to(dtype=weights.dtype)
    denominator = weights.sum(dim=-1, keepdim=True)
    normalized = weights / denominator.clamp_min(1e-8)
    return torch.where(denominator > 0, normalized,
                       torch.zeros_like(normalized))


@dataclass(frozen=True)
class Packet:
    """One immutable structured observation/feedback packet."""

    schema: str
    producer_revision: str
    representation_version: str
    codec_version: str
    scope: str
    stream_id: str
    event_id: str
    sequence: int
    observation_start: int
    observation_end: int
    available_at: int
    state_revision: int
    payload_type: str
    source_lineage: Mapping[str, Any]
    payload: torch.Tensor = field(compare=False)
    availability: torch.Tensor = field(compare=False)
    validity_start: int | None = None
    validity_end: int | None = None
    deadline: int | None = None
    units: str = "logical_unit"
    coordinate_frame: str = "text"
    provider_confidence: float | None = None
    verified: bool = False

    def __post_init__(self) -> None:
        if self.schema != PACKET_SCHEMA:
            raise SpineError(f"packet schema must be {PACKET_SCHEMA}")
        for name in ("producer_revision", "representation_version", "codec_version",
                     "scope", "stream_id", "event_id"):
            _text(getattr(self, name), name)
        _int(self.sequence, "sequence")
        _int(self.observation_start, "observation_start")
        _int(self.observation_end, "observation_end")
        _int(self.available_at, "available_at")
        _int(self.state_revision, "state_revision")
        if self.observation_end < self.observation_start:
            raise SpineError("observation interval is reversed")
        if self.available_at < self.observation_end:
            raise SpineError("packet availability precedes observation end")
        if self.payload_type not in PAYLOAD_TYPES:
            raise SpineError("unknown packet payload type")
        if not isinstance(self.source_lineage, Mapping) or not self.source_lineage:
            raise SpineError("source_lineage must be a nonempty mapping")
        if not torch.is_tensor(self.payload) or not torch.is_floating_point(self.payload):
            raise SpineError("packet payload must be a floating tensor")
        if self.payload.ndim != 3 or self.payload.shape[-1] != 512:
            raise SpineError("packet payload must have shape [B,S,512]")
        if not bool(torch.isfinite(self.payload).all()):
            raise SpineError("packet payload contains nonfinite values")
        availability = _bool_tensor(self.availability, "availability")
        if availability.ndim != 3 or availability.shape[1] <= 0:
            raise SpineError("availability must have shape [B,Q,S] with Q > 0")
        if availability.shape[0] != self.payload.shape[0] \
                or availability.shape[2] != self.payload.shape[1]:
            raise SpineError("availability and payload batch/slot dimensions differ")
        # Own all tensor storage at the public packet boundary.
        object.__setattr__(self, "payload", self.payload.detach().clone())
        object.__setattr__(self, "availability", availability.detach().clone())
        if self.validity_start is not None:
            _int(self.validity_start, "validity_start")
        if self.validity_end is not None:
            _int(self.validity_end, "validity_end")
            if self.validity_end < self.available_at:
                raise SpineError("validity interval expires before availability")
        if self.deadline is not None:
            _int(self.deadline, "deadline")
            if self.deadline < self.available_at:
                raise SpineError("deadline precedes availability")
        if self.provider_confidence is not None and not 0 <= self.provider_confidence <= 1:
            raise SpineError("provider_confidence must lie in [0,1]")
        if self.verified and self.payload_type != "verified_feedback":
            raise SpineError("only verified_feedback may carry verified=true")

    @property
    def batch_size(self) -> int:
        return int(self.payload.shape[0])

    @property
    def slot_count(self) -> int:
        return int(self.payload.shape[1])

    @property
    def query_count(self) -> int:
        return int(self.availability.shape[1])

    @property
    def payload_bytes(self) -> int:
        return int(self.payload.numel() * self.payload.element_size())

    def clone(self) -> "Packet":
        """Return an ownership-independent packet, including nested lineage."""
        return replace(self, source_lineage=copy.deepcopy(self.source_lineage))


    def metadata(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "producer_revision": self.producer_revision,
            "representation_version": self.representation_version,
            "codec_version": self.codec_version,
            "scope": self.scope,
            "stream_id": self.stream_id,
            "event_id": self.event_id,
            "sequence": self.sequence,
            "observation_start": self.observation_start,
            "observation_end": self.observation_end,
            "available_at": self.available_at,
            "state_revision": self.state_revision,
            "payload_type": self.payload_type,
            "source_lineage": copy.deepcopy(self.source_lineage),
            "validity_start": self.validity_start,
            "validity_end": self.validity_end,
            "deadline": self.deadline,
            "units": self.units,
            "coordinate_frame": self.coordinate_frame,
            "provider_confidence": self.provider_confidence,
            "verified": self.verified,
            "payload_bytes": self.payload_bytes,
        }


@dataclass(frozen=True)
class SpineSpec:
    hidden_size: int
    packet_width: int = 512
    workspace_width: int = 256
    slots: int = 8
    sites: tuple[int, ...] = (11, 23)
    heads: int = 4

    def __post_init__(self) -> None:
        for name in ("hidden_size", "packet_width", "workspace_width", "slots", "heads"):
            _int(getattr(self, name), name, minimum=1)
        if self.packet_width != 512 or self.workspace_width != 256:
            raise SpineError("initial spine uses packet width 512 and workspace width 256")
        if self.workspace_width % self.heads:
            raise SpineError("workspace width must divide heads")
        if len(set(self.sites)) != len(self.sites) or any(
                not isinstance(site, int) or isinstance(site, bool) or site < 0
                for site in self.sites):
            raise SpineError("spine sites must be distinct nonnegative integers")
        if not self.sites:
            raise SpineError("at least one spine site is required")

    def to_dict(self) -> dict[str, Any]:
        return {"hidden_size": self.hidden_size, "packet_width": self.packet_width,
                "workspace_width": self.workspace_width, "slots": self.slots,
                "sites": list(self.sites), "heads": self.heads}


class IngestionCodec(nn.Module):
    """Trainable packet-to-slot codec with causal availability masking."""

    def __init__(self, spec: SpineSpec) -> None:
        super().__init__()
        self.spec = spec
        width = spec.packet_width
        self.input_norm = nn.LayerNorm(width)
        self.key_projection = nn.Linear(width, width, bias=False)
        self.value_projection = nn.Linear(width, width, bias=False)
        self.slot_queries = nn.Parameter(torch.empty(spec.slots, width))
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, width, bias=False)
        nn.init.normal_(self.slot_queries, std=0.02)

    def forward(self, payload: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        if payload.ndim != 3 or payload.shape[-1] != self.spec.packet_width:
            raise SpineError("codec payload must be [B,S,512]")
        if available.shape != payload.shape[:2]:
            raise SpineError("codec availability must be [B,S]")
        if payload.shape[1] > self.spec.slots:
            raise SpineError("packet slot count exceeds the declared spine cap")
        if not bool(available.any()):
            return payload.new_zeros(
                payload.shape[0], payload.shape[1], self.spec.packet_width)
        source = payload.masked_fill(~available.unsqueeze(-1), 0)
        source = self.input_norm(source)
        key = self.key_projection(source)
        value = self.value_projection(source)
        query = self.slot_queries[:payload.shape[1]].unsqueeze(0).expand(
            payload.shape[0], -1, -1)
        scores = torch.einsum("bqd,bsd->bqs", query, key) / (key.shape[-1] ** 0.5)
        weights = _masked_attention(
            scores, available.unsqueeze(1).expand(-1, payload.shape[1], -1))
        read = torch.einsum("bqs,bsd->bqd", weights, value)
        return self.output_projection(self.output_norm(read))


def _packet_digest(packet: Packet) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(json.dumps(packet.metadata(), sort_keys=True,
                             default=str).encode("utf-8"))
    for tensor in (packet.payload, packet.availability):
        value = tensor.detach().to("cpu").contiguous()
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class PacketBuffer:
    """Bounded host-side packet queue with content deduplication and expiry."""

    def __init__(self, *, max_packets: int = 64, max_payload_bytes: int = 1 << 20,
                 max_pending: int = 64) -> None:
        for name, value in (("max_packets", max_packets),
                            ("max_payload_bytes", max_payload_bytes),
                            ("max_pending", max_pending)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SpineError(f"{name} must be positive")
        self.max_packets = max_packets
        self.max_payload_bytes = max_payload_bytes
        self.max_pending = max_pending
        self._packets: dict[tuple, tuple[Packet, str]] = {}
        self._last_sequence: dict[tuple[str, str], int] = {}
        self._payload_bytes = 0

    @staticmethod
    def _key(packet: Packet) -> tuple:
        return (packet.schema, packet.producer_revision,
                packet.representation_version, packet.codec_version,
                packet.scope, packet.stream_id, packet.event_id,
                packet.sequence, packet.state_revision)

    @property
    def pending(self) -> int:
        return len(self._packets)

    @property
    def payload_bytes(self) -> int:
        return self._payload_bytes

    def append(self, packet: Packet, *, now: int) -> bool:
        _int(now, "now")
        if packet.validity_end is not None and packet.validity_end < now:
            return False
        if packet.deadline is not None and packet.deadline < now:
            return False
        key = self._key(packet)
        digest = _packet_digest(packet)
        if key in self._packets:
            if self._packets[key][1] != digest:
                raise SpineError("duplicate packet has changed payload")
            return False
        stream = (packet.scope, packet.stream_id)
        previous = self._last_sequence.get(stream, -1)
        if packet.sequence != previous + 1:
            raise SpineError(
                f"missing or reordered packet sequence: {previous} -> {packet.sequence}")
        if self.pending >= self.max_packets or self.pending >= self.max_pending:
            raise SpineError("packet queue/pending limit exceeded")
        if self._payload_bytes + packet.payload_bytes > self.max_payload_bytes:
            raise SpineError("packet payload byte limit exceeded")
        owned = packet.clone()
        self._packets[key] = (owned, digest)
        self._payload_bytes += owned.payload_bytes
        self._last_sequence[stream] = packet.sequence
        return True

    def expire(self, *, now: int) -> int:
        _int(now, "now")
        removed = 0
        for key, (packet, _) in list(self._packets.items()):
            if ((packet.validity_end is not None and packet.validity_end < now)
                    or (packet.deadline is not None and packet.deadline < now)):
                del self._packets[key]
                self._payload_bytes -= packet.payload_bytes
                removed += 1
        return removed

    def snapshot(self) -> tuple[Packet, ...]:
        return tuple(packet.clone() for packet, _ in sorted(
            self._packets.values(),
            key=lambda item: (item[0].scope, item[0].stream_id,
                              item[0].sequence)))


class SpinePort(nn.Module):
    """One independently gated residual reader at one Spark layer."""

    def __init__(self, spec: SpineSpec, branch: str, site: int) -> None:
        super().__init__()
        if branch not in BRANCHES:
            raise SpineError(f"unknown spine branch {branch!r}")
        self.spec, self.branch, self.site = spec, branch, site
        width = spec.workspace_width
        self.query_norm = nn.LayerNorm(spec.hidden_size)
        self.query_projection = nn.Linear(spec.hidden_size, width, bias=False)
        self.slot_norm = nn.LayerNorm(width)
        self.key_projection = nn.Linear(width, width, bias=False)
        self.value_projection = nn.Linear(width, width, bias=False)
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, spec.hidden_size, bias=False)
        self.gate_logit = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, hidden: torch.Tensor, slots: torch.Tensor,
                availability: torch.Tensor, return_attention: bool = False,
                key_slots=None) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.spec.hidden_size:
            raise SpineError("port hidden state has the wrong shape")
        if slots.ndim != 4 or slots.shape[-1] != self.spec.workspace_width:
            raise SpineError("port slots must be [B,Q,S,W]")
        if availability.ndim != 3 or availability.shape[:2] != hidden.shape[:2] \
                or availability.shape[:2] != slots.shape[:2] \
                or availability.shape[2] != slots.shape[2]:
            raise SpineError("port availability shape differs from hidden/slots")
        if not bool(availability.any()):
            delta = hidden.new_zeros(hidden.shape)
            if not return_attention:
                return delta
            query = self.query_projection(self.query_norm(hidden)).unsqueeze(1)
            empty = slots.new_zeros(slots.shape)
            weights = hidden.new_zeros(
                (hidden.shape[0], 1, hidden.shape[1], slots.shape[2]))
            read = hidden.new_zeros(
                (hidden.shape[0], 1, hidden.shape[1], self.spec.workspace_width))
            return delta, {'query': query, 'key': empty, 'value': empty,
                           'weights': weights, 'read': read}
        if key_slots is not None and key_slots.shape != slots.shape:
            raise SpineError("content-key slots must match value-slot shape")
        source = slots.masked_fill(~availability.unsqueeze(-1), 0)
        key_source = slots if key_slots is None else key_slots
        key_source = key_source.masked_fill(~availability.unsqueeze(-1), 0)
        source = self.slot_norm(source)
        key = self.key_projection(self.slot_norm(key_source))
        value = self.value_projection(source)
        query = self.query_projection(self.query_norm(hidden))
        scores = (query.unsqueeze(2) * key).sum(dim=-1) / (key.shape[-1] ** 0.5)
        weights = _masked_attention(scores, availability)
        read = (weights.unsqueeze(-1) * value).sum(dim=2)
        delta = self.output_projection(self.output_norm(read))
        delta = delta * self.gate_logit.sigmoid()
        delta = delta * availability.any(dim=-1).to(delta.dtype).unsqueeze(-1)
        if return_attention:
            return delta, {'query': query.unsqueeze(1), 'key': key, 'value': value,
                           'weights': weights.unsqueeze(1), 'read': read.unsqueeze(1)}
        return delta.to(dtype=hidden.dtype)


class SpineCore(nn.Module):
    """Typed external connector, retaining composite branches for legacy loads.

    Intrinsic v3 transfers the old memory reader to its independent owner. New
    memory execution never constructs a packet or calls this core. Historical
    full-extension checkpoints can still load before the explicit transfer.
    """

    def __init__(self, spec: SpineSpec, *, codec_version: str = "codec-1",
                 representation_version: str = "repr-1") -> None:
        super().__init__()
        self.spec = spec
        self.codec_version = _text(codec_version, "codec_version")
        self.representation_version = _text(
            representation_version, "representation_version")
        self.codec = IngestionCodec(spec)
        self.spine_projection = nn.Linear(spec.packet_width, spec.workspace_width,
                                          bias=False)
        self.memory_projection = nn.Linear(spec.packet_width, spec.workspace_width,
                                           bias=False)
        self.workspace_projection = nn.Linear(spec.packet_width, spec.workspace_width,
                                              bias=False)
        self.ports = nn.ModuleDict({
            branch: nn.ModuleDict({
                str(site): SpinePort(spec, branch, site)
                for site in spec.sites})
            for branch in BRANCHES})
        self.float()

    def encode(self, packet: Packet) -> dict[str, torch.Tensor]:
        if packet.codec_version != self.codec_version \
                or packet.representation_version != self.representation_version:
            raise SpineError("packet representation/codec differs from the spine core")
        device = self.codec.input_norm.weight.device
        dtype = self.codec.input_norm.weight.dtype
        payload = packet.payload.to(device=device, dtype=dtype)
        availability = packet.availability.to(device=device)
        if payload is not packet.payload or availability is not packet.availability:
            packet = replace(packet, payload=payload, availability=availability)
        batch, query_count, slot_count, width = (
            payload.shape[0], availability.shape[1], payload.shape[1], payload.shape[2])
        expanded = payload.unsqueeze(1).expand(
            batch, query_count, slot_count, width)
        encoded = self.codec(
            expanded.reshape(batch * query_count, slot_count, width),
            availability.reshape(batch * query_count, slot_count))
        encoded = encoded.reshape(batch, query_count, slot_count, width)
        # Legacy checkpoints retain all three branches until memory ownership
        # is explicitly migrated to extension.intrinsic_memory. External
        # connections continue to use the remaining typed branches afterward.
        return {branch: getattr(self, branch + "_projection")(encoded)
                for branch in self.ports}

    def port(self, branch: str, site: int, hidden: torch.Tensor,
             slots: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        if branch not in self.ports or site not in self.spec.sites:
            raise SpineError(f"undeclared spine port {branch}/{site}")
        return self.ports[branch][str(site)](hidden, slots, availability)

    def inventory(self) -> dict[str, Any]:
        tensors = [{"name": f"extension.spine.{name}", "shape": list(parameter.shape),
                    "dtype": "FLOAT32"}
                   for name, parameter in self.named_parameters()]
        return {
            "schema": SPINE_SCHEMA,
            "spec": self.spec.to_dict(),
            "codec_version": self.codec_version,
            "representation_version": self.representation_version,
            "trainable": True,
            "zero_effect_final_projections": True,
            "live_gate_path": True,
            "tensors": tensors,
            "parameter_elements": sum(parameter.numel()
                                       for parameter in self.parameters()),
            "parameter_inventory_sha256": fingerprint(tensors),
        }


@dataclass(frozen=True)
class DecisionSnapshot:
    decision_id: str
    scope: str
    stream_id: str
    state_revision: int
    arm: str
    event: str
    phase: str


class SpineForwardContext:
    """One local, immutable decision context with an explicit work charge."""

    def __init__(self, core: SpineCore, packet: Packet, snapshot: DecisionSnapshot,
                 *, enabled_branches: Iterable[str] = BRANCHES,
                 memory_slots: torch.Tensor | None = None,
                 memory_source: str | None = None,
                 memory_availability: torch.Tensor | None = None) -> None:
        if memory_source not in (None, "E", "P", "E_plus_P", "external"):
            raise SpineError("unknown memory slot source")
        if memory_source in ("P", "E_plus_P") and memory_slots is None:
            raise SpineError(f"{memory_source} memory source requires explicit slots")
        if memory_source == "E" and memory_slots is not None:
            raise SpineError("E memory source cannot use replacement slots")
        self.memory_source = memory_source or ("external" if memory_slots is not None else "E")
        self.p_memory_slots: torch.Tensor | None = None
        self.p_memory_availability: torch.Tensor | None = None
        self._invalidated = False
        self.core = core
        device = core.codec.input_norm.weight.device
        dtype = core.codec.input_norm.weight.dtype
        if packet.codec_version != core.codec_version \
                or packet.representation_version != core.representation_version:
            raise SpineError("packet representation/codec differs from the spine core")
        packet = replace(
            packet,
            payload=packet.payload.to(device=device, dtype=dtype),
            availability=packet.availability.to(device=device))
        self.packet = packet
        self.snapshot = snapshot
        self.enabled_branches = tuple(enabled_branches)
        if len(set(self.enabled_branches)) != len(self.enabled_branches):
            raise SpineError("duplicate enabled spine branch")
        if any(branch not in BRANCHES for branch in self.enabled_branches):
            raise SpineError("unknown enabled spine branch")
        if packet.scope != snapshot.scope or packet.stream_id != snapshot.stream_id \
                or packet.state_revision != snapshot.state_revision:
            raise SpineError("packet and decision snapshot identity differ")
        self.slots = core.encode(packet) if self.enabled_branches else {}
        self.branch_availability = {
            branch: packet.availability for branch in self.enabled_branches}
        if memory_slots is not None:
            if not torch.is_tensor(memory_slots) or memory_slots.ndim != 4 \
                    or tuple(memory_slots.shape[:2]) != (packet.batch_size,
                                                        packet.query_count) \
                    or memory_slots.shape[2] <= 0 \
                    or memory_slots.shape[2] > core.spec.slots \
                    or memory_slots.shape[3] != core.spec.workspace_width:
                raise SpineError(
                    "memory_slots must have shape [B,Q,S,W] with 0 < S <= slots")
            if not torch.is_floating_point(memory_slots) \
                    or not bool(torch.isfinite(memory_slots).all()):
                raise SpineError("memory_slots must be finite floating point")
            if memory_availability is None:
                if self.memory_source == "E_plus_P":
                    memory_availability = torch.ones(
                        memory_slots.shape[:3], dtype=torch.bool,
                        device=memory_slots.device)
                elif memory_slots.shape[2] == packet.slot_count:
                    memory_availability = packet.availability
                elif self.memory_source == "P":
                    memory_availability = torch.ones(
                        memory_slots.shape[:3], dtype=torch.bool,
                        device=memory_slots.device)
                else:
                    raise SpineError(
                        "replacement memory slots require memory_availability")
            memory_availability = _bool_tensor(
                memory_availability, "memory_availability")
            if tuple(memory_availability.shape) != tuple(memory_slots.shape[:3]):
                raise SpineError("memory_availability shape differs from memory_slots")
            owned_slots = memory_slots.to(device=device, dtype=dtype).clone()
            owned_availability = memory_availability.to(device=device).clone()
            if self.memory_source == "E_plus_P":
                self.p_memory_slots = owned_slots
                self.p_memory_availability = owned_availability
            else:
                self.slots["memory"] = owned_slots
                self.branch_availability["memory"] = owned_availability
        elif memory_availability is not None:
            raise SpineError("memory_availability requires memory_slots")
        self.charges = {
            "codec_calls": self.packet.query_count if self.enabled_branches else 0,
            "codec_positions": (self.packet.batch_size * self.packet.query_count
                                 if self.enabled_branches else 0),
            "port_calls": 0,
            "packet_count": packet.slot_count,
            "packet_bytes": packet.payload_bytes,
            "memory_read_slots": (
                (packet.slot_count if self.memory_source == "E_plus_P" else 0)
                + (0 if memory_slots is None else int(memory_slots.shape[2]))),
            "e_memory_read_slots": (
                packet.slot_count if self.memory_source == "E_plus_P" else 0),
            "p_memory_read_slots": (
                int(memory_slots.shape[2])
                if memory_slots is not None
                and self.memory_source in ("P", "E_plus_P") else 0),
            "exchange_rounds": 0,
            "specialist_calls": 0,
        }
    @property
    def invalidated(self) -> bool:
        return self._invalidated

    def invalidate(self) -> None:
        self._invalidated = True


    def delta(self, branch: str, site: int, hidden: torch.Tensor) -> torch.Tensor:
        if self._invalidated:
            raise SpineError("decision context is stale")
        if branch not in self.enabled_branches:
            return hidden.new_zeros(hidden.shape)
        if site not in self.core.spec.sites:
            raise SpineError(f"undeclared spine site {site}")
        if branch == "memory" and self.memory_source == "E_plus_P":
            self.charges["port_calls"] += 2
            e_delta = self.core.port(
                branch, site, hidden, self.slots[branch],
                self.branch_availability[branch])
            p_delta = self.core.port(
                branch, site, hidden, self.p_memory_slots,
                self.p_memory_availability)
            return e_delta + p_delta
        self.charges["port_calls"] += 1
        return self.core.port(branch, site, hidden, self.slots[branch],
                              self.branch_availability[branch])


def spine_inventory(core: SpineCore) -> dict[str, Any]:
    """Return the actual selected dimensions and tensor inventory."""
    return core.inventory()


__all__ = [
    "SPINE_SCHEMA", "SPINE_EXTENSION_SCHEMA", "PACKET_SCHEMA", "PAYLOAD_TYPES",
    "BRANCHES", "SpineError", "Packet", "PacketBuffer", "SpineSpec", "IngestionCodec",
    "SpinePort", "SpineCore", "DecisionSnapshot", "SpineForwardContext",
    "spine_inventory",
]
