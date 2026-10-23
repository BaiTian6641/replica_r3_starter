"""Explicit E1/E2 ownership and provenance; no model or storage side effects."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import hashlib
import json
from typing import Any
import torch


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class PortSpec:
    injection_layers: tuple[int, ...] = (13, 27)
    observer_layer: int = 41
    source_layer: int = 13
    memory_width: int = 256
    slots_per_unit: int = 4
    max_units: int = 16
    source_tokens: int = 128
    heads: int = 4

    def validate(self, layers: int) -> None:
        if not self.injection_layers or len(set(self.injection_layers)) != len(self.injection_layers):
            raise ValueError('Distinct injection ports required')
        if any(type(i) is not int or not 0 <= i < layers for i in
               (*self.injection_layers, self.observer_layer, self.source_layer)):
            raise ValueError('Port outside native backbone')
        if any(type(i) is not int or i <= 0 for i in
               (self.memory_width, self.slots_per_unit, self.max_units, self.source_tokens, self.heads)):
            raise ValueError('Positive integer port dimensions required')
        if self.memory_width % self.heads:
            raise ValueError('Memory width must divide heads')


@dataclass(frozen=True)
class EvidencePin:
    ledger_id: str
    epoch: int
    subject_id: str
    branch_id: str
    purpose: str
    as_of: int
    mode: str
    snapshot_at: int | None
    records: tuple[tuple[str, str], ...]

    @property
    def digest(self):
        return fingerprint(asdict(self))


@dataclass(frozen=True)
class SourceSpan:
    record_id: str
    record_hash: str
    source_hash: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    token_ids: tuple[int, ...]
    span_hash: str

    @property
    def key(self):
        return fingerprint(asdict(self))


@dataclass(frozen=True)
class MemoryRead:
    """One per-record unit retains provenance through flattening/selection."""
    spans: tuple[SourceSpan, ...]
    slots: torch.Tensor  # [1,R,K,D]; one request/batch row for now
    valid: torch.Tensor  # [1,R,K]
    producer: str


@dataclass(frozen=True)
class SenseFrame:
    summaries: tuple[tuple[int, torch.Tensor], ...]
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    producer: str


@dataclass(frozen=True)
class TaskGraph:
    task_id: str
    goal: str
    revision: int = 0
    constraints: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    open_obligations: tuple[str, ...] = ()
    # E1 carries exact state; learned task transitions belong to E3.


@dataclass(frozen=True)
class PendingWrite:
    event_id: str
    record: Any
    expected_epoch: int


@dataclass(frozen=True)
class ComponentManifest:
    source_revision: str
    source_seal_digest: str
    tokenizer_digest: str
    native_config_digest: str
    native_abi_digest: str
    ports: PortSpec
    format_version: int = 1
    training_scope: str = 'text_only'
    cached_generation_supported: bool = False

    def to_dict(self):
        return asdict(self)

    @property
    def digest(self):
        return fingerprint(self.to_dict())


def tensor_stamp(tensor: torch.Tensor):
    try:
        version = tensor._version
    except RuntimeError as error:
        raise ValueError('Owned frame tensors require version tracking; build under no_grad, not inference_mode') from error
    return (id(tensor), version, tuple(tensor.shape), str(tensor.device), str(tensor.dtype), tensor.data_ptr())


@dataclass(frozen=True)
class DecisionFrame:
    decision_id: str
    evidence: EvidencePin
    memory: MemoryRead
    sense: SenseFrame
    manifest_digest: str
    parameter_stamp: tuple
    enabled_ports: tuple[int, ...]
    _tensor_stamps: tuple = field(init=False, repr=False, compare=False)

    def tensors(self):
        return (self.memory.slots, self.memory.valid, self.sense.prompt_ids,
                self.sense.prompt_mask, *(x for _, x in self.sense.summaries))

    def __post_init__(self):
        object.__setattr__(self, '_tensor_stamps', tuple(tensor_stamp(t) for t in self.tensors()))

    def assert_immutable(self):
        if self._tensor_stamps != tuple(tensor_stamp(t) for t in self.tensors()):
            raise RuntimeError('DecisionFrame tensor mutated; rebuild at a new decision boundary')


@dataclass(frozen=True)
class InjectFrame:
    decision: DecisionFrame
    layer: int

    @property
    def enabled(self):
        return self.layer in self.decision.enabled_ports


@dataclass
class RequestContext:
    request_id: str
    task: TaskGraph
    frame: DecisionFrame | None
    selection: Any  # host-owned immutable RetrievalSelection
    state: str = 'ready'
    native_cache: Any = None
    generated_ids: tuple[int, ...] = ()

    def assert_ready(self):
        if self.state not in {'ready', 'training'}:
            raise RuntimeError(f'Request is {self.state}; rebuild its decision')
        if self.native_cache is not None:
            raise ValueError('Native cache reuse is not qualified by the E1/E2 reference')
        if self.frame is None:
            raise RuntimeError('Request tensors were released')
        self.frame.assert_immutable()

    def invalidate(self):
        self.native_cache = None
        self.generated_ids = ()
        self.frame = None
        self.selection = None
        self.state = 'invalid'

    def release(self):
        self.native_cache = None
        self.frame = None
        self.selection = None
        self.state = 'released'
