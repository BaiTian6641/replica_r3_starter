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
class CoreSpec:
    """Declared shared-core geometry; validated against the native port layout.

    ``width`` is the workspace/record-memory width so workspace readers reuse the
    same projection arithmetic as memory readers. ``ffn_hidden`` is the tied
    core's four-width feed-forward; ``control_hidden`` sizes the feed-forward
    control arm to approximately the recurrent core's parameter count.
    """
    width: int = 256
    slots: int = 8
    steps: int = 2
    heads: int = 4
    task_features: int = 24
    ffn_hidden: int = 1024
    control_hidden: int = 340

    def validate(self, ports: PortSpec) -> None:
        for name in ('width', 'slots', 'steps', 'heads', 'task_features',
                     'ffn_hidden', 'control_hidden'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'CoreSpec.{name} must be a positive integer')
        if self.width != ports.memory_width:
            raise ValueError('Workspace width must equal the record-memory width')
        if self.width % self.heads:
            raise ValueError('Workspace width must divide core heads')
        if self.steps > 8:
            raise ValueError('Core recurrence is bounded to eight tied iterations')


@dataclass(frozen=True)
class CoreState:
    """One task-owned latent workspace plus its exact task-feature binding.

    The tensors are owned by one decision boundary. ``producer`` binds the core,
    task encoder, sensors and memory writer; ``task_digest`` binds the exact task
    revision; ``revision`` is the exact journal revision that produced it.
    """
    workspace: torch.Tensor  # [1, slots, width]
    valid: torch.Tensor      # [1, slots] bool
    task_features: torch.Tensor  # [1, task_features] float
    producer: str
    task_digest: str
    revision: int
    steps: int
    implementation: str = 'recurrent'
    identity_digest: str | None = None
    _tensor_stamps: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if (self.workspace.ndim != 3 or self.workspace.shape[0] != 1
                or self.workspace.shape[1] < 1 or self.workspace.shape[2] < 1):
            raise ValueError('Core workspace must be [1, slots, width]')
        if not self.workspace.is_floating_point() or not bool(torch.isfinite(self.workspace).all()):
            raise ValueError('Core workspace must be finite floating point')
        if (self.valid.dtype != torch.bool
                or tuple(self.valid.shape) != tuple(self.workspace.shape[:2])
                or self.valid.device != self.workspace.device):
            raise ValueError('Core validity must be a matching boolean mask')
        if (self.task_features.ndim != 2 or self.task_features.shape[0] != 1
                or self.task_features.shape[1] < 1
                or not self.task_features.is_floating_point()
                or not bool(torch.isfinite(self.task_features).all())
                or self.task_features.device != self.workspace.device):
            raise ValueError('Core task features must be finite [1, features] on the workspace device')
        if not isinstance(self.producer, str) or not self.producer:
            raise ValueError('Core producer identity required')
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError('Core revision must be a nonnegative integer')
        if type(self.steps) is not int or self.steps < 0:
            raise ValueError('Core steps must be nonnegative')
        if self.implementation not in {'recurrent', 'feedforward'}:
            raise ValueError('Unknown core implementation')
        if self.identity_digest is not None and (
                not isinstance(self.identity_digest, str) or len(self.identity_digest) != 64):
            raise ValueError('identity_digest must be a SHA256 or None')
        object.__setattr__(self, '_tensor_stamps', tuple(tensor_stamp(t) for t in self.tensors()))

    def tensors(self):
        return (self.workspace, self.valid, self.task_features)

    def assert_immutable(self):
        if self._tensor_stamps != tuple(tensor_stamp(t) for t in self.tensors()):
            raise RuntimeError('CoreState tensor mutated; rebuild at a new decision boundary')

    def detached(self):
        return CoreState(self.workspace.detach(), self.valid.detach(),
                         self.task_features.detach(), self.producer, self.task_digest,
                         self.revision, self.steps, self.implementation,
                         self.identity_digest)

    def stamp(self):
        """Stable lineage stamp for journals; never a confidence or truth score."""
        return fingerprint({'producer': self.producer, 'task_digest': self.task_digest,
                            'revision': self.revision, 'steps': self.steps,
                            'implementation': self.implementation,
                            'identity_digest': self.identity_digest,
                            'workspace': [float(v) for v in self.workspace.detach().cpu().flatten().tolist()],
                            'valid': [bool(v) for v in self.valid.detach().cpu().flatten().tolist()]})


@dataclass(frozen=True)
class FastMemorySpec:
    """Fast-state geometry and update rule; no native reader is implied."""
    heads: int = 4
    key_size: int = 64
    value_size: int = 64
    update_rule: str = 'linear'
    def __post_init__(self):
        for name in ('heads', 'key_size', 'value_size'):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 1024:
                raise ValueError(f'FastMemorySpec.{name} must be in 1..1024')
        if self.update_rule not in {'linear', 'sgdn2'}:
            raise ValueError('FastMemorySpec.update_rule must be linear or sgdn2')


@dataclass(frozen=True)
class CopySpec:
    """Bounded exact record-token copy interface for answer-time copying."""
    max_tokens: int = 512
    mode: str = 'logit_bias'
    max_span: int = 4

    def __post_init__(self):
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 2048:
            raise ValueError('CopySpec.max_tokens must be in 1..2048')
        if self.mode not in {'logit_bias', 'pointer'}:
            raise ValueError('CopySpec.mode must be logit_bias or pointer')
        if type(self.max_span) is not int or not 1 <= self.max_span <= 32:
            raise ValueError('CopySpec.max_span must be in 1..32')


@dataclass(frozen=True)
class IdentitySpec:
    """Optional workspace identity reader; width must match the core workspace."""
    width: int = 256
    heads: int = 4

    def __post_init__(self):
        if type(self.width) is not int or self.width < 1:
            raise ValueError('IdentitySpec.width must be positive')
        if type(self.heads) is not int or self.heads < 1 or self.width % self.heads:
            raise ValueError('IdentitySpec.heads must divide width')

    def validate(self, ports, core) -> None:
        if core is None:
            raise ValueError('Identity conditioning requires a declared core')
        if self.width != ports.memory_width or self.width != core.width:
            raise ValueError('Identity width must match memory and core width')


@dataclass(frozen=True)
class IdentityRead:
    """Host-approved identity slots and their exact context binding.

    Slots are supplied by the host from an already resolved IdentityContext or a
    future separately trained encoder. They are not read from the archive here.
    """
    slots: torch.Tensor  # [1, identity_slots, width]
    valid: torch.Tensor  # [1, identity_slots] bool
    producer: str
    context_digest: str
    subject_id: str
    branch_id: str
    mode: str
    evidence_ids: tuple[str, ...] = ()
    _tensor_stamps: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if (self.slots.ndim != 3 or self.slots.shape[0] != 1
                or self.slots.shape[1] < 0 or self.slots.shape[2] < 1
                or not self.slots.is_floating_point()
                or not bool(torch.isfinite(self.slots).all())):
            raise ValueError('identity slots must be finite [1, slots, width]')
        if (self.valid.dtype != torch.bool
                or tuple(self.valid.shape) != tuple(self.slots.shape[:2])
                or self.valid.device != self.slots.device):
            raise ValueError('identity validity must be a matching boolean mask')
        for name in ('producer', 'context_digest', 'subject_id', 'branch_id', 'mode'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f'{name} required')
        if len(self.context_digest) != 64:
            raise ValueError('identity context digest must be SHA256')
        if not isinstance(self.evidence_ids, tuple) or len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError('identity evidence IDs must be distinct')
        object.__setattr__(self, '_tensor_stamps',
                           (tensor_stamp(self.slots), tensor_stamp(self.valid)))

    def tensors(self):
        return (self.slots, self.valid)

    def assert_immutable(self):
        if self._tensor_stamps != (tensor_stamp(self.slots), tensor_stamp(self.valid)):
            raise RuntimeError('IdentityRead tensors mutated; rebuild at a decision boundary')


@dataclass(frozen=True)
class CopyRead:
    """Exact source token IDs supplied to the optional copy head."""
    token_ids: torch.Tensor  # [1, candidates] int64
    mask: torch.Tensor  # [1, candidates] bool
    producer: str
    source_digest: str
    _tensor_stamps: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if (self.token_ids.ndim != 2 or self.token_ids.shape[0] != 1
                or self.token_ids.dtype != torch.long
                or tuple(self.mask.shape) != tuple(self.token_ids.shape)
                or self.mask.dtype != torch.bool):
            raise ValueError('copy tensors must be [1, candidates] int64/bool')
        if not isinstance(self.producer, str) or not self.producer:
            raise ValueError('copy producer identity required')
        if not isinstance(self.source_digest, str) or not self.source_digest:
            raise ValueError('copy source digest required')
        object.__setattr__(self, '_tensor_stamps',
                           (tensor_stamp(self.token_ids), tensor_stamp(self.mask)))

    def tensors(self):
        return (self.token_ids, self.mask)

    def assert_immutable(self):
        if self._tensor_stamps != (tensor_stamp(self.token_ids), tensor_stamp(self.mask)):
            raise RuntimeError('CopyRead tensors mutated; rebuild at a decision boundary')


@dataclass(frozen=True)
class FastMemoryRead:
    """One immutable committed fast-state snapshot supplied to a decision."""
    weights: torch.Tensor  # [1, heads, key, value]
    producer: str
    state_digest: str
    _tensor_stamps: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if (self.weights.ndim != 4 or self.weights.shape[0] != 1
                or not self.weights.is_floating_point()
                or not bool(torch.isfinite(self.weights).all())):
            raise ValueError('fast weights must be finite [1, heads, key, value]')
        if not isinstance(self.producer, str) or not self.producer:
            raise ValueError('fast-memory producer identity required')
        if not isinstance(self.state_digest, str) or not self.state_digest:
            raise ValueError('fast-memory state digest required')
        object.__setattr__(self, '_tensor_stamps', (tensor_stamp(self.weights),))

    def tensors(self):
        return (self.weights,)

    def assert_immutable(self):
        if self._tensor_stamps != (tensor_stamp(self.weights),):
            raise RuntimeError('FastMemoryRead tensor mutated; rebuild at a decision boundary')


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
    """Request-level projection of the authoritative exact task journal.

    The runtime ``ExactTaskState`` owns goal authority, claims, obligations and
    outcomes. This projection is what a decision frame may carry; ``state_digest``
    binds the frame to one journal revision.
    """
    task_id: str
    goal: str
    revision: int = 0
    constraints: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    open_obligations: tuple[str, ...] = ()
    success_conditions: tuple[str, ...] = ()
    state_digest: str = ''
    budget_remaining: int | None = None

    @property
    def digest(self):
        return fingerprint(asdict(self))


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
    core: CoreSpec | None = None
    fast_memory: FastMemorySpec | None = None
    copy: CopySpec | None = None
    identity: IdentitySpec | None = None

    def shared_dict(self):
        """Manifest fields that identify the parent, excluding additive state."""
        return {key: value for key, value in self.to_dict().items()
                if key not in ('core', 'fast_memory', 'copy', 'identity')}

    def to_dict(self):
        # A core-less manifest must keep the exact E1/E2 shape and digest so the
        # selected parent extension remains byte-compatible with its receipt.
        value = asdict(self)
        for name in ('core', 'fast_memory', 'copy', 'identity'):
            if value.get(name) is None:
                value.pop(name, None)
        return value

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
    core: CoreState | None = None
    fast: FastMemoryRead | None = None
    copy: CopyRead | None = None
    enabled_branches: tuple[str, ...] = ('memory',)
    _tensor_stamps: tuple = field(init=False, repr=False, compare=False)

    def tensors(self):
        owned = (self.memory.slots, self.memory.valid, self.sense.prompt_ids,
                 self.sense.prompt_mask, *(x for _, x in self.sense.summaries))
        if self.core is not None:
            owned = (*owned, *self.core.tensors())
        if self.fast is not None:
            owned = (*owned, *self.fast.tensors())
        if self.copy is not None:
            owned = (*owned, *self.copy.tensors())
        return owned

    def __post_init__(self):
        if (not self.enabled_branches
                or len(set(self.enabled_branches)) != len(self.enabled_branches)
                or not set(self.enabled_branches) <= {'memory', 'workspace', 'fast', 'copy'}):
            raise ValueError('Enabled branches must be a nonempty distinct subset of memory/workspace/fast/copy')
        if 'workspace' in self.enabled_branches and self.core is None:
            raise ValueError('Workspace branch requires an owned CoreState')
        if 'fast' in self.enabled_branches and self.fast is None:
            raise ValueError('Fast branch requires an owned FastMemoryRead')
        if 'copy' in self.enabled_branches and self.copy is None:
            raise ValueError('Copy branch requires an owned CopyRead')
        if self.core is not None:
            self.core.assert_immutable()
        if self.fast is not None:
            self.fast.assert_immutable()
        if self.copy is not None:
            self.copy.assert_immutable()
        object.__setattr__(self, '_tensor_stamps', tuple(tensor_stamp(t) for t in self.tensors()))

    def branch_enabled(self, name):
        return name in self.enabled_branches

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
