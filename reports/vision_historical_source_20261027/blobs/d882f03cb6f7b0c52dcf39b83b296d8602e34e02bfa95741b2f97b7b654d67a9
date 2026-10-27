"""Explicit E1/E2 ownership and provenance; no model or storage side effects."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import hashlib
import json
from typing import Any, Mapping
from types import MappingProxyType
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
    depth_mode: str = 'shared'
    depth_writers: bool = False
    def __post_init__(self):
        for name in ('heads', 'key_size', 'value_size'):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 1024:
                raise ValueError(f'FastMemorySpec.{name} must be in 1..1024')
        if self.depth_mode not in {'shared', 'private'}:
            raise ValueError('FastMemorySpec.depth_mode must be shared or private')
        if self.update_rule not in {'linear', 'sgdn2'}:
            raise ValueError('FastMemorySpec.update_rule must be linear or sgdn2')
        if type(self.depth_writers) is not bool:
            raise ValueError('FastMemorySpec.depth_writers must be boolean')
        if self.depth_writers and self.update_rule != 'sgdn2':
            raise ValueError('Per-depth writers require SGDN-2')

    def to_dict(self):
        value = asdict(self)
        if self.depth_mode == 'shared' and not self.depth_writers:
            value.pop('depth_mode')
            value.pop('depth_writers')
        elif not self.depth_writers:
            value.pop('depth_writers')
        return value


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
    private_weights: Mapping[int, torch.Tensor] | None = None
    encoding_identity: str | None = None
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
        if self.encoding_identity is not None and (
                not isinstance(self.encoding_identity, str) or len(self.encoding_identity) != 64
                or any(char not in '0123456789abcdef' for char in self.encoding_identity)):
            raise ValueError('fast-memory encoding identity must be a SHA256')
        if self.private_weights is not None:
            if not isinstance(self.private_weights, Mapping) or not self.private_weights:
                raise ValueError('private fast weights must be a nonempty layer mapping')
            private = dict(self.private_weights)
            object.__setattr__(self, 'private_weights', MappingProxyType(private))
            for layer, weights in private.items():
                if type(layer) is not int or layer < 0:
                    raise ValueError('private fast port must be a nonnegative integer')
                if (weights.shape != self.weights.shape or weights.device != self.weights.device
                        or weights.dtype != self.weights.dtype
                        or not weights.is_floating_point()
                        or not bool(torch.isfinite(weights).all())):
                    raise ValueError('private fast weights must match finite [1, heads, key, value] state')
            if not any(self.weights is weights for weights in private.values()):
                raise ValueError('private fast weights must alias one owned layer state')
        private = tuple(sorted((layer, tensor_stamp(weights))
                               for layer, weights in (self.private_weights or {}).items()))
        object.__setattr__(self, '_tensor_stamps', (tensor_stamp(self.weights), private))

    @property
    def cache_key(self):
        """Never merge states produced by different persistent encodings."""
        return fingerprint({'producer': self.producer, 'encoding': self.encoding_identity,
                            'state': self.state_digest})

    def for_layer(self, layer: int) -> torch.Tensor:
        if self.private_weights is not None:
            if layer not in self.private_weights:
                raise ValueError(f'private fast state missing port {layer}')
            return self.private_weights[layer]
        return self.weights
    def tensors(self):
        if self.private_weights is None:
            return (self.weights,)
        return tuple(weights for _, weights in sorted(self.private_weights.items()))

    def assert_immutable(self):
        private = tuple(sorted((layer, tensor_stamp(weights))
                               for layer, weights in (self.private_weights or {}).items()))
        if self._tensor_stamps != (tensor_stamp(self.weights), private):
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
class VisionSpec:
    """One pinned frozen patch encoder and a new model-specific projector."""
    encoder_repo: str
    encoder_revision: str
    encoder_source_digest: str
    processor_digest: str
    encoder_width: int = 768
    image_size: int = 224
    patch_size: int = 16
    max_visual_tokens: int = 196
    projector_hidden: int = 2048
    feature_version: str = 'siglip_patch_last_hidden.v1'
    compute_dtype: str = 'float32'

    def validate(self):
        if not isinstance(self.encoder_repo, str) or not self.encoder_repo or '/' not in self.encoder_repo:
            raise ValueError('Pinned vision encoder repository required')
        for name, length in (('encoder_revision', 40), ('encoder_source_digest', 64),
                             ('processor_digest', 64)):
            value = getattr(self, name)
            if (not isinstance(value, str) or len(value) != length
                    or any(char not in '0123456789abcdef' for char in value)):
                raise ValueError(f'VisionSpec.{name} requires an immutable digest')
        for name in ('encoder_width', 'image_size', 'patch_size', 'max_visual_tokens', 'projector_hidden'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'VisionSpec.{name} must be a positive integer')
        if (self.image_size % self.patch_size or self.max_visual_tokens > 1024
                or (self.image_size // self.patch_size) ** 2 > self.max_visual_tokens):
            raise ValueError('Bounded fixed spatial patch layout required')
        if self.feature_version != 'siglip_patch_last_hidden.v1' or self.compute_dtype != 'float32':
            raise ValueError('Unsupported visual feature or computation version')

    @property
    def digest(self):
        return fingerprint(asdict(self))


@dataclass(frozen=True)
class VisualInput:
    """Observed, detached spatial patches owned by one task and source scope."""
    features: torch.Tensor
    valid: torch.Tensor
    producer: str
    encoder_producer: str
    image_sha256: str
    source_id: str
    task_id: str
    subject_id: str
    branch_id: str
    purpose: str
    observed_at: int
    source_epoch: int
    original_size: tuple[int, int]
    crop: tuple[int, int, int, int]
    grid: tuple[int, int]
    _tensor_stamps: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        for name in ('original_size', 'crop', 'grid'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (self.features.ndim != 3 or self.features.shape[0] != 1 or self.features.shape[1] < 1
                or not self.features.is_floating_point() or self.features.requires_grad
                or not bool(torch.isfinite(self.features).all())):
            raise ValueError('Visual features must be detached finite [1, patches, width]')
        if (self.valid.shape != self.features.shape[:2] or self.valid.dtype != torch.bool
                or self.valid.device != self.features.device or not bool(self.valid.any())):
            raise ValueError('Visual patch mask must be matching nonempty boolean state')
        for name in ('producer', 'encoder_producer', 'image_sha256'):
            value = getattr(self, name)
            if (not isinstance(value, str) or len(value) != 64
                    or any(char not in '0123456789abcdef' for char in value)):
                raise ValueError(f'VisualInput.{name} must be a SHA256')
        for name in ('source_id', 'task_id', 'subject_id', 'branch_id', 'purpose'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f'VisualInput.{name} is required')
        for name in ('observed_at', 'source_epoch'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'VisualInput.{name} must be a nonnegative integer')
        if (len(self.original_size) != 2 or any(type(value) is not int or value < 1 for value in self.original_size)
                or len(self.crop) != 4 or any(type(value) is not int for value in self.crop)
                or not 0 <= self.crop[0] < self.crop[2] <= self.original_size[0]
                or not 0 <= self.crop[1] < self.crop[3] <= self.original_size[1]
                or len(self.grid) != 2 or any(type(value) is not int or value < 1 for value in self.grid)
                or self.grid[0] * self.grid[1] != self.features.shape[1]):
            raise ValueError('Visual crop, dimensions or spatial patch layout differs')
        object.__setattr__(self, '_tensor_stamps', tuple(tensor_stamp(value) for value in self.tensors()))

    def tensors(self):
        return self.features, self.valid

    @property
    def cache_key(self):
        return fingerprint({name: getattr(self, name) for name in (
            'producer', 'encoder_producer', 'image_sha256', 'source_id', 'task_id',
            'subject_id', 'branch_id', 'purpose', 'observed_at', 'source_epoch',
            'original_size', 'crop', 'grid')})

    def assert_immutable(self):
        if self.features.requires_grad or self._tensor_stamps != tuple(tensor_stamp(value) for value in self.tensors()):
            raise RuntimeError('VisualInput changed; build a new pinned decision')


@dataclass(frozen=True)
class SenseFrame:
    summaries: tuple[tuple[int, torch.Tensor], ...]
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    producer: str
    visual: VisualInput | None = None


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
class PatternSpec:
    """Static causal n-gram table geometry; versioned separately from episodes."""
    injection_layer: int = 2
    orders: tuple[int, ...] = (2, 3)
    heads: int = 4
    table_rows: int = 65536
    width: int = 64
    sparse: bool = True

    def __post_init__(self):
        if type(self.injection_layer) is not int or self.injection_layer < 0:
            raise ValueError('Pattern injection layer must be a nonnegative integer')
        if (not self.orders or len(set(self.orders)) != len(self.orders)
                or any(type(order) is not int or order not in {2, 3} for order in self.orders)):
            raise ValueError('Pattern orders must be distinct causal 2/3-grams')
        if type(self.heads) is not int or not 1 <= self.heads <= 16:
            raise ValueError('Pattern heads must be in 1..16')
        if type(self.table_rows) is not int or not 1 <= self.table_rows <= 2**20:
            raise ValueError('Pattern table rows must be in 1..2^20')
        if type(self.width) is not int or not 1 <= self.width <= 256:
            raise ValueError('Pattern width must be in 1..256')
        if type(self.sparse) is not bool:
            raise ValueError('Pattern sparse flag must be boolean')


@dataclass(frozen=True)
class PatternRead:
    """One immutable pattern-table version; hashes are recomputed causally per pass."""
    producer: str
    version: str

    def __post_init__(self):
        if not isinstance(self.producer, str) or not self.producer:
            raise ValueError('pattern producer identity required')
        if not isinstance(self.version, str) or not self.version:
            raise ValueError('pattern table version required')


@dataclass(frozen=True)
class PlasticSpec:
    """Declared selected-expert plastic bank plus its fixed read projection.

    The bank geometry is metadata for the read path; bank weights, moments,
    transactions and observation state are owned by
    :mod:`replica_lab.nested_learning`, never by the native adapter.
    """
    experts: int = 16
    width: int = 256
    hidden_width: int = 1024
    sites: tuple[int, ...] = (13, 27)
    route_mode: str = 'fixed'
    route_seed: int = 0
    projection_seed: int = 0
    read_gain: float = 0.1
    feature_version: str = 'sites_masked_mean_projection.v1'
    p_init: str = 'zero_read_down'

    def __post_init__(self):
        object.__setattr__(self, 'sites', tuple(self.sites))
        for name in ('experts', 'width', 'hidden_width'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'PlasticSpec.{name} must be a positive integer')
        if (not self.sites or len(set(self.sites)) != len(self.sites)
                or any(type(site) is not int or site < 0 for site in self.sites)):
            raise ValueError('PlasticSpec.sites must be distinct nonnegative integers')
        if self.route_mode not in {'fixed', 'shuffled'}:
            raise ValueError("PlasticSpec.route_mode must be 'fixed' or 'shuffled'")
        if type(self.route_seed) is not int:
            raise ValueError('PlasticSpec.route_seed must be an integer')
        if type(self.projection_seed) is not int:
            raise ValueError('PlasticSpec.projection_seed must be an integer')
        if (not isinstance(self.read_gain, (int, float))
                or not torch.isfinite(torch.tensor(float(self.read_gain)))
                or self.read_gain < 0):
            raise ValueError('PlasticSpec.read_gain must be finite and nonnegative')
        if self.feature_version != 'sites_masked_mean_projection.v1':
            raise ValueError('Unsupported plastic feature projection version')
        if self.p_init != 'zero_read_down':
            raise ValueError("PlasticSpec.p_init must be 'zero_read_down'")

    @property
    def digest(self):
        return fingerprint(asdict(self))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PlasticRead:
    """One immutable selected-expert read value handed to a decision frame.

    ``slot`` is the expert output in the declared plastic width.  The slot is
    detached and carries no source, label or task identity.  The decoder and
    gain are fixed by the model manifest; ``read_path_sha256`` proves the
    adapter is using exactly that fixed map.
    """
    slot: torch.Tensor  # [1, width]
    expert_id: int
    producer: str
    revision: int
    route_mode: str
    route_permutation: tuple[int, ...]
    read_path_sha256: str
    _tensor_stamps: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if (self.slot.ndim != 2 or self.slot.shape[0] != 1
                or not self.slot.is_floating_point() or self.slot.requires_grad
                or not bool(torch.isfinite(self.slot).all())):
            raise ValueError('Plastic slot must be detached finite [1, width]')
        if type(self.expert_id) is not int or self.expert_id < 0:
            raise ValueError('Plastic expert_id must be a nonnegative integer')
        if not isinstance(self.producer, str) or not self.producer:
            raise ValueError('Plastic producer identity required')
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError('Plastic revision must be a nonnegative integer')
        if self.route_mode not in {'fixed', 'shuffled'}:
            raise ValueError('Plastic route mode must be fixed or shuffled')
        object.__setattr__(self, 'route_permutation', tuple(self.route_permutation))
        permutation = self.route_permutation
        if (not permutation or len(set(permutation)) != len(permutation)
                or any(type(item) is not int or item < 0 for item in permutation)):
            raise ValueError('Plastic route permutation must be distinct nonnegative integers')
        if (not isinstance(self.read_path_sha256, str)
                or len(self.read_path_sha256) != 64
                or any(char not in '0123456789abcdef' for char in self.read_path_sha256)):
            raise ValueError('Plastic read path identity must be a SHA256')
        object.__setattr__(self, '_tensor_stamps', (tensor_stamp(self.slot),))

    def tensors(self):
        return (self.slot,)

    def assert_immutable(self):
        if self._tensor_stamps != (tensor_stamp(self.slot),):
            raise RuntimeError('PlasticRead tensor mutated; rebuild at a decision boundary')


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
    pattern: PatternSpec | None = None
    vision: VisionSpec | None = None
    plastic: PlasticSpec | None = None

    def shared_dict(self):
        """Manifest fields that identify the parent, excluding additive state."""
        return {key: value for key, value in self.to_dict().items()
                if key not in ('core', 'fast_memory', 'copy', 'identity', 'pattern', 'vision', 'plastic')}

    def to_dict(self):
        # Absent additive components preserve the exact parent manifest digest.
        value = asdict(self)
        for name in ('core', 'fast_memory', 'copy', 'identity', 'pattern', 'vision', 'plastic'):
            if value.get(name) is None:
                value.pop(name, None)
        if self.fast_memory is not None:
            value['fast_memory'] = self.fast_memory.to_dict()
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
    pattern: PatternRead | None = None
    enabled_branches: tuple[str, ...] = ('memory',)
    plastic: PlasticRead | None = None
    _tensor_stamps: tuple = field(init=False, repr=False, compare=False)

    def tensors(self):
        owned = (self.memory.slots, self.memory.valid, self.sense.prompt_ids,
                 self.sense.prompt_mask, *(x for _, x in self.sense.summaries))
        if self.core is not None:
            owned = (*owned, *self.core.tensors())
        if self.sense.visual is not None:
            owned = (*owned, *self.sense.visual.tensors())
        if self.fast is not None:
            owned = (*owned, *self.fast.tensors())
        if self.copy is not None:
            owned = (*owned, *self.copy.tensors())
        if self.plastic is not None:
            owned = (*owned, *self.plastic.tensors())
        return owned

    def __post_init__(self):
        if (not self.enabled_branches
                or len(set(self.enabled_branches)) != len(self.enabled_branches)
                or not set(self.enabled_branches) <= {'memory', 'workspace', 'fast', 'copy', 'pattern', 'plastic'}):
            raise ValueError('Enabled branches must be a nonempty distinct subset of memory/workspace/fast/copy/pattern/plastic')
        if 'workspace' in self.enabled_branches and self.core is None:
            raise ValueError('Workspace branch requires an owned CoreState')
        if 'fast' in self.enabled_branches and self.fast is None:
            raise ValueError('Fast branch requires an owned FastMemoryRead')
        if 'copy' in self.enabled_branches and self.copy is None:
            raise ValueError('Copy branch requires an owned CopyRead')
        if 'pattern' in self.enabled_branches and self.pattern is None:
            raise ValueError('Pattern branch requires an owned PatternRead')
        if 'plastic' in self.enabled_branches and self.plastic is None:
            raise ValueError('Plastic branch requires an owned PlasticRead')
        if self.plastic is not None:
            self.plastic.assert_immutable()
        if self.core is not None:
            self.core.assert_immutable()
        if self.fast is not None:
            self.fast.assert_immutable()
        if self.sense.visual is not None:
            self.sense.visual.assert_immutable()
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
