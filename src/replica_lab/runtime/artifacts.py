"""Typed runtime-compilation artifacts with distinct lifecycles (E4/E5 boundary).

Artifacts are metadata records plus external payload references. They never
execute tools, load a model or self-certify quality. Context packs, record
latents and branch images are host-compiled views; procedure and neural
candidates remain proposals until an independent evaluator accepts them. Source
or parent invalidation propagates to dependent artifacts; this is not neural
unlearning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

KINDS = ('context_pack', 'record_latent', 'branch_memory_image',
         'procedure', 'neural_candidate')
VIEW_STATUS = ('built', 'active', 'invalidated', 'expired')
CANDIDATE_STATUS = ('proposed', 'validated', 'accepted', 'rejected', 'deferred', 'revoked')
STATUS = VIEW_STATUS + CANDIDATE_STATUS
TRANSITIONS = {
    'context_pack': {'built': {'active', 'invalidated', 'expired'},
                     'active': {'invalidated', 'expired'},
                     'invalidated': set(), 'expired': set()},
    'record_latent': {'built': {'active', 'invalidated', 'expired'},
                      'active': {'invalidated', 'expired'},
                      'invalidated': set(), 'expired': set()},
    'branch_memory_image': {'built': {'active', 'invalidated', 'expired'},
                            'active': {'invalidated', 'expired'},
                            'invalidated': set(), 'expired': set()},
    'procedure': {'proposed': {'validated', 'rejected', 'deferred'},
                  'validated': {'accepted', 'rejected', 'deferred'},
                  'deferred': {'validated', 'rejected'},
                  'accepted': {'revoked'}, 'rejected': set(), 'revoked': set()},
    'neural_candidate': {'proposed': {'validated', 'rejected', 'deferred'},
                         'validated': {'accepted', 'rejected', 'deferred'},
                         'deferred': {'validated', 'rejected'},
                         'accepted': {'revoked'}, 'rejected': set(), 'revoked': set()},
}
MAX_TEXT = 64 * 1024


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip() or any(c.isspace() for c in value):
        raise ValueError(f'{name} must be an identifier')
    return value


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(
            c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{name} must be a lowercase SHA256')


def _producers(value):
    if not isinstance(value, Mapping) or not value:
        raise ValueError('producer versions are required')
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()):
        raise ValueError('producer versions must map strings to strings')
    return MappingProxyType(dict(value))


def _sources(value):
    if not isinstance(value, tuple) or not value:
        raise ValueError('at least one source/evidence ID is required')
    if len(set(value)) != len(value):
        raise ValueError('source IDs must be distinct')
    for item in value:
        _identifier(item, 'source ID')
    return value


@dataclass(frozen=True)
class ContextPack:
    artifact_id: str
    source_ids: tuple[str, ...]
    subject_id: str
    branch_id: str
    purpose: str
    cutoff: int
    text: str
    text_sha256: str
    producer_versions: Mapping = field(default_factory=dict)
    parent_id: str | None = None
    status: str = 'built'
    created_at: int = 0
    expires_at: int | None = None
    kind: str = 'context_pack'

    def __post_init__(self):
        _identifier(self.artifact_id, 'artifact_id')
        object.__setattr__(self, 'source_ids', _sources(self.source_ids))
        for name in ('subject_id', 'branch_id', 'purpose'):
            _identifier(getattr(self, name), name)
        if type(self.cutoff) is not int or self.cutoff < 0:
            raise ValueError('cutoff must be nonnegative')
        if not isinstance(self.text, str) or not self.text or len(self.text.encode()) > MAX_TEXT:
            raise ValueError('context pack text exceeds bounds')
        _sha(self.text_sha256, 'text_sha256')
        if hashlib.sha256(self.text.encode()).hexdigest() != self.text_sha256:
            raise ValueError('context pack text hash differs')
        object.__setattr__(self, 'producer_versions', _producers(self.producer_versions))
        if self.status not in VIEW_STATUS:
            raise ValueError('unknown context-pack status')

    @property
    def digest(self):
        return fingerprint({'kind': self.kind, 'artifact_id': self.artifact_id,
                            'source_ids': list(self.source_ids),
                            'subject_id': self.subject_id, 'branch_id': self.branch_id,
                            'purpose': self.purpose, 'cutoff': self.cutoff,
                            'text_sha256': self.text_sha256,
                            'producer_versions': dict(self.producer_versions),
                            'parent_id': self.parent_id, 'status': self.status,
                            'created_at': self.created_at, 'expires_at': self.expires_at})


@dataclass(frozen=True)
class RecordLatent:
    artifact_id: str
    record_id: str
    source_hash: str
    span_hash: str
    slots: int
    width: int
    payload_sha256: str
    writer_id: str
    producer_versions: Mapping = field(default_factory=dict)
    source_ids: tuple[str, ...] = ()
    parent_id: str | None = None
    status: str = 'built'
    created_at: int = 0
    expires_at: int | None = None
    kind: str = 'record_latent'

    def __post_init__(self):
        _identifier(self.artifact_id, 'artifact_id')
        for name in ('record_id', 'writer_id'):
            _identifier(getattr(self, name), name)
        _sha(self.source_hash, 'source_hash')
        _sha(self.span_hash, 'span_hash')
        _sha(self.payload_sha256, 'payload_sha256')
        for name in ('slots', 'width'):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 1024:
                raise ValueError(f'{name} must be in 1..1024')
        object.__setattr__(self, 'producer_versions', _producers(self.producer_versions))
        object.__setattr__(self, 'source_ids', _sources(self.source_ids or
                                                       (self.record_id,)))
        if self.status not in VIEW_STATUS:
            raise ValueError('unknown record-latent status')

    @property
    def digest(self):
        return fingerprint({'kind': self.kind, 'artifact_id': self.artifact_id,
                            'record_id': self.record_id, 'source_hash': self.source_hash,
                            'span_hash': self.span_hash, 'slots': self.slots,
                            'width': self.width, 'payload_sha256': self.payload_sha256,
                            'writer_id': self.writer_id,
                            'producer_versions': dict(self.producer_versions),
                            'source_ids': list(self.source_ids),
                            'parent_id': self.parent_id, 'status': self.status,
                            'created_at': self.created_at, 'expires_at': self.expires_at})


@dataclass(frozen=True)
class BranchMemoryImage:
    artifact_id: str
    branch_id: str
    subject_id: str
    mode: str
    fast_state_digest: str
    producer: str
    chunk_index: int
    cursor: int
    generation: int
    lineage: tuple[tuple[str, str], ...]
    payload_sha256: str
    source_ids: tuple[str, ...]
    producer_versions: Mapping = field(default_factory=dict)
    parent_id: str | None = None
    status: str = 'built'
    created_at: int = 0
    expires_at: int | None = None
    kind: str = 'branch_memory_image'

    def __post_init__(self):
        _identifier(self.artifact_id, 'artifact_id')
        for name in ('branch_id', 'subject_id', 'mode', 'producer'):
            _identifier(getattr(self, name), name)
        _sha(self.fast_state_digest, 'fast_state_digest')
        _sha(self.payload_sha256, 'payload_sha256')
        for name in ('chunk_index', 'cursor', 'generation'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative')
        if not isinstance(self.lineage, tuple) or any(
                not (isinstance(item, tuple) and len(item) == 2
                     and all(isinstance(v, str) for v in item)) for item in self.lineage):
            raise ValueError('lineage must be (event_id, source_hash) tuples')
        object.__setattr__(self, 'source_ids', _sources(self.source_ids))
        object.__setattr__(self, 'producer_versions', _producers(self.producer_versions))
        if self.status not in VIEW_STATUS:
            raise ValueError('unknown branch-image status')

    @property
    def digest(self):
        return fingerprint({'kind': self.kind, 'artifact_id': self.artifact_id,
                            'branch_id': self.branch_id, 'subject_id': self.subject_id,
                            'mode': self.mode, 'fast_state_digest': self.fast_state_digest,
                            'producer': self.producer, 'chunk_index': self.chunk_index,
                            'cursor': self.cursor, 'generation': self.generation,
                            'lineage': [list(item) for item in self.lineage],
                            'payload_sha256': self.payload_sha256,
                            'source_ids': list(self.source_ids),
                            'producer_versions': dict(self.producer_versions),
                            'parent_id': self.parent_id, 'status': self.status,
                            'created_at': self.created_at, 'expires_at': self.expires_at})


@dataclass(frozen=True)
class ProcedureStep:
    tool: str
    argument_schema: Mapping
    purpose: str = ''

    def __post_init__(self):
        _identifier(self.tool, 'tool')
        if not isinstance(self.argument_schema, Mapping):
            raise ValueError('procedure step schema must be a mapping')
        object.__setattr__(self, 'argument_schema',
                           MappingProxyType(dict(self.argument_schema)))
        if self.purpose and not isinstance(self.purpose, str):
            raise ValueError('step purpose must be text')

    def to_dict(self):
        return {'tool': self.tool, 'argument_schema': dict(self.argument_schema),
                'purpose': self.purpose}


@dataclass(frozen=True)
class ProcedureArtifact:
    artifact_id: str
    name: str
    version: int
    preconditions: tuple[str, ...]
    steps: tuple[ProcedureStep, ...]
    tests: tuple[str, ...]
    source_event_ids: tuple[str, ...]
    producer_versions: Mapping = field(default_factory=dict)
    parent_id: str | None = None
    status: str = 'proposed'
    created_at: int = 0
    expires_at: int | None = None
    kind: str = 'procedure'

    def __post_init__(self):
        _identifier(self.artifact_id, 'artifact_id')
        _identifier(self.name, 'name')
        if type(self.version) is not int or self.version < 1:
            raise ValueError('procedure version must be positive')
        for name in ('preconditions', 'tests', 'source_event_ids'):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(not isinstance(x, str) or not x for x in value):
                raise ValueError(f'{name} must be text')
        if not self.steps or any(not isinstance(step, ProcedureStep) for step in self.steps):
            raise ValueError('procedure requires typed steps')
        if not self.tests:
            raise ValueError('procedure requires at least one independent test')
        if not self.source_event_ids:
            raise ValueError('procedure requires source experience')
        object.__setattr__(self, 'producer_versions', _producers(self.producer_versions))
        if self.status not in CANDIDATE_STATUS:
            raise ValueError('unknown procedure status')

    @property
    def digest(self):
        return fingerprint({'kind': self.kind, 'artifact_id': self.artifact_id,
                            'name': self.name, 'version': self.version,
                            'preconditions': list(self.preconditions),
                            'steps': [step.to_dict() for step in self.steps],
                            'tests': list(self.tests),
                            'source_event_ids': list(self.source_event_ids),
                            'producer_versions': dict(self.producer_versions),
                            'parent_id': self.parent_id, 'status': self.status,
                            'created_at': self.created_at, 'expires_at': self.expires_at})


@dataclass(frozen=True)
class NeuralCandidate:
    artifact_id: str
    parent_manifest_digest: str
    changed_parameters: tuple[str, ...]
    weights_sha256: str
    training_scope: str
    optimizer: str
    producer_versions: Mapping = field(default_factory=dict)
    source_ids: tuple[str, ...] = ()
    parent_id: str | None = None
    status: str = 'proposed'
    evaluation: str = 'unevaluated'
    created_at: int = 0
    expires_at: int | None = None
    kind: str = 'neural_candidate'

    def __post_init__(self):
        _identifier(self.artifact_id, 'artifact_id')
        _sha(self.parent_manifest_digest, 'parent_manifest_digest')
        _sha(self.weights_sha256, 'weights_sha256')
        for name in ('training_scope', 'optimizer'):
            _identifier(getattr(self, name), name)
        if not isinstance(self.changed_parameters, tuple) or not self.changed_parameters:
            raise ValueError('candidate must declare changed parameters')
        for name in self.changed_parameters:
            _identifier(name, 'changed parameter')
        object.__setattr__(self, 'source_ids', _sources(self.source_ids))
        object.__setattr__(self, 'producer_versions', _producers(self.producer_versions))
        if self.status not in CANDIDATE_STATUS:
            raise ValueError('unknown candidate status')
        if self.evaluation not in ('unevaluated', 'passed', 'failed', 'inconclusive'):
            raise ValueError('unknown evaluation state')

    @property
    def digest(self):
        return fingerprint({'kind': self.kind, 'artifact_id': self.artifact_id,
                            'parent_manifest_digest': self.parent_manifest_digest,
                            'changed_parameters': list(self.changed_parameters),
                            'weights_sha256': self.weights_sha256,
                            'training_scope': self.training_scope,
                            'optimizer': self.optimizer,
                            'producer_versions': dict(self.producer_versions),
                            'source_ids': list(self.source_ids),
                            'parent_id': self.parent_id, 'status': self.status,
                            'evaluation': self.evaluation,
                            'created_at': self.created_at, 'expires_at': self.expires_at})


ARTIFACT_TYPES = (ContextPack, RecordLatent, BranchMemoryImage, ProcedureArtifact,
                  NeuralCandidate)


class ArtifactRegistry:
    """Host catalog with status transitions and dependency invalidation."""
    def __init__(self):
        self._items = {}
        self._source_index = {}
        self._children = {}

    def add(self, artifact):
        if not isinstance(artifact, ARTIFACT_TYPES):
            raise ValueError('unknown artifact type')
        if artifact.artifact_id in self._items:
            raise ValueError('duplicate artifact ID')
        if artifact.parent_id is not None and artifact.parent_id not in self._items:
            raise ValueError('parent artifact is absent')
        self._items[artifact.artifact_id] = artifact
        sources = getattr(artifact, 'source_ids', None)
        if sources is None:
            sources = getattr(artifact, 'source_event_ids', ())
        for source in sources or ():
            self._source_index.setdefault(source, set()).add(artifact.artifact_id)
            self._children.setdefault(artifact.parent_id, set()).add(artifact.artifact_id)
        return artifact

    def get(self, artifact_id):
        return self._items.get(artifact_id)

    def list(self, kind=None, status=None):
        values = list(self._items.values())
        if kind is not None:
            values = [item for item in values if item.kind == kind]
        if status is not None:
            values = [item for item in values if item.status == status]
        return tuple(sorted(values, key=lambda item: item.artifact_id))

    def transition(self, artifact_id, new_status, *, reason='', now=0):
        item = self._items.get(artifact_id)
        if item is None:
            raise KeyError(artifact_id)
        if new_status not in STATUS:
            raise ValueError('unknown artifact status')
        if new_status not in TRANSITIONS[item.kind].get(item.status, set()):
            raise ValueError(f'illegal transition {item.kind}:{item.status}->{new_status}')
        if item.expires_at is not None and now > item.expires_at and new_status != 'expired':
            raise ValueError('artifact has expired')
        updated = type(item)(**{**{name: getattr(item, name)
                                  for name in item.__dataclass_fields__
                                  if name != 'status'}, 'status': new_status})
        self._items[artifact_id] = updated
        if new_status in ('invalidated', 'expired', 'revoked'):
            self._invalidate_children(artifact_id)
        return updated

    def _invalidate_children(self, artifact_id):
        for child_id in list(self._children.get(artifact_id, ())):
            child = self._items.get(child_id)
            if child is not None and child.status not in ('invalidated', 'expired',
                                                          'rejected', 'revoked'):
                target = self._invalid_target(child)
                if target in TRANSITIONS[child.kind].get(child.status, set()):
                    self.transition(child_id, target, reason='parent invalidated')

    @staticmethod
    def _invalid_target(item):
        if item.kind in ('procedure', 'neural_candidate'):
            return 'revoked' if item.status == 'accepted' else 'rejected'
        return 'invalidated'

    def invalidate_source(self, source_id, *, reason=''):
        _identifier(source_id, 'source_id')
        affected = []
        for artifact_id in list(self._source_index.get(source_id, ())):
            item = self._items.get(artifact_id)
            if item is None or item.status in ('invalidated', 'expired', 'revoked'):
                continue
            target = self._invalid_target(item)
            if target in TRANSITIONS[item.kind].get(item.status, set()):
                self.transition(artifact_id, target, reason=reason)
                affected.append(artifact_id)
        return tuple(affected)

    def payload(self, artifact_id):
        item = self._items.get(artifact_id)
        if item is None:
            raise KeyError(artifact_id)
        if item.status in ('invalidated', 'expired', 'revoked', 'rejected'):
            raise PermissionError('artifact is not usable')
        return item

    def save(self, path):
        value = {'format': 'replica.artifacts.catalog.v1',
                 'artifacts': [{'artifact_id': item.artifact_id,
                                'kind': item.kind, 'status': item.status,
                                'digest': item.digest}
                               for item in self.list()]}
        Path(path).write_text(json.dumps(value, indent=2) + '\n')
        return value
