"""Typed runtime-compilation artifacts with distinct lifecycles (E4/E5 boundary).

Artifacts are metadata records plus external payload references. They never
execute tools, load a model or self-certify quality. Context packs, record
latents and branch images are host-compiled views; procedure and neural
candidates remain proposals until an independent evaluator accepts them. Source
or parent invalidation propagates to dependent artifacts; this is not neural
unlearning.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
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
CATALOG_FORMAT = 'replica.artifacts.catalog.v2'
LEGACY_CATALOG_FORMAT = 'replica.artifacts.catalog.v1'


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def atomic_write_text(path: Any, text: str) -> None:
    """Durable single-file replacement: fsync the file, replace, fsync the dir."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='pending-', dir=target.parent)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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


def _jsonable(value):
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def artifact_fields(artifact) -> dict:
    """All stored fields of one artifact, status excluded, JSON-safe."""
    value = {}
    for name in artifact.__dataclass_fields__:
        if name == 'status':
            continue
        item = getattr(artifact, name)
        if name == 'steps':
            value[name] = [step.to_dict() for step in item]
        else:
            value[name] = _jsonable(item)
    return value


def artifact_content_digest(artifact) -> str:
    """Digest of the artifact payload that must not change across transitions."""
    return fingerprint({'kind': artifact.kind, **artifact_fields(artifact)})


def artifact_to_dict(artifact, *, content_digest: bool = False) -> dict:
    """Canonical record used by catalogs and v2 manifest payloads."""
    if not isinstance(artifact, ARTIFACT_TYPES):
        raise ValueError('unknown artifact type')
    value = {'kind': artifact.kind, **artifact_fields(artifact),
             'status': artifact.status, 'digest': artifact.digest}
    if content_digest:
        value['content_digest'] = artifact_content_digest(artifact)
    return value


_ARTIFACT_FIELDS = {
    'context_pack': (
        ('artifact_id', 'source_ids', 'subject_id', 'branch_id', 'purpose',
         'cutoff', 'text', 'text_sha256', 'producer_versions', 'parent_id',
         'status', 'created_at', 'expires_at'),
        ('source_ids',), ('producer_versions',)),
    'record_latent': (
        ('artifact_id', 'record_id', 'source_hash', 'span_hash', 'slots', 'width',
         'payload_sha256', 'writer_id', 'producer_versions', 'source_ids',
         'parent_id', 'status', 'created_at', 'expires_at'),
        ('source_ids',), ('producer_versions',)),
    'branch_memory_image': (
        ('artifact_id', 'branch_id', 'subject_id', 'mode', 'fast_state_digest',
         'producer', 'chunk_index', 'cursor', 'generation', 'lineage',
         'payload_sha256', 'source_ids', 'producer_versions', 'parent_id',
         'status', 'created_at', 'expires_at'),
        ('source_ids', 'lineage'), ('producer_versions',)),
    'procedure': (
        ('artifact_id', 'name', 'version', 'preconditions', 'steps', 'tests',
         'source_event_ids', 'producer_versions', 'parent_id', 'status',
         'created_at', 'expires_at'),
        ('preconditions', 'steps', 'tests', 'source_event_ids'),
        ('producer_versions',)),
    'neural_candidate': (
        ('artifact_id', 'parent_manifest_digest', 'changed_parameters',
         'weights_sha256', 'training_scope', 'optimizer', 'producer_versions',
         'source_ids', 'parent_id', 'status', 'evaluation', 'created_at',
         'expires_at'),
        ('changed_parameters', 'source_ids'), ('producer_versions',)),
}


_ARTIFACT_CLASSES = {item.kind: item for item in ARTIFACT_TYPES}


def artifact_from_dict(value, *, content_digest: bool = False, kind=None,
                       require_kind: bool = False):
    """Strictly reconstruct one artifact record and re-verify its digests."""
    if not isinstance(value, Mapping):
        raise TypeError('artifact record must be a mapping')
    if kind is not None and kind not in _ARTIFACT_FIELDS:
        raise ValueError('artifact record has no known kind')
    artifact_kind = kind if kind is not None else value.get('kind')
    if artifact_kind not in _ARTIFACT_FIELDS:
        raise ValueError('artifact record has no known kind')
    names, tuple_fields, mapping_fields = _ARTIFACT_FIELDS[artifact_kind]
    expected = set(names) | {'digest'}
    if kind is None or require_kind:
        expected.add('kind')
    if 'kind' in value:
        if value['kind'] != artifact_kind:
            raise ValueError('artifact record kind differs from the declared kind')
    elif require_kind:
        raise ValueError('artifact record kind is required')
    if content_digest:
        expected.add('content_digest')
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(
            f'artifact record schema mismatch; missing={missing} extra={extra}')
    _sha(value['digest'], 'artifact digest')
    fields = {}
    for name in names:
        item = value[name]
        if name in tuple_fields:
            if not isinstance(item, list):
                raise ValueError(f'{name} must be a serialized list')
            if name == 'steps':
                steps = []
                for step in item:
                    if not isinstance(step, Mapping) or not {
                            'tool', 'argument_schema'} <= set(step) or \
                            not set(step) <= {'tool', 'argument_schema', 'purpose'}:
                        raise ValueError('procedure step schema mismatch')
                    steps.append(ProcedureStep(**step))
                item = tuple(steps)
            elif name == 'lineage':
                item = tuple(tuple(pair) for pair in item)
            else:
                item = tuple(item)
        elif name in mapping_fields:
            if not isinstance(item, Mapping):
                raise ValueError(f'{name} must be a mapping')
            item = dict(item)
        fields[name] = item
    artifact = _ARTIFACT_CLASSES[artifact_kind](**fields)
    if artifact.digest != value['digest']:
        raise ValueError('artifact digest differs from its recorded payload')
    if content_digest and artifact_content_digest(artifact) != value['content_digest']:
        raise ValueError('artifact content digest differs from its payload')
    return artifact


def verify_catalog_summary(value) -> tuple[dict, ...]:
    """Validate a v1 summary catalog.  Payloads are absent by construction."""
    if not isinstance(value, Mapping):
        raise TypeError('artifact catalog must be a mapping')
    if set(value) != {'format', 'artifacts'}:
        raise ValueError('artifact catalog schema mismatch')
    if value['format'] != LEGACY_CATALOG_FORMAT:
        raise ValueError(f"unknown summary catalog format {value['format']!r}")
    if not isinstance(value['artifacts'], list):
        raise ValueError('artifact catalog entries must be a list')
    seen = set()
    records = []
    for record in value['artifacts']:
        if not isinstance(record, Mapping) or set(record) != {
                'artifact_id', 'kind', 'status', 'digest'}:
            raise ValueError('summary catalog record schema mismatch')
        _identifier(record['artifact_id'], 'artifact_id')
        if record['artifact_id'] in seen:
            raise ValueError('duplicate artifact ID in summary catalog')
        seen.add(record['artifact_id'])
        if record['kind'] not in KINDS:
            raise ValueError('unknown artifact kind in summary catalog')
        allowed = VIEW_STATUS if record['kind'] in (
            'context_pack', 'record_latent', 'branch_memory_image') \
            else CANDIDATE_STATUS
        if record['status'] not in allowed:
            raise ValueError(
                f'impossible {record["kind"]} status in summary catalog: '
                f'{record["status"]!r}')
        _sha(record['digest'], 'artifact digest')
        records.append(dict(record))
    return tuple(records)


class ArtifactRegistry:
    """Host catalog with status transitions, publications and invalidation."""
    def __init__(self):
        self._items = {}
        self._source_index = {}
        self._children = {}
        self._publications = {}

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
        if new_status in ('revoked', 'rejected', 'invalidated', 'expired'):
            self._deactivate_publication(artifact_id, reason or new_status)
        if new_status in ('invalidated', 'expired', 'revoked'):
            self._invalidate_children(artifact_id)
        return updated

    # -- publication binding -------------------------------------------------
    def record_publication(self, artifact_id, manifest_id, payload_digest, *,
                           reason='published', active=True, imported=False):
        """Bind an accepted artifact to the manifest that carries it."""
        item = self._items.get(artifact_id)
        if item is None:
            raise KeyError(artifact_id)
        if item.kind != 'procedure':
            raise ValueError('only procedures publish through manifests')
        _sha(manifest_id, 'manifest_id')
        _sha(payload_digest, 'payload_digest')
        if type(active) is not bool or type(imported) is not bool:
            raise TypeError('publication flags must be boolean')
        content = artifact_content_digest(item)
        existing = self._publications.get(artifact_id)
        if existing is not None:
            same = (existing['manifest_id'] == manifest_id and
                    existing['payload_digest'] == payload_digest and
                    existing['artifact_content_digest'] == content and
                    existing['active'] == active)
            if not same:
                raise ValueError('artifact publication would be rewritten')
        record = {
            'artifact_id': artifact_id,
            'manifest_id': manifest_id,
            'payload_digest': payload_digest,
            'artifact_content_digest': content,
            'active': active,
            'imported': imported,
            'reason': reason,
        }
        self._publications[artifact_id] = record
        return dict(record)

    def publication(self, artifact_id):
        record = self._publications.get(artifact_id)
        return dict(record) if record is not None else None

    def restore_status(self, artifact_id, status):
        """Reconstruction-only status restoration for a manifest reload.

        The payload bytes (and therefore the content digest) are unchanged; this
        is not a lifecycle transition and is never used by publish/rollback.
        """
        item = self._items.get(artifact_id)
        if item is None:
            raise KeyError(artifact_id)
        if status not in CANDIDATE_STATUS:
            raise ValueError('reconstructed status must be a candidate status')
        if item.status == status:
            return item
        updated = replace(item, status=status)
        if artifact_content_digest(updated) != artifact_content_digest(item):
            raise ValueError('status restoration changed artifact content')
        self._items[artifact_id] = updated
        return updated

    def _deactivate_publication(self, artifact_id, reason):
        record = self._publications.get(artifact_id)
        if record is None or record['active'] is False:
            return
        self._publications[artifact_id] = {
            **record, 'active': False,
            'reason': f'{record["reason"]};{reason}'}

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

    def assert_consistent(self):
        """Reject catalog states that no transaction could have produced."""
        for item in self._items.values():
            record = self._publications.get(item.artifact_id)
            if record is not None:
                if record['artifact_content_digest'] != artifact_content_digest(item):
                    raise ValueError('artifact content changed under its publication')
                if record['active'] and item.status != 'accepted':
                    raise ValueError(
                        f'publication is active for non-accepted artifact {item.artifact_id}')
            if item.kind == 'procedure' and item.status == 'accepted':
                if record is None or not record['active']:
                    raise ValueError(
                        f'accepted artifact has no active publication: {item.artifact_id}')

    # -- durable catalog -----------------------------------------------------
    def export(self) -> dict:
        return {
            'format': CATALOG_FORMAT,
            'artifacts': [artifact_to_dict(item, content_digest=True)
                          for item in self.list()],
            'publications': [dict(self._publications[key])
                             for key in sorted(self._publications)],
        }

    def snapshot(self) -> dict:
        return self.export()

    def summary(self) -> dict:
        """Emit the historical v1 summary for readers that only audit state.

        The v1 contract carries no payloads, so it is a derived, validated view,
        never a substitute for the v2 catalog.
        """
        return {'format': LEGACY_CATALOG_FORMAT,
                'artifacts': [{'artifact_id': item.artifact_id,
                               'kind': item.kind, 'status': item.status,
                               'digest': item.digest}
                              for item in self.list()]}

    @classmethod
    def import_state(cls, value) -> 'ArtifactRegistry':
        if not isinstance(value, Mapping):
            raise TypeError('artifact catalog must be a mapping')
        if value.get('format') == LEGACY_CATALOG_FORMAT:
            raise ValueError(
                'catalog format replica.artifacts.catalog.v1 has no artifact '
                'payloads; validate it with verify_catalog_summary and migrate '
                'explicitly instead of loading it as a live registry')
        if set(value) != {'format', 'artifacts', 'publications'}:
            raise ValueError('artifact catalog schema mismatch')
        if value['format'] != CATALOG_FORMAT:
            raise ValueError(f"unknown artifact catalog format {value['format']!r}")
        if not isinstance(value['artifacts'], list) or \
                not isinstance(value['publications'], list):
            raise ValueError('artifact catalog entries must be lists')
        registry = cls()
        pending = list(value['artifacts'])
        while pending:
            progressed = False
            for record in list(pending):
                artifact = artifact_from_dict(record, content_digest=True)
                if artifact.parent_id is not None and \
                        artifact.parent_id not in registry._items:
                    continue
                registry.add(artifact)
                pending.remove(record)
                progressed = True
            if not progressed:
                raise ValueError('artifact catalog has missing or cyclic parents')
        seen_publications = set()
        seen_manifests = set()
        for record in value['publications']:
            if not isinstance(record, Mapping) or set(record) != {
                    'artifact_id', 'manifest_id', 'payload_digest',
                    'artifact_content_digest', 'active', 'imported', 'reason'}:
                raise ValueError('catalog publication schema mismatch')
            if record['artifact_id'] in seen_publications:
                raise ValueError('duplicate artifact publication record')
            if record['manifest_id'] in seen_manifests:
                raise ValueError('duplicate manifest publication record')
            seen_publications.add(record['artifact_id'])
            seen_manifests.add(record['manifest_id'])
            _sha(record['manifest_id'], 'manifest_id')
            _sha(record['payload_digest'], 'payload_digest')
            if type(record['active']) is not bool or type(record['imported']) is not bool:
                raise ValueError('catalog publication flags must be boolean')
            if not isinstance(record['reason'], str):
                raise ValueError('catalog publication reason must be text')
            artifact = registry._items.get(record['artifact_id'])
            if artifact is None:
                raise ValueError('catalog publication references an absent artifact')
            if record['artifact_content_digest'] != artifact_content_digest(artifact):
                raise ValueError('catalog publication content differs from artifact')
            registry.record_publication(
                record['artifact_id'], record['manifest_id'],
                record['payload_digest'], reason=record['reason'],
                active=record['active'], imported=record['imported'])
        registry.assert_consistent()
        return registry

    def restore(self, snapshot) -> None:
        restored = type(self).import_state(snapshot)
        self._items = restored._items
        self._source_index = restored._source_index
        self._children = restored._children
        self._publications = restored._publications

    def save(self, path):
        self.assert_consistent()
        value = self.export()
        atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + '\n')
        return value

    @classmethod
    def load(cls, path) -> 'ArtifactRegistry':
        target = Path(path)
        try:
            value = json.loads(target.read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f'artifact catalog is unreadable: {target}') from error
        return cls.import_state(value)

    @classmethod
    def load_legacy_catalog(cls, path) -> dict:
        """Explicit migration path for a v1 summary; no payloads are invented."""
        target = Path(path)
        try:
            value = json.loads(target.read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f'artifact catalog is unreadable: {target}') from error
        return {'format': LEGACY_CATALOG_FORMAT,
                'artifacts': list(verify_catalog_summary(value)),
                'reconstructable': False}

