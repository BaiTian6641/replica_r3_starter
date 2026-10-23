"""Versioned Replica identity contracts; no authentic personal data is bundled.

The source archive and a Replica branch are separate ownership domains. A
snapshot is a dated, host-approved selection of source episodes and principle
hypotheses. A branch may add its own attributed experiences, but it may not
rewrite source testimony, duplicate itself into the human archive, or convert a
preference into task-success authority.

These are contracts and a resolver, not a personalization result. Authentic
conditioning remains dependent on permitted source material and a separately
evaluated candidate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

MODES = frozenset({'source_snapshot', 'source_current', 'parallel_replica',
                   'assistant_for_source'})
USES = frozenset({'retrieval', 'local_training', 'evaluation', 'teacher', 'sharing'})
EPISODE_KINDS = frozenset({'reported_event', 'interpretation', 'preference',
                           'hypothetical', 'agent_experience'})
AUTHENTICITY = frozenset({'unverified', 'source_report', 'verified_source',
                          'observed_tool', 'agent_report'})
PRINCIPLE_STATUS = frozenset({'proposed', 'approved', 'revoked'})
MAX_TEXT = 4000
MAX_SUMMARY = 2400


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def _text(value, name, limit=MAX_TEXT):
    if not isinstance(value, str) or not value.strip() or len(value.encode()) > limit:
        raise ValueError(f'{name} must be bounded nonempty text')
    value.encode('utf-8')
    return value


def _identifier(value, name):
    _text(value, name, 128)
    if '\x00' in value or any(c.isspace() for c in value):
        raise ValueError(f'{name} must be an identifier without whitespace')
    return value


def _uses(value):
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError('permitted uses must be a nonempty sequence')
    result = tuple(value)
    if len(set(result)) != len(result) or not set(result) <= USES:
        raise ValueError('permitted uses must be distinct known uses')
    return result


@dataclass(frozen=True, slots=True)
class SourceEpisode:
    """One attributed source statement/decision, not a verified world fact."""
    episode_id: str
    subject_id: str
    source_id: str
    source_hash: str
    observed_at: int
    event_at: int
    text: str
    kind: str = 'reported_event'
    authenticity: str = 'unverified'
    permitted_uses: tuple[str, ...] = ('retrieval',)
    supersedes: str | None = None
    uncertainty: str = ''

    def __post_init__(self):
        _identifier(self.episode_id, 'episode_id')
        _identifier(self.subject_id, 'subject_id')
        _identifier(self.source_id, 'source_id')
        if not isinstance(self.source_hash, str) or len(self.source_hash) != 64 \
                or any(c not in '0123456789abcdef' for c in self.source_hash):
            raise ValueError('source_hash must be a lowercase SHA256')
        for name in ('observed_at', 'event_at'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        _text(self.text, 'text')
        if self.kind not in EPISODE_KINDS:
            raise ValueError('unknown episode kind')
        if self.authenticity not in AUTHENTICITY:
            raise ValueError('unknown authenticity label')
        object.__setattr__(self, 'permitted_uses', _uses(self.permitted_uses))
        if self.supersedes is not None:
            _identifier(self.supersedes, 'supersedes')
            if self.supersedes == self.episode_id:
                raise ValueError('an episode cannot supersede itself')
        if self.uncertainty:
            _text(self.uncertainty, 'uncertainty', 512)

    @property
    def digest(self):
        return fingerprint({name: getattr(self, name) for name in
                            ('episode_id', 'subject_id', 'source_id', 'source_hash',
                             'observed_at', 'event_at', 'text', 'kind',
                             'authenticity', 'permitted_uses', 'supersedes',
                             'uncertainty')})


@dataclass(frozen=True, slots=True)
class PrincipleHypothesis:
    """An inferred conditional priority; never an automatically verified trait."""
    principle_id: str
    subject_id: str
    statement: str
    condition: str = ''
    exceptions: tuple[str, ...] = ()
    supporting_episode_ids: tuple[str, ...] = ()
    contradicting_episode_ids: tuple[str, ...] = ()
    status: str = 'proposed'

    def __post_init__(self):
        _identifier(self.principle_id, 'principle_id')
        _identifier(self.subject_id, 'subject_id')
        _text(self.statement, 'statement', 512)
        if self.condition:
            _text(self.condition, 'condition', 512)
        if not isinstance(self.exceptions, tuple) or any(
                not isinstance(item, str) or not item for item in self.exceptions):
            raise ValueError('exceptions must be text')
        for name in ('supporting_episode_ids', 'contradicting_episode_ids'):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(not isinstance(x, str) for x in value):
                raise ValueError(f'{name} must be a tuple of IDs')
            if len(set(value)) != len(value):
                raise ValueError(f'{name} contains duplicates')
        if self.status not in PRINCIPLE_STATUS:
            raise ValueError('unknown principle status')
        if self.status == 'approved' and not self.supporting_episode_ids:
            raise ValueError('approved principle requires supporting evidence')

    @property
    def digest(self):
        return fingerprint({name: getattr(self, name) for name in
                            ('principle_id', 'subject_id', 'statement', 'condition',
                             'exceptions', 'supporting_episode_ids',
                             'contradicting_episode_ids', 'status')})


@dataclass(frozen=True, slots=True)
class IdentitySnapshot:
    """A dated approved view; changing it requires a new version."""
    snapshot_id: str
    subject_id: str
    version: int
    created_at: int
    source_episode_ids: tuple[str, ...]
    source_hashes: tuple[str, ...]
    principle_ids: tuple[str, ...]
    unresolved_questions: tuple[str, ...]
    permitted_uses: tuple[str, ...]
    schema_version: str = 'replica.identity.snapshot.v1'
    status: str = 'active'

    def __post_init__(self):
        _identifier(self.snapshot_id, 'snapshot_id')
        _identifier(self.subject_id, 'subject_id')
        if type(self.version) is not int or self.version < 1:
            raise ValueError('snapshot version must be positive')
        if type(self.created_at) is not int or self.created_at < 0:
            raise ValueError('created_at must be nonnegative')
        for name in ('source_episode_ids', 'source_hashes', 'principle_ids'):
            value = getattr(self, name)
            if not isinstance(value, tuple) or len(set(value)) != len(value):
                raise ValueError(f'{name} must be a distinct tuple')
        if len(self.source_episode_ids) != len(self.source_hashes):
            raise ValueError('one source hash per episode is required')
        if not isinstance(self.unresolved_questions, tuple) or any(
                not isinstance(x, str) or not x for x in self.unresolved_questions):
            raise ValueError('unresolved questions must be text')
        object.__setattr__(self, 'permitted_uses', _uses(self.permitted_uses))
        if self.schema_version != 'replica.identity.snapshot.v1':
            raise ValueError('unknown identity snapshot schema')
        if self.status not in {'active', 'revoked'}:
            raise ValueError('unknown snapshot status')

    @property
    def digest(self):
        return fingerprint({name: getattr(self, name) for name in
                            ('snapshot_id', 'subject_id', 'version', 'created_at',
                             'source_episode_ids', 'source_hashes', 'principle_ids',
                             'unresolved_questions', 'permitted_uses',
                             'schema_version', 'status')})


@dataclass(frozen=True, slots=True)
class IdentityManifest:
    subject_id: str
    snapshot_id: str
    snapshot_digest: str
    schema_version: str = 'replica.identity.context.v1'
    producer_versions: Mapping = field(default_factory=dict)

    def __post_init__(self):
        _identifier(self.subject_id, 'subject_id')
        _identifier(self.snapshot_id, 'snapshot_id')
        if len(self.snapshot_digest) != 64:
            raise ValueError('snapshot digest must be SHA256')
        object.__setattr__(self, 'producer_versions',
                           MappingProxyType(dict(self.producer_versions)))

    def compatible(self, other: 'IdentityManifest') -> bool:
        return (self.subject_id, self.snapshot_id, self.snapshot_digest,
                self.schema_version, dict(self.producer_versions)) == (
            other.subject_id, other.snapshot_id, other.snapshot_digest,
            other.schema_version, dict(other.producer_versions))


@dataclass(frozen=True, slots=True)
class IdentityContext:
    """Host-resolved, mode-bound conditioning; never task-success authority."""
    subject_id: str
    snapshot_id: str | None
    snapshot_digest: str | None
    mode: str | None
    branch_id: str
    purpose: str
    cutoff: int
    evidence_ids: tuple[str, ...]
    summary: str
    uncertainty: tuple[str, ...] = ()
    producer_versions: Mapping = field(default_factory=dict)
    schema_version: str = 'replica.identity.context.v1'
    is_null: bool = False

    def __post_init__(self):
        _identifier(self.subject_id, 'subject_id')
        _identifier(self.branch_id, 'branch_id')
        _identifier(self.purpose, 'purpose')
        if type(self.cutoff) is not int or self.cutoff < 0:
            raise ValueError('cutoff must be nonnegative')
        if self.mode is not None and self.mode not in MODES:
            raise ValueError('unknown identity mode')
        if self.is_null:
            if self.snapshot_id is not None or self.snapshot_digest is not None:
                raise ValueError('null context cannot carry a snapshot')
        else:
            if self.mode is None or self.snapshot_id is None or not self.snapshot_digest:
                raise ValueError('active context requires mode and snapshot')
            if len(self.snapshot_digest) != 64:
                raise ValueError('snapshot digest must be SHA256')
        if not isinstance(self.evidence_ids, tuple) or len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError('evidence IDs must be distinct')
        if len(self.summary.encode()) > MAX_SUMMARY:
            raise ValueError('identity summary exceeds its bounded size')
        if not isinstance(self.uncertainty, tuple) or any(
                not isinstance(x, str) or not x for x in self.uncertainty):
            raise ValueError('uncertainty must be text')
        object.__setattr__(self, 'producer_versions',
                           MappingProxyType(dict(self.producer_versions)))
        if self.schema_version != 'replica.identity.context.v1':
            raise ValueError('unknown identity context schema')

    @classmethod
    def null(cls, *, subject_id='fictional', branch_id='main',
             purpose='local', cutoff=0):
        return cls(subject_id, None, None, None, branch_id, purpose, cutoff, (),
                   '', (), {}, is_null=True)

    @property
    def digest(self):
        return fingerprint({name: getattr(self, name) for name in
                            ('subject_id', 'snapshot_id', 'snapshot_digest', 'mode',
                             'branch_id', 'purpose', 'cutoff', 'evidence_ids',
                             'summary', 'uncertainty', 'producer_versions',
                             'schema_version', 'is_null')})

    def manifest(self):
        if self.is_null:
            return None
        return IdentityManifest(self.subject_id, self.snapshot_id,
                                self.snapshot_digest, self.schema_version,
                                dict(self.producer_versions))


@dataclass(frozen=True, slots=True)
class BranchExperience:
    """Agent-owned event; cannot be converted into source-person testimony."""
    experience_id: str
    branch_id: str
    subject_id: str
    observed_at: int
    text: str
    outcome: str = 'unknown'
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self):
        _identifier(self.experience_id, 'experience_id')
        _identifier(self.branch_id, 'branch_id')
        _identifier(self.subject_id, 'subject_id')
        if type(self.observed_at) is not int or self.observed_at < 0:
            raise ValueError('observed_at must be nonnegative')
        _text(self.text, 'text')
        if self.outcome not in {'unknown', 'succeeded', 'failed', 'denied'}:
            raise ValueError('unknown outcome')
        if not isinstance(self.evidence_ids, tuple):
            raise ValueError('evidence IDs must be a tuple')


@dataclass(frozen=True, slots=True)
class ReplicaBranchState:
    branch_id: str
    subject_id: str
    mode: str
    source_snapshot_id: str
    experiences: tuple[BranchExperience, ...] = ()
    skills: tuple[str, ...] = ()
    revision: int = 0

    def __post_init__(self):
        _identifier(self.branch_id, 'branch_id')
        _identifier(self.subject_id, 'subject_id')
        if self.mode != 'parallel_replica':
            raise ValueError('branch state is for parallel_replica only')
        _identifier(self.source_snapshot_id, 'source_snapshot_id')
        if any(item.branch_id != self.branch_id or item.subject_id != self.subject_id
               for item in self.experiences):
            raise ValueError('branch experiences must belong to this branch')
        if len(set(item.experience_id for item in self.experiences)) != len(self.experiences):
            raise ValueError('duplicate branch experience')
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError('revision must be nonnegative')

    def add(self, experience: BranchExperience, *, skill=None):
        if experience.branch_id != self.branch_id or experience.subject_id != self.subject_id:
            raise PermissionError('experience belongs to another branch or subject')
        skills = self.skills if skill is None else (*self.skills, skill)
        return ReplicaBranchState(self.branch_id, self.subject_id, self.mode,
                                  self.source_snapshot_id,
                                  (*self.experiences, experience), skills,
                                  self.revision + 1)


class IdentityArchive:
    """Host-owned archive and resolver boundary; fictional data in tests only."""
    def __init__(self, subject_id):
        _identifier(subject_id, 'subject_id')
        self.subject_id = subject_id
        self._episodes = {}
        self._principles = {}
        self._snapshots = {}

    def add_episode(self, episode: SourceEpisode):
        if episode.subject_id != self.subject_id:
            raise PermissionError('episode subject differs from archive')
        if episode.episode_id in self._episodes:
            raise ValueError('duplicate episode ID')
        if episode.supersedes is not None and episode.supersedes not in self._episodes:
            raise ValueError('supersedes targets an unknown episode')
        self._episodes[episode.episode_id] = episode

    def add_principle(self, principle: PrincipleHypothesis):
        if principle.subject_id != self.subject_id:
            raise PermissionError('principle subject differs from archive')
        known = set(self._episodes)
        if not set(principle.supporting_episode_ids + principle.contradicting_episode_ids) <= known:
            raise ValueError('principle references unknown episodes')
        if principle.principle_id in self._principles:
            raise ValueError('duplicate principle ID')
        self._principles[principle.principle_id] = principle

    def build_snapshot(self, snapshot_id, *, version, created_at,
                       episode_ids, principle_ids, unresolved_questions=(),
                       permitted_uses=('retrieval', 'evaluation')):
        episodes = tuple(self._episodes[item] for item in episode_ids)
        principles = tuple(self._principles[item] for item in principle_ids)
        if any(item.status != 'approved' for item in principles):
            raise ValueError('snapshot principles must be approved')
        snapshot = IdentitySnapshot(
            snapshot_id, self.subject_id, version, created_at,
            tuple(item.episode_id for item in episodes),
            tuple(item.source_hash for item in episodes),
            tuple(item.principle_id for item in principles),
            tuple(unresolved_questions), tuple(permitted_uses))
        self._snapshots[snapshot_id] = snapshot
        return snapshot

    def revoke_episode(self, episode_id):
        episode = self._episodes.pop(episode_id)
        invalidated = []
        for snapshot_id, snapshot in list(self._snapshots.items()):
            if episode_id in snapshot.source_episode_ids and snapshot.status == 'active':
                self._snapshots[snapshot_id] = IdentitySnapshot(
                    **{**{name: getattr(snapshot, name) for name in
                          ('snapshot_id', 'subject_id', 'version', 'created_at',
                           'source_episode_ids', 'source_hashes', 'principle_ids',
                           'unresolved_questions', 'permitted_uses',
                           'schema_version')}, 'status': 'revoked'})
                invalidated.append(snapshot_id)
        return episode, tuple(invalidated)

    def resolve(self, snapshot_id, *, mode, branch_id, purpose, cutoff,
                branch_state: ReplicaBranchState | None = None,
                allowed_uses=('retrieval',)):
        if mode not in MODES:
            raise ValueError('unknown identity mode')
        if type(cutoff) is not int or cutoff < 0:
            raise ValueError('cutoff must be nonnegative')
        snapshot = self._snapshots.get(snapshot_id)
        if snapshot is None or snapshot.status != 'active':
            raise PermissionError('snapshot is absent or revoked')
        _identifier(purpose, 'purpose')
        if branch_id == 'source' and mode not in {'source_snapshot', 'source_current'}:
            raise ValueError('source branch is reserved for source modes')
        episodes = [self._episodes[item] for item in snapshot.source_episode_ids]
        # A correction observed by the cutoff supersedes its parent; a later
        # correction must not leak into an earlier historical view.
        superseded = {item.supersedes for item in episodes
                      if item.supersedes is not None and item.observed_at <= cutoff}
        episodes = [item for item in episodes if item.observed_at <= cutoff
                    and item.episode_id not in superseded
                    and item.authenticity in AUTHENTICITY
                    and set(item.permitted_uses) & set(allowed_uses)]
        principles = [self._principles[item] for item in snapshot.principle_ids
                      if self._principles[item].status == 'approved']
        if mode in {'source_snapshot', 'source_current'}:
            branch_id = 'source'
            branch_slots = ()
        else:
            branch_slots = ()
            if mode == 'parallel_replica':
                if branch_state is None:
                    raise ValueError('parallel_replica requires branch state')
                if (branch_state.branch_id != branch_id
                        or branch_state.subject_id != self.subject_id
                        or branch_state.source_snapshot_id != snapshot.snapshot_id):
                    raise PermissionError('branch state is not compatible with snapshot')
                branch_slots = tuple(
                    f'{item.experience_id}: {item.text} [{item.outcome}]'
                    for item in branch_state.experiences if item.observed_at <= cutoff)
        lines = [f'IDENTITY MODE: {mode}', f'SNAPSHOT: {snapshot.snapshot_id} v{snapshot.version}']
        if principles:
            lines.append('APPROVED PRINCIPLE HYPOTHESES (non-factual):')
            lines += [f'- {item.statement}' + (f' when {item.condition}' if item.condition else '')
                      for item in principles]
        if branch_slots:
            lines.append('BRANCH-OWNED EXPERIENCES (not source testimony):')
            lines += [f'- {item}' for item in branch_slots[-8:]]
        if snapshot.unresolved_questions:
            lines.append('UNRESOLVED / DO NOT INVENT:')
            lines += [f'- {item}' for item in snapshot.unresolved_questions]
        summary = '\n'.join(lines)[:MAX_SUMMARY]
        return IdentityContext(
            self.subject_id, snapshot.snapshot_id, snapshot.digest, mode, branch_id,
            purpose, cutoff, tuple(item.episode_id for item in episodes), summary,
            tuple(snapshot.unresolved_questions),
            {'identity_snapshot': snapshot.schema_version})


class IdentityResolver:
    """Thin explicit resolver wrapper; truth and authority remain host-owned."""
    def __init__(self, archive: IdentityArchive):
        self.archive = archive

    def resolve(self, snapshot_id, **kwargs):
        return self.archive.resolve(snapshot_id, **kwargs)


def render_identity_context(context: IdentityContext | None) -> str:
    """Bounded prompt section; never changes success verification or permissions."""
    if context is None or context.is_null or not context.summary.strip():
        return ''
    return ('IDENTITY CONTEXT (approved source; preferences only, not evidence '
            'authority and not task success):\n' + context.summary)
