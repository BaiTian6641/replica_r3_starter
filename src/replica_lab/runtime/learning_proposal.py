"""Bounded learning proposals; host/independent-evaluator boundaries only."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

FORMAT = 'replica.learning.proposal.v1'
KINDS = ('procedure', 'record', 'neural')
_KEYS = ('format', 'proposal_id', 'parent_manifest_id', 'source_event_ids',
         'target_kind', 'target_id', 'data_manifest', 'budget', 'optimizer',
         'evaluation_plan', 'created_at', 'digest')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def _body(proposal):
    return {'format': FORMAT, 'proposal_id': proposal.proposal_id,
            'parent_manifest_id': proposal.parent_manifest_id,
            'source_event_ids': list(proposal.source_event_ids),
            'target_kind': proposal.target_kind, 'target_id': proposal.target_id,
            'data_manifest': dict(proposal.data_manifest),
            'budget': dict(proposal.budget), 'optimizer': proposal.optimizer,
            'evaluation_plan': proposal.evaluation_plan,
            'created_at': proposal.created_at}


@dataclass(frozen=True)
class LearningProposal:
    proposal_id: str
    parent_manifest_id: str | None
    source_event_ids: tuple[str, ...]
    target_kind: str
    target_id: str
    data_manifest: Mapping
    budget: Mapping
    optimizer: str
    evaluation_plan: str
    created_at: int = 0

    def __post_init__(self):
        for name in ('proposal_id', 'target_id', 'optimizer', 'evaluation_plan'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{name} required')
        if self.target_kind not in KINDS:
            raise ValueError('unknown learning target kind')
        if not isinstance(self.source_event_ids, tuple) or not self.source_event_ids:
            raise ValueError('proposal requires source event IDs')
        if len(set(self.source_event_ids)) != len(self.source_event_ids):
            raise ValueError('proposal source event IDs must be distinct')
        for name in ('data_manifest', 'budget'):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not value:
                raise ValueError(f'{name} must be a nonempty mapping')
            object.__setattr__(self, name, MappingProxyType(dict(value)))
        if type(self.created_at) is not int or self.created_at < 0:
            raise ValueError('created_at must be nonnegative')
        if self.parent_manifest_id is not None and not isinstance(
                self.parent_manifest_id, str):
            raise ValueError('parent manifest must be text or None')

    @property
    def digest(self):
        return _digest(_body(self))

    def to_dict(self):
        return {**_body(self), 'digest': self.digest}

    @classmethod
    def from_dict(cls, value: Mapping) -> 'LearningProposal':
        """Reconstruct and authenticate one recorded proposal.

        The recorded digest and format are mandatory: a proposal snapshot is
        never trusted just because its fields parse.
        """
        if not isinstance(value, Mapping):
            raise TypeError('proposal snapshot must be a mapping')
        if set(value) != set(_KEYS):
            missing = sorted(set(_KEYS) - set(value))
            extra = sorted(set(value) - set(_KEYS))
            raise ValueError(
                f'proposal snapshot schema mismatch; missing={missing} extra={extra}')
        if value['format'] != FORMAT:
            raise ValueError('unknown proposal format')
        if not isinstance(value['digest'], str) or len(value['digest']) != 64:
            raise ValueError('proposal digest must be a SHA256')
        sources = value['source_event_ids']
        if not isinstance(sources, (list, tuple)) or not sources:
            raise ValueError('proposal source_event_ids must be a nonempty sequence')
        if any(not isinstance(item, str) or not item for item in sources):
            raise ValueError('proposal source_event_ids must be text')
        proposal = cls(
            proposal_id=value['proposal_id'],
            parent_manifest_id=value['parent_manifest_id'],
            source_event_ids=tuple(sources),
            target_kind=value['target_kind'],
            target_id=value['target_id'],
            data_manifest=value['data_manifest'],
            budget=value['budget'],
            optimizer=value['optimizer'],
            evaluation_plan=value['evaluation_plan'],
            created_at=value['created_at'])
        if proposal.digest != value['digest']:
            raise ValueError('proposal content does not match its digest')
        return proposal


def verify_proposal_snapshot(value: Mapping) -> LearningProposal:
    """Public alias used by manifest reload and candidate authority checks."""
    return LearningProposal.from_dict(value)
