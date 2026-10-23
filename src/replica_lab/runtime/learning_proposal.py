"""Bounded learning proposals; host/independent-evaluator boundaries only."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

FORMAT = 'replica.learning.proposal.v1'
KINDS = ('procedure', 'record', 'neural')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


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
        for name in ('data_manifest', 'budget'):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not value:
                raise ValueError(f'{name} must be a nonempty mapping')
            object.__setattr__(self, name, MappingProxyType(dict(value)))
        if type(self.created_at) is not int or self.created_at < 0:
            raise ValueError('created_at must be nonnegative')
        if self.parent_manifest_id is not None and not isinstance(self.parent_manifest_id, str):
            raise ValueError('parent manifest must be text or None')

    @property
    def digest(self):
        return _digest({'format': FORMAT, 'proposal_id': self.proposal_id,
                        'parent_manifest_id': self.parent_manifest_id,
                        'source_event_ids': list(self.source_event_ids),
                        'target_kind': self.target_kind, 'target_id': self.target_id,
                        'data_manifest': dict(self.data_manifest),
                        'budget': dict(self.budget), 'optimizer': self.optimizer,
                        'evaluation_plan': self.evaluation_plan,
                        'created_at': self.created_at})

    def to_dict(self):
        return {'format': FORMAT, 'proposal_id': self.proposal_id,
                'parent_manifest_id': self.parent_manifest_id,
                'source_event_ids': list(self.source_event_ids),
                'target_kind': self.target_kind, 'target_id': self.target_id,
                'data_manifest': dict(self.data_manifest),
                'budget': dict(self.budget), 'optimizer': self.optimizer,
                'evaluation_plan': self.evaluation_plan,
                'created_at': self.created_at, 'digest': self.digest}
