"""C13 experience diagnosis and bounded learning-director baseline.

This is a deterministic catalog selector, not a trained policy. It consumes
actual outcome summaries from ordinary training/development roles and emits at
most one LearningProposal for host validation. It cannot execute a candidate,
publish a manifest, change evaluator rules, edit identity/goals or spend beyond
its declared budget. Audit/final outcomes are excluded by construction.

Catalog actions are deliberately coarse: record/memory repair, reusable verified
procedure, neural policy/representation candidate, deferral, or no update. A
learned ranking policy is a later, separately evaluated candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

from .learning_proposal import LearningProposal

ROLES = ('train', 'development')
FAILURE_CLASSES = ('retrieval', 'memory', 'action', 'representation', 'answer',
                   'procedure', 'insufficient_budget', 'implementation',
                   'evaluation_ambiguity')
TARGETS = {'retrieval': 'record', 'memory': 'record', 'procedure': 'procedure',
           'action': 'neural', 'representation': 'neural', 'answer': 'neural'}
PROTECTED_TARGETS = frozenset({'evaluator', 'goal', 'authority', 'identity',
                               'calibration', 'director'})


def _digest(value):
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be nonempty text')
    return value


@dataclass(frozen=True)
class OutcomeSummary:
    """Actual outcomes for one task family and failure class.

    ``source_event_ids`` must identify the verified experience used for the
    diagnosis. ``verified_receipts`` counts actual tool/verifier receipts; a
    model-written success sentence never counts.
    """
    task_family: str
    failure_class: str
    attempts: int
    successes: int
    source_event_ids: tuple[str, ...]
    role: str = 'train'
    verified_receipts: int = 0
    cost_units: float = 0.0
    notes: str = ''

    def __post_init__(self):
        _text(self.task_family, 'task_family')
        if self.failure_class not in FAILURE_CLASSES:
            raise ValueError('unknown failure class')
        if type(self.attempts) is not int or self.attempts < 1:
            raise ValueError('attempts must be positive')
        if type(self.successes) is not int or not 0 <= self.successes <= self.attempts:
            raise ValueError('successes must be within attempts')
        if not isinstance(self.source_event_ids, tuple) or not self.source_event_ids \
                or len(set(self.source_event_ids)) != len(self.source_event_ids):
            raise ValueError('distinct source event IDs are required')
        if self.role not in ROLES:
            raise ValueError('director accepts only train/development outcomes')
        if type(self.verified_receipts) is not int or self.verified_receipts < 0:
            raise ValueError('verified receipts must be nonnegative')
        if not isinstance(self.cost_units, (int, float)) or self.cost_units < 0:
            raise ValueError('cost must be nonnegative')

    @property
    def failures(self):
        return self.attempts - self.successes

    @property
    def digest(self):
        return _digest({'task_family': self.task_family,
                        'failure_class': self.failure_class,
                        'attempts': self.attempts, 'successes': self.successes,
                        'source_event_ids': list(self.source_event_ids),
                        'role': self.role, 'verified_receipts': self.verified_receipts,
                        'cost_units': self.cost_units, 'notes': self.notes})


@dataclass(frozen=True)
class DirectorBudget:
    max_proposals: int = 1
    max_cost_units: float = 100.0
    min_repeated_failures: int = 2
    min_neural_attempts: int = 8
    min_verified_receipts_for_procedure: int = 1

    def __post_init__(self):
        if self.max_proposals not in (0, 1):
            raise ValueError('this reference emits at most one proposal')
        if self.max_cost_units < 0:
            raise ValueError('cost budget must be nonnegative')
        if self.min_repeated_failures < 1 or self.min_neural_attempts < 1 \
                or self.min_verified_receipts_for_procedure < 1:
            raise ValueError('minimum-evidence bounds must be positive')


@dataclass(frozen=True)
class DirectorDecision:
    status: str  # proposed | deferred | no_update
    reason: str
    proposal: LearningProposal | None = None
    diagnostic: dict | None = None

    def __post_init__(self):
        if self.status not in {'proposed', 'deferred', 'no_update'}:
            raise ValueError('unknown director status')
        if (self.status == 'proposed') != (self.proposal is not None):
            raise ValueError('only a proposed decision carries a proposal')


class LearningDirector:
    """Deterministic catalog baseline; no training, execution or authority."""

    def __init__(self, *, budget: DirectorBudget = DirectorBudget()):
        self.budget = budget

    def review(self, outcomes: Iterable[OutcomeSummary], *,
               parent_manifest_id: str | None = None,
               data_manifest: dict | None = None) -> DirectorDecision:
        items = tuple(outcomes)
        if not items:
            return DirectorDecision('no_update', 'no_outcomes')
        if any(not isinstance(item, OutcomeSummary) for item in items):
            raise ValueError('director consumes OutcomeSummary values')
        # Aggregate by failure class. Repeated means actual failures, not
        # attempts or model-stated failures.
        groups = {}
        for item in items:
            group = groups.setdefault(item.failure_class, {
                'failures': 0, 'attempts': 0, 'successes': 0, 'cost': 0.0,
                'sources': set(), 'receipts': 0, 'families': set()})
            group['failures'] += item.failures
            group['attempts'] += item.attempts
            group['successes'] += item.successes
            group['cost'] += float(item.cost_units)
            group['sources'].update(item.source_event_ids)
            group['receipts'] += item.verified_receipts
            group['families'].add(item.task_family)
        diagnostic = {name: {'failures': value['failures'],
                             'attempts': value['attempts'],
                             'successes': value['successes'],
                             'cost_units': value['cost'],
                             'source_event_ids': sorted(value['sources']),
                             'verified_receipts': value['receipts'],
                             'task_families': sorted(value['families'])}
                      for name, value in groups.items()}
        total_cost = sum(value['cost'] for value in groups.values())
        if total_cost > self.budget.max_cost_units:
            return DirectorDecision('deferred', 'budget_exceeded', diagnostic=diagnostic)
        if self.budget.max_proposals == 0:
            return DirectorDecision('deferred', 'proposal_budget_zero', diagnostic=diagnostic)
        # Choose the failure class with the most actual failures; ties follow the
        # documented catalog order for reproducibility.
        order = {name: index for index, name in enumerate(FAILURE_CLASSES)}
        ranked = sorted(groups.items(),
                        key=lambda item: (-item[1]['failures'], order[item[0]]))
        failure_class, group = ranked[0]
        if group['failures'] < self.budget.min_repeated_failures:
            return DirectorDecision('no_update', 'no_repeated_failure', diagnostic=diagnostic)
        if failure_class in {'insufficient_budget', 'implementation',
                             'evaluation_ambiguity'}:
            return DirectorDecision('no_update', f'not_a_learning_target:{failure_class}',
                                    diagnostic=diagnostic)
        target_kind = TARGETS.get(failure_class)
        if target_kind is None:
            return DirectorDecision('no_update', 'no_catalog_target', diagnostic=diagnostic)
        target_id = {'record': 'memory-selection',
                     'procedure': 'repair-config',
                     'neural': 'action-policy'}[target_kind]
        if target_id in PROTECTED_TARGETS:
            return DirectorDecision('no_update', 'protected_target', diagnostic=diagnostic)
        if target_kind == 'procedure' and group['receipts'] < \
                self.budget.min_verified_receipts_for_procedure:
            return DirectorDecision('deferred', 'procedure_receipts_missing',
                                    diagnostic=diagnostic)
        if target_kind == 'neural' and group['attempts'] < self.budget.min_neural_attempts:
            return DirectorDecision('deferred', 'neural_labels_insufficient',
                                    diagnostic=diagnostic)
        sources = tuple(sorted(group['sources']))
        if not sources:
            return DirectorDecision('deferred', 'source_events_missing',
                                    diagnostic=diagnostic)
        manifest = dict(data_manifest or {'task_families': sorted(group['families'])})
        if any(name in manifest for name in ('audit', 'final_evaluation', 'gold')):
            return DirectorDecision('no_update', 'audit_or_gold_forbidden',
                                    diagnostic=diagnostic)
        optimizer = {'record': 'none', 'procedure': 'host_tools',
                     'neural': 'muon8bit+adamw_fp32'}[target_kind]
        proposal = LearningProposal(
            proposal_id=f'director-{failure_class}-{_digest([sources, target_kind])[:16]}',
            parent_manifest_id=parent_manifest_id,
            source_event_ids=sources,
            target_kind=target_kind,
            target_id=target_id,
            data_manifest=manifest,
            budget={'cost_units': total_cost,
                    'attempts': group['attempts'],
                    'failures': group['failures'],
                    'max_proposals': self.budget.max_proposals},
            optimizer=optimizer,
            evaluation_plan=('independent grouped evaluation against the parent; '
                             'retention and cost reported; no auditor/gold exposure'),
            created_at=0)
        return DirectorDecision('proposed', f'repeated_{failure_class}_failure',
                                proposal=proposal, diagnostic=diagnostic)
