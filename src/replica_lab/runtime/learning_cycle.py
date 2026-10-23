"""C13 boundary: actual runtime outcome -> bounded learning-director review.

This adapter never executes a proposal, publishes a manifest, changes a goal or
uses evaluator gold. It converts the observable trace counters into one
OutcomeSummary and delegates selection to the existing deterministic director.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .director import DirectorDecision, LearningDirector, OutcomeSummary


@dataclass(frozen=True)
class CycleEvidence:
    task_id: str
    task_family: str
    success: bool
    decisions: int
    tool_calls: int
    prompt_tokens: int
    generated_tokens: int
    wall_seconds: float
    source_event_ids: tuple[str, ...]
    verified_receipts: int = 0
    malformed_actions: int = 0
    false_success_claims: int = 0
    repeated_read: bool = False
    state_digest: str = ''

    def __post_init__(self):
        for name in ('task_id', 'task_family', 'state_digest'):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f'{name} must be text')
        if type(self.success) is not bool:
            raise ValueError('success must be boolean')
        for name in ('decisions', 'tool_calls', 'prompt_tokens',
                     'generated_tokens', 'verified_receipts', 'malformed_actions',
                     'false_success_claims'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be nonnegative integer')
        if not isinstance(self.wall_seconds, (int, float)) or self.wall_seconds < 0:
            raise ValueError('wall_seconds must be nonnegative')
        if not isinstance(self.source_event_ids, tuple) or not self.source_event_ids \
                or len(set(self.source_event_ids)) != len(self.source_event_ids):
            raise ValueError('distinct source event IDs required')
        if not isinstance(self.repeated_read, bool):
            raise ValueError('repeated_read must be boolean')


def classify_failure(evidence: CycleEvidence) -> str | None:
    if evidence.success:
        return None
    if evidence.malformed_actions:
        return 'action'
    if evidence.false_success_claims:
        return 'action'
    if evidence.repeated_read:
        return 'action'
    if evidence.tool_calls == 0:
        return 'retrieval'
    if evidence.verified_receipts:
        return 'procedure'
    return 'action'


def review_cycle(evidence: CycleEvidence, *,
                 director: LearningDirector | None = None,
                 parent_manifest_id: str | None = None) -> DirectorDecision:
    if not isinstance(evidence, CycleEvidence):
        raise ValueError('review_cycle requires CycleEvidence')
    if evidence.success:
        return DirectorDecision('no_update', 'successful_episode')
    failure_class = classify_failure(evidence)
    if failure_class is None:
        return DirectorDecision('no_update', 'no_failure')
    cost = (evidence.prompt_tokens + evidence.generated_tokens
            + evidence.tool_calls * 8 + evidence.decisions)
    summary = OutcomeSummary(
        task_family=evidence.task_family,
        failure_class=failure_class,
        attempts=max(1, evidence.decisions),
        successes=0,
        source_event_ids=evidence.source_event_ids,
        role='development',
        verified_receipts=evidence.verified_receipts,
        cost_units=float(cost),
        notes=f'state_digest={evidence.state_digest[:16]}')
    return (director or LearningDirector()).review(
        [summary], parent_manifest_id=parent_manifest_id,
        data_manifest={'task_family': evidence.task_family,
                       'task_id': evidence.task_id,
                       'source_event_ids': list(evidence.source_event_ids)})


def evidence_from_summary(summary: Mapping, *, task_family: str,
                          task_id: str, state_digest: str = '',
                          source_event_ids: tuple[str, ...] | None = None,
                          verified_receipts: int = 0) -> CycleEvidence:
    """Bounded adapter for TaskRuntime.summary; no future/gold fields."""
    if not isinstance(summary, Mapping):
        raise ValueError('summary must be a mapping')
    trace = summary.get('trace') or ()
    decisions = int(summary.get('decisions') or len(trace))
    actions = [item.get('action') or {} for item in trace]
    malformed = int(summary.get('malformed_actions') or 0)
    false_claims = int(summary.get('false_success_claims') or 0)
    repeated_read = any((actions[i].get('kind') == 'read'
                         and actions[i - 1].get('kind') == 'read')
                        for i in range(1, len(actions)))
    if source_event_ids is None:
        head = summary.get('journal_head')
        digest = summary.get('state_digest') or state_digest or 'unknown'
        source_event_ids = (str(head or task_id), str(digest))
    budget = summary.get('budget') or {}
    return CycleEvidence(
        task_id=task_id, task_family=task_family,
        success=bool(summary.get('success')),
        decisions=decisions,
        tool_calls=int(budget.get('used_tool_calls') or 0),
        prompt_tokens=int(budget.get('used_prompt_tokens') or 0),
        generated_tokens=int(budget.get('used_generated_tokens') or 0),
        wall_seconds=float(sum([(item.get('decision_wall_seconds') or 0.0)
                                for item in trace])),
        source_event_ids=tuple(source_event_ids),
        verified_receipts=int(verified_receipts),
        malformed_actions=malformed, false_success_claims=false_claims,
        repeated_read=repeated_read, state_digest=str(state_digest or ''))
