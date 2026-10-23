"""Exact task state: authorized goal, claims, obligations and typed patches (E3 §3).

The model proposes; this module decides. ``apply_patch`` is the only commit path
for task-state edits and every denial is a typed, recordable result. All
collections are immutable and validated; digests are canonical-JSON fingerprints
shared with the architecture contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

from ..architecture.contracts import TaskGraph, fingerprint

CLAIM_STATUSES = ('known', 'inferred', 'unknown', 'hypothetical')
SUBGOAL_STATES = ('open', 'active', 'done', 'blocked')
ACTION_STATES = ('proposed', 'attempted', 'completed', 'denied', 'failed')
OUTCOME_STATUSES = ('completed', 'denied', 'failed', 'pending_input')
OBLIGATION_STATES = ('unresolved', 'satisfied')
EVIDENCE_KINDS = ('receipt', 'observation', 'record')
TOOL_KINDS = ('read_file', 'write_file', 'set_constant', 'run_test')
PATCH_OPS = ('add_claim', 'revise_claim', 'add_subgoal', 'add_obligation',
             'satisfy_obligation', 'add_evidence', 'set_subgoal_state', 'note')
AUTHORS = ('host', 'model')

# Bounded resources; exact state stays replayable and feature-bounded.
MAX_SUCCESS_CONDITIONS = 4
MAX_CONSTRAINTS = 16
MAX_SUBGOALS = 32
MAX_CLAIMS = 64
MAX_EVIDENCE = 64
MAX_OBLIGATIONS = 32
MAX_ACTIONS = 64
MAX_OUTCOMES = 64
MAX_NOTES = 32
MAX_TEXT = 2000
MAX_NOTE = 512
MAX_ID = 128

_SUBGOAL_TRANSITIONS = {
    'open': frozenset({'open', 'active', 'blocked'}),
    'active': frozenset({'active', 'done', 'blocked'}),
    'blocked': frozenset({'blocked', 'active'}),
    'done': frozenset({'done'}),
}

_FEATURE_LAYOUT = (
    'revision_over_32', 'open_obligations_over_8', 'constraints_over_8',
    'claims_known_over_8', 'claims_inferred_over_8', 'claims_unknown_over_8',
    'claims_hypothetical_over_8', 'evidence_over_8', 'subgoals_total_over_8',
    'subgoals_done_over_8', 'actions_proposed_over_8', 'actions_attempted_over_8',
    'actions_completed_over_8', 'actions_denied_over_8', 'actions_failed_over_8',
    'outcomes_over_8', 'decisions_used_fraction', 'tool_calls_used_fraction',
    'token_budget_used_fraction', 'last_outcome_failed', 'last_outcome_denied',
    'last_outcome_completed', 'success_conditions_over_4', 'remaining_budget_fraction',
)
TASK_FEATURE_COUNT = 24


def _text(value: Any, name: str, limit: int = MAX_TEXT) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    if len(value) > limit:
        raise ValueError(f'{name} exceeds the {limit} character bound')
    return value


def _id(value: Any, name: str) -> str:
    return _text(value, name, MAX_ID)


def _freeze(value: Any):
    if isinstance(value, MappingProxyType):
        return value
    if isinstance(value, dict):
        return MappingProxyType({str(k): _freeze(v) for k, v in sorted(value.items())})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(f'Unsupported exact-state value type: {type(value).__name__}')


def _plain(value: Any):
    if isinstance(value, MappingProxyType):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def _count(collection, limit, name) -> int:
    if len(collection) > limit:
        raise ValueError(f'{name} exceeds the bounded limit of {limit}')
    return len(collection)


@dataclass(frozen=True)
class SuccessCondition:
    """Independently checkable condition; satisfied only by a host receipt."""
    condition_id: str
    description: str

    def __post_init__(self):
        _id(self.condition_id, 'condition_id')
        _text(self.description, 'condition description')

    def to_dict(self):
        return {'condition_id': self.condition_id, 'description': self.description}


@dataclass(frozen=True)
class Constraint:
    """A host-declared restriction the plan must respect."""
    constraint_id: str
    text: str

    def __post_init__(self):
        _id(self.constraint_id, 'constraint_id')
        _text(self.text, 'constraint text')

    def to_dict(self):
        return {'constraint_id': self.constraint_id, 'text': self.text}


@dataclass(frozen=True)
class Subgoal:
    """One DAG node of the decomposed plan; no cycles, dependencies must exist."""
    subgoal_id: str
    description: str
    state: str = 'open'
    depends_on: tuple[str, ...] = ()

    def __post_init__(self):
        _id(self.subgoal_id, 'subgoal_id')
        _text(self.description, 'subgoal description')
        if self.state not in SUBGOAL_STATES:
            raise ValueError(f'Unknown subgoal state: {self.state!r}')
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError('Subgoal dependencies must be distinct')
        if self.subgoal_id in self.depends_on:
            raise ValueError('A subgoal cannot depend on itself')
        object.__setattr__(self, 'depends_on', tuple(_id(d, 'dependency') for d in self.depends_on))

    def to_dict(self):
        return {'subgoal_id': self.subgoal_id, 'description': self.description,
                'state': self.state, 'depends_on': list(self.depends_on)}


@dataclass(frozen=True)
class EvidenceRef:
    """A pointer at host-observed material; never model text."""
    evidence_id: str
    kind: str
    digest: str
    origin: str

    def __post_init__(self):
        _id(self.evidence_id, 'evidence_id')
        if self.kind not in EVIDENCE_KINDS:
            raise ValueError(f'Unknown evidence kind: {self.kind!r}')
        if (not isinstance(self.digest, str) or len(self.digest) != 64
                or any(c not in '0123456789abcdef' for c in self.digest)):
            raise ValueError('Evidence digest must be a sha256 hex string')
        _text(self.origin, 'evidence origin', MAX_ID)

    def to_dict(self):
        return {'evidence_id': self.evidence_id, 'kind': self.kind,
                'digest': self.digest, 'origin': self.origin}


@dataclass(frozen=True)
class Claim:
    """A proposition with an explicit epistemic status and evidence pointers."""
    claim_id: str
    text: str
    status: str
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self):
        _id(self.claim_id, 'claim_id')
        _text(self.text, 'claim text')
        if self.status not in CLAIM_STATUSES:
            raise ValueError(f'Unknown claim status: {self.status!r}')
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError('Claim evidence references must be distinct')
        object.__setattr__(self, 'evidence_ids', tuple(_id(e, 'evidence id') for e in self.evidence_ids))

    def to_dict(self):
        return {'claim_id': self.claim_id, 'text': self.text, 'status': self.status,
                'evidence_ids': list(self.evidence_ids)}


@dataclass(frozen=True)
class Obligation:
    """An unresolved duty; satisfaction requires a host-verified receipt."""
    obligation_id: str
    description: str
    state: str = 'unresolved'
    receipt_digest: str | None = None

    def __post_init__(self):
        _id(self.obligation_id, 'obligation_id')
        _text(self.description, 'obligation description')
        if self.state not in OBLIGATION_STATES:
            raise ValueError(f'Unknown obligation state: {self.state!r}')
        if self.state == 'satisfied' and not isinstance(self.receipt_digest, str):
            raise ValueError('A satisfied obligation requires its receipt digest')
        if self.receipt_digest is not None:
            if (len(self.receipt_digest) != 64
                    or any(c not in '0123456789abcdef' for c in self.receipt_digest)):
                raise ValueError('Obligation receipt digest must be a sha256 hex string')

    def to_dict(self):
        return {'obligation_id': self.obligation_id, 'description': self.description,
                'state': self.state, 'receipt_digest': self.receipt_digest}


@dataclass(frozen=True)
class ActionRecord:
    """One proposed action and its host-tracked lifecycle state."""
    action_id: str
    kind: str
    args: Mapping
    state: str
    revision: int
    idempotency_key: str
    tool: str | None = None
    subgoal_id: str | None = None

    def __post_init__(self):
        _id(self.action_id, 'action_id')
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError('Action kind required')
        if self.state not in ACTION_STATES:
            raise ValueError(f'Unknown action state: {self.state!r}')
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError('Action revision must be a nonnegative integer')
        _text(self.idempotency_key, 'idempotency key', MAX_ID)
        _freeze(self.args)
        if self.tool is not None:
            _id(self.tool, 'tool')
        if self.subgoal_id is not None:
            _id(self.subgoal_id, 'subgoal_id')

    def to_dict(self):
        return {'action_id': self.action_id, 'kind': self.kind, 'args': _plain(self.args),
                'state': self.state, 'revision': self.revision,
                'idempotency_key': self.idempotency_key, 'tool': self.tool,
                'subgoal_id': self.subgoal_id}


@dataclass(frozen=True)
class OutcomeRecord:
    """An actual result with its receipt; model claims are never outcomes."""
    outcome_id: str
    action_id: str
    status: str
    actual: bool
    result: Mapping
    revision: int
    observation_ids: tuple[str, ...] = ()
    receipt: Mapping | None = None
    error: str | None = None
    spent: Mapping | None = None

    def __post_init__(self):
        _id(self.outcome_id, 'outcome_id')
        _id(self.action_id, 'action_id')
        if self.status not in OUTCOME_STATUSES:
            raise ValueError(f'Unknown outcome status: {self.status!r}')
        if self.actual is not True:
            raise ValueError('Outcome records are always actual; hypothetical results are claims')
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError('Outcome revision must be a nonnegative integer')
        _freeze(self.result)
        if self.receipt is not None:
            _freeze(self.receipt)
        if self.spent is not None:
            _freeze(self.spent)
        object.__setattr__(self, 'observation_ids',
                           tuple(_id(o, 'observation id') for o in self.observation_ids))

    def to_dict(self):
        return {'outcome_id': self.outcome_id, 'action_id': self.action_id,
                'status': self.status, 'actual': True, 'result': _plain(self.result),
                'revision': self.revision, 'observation_ids': list(self.observation_ids),
                'receipt': _plain(self.receipt) if self.receipt is not None else None,
                'error': self.error, 'spent': _plain(self.spent) if self.spent is not None else None}

    @classmethod
    def from_dict(cls, value):
        return cls(outcome_id=value['outcome_id'], action_id=value['action_id'],
                   status=value['status'], actual=True, result=dict(value['result']),
                   revision=value['revision'], observation_ids=tuple(value['observation_ids']),
                   receipt=value['receipt'] if value.get('receipt') is not None else None,
                   error=value.get('error'),
                   spent=value['spent'] if value.get('spent') is not None else None)


@dataclass(frozen=True)
class Budget:
    """Declared decision/tool/token budgets and their actual usage."""
    max_decisions: int
    max_tool_calls: int
    max_prompt_tokens: int
    max_generated_tokens: int
    used_decisions: int = 0
    used_tool_calls: int = 0
    used_prompt_tokens: int = 0
    used_generated_tokens: int = 0

    def __post_init__(self):
        for name in ('max_decisions', 'max_tool_calls', 'max_prompt_tokens', 'max_generated_tokens',
                     'used_decisions', 'used_tool_calls', 'used_prompt_tokens', 'used_generated_tokens'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'Budget.{name} must be a nonnegative integer')

    def allows(self, *, decisions=0, tool_calls=0, prompt_tokens=0, generated_tokens=0) -> bool:
        return (self.used_decisions + decisions <= self.max_decisions
                and self.used_tool_calls + tool_calls <= self.max_tool_calls
                and self.used_prompt_tokens + prompt_tokens <= self.max_prompt_tokens
                and self.used_generated_tokens + generated_tokens <= self.max_generated_tokens)

    def spend(self, *, decisions=0, tool_calls=0, prompt_tokens=0, generated_tokens=0) -> 'Budget':
        if not self.allows(decisions=decisions, tool_calls=tool_calls,
                           prompt_tokens=prompt_tokens, generated_tokens=generated_tokens):
            raise ValueError('Budget exhausted; spend denied before execution')
        return Budget(self.max_decisions, self.max_tool_calls, self.max_prompt_tokens,
                      self.max_generated_tokens, self.used_decisions + decisions,
                      self.used_tool_calls + tool_calls, self.used_prompt_tokens + prompt_tokens,
                      self.used_generated_tokens + generated_tokens)

    def to_dict(self):
        return {name: getattr(self, name) for name in
                ('max_decisions', 'max_tool_calls', 'max_prompt_tokens', 'max_generated_tokens',
                 'used_decisions', 'used_tool_calls', 'used_prompt_tokens', 'used_generated_tokens')}

    @classmethod
    def from_dict(cls, value):
        return cls(**{name: value[name] for name in
                      ('max_decisions', 'max_tool_calls', 'max_prompt_tokens', 'max_generated_tokens',
                       'used_decisions', 'used_tool_calls', 'used_prompt_tokens', 'used_generated_tokens')})


def _validated_binding(value: Mapping) -> Mapping:
    if not isinstance(value, Mapping) or not value:
        raise ValueError('Binding must be a nonempty dict of component versions')
    frozen = {}
    for key, item in value.items():
        _id(key, 'binding key')
        if isinstance(item, dict):
            frozen[key] = MappingProxyType({_id(k, 'binding subkey'): _plain_scalar(v)
                                            for k, v in sorted(item.items())})
        else:
            frozen[key] = _plain_scalar(item)
    return MappingProxyType(frozen)


def _plain_scalar(value):
    if isinstance(value, bool) or type(value) in (int, float) or isinstance(value, str):
        return value
    raise ValueError('Binding values must be JSON scalars')


def _creates_cycle(edges: dict, new_id: str | None, deps: tuple[str, ...]) -> bool:
    edges = dict(edges)
    if new_id is not None:
        edges[new_id] = deps
    seen, stack = set(), set()

    def visit(node):
        if node in stack:
            return True
        if node in seen:
            return False
        seen.add(node)
        stack.add(node)
        if any(dep in edges and visit(dep) for dep in edges[node]):
            return True
        stack.discard(node)
        return False

    return any(visit(node) for node in edges)


@dataclass(frozen=True)
class ExactTaskState:
    """Host-owned exact task state; the journal is the only required copy."""
    task_id: str
    subject: str
    branch: str
    purpose: str
    source_epoch: int
    goal: str
    success_conditions: tuple[SuccessCondition, ...]
    constraints: tuple[Constraint, ...]
    subgoals: tuple[Subgoal, ...]
    claims: tuple[Claim, ...]
    evidence: tuple[EvidenceRef, ...]
    obligations: tuple[Obligation, ...]
    actions: tuple[ActionRecord, ...]
    outcomes: tuple[OutcomeRecord, ...]
    notes: tuple[str, ...]
    budget: Budget
    binding: Mapping
    allowed_tools: tuple[str, ...]
    revision: int = 0

    def __post_init__(self):
        _id(self.task_id, 'task_id')
        _id(self.subject, 'subject')
        _id(self.branch, 'branch')
        _id(self.purpose, 'purpose')
        if type(self.source_epoch) is not int or self.source_epoch < 0:
            raise ValueError('source_epoch must be a nonnegative integer')
        _text(self.goal, 'goal')
        _count(self.success_conditions, MAX_SUCCESS_CONDITIONS, 'success conditions')
        _count(self.constraints, MAX_CONSTRAINTS, 'constraints')
        _count(self.subgoals, MAX_SUBGOALS, 'subgoals')
        _count(self.claims, MAX_CLAIMS, 'claims')
        _count(self.evidence, MAX_EVIDENCE, 'evidence')
        _count(self.obligations, MAX_OBLIGATIONS, 'obligations')
        _count(self.actions, MAX_ACTIONS, 'actions')
        _count(self.outcomes, MAX_OUTCOMES, 'outcomes')
        _count(self.notes, MAX_NOTES, 'notes')
        if not isinstance(self.budget, Budget):
            raise ValueError('Exact state requires a typed Budget')
        object.__setattr__(self, 'binding', _validated_binding(self.binding))
        if not self.allowed_tools or not set(self.allowed_tools) <= set(TOOL_KINDS):
            raise ValueError(f'allowed_tools must be a nonempty subset of {TOOL_KINDS}')
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError('revision must be a nonnegative integer')
        ids = ([c.condition_id for c in self.success_conditions]
               + [c.constraint_id for c in self.constraints]
               + [s.subgoal_id for s in self.subgoals]
               + [c.claim_id for c in self.claims]
               + [e.evidence_id for e in self.evidence]
               + [o.obligation_id for o in self.obligations])
        if len(set(ids)) != len(ids):
            raise ValueError('Exact-state identifiers must be unique across collections')
        for note in self.notes:
            _text(note, 'note', MAX_NOTE)
        subgoal_map = {s.subgoal_id: s for s in self.subgoals}
        edges = {s.subgoal_id: tuple(s.depends_on) for s in self.subgoals}
        evidence_ids = {e.evidence_id for e in self.evidence}
        for subgoal in self.subgoals:
            if any(dep not in subgoal_map for dep in subgoal.depends_on):
                raise ValueError(f'Subgoal {subgoal.subgoal_id} has an unknown dependency')
        if _creates_cycle(edges, None, ()):
            raise ValueError('Subgoal graph must be acyclic')
        for claim in self.claims:
            if any(e not in evidence_ids for e in claim.evidence_ids):
                raise ValueError(f'Claim {claim.claim_id} references unknown evidence')

    # ---- projections -----------------------------------------------------
    @property
    def digest(self):
        return fingerprint(self.to_dict())

    def to_dict(self):
        return {'task_id': self.task_id, 'subject': self.subject, 'branch': self.branch,
                'purpose': self.purpose, 'source_epoch': self.source_epoch, 'goal': self.goal,
                'success_conditions': [c.to_dict() for c in self.success_conditions],
                'constraints': [c.to_dict() for c in self.constraints],
                'subgoals': [s.to_dict() for s in self.subgoals],
                'claims': [c.to_dict() for c in self.claims],
                'evidence': [e.to_dict() for e in self.evidence],
                'obligations': [o.to_dict() for o in self.obligations],
                'actions': [a.to_dict() for a in self.actions],
                'outcomes': [o.to_dict() for o in self.outcomes],
                'notes': list(self.notes), 'budget': self.budget.to_dict(),
                'binding': _plain(self.binding), 'allowed_tools': list(self.allowed_tools),
                'revision': self.revision}

    @classmethod
    def from_dict(cls, value):
        return cls(
            task_id=value['task_id'], subject=value['subject'], branch=value['branch'],
            purpose=value['purpose'], source_epoch=value['source_epoch'], goal=value['goal'],
            success_conditions=tuple(SuccessCondition(**c) for c in value['success_conditions']),
            constraints=tuple(Constraint(**c) for c in value['constraints']),
            subgoals=tuple(Subgoal(**s) for s in value['subgoals']),
            claims=tuple(Claim(**c) for c in value['claims']),
            evidence=tuple(EvidenceRef(**e) for e in value['evidence']),
            obligations=tuple(Obligation(**o) for o in value['obligations']),
            actions=tuple(ActionRecord(**a) for a in value['actions']),
            outcomes=tuple(OutcomeRecord(**o) for o in value['outcomes']),
            notes=tuple(value['notes']), budget=Budget.from_dict(value['budget']),
            binding=dict(value['binding']), allowed_tools=tuple(value['allowed_tools']),
            revision=value['revision'])

    def task_graph(self) -> TaskGraph:
        """Request-level projection; the frame may carry this, never the full state."""
        return TaskGraph(
            task_id=self.task_id, goal=self.goal, revision=self.revision,
            constraints=tuple(c.text for c in self.constraints),
            evidence_ids=tuple(e.evidence_id for e in self.evidence),
            open_obligations=tuple(o.obligation_id for o in self.obligations
                                   if o.state == 'unresolved'),
            success_conditions=tuple(c.condition_id for c in self.success_conditions),
            state_digest=self.digest,
            budget_remaining=self.budget.max_decisions - self.budget.used_decisions)

    def features(self) -> tuple[float, ...]:
        """Host-computed 24-float task vector; actual state, never a model claim."""
        budget = self.budget
        token_max = budget.max_prompt_tokens + budget.max_generated_tokens
        token_used = budget.used_prompt_tokens + budget.used_generated_tokens
        last = self.outcomes[-1] if self.outcomes else None
        open_obligations = sum(1 for o in self.obligations if o.state == 'unresolved')
        subgoals_done = sum(1 for s in self.subgoals if s.state == 'done')
        values = (
            min(1.0, self.revision / 32.0),
            min(1.0, open_obligations / 8.0),
            min(1.0, len(self.constraints) / 8.0),
            min(1.0, sum(1 for c in self.claims if c.status == 'known') / 8.0),
            min(1.0, sum(1 for c in self.claims if c.status == 'inferred') / 8.0),
            min(1.0, sum(1 for c in self.claims if c.status == 'unknown') / 8.0),
            min(1.0, sum(1 for c in self.claims if c.status == 'hypothetical') / 8.0),
            min(1.0, len(self.evidence) / 8.0),
            min(1.0, len(self.subgoals) / 8.0),
            min(1.0, subgoals_done / 8.0),
            min(1.0, sum(1 for a in self.actions if a.state == 'proposed') / 8.0),
            min(1.0, sum(1 for a in self.actions if a.state == 'attempted') / 8.0),
            min(1.0, sum(1 for a in self.actions if a.state == 'completed') / 8.0),
            min(1.0, sum(1 for a in self.actions if a.state == 'denied') / 8.0),
            min(1.0, sum(1 for a in self.actions if a.state == 'failed') / 8.0),
            min(1.0, len(self.outcomes) / 8.0),
            min(1.0, budget.used_decisions / max(1, budget.max_decisions)),
            min(1.0, budget.used_tool_calls / max(1, budget.max_tool_calls)),
            min(1.0, token_used / max(1, token_max)),
            1.0 if last is not None and last.status == 'failed' else 0.0,
            1.0 if last is not None and last.status == 'denied' else 0.0,
            1.0 if last is not None and last.status == 'completed' else 0.0,
            min(1.0, len(self.success_conditions) / 4.0),
            max(0.0, 1.0 - budget.used_decisions / max(1, budget.max_decisions)),
        )
        if len(values) != TASK_FEATURE_COUNT:
            raise RuntimeError('Feature layout drift')
        return tuple(min(1.0, max(0.0, float(v))) for v in values)

    def summary_text(self, max_chars: int = 1600) -> str:
        """Bounded human-readable rendering; observations appear as excerpts only."""
        lines = [f'GOAL: {self.goal}',
                 f'REVISION: {self.revision}',
                 'SUCCESS: ' + '; '.join(f'{c.condition_id}: {c.description}'
                                         for c in self.success_conditions)]
        if self.constraints:
            lines.append('CONSTRAINTS: ' + '; '.join(c.text for c in self.constraints))
        if self.subgoals:
            lines.append('SUBGOALS: ' + '; '.join(f'{s.subgoal_id}[{s.state}] {s.description}'
                                                  for s in self.subgoals))
        if self.claims:
            lines.append('CLAIMS: ' + '; '.join(f'{c.claim_id}[{c.status}] {c.text}'
                                                for c in self.claims[-8:]))
        open_obligations = [o for o in self.obligations if o.state == 'unresolved']
        if open_obligations:
            lines.append('OPEN OBLIGATIONS: ' + '; '.join(f'{o.obligation_id} {o.description}'
                                                          for o in open_obligations))
        if self.outcomes:
            last = self.outcomes[-1]
            lines.append(f'LAST OUTCOME: {last.status} {last.error or ""}'.strip())
        text = '\n'.join(lines)
        return text if len(text) <= max_chars else text[:max_chars]

    # ---- host-authoritative transitions (journal-driven, never model-called) ----
    def spend(self, *, decisions=0, tool_calls=0, prompt_tokens=0, generated_tokens=0) -> 'ExactTaskState':
        return replace(self, budget=self.budget.spend(
            decisions=decisions, tool_calls=tool_calls, prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens), revision=self.revision + 1)

    def with_action(self, action: ActionRecord) -> 'ExactTaskState':
        if not isinstance(action, ActionRecord):
            raise ValueError('with_action requires an ActionRecord')
        if any(a.action_id == action.action_id for a in self.actions):
            raise ValueError(f'Duplicate action id {action.action_id}')
        return replace(self, actions=(*self.actions, action), revision=self.revision + 1)

    def with_action_state(self, action_id: str, state: str) -> 'ExactTaskState':
        if state not in ACTION_STATES:
            raise ValueError(f'Unknown action state: {state!r}')
        if not any(a.action_id == action_id for a in self.actions):
            raise ValueError(f'Unknown action {action_id}')
        actions = tuple(replace(a, state=state) if a.action_id == action_id else a
                        for a in self.actions)
        return replace(self, actions=actions, revision=self.revision + 1)

    def with_outcome(self, outcome: OutcomeRecord) -> 'ExactTaskState':
        if not isinstance(outcome, OutcomeRecord):
            raise ValueError('with_outcome requires an OutcomeRecord')
        if outcome.revision != self.revision:
            raise ValueError('Outcome must be recorded at the current exact revision')
        if not any(a.action_id == outcome.action_id for a in self.actions):
            raise ValueError(f'Outcome references unknown action {outcome.action_id}')
        if any(o.outcome_id == outcome.outcome_id for o in self.outcomes):
            raise ValueError(f'Duplicate outcome id {outcome.outcome_id}')
        status_by_action = {o.action_id: o.status for o in self.outcomes}
        status_by_action[outcome.action_id] = outcome.status
        outcome_to_action = {'completed': 'completed', 'denied': 'denied',
                             'failed': 'failed', 'pending_input': 'attempted'}
        actions = tuple(replace(a, state=outcome_to_action[status_by_action[a.action_id]])
                        if a.action_id in status_by_action else a for a in self.actions)
        return replace(self, actions=actions, outcomes=(*self.outcomes, outcome),
                       revision=self.revision + 1)

    def subgoal_map(self):
        return {s.subgoal_id: s for s in self.subgoals}

    def claim_map(self):
        return {c.claim_id: c for c in self.claims}

    def paused(self) -> bool:
        return bool(self.outcomes) and self.outcomes[-1].status == 'pending_input'

    def satisfied_conditions(self, verifier) -> tuple[str, ...]:
        """Success conditions with a passing independent host receipt only."""
        met = set()
        for outcome in self.outcomes:
            receipt = dict(outcome.receipt) if outcome.receipt is not None else {}
            if (outcome.status == 'completed' and receipt.get('kind') == 'verify_success'
                    and receipt.get('passed') is True and receipt.get('exit_code') == 0
                    and verifier.receipt_verified(receipt.get('receipt_sha256'))):
                met.update(receipt.get('condition_ids', ()))
        return tuple(c.condition_id for c in self.success_conditions if c.condition_id in met)

    def is_complete(self, verifier) -> bool:
        conditions = {c.condition_id for c in self.success_conditions}
        return bool(conditions) and conditions <= set(self.satisfied_conditions(verifier))


@dataclass(frozen=True)
class TaskPatch:
    """One typed edit request; the op set is closed and authority is explicit."""
    op: str
    author: str
    payload: Mapping

    def __post_init__(self):
        if not isinstance(self.op, str) or not self.op:
            raise ValueError('Patch op required')
        if self.author not in AUTHORS:
            raise ValueError(f'Patch author must be one of {AUTHORS}')
        object.__setattr__(self, 'payload', _freeze(self.payload))

    def to_dict(self):
        return {'op': self.op, 'author': self.author, 'payload': _plain(self.payload)}

    @classmethod
    def from_dict(cls, value):
        return cls(op=value['op'], author=value['author'], payload=dict(value['payload']))


@dataclass(frozen=True)
class PatchResult:
    """Typed patch outcome; denials are explicit and never silently dropped."""
    applied: bool
    reason: str
    patch: TaskPatch
    state: ExactTaskState | None = None

    def to_dict(self):
        return {'applied': self.applied, 'reason': self.reason, 'patch': self.patch.to_dict()}


class PatchVerifier:
    """Host-side verification callbacks used by apply_patch."""

    def host_evidence(self, ref: EvidenceRef) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def receipt_verified(self, receipt_digest: Any) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


def _denied(patch: TaskPatch, reason: str) -> PatchResult:
    return PatchResult(False, reason, patch)


def _payload_text(payload: Mapping, key: str, *, required: bool, limit: int = MAX_TEXT):
    value = payload.get(key)
    if value is None:
        if required:
            raise ValueError(f'Patch payload requires {key}')
        return None
    return _text(value, f'payload {key}', limit)


def _payload_id(payload: Mapping, key: str, *, required: bool = True):
    value = payload.get(key)
    if value is None:
        if required:
            raise ValueError(f'Patch payload requires {key}')
        return None
    return _id(value, f'payload {key}')


def _evidence_ok(state: ExactTaskState, evidence_ids: tuple[str, ...], verifier) -> bool:
    refs = {e.evidence_id: e for e in state.evidence}
    return all(e in refs and verifier.host_evidence(refs[e]) for e in evidence_ids)


def apply_patch(state: ExactTaskState, patch: TaskPatch, verifier) -> PatchResult:
    """The only commit path for task-state edits. Returns a typed result.

    Denied requests never mutate state; the caller records the denial in the
    journal so every rejection stays observable.
    """
    if not isinstance(state, ExactTaskState) or not isinstance(patch, TaskPatch):
        raise ValueError('apply_patch requires ExactTaskState and TaskPatch')
    if not isinstance(verifier, PatchVerifier):
        raise ValueError('apply_patch requires a PatchVerifier')
    payload = patch.payload
    model = patch.author == 'model'
    try:
        if patch.op == 'add_claim':
            claim_id = _payload_id(payload, 'claim_id')
            text = _payload_text(payload, 'text', required=True)
            status = payload.get('status', 'inferred' if model else 'known')
            if status not in CLAIM_STATUSES:
                return _denied(patch, 'unknown_claim_status')
            if model and status == 'known':
                return _denied(patch, 'model_may_not_add_known_claim')
            evidence_ids = tuple(payload.get('evidence_ids') or ())
            if status == 'known':
                if not evidence_ids or not _evidence_ok(state, evidence_ids, verifier):
                    return _denied(patch, 'known_claim_requires_host_evidence')
            if any(c.claim_id == claim_id for c in state.claims):
                return _denied(patch, 'duplicate_claim_id')
            if len(state.claims) >= MAX_CLAIMS:
                return _denied(patch, 'claim_limit_reached')
            claim = Claim(claim_id, text, status, evidence_ids)
            updated = replace(state, claims=(*state.claims, claim), revision=state.revision + 1)
            return PatchResult(True, 'applied', patch, updated)

        if patch.op == 'revise_claim':
            claim_id = _payload_id(payload, 'claim_id')
            claims = state.claim_map()
            if claim_id not in claims:
                return _denied(patch, 'unknown_claim')
            claim = claims[claim_id]
            if model and claim.status == 'known':
                return _denied(patch, 'known_claim_immutable')
            text = _payload_text(payload, 'text', required=False) or claim.text
            status = payload.get('status', claim.status)
            if status not in CLAIM_STATUSES:
                return _denied(patch, 'unknown_claim_status')
            if model and status == 'known':
                return _denied(patch, 'model_may_not_add_known_claim')
            evidence_ids = tuple(payload.get('evidence_ids', claim.evidence_ids))
            if status == 'known' and (not evidence_ids
                                      or not _evidence_ok(state, evidence_ids, verifier)):
                return _denied(patch, 'known_claim_requires_host_evidence')
            if any(e not in {x.evidence_id for x in state.evidence} for e in evidence_ids):
                return _denied(patch, 'unknown_evidence_reference')
            revised = replace(claim, text=text, status=status, evidence_ids=evidence_ids)
            new_claims = tuple(revised if c.claim_id == claim_id else c for c in state.claims)
            updated = replace(state, claims=new_claims, revision=state.revision + 1)
            return PatchResult(True, 'applied', patch, updated)

        if patch.op == 'add_subgoal':
            subgoal_id = _payload_id(payload, 'subgoal_id')
            description = _payload_text(payload, 'description', required=True)
            deps = tuple(payload.get('depends_on') or ())
            existing = state.subgoal_map()
            if subgoal_id in existing:
                return _denied(patch, 'duplicate_subgoal_id')
            if any(dep not in existing for dep in deps):
                return _denied(patch, 'unknown_subgoal_dependency')
            if _creates_cycle({k: tuple(v.depends_on) for k, v in existing.items()},
                              subgoal_id, deps):
                return _denied(patch, 'subgoal_cycle_rejected')
            if len(state.subgoals) >= MAX_SUBGOALS:
                return _denied(patch, 'subgoal_limit_reached')
            subgoal = Subgoal(subgoal_id, description, 'open', deps)
            updated = replace(state, subgoals=(*state.subgoals, subgoal),
                              revision=state.revision + 1)
            return PatchResult(True, 'applied', patch, updated)

        if patch.op == 'add_obligation':
            obligation_id = _payload_id(payload, 'obligation_id')
            description = _payload_text(payload, 'description', required=True)
            if any(o.obligation_id == obligation_id for o in state.obligations):
                return _denied(patch, 'duplicate_obligation_id')
            if len(state.obligations) >= MAX_OBLIGATIONS:
                return _denied(patch, 'obligation_limit_reached')
            obligation = Obligation(obligation_id, description)
            updated = replace(state, obligations=(*state.obligations, obligation),
                              revision=state.revision + 1)
            return PatchResult(True, 'applied', patch, updated)

        if patch.op == 'satisfy_obligation':
            obligation_id = _payload_id(payload, 'obligation_id')
            receipt_digest = payload.get('receipt_digest')
            obligations = {o.obligation_id: o for o in state.obligations}
            if obligation_id not in obligations:
                return _denied(patch, 'unknown_obligation')
            obligation = obligations[obligation_id]
            if obligation.state == 'satisfied':
                return _denied(patch, 'obligation_already_satisfied')
            if (not isinstance(receipt_digest, str)
                    or not verifier.receipt_verified(receipt_digest)):
                return _denied(patch, 'obligation_requires_host_receipt')
            satisfied = replace(obligation, state='satisfied', receipt_digest=receipt_digest)
            new_obligations = tuple(satisfied if o.obligation_id == obligation_id else o
                                    for o in state.obligations)
            updated = replace(state, obligations=new_obligations, revision=state.revision + 1)
            return PatchResult(True, 'applied', patch, updated)

        if patch.op == 'add_evidence':
            if model:
                return _denied(patch, 'host_authority_required')
            evidence_id = _payload_id(payload, 'evidence_id')
            kind = payload.get('kind')
            digest = payload.get('digest')
            origin = _payload_text(payload, 'origin', required=True, limit=MAX_ID)
            if kind not in EVIDENCE_KINDS:
                return _denied(patch, 'unknown_evidence_kind')
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in '0123456789abcdef' for c in digest)):
                return _denied(patch, 'invalid_evidence_digest')
            if any(e.evidence_id == evidence_id for e in state.evidence):
                return _denied(patch, 'duplicate_evidence_id')
            if len(state.evidence) >= MAX_EVIDENCE:
                return _denied(patch, 'evidence_limit_reached')
            ref = EvidenceRef(evidence_id, kind, digest, origin)
            updated = replace(state, evidence=(*state.evidence, ref), revision=state.revision + 1)
            return PatchResult(True, 'applied', patch, updated)

        if patch.op == 'set_subgoal_state':
            subgoal_id = _payload_id(payload, 'subgoal_id')
            new_state = payload.get('state')
            subgoals = state.subgoal_map()
            if subgoal_id not in subgoals:
                return _denied(patch, 'unknown_subgoal')
            if new_state not in SUBGOAL_STATES:
                return _denied(patch, 'unknown_subgoal_state')
            subgoal = subgoals[subgoal_id]
            if new_state not in _SUBGOAL_TRANSITIONS[subgoal.state]:
                return _denied(patch, 'invalid_subgoal_transition')
            changed = replace(subgoal, state=new_state)
            new_subgoals = tuple(changed if s.subgoal_id == subgoal_id else s
                                 for s in state.subgoals)
            updated = replace(state, subgoals=new_subgoals, revision=state.revision + 1)
            return PatchResult(True, 'applied', patch, updated)

        if patch.op == 'note':
            text = _payload_text(payload, 'text', required=True, limit=MAX_NOTE)
            if len(state.notes) >= MAX_NOTES:
                return _denied(patch, 'note_limit_reached')
            updated = replace(state, notes=(*state.notes, text), revision=state.revision + 1)
            return PatchResult(True, 'applied', patch, updated)

        return _denied(patch, 'unknown_op')
    except ValueError as error:
        return _denied(patch, f'invalid_payload: {error}')


class JournalVerifier(PatchVerifier):
    """Verifies evidence and receipts against what the journal actually recorded."""

    def __init__(self, digests):
        self._digests = frozenset(digests)

    @property
    def digests(self) -> frozenset:
        return self._digests

    def host_evidence(self, ref: EvidenceRef) -> bool:
        return ref.kind in EVIDENCE_KINDS and ref.digest in self._digests

    def receipt_verified(self, receipt_digest) -> bool:
        return isinstance(receipt_digest, str) and receipt_digest in self._digests


__all__ = [
    'ACTION_STATES', 'AUTHORS', 'Budget', 'CLAIM_STATUSES', 'Claim', 'Constraint',
    'EVIDENCE_KINDS', 'EvidenceRef', 'ActionRecord', 'Obligation',
    'OBLIGATION_STATES', 'OUTCOME_STATUSES', 'OutcomeRecord', 'PATCH_OPS', 'PatchResult',
    'PatchVerifier', 'JournalVerifier', 'SUBGOAL_STATES', 'SuccessCondition', 'Subgoal',
    'TASK_FEATURE_COUNT', 'TASK_FEATURE_NAMES', 'TaskPatch', 'TOOL_KINDS', 'ExactTaskState',
    'apply_patch',
]

TASK_FEATURE_NAMES = _FEATURE_LAYOUT
