"""Typed action proposals, strict parsing, validation and outcome frames (E3 §4).

The model supplies raw text; the host assigns authority fields, parses exactly
one JSON object, validates schema/scope/tool/budget and records denials with the
raw bytes preserved. Malformed output never reaches a tool.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
import json
import re
import uuid

PROPOSAL_KINDS = ('retrieve', 'read', 'invoke_tool', 'verify', 'revise', 'answer',
                  'stop', 'ask')
OUTCOME_STATUSES = ('completed', 'denied', 'failed', 'pending_input')
MAX_PROPOSAL_BYTES = 16 * 1024
MAX_TEXT_ARGUMENT = 4000
MAX_PATH_ARGUMENT = 512

# Model-supplied keys that would usurp host authority; never accepted.
AUTHORITY_FIELDS = frozenset({
    'action_id', 'idempotency_key', 'actual', 'proposed_by', 'status', 'outcome',
    'receipt', 'spent', 'revision', 'verifier',
})
RESERVED_FIELDS = frozenset({'kind', 'goal_ref', 'subgoal_id', 'evidence_refs',
                             'budget_estimate', 'predicted_outcome',
                             'hypothetical'}) | AUTHORITY_FIELDS

# Strict per-kind argument schemas: key -> (type, required, max_length).
KIND_SCHEMAS = {
    'retrieve': {'query': (str, True, 512)},
    'read': {'path': (str, True, MAX_PATH_ARGUMENT)},
    'invoke_tool': {'tool': (str, True, 128), 'arguments': (dict, True, None)},
    'verify': {'condition_id': (str, False, 128)},
    'revise': {'note': (str, True, MAX_TEXT_ARGUMENT)},
    'answer': {'text': (str, True, MAX_TEXT_ARGUMENT)},
    'stop': {},
    'ask': {'question': (str, True, 1000)},
}
# invoke_tool arguments are validated against the per-tool schemas below.
TOOL_ARGUMENT_SCHEMAS = {
    'read_file': {'path': (str, True, MAX_PATH_ARGUMENT)},
    'write_file': {'path': (str, True, MAX_PATH_ARGUMENT), 'content': (str, True, 8 * 1024)},
    'run_test': {},
}


def validate_tool_arguments(tool: str, arguments) -> dict:
    """Shared strict tool-argument validation used by proposals and execution."""
    schema = TOOL_ARGUMENT_SCHEMAS.get(tool)
    if schema is None:
        raise ValueError(f'Unknown tool {tool!r}')
    if not isinstance(arguments, dict):
        raise ValueError('Tool arguments must be an object')
    unexpected = sorted(set(arguments) - set(schema))
    if unexpected:
        raise ValueError(f'Unexpected tool arguments: {unexpected}')
    missing = sorted(name for name, (kind, required, _) in schema.items()
                     if required and name not in arguments)
    if missing:
        raise ValueError(f'Missing tool arguments: {missing}')
    for name, (kind, _, limit) in schema.items():
        if name in arguments:
            value = arguments[name]
            if not isinstance(value, kind) or isinstance(value, bool):
                raise ValueError(f'Tool argument {name} must be {kind.__name__}')
            if len(value) > limit:
                raise ValueError(f'Tool argument {name} exceeds the {limit} character bound')
    return dict(arguments)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({str(k): _freeze(v) for k, v in sorted(value.items())})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(f'Unsupported proposal value type: {type(value).__name__}')


def _plain(value):
    if isinstance(value, MappingProxyType):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def _validate_kind_arguments(kind: str, args: dict) -> dict:
    schema = KIND_SCHEMAS[kind]
    unexpected = sorted(set(args) - set(schema))
    if unexpected:
        raise ValueError(f'unexpected_argument: {unexpected}')
    missing = sorted(name for name, (value_type, required, _) in schema.items()
                     if required and name not in args)
    if missing:
        raise ValueError(f'missing_argument: {missing}')
    for name, (value_type, _, limit) in schema.items():
        if name in args:
            value = args[name]
            if value_type is dict:
                if not isinstance(value, dict):
                    raise ValueError(f'invalid_argument_type: {name}')
                continue
            if not isinstance(value, value_type) or isinstance(value, bool):
                raise ValueError(f'invalid_argument_type: {name}')
            if not value or len(value) > limit:
                raise ValueError(f'invalid_argument_bound: {name}')
    return args


@dataclass(frozen=True)
class ActionProposal:
    """One model action request; identity/authority fields are host-assigned."""
    action_id: str
    idempotency_key: str
    kind: str
    args: dict
    goal_ref: str
    budget_estimate: dict = field(default_factory=dict)
    subgoal_id: str | None = None
    evidence_refs: tuple[str, ...] = ()
    predicted_outcome: str = ''
    hypothetical: bool = True
    proposed_by: str = 'model'

    def __post_init__(self):
        if not isinstance(self.action_id, str) or not self.action_id:
            raise ValueError('Host must assign action_id')
        if not isinstance(self.idempotency_key, str) or not self.idempotency_key:
            raise ValueError('Host must assign idempotency_key')
        if self.kind not in PROPOSAL_KINDS:
            raise ValueError(f'Unknown proposal kind: {self.kind!r}')
        if self.proposed_by != 'model':
            raise ValueError('Action proposals originate from the model author')
        if self.hypothetical is not True:
            raise ValueError('Predicted outcomes are hypothetical; only OutcomeFrame is actual')
        if not isinstance(self.goal_ref, str):
            raise ValueError('goal_ref binding required')
        object.__setattr__(self, 'args', MappingProxyType(
            _validate_kind_arguments(self.kind, dict(self.args))))
        estimate = dict(self.budget_estimate)
        for name in ('tool_calls', 'prompt_tokens', 'generated_tokens'):
            value = estimate.get(name, 0)
            if type(value) is not int or not 0 <= value <= 10 ** 9:
                raise ValueError(f'budget_estimate.{name} must be a nonnegative integer')
        object.__setattr__(self, 'budget_estimate', MappingProxyType(estimate))
        if self.subgoal_id is not None and (not isinstance(self.subgoal_id, str)
                                            or not self.subgoal_id):
            raise ValueError('subgoal_id must be a nonempty string when present')
        for ref in self.evidence_refs:
            if not isinstance(ref, str) or not ref:
                raise ValueError('evidence_refs must be nonempty strings')
        if len(self.predicted_outcome) > 200:
            raise ValueError('predicted_outcome exceeds 200 characters')

    def to_dict(self):
        return {'action_id': self.action_id, 'idempotency_key': self.idempotency_key,
                'kind': self.kind, 'args': _plain(self.args), 'goal_ref': self.goal_ref,
                'budget_estimate': _plain(self.budget_estimate), 'subgoal_id': self.subgoal_id,
                'evidence_refs': list(self.evidence_refs),
                'predicted_outcome': self.predicted_outcome,
                'hypothetical': True, 'proposed_by': self.proposed_by}


@dataclass(frozen=True)
class ParsedProposal:
    """Parse result: an authorized proposal or a typed denial with raw bytes."""
    status: str
    raw: bytes
    proposal: ActionProposal | None = None
    reason: str | None = None
    kind: str | None = None

    def __post_init__(self):
        if self.status not in ('ok', 'denied'):
            raise ValueError('ParsedProposal status must be ok or denied')
        if self.status == 'ok' and not isinstance(self.proposal, ActionProposal):
            raise ValueError('An ok parse requires an ActionProposal')
        if self.status == 'denied' and not (isinstance(self.reason, str) and self.reason):
            raise ValueError('A denial requires a typed reason')
        if not isinstance(self.raw, bytes):
            raise ValueError('Raw proposal bytes must be preserved')


# Native tokenizers append decoder stop markers after the emitted JSON. They are
# transport artifacts, not a second action; arbitrary prose remains rejected.
_TRAILING_SPECIAL_TOKENS = re.compile(
    r'(?:</s>|<\|im_end\|>|<\|endoftext\|>|<\|end\|>|<pad>)+\s*$')


def _extract_single_object(text: str):
    stripped = _TRAILING_SPECIAL_TOKENS.sub('', text).strip()
    fence = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', stripped, re.DOTALL)
    if fence is not None:
        stripped = fence.group(1).strip()
    decoder = json.JSONDecoder()
    index = 0
    value = None
    while index < len(stripped):
        while index < len(stripped) and stripped[index].isspace():
            index += 1
        if index >= len(stripped):
            break
        if value is not None:
            raise ValueError('multiple_json_objects')
        try:
            value, end = decoder.raw_decode(stripped, index)
        except json.JSONDecodeError:
            raise ValueError('malformed_json')
        index = end
    if value is None:
        raise ValueError('empty_proposal')
    return value


def parse_action_proposal(raw, allowed_tools, *, max_bytes: int = MAX_PROPOSAL_BYTES,
                          action_id: str | None = None,
                          idempotency_key: str | None = None) -> ParsedProposal:
    """Extract exactly one JSON action object; denials preserve the raw bytes."""
    if isinstance(raw, str):
        raw = raw.encode('utf-8', errors='strict')
    if not isinstance(raw, bytes):
        raise ValueError('raw proposal must be bytes or str')
    if len(raw) > max_bytes:
        return ParsedProposal('denied', raw, reason='oversize')
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        return ParsedProposal('denied', raw, reason='invalid_encoding')
    try:
        value = _extract_single_object(text)
    except ValueError as error:
        return ParsedProposal('denied', raw, reason=str(error))
    if not isinstance(value, dict):
        return ParsedProposal('denied', raw, reason='json_not_object')
    supplied_authority = sorted(AUTHORITY_FIELDS & set(value))
    if supplied_authority:
        return ParsedProposal('denied', raw, reason='authority_field_supplied')
    kind = value.get('kind')
    if not isinstance(kind, str) or kind not in PROPOSAL_KINDS:
        return ParsedProposal('denied', raw, reason='unknown_kind', kind=kind
                              if isinstance(kind, str) else None)
    if kind == 'read' and 'read_file' not in allowed_tools:
        return ParsedProposal('denied', raw, reason='tool_not_authorized', kind=kind)
    args = {key: item for key, item in value.items()
            if key not in RESERVED_FIELDS}
    estimate = value.get('budget_estimate', {})
    refs = value.get('evidence_refs', [])
    if not isinstance(estimate, dict) or not isinstance(refs, list) \
            or not all(isinstance(r, str) for r in refs):
        return ParsedProposal('denied', raw, reason='invalid_metadata', kind=kind)
    if kind == 'invoke_tool':
        tool = args.get('tool')
        if not isinstance(tool, str) or tool not in allowed_tools:
            return ParsedProposal('denied', raw, reason='tool_not_authorized', kind=kind)
        try:
            args['arguments'] = validate_tool_arguments(tool, args.get('arguments', {}))
        except ValueError as error:
            return ParsedProposal('denied', raw, reason=str(error), kind=kind)
    hypothetical = value.get('hypothetical', True)
    if hypothetical is not True:
        return ParsedProposal('denied', raw, reason='predicted_outcomes_are_hypothetical',
                              kind=kind)
    try:
        proposal = ActionProposal(
            action_id=action_id or uuid.uuid4().hex,
            idempotency_key=idempotency_key or uuid.uuid4().hex,
            kind=kind, args=args, goal_ref=value.get('goal_ref', ''),
            budget_estimate=estimate, subgoal_id=value.get('subgoal_id'),
            evidence_refs=tuple(refs),
            predicted_outcome=value.get('predicted_outcome', ''),
            hypothetical=True)
    except ValueError as error:
        return ParsedProposal('denied', raw, reason=str(error), kind=kind)
    return ParsedProposal('ok', raw, proposal=proposal, kind=kind)


@dataclass(frozen=True)
class OutcomeFrame:
    """The host's actual result of one action; model text is never a receipt."""
    outcome_id: str
    action_id: str
    status: str
    result: dict
    verifier: str
    spent: dict
    observation_ids: tuple[str, ...] = ()
    receipt: dict | None = None
    error: str | None = None
    actual: bool = True

    def __post_init__(self):
        if self.actual is not True:
            raise ValueError('OutcomeFrame is always actual')
        if self.status not in OUTCOME_STATUSES:
            raise ValueError(f'Unknown outcome status: {self.status!r}')
        if not isinstance(self.outcome_id, str) or not self.outcome_id:
            raise ValueError('outcome_id required')
        if not isinstance(self.action_id, str) or not self.action_id:
            raise ValueError('action_id required')
        if not isinstance(self.verifier, str) or not self.verifier:
            raise ValueError('OutcomeFrame requires its host verifier identity')
        if not isinstance(self.result, dict) or not isinstance(self.spent, dict):
            raise ValueError('OutcomeFrame result/spent must be objects')
        for name, value in self.spent.items():
            if type(value) is not int or value < 0:
                raise ValueError(f'Spent budget {name} must be a nonnegative integer')
        if self.error is not None and not isinstance(self.error, str):
            raise ValueError('OutcomeFrame error must be a string')

    def to_dict(self):
        return {'outcome_id': self.outcome_id, 'action_id': self.action_id,
                'status': self.status, 'result': _plain(self.result),
                'verifier': self.verifier, 'spent': dict(self.spent),
                'observation_ids': list(self.observation_ids),
                'receipt': _plain(self.receipt) if self.receipt is not None else None,
                'error': self.error, 'actual': True}

    def to_record(self, revision: int):
        from .task_graph import OutcomeRecord
        return OutcomeRecord(
            outcome_id=self.outcome_id, action_id=self.action_id, status=self.status,
            actual=True, result=dict(self.result), revision=revision,
            observation_ids=self.observation_ids,
            receipt=dict(self.receipt) if self.receipt is not None else None,
            error=self.error, spent=dict(self.spent))


@dataclass(frozen=True)
class ProposalValidation:
    """Typed validation result with the host-computed execution estimate."""
    ok: bool
    reason: str | None = None
    tool_calls: int = 0
    tool: str | None = None


def validate_proposal(proposal: ActionProposal, state, registry, budget) -> ProposalValidation:
    """Schema/scope/tool authorization and budget checks before any execution."""
    if not isinstance(proposal, ActionProposal):
        raise ValueError('validate_proposal requires an ActionProposal')
    if proposal.goal_ref not in ('', state.task_id):
        return ProposalValidation(False, 'goal_binding_mismatch')
    if proposal.subgoal_id is not None and proposal.subgoal_id not in state.subgoal_map():
        return ProposalValidation(False, 'unknown_subgoal')
    known_evidence = {e.evidence_id for e in state.evidence}
    if any(ref not in known_evidence for ref in proposal.evidence_refs):
        return ProposalValidation(False, 'unknown_evidence_reference')
    tool = None
    tool_calls = 0
    if proposal.kind == 'invoke_tool':
        tool = proposal.args.get('tool')
        if not isinstance(tool, str) or tool not in state.allowed_tools:
            return ProposalValidation(False, 'tool_not_authorized')
        if registry is None or tool not in registry.names():
            return ProposalValidation(False, 'tool_not_available')
        try:
            validate_tool_arguments(tool, proposal.args.get('arguments', {}))
        except ValueError as error:
            return ProposalValidation(False, f'invalid_tool_arguments: {error}')
        tool_calls = 1
    elif proposal.kind == 'read':
        if 'read_file' not in state.allowed_tools:
            return ProposalValidation(False, 'tool_not_authorized')
        tool_calls = 1
    elif proposal.kind in ('verify', 'stop', 'answer'):
        if not state.success_conditions:
            return ProposalValidation(False, 'no_success_condition')
        condition_id = proposal.args.get('condition_id')
        if condition_id is not None and condition_id not in {
                c.condition_id for c in state.success_conditions}:
            return ProposalValidation(False, 'unknown_success_condition')
        tool_calls = 1 if proposal.kind == 'verify' else 0
    if not budget.allows(tool_calls=tool_calls):
        return ProposalValidation(False, 'budget_exhausted', tool_calls=tool_calls, tool=tool)
    return ProposalValidation(True, tool_calls=tool_calls, tool=tool)


__all__ = ['AUTHORITY_FIELDS', 'ActionProposal', 'KIND_SCHEMAS', 'MAX_PROPOSAL_BYTES',
           'OUTCOME_STATUSES', 'OutcomeFrame', 'PROPOSAL_KINDS', 'ParsedProposal',
           'ProposalValidation', 'TOOL_ARGUMENT_SCHEMAS', 'parse_action_proposal',
           'validate_proposal', 'validate_tool_arguments']
