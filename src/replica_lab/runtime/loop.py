"""TaskRuntime: one exact decision at a time, journaled end to end (E3 §3-§4).

Each decision rebuilds the exact state from the journal, builds a bounded prompt
from actual observations only, computes the 24 task features, calls an injectable
proposer, journals the decision, parses and validates the proposal, executes one
authorized action with real tools, journals the outcome and receipts, applies
host patches, and persists the next workspace detached. Only a passing
independent ``verify_success`` check satisfies a success condition; false claims
become failed outcomes and the loop continues within budget.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import time
from typing import Any

import torch

from ..architecture.contracts import CoreState, fingerprint
from .actions import (MAX_PROPOSAL_BYTES, ActionProposal, OutcomeFrame, ParsedProposal,
                      ProposalValidation, parse_action_proposal, validate_proposal)
from .journal import PendingIntent, TaskJournal
from .task_graph import (ActionRecord, Budget, ExactTaskState, JournalVerifier,
                         OutcomeRecord, TaskPatch, apply_patch)
from .tools import sha256_bytes
from .workspace_store import WorkspaceStore, detach_to_cpu

MODES = ('full', 'core_disabled', 'workspace_zero', 'workspace_shuffled', 'ff_core',
         'extra_inference')
MAX_HISTORY_ITEMS = 6
MAX_OBSERVATION_EXCERPT = 600
MAX_RAW_TEXT_BYTES = 16 * 1024
DEFAULT_WORKSPACE_WIDTH = 256


class BudgetExhausted(RuntimeError):
    """Raised when a decision is attempted beyond the declared budget."""


@dataclass(frozen=True)
class ProposalRequest:
    """Everything a proposer may see; never future observations or expected values."""
    prompt_text: str
    task_features: tuple
    previous_workspace: CoreState | None
    mode: str
    prompt_token_budget: int
    generated_token_budget: int
    task_digest: str
    revision: int
    fast_state: Any | None = None

    def __post_init__(self):
        if not isinstance(self.prompt_text, str) or not self.prompt_text:
            raise ValueError('ProposalRequest requires prompt_text')
        if len(self.task_features) != 24:
            raise ValueError('ProposalRequest carries the 24-float task feature vector')
        for value in self.task_features:
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError('Task features must be bounded to [0, 1]')
        if self.previous_workspace is not None and not isinstance(self.previous_workspace, CoreState):
            raise ValueError('previous_workspace must be a CoreState or None')
        if self.mode not in MODES:
            raise ValueError(f'Unknown runtime mode: {self.mode!r}')
        if type(self.prompt_token_budget) is not int or self.prompt_token_budget <= 0 \
                or type(self.generated_token_budget) is not int or self.generated_token_budget <= 0:
            raise ValueError('Token budgets must be positive integers')
        if not isinstance(self.task_digest, str) or not self.task_digest:
            raise ValueError('task_digest binding required')
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError('revision must be a nonnegative integer')


@dataclass(frozen=True)
class ProposalOutput:
    """One proposer response; workspace ownership passes to the host here."""
    raw_text: str
    prompt_tokens: int
    generated_token_ids: tuple
    generated_tokens: int
    frame_digest: str
    workspace: CoreState | None
    workspace_stamp: str | None
    wall_seconds: float
    notes: tuple = ()

    def __post_init__(self):
        if not isinstance(self.raw_text, str):
            raise ValueError('raw_text must be a string')
        if len(self.raw_text.encode('utf-8')) > MAX_RAW_TEXT_BYTES * 4:
            raise ValueError('Raw proposal text exceeds the bounded size')
        if type(self.prompt_tokens) is not int or self.prompt_tokens < 0:
            raise ValueError('prompt_tokens must be a nonnegative integer')
        if type(self.generated_tokens) is not int or self.generated_tokens < 0:
            raise ValueError('generated_tokens must be a nonnegative integer')
        object.__setattr__(self, 'generated_token_ids', tuple(self.generated_token_ids))
        if self.generated_tokens != len(self.generated_token_ids):
            raise ValueError('generated_tokens must match generated_token_ids')
        if any(type(t) is not int or t < 0 for t in self.generated_token_ids):
            raise ValueError('Generated token ids must be nonnegative integers')
        if not isinstance(self.frame_digest, str) or not self.frame_digest:
            raise ValueError('frame_digest required')
        if self.workspace is not None and not isinstance(self.workspace, CoreState):
            raise ValueError('workspace must be a CoreState or None')
        if self.workspace is None and self.workspace_stamp is not None:
            raise ValueError('workspace_stamp requires an owned workspace')
        if not isinstance(self.wall_seconds, (int, float)) \
                or isinstance(self.wall_seconds, bool) or self.wall_seconds < 0:
            raise ValueError('wall_seconds must be a nonnegative number')
        object.__setattr__(self, 'notes', tuple(self.notes))


@dataclass(frozen=True)
class DecisionTrace:
    """One journaled decision; persisted verbatim in the session report."""
    sequence: int
    prompt_text: str
    prompt_sha256: str
    prompt_tokens: int
    task_features: tuple
    state_digest: str
    revision: int
    raw_text: str
    generated_token_ids: tuple
    generated_tokens: int
    frame_digest: str
    workspace_stamp: str | None
    mode: str
    action: dict | None
    outcome: dict
    receipts: list
    success_condition_met: bool
    success_claimed: bool
    decision_wall_seconds: float
    workspace_pool: tuple | None = None
    notes: tuple = ()


def _outcome_public(frame: OutcomeFrame) -> dict:
    return {'outcome_id': frame.outcome_id, 'action_id': frame.action_id,
            'status': frame.status, 'result': dict(frame.result), 'error': frame.error,
            'spent': dict(frame.spent), 'verifier': frame.verifier,
            'observation_ids': list(frame.observation_ids)}


class TaskRuntime:
    """Host orchestration; authority stays here, generation stays injectable."""

    def __init__(self, tools, journal: TaskJournal, *, workspace_store: WorkspaceStore | None = None,
                 mode: str = 'full', retriever=None, budget: Budget | None = None,
                 max_prompt_tokens: int = 2048, generated_token_budget: int = 128,
                 max_proposal_bytes: int = MAX_PROPOSAL_BYTES,
                 workspace_width: int = DEFAULT_WORKSPACE_WIDTH,
                 shuffled_workspace: CoreState | None = None,
                 identity=None, fast_memory=None):
        if mode not in MODES:
            raise ValueError(f'Unknown runtime mode: {mode!r}')
        self.tools = tools
        self.journal = journal
        self.workspace_store = workspace_store
        self.mode = mode
        self.retriever = retriever
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.generated_token_budget = int(generated_token_budget)
        self.max_proposal_bytes = int(max_proposal_bytes)
        self.workspace_width = int(workspace_width)
        if self.max_prompt_tokens <= 0 or self.generated_token_budget <= 0:
            raise ValueError('Per-decision token budgets must be positive integers')
        if self.max_proposal_bytes <= 0 or self.workspace_width <= 0:
            raise ValueError('Proposal and workspace bounds must be positive integers')
        self.shuffled_workspace = shuffled_workspace
        if identity is not None:
            from ..replica.identity import IdentityContext
            if not isinstance(identity, IdentityContext):
                raise ValueError('identity must be an IdentityContext or None')
        self.identity = identity
        if fast_memory is not None and not hasattr(fast_memory, 'observe'):
            raise ValueError('fast_memory must expose observe() and snapshot()')
        self.fast_memory = fast_memory
        self.fast_state = fast_memory.snapshot() if fast_memory is not None else None
        self.fast_observed = 0
        self.fast_error = None
        self.state: ExactTaskState | None = None
        self.previous_workspace: CoreState | None = None
        self._verifier: JournalVerifier | None = None
        self._sequence = 0
        self._budget_exhausted = False
        self.last_workspace_discard: str | None = None

    # ---- public read access (no tool execution on load) ---------------------
    @property
    def verifier(self) -> JournalVerifier:
        if self._verifier is None:
            self._load()
        return self._verifier

    def initialize(self, state: ExactTaskState) -> None:
        """Host-only first write of the exact state into a fresh journal."""
        if not isinstance(state, ExactTaskState):
            raise ValueError('initialize requires an ExactTaskState')
        if self.journal.initialize_done():
            raise RuntimeError('Journal already initialized')
        self.journal.initialize(state)
        self._load()

    def correct(self, author: str = 'host', *, note=None, constraint=None,
                claim=None, evidence=None) -> ExactTaskState:
        """Host-only correction: bumps the exact revision and reassigns lineage.

        A correction may add a constraint, record a host note, admit host
        evidence and/or revise a claim (including a ``known`` claim, which model
        authors may never revise). The persisted workspace is bound to the
        pre-correction revision, so the next decision rejects it as
        ``stale_revision`` and rebuilds.
        """
        if author != 'host':
            raise ValueError('Corrections are host-authority only')
        if note is None and constraint is None and claim is None and evidence is None:
            raise ValueError('A correction requires a note, constraint, claim or evidence')
        self._load()
        evidence_items = []
        if evidence is not None:
            if isinstance(evidence, dict):
                evidence_items = [evidence]
            elif isinstance(evidence, (list, tuple)):
                evidence_items = list(evidence)
            else:
                raise ValueError('Correction evidence must be an object or list of objects')
        for payload in evidence_items:
            if not isinstance(payload, dict):
                raise ValueError('Correction evidence entries must be objects')
            result = apply_patch(self.state, TaskPatch('add_evidence', 'host', payload),
                                 self.verifier)
            self._journal_patch(result)
            if not result.applied or result.state is None:
                raise ValueError(f'Correction evidence denied: {result.reason}')
            self.state = result.state
            self._load()
        if claim is not None:
            if not isinstance(claim, dict):
                raise ValueError('Correction claim must be an object')
            result = apply_patch(self.state, TaskPatch('revise_claim', 'host', claim),
                                 self.verifier)
            self._journal_patch(result)
            if not result.applied or result.state is None:
                raise ValueError(f'Correction claim denied: {result.reason}')
            self.state = result.state
            self._load()
        if note is not None or constraint is not None:
            self.journal.correct(note=note, constraint=constraint)
        self.previous_workspace = None
        self.last_workspace_discard = 'host_correction'
        self._load()
        return self.state

    def _load(self) -> None:
        snapshot = self.journal.load()
        self.state = snapshot.state
        self._verifier = snapshot.verifier
        self._sequence = sum(1 for event in self.journal.events() if event.kind == 'decision')

    def recover(self) -> int:
        """Reconcile pending intents exactly once before new decisions."""
        self._load()
        pending = self.journal.pending_intents()
        if not pending:
            return 0
        self.journal.recover(self._make_reconciler())
        self._load()
        return len(pending)

    # ---- workspace persistence ---------------------------------------------
    def _load_previous_workspace(self, state: ExactTaskState):
        if self.mode == 'workspace_shuffled':
            if self.shuffled_workspace is None:
                return None, 'shuffled_workspace_not_supplied'
            return detach_to_cpu(self.shuffled_workspace), None
        from_store = self.workspace_store is not None
        if from_store:
            source = self.workspace_store.load()
            if source is None:
                return None, 'missing_or_incompatible'
        else:
            source = self.previous_workspace
        if source is None:
            return None, 'no_previous_workspace'
        core = detach_to_cpu(source)
        if core.task_digest != state.digest or core.revision != state.revision:
            if from_store:
                # A correction or any intervening revision invalidates the
                # persisted payload; rebuild rather than reuse stale lineage.
                self.workspace_store.clear()
            return None, 'stale_revision'
        expected_implementation = 'feedforward' if self.mode == 'ff_core' else 'recurrent'
        if core.implementation != expected_implementation:
            if from_store:
                self.workspace_store.clear()
            return None, 'incompatible_implementation'
        return core, None

    # ---- prompt -------------------------------------------------------------
    def _build_prompt(self, state: ExactTaskState) -> str:
        lines = [f'TASK {state.task_id} (revision {state.revision})',
                 f'AUTHORITATIVE GOAL (immutable): {state.goal}',
                 'SUCCESS CONDITIONS (independent host check only):']
        lines += [f'- [{c.condition_id}] {c.description}' for c in state.success_conditions]
        if state.constraints:
            lines.append('CONSTRAINTS:')
            lines += [f'- {c.text}' for c in state.constraints]
        if self.identity is not None:
            from ..replica.identity import render_identity_context
            identity_text = render_identity_context(self.identity)
            if identity_text:
                lines.append(identity_text)
        if state.subgoals:
            lines.append('SUBGOALS:')
            lines += [f'- [{s.state}] {s.subgoal_id}: {s.description}'
                      for s in state.subgoals[-MAX_HISTORY_ITEMS:]]
        if state.claims:
            lines.append('CLAIMS:')
            lines += [f'- [{c.status}] {c.claim_id}: {c.text}'
                      for c in state.claims[-8:]]
        if state.notes:
            lines.append('HOST NOTES (authoritative corrections):')
            lines += [f'- {note}' for note in state.notes[-4:]]
        open_obligations = [o for o in state.obligations if o.state == 'unresolved']
        if open_obligations:
            lines.append('OPEN OBLIGATIONS:')
            lines += [f'- {o.obligation_id}: {o.description}'
                      for o in open_obligations[-MAX_HISTORY_ITEMS:]]
        if state.actions:
            lines.append('ACTION/OBSERVATION HISTORY (most recent last):')
            outcomes = {outcome.action_id: outcome for outcome in state.outcomes}
            for record in state.actions[-MAX_HISTORY_ITEMS:]:
                outcome = outcomes.get(record.action_id)
                status = outcome.status if outcome is not None else record.state
                detail = ''
                if outcome is not None:
                    detail = outcome.error or ''
                    if not detail and isinstance(outcome.result, dict):
                        detail = str(outcome.result.get('error')
                                     or outcome.result.get('reason') or '')
                label = record.kind
                if record.tool:
                    label += f'({record.tool})'
                lines.append(f'- {label} -> {status} {detail}'[:MAX_OBSERVATION_EXCERPT])
        if state.outcomes:
            last = state.outcomes[-1]
            lines.append(f'LAST OUTCOME: {last.status} {last.error or ""}'.rstrip())
            lines.append(f'LAST OBSERVATION (excerpt, status {last.status}):')
            lines.append(_observation_excerpt(last))
        lines.append('ALLOWED TOOLS: ' + (', '.join(state.allowed_tools) or 'none'))

        instructions = ['Respond with exactly one JSON object and nothing else.',
                        'Kinds and arguments:',
                        '{"kind": "retrieve", "query": "..."}']
        if 'read_file' in state.allowed_tools:
            instructions.append('{"kind": "read", "path": "workspace/..."}')
        if 'read_file' in state.allowed_tools:
            instructions.append('{"kind": "invoke_tool", "tool": "read_file",'
                                ' "arguments": {"path": "workspace/..."}}')
        if 'write_file' in state.allowed_tools:
            instructions.append('{"kind": "invoke_tool", "tool": "write_file",'
                                ' "arguments": {"path": "workspace/...", "content": "..."}}')
        if 'set_constant' in state.allowed_tools:
            instructions.append('{"kind": "invoke_tool", "tool": "set_constant",'
                                ' "arguments": {"name": "CONSTANT_NAME", "value": 0}}')
        if 'run_test' in state.allowed_tools:
            instructions.append('{"kind": "invoke_tool", "tool": "run_test",'
                                ' "arguments": {}}')
        instructions += ['{"kind": "verify"}',
                         '{"kind": "revise", "note": "..."}',
                         '{"kind": "answer", "text": "..."}',
                         '{"kind": "stop"}',
                         '{"kind": "ask", "question": "..."}',
                         'Predictions are hypothetical; success is decided only by the '
                         'host check after verify/stop/answer.']
        tail = '\n'.join(instructions)
        budget_chars = self.max_prompt_tokens * 4
        if len(tail) >= budget_chars:
            raise ValueError('Per-decision prompt budget is too small for the action schema')
        body = '\n'.join(lines)
        allowance = budget_chars - len(tail) - 1
        if len(body) > allowance:
            body = body[:allowance]
        return body + '\n' + tail

    # ---- one decision -------------------------------------------------------
    def decide(self, proposer) -> DecisionTrace:
        started = time.monotonic()
        self._load()
        if self.state.paused():
            raise RuntimeError('Task is paused on pending_input; resume explicitly')
        if not self.state.budget.allows(decisions=1):
            self.journal.append('budget', {'denied': 'decision', 'reason': 'budget_exhausted'},
                                idempotency_key='budget:' + self.state.digest)
            raise BudgetExhausted('Decision budget exhausted')
        if self.journal.pending_intents():
            self.recover()
            self._load()
            if self.state.paused():
                raise RuntimeError('Recovered task is paused on pending_input')
        state = self.state
        previous_workspace, discard_reason = self._load_previous_workspace(state)
        self.last_workspace_discard = discard_reason
        features = state.features()
        prompt = self._build_prompt(state)
        request = ProposalRequest(prompt, features, previous_workspace, self.mode,
                                  self.max_prompt_tokens, self.generated_token_budget,
                                  state.digest, state.revision, self.fast_state)
        output = proposer(request)
        if not isinstance(output, ProposalOutput):
            raise ValueError('Proposer must return a ProposalOutput')
        decision_seq = self._sequence + 1
        prompt_sha = sha256_bytes(prompt.encode('utf-8'))
        over_per_decision = (output.prompt_tokens > self.max_prompt_tokens
                             or output.generated_tokens > self.generated_token_budget)
        over_cumulative = not state.budget.allows(
            decisions=1, prompt_tokens=output.prompt_tokens,
            generated_tokens=output.generated_tokens)
        if over_per_decision or over_cumulative:
            reason = ('per_decision_token_budget_exceeded' if over_per_decision
                      else 'cumulative_token_budget_exceeded')
            return self._finish_budget_exceeded(
                decision_seq, state, output, prompt, prompt_sha, features, started, reason)
        raw_sha = sha256_bytes(output.raw_text.encode('utf-8'))
        decision_event = self.journal.append('decision', {
            'sequence': decision_seq, 'mode': self.mode,
            'prompt_sha256': prompt_sha, 'prompt_tokens': output.prompt_tokens,
            'generated_token_ids': list(output.generated_token_ids),
            'generated_tokens': output.generated_tokens,
            'frame_digest': output.frame_digest,
            'workspace_stamp': output.workspace_stamp,
            'action_raw_sha256': raw_sha, 'action_raw_text': output.raw_text,
        }, idempotency_key=f'decision:{state.digest}:{decision_seq}')
        self.state = state = state.spend(decisions=1, prompt_tokens=output.prompt_tokens,
                                         generated_tokens=output.generated_tokens)
        action_id = fingerprint(('action', decision_event.event_id))
        idempotency_key = fingerprint(('intent', state.digest, decision_event.event_id))
        parsed = parse_action_proposal(
            output.raw_text, allowed_tools=state.allowed_tools,
            max_bytes=self.max_proposal_bytes, action_id=action_id,
            idempotency_key=idempotency_key)
        if parsed.status != 'ok':
            return self._finish_denied(decision_seq, output, prompt, prompt_sha,
                                       features, started, parsed, action_id, idempotency_key,
                                       f'parse_denied: {parsed.reason}')
        validation = validate_proposal(parsed.proposal, state,
                                       self.tools.registry(state.allowed_tools)
                                       if self.tools is not None else None, state.budget)
        if not validation.ok:
            return self._finish_denied(decision_seq, output, prompt, prompt_sha,
                                       features, started, parsed, action_id, idempotency_key,
                                       f'validation_denied: {validation.reason}')
        frame = self._journal_intent_and_execute(parsed.proposal, validation, state,
                                                 idempotency_key)
        self._observe_fast_memory(frame)
        self._load()
        met = bool(self.state.satisfied_conditions(self.verifier))
        claimed = parsed.proposal.kind in ('stop', 'answer')
        self._persist_workspace(output)
        return self._trace(
            decision_seq, output, prompt, prompt_sha, features, started, parsed, frame,
            met=met, claimed=claimed,
            action=parsed.proposal.to_dict(), receipts=_receipts(frame))

    def _finish_budget_exceeded(self, decision_seq, state, output, prompt, prompt_sha,
                                features, started, reason) -> DecisionTrace:
        """Record generation beyond the declared cap as a denied outcome.

        The decision slot is charged (the model did run), the actual generated
        usage stays in the decision/budget events, and the loop stops cleanly
        with ``budget_exhausted`` instead of raising mid-decision.
        """
        raw_sha = sha256_bytes(output.raw_text.encode('utf-8'))
        decision_event = self.journal.append('decision', {
            'sequence': decision_seq, 'mode': self.mode,
            'prompt_sha256': prompt_sha, 'prompt_tokens': output.prompt_tokens,
            'generated_token_ids': list(output.generated_token_ids),
            'generated_tokens': output.generated_tokens,
            'frame_digest': output.frame_digest,
            'workspace_stamp': output.workspace_stamp,
            'action_raw_sha256': raw_sha, 'action_raw_text': output.raw_text,
            'budget_exceeded': reason,
        }, idempotency_key=f'decision:{state.digest}:{decision_seq}')
        self.state = state = state.spend(decisions=1)
        self.journal.append('budget', {
            'denied': 'decision', 'reason': reason,
            'prompt_tokens': output.prompt_tokens,
            'generated_tokens': output.generated_tokens, 'sequence': decision_seq,
        }, idempotency_key=f'budget:{decision_seq}')
        action_id = fingerprint(('action', decision_event.event_id))
        idempotency_key = fingerprint(('intent', state.digest, decision_event.event_id))
        parsed = parse_action_proposal(
            output.raw_text, allowed_tools=state.allowed_tools,
            max_bytes=self.max_proposal_bytes, action_id=action_id,
            idempotency_key=idempotency_key)
        self._budget_exhausted = True
        return self._finish_denied(
            decision_seq, output, prompt, prompt_sha, features, started, parsed,
            action_id, idempotency_key, f'budget_exhausted: {reason}',
            notes=('budget_exhausted',))

    def _observe_fast_memory(self, frame: OutcomeFrame) -> None:
        """Feed one actual host observation into the optional fast state."""
        if self.fast_memory is None or frame.verifier != 'host_tools':
            return
        if frame.status not in ('completed', 'failed'):
            return
        result = dict(frame.result) if isinstance(frame.result, dict) else {}
        if result.get('kind') in ('verify_success', 'stop', 'answer'):
            return
        text = ''
        for key in ('content', 'excerpt', 'stdout_excerpt', 'stderr_excerpt', 'text'):
            value = result.get(key)
            if isinstance(value, str) and value:
                text = value
                break
        source_hash = result.get('sha256') if isinstance(result.get('sha256'), str) else None
        receipt = dict(frame.receipt or {})
        if not text:
            for key in ('stdout_excerpt', 'stderr_excerpt'):
                value = receipt.get(key)
                if isinstance(value, str) and value:
                    text = value
                    break
        if not text:
            return
        if source_hash is None:
            source_hash = receipt.get('receipt_sha256')
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            source_hash = hashlib.sha256(
                json.dumps(result, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        try:
            receipt_value = self.fast_memory.observe(
                frame.action_id, text, source_hash=source_hash)
            if receipt_value is not None:
                self.fast_observed += 1
                self.fast_state = self.fast_memory.snapshot()
        except Exception as error:
            self.fast_error = f'{type(error).__name__}: {error}'

    def _finish_denied(self, decision_seq, output, prompt, prompt_sha, features,
                       started, parsed: ParsedProposal, action_id, idempotency_key,
                       reason, notes=()) -> DecisionTrace:
        kind = parsed.kind or 'invalid'
        self.journal.append('action_intent', {
            'action_id': action_id, 'kind': kind, 'tool': None, 'args': {},
            'authorized': False, 'idempotency_key': idempotency_key, 'reason': reason,
        }, idempotency_key='intent:' + idempotency_key)
        record = ActionRecord(action_id=action_id, kind=kind, args={}, state='proposed',
                              revision=self.state.revision, idempotency_key=idempotency_key)
        self.state = state = self.state.with_action(record)
        frame = OutcomeFrame(
            outcome_id='out-' + action_id[:16], action_id=action_id, status='denied',
            result={'reason': reason, 'raw_sha256': sha256_bytes(parsed.raw),
                    'raw_bytes': len(parsed.raw)},
            verifier='host_runtime', spent={})
        self._journal_outcome(frame, tool_calls=0)
        self._load()
        self._persist_workspace(output)
        return self._trace(
            decision_seq, output, prompt, prompt_sha, features, started, parsed, frame,
            met=False, claimed=False,
            action=parsed.proposal.to_dict() if parsed.proposal is not None else None,
            receipts=[], notes=tuple(notes))

    def _trace(self, decision_seq, output, prompt, prompt_sha, features, started, parsed,
               frame, *, met, claimed, action, receipts, notes=()) -> DecisionTrace:
        extra_notes = list(output.notes)
        extra_notes.extend(notes)
        if self.last_workspace_discard:
            extra_notes.append(f'workspace_discarded: {self.last_workspace_discard}')
        return DecisionTrace(
            sequence=decision_seq, prompt_text=prompt, prompt_sha256=prompt_sha,
            prompt_tokens=output.prompt_tokens, task_features=features,
            state_digest=self.state.digest, revision=self.state.revision,
            raw_text=output.raw_text, generated_token_ids=output.generated_token_ids,
            generated_tokens=output.generated_tokens, frame_digest=output.frame_digest,
            workspace_stamp=output.workspace_stamp, mode=self.mode, action=action,
            outcome=_outcome_public(frame), receipts=list(receipts),
            success_condition_met=met, success_claimed=claimed,
            decision_wall_seconds=time.monotonic() - started,
            workspace_pool=self._pool(output.workspace), notes=tuple(extra_notes))

    def _journal_intent_and_execute(self, proposal: ActionProposal,
                                    validation: ProposalValidation, state: ExactTaskState,
                                    idempotency_key: str) -> OutcomeFrame:
        args = dict(proposal.args)
        payload = {'action_id': proposal.action_id, 'kind': proposal.kind,
                   'tool': validation.tool, 'args': args, 'authorized': True,
                   'idempotency_key': idempotency_key,
                   'subgoal_id': proposal.subgoal_id,
                   'budget_estimate': dict(proposal.budget_estimate)}
        if proposal.kind == 'invoke_tool' and validation.tool == 'write_file':
            content = args.get('arguments', {}).get('content', '')
            payload['content'] = content
            payload['content_sha256'] = sha256_bytes(content.encode('utf-8'))
        self.journal.append('action_intent', payload,
                            idempotency_key='intent:' + idempotency_key)
        record = ActionRecord(action_id=proposal.action_id, kind=proposal.kind, args=args,
                              state='attempted', revision=state.revision,
                              idempotency_key=idempotency_key, tool=validation.tool,
                              subgoal_id=proposal.subgoal_id)
        self.state = state = self.state.with_action(record)
        frame = self._execute(proposal, validation)
        self._journal_outcome(frame, tool_calls=validation.tool_calls)
        if proposal.kind == 'revise':
            # Exact revision order: intent -> outcome -> patch. The preview in
            # _execute only decided the reported status; the patch commits here.
            patch = TaskPatch('note', 'model', {'text': args['note']})
            result = apply_patch(self.state, patch, self.verifier)
            self._journal_patch(result)
            if result.applied:
                self.state = result.state
        self._journal_observation_evidence(frame)
        return frame

    def _execute(self, proposal: ActionProposal, validation: ProposalValidation) -> OutcomeFrame:
        kind = proposal.kind
        args = dict(proposal.args)
        tool_calls = validation.tool_calls
        spent = {'tool_calls': tool_calls}
        try:
            if kind == 'read':
                result = self.tools.read_file({'path': args['path']})
            elif kind == 'invoke_tool':
                registry = self.tools.registry(self.state.allowed_tools)
                result = registry.call(validation.tool, args.get('arguments', {}))
            elif kind == 'retrieve':
                if self.retriever is None:
                    result = {'status': 'denied', 'kind': 'retrieve',
                              'error': 'retrieval_unavailable'}
                else:
                    observation = str(self.retriever(args['query']))[:MAX_OBSERVATION_EXCERPT]
                    result = {'status': 'ok', 'kind': 'retrieve', 'content': observation,
                              'sha256': sha256_bytes(observation.encode('utf-8'))}
            elif kind == 'verify':
                condition_id = args.get('condition_id')
                condition_ids = ([condition_id] if condition_id
                                 else [c.condition_id for c in self.state.success_conditions])
                verdict = self.tools.verify_success(condition_ids)
                result = {'status': 'ok' if verdict['passed'] else 'error', 'kind': 'verify',
                          'passed': verdict['passed'], 'receipt': verdict['receipt']}
            elif kind == 'revise':
                # Pure preview; the committed patch follows the outcome event so
                # every action has the same intent -> outcome -> patch order.
                patch = TaskPatch('note', 'model', {'text': args['note']})
                outcome = apply_patch(self.state, patch, self.verifier)
                result = {'status': 'ok' if outcome.applied else 'denied', 'kind': 'revise',
                          'applied': outcome.applied, 'reason': outcome.reason}
            elif kind in ('stop', 'answer'):
                condition_ids = [c.condition_id for c in self.state.success_conditions]
                verdict = self.tools.verify_success(condition_ids)
                met = verdict['passed']
                result = {'status': 'ok' if met else 'error', 'kind': kind,
                          'claimed': True, 'verified': met, 'receipt': verdict['receipt']}
                if kind == 'answer':
                    result['text'] = args['text']
                frame = self._frame(proposal, result,
                                    'completed' if met else 'failed',
                                    None if met else 'success_condition_not_met', spent)
                return frame
            elif kind == 'ask':
                result = {'status': 'ok', 'kind': 'ask', 'question': args['question']}
            else:  # pragma: no cover - kinds are closed
                result = {'status': 'denied', 'kind': kind, 'error': 'unknown_kind'}
        except Exception as error:  # real tool failures become failed outcomes
            result = {'status': 'error', 'kind': kind, 'error': f'execution_error: {error}'}
        status = {'ok': 'completed', 'error': 'failed', 'denied': 'denied'}.get(
            result.get('status'), 'failed')
        if kind == 'ask' and status == 'completed':
            status = 'pending_input'
        return self._frame(proposal, result, status, result.get('error'), spent)

    def _frame(self, proposal: ActionProposal, result: dict, status: str, error, spent) -> OutcomeFrame:
        observation = fingerprint(result)
        receipt = result.get('receipt')
        return OutcomeFrame(
            outcome_id='out-' + proposal.action_id[:16], action_id=proposal.action_id,
            status=status, result=result, verifier='host_tools', spent=spent,
            observation_ids=(observation,), receipt=receipt, error=error)

    def _journal_outcome(self, frame: OutcomeFrame, *, tool_calls: int) -> None:
        record = frame.to_record(self.state.revision)
        self.journal.append('action_outcome',
                            {'outcome': record.to_dict(), 'spent': {'tool_calls': tool_calls}},
                            idempotency_key='outcome:' + frame.action_id)
        self.state = self.state.with_outcome(record)
        if tool_calls:
            self.state = self.state.spend(tool_calls=tool_calls)

    def _journal_patch(self, result) -> None:
        self.journal.append('patch', {'patch': result.patch.to_dict(),
                                      'result': {'applied': result.applied,
                                                 'reason': result.reason}})

    def _journal_observation_evidence(self, frame: OutcomeFrame) -> None:
        """Host evidence for real observations; generated text is never evidence."""
        self._journal_evidence(frame.action_id, dict(frame.result),
                               dict(frame.receipt or {}))

    def _journal_evidence(self, action_id: str, result: dict, receipt: dict) -> None:
        candidates = []
        sha = result.get('sha256')
        if isinstance(sha, str) and len(sha) == 64:
            candidates.append(('observation', sha))
        receipt_sha = receipt.get('receipt_sha256')
        if isinstance(receipt_sha, str) and len(receipt_sha) == 64:
            candidates.append(('receipt', receipt_sha))
        existing = {ref.evidence_id for ref in self.state.evidence}
        for index, (kind, digest) in enumerate(candidates):
            evidence_id = f'ev-{action_id[:12]}-{index}'
            if evidence_id in existing:
                continue
            patch = TaskPatch('add_evidence', 'host', {
                'evidence_id': evidence_id, 'kind': kind, 'digest': digest,
                'origin': f'{result.get("kind", "tool")}:{action_id}',
            })
            outcome = apply_patch(self.state, patch, self.verifier)
            self._journal_patch(outcome)
            if outcome.applied:
                self.state = outcome.state
                existing.add(evidence_id)

    def _bind_workspace(self, core: CoreState) -> CoreState:
        """Bind a proposer workspace to the post-decision exact digest/revision.

        The proposer produced it at the request revision; the outcome, patches
        and budget spend moved the exact state forward. The next decision loads
        exactly this binding, so it must describe the state the next decision
        will see.
        """
        core = detach_to_cpu(core)
        if core.task_digest == self.state.digest and core.revision == self.state.revision:
            return core
        return CoreState(core.workspace, core.valid, core.task_features, core.producer,
                         self.state.digest, self.state.revision, core.steps,
                         core.implementation)

    def _persist_workspace(self, output: ProposalOutput) -> None:
        if output.workspace is None:
            self.previous_workspace = None
            return
        core = self._bind_workspace(output.workspace)
        self.previous_workspace = core
        if self.workspace_store is not None:
            self.workspace_store.save(core)

    def _pool(self, workspace: CoreState | None) -> tuple | None:
        if workspace is None:
            return None
        valid = workspace.valid[0]
        slots = workspace.workspace[0]
        if bool(valid.any()):
            pooled = slots[valid].mean(dim=0)
        else:
            pooled = torch.zeros(workspace.workspace.shape[2], dtype=slots.dtype)
        return tuple(float(v) for v in pooled.detach().cpu().tolist())

    # ---- recovery -----------------------------------------------------------
    def _make_reconciler(self):
        tools = self.tools
        allowed = tuple(self.state.allowed_tools) if self.state is not None else ()

        def arguments(intent: PendingIntent) -> dict:
            args = dict(intent.args or {})
            nested = args.get('arguments')
            return dict(nested) if isinstance(nested, dict) else {
                key: value for key, value in args.items() if key != 'tool'}

        def reconcile(intent: PendingIntent, revision: int) -> OutcomeRecord | None:
            if tools is None:
                result = {'status': 'error', 'kind': intent.kind,
                          'error': 'tools_unavailable'}
                return _frame_from_tool_result(intent, result, tool_calls=0).to_record(revision)
            if intent.kind == 'invoke_tool' and intent.tool == 'write_file':
                args = arguments(intent)
                path = args.get('path')
                content = intent.content if intent.content is not None else args.get('content')
                if not isinstance(path, str) or not isinstance(content, str):
                    result = {'status': 'error', 'kind': 'write_file',
                              'error': 'unreconcilable_write_payload'}
                    return _frame_from_tool_result(
                        intent, result, tool_calls=0).to_record(revision)
                expected = intent.content_sha256 or sha256_bytes(content.encode('utf-8'))
                result = tools.reconcile_write(path, content,
                                               expected_sha256=expected)
                tool_calls = int(result.pop('tool_calls', 0))
                return _frame_from_tool_result(
                    intent, result, tool_calls=tool_calls).to_record(revision)
            # Read-only tools are safe to re-run only because a pending intent has
            # no receipt; a landed outcome is never pending.
            if intent.kind == 'read':
                result = tools.read_file(arguments(intent))
            elif intent.kind == 'invoke_tool':
                registry = tools.registry(allowed)
                result = registry.call(intent.tool, arguments(intent))
            elif intent.kind == 'retrieve':
                if self.retriever is None:
                    result = {'status': 'denied', 'kind': 'retrieve',
                              'error': 'retrieval_unavailable'}
                else:
                    observation = str(self.retriever(intent.args.get('query', '')))[
                        :MAX_OBSERVATION_EXCERPT]
                    result = {'status': 'ok', 'kind': 'retrieve', 'content': observation,
                              'sha256': sha256_bytes(observation.encode('utf-8'))}
            elif intent.kind == 'verify':
                condition_id = intent.args.get('condition_id')
                condition_ids = ([condition_id] if condition_id
                                 else [c.condition_id for c in self.state.success_conditions])
                verdict = tools.verify_success(condition_ids)
                result = {'status': 'ok' if verdict['passed'] else 'error',
                          'kind': 'verify', 'passed': verdict['passed'],
                          'receipt': verdict['receipt']}
            elif intent.kind in ('stop', 'answer'):
                condition_ids = [c.condition_id for c in self.state.success_conditions]
                verdict = tools.verify_success(condition_ids)
                met = verdict['passed']
                result = {'status': 'ok' if met else 'error', 'kind': intent.kind,
                          'claimed': True, 'verified': met, 'receipt': verdict['receipt']}
                if intent.kind == 'answer':
                    result['text'] = intent.args.get('text', '')
            elif intent.kind == 'ask':
                result = {'status': 'ok', 'kind': 'ask',
                          'question': intent.args.get('question', '')}
                return _frame_from_tool_result(
                    intent, result, status='pending_input', tool_calls=0).to_record(revision)
            else:
                result = {'status': 'error', 'kind': intent.kind,
                          'error': 'unreconcilable_intent'}
            tool_calls = 1 if intent.kind in ('read', 'invoke_tool', 'verify',
                                              'stop', 'answer') else 0
            return _frame_from_tool_result(
                intent, result, tool_calls=tool_calls).to_record(revision)

        return reconcile

    # ---- run loop -----------------------------------------------------------
    def summary(self) -> dict:
        """Stable host summary of the current journal state; no tool execution."""
        self._load()
        state = self.state
        return {
            'task_id': state.task_id, 'case_id': getattr(self.tools, 'case_id', None),
            'goal': state.goal, 'mode': self.mode, 'digest': state.digest,
            'revision': state.revision,
            'journal_head': self.journal.head,
            'budget': state.budget.to_dict(),
            'satisfied_conditions': list(state.satisfied_conditions(self.verifier)),
            'success_conditions': [c.condition_id for c in state.success_conditions],
            'actions': [a.to_dict() for a in state.actions],
            'outcomes': [o.to_dict() for o in state.outcomes],
        }

    def run(self, proposer, max_decisions: int) -> dict:
        if type(max_decisions) is not int or max_decisions < 0:
            raise ValueError('max_decisions must be a nonnegative integer')
        self.recover()
        trace = []
        first_features = None
        first_pool = None
        success = pending = exhausted = False
        self._budget_exhausted = False
        for _ in range(max_decisions):
            self._load()
            state = self.state
            if state.is_complete(self.verifier):
                success = True
                break
            if state.paused():
                pending = True
                break
            if not state.budget.allows(decisions=1):
                exhausted = True
                break
            if first_features is None:
                first_features = state.features()
            try:
                item = self.decide(proposer)
            except BudgetExhausted:
                exhausted = True
                break
            if first_pool is None:
                first_pool = item.workspace_pool
            trace.append(item)
            if self._budget_exhausted:
                exhausted = True
                break
        # Final status from the journal; covers max_decisions == 0 and mid-decision pauses.
        self._load()
        if not success and self.state.is_complete(self.verifier):
            success = True
        if not success and not pending and not exhausted:
            if self.state.paused():
                pending = True
            elif not self.state.budget.allows(decisions=1):
                exhausted = True
        feature_vector = list(first_features or ()) + list(
            first_pool if first_pool is not None else (0.0,) * self.workspace_width)
        return {
            'mode': self.mode,
            'case_id': self.tools.case_id if self.tools is not None else None,
            'task_id': self.state.task_id,
            'decisions': len(trace),
            'success': bool(success),
            'pending_input': bool(pending),
            'budget_exhausted': bool(exhausted),
            'state_digest': self.state.digest,
            'revision': self.state.revision,
            'journal_head': self.journal.head,
            'satisfied_conditions': list(self.state.satisfied_conditions(self.verifier)),
            'budget': self.state.budget.to_dict(),
            'feature_vector': feature_vector,
            'fast_memory': {'observed': self.fast_observed,
                            'error': self.fast_error} if self.fast_memory is not None else None,
            'trace': [asdict(item) for item in trace],
        }


def _frame_from_tool_result(intent: PendingIntent, result: dict, *, status=None,
                            tool_calls=None) -> OutcomeFrame:
    resolved = status or {'ok': 'completed', 'error': 'failed', 'denied': 'denied'}.get(
        result.get('status'), 'failed')
    if tool_calls is None:
        tool_calls = 0 if result.get('reconciled') else 1
    return OutcomeFrame(
        outcome_id='out-' + intent.action_id[:16], action_id=intent.action_id,
        status=resolved, result=result, verifier='host_recovery',
        spent={'tool_calls': tool_calls},
        observation_ids=(fingerprint(result),),
        receipt=result.get('receipt'), error=result.get('error'))


def _receipts(frame: OutcomeFrame) -> list:
    receipts = []
    if frame.receipt is not None:
        receipts.append(dict(frame.receipt))
    return receipts


def _observation_excerpt(outcome: OutcomeRecord) -> str:
    result = dict(outcome.result)
    for key in ('excerpt', 'content', 'stdout_excerpt', 'stderr_excerpt', 'question', 'text'):
        value = result.get(key)
        if isinstance(value, str) and value:
            return value[:MAX_OBSERVATION_EXCERPT]
    receipt = dict(outcome.receipt or {})
    for key in ('stdout_excerpt', 'stderr_excerpt'):
        value = receipt.get(key)
        if isinstance(value, str) and value:
            return value[:MAX_OBSERVATION_EXCERPT]
    return result.get('error') or result.get('reason') or '(no text observation)'


class SessionProposer:
    """Native CLI proposer wrapping one MemorySession and its tokenizer.

    Implements the documented E3 session API (``prepare(training=False, ...,
    task_features=..., previous_core=..., core_impl=..., enabled_branches=...)``
    followed by a bounded ``generate``). Mode controls live here, never inside
    TaskRuntime; the runtime only enforces authority.
    """

    def __init__(self, session, tokenizer, *, task_id: str, goal: str, scope,
                 device: str = 'mps', core_impl: str = 'recurrent',
                 enabled_branches=None, enabled_inputs=('sense', 'memory', 'task'),
                 zero_previous_workspace: bool = False, extra_inference_pass: bool = False,
                 core_disabled: bool = False):
        self.session = session
        self.tokenizer = tokenizer
        self.task_id = task_id
        self.goal = goal
        self.scope = scope
        self.device = device
        self.core_impl = core_impl
        self.enabled_branches = tuple(enabled_branches) if enabled_branches else None
        self.enabled_inputs = tuple(enabled_inputs)
        self.zero_previous_workspace = bool(zero_previous_workspace)
        self.extra_inference_pass = bool(extra_inference_pass)
        self.core_disabled = bool(core_disabled)

    def _on_device(self, core: CoreState) -> CoreState:
        if (core.workspace.device.type == self.device
                and core.task_features.device.type == self.device):
            return core
        return CoreState(core.workspace.to(self.device), core.valid.to(self.device),
                         core.task_features.to(self.device), core.producer,
                         core.task_digest, core.revision, core.steps,
                         core.implementation)

    def __call__(self, request: ProposalRequest) -> ProposalOutput:
        import torch
        from ..architecture.contracts import FastMemoryRead, TaskGraph, fingerprint
        from ..architecture.loading import encode_prompt
        mode = request.mode
        core_disabled = self.core_disabled or mode == 'core_disabled'
        zero_previous = self.zero_previous_workspace or mode == 'workspace_zero'
        if core_disabled or self.core_impl == 'none':
            core_impl = 'none'
        elif self.core_impl == 'feedforward' or mode == 'ff_core':
            core_impl = 'feedforward'
        else:
            core_impl = 'recurrent'
        extra_inference = self.extra_inference_pass or mode == 'extra_inference'
        adapter_has_core = self.session.adapter.core is not None
        no_core = core_impl == 'none' or not adapter_has_core
        branches = self.enabled_branches
        if no_core or extra_inference:
            branches = ('memory',)
        started = time.monotonic()
        prompt_ids = encode_prompt(self.tokenizer, request.prompt_text, self.device)
        if prompt_ids.shape[1] > request.prompt_token_budget:
            raise ValueError('Prompt exceeds the declared token budget; narrow the prompt')
        prompt_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
        task_features = None if no_core else torch.tensor(
            [request.task_features], dtype=torch.float32, device=self.device)
        notes = [f'mode={mode}']
        fast_read = None
        if request.fast_state is not None and getattr(self.session.adapter, 'fast_ports', None) is not None:
            state = request.fast_state
            if tuple(state.weights.shape) == (1, self.session.adapter.manifest.fast_memory.heads,
                                              self.session.adapter.manifest.fast_memory.key_size,
                                              self.session.adapter.manifest.fast_memory.value_size):
                fast_read = FastMemoryRead(
                    state.weights.to(device=self.device),
                    self.session.adapter.fast_producer_key(), state.digest)
                notes.append('fast_state=committed')
            else:
                notes.append('fast_state_discarded: shape_mismatch')
        elif request.fast_state is not None:
            notes.append('fast_state_discarded: no_native_fast_ports')
        if extra_inference:
            # Exactly one additional known-input sense pass; it creates no frame
            # and feeds no reader, and the normal prepare starts from scratch.
            with torch.no_grad():
                self.session.adapter.sense(prompt_ids, prompt_mask)
            notes.append('extra_inference_passes=1')
        previous_core = request.previous_workspace
        if previous_core is not None and (no_core or core_impl == 'feedforward'):
            # Neither control uses a previous recurrent workspace: the core is
            # disabled, or the feed-forward control ignores recurrence.
            previous_core = None
        if previous_core is not None:
            expected = self.session.adapter.core_producer_key()
            if previous_core.producer != expected:
                # Producer stamps include process-local parameter identities, so a
                # workspace from a fresh process must be rebuilt, not rejected.
                previous_core = None
                notes.append('workspace_discarded: producer_mismatch')
            else:
                previous_core = self._on_device(previous_core)
                if zero_previous:
                    previous_core = CoreState(
                        torch.zeros_like(previous_core.workspace),
                        previous_core.valid.clone(),
                        previous_core.task_features.clone(), previous_core.producer,
                        previous_core.task_digest, previous_core.revision,
                        previous_core.steps, previous_core.implementation)
        task = TaskGraph(task_id=self.task_id, goal=self.goal,
                         revision=request.revision, state_digest=request.task_digest)
        kwargs = dict(prompt_ids=prompt_ids, prompt_mask=prompt_mask, query=self.goal,
                      scope=self.scope, task=task, training=False,
                      task_features=task_features, previous_core=previous_core,
                      core_impl=core_impl, enabled_inputs=self.enabled_inputs,
                      enabled_branches=branches,
                      task_digest=request.task_digest, revision=request.revision,
                      fast_state=fast_read)
        leased = self.session.prepare(**kwargs)
        try:
            result = self.session.generate(leased,
                                           max_new_tokens=request.generated_token_budget)
            self.session.verify(leased)
            prompt_length = int(leased.frame.sense.prompt_ids.shape[1])
            new_ids = tuple(int(token) for token in result[0, prompt_length:].tolist())
            if len(new_ids) > request.generated_token_budget:
                raise ValueError('Generation exceeded the declared token budget')
            core = leased.frame.core
            workspace = detach_to_cpu(core) if core is not None else None
            frame_digest = fingerprint({
                'manifest': self.session.adapter.manifest.digest,
                'decision': leased.frame.decision_id,
                'task_digest': request.task_digest,
                'revision': request.revision,
                'mode': mode,
                'core': core.stamp() if core is not None else None})
            raw_text = self.tokenizer.decode(new_ids, skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False)
            return ProposalOutput(
                raw_text=raw_text, prompt_tokens=prompt_length,
                generated_token_ids=new_ids, generated_tokens=len(new_ids),
                frame_digest=frame_digest, workspace=workspace,
                workspace_stamp=core.stamp() if core is not None else None,
                wall_seconds=time.monotonic() - started, notes=tuple(notes))
        finally:
            leased.release()


__all__ = ['BudgetExhausted', 'DecisionTrace', 'MODES', 'ProposalOutput', 'ProposalRequest',
           'SessionProposer', 'TaskRuntime']
