"""Append-only hash-chained SQLite journal bound to one exact task (E3 §3).

Every state change reaches the exact state only through journaled events; the
journal is the only required copy of the task. ``load()`` reconstructs the exact
state and pending intents without executing tools; ``recover()`` reconciles each
pending intent exactly once through a host reconciler. Tampered chains are
rejected on open.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid

from ..architecture.contracts import fingerprint
from .task_graph import (ActionRecord, Constraint, ExactTaskState, JournalVerifier,
                         MAX_CONSTRAINTS, MAX_NOTE, MAX_NOTES, OutcomeRecord, TaskPatch,
                         apply_patch)

GENESIS_HASH = '0' * 64
MAX_EVENTS_DEFAULT = 4096
MAX_PAYLOAD_BYTES = 128 * 1024
MAX_RAW_TEXT_BYTES = 16 * 1024

BINDING_FIELDS = ('task_id', 'subject', 'branch', 'purpose', 'source_epoch',
                  'goal_digest', 'component_versions')
STATE_EVENT_KINDS = ('binding', 'init', 'decision', 'action_intent', 'action_outcome',
                     'patch', 'revision', 'budget', 'correction', 'invalidation')


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _validate_binding(binding) -> dict:
    if not isinstance(binding, dict):
        raise ValueError('Journal binding must be a dict')
    missing = [name for name in BINDING_FIELDS if name not in binding]
    if missing:
        raise ValueError(f'Journal binding requires {missing}')
    if type(binding['source_epoch']) is not int or binding['source_epoch'] < 0:
        raise ValueError('Binding source_epoch must be a nonnegative integer')
    for name in ('task_id', 'subject', 'branch', 'purpose', 'goal_digest'):
        if not isinstance(binding[name], str) or not binding[name]:
            raise ValueError(f'Binding {name} must be a nonempty string')
    versions = binding['component_versions']
    if not isinstance(versions, dict) or not versions:
        raise ValueError('Binding component_versions must be a nonempty dict')
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in versions.items()):
        raise ValueError('Binding component_versions must map names to version strings')
    return {name: binding[name] for name in BINDING_FIELDS}


@dataclass(frozen=True)
class Event:
    seq: int
    event_id: str
    kind: str
    payload: dict
    idempotency_key: str | None
    prev_hash: str
    hash: str

    def to_dict(self):
        return {'seq': self.seq, 'event_id': self.event_id, 'kind': self.kind,
                'payload': self.payload, 'idempotency_key': self.idempotency_key,
                'prev_hash': self.prev_hash, 'hash': self.hash}


@dataclass(frozen=True)
class PendingIntent:
    """A journaled action intent whose outcome never landed (crash window)."""
    event_id: str
    action_id: str
    kind: str
    args: dict
    idempotency_key: str
    revision: int
    tool: str | None = None
    subgoal_id: str | None = None
    content: str | None = None
    content_sha256: str | None = None


@dataclass(frozen=True)
class LoadedSnapshot:
    state: ExactTaskState
    pending: tuple[PendingIntent, ...]
    verifier: object
    head: str
    event_count: int


class JournalError(RuntimeError):
    pass


class BindingMismatch(JournalError):
    pass


class JournalTampered(JournalError):
    pass


class JournalFull(JournalError):
    pass


def _event_hash(prev_hash: str, event_id: str, kind: str, payload_json: str) -> str:
    return hashlib.sha256(
        (prev_hash + '\n' + event_id + '\n' + kind + '\n' + payload_json).encode('utf-8')
    ).hexdigest()


class TaskJournal:
    """One writer, append-only, hash-chained; bounded event count."""

    def __init__(self, path, *, max_events: int = MAX_EVENTS_DEFAULT):
        if type(max_events) is not int or max_events < 2:
            raise ValueError('max_events must be a positive integer >= 2')
        self.path = Path(path)
        self.max_events = max_events
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.execute('PRAGMA journal_mode=WAL')
        self._db.execute(
            'CREATE TABLE IF NOT EXISTS events ('
            ' seq INTEGER PRIMARY KEY AUTOINCREMENT,'
            ' event_id TEXT NOT NULL UNIQUE,'
            ' kind TEXT NOT NULL,'
            ' payload TEXT NOT NULL,'
            ' idempotency_key TEXT UNIQUE,'
            ' prev_hash TEXT NOT NULL,'
            ' hash TEXT NOT NULL)')
        self._db.commit()

    # ---- low-level append --------------------------------------------------
    def append(self, kind, payload, *, idempotency_key=None, event_id=None) -> Event:
        if kind not in STATE_EVENT_KINDS:
            raise ValueError(f'Unknown journal event kind: {kind!r}')
        payload_json = canonical_json(payload)
        if len(payload_json.encode('utf-8')) > MAX_PAYLOAD_BYTES:
            raise ValueError('Journal payload exceeds the bounded size')
        self._db.execute('BEGIN IMMEDIATE')
        try:
            if idempotency_key is not None:
                row = self._db.execute(
                    'SELECT seq, event_id, kind, payload, idempotency_key, prev_hash, hash'
                    ' FROM events WHERE idempotency_key = ?', (idempotency_key,)).fetchone()
                if row is not None:
                    if row[2] != kind:
                        raise ValueError('Idempotency key reused across event kinds')
                    existing = json.loads(row[3])
                    if existing != payload:
                        raise ValueError('Idempotency key reused with a different payload')
                    self._db.rollback()
                    return Event(row[0], row[1], row[2], existing, row[4], row[5], row[6])
            count = self._db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
            if count >= self.max_events:
                raise JournalFull(f'Journal reached its bounded event count {self.max_events}')
            prev = self._db.execute(
                'SELECT hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
            prev_hash = prev[0] if prev is not None else GENESIS_HASH
            resolved_id = event_id or uuid.uuid4().hex
            event_hash = _event_hash(prev_hash, resolved_id, kind, payload_json)
            cursor = self._db.execute(
                'INSERT INTO events (event_id, kind, payload, idempotency_key, prev_hash, hash)'
                ' VALUES (?, ?, ?, ?, ?, ?)',
                (resolved_id, kind, payload_json, idempotency_key, prev_hash, event_hash))
            self._db.commit()
            return Event(cursor.lastrowid, resolved_id, kind, payload, idempotency_key,
                         prev_hash, event_hash)
        except BaseException:
            self._db.rollback()
            raise

    def events(self) -> tuple[Event, ...]:
        rows = self._db.execute(
            'SELECT seq, event_id, kind, payload, idempotency_key, prev_hash, hash'
            ' FROM events ORDER BY seq').fetchall()
        return tuple(Event(r[0], r[1], r[2], json.loads(r[3]), r[4], r[5], r[6]) for r in rows)

    def verify_chain(self) -> None:
        prev = GENESIS_HASH
        for event in self.events():
            expected = _event_hash(prev, event.event_id, event.kind,
                                   canonical_json(event.payload))
            if event.prev_hash != prev or event.hash != expected:
                raise JournalTampered(f'Hash chain broken at seq {event.seq}')
            prev = event.hash

    # ---- binding / lifecycle ----------------------------------------------
    @classmethod
    def open(cls, path, binding, *, max_events: int = MAX_EVENTS_DEFAULT) -> 'TaskJournal':
        validated = _validate_binding(binding)
        journal = cls(path, max_events=max_events)
        try:
            journal.verify_chain()
        except BaseException:
            journal.close()
            raise
        existing = [event for event in journal.events() if event.kind == 'binding']
        if not existing:
            journal.append('binding', {'binding': validated})
            return journal
        recorded = existing[0].payload.get('binding', {})
        if recorded != validated:
            differing = sorted(name for name in BINDING_FIELDS
                               if recorded.get(name) != validated[name])
            journal.close()
            raise BindingMismatch(
                'Journal is bound to a different task identity; differing fields: '
                + ','.join(differing))
        return journal

    @property
    def binding(self) -> dict:
        for event in self.events():
            if event.kind == 'binding':
                return _validate_binding(event.payload['binding'])
        raise JournalError('Journal has no binding event')

    @property
    def epoch(self) -> int:
        return self.binding['source_epoch']

    @property
    def head(self) -> str:
        row = self._db.execute('SELECT hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
        return row[0] if row is not None else GENESIS_HASH

    def initialize(self, state: ExactTaskState) -> Event:
        if not isinstance(state, ExactTaskState):
            raise ValueError('initialize requires an ExactTaskState')
        binding = self.binding
        if state.task_id != binding['task_id'] or state.subject != binding['subject'] \
                or state.branch != binding['branch'] or state.purpose != binding['purpose'] \
                or state.source_epoch != binding['source_epoch'] \
                or fingerprint(state.goal) != binding['goal_digest']:
            raise ValueError('Initial state contradicts the journal binding')
        return self.append('init', {'state': state.to_dict()}, idempotency_key='init')

    def initialize_done(self) -> bool:
        return any(event.kind == 'init' for event in self.events())

    def correct(self, *, note=None, constraint=None) -> Event:
        """Host-only correction: appends an authorized correction event."""
        if note is None and constraint is None:
            raise ValueError('A correction requires a note or a constraint')
        if note is not None and (not isinstance(note, str) or not note.strip()
                                 or len(note) > MAX_NOTE):
            raise ValueError('Correction note must be a bounded nonempty string')
        payload = {'author': 'host', 'note': note, 'constraint': constraint}
        key = 'correction:' + fingerprint(payload)
        recorded = self._db.execute(
            'SELECT 1 FROM events WHERE idempotency_key = ?', (key,)).fetchone()
        if recorded is None:
            state = self.load().state
            if note is not None and len(state.notes) >= MAX_NOTES:
                raise ValueError('Correction would exceed the bounded note count')
            if constraint is not None:
                if (not isinstance(constraint, dict)
                        or set(constraint) != {'constraint_id', 'text'}
                        or not isinstance(constraint.get('constraint_id'), str)
                        or not isinstance(constraint.get('text'), str)):
                    raise ValueError(
                        'Correction constraint requires exactly constraint_id and text')
                try:
                    Constraint(constraint['constraint_id'], constraint['text'])
                except ValueError as error:
                    raise ValueError(f'Invalid correction constraint: {error}') from error
                if any(c.constraint_id == constraint['constraint_id']
                       for c in state.constraints):
                    raise ValueError('Correction constraint id already exists')
                if len(state.constraints) >= MAX_CONSTRAINTS:
                    raise ValueError('Correction would exceed the bounded constraint count')
        return self.append('correction', payload, idempotency_key=key)

    def invalidate(self, *, claim_id: str, reason: str) -> Event:
        if not isinstance(claim_id, str) or not claim_id:
            raise ValueError('Invalidation requires a claim id')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('Invalidation requires a reason')
        if claim_id not in self.load().state.claim_map():
            raise ValueError(f'Invalidation targets unknown claim {claim_id}')
        payload = {'claim_id': claim_id, 'reason': reason}
        return self.append('invalidation', payload,
                           idempotency_key='invalidation:' + fingerprint(payload))

    # ---- reconstruction ----------------------------------------------------
    def _replay_digests(self, outcome_payload: dict, digests: set) -> None:
        outcome = outcome_payload.get('outcome', {})
        receipt = outcome.get('receipt') or {}
        for candidate in (receipt.get('receipt_sha256'),
                          (outcome.get('result') or {}).get('sha256')):
            if isinstance(candidate, str) and len(candidate) == 64:
                digests.add(candidate)

    def load(self) -> LoadedSnapshot:
        """Rebuild the exact state and pending intents; never executes tools."""
        self.verify_chain()
        events = self.events()
        state = None
        digests: set = set()
        pending: list[PendingIntent] = []
        resolved_actions: set[str] = set()
        for event in events:
            payload = event.payload
            try:
                if event.kind == 'binding':
                    _validate_binding(payload['binding'])
                    continue
                if event.kind == 'init':
                    if state is not None:
                        raise JournalTampered('Repeated init event')
                    state = ExactTaskState.from_dict(payload['state'])
                    continue
                if state is None:
                    raise JournalError(
                        f'Journal event {event.kind!r} precedes the init event')
                if event.kind == 'decision':
                    if payload.get('budget_exceeded'):
                        # Generation happened, but the cumulative token cap would
                        # be crossed; only the decision slot is charged and the
                        # actual usage stays recorded in this event.
                        state = state.spend(decisions=1)
                    else:
                        state = state.spend(
                            decisions=1, prompt_tokens=int(payload['prompt_tokens']),
                            generated_tokens=int(payload['generated_tokens']))
                elif event.kind == 'action_intent':
                    record = ActionRecord(
                        action_id=payload['action_id'], kind=payload['kind'],
                        args=payload['args'],
                        state='attempted' if payload.get('authorized') else 'proposed',
                        revision=state.revision, idempotency_key=payload['idempotency_key'],
                        tool=payload.get('tool'), subgoal_id=payload.get('subgoal_id'))
                    state = state.with_action(record)
                    if payload.get('authorized'):
                        pending.append(PendingIntent(
                            event_id=event.event_id, action_id=payload['action_id'],
                            kind=payload['kind'], args=payload['args'],
                            idempotency_key=payload['idempotency_key'],
                            revision=record.revision, tool=payload.get('tool'),
                            subgoal_id=payload.get('subgoal_id'),
                            content=payload.get('content'),
                            content_sha256=payload.get('content_sha256')))
                elif event.kind == 'action_outcome':
                    outcome = OutcomeRecord.from_dict(payload['outcome'])
                    state = state.with_outcome(outcome)
                    resolved_actions.add(outcome.action_id)
                    if payload.get('spent', {}).get('tool_calls'):
                        state = state.spend(tool_calls=int(payload['spent']['tool_calls']))
                    self._replay_digests(payload, digests)
                elif event.kind == 'patch':
                    patch = TaskPatch.from_dict(payload['patch'])
                    if payload['result']['applied']:
                        result = apply_patch(state, patch, JournalVerifier(digests))
                        if not result.applied or result.state is None:
                            raise JournalTampered(
                                f'Patch replay diverged at seq {event.seq}: {result.reason}')
                        state = result.state
                        if patch.op == 'add_evidence' and patch.author == 'host':
                            digest = patch.payload.get('digest')
                            if isinstance(digest, str) and len(digest) == 64:
                                digests.add(digest)
                elif event.kind == 'revision':
                    if (state.revision != payload['revision']
                            or state.digest != payload['state_digest']):
                        raise JournalTampered(
                            f'Revision checkpoint mismatch at seq {event.seq}')
                elif event.kind == 'correction':
                    if payload.get('author') != 'host':
                        raise JournalTampered('Corrections require host authority')
                    state = _apply_correction(state, payload)
                elif event.kind == 'invalidation':
                    state = _apply_invalidation(state, payload)
                elif event.kind == 'budget':
                    # Budget denials are audit records; spend is carried by the
                    # decision/outcome events that actually consumed resources.
                    continue
            except JournalTampered:
                raise
            except (KeyError, TypeError, ValueError) as error:
                raise JournalTampered(
                    f'Invalid {event.kind} event at seq {event.seq}: {error}') from error
        if state is None:
            raise JournalError('Journal has no init event; the exact state is unknown')
        pending = tuple(p for p in pending if p.action_id not in resolved_actions)
        return LoadedSnapshot(state, pending, JournalVerifier(digests), self.head, len(events))

    def pending_intents(self) -> tuple[PendingIntent, ...]:
        return self.load().pending

    def recover(self, reconciler=None) -> tuple[OutcomeRecord, ...]:
        """Reconcile each pending intent exactly once via the host reconciler.

        Reconciliation is journal-driven: the replay state advances through the
        same ``with_outcome`` then ``spend`` transitions as a live decision, so
        subsequent intents see the exact revision they were recorded against.
        Calling this with no pending intents is a no-op.
        """
        snapshot = self.load()
        if not snapshot.pending:
            return tuple()
        if reconciler is None:
            raise ValueError('Pending intents require a host reconciler')
        state = snapshot.state
        results = []
        for intent in snapshot.pending:
            outcome = reconciler(intent, state.revision)
            if outcome is None:
                continue
            if not isinstance(outcome, OutcomeRecord):
                raise ValueError('Reconciler must return an OutcomeRecord')
            if outcome.action_id != intent.action_id:
                raise ValueError('Reconciled outcome must reference its pending action')
            if outcome.revision != state.revision:
                raise ValueError('Reconciled outcome must be recorded at the replay revision')
            tool_calls = int((outcome.spent or {}).get('tool_calls', 0))
            if tool_calls and not state.budget.allows(tool_calls=tool_calls):
                raise ValueError('Recovery cannot charge the declared tool-call budget')
            event = self.append('action_outcome',
                                {'outcome': outcome.to_dict(),
                                 'spent': {'tool_calls': tool_calls}},
                                idempotency_key='outcome:' + intent.action_id)
            record = OutcomeRecord.from_dict(event.payload['outcome'])
            state = state.with_outcome(record)
            if tool_calls:
                state = state.spend(tool_calls=tool_calls)
            results.append(record)
        return tuple(results)

    def append_revision_checkpoint(self, state: ExactTaskState) -> Event:
        return self.append('revision', {'revision': state.revision,
                                        'state_digest': state.digest})

    def close(self):
        self._db.close()


def _apply_correction(state: ExactTaskState, payload: dict) -> ExactTaskState:
    """Host-authoritative correction; bypasses TaskPatch because the host writes it."""
    constraint_payload = payload.get('constraint')
    note = payload.get('note')
    constraints = state.constraints
    notes = state.notes
    if constraint_payload is not None:
        if (not isinstance(constraint_payload, dict)
                or set(constraint_payload) != {'constraint_id', 'text'}):
            raise JournalTampered('Correction constraint has an invalid shape')
        if any(c.constraint_id == constraint_payload['constraint_id']
               for c in state.constraints):
            raise JournalTampered('Correction repeats a constraint identifier')
        if len(state.constraints) >= MAX_CONSTRAINTS:
            raise JournalTampered('Correction exceeds the bounded constraint count')
        constraints = (*state.constraints,
                       Constraint(constraint_payload['constraint_id'], constraint_payload['text']))
    if note is not None:
        if not isinstance(note, str) or not note.strip() or len(note) > MAX_NOTE:
            raise JournalTampered('Correction note violates its bound')
        if len(state.notes) >= MAX_NOTES:
            raise JournalTampered('Correction exceeds the bounded note count')
        notes = (*state.notes, note)
    return ExactTaskState(
        task_id=state.task_id, subject=state.subject, branch=state.branch,
        purpose=state.purpose, source_epoch=state.source_epoch, goal=state.goal,
        success_conditions=state.success_conditions, constraints=constraints,
        subgoals=state.subgoals, claims=state.claims, evidence=state.evidence,
        obligations=state.obligations, actions=state.actions, outcomes=state.outcomes,
        notes=notes, budget=state.budget, binding=dict(state.binding),
        allowed_tools=state.allowed_tools, revision=state.revision + 1)


def _apply_invalidation(state: ExactTaskState, payload: dict) -> ExactTaskState:
    claim_id = payload['claim_id']
    claims = state.claim_map()
    if claim_id not in claims:
        raise JournalTampered(f'Invalidation targets unknown claim {claim_id}')
    invalidated = replace(claims[claim_id], status='unknown')
    new_claims = tuple(invalidated if c.claim_id == claim_id else c for c in state.claims)
    return ExactTaskState(
        task_id=state.task_id, subject=state.subject, branch=state.branch,
        purpose=state.purpose, source_epoch=state.source_epoch, goal=state.goal,
        success_conditions=state.success_conditions, constraints=state.constraints,
        subgoals=state.subgoals, claims=new_claims, evidence=state.evidence,
        obligations=state.obligations, actions=state.actions, outcomes=state.outcomes,
        notes=state.notes, budget=state.budget, binding=dict(state.binding),
        allowed_tools=state.allowed_tools, revision=state.revision + 1)
