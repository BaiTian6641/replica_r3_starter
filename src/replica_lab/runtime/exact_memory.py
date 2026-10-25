"""Durable exact-record reference and query/evidence coverage gating.
The RecordLedger remains the source of truth. Fast tensors are derived state,
never a second authoritative store. Query routing uses only query text and
eligible record text; split, case role, expected answers and audit labels are
not accepted by this module.
"""
from __future__ import annotations
import hashlib

from dataclasses import asdict, dataclass
from pathlib import Path
import re

from ..architecture.contracts import fingerprint
from ..memory_system import (EligibilityScope, ExactSpan, MemoryRecord,
                             RecordLedger, RetrievalIndex, RetrievalSelection)


_STOPWORDS = frozenset({
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'being', 'by', 'can',
    'could', 'do', 'does', 'for', 'from', 'give', 'had', 'has', 'have', 'how',
    'if', 'in', 'is', 'it', 'its', 'me', 'not', 'of', 'on', 'or', 'please',
    'project', 'return', 'tell', 'that', 'the', 'their', 'this', 'to', 'was',
    'what', 'when', 'where', 'which', 'who', 'why', 'will', 'with', 'would',
    'you', 'your', 'about', 'assigned', 'value', 'values',
})
_FACT = re.compile(
    r'^(?:Update:\s*)?The\s+(?P<attribute>.+?)\s+for\s+project\s+'
    r'(?P<entity>\S+)\s+is\s+(?P<value>.+)\.$', re.IGNORECASE)


def _fact_line(fact):
    return f"The {fact['attribute']} for project {fact['entity']} is {fact['value']}."


def _parse_fact(text):
    match = _FACT.match(text)
    if match is None:
        return None
    return tuple(match.group(name).casefold() for name in ('entity', 'attribute', 'value'))


def _digest(scope, epoch, records):
    return fingerprint({'scope': asdict(scope), 'epoch': epoch,
                        'records': [(record.record_id, record.record_hash) for record in records]})


@dataclass(frozen=True)
class SourceView:
    ledger_id: str
    epoch: int
    scope: EligibilityScope
    records: tuple[MemoryRecord, ...]
    digest: str


@dataclass(frozen=True)
class ExactContext:
    selection: RetrievalSelection
    records: tuple[MemoryRecord, ...]
    spans: tuple[ExactSpan, ...]
    status: str
    reason: str
    view_digest: str
    duplicates_removed: int = 0

    @property
    def enabled(self):
        return self.status == 'relevant'

    @property
    def text(self):
        return '\n'.join(span.text for span in self.spans) if self.enabled else ''

    @property
    def record_ids(self):
        return tuple(record.record_id for record in self.records)

    def metadata(self):
        return {'gate': self.status, 'gate_reason': self.reason,
                'gate_record_ids': list(self.record_ids),
                'gate_ledger_id': self.selection.ledger_id,
                'gate_epoch': self.selection.epoch,
                'gate_view_digest': self.view_digest,
                'gate_duplicates_removed': self.duplicates_removed}


class ExactMemoryReference:
    """Authorized exact records, current views and selective context."""

    def __init__(self, ledger):
        self.ledger = ledger if isinstance(ledger, RecordLedger) else RecordLedger(ledger)
        self.index = RetrievalIndex(self.ledger)

    def close(self):
        self.ledger.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    def admit(self, records):
        changed = 0
        for record in records:
            if not isinstance(record, MemoryRecord):
                raise TypeError('Exact admission requires immutable MemoryRecord values')
            changed += self.ledger.admit(
                record, authorized_subject=record.subject_id,
                authorized_branch=record.branch_id)
        return changed

    def admit_episode(self, episode):
        initial, _ = episode_records(episode)
        return self.admit(initial)

    def correct_episode(self, episode):
        _, correction = episode_records(episode)
        return self.admit((correction,))

    def revoke(self, record_id, *, authorized_subject, authorized_branch):
        return self.ledger.revoke(
            record_id, authorized_subject=authorized_subject,
            authorized_branch=authorized_branch)

    def view(self, scope):
        snapshot = self.ledger.snapshot(scope)
        return SourceView(snapshot.ledger_id, snapshot.epoch, scope, snapshot.records,
                          _digest(scope, snapshot.epoch, snapshot.records))

    def select(self, query, scope, *, limit=64):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('Evidence gate requires a nonempty query')
        selection = self.index.retrieve(query, **asdict(scope), limit=limit)
        self.index.verify_selection(selection, expected_scope=scope)
        raw_terms = re.findall(r'\w+', query, flags=re.UNICODE)
        content = {term.casefold() for term in raw_terms if term.casefold() not in _STOPWORDS}
        entities = {term.casefold() for term in raw_terms
                    if term.casefold() not in _STOPWORDS
                    and (any(char.isdigit() for char in term) or term[:1].isupper())}
        status, reason = 'no_evidence', 'query has no content or entity-like terms'
        covered = []
        if content and entities:
            for record in selection.records:
                tokens = set(re.findall(r'\w+', record.text.casefold(), flags=re.UNICODE))
                if content <= tokens and entities & tokens:
                    covered.append(record)
            if covered:
                status, reason = 'relevant', 'eligible record covers query content and entity terms'
            else:
                reason = 'no eligible record covers all query content and entity terms'
        else:
            reason = 'query lacks content or entity-like terms'
        covered = [record for record in covered if _parse_fact(record.text) is not None]
        if not covered and status == 'relevant':
            status, reason = 'no_evidence', 'no eligible structured fact matches the query'
        unique, seen_text = [], set()
        for record in covered:
            if record.text in seen_text:
                continue
            seen_text.add(record.text)
            unique.append(record)
        duplicates = len(covered) - len(unique)
        matched_keys = {parsed[:2] for record in unique if (parsed := _parse_fact(record.text)) is not None}
        values = {key: set() for key in matched_keys}
        for record in self.ledger.snapshot(scope).records:
            parsed = _parse_fact(record.text)
            if parsed is not None and parsed[:2] in matched_keys:
                values[parsed[:2]].add(parsed[2])
        if any(len(items) > 1 for items in values.values()):
            status, reason = 'conflict', 'eligible records contain conflicting current values'
            unique = []
        spans = []
        if status == 'relevant':
            for record in unique:
                spans.append(self.ledger.fetch_exact(
                    record.record_id, **asdict(scope), start=0, end=len(record.text),
                    expected_hash=record.record_hash, expected_epoch=selection.epoch))
        return ExactContext(selection, tuple(unique), tuple(spans), status, reason,
                            _digest(scope, selection.epoch, selection.records), duplicates)

    def verify_context(self, context, *, query=None):
        """Authenticate the exact payload, not only its retrieval receipt."""
        if not isinstance(context, ExactContext):
            raise TypeError('Expected ExactContext')
        if query is not None and context.selection.query != query:
            raise ValueError('Exact context query differs from the consumer query')
        if not isinstance(context.selection.query, str) or not context.selection.query.strip():
            raise ValueError('Exact context query is empty')
        self.index.verify_selection(context.selection,
                                    expected_scope=context.selection.scope)
        expected = _digest(context.selection.scope, context.selection.epoch,
                           context.selection.records)
        if expected != context.view_digest:
            raise ValueError('Exact context view digest differs')
        if context.status not in {'relevant', 'no_evidence', 'conflict'}:
            raise ValueError('Unknown exact-context status')
        if not isinstance(context.reason, str) or not context.reason.strip():
            raise ValueError('Exact-context status reason is missing')
        if context.status != 'relevant':
            if context.records or context.spans:
                raise ValueError('Disabled exact context carries consumed evidence')
            return True
        if not context.records or len(context.records) != len(context.spans):
            raise ValueError('Relevant exact context requires one span per record')
        if type(context.duplicates_removed) is not int or context.duplicates_removed < 0:
            raise ValueError('Invalid exact-context duplicate count')
        selected = {record.record_id: record for record in context.selection.records}
        if len(selected) != len(context.selection.records):
            raise ValueError('Duplicate selected record identifier')
        consumed_ids = []
        for record, span in zip(context.records, context.spans):
            if record.record_id not in selected or record != selected[record.record_id]:
                raise ValueError('Consumed record is substituted or ineligible')
            if record.record_id in consumed_ids:
                raise ValueError('Duplicate consumed record')
            consumed_ids.append(record.record_id)
            if span.record_id != record.record_id:
                raise ValueError('Span and record identifiers differ')
            if span.start != 0 or span.end != len(record.text) or span.text != record.text:
                raise ValueError('Span offsets or text differ from the consumed record')
            if span.record_hash != record.record_hash or span.source_hash != record.source_hash:
                raise ValueError('Span provenance hash differs')
            if span.span_hash != hashlib.sha256(span.text.encode()).hexdigest():
                raise ValueError('Span content hash differs')
            if span.epoch != context.selection.epoch:
                raise ValueError('Span epoch differs from selection epoch')
        if consumed_ids != [record.record_id for record in context.records]:
            raise ValueError('Consumed record inventory differs')
        if context.duplicates_removed > len(context.selection.records) - len(consumed_ids):
            raise ValueError('Duplicate count exceeds selected inventory')
        return True

    @classmethod
    def rebuild(cls, path, records):
        """Build a fresh ledger only from explicitly permitted immutable records."""
        reference = cls(Path(path))
        try:
            reference.admit(records)
        except BaseException:
            reference.close()
            raise
        return reference


def episode_records(episode):
    """Render one immutable record per source fact and one superseding correction.

    Page-level supersession would incorrectly delete unaffected facts. Every
    fact line is authenticated against exactly one rendered source page first.
    """
    episode_id = episode['episode_id']
    displayed = list(episode['facts']) + list(episode['distractors'])
    lines = [_fact_line(fact) for fact in displayed]
    pages = [page.splitlines() for page in episode['observations']]
    locations = {}
    for line in lines:
        matches = [(page_index, line_index) for page_index, page in enumerate(pages)
                   for line_index, value in enumerate(page) if value == line]
        if len(matches) != 1:
            raise ValueError('Structured fact does not match exactly one source line')
        locations[line] = matches[0][0]
    records = []
    for index, (fact, line) in enumerate(zip(displayed, lines)):
        page_index = locations[line]
        records.append(MemoryRecord(
            record_id=f'{episode_id}:fact:{index}', subject_id='memory_reliability',
            branch_id=episode_id, text=line, observed_at=page_index + 1,
            event_at=index + 1, purpose='reliability', kind='source_fact',
            source_id=f'{episode_id}:source:{page_index}', authenticity='verified_source'))
    correction = episode['correction']
    parent = next(index for index, fact in enumerate(episode['facts'])
                  if (fact['entity'], fact['attribute'])
                  == (correction['entity'], correction['attribute']))
    correction_record = MemoryRecord(
        record_id=f'{episode_id}:correction', subject_id='memory_reliability',
        branch_id=episode_id, text=episode['correction_observation'],
        observed_at=3, event_at=len(records) + 1, purpose='reliability',
        kind='source_correction', supersedes=f'{episode_id}:fact:{parent}',
        source_id=f'{episode_id}:correction', authenticity='verified_source')
    return tuple(records), correction_record
