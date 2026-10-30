"""Append-only durable work reservations for the v12 source-feature experiment.

This is a separately versioned successor to the sealed v5 ledger; v5 bytes and
its consumed receipts are untouched. Totals are counted over *all* work, not
only successful work. Nothing here is an authorization or a scientific gate.

Design changes from v5, declared so they can be audited rather than inferred:

* The journal is append-only ``.jsonl``. v5 rewrote one JSON document per event,
  so its logical serialized volume grew quadratically (37,101,387,705 logical
  bytes for 14,093 calls in the completed v11 attempt). This version appends one
  canonical line and fsyncs it. Logical appended bytes therefore equal the
  physical append request volume; physical SSD amplification is still *not*
  measured and is stamped as unmeasured rather than estimated.
* Both a monotonic and a wall-clock deadline are persisted. A reader uses the
  stricter remaining time, so a monotonic-clock restart cannot extend a budget.
* ``finish`` accepts only observations covered by the reservation, and a
  reservation is never refunded. A reservation without a terminal record stays
  charged; :meth:`WorkLedger.recover_interrupted` records it explicitly.
* A torn final line (partial append) is a fail-closed condition, not a silent
  repair: the caller must inspect it. Unknown work is never assumed to be zero.

Source-feature vocabulary added by v12 (all in addition to the v5 counters):
``source_feature_forwards`` (cache-missing frozen source forwards),
``tokenizer_calls``/``source_token_bytes`` (canonical-span tokenizer work),
``cache_misses``/``cache_evictions``/``cache_recomputations`` (bounded RAM C),
``artifact_read_bytes`` (restore reads) and ``restart_restores`` (fresh-process
checkpoint/teacher-cache restores).
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

FORMAT = 'replica.intrinsic_memory.reservations.v12'
COUNTERS = (
    'forwards', 'backwards', 'updates', 'input_tokens', 'generated_tokens',
    'model_loads', 'artifact_bytes', 'source_encodes', 'retrievals', 'writes',
    'source_feature_forwards', 'tokenizer_calls', 'prompt_tokenizer_calls',
    'source_token_bytes',
    'cache_misses', 'cache_evictions', 'cache_recomputations',
    'artifact_read_bytes', 'restart_restores',
)
LEGACY_COUNTERS = frozenset(COUNTERS[:10])
CLOCK_TOLERANCE_SECONDS = 5.0
RECORD_TYPES = frozenset(('plan', 'enter', 'reserve', 'finish', 'leave', 'recover', 'torn'))


class BudgetError(ValueError):
    pass


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic(path, value):
    """Durable single-document write for receipts; never a substituted path."""
    path = Path(path)
    if path.is_symlink(): raise BudgetError('substituted receipt path')
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def validate_plan(plan):
    if not isinstance(plan, dict) or not plan: raise BudgetError('stage plan missing')
    for stage, caps in plan.items():
        if not isinstance(stage, str) or not stage or set(caps) != set(COUNTERS)|{'wall_seconds'}:
            raise BudgetError('incomplete stage caps')
        if any(type(caps[k]) is not int or caps[k] < 0 for k in COUNTERS):
            raise BudgetError('invalid count reservation')
        if (isinstance(caps['wall_seconds'], bool) or not math.isfinite(caps['wall_seconds'])
                or caps['wall_seconds'] <= 0): raise BudgetError('invalid wall allocation')


def _amounts(value, *, where):
    if not isinstance(value, dict) or not set(value) <= set(COUNTERS):
        raise BudgetError('invalid reserved counters at '+where)
    for key, amount in value.items():
        if type(amount) is not int or amount < 0: raise BudgetError('invalid reservation amount at '+where)
    return {k: v for k, v in value.items() if v}


def _remaining(stage_deadline_monotonic, stage_deadline_unix, clock):
    """Stricter remaining time across both clocks; negative means expired."""
    now_monotonic, now_unix = time.monotonic(), time.time()
    outer_monotonic = clock['deadline_monotonic'] - now_monotonic
    outer_unix = clock['deadline_unix'] - now_unix
    stage_monotonic = (stage_deadline_monotonic - now_monotonic
                       if stage_deadline_monotonic is not None else math.inf)
    stage_unix = (stage_deadline_unix - now_unix if stage_deadline_unix is not None else math.inf)
    return (min(outer_monotonic, outer_unix), min(stage_monotonic, stage_unix),
            outer_monotonic - outer_unix)


def _canonical(record):
    if not isinstance(record, dict) or record.get('t') not in RECORD_TYPES:
        raise BudgetError('invalid journal record')
    return json.dumps(record, sort_keys=True, separators=(',', ':'))


def _replay(lines, *, identity):
    """Rebuild authoritative state from complete journal lines; verify as we go."""
    if not lines: raise BudgetError('empty reservation journal')
    plan_record = lines[0]
    if plan_record.get('t') != 'plan': raise BudgetError('reservation journal lacks its plan')
    plan = plan_record.get('plan')
    validate_plan(plan)
    if (plan_record.get('format') != FORMAT or plan_record.get('identity') != identity
            or plan_record.get('plan_sha256') != sha_json(plan)):
        raise BudgetError('ledger identity differs')
    clock = plan_record.get('clock')
    if (not isinstance(clock, dict)
            or set(clock) != {'started_monotonic', 'started_unix',
                              'deadline_monotonic', 'deadline_unix'}
            or any(isinstance(v, bool) or not math.isfinite(v) for v in clock.values())):
        raise BudgetError('ledger deadline state missing')
    stages, events = {}, []
    reserved = {k: 0 for k in COUNTERS}
    reserved_wall = 0
    for record in lines[1:]:
        kind = record.get('t')
        if kind == 'enter':
            stage = record['stage']
            if stage not in plan or stage in stages: raise BudgetError('stage missing or already consumed')
            if any(v['status'] == 'active' for v in stages.values()):
                raise BudgetError('two stages active in one journal')
            reserved_wall += plan[stage]['wall_seconds']
            stages[stage] = {'status': 'active', 'started': record['started'],
                'deadline': record['deadline'], 'deadline_unix': record['deadline_unix'],
                'reserved': {k: 0 for k in COUNTERS}}
        elif kind == 'reserve':
            stage = record['stage']
            state = stages.get(stage)
            amounts = _amounts(record['reserved'], where='replay')
            if state is None or state['status'] != 'active':
                raise BudgetError('reservation outside an active stage')
            if any(state['reserved'][k]+v > plan[stage][k] for k, v in amounts.items()):
                raise BudgetError('stage work budget exceeded in journal')
            if record['id'] != len(events): raise BudgetError('reservation id order differs')
            for k, v in amounts.items():
                state['reserved'][k] += v; reserved[k] += v
            events.append({'id': record['id'], 'stage': stage, 'operation': record['operation'],
                'reserved': amounts, 'status': 'reserved', 'started': record['started'],
                'pid': record['pid']})
        elif kind == 'finish':
            event = events[record['id']] if type(record.get('id')) is int and record['id'] < len(events) else None
            if event is None or event['status'] != 'reserved':
                raise BudgetError('terminal record does not match a reservation')
            if record['status'] not in ('completed', 'failed'): raise BudgetError('invalid terminal status')
            actual = _amounts(record.get('actual') or {}, where='actual')
            if any(k not in event['reserved'] or v > event['reserved'][k] for k, v in actual.items()):
                raise BudgetError('actual work exceeds reservation')
            event.update(status=record['status'], actual=actual, detail=record.get('detail'),
                         elapsed_seconds=record.get('elapsed_seconds'))
        elif kind == 'leave':
            state = stages.get(record['stage'])
            if state is None or state['status'] != 'active': raise BudgetError('stage already closed')
            if any(e['stage'] == record['stage'] and e['status'] == 'reserved' for e in events):
                raise BudgetError('stage has uncertain work')
            state.update(status=record['status'], elapsed_seconds=record.get('elapsed_seconds'))
        elif kind == 'recover':
            for identifier in record['ids']:
                event = events[identifier] if type(identifier) is int and identifier < len(events) else None
                if event is None or event['status'] != 'reserved':
                    raise BudgetError('recovery record does not match a reservation')
                event.update(status='failed', actual={},
                             detail={'error': 'BudgetError',
                                     'message': 'process terminated before terminal record; charged'},
                             elapsed_seconds=None)
        elif kind == 'torn':
            if type(record.get('bytes')) is not int or record['bytes'] <= 0:
                raise BudgetError('invalid torn-tail record')
        else:
            raise BudgetError('unknown journal record')
    return {'format': FORMAT, 'identity': identity, 'plan': plan, 'plan_sha256': sha_json(plan),
            'clock': clock, 'stages': stages, 'events': events, 'reserved': reserved,
            'reserved_wall_seconds': reserved_wall}


class WorkLedger:
    def __init__(self, path, *, identity):
        if not isinstance(identity, str) or not identity:
            raise BudgetError('execution identity missing')
        self.path, self.identity = Path(path), identity
        self._cache = None

    # ------------------------------------------------------------------
    # Journal access

    @staticmethod
    def _lines(payload):
        """Parse complete records; an incomplete final line is reported, not parsed.

        A torn tail means the process died during an append. The reservation in
        that line was never granted (``reserve`` returns only after fsync), so it
        authorised no work; the line is still reported so recovery can refuse
        instead of silently ignoring an unknown record.
        """
        if not payload: raise BudgetError('empty reservation journal')
        torn = not payload.endswith(b'\n')
        text = payload.decode('utf-8', 'strict')
        parts = text.split('\n')
        if parts and parts[-1] == '': parts.pop()
        if torn and parts: parts.pop()
        lines = []
        for index, part in enumerate(parts):
            try: lines.append(json.loads(part))
            except ValueError as error:
                raise BudgetError(f'corrupt journal line {index}') from error
        if not lines: raise BudgetError('reservation journal has no complete record')
        return lines, torn

    def _read(self):
        if self.path.is_symlink() or not self.path.is_file():
            raise BudgetError('missing or substituted reservation journal')
        payload = self.path.read_bytes()
        lines, torn = self._lines(payload)
        doc = _replay(lines, identity=self.identity)
        doc['journal'] = {'appended_bytes': len(payload), 'append_calls': len(lines),
                          'torn_tail': torn, 'path': str(self.path)}
        return doc

    @contextmanager
    def _transaction(self):
        """Exclusive read-then-append transaction; one line per mutation.

        The journal is append-only, so an unchanged (size, mtime) pair proves
        the cached state is current. That keeps per-operation cost constant
        instead of re-parsing a growing journal, without weakening durability:
        any append changes the size and forces a fresh verified replay.
        """
        lock = self.path.with_suffix('.lock')
        if lock.is_symlink() or self.path.is_symlink(): raise BudgetError('substituted ledger')
        with lock.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                doc = self._cached_read()
                yield doc, self._write
            finally:
                self._cache = None
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _cached_read(self):
        stat = self.path.stat()
        key = (stat.st_size, stat.st_mtime_ns)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        doc = self._read()
        stat = self.path.stat()
        self._cache = ((stat.st_size, stat.st_mtime_ns), doc)
        return doc

    def _write(self, record):
        """Append one record to an existing journal; never create it."""
        if self.path.is_symlink(): raise BudgetError('substituted ledger')
        line = (_canonical(record)+'\n').encode()
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        try:
            written = 0
            while written < len(line):
                written += os.write(fd, line[written:])
            os.fsync(fd)
        finally:
            os.close(fd)
        return len(line)

    def _write_new(self, record):
        """Create the journal exclusively with its first record; fsync file and dir."""
        if self.path.is_symlink(): raise BudgetError('substituted ledger')
        line = (_canonical(record)+'\n').encode()
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600)
        except FileExistsError as error:
            raise BudgetError('ledger must be fresh') from error
        try:
            written = 0
            while written < len(line):
                written += os.write(fd, line[written:])
            os.fsync(fd)
        finally:
            os.close(fd)
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return len(line)

    # ------------------------------------------------------------------
    # Construction, restart and recovery

    @classmethod
    def create(cls, path, *, identity, plan, deadline):
        validate_plan(plan)
        if isinstance(deadline, bool) or not math.isfinite(deadline):
            raise BudgetError('execution deadline missing')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink(): raise BudgetError('ledger must be fresh')
        ledger = cls(path, identity=identity)
        now_monotonic, now_unix = time.monotonic(), time.time()
        if now_monotonic >= deadline: raise BudgetError('outer deadline already expired')
        clock = {'started_monotonic': now_monotonic, 'started_unix': now_unix,
                 'deadline_monotonic': deadline,
                 'deadline_unix': now_unix + (deadline - now_monotonic)}
        lock = path.with_suffix('.lock')
        with lock.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                if path.exists() or path.is_symlink(): raise BudgetError('ledger must be fresh')
                ledger._write_new({'t': 'plan', 'format': FORMAT, 'identity': identity,
                                   'plan': plan, 'plan_sha256': sha_json(plan), 'clock': clock})
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
        return ledger

    def snapshot(self, *, cached=False):
        return self._cached_read() if cached else self._read()

    def deadline_state(self, doc=None):
        """Stricter remaining outer/stage seconds plus observed clock skew."""
        doc = self.snapshot() if doc is None else doc
        outer, _, skew = _remaining(None, None, doc['clock'])
        stages = {}
        for name, state in doc['stages'].items():
            if state['status'] != 'active': continue
            _, stage_remaining, _ = _remaining(state['deadline'], state['deadline_unix'], doc['clock'])
            stages[name] = stage_remaining
        return {'outer_remaining_seconds': outer, 'stage_remaining_seconds': stages,
                'clock_skew_seconds': skew,
                'clock_reanchored': abs(skew) > CLOCK_TOLERANCE_SECONDS}

    def recover_interrupted(self):
        """Record terminated work without refunding it. Refuses a torn tail.

        Operator-only: the planned main->restart hand-off leaves no outstanding
        reservation, so a healthy continuation never needs this call.
        """
        with self._transaction() as (doc, write):
            if doc['journal']['torn_tail']:
                raise BudgetError('torn journal tail: unknown reservation requires explicit operator review')
            ids = [e['id'] for e in doc['events'] if e['status'] == 'reserved']
            active = [name for name, state in doc['stages'].items() if state['status'] == 'active']
            if ids: write({'t': 'recover', 'ids': ids, 'at': time.monotonic()})
            for name in active:
                write({'t': 'leave', 'stage': name, 'status': 'interrupted',
                       'elapsed_seconds': None})
        return {'charged_events': ids, 'interrupted_stages': active, 'refunded': False}

    # ------------------------------------------------------------------
    # Reservations

    def enter(self, stage):
        with self._transaction() as (doc, write):
            if stage not in doc['plan'] or stage in doc['stages']:
                raise BudgetError('stage missing or already consumed')
            if any(v['status'] == 'active' for v in doc['stages'].values()):
                raise BudgetError('another stage is active')
            outer, _, _ = _remaining(None, None, doc['clock'])
            if outer <= 0: raise BudgetError('outer deadline exceeded')
            now_monotonic, now_unix = time.monotonic(), time.time()
            wall = doc['plan'][stage]['wall_seconds']
            write({'t': 'enter', 'stage': stage, 'started': now_monotonic,
                   'deadline': min(doc['clock']['deadline_monotonic'], now_monotonic+wall),
                   'deadline_unix': min(doc['clock']['deadline_unix'], now_unix+wall)})

    def reserve(self, stage, operation, **amounts):
        if not isinstance(operation, str) or not operation:
            raise BudgetError('invalid reservation operation')
        amounts = _amounts(amounts, where='reserve')
        with self._transaction() as (doc, write):
            state = doc['stages'].get(stage)
            if state is None or state['status'] != 'active': raise BudgetError('stage is not active')
            outer, stage_remaining, _ = _remaining(state['deadline'],
                                                   state['deadline_unix'], doc['clock'])
            if outer <= 0 or stage_remaining <= 0:
                raise BudgetError('stage/outer wall budget exceeded')
            if any(state['reserved'][k]+v > doc['plan'][stage][k] for k, v in amounts.items()):
                raise BudgetError('stage work budget exceeded before dispatch')
            token = len(doc['events'])
            write({'t': 'reserve', 'id': token, 'stage': stage, 'operation': operation,
                   'reserved': amounts, 'started': time.monotonic(), 'pid': os.getpid()})
            return token

    def finish(self, token, *, status='completed', actual=None, detail=None):
        if status not in ('completed', 'failed'): raise BudgetError('invalid terminal status')
        actual = _amounts(actual or {}, where='actual')
        with self._transaction() as (doc, write):
            if type(token) is not int or not 0 <= token < len(doc['events']):
                raise BudgetError('unknown reservation id')
            event = doc['events'][token]
            if event['status'] != 'reserved': raise BudgetError('reservation already terminal')
            if any(k not in event['reserved'] or v > event['reserved'][k] for k, v in actual.items()):
                raise BudgetError('actual work exceeds reservation')
            now = time.monotonic()
            elapsed = now-event['started'] if now >= event['started'] else None
            write({'t': 'finish', 'id': token, 'status': status, 'actual': actual,
                   'detail': detail, 'elapsed_seconds': elapsed})

    @contextmanager
    def work(self, stage, operation, **amounts):
        token = self.reserve(stage, operation, **amounts)
        observation = {}
        try:
            yield observation
        except BaseException as error:
            self.finish(token, status='failed',
                        detail={'error': type(error).__name__, 'message': str(error)})
            raise
        else:
            self.finish(token, actual=observation.pop('actual', None), detail=observation)

    def leave(self, stage, *, status='completed'):
        with self._transaction() as (doc, write):
            state = doc['stages'].get(stage)
            if state is None or state['status'] != 'active': raise BudgetError('stage already closed')
            if any(e['stage'] == stage and e['status'] == 'reserved' for e in doc['events']):
                raise BudgetError('stage has uncertain work; close as interrupted externally')
            now = time.monotonic()
            elapsed = now-state['started'] if now >= state['started'] else None
            write({'t': 'leave', 'stage': stage, 'status': status, 'elapsed_seconds': elapsed})


def audit_io(ledger):
    """Logical appended bytes versus physical I/O; never conflated."""
    doc = ledger.snapshot()
    journal = doc['journal']
    return {'logical_appended_bytes': journal['appended_bytes'],
            'append_calls': journal['append_calls'],
            'fsync_calls': journal['append_calls'],
            'physical_write_bytes_measured': False,
            'physical_io_note': ('append-only journal: requested bytes equal appended bytes; '
                                 'SSD write amplification, page cache and flush cost are not measured'),
            'reservations_charged': dict(doc['reserved']),
            'failed_or_interrupted_events': sum(1 for e in doc['events']
                                                if e['status'] in ('failed', 'interrupted'))}
