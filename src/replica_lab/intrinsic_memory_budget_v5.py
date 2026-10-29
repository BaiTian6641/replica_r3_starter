"""Durable, process-shared reservations for the v5 experiment, not authorization.

Work is charged before dispatch, including failed/uncertain work. Completion
records observations but never refunds a reservation. A child uses the same
ledger and absolute monotonic deadline; restart cannot replenish a budget.
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

FORMAT = 'replica.intrinsic_memory.reservations.v5'
COUNTERS = ('forwards', 'backwards', 'updates', 'input_tokens', 'generated_tokens',
            'model_loads', 'artifact_bytes', 'source_encodes', 'retrievals', 'writes')


class BudgetError(ValueError):
    pass


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    if path.is_symlink(): raise BudgetError('substituted ledger path')
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


def audit_write(path,doc):
    audit=doc.setdefault('audit_io',{'serialized_write_bytes':0,'write_calls':0})
    previous=audit['serialized_write_bytes'];audit['write_calls']+=1;size=0
    for _ in range(10):
        audit['serialized_write_bytes']=previous+size
        actual=len(json.dumps(doc,indent=2,sort_keys=True).encode())+1
        if size==actual:break
        size=actual
    atomic(path,doc)


class WorkLedger:
    def __init__(self, path, *, identity):
        self.path, self.identity = Path(path), identity

    @classmethod
    def create(cls, path, *, identity, plan, deadline):
        validate_plan(plan)
        if not isinstance(identity, str) or not identity or not math.isfinite(deadline):
            raise BudgetError('execution identity/deadline missing')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink(): raise BudgetError('ledger must be fresh')
        now = time.monotonic()
        if now >= deadline: raise BudgetError('outer deadline already expired')
        doc = {'format': FORMAT, 'identity': identity, 'plan': plan, 'plan_sha256': sha_json(plan),
               'started': now, 'deadline': deadline, 'stages': {}, 'events': [],
               'reserved': {k: 0 for k in COUNTERS}, 'reserved_wall_seconds': 0}
        audit_write(path, doc)
        return cls(path, identity=identity)

    @contextmanager
    def _locked(self, *, write=True):
        lock = self.path.with_suffix('.lock')
        if lock.is_symlink() or self.path.is_symlink(): raise BudgetError('substituted ledger')
        with lock.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            doc = json.loads(self.path.read_text())
            if (doc.get('format') != FORMAT or doc.get('identity') != self.identity
                    or doc.get('plan_sha256') != sha_json(doc.get('plan'))):
                raise BudgetError('ledger identity differs')
            yield doc
            if write:audit_write(self.path, doc)

    def snapshot(self):
        with self._locked(write=False) as doc: return json.loads(json.dumps(doc))

    def enter(self, stage):
        with self._locked() as doc:
            if stage not in doc['plan'] or stage in doc['stages']:
                raise BudgetError('stage missing or already consumed')
            if any(v['status'] == 'active' for v in doc['stages'].values()):
                raise BudgetError('another stage is active')
            now = time.monotonic()
            if now >= doc['deadline']: raise BudgetError('outer deadline exceeded')
            wall = doc['plan'][stage]['wall_seconds']
            doc['reserved_wall_seconds'] += wall
            doc['stages'][stage] = {'status': 'active', 'started': now,
                'deadline': min(doc['deadline'], now+wall), 'reserved': {k: 0 for k in COUNTERS}}

    def reserve(self, stage, operation, **amounts):
        if not isinstance(operation, str) or not operation or not set(amounts) <= set(COUNTERS):
            raise BudgetError('invalid reservation operation')
        if any(type(v) is not int or v < 0 for v in amounts.values()):
            raise BudgetError('invalid reservation amount')
        with self._locked() as doc:
            s = doc['stages'].get(stage)
            if s is None or s['status'] != 'active': raise BudgetError('stage is not active')
            now = time.monotonic()
            if now >= min(s['deadline'], doc['deadline']): raise BudgetError('stage/outer wall budget exceeded')
            if any(s['reserved'][k]+v > doc['plan'][stage][k] for k,v in amounts.items()):
                raise BudgetError('stage work budget exceeded before dispatch')
            for k,v in amounts.items():
                s['reserved'][k] += v; doc['reserved'][k] += v
            token = len(doc['events'])
            doc['events'].append({'id': token, 'stage': stage, 'operation': operation,
                'reserved': amounts, 'status': 'reserved', 'started': now, 'pid': os.getpid()})
            return token

    def finish(self, token, *, status='completed', actual=None, detail=None):
        with self._locked() as doc:
            event = doc['events'][token]
            if event['status'] != 'reserved': raise BudgetError('reservation already terminal')
            if status not in ('completed', 'failed'): raise BudgetError('invalid terminal status')
            actual = actual or {}
            if any(k not in event['reserved'] or type(v) is not int or v < 0
                   or v > event['reserved'][k] for k,v in actual.items()):
                raise BudgetError('actual work exceeds reservation')
            event.update(status=status, actual=actual, detail=detail,
                         elapsed_seconds=time.monotonic()-event['started'])

    @contextmanager
    def work(self, stage, operation, **amounts):
        token = self.reserve(stage, operation, **amounts)
        observation = {}
        try:
            yield observation
        except BaseException as error:
            self.finish(token, status='failed', detail={'error': type(error).__name__, 'message': str(error)})
            raise
        else:
            self.finish(token, actual=observation.pop('actual', None), detail=observation)

    def leave(self, stage, *, status='completed'):
        with self._locked() as doc:
            s = doc['stages'][stage]
            if s['status'] != 'active': raise BudgetError('stage already closed')
            if any(e['stage'] == stage and e['status'] == 'reserved' for e in doc['events']):
                raise BudgetError('stage has uncertain work; close as interrupted externally')
            s.update(status=status, elapsed_seconds=time.monotonic()-s['started'])
