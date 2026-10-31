"""Versioned contextual correctness successor; sealed v3/v4 bytes are unchanged.

Live values remain differentiable. Durable reads/export detach. Every active
reader rechecks authority; only producer hashing caches normal PyTorch mutation
versions. The fixed eight-slot snapshot masks unused slots explicitly.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import torch

from .contextual_memory import (CanonicalSpan, ContextualMemoryError, ValueBatch,
    PRODUCER, REPRESENTATION, CODEC, SLOTS, WIDTH, tensor_digest)
from .contextual_memory_layout_v4 import ContextualMemoryLayoutV4
from .tiered_memory import (MemoryScope, RecallTicket, RecallResult, RecalledRecord,
    Outcome, EvidenceRecordDraft, _read_verified, unpack_values)
from .tiered_readout import ReadoutBudget, read_scoped_memory, resume_scoped_readout

FORMAT = 'replica.contextual_memory.v5'
SLOT_POLICY = 'fixed_eight_masked_padding_v5'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    if path.is_symlink() or not path.is_file():
        raise ContextualMemoryError('missing or substituted file')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    """Fsync new bytes and directory, then atomically replace one authority pointer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ContextualMemoryError('substituted authority path')
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def effective_scope(scope):
    if not isinstance(scope, MemoryScope):
        raise ContextualMemoryError('MemoryScope required')
    return replace(scope, as_of=scope.cutoff)


@dataclass(frozen=True)
class PendingContextualRead:
    ticket_id: str
    store_root: str
    scope: dict
    descriptor_sha256: str
    costs: dict
    status: str = 'pending'


class ContextualReadFailure(ContextualMemoryError):
    def __init__(self, reason, costs):
        super().__init__(reason)
        self.costs = dict(costs)

class AuthorityReadFailure(ContextualReadFailure):
    """A cold read that failed specifically because authority was invalidated."""

    def __init__(self, authority_reason, message, costs):
        super().__init__(message, costs)
        self.authority_reason = authority_reason


def _cold_payloads(root, tickets):
    # Worker performs bounded file I/O only, never touches SQLite/model/state.
    accounting = {'file_reads': 0, 'io_bytes': 0}
    staged = {}
    try:
        for ticket in tickets:
            for pin in ticket['pinned']:
                text = _read_verified(pin['text_path'], root, pin['text_sha256'],
                                      pin['text_bytes'], accounting=accounting)
                values = tuple(_read_verified(v['path'], root, v['sha256'], v['bytes'],
                                             accounting=accounting) for v in pin['values'])
                staged[pin['record_id']] = (text, values)
        return staged, accounting, None
    except Exception as error:
        return {}, accounting, f'{type(error).__name__}: {error}'


def _release_payloads(root, payloads):
    accounting = {'file_reads': 0, 'io_bytes': 0}
    try:
        staged = {p['record_id']: _read_verified(p['path'], root, p['sha256'],
                    p['bytes'], accounting=accounting) for p in payloads}
        return staged, accounting, None
    except Exception as error:
        return {}, accounting, f'{type(error).__name__}: {error}'


def _descriptor(record):
    return {key: getattr(record, key) for key in (
        'record_id', 'text', 'values', 'source_id', 'source_epoch', 'source_provenance',
        'value_producer_id', 'value_producer_version', 'scope_key', 'text_sha256',
        'supersedes', 'cluster_id', 'cluster_generation', 'revision')}


class ContextualMemoryV5(ContextualMemoryLayoutV4):
    def __init__(self, *args, objective_identity=None, execution_identity='CPU_engineering_unbound', **kwargs):
        super().__init__(*args, **kwargs)
        self.objective_identity = objective_identity or {'scope': 'CPU_correctness_only'}
        if not isinstance(execution_identity,str) or not execution_identity:
            raise ContextualMemoryError('explicit execution identity required')
        self.execution_identity=execution_identity
        self._release_original = self.encoder.release
        self._release_cache = None
        self.encoder.release = self._producer_release
        # Fused AdamW does not necessarily advance Tensor._version. Invalidate
        # before its work (also on a partially failed update), not only by stamps.
        self._optimizer_release_hook = self.optimizer.register_step_pre_hook(self._before_step)
        self._pool = None
        self._jobs = {}
        self.release_corpus = None
        self.cost.update({k: 0 for k in ('producer_hash_calls', 'producer_hash_ns',
            'producer_hash_cache_hits', 'authority_validation_ns', 'cold_wait_ns',
            'cold_reserved_bytes', 'cold_reserved_files', 'cold_completed_bytes',
            'cold_completed_files', 'durable_write_bytes', 'durable_write_ns',
            'durable_artifact_growth_bytes', 'admitted_payload_bytes',
            'ticket_descriptor_read_bytes','ticket_descriptor_read_calls')})

    def _before_step(self, optimizer, args, kwargs):
        self._release_cache = None

    def _producer_release(self):
        stamp = tuple((name, value._version, value.data_ptr(), str(value.device),
                       str(value.dtype), tuple(value.shape))
                      for name, value in self.encoder.named_parameters())
        if self._release_cache is not None and self._release_cache[0] == stamp:
            self.cost['producer_hash_cache_hits'] += 1
            return self._release_cache[1]
        start = time.monotonic_ns()
        release = self._release_original()
        self.cost['producer_hash_calls'] += 1
        self.cost['producer_hash_ns'] += time.monotonic_ns()-start
        self._release_cache = (stamp, release)
        return release

    def training_values(self, spans, *, scope):
        return super().training_values(spans, scope=effective_scope(scope))

    def admit(self, store, span, *, scope, cluster='contextual', placement='warm'):
        span.validate()
        version = self.encoder.release()
        with torch.no_grad(): values = self._encode([span])[0].detach().cpu().tolist()
        draft = EvidenceRecordDraft(span.record_id, span.text, span.source_id, span.source_epoch,
            span.available_at, valid_to=span.valid_until, authenticity='source_report',
            verification_state='verified_source', values=tuple(tuple(v) for v in values),
            value_producer_id=PRODUCER, value_producer_version=version,
            source_lineage=(json.dumps(self.provenance(span, version), sort_keys=True, separators=(',', ':')),))
        start = time.monotonic_ns()
        before = sum(p.stat().st_size for p in store.root.rglob('*') if p.is_file())
        self.cost['admission_calls'] += 1
        self.cost['admitted_payload_bytes'] += len(span.text.encode()) + SLOTS*WIDTH*4
        try:
            result = store.admit_cluster(cluster, scope=scope, records=[draft], placement=placement)
        finally:
            self.cost['durable_write_ns'] += time.monotonic_ns()-start
            self.cost['durable_artifact_growth_bytes'] += max(0, sum(p.stat().st_size for p in store.root.rglob('*') if p.is_file())-before)
        return result

    def validate(self, batch, scope):
        if not isinstance(batch, ValueBatch):
            raise ContextualMemoryError('complete ValueBatch required; pending is not usable memory')
        start = time.monotonic_ns()
        try:
            super().validate(batch, effective_scope(scope))
            if (not batch.training and self.release_corpus is None
                    and any('producer_generation' in pin for pin in batch.pins)):
                raise ContextualMemoryError('published release authority is required for durable overlay values')
            if self.release_corpus is not None:
                self.release_corpus.validate_batch(batch, scope=scope)
        finally:
            self.cost['authority_validation_ns'] += time.monotonic_ns()-start

    def snapshot(self, ids, batch, *, scope, active_sites=None, padding_value=0.0):
        self.validate(batch, scope)
        slots = batch.slots + self.order(torch.arange(len(batch.slots), device=ids.device))
        count = len(slots)
        pad = torch.full((self.spec.slots-count, WIDTH), padding_value,
                         device=slots.device, dtype=slots.dtype)
        values = torch.cat((slots, pad))[None, None].expand(ids.shape[0], ids.shape[1], -1, -1)
        available = (torch.arange(self.spec.slots, device=ids.device) < count)
        available = available[None, None].expand(values.shape[:3])
        context = self.reader.snapshot(values, available,
            validate=lambda: self.validate(batch, scope), active_sites=active_sites)
        self.cost['snapshot_copies'] += 1
        self.cost['snapshot_bytes'] += context.values.numel()*context.values.element_size()
        self.cost['value_slots_read'] += count
        return context

    def _batch(self, records, store, scope):
        version = self.encoder.release()
        spans, values, pins = [], [], []
        for r in sorted(records, key=lambda r: (r['source_id'], r['source_epoch'],
                                               r['source_provenance'].get('source_order', 0))):
            p = r['source_provenance']
            if (r['scope_key'] != scope.scope_key or r['value_producer_id'] != PRODUCER
                    or p.get('record_id') != r['record_id'] or p.get('text_sha256') != r['text_sha256']
                    or p.get('supersedes') != r['supersedes']
                    or p.get('representation_version') != REPRESENTATION or p.get('codec_version') != CODEC):
                raise ContextualMemoryError('canonical value provenance differs')
            metadata = {**p, 'text': r['text'], 'source_id': r['source_id']}
            span = CanonicalSpan(**{k: metadata[k] for k in CanonicalSpan.__dataclass_fields__}).validate()
            if self.provenance(span, p.get('producer_revision')) != p:
                raise ContextualMemoryError('slot provenance differs')
            pin = {'record_id': r['record_id'], 'cluster_id': r['cluster_id'],
                   'generation': r['cluster_generation'], 'revision': r['revision'],
                   'text_sha256': r['text_sha256']}
            if self.release_corpus is None:
                if r['value_producer_version'] != version or p.get('producer_revision') != version:
                    raise ContextualMemoryError('stale producer release')
                value = torch.tensor(r['values'], dtype=torch.float32, device=self.order.weight.device)
            else:
                value, release_pin = self.release_corpus.values(pin, span, version=version, scope=scope,
                                                               payload=r.get('release_payload'))
                pin.update(release_pin)
                value = value.to(self.order.weight.device)
            if tuple(value.shape) != (SLOTS, WIDTH):
                raise ContextualMemoryError('stored slot shape differs')
            pin['slots_sha256'] = tensor_digest(value)
            pins.append(pin); spans.append(span); values.append(value)
        if not values:
            raise ContextualMemoryError('no eligible contextual values')
        batch = ValueBatch(torch.cat(values).detach(), tuple(spans), scope.scope_key, version,
                           pins=tuple(pins), store=store)
        self.validate(batch, scope)
        return batch

    def read(self, store, question, *, scope, wait_seconds=0.0):
        """Return a complete batch or a typed ticket; never expose partial values."""
        self._wait_bound(wait_seconds)
        if self.release_corpus is not None:
            if self.release_corpus.store is not store:
                raise ContextualMemoryError('producer corpus belongs to a different store')
            return self.release_corpus.read(self, question, scope=scope, wait_seconds=wait_seconds)
        self.cost['retrieval_calls'] += 1
        start = time.monotonic_ns()
        budget = ReadoutBudget(max_records=self.spec.slots//SLOTS, max_value_slots=self.spec.slots,
                               cold_wait_seconds=0.0, cache_bytes=0)
        result = read_scoped_memory(store, question, scope, budget)
        costs = {'retrieval_wall_ns': time.monotonic_ns()-start,
                 'read_file_bytes': result.metrics.total_io_bytes,
                 'read_file_calls': result.metrics.total_file_reads,
                 'cold_reserved_bytes': 0, 'cold_reserved_files': 0,
                 'cold_completed_bytes': 0, 'cold_completed_files': 0, 'cold_wait_ns': 0}
        for k, v in costs.items(): self.cost[k] += v
        if result.new_decision_boundary or result.truncated:
            raise ContextualReadFailure('incomplete or invalidated selection', costs)
        if not result.pending_cold:
            if result.status not in ('ok', 'degraded') or any(r.status != 'ok' for r in result.records):
                raise ContextualReadFailure('incomplete contextual read', costs)
            batch = self._batch([_descriptor(r) for r in result.records], store, scope)
            batch.read_costs = dict(costs)
            return batch
        pending_ids = {p.record_id for p in result.pending_cold}
        if any(r.status != 'ok' and r.record_id not in pending_ids for r in result.records):
            raise ContextualReadFailure('partial selection refused', costs)
        token = uuid.uuid4().hex
        tickets = [asdict(store.ticket(p.ticket_id)) for p in result.pending_cold]
        doc = {'format': FORMAT+'.pending', 'id': token, 'store_root': str(store.root.resolve()),
               'scope': asdict(scope), 'records': [_descriptor(r) for r in result.records],
               'tickets': tickets, 'costs': costs, 'status': 'pending', 'attempts': 0,
               'producer_release': self.encoder.release(), 'max_attempts': 2}
        self._persist(store, token, doc)
        pending = self._pending(doc)
        return self.resume(store, pending, scope=scope, wait_seconds=wait_seconds) if wait_seconds else pending

    def begin_release_read(self, store, scope, records, payloads, pointer, *, wait_seconds):
        token = uuid.uuid4().hex
        costs = {k: 0 for k in ('cold_reserved_bytes', 'cold_reserved_files',
            'cold_completed_bytes', 'cold_completed_files', 'cold_wait_ns',
            'read_file_bytes', 'read_file_calls')}
        doc = {'format': FORMAT+'.pending', 'id': token, 'store_root': str(store.root.resolve()),
               'scope': asdict(scope), 'records': records, 'release_payloads': payloads,
               'release_pointer': pointer, 'costs': costs, 'status': 'pending',
               'attempts': 0, 'producer_release': self.encoder.release(), 'max_attempts': 2}
        self._persist(store, token, doc)
        pending = self._pending(doc)
        return self.resume(store, pending, scope=scope, wait_seconds=wait_seconds) if wait_seconds else pending

    @staticmethod
    def _wait_bound(seconds):
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 <= seconds <= 5:
            raise ContextualMemoryError('cold wait must be bounded to 0..5 seconds')

    @staticmethod
    def _ticket_path(store, token):
        if len(token) != 32 or any(c not in '0123456789abcdef' for c in token):
            raise ContextualMemoryError('invalid contextual ticket identity')
        return store.root/'artifacts'/'contextual_v5'/'tickets'/(token+'.json')

    def _persist(self, store, token, doc):
        path = self._ticket_path(store, token)
        # Charge the descriptor write before it happens. Persisting the cumulative
        # reservation makes timeout/restart/failure cost visible to the caller.
        costs = doc['costs']
        previous = costs.get('ticket_write_bytes_reserved', 0)
        costs['ticket_write_calls'] = costs.get('ticket_write_calls', 0)+1
        size = 0
        for _ in range(10):
            costs['ticket_write_bytes_reserved'] = previous+size
            actual = len(json.dumps(doc, sort_keys=True, indent=2).encode())+1
            if actual == size: break
            size = actual
        # Ticket overlap and retained descriptor are charged to existing cold quota.
        store._charge('cold', 'artifacts', size, 'contextual_ticket:'+token+':write', 'pending descriptor')
        start = time.monotonic_ns()
        try:
            atomic_json(path, doc)
        finally:
            store._release('contextual_ticket:'+token+':write')
        store._set_component_locked('contextual_ticket:'+token, 'cold', 'artifacts', size, 'pending descriptor')
        self.cost['durable_write_bytes'] += size
        self.cost['durable_write_ns'] += time.monotonic_ns()-start

    @staticmethod
    def _pending(doc):
        return PendingContextualRead(doc['id'], doc['store_root'], doc['scope'], digest(doc), dict(doc['costs']))

    def recover_pending(self, store, ticket_id, *, scope):
        """Recover the latest committed descriptor after an interrupted wait.

        This does not dispatch I/O or reset uncertain reservations. Original E
        authority is checked again after any resumed payload read.
        """
        doc,size=self._load_ticket(store,ticket_id)
        if (doc.get('format')!=FORMAT+'.pending' or doc.get('id')!=ticket_id
                or doc.get('store_root')!=str(store.root.resolve()) or doc.get('scope')!=asdict(scope)
                or doc.get('producer_release')!=self.encoder.release() or doc.get('status')!='pending'):
            raise ContextualReadFailure('pending recovery authority differs',doc.get('costs',{}))
        self._charge_ticket_read(doc,size);self._persist(store,ticket_id,doc)
        return self._pending(doc)

    def _load_ticket(self,store,ticket_id):
        path=self._ticket_path(store,ticket_id)
        if path.is_symlink() or not path.is_file() or path.stat().st_size>256*1024:
            raise ContextualMemoryError('missing/substituted/oversized pending descriptor')
        payload=path.read_bytes()
        self.cost['ticket_descriptor_read_bytes']+=len(payload)
        self.cost['ticket_descriptor_read_calls']+=1
        return json.loads(payload),len(payload)

    @staticmethod
    def _charge_ticket_read(doc,size):
        doc['costs']['ticket_descriptor_read_bytes']=doc['costs'].get('ticket_descriptor_read_bytes',0)+size
        doc['costs']['ticket_descriptor_read_calls']=doc['costs'].get('ticket_descriptor_read_calls',0)+1

    def resume(self, store, pending, *, scope, wait_seconds=0.0):
        self._wait_bound(wait_seconds)
        doc,size = self._load_ticket(store,pending.ticket_id)
        if (doc['format'] != FORMAT+'.pending' or doc['store_root'] != str(store.root.resolve())
                or pending.store_root != doc['store_root'] or doc['scope'] != asdict(scope)
                or pending.scope != doc['scope'] or doc['producer_release'] != self.encoder.release()
                or digest(doc) != pending.descriptor_sha256):
            raise ContextualReadFailure('pending scope/release/descriptor identity differs', doc['costs'])
        if doc['status'] != 'pending':
            raise ContextualReadFailure('ticket already terminal', doc['costs'])
        self._charge_ticket_read(doc,size)
        if wait_seconds == 0:
            self._persist(store,pending.ticket_id,doc)
            return self._pending(doc)
        if pending.ticket_id not in self._jobs:
            # Reserve the maximum I/O before dispatch. If the process dies, its
            # reservation remains charged as uncertain work; a retry adds a new one.
            release = doc.get('release_payloads')
            byte_count = sum(p['bytes'] for p in release) if release else sum(t['required_bytes'] for t in doc['tickets'])
            if doc['attempts'] >= doc['max_attempts'] or any(t['required_bytes'] > t['max_bytes'] for t in doc.get('tickets', [])):
                raise ContextualReadFailure('cold attempt/byte budget exhausted', doc['costs'])
            files = len(release) if release else sum(1+len(p['values']) for t in doc['tickets'] for p in t['pinned'])
            for k, amount in [('cold_reserved_bytes', byte_count), ('cold_reserved_files', files)]:
                doc['costs'][k] += amount; self.cost[k] += amount
            doc['attempts'] += 1
            self._persist(store, pending.ticket_id, doc)
            if self._pool is None: self._pool = ThreadPoolExecutor(max_workers=1)
            fn, args = (_release_payloads, release) if release else (_cold_payloads, doc['tickets'])
            self._jobs[pending.ticket_id] = self._pool.submit(fn, store.root, args)
        start = time.monotonic_ns()
        try:
            staged, io, error = self._jobs[pending.ticket_id].result(timeout=wait_seconds)
        except TimeoutError:
            elapsed = time.monotonic_ns()-start
            doc['costs']['cold_wait_ns'] += elapsed; self.cost['cold_wait_ns'] += elapsed
            self._persist(store, pending.ticket_id, doc)
            return self._pending(doc)
        elapsed = time.monotonic_ns()-start
        doc['costs']['cold_wait_ns'] += elapsed; self.cost['cold_wait_ns'] += elapsed
        self._jobs.pop(pending.ticket_id)
        for k, v in [('cold_completed_bytes', io['io_bytes']), ('cold_completed_files', io['file_reads']),
                     ('read_file_bytes', io['io_bytes']), ('read_file_calls', io['file_reads'])]:
            doc['costs'][k] += v; self.cost[k] += v
        try:
            if error: raise ContextualMemoryError(error)
            if 'release_payloads' in doc:
                if self.release_corpus is None or self.release_corpus._current()[0] != doc['release_pointer']:
                    raise ContextualMemoryError('producer publication invalidated pending read')
                descriptors = [{**r, 'release_payload': staged[r['record_id']]} for r in doc['records']]
                batch = self._batch(descriptors, store, scope)
                doc['status'] = 'complete'
                self._persist(store, pending.ticket_id, doc)
                batch.read_costs = dict(doc['costs'])
                return batch
            updates = {}
            for t in doc['tickets']:
                ticket = RecallTicket(**{**t, 'scope': MemoryScope(**t['scope']),
                                          'outcome': Outcome(**t['outcome'])})
                status, reason, checks = store._validate_after_io(ticket, staged)
                if status != 'ready': raise ContextualMemoryError(reason)
                records = tuple(RecalledRecord(record_id=rid, text=staged[rid][0].decode(),
                    text_sha256=hashlib.sha256(staged[rid][0]).hexdigest(),
                    values=unpack_values(staged[rid][1]), value_slots=SLOTS,
                    value_sha256=tuple(hashlib.sha256(v).hexdigest() for v in staged[rid][1]),
                    placement='cold') for rid in ticket.record_ids)
                ready = RecallResult(ticket_id=ticket.ticket_id, selection_id=ticket.selection_id,
                    status='ready', outcome=Outcome('ok', 'bounded_cold_ready'), records=records,
                    io_bytes=sum(len(staged[rid][0])+sum(map(len,staged[rid][1])) for rid in ticket.record_ids),
                    checks=checks, new_decision_boundary=False,
                    file_reads=sum(1+len(staged[rid][1]) for rid in ticket.record_ids))
                store._tickets[ticket.ticket_id] = {'ticket': replace(ticket, status='ready')}
                store._ticket_results[ticket.ticket_id] = ready
                result = resume_scoped_readout(store, ticket.ticket_id, scope=scope,
                    budget=ReadoutBudget(max_records=2, max_value_slots=8, cache_bytes=0))
                if not result.ok or result.new_decision_boundary or any(r.status != 'ok' for r in result.records):
                    raise ContextualMemoryError('post-I/O read invalidated')
                updates.update({r.record_id: _descriptor(r) for r in result.records})
            descriptors = [updates.get(r['record_id'], r) for r in doc['records']]
            batch = self._batch(descriptors, store, scope)
        except Exception as error:
            doc['status'] = 'failed'; doc['reason'] = str(error)
            self._persist(store, pending.ticket_id, doc)
            raise ContextualReadFailure(str(error), doc['costs']) from error
        doc['status'] = 'complete'
        self._persist(store, pending.ticket_id, doc)
        batch.read_costs = dict(doc['costs'])
        return batch

    def save(self, directory):
        directory = Path(directory); directory.mkdir(exist_ok=False, parents=True)
        inner = super().save(directory/'components')
        manifest = {'format': FORMAT+'.checkpoint', 'objective': self.objective_identity,
                    'execution_identity':self.execution_identity,
                    'slot_policy': SLOT_POLICY, 'temporal_rule': 'MemoryScope.cutoff',
                    'components_manifest_sha256': file_hash(directory/'components'/'manifest.json')}
        atomic_json(directory/'manifest.json', manifest)
        return {**manifest, 'components': inner}

    def load(self, directory):
        directory = Path(directory); file_hash(directory/'manifest.json')
        m = json.loads((directory/'manifest.json').read_text())
        expected = {'format': FORMAT+'.checkpoint', 'objective': self.objective_identity,
                    'execution_identity':self.execution_identity,
                    'slot_policy': SLOT_POLICY, 'temporal_rule': 'MemoryScope.cutoff'}
        if any(m.get(k) != v for k, v in expected.items()) or m.get('components_manifest_sha256') != file_hash(directory/'components'/'manifest.json'):
            raise ContextualMemoryError('v5 objective/slots/checkpoint identity differs')
        result = super().load(directory/'components')
        self._release_cache = None
        return {**m, 'components': result}

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
