"""Immutable producer-release overlays over existing canonical evidence authority.

A release never rewrites E records or historical values. It re-encodes current,
scoped canonical sources and atomically publishes a digest-bound generation.
Interrupted preparation leaves the previous committed generation usable. Readers
refuse a different/mixed release and revalidate original E authority at each site.
"""
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import torch

from .contextual_memory import CanonicalSpan, ContextualMemoryError, CODEC, REPRESENTATION, tensor_digest
from .contextual_memory_v5 import atomic_json, digest, file_hash
from .tiered_memory import _read_verified, pack_values, unpack_values, RetrievalBudget, MemoryScope

FORMAT = 'replica.contextual_producer_release.v5'


class ProducerReleaseCorpus:
    def __init__(self, store, scope, *, max_records=1024):
        self.store, self.scope, self.max_records = store, scope, max_records
        self.root = store.root/'artifacts'/'contextual_v5'/'releases'/scope.scope_key
        self.root.mkdir(parents=True, exist_ok=True)
        self.cost = dict(source_read_bytes=0, source_read_files=0, write_bytes=0,
                         write_ns=0, read_bytes=0, read_files=0, publications=0,
                         authority_read_bytes=0, authority_read_files=0)
        # Generation zero is explicit authority, so an interrupted first build
        # can resume without confusing a lost committed pointer with an empty
        # corpus. Never reconstruct authority from whichever payload survived.
        lock = self.root/'writer.lock'
        if lock.is_symlink(): raise ContextualMemoryError('substituted release lock')
        with lock.open('a') as locked:
            fcntl.flock(locked, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pointer = self.root/'current.json'
            if not pointer.exists():
                if any(p.is_dir() and p.name.isdigit() for p in self.root.iterdir()):
                    raise ContextualMemoryError('committed producer pointer missing; recovery required')
                origin={'format': FORMAT+'.pointer', 'scope': asdict(scope),
                        'generation': 0, 'manifest_sha256': None}
                size=len(json.dumps(origin,sort_keys=True,indent=2).encode())+1
                store._charge('cold','artifacts',size,'release_pointer:'+scope.scope_key,'release origin')
                atomic_json(pointer,origin);self.cost['write_bytes']+=size
        self._manifest_cache = None

    def _authority(self,path):
        if path.is_symlink() or not path.is_file() or path.stat().st_size>4*1024**2:
            raise ContextualMemoryError('missing/substituted/oversized release authority')
        payload=path.read_bytes()
        self.cost['authority_read_bytes']+=len(payload);self.cost['authority_read_files']+=1
        return hashlib.sha256(payload).hexdigest(),payload

    def _write(self, path, payload):
        component = 'release_v5:'+str(path.relative_to(self.store.root))
        self.store._charge('cold', 'artifacts', len(payload), component, 'immutable producer release')
        start = time.monotonic_ns()
        with path.open('xb') as out:
            out.write(payload); out.flush()
            import os
            os.fsync(out.fileno())
        self.cost['write_bytes'] += len(payload)
        self.cost['write_ns'] += time.monotonic_ns()-start

    def _current(self):
        pointer = self.root/'current.json'
        if not pointer.exists(): raise ContextualMemoryError('producer pointer missing')
        _,payload=self._authority(pointer);pin=json.loads(payload)
        if pin.get('format') != FORMAT+'.pointer' or MemoryScope(**pin['scope']).scope_key != self.scope.scope_key:
            raise ContextualMemoryError('release pointer scope/format differs')
        generation = pin.get('generation')
        if type(generation) is int and generation == 0 and pin.get('manifest_sha256') is None: return None
        if type(generation) is not int or generation < 1:
            raise ContextualMemoryError('release generation invalid')
        path = self.root/f'{generation:08d}'/'manifest.json'
        identity,payload = self._authority(path)
        if identity != pin.get('manifest_sha256'):
            raise ContextualMemoryError('committed producer authority missing/corrupt')
        if self._manifest_cache is None or self._manifest_cache[0] != identity:
            m = json.loads(payload)
            if (m.get('format') != FORMAT or m.get('generation') != generation
                    or MemoryScope(**m['scope']).scope_key != self.scope.scope_key or m.get('representation') != REPRESENTATION
                    or m.get('codec') != CODEC or m.get('record_digest') != digest(m.get('records'))):
                raise ContextualMemoryError('release manifest identity differs')
            if any(r['producer_release'] != m['producer_release'] for r in m['records'].values()):
                raise ContextualMemoryError('mixed producer releases refused')
            self._manifest_cache = (identity, m)
        return pin, self._manifest_cache[1]

    def _source(self, row):
        source = None
        for item in json.loads(row['source_lineage']):
            try:
                value = json.loads(item)
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict) and value.get('record_id') == row['record_id']: source = value
        if source is None: raise ContextualMemoryError('canonical provenance missing')
        text = _read_verified(row['text_path'], self.store.root, row['text_sha256'], row['text_bytes'],
                              accounting=None).decode()
        self.cost['source_read_bytes'] += row['text_bytes']; self.cost['source_read_files'] += 1
        data = {**source, 'source_id': row['source_id'], 'text': text}
        span = CanonicalSpan(**{k: data[k] for k in CanonicalSpan.__dataclass_fields__}).validate()
        pin = {'record_id': row['record_id'], 'cluster_id': row['cluster_id'],
               'generation': row['cluster_generation'], 'revision': row['revision'],
               'text_sha256': row['text_sha256']}
        return span, pin, source

    def _validate_source_pin(self, pin):
        row = self.store.connection.execute('SELECT r.*,c.current_generation FROM records r JOIN clusters c ON r.cluster_id=c.cluster_id WHERE r.record_id=?',
                                            (pin['record_id'],)).fetchone()
        cutoff = self.scope.cutoff
        if (row is None or row['status'] != 'current' or row['scope_key'] != self.scope.scope_key
                or row['revision'] != pin['revision'] or row['text_sha256'] != pin['text_sha256']
                or row['current_generation'] != pin['generation'] or row['observed_at'] > cutoff
                or row['valid_from'] is not None and row['valid_from'] > cutoff
                or row['valid_to'] is not None and row['valid_to'] <= cutoff):
            raise ContextualMemoryError('corrected/revoked/time/generation source invalidated release')

    def rebuild(self, path, *, before_publish=None):
        """One explicit release transaction; hook supports portable crash testing."""
        if path.source_identity == '': raise ContextualMemoryError('source identity missing')
        lock = self.root/'writer.lock'
        if lock.is_symlink(): raise ContextualMemoryError('substituted release lock')
        with lock.open('a') as locked:
            fcntl.flock(locked, fcntl.LOCK_EX | fcntl.LOCK_NB)
            old = self._current()
            # Orphaned preparation directories remain preserved and are never reused.
            existing = [int(p.name) for p in self.root.iterdir() if p.is_dir() and p.name.isdigit()]
            generation = max(existing, default=0)+1
            dest = self.root/f'{generation:08d}'; dest.mkdir()
            version = path.encoder.release()
            rows = self.store._eligible_record_rows(self.scope)
            if not rows or len(rows) > self.max_records: raise ContextualMemoryError('release corpus bound/empty')
            records = {}
            for row in rows:
                span, pin, source = self._source(row)
                if path.provenance(span, source.get('producer_revision')) != source:
                    raise ContextualMemoryError('canonical provenance differs')
                with torch.no_grad(): values = path._encode([span])[0].detach().cpu()
                payload = b''.join(pack_values(values.tolist()))
                name = hashlib.sha256(span.record_id.encode()).hexdigest()+'.slots'
                self._write(dest/name, payload)
                records[span.record_id] = {'source_pin': pin, 'source_provenance': source,
                    'span': asdict(span), 'producer_release': version, 'value_file': name,
                    'value_bytes': len(payload), 'value_sha256': hashlib.sha256(payload).hexdigest()}
            if path.encoder.release() != version: raise ContextualMemoryError('producer changed during rebuild')
            m = {'format': FORMAT, 'generation': generation, 'scope': asdict(self.scope),
                 'parent': old[0] if old else None, 'producer_release': version,
                 'source_identity': path.source_identity, 'representation': REPRESENTATION,
                 'codec': CODEC, 'records': records, 'record_digest': digest(records)}
            self._write(dest/'manifest.json', (json.dumps(m,sort_keys=True,indent=2)+'\n').encode())
            fd = os.open(dest, os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
            if before_publish is not None: before_publish()
            # E's SQLite write lock closes the authority-change race at publication.
            with self.store._transaction():
                if path.encoder.release() != version: raise ContextualMemoryError('producer changed before publication')
                for record in records.values(): self._validate_source_pin(record['source_pin'])
                pointer = {'format': FORMAT+'.pointer', 'scope': asdict(self.scope),
                           'generation': generation, 'manifest_sha256': file_hash(dest/'manifest.json')}
                pointer_bytes = len(json.dumps(pointer,sort_keys=True,indent=2).encode())+1
                self.store._ensure_capacity_locked('cold', 'artifacts', pointer_bytes*2)
                atomic_json(self.root/'current.json', pointer)
                self.store._set_component_locked('release_pointer:'+self.scope.scope_key,'cold','artifacts',pointer_bytes,'release authority')
            self.cost['write_bytes'] += pointer_bytes
            self.cost['publications'] += 1
            return m

    def values(self, pin, span, *, version, scope, payload=None):
        if scope != self.scope: raise ContextualMemoryError('release scope/time differs')
        current = self._current()
        if current is None: raise ContextualMemoryError('no committed producer release')
        pointer, m = current
        record = m['records'].get(span.record_id)
        if (m['producer_release'] != version or record is None or record['source_pin'] != pin
                or record['span'] != asdict(span)):
            raise ContextualMemoryError('missing/mixed/stale producer release corpus')
        self._validate_source_pin(pin)
        p = self.root/f"{m['generation']:08d}"/record['value_file']
        if p.name != record['value_file'] or p.is_symlink(): raise ContextualMemoryError('substituted release payload')
        if payload is None:
            payload = p.read_bytes(); self.cost['read_bytes'] += len(payload); self.cost['read_files'] += 1
        else:
            payload = bytes(payload)
        if len(payload) != record['value_bytes'] or hashlib.sha256(payload).hexdigest() != record['value_sha256']:
            raise ContextualMemoryError('release value digest differs')
        values = torch.tensor(unpack_values(tuple(payload[i:i+1024] for i in range(0,len(payload),1024))))
        return values, {'producer_generation': m['generation'], 'producer_manifest': pointer['manifest_sha256']}

    def validate_batch(self, batch, *, scope):
        # Live training sources may originate from an older release. Their E
        # authority is still checked; inference must bind the published release.
        if batch.training: return
        current = self._current()
        if current is None: raise ContextualMemoryError('producer authority unavailable')
        pointer, m = current
        if scope != self.scope or batch.producer_version != m['producer_release']:
            raise ContextualMemoryError('mixed producer release batch')
        for pin in batch.pins:
            if pin.get('producer_generation') != m['generation'] or pin.get('producer_manifest') != pointer['manifest_sha256']:
                raise ContextualMemoryError('producer generation invalidated batch')
            self._validate_source_pin(pin)

    def read(self, path, question, *, scope, wait_seconds=0.0):
        result = self.store.select(question, scope=scope, budget=RetrievalBudget(max_records=2,max_value_slots=8),
                                   include_text=False, admit_payload=False)
        path.cost['retrieval_calls'] += 1
        path.cost['read_file_bytes'] += result.io_bytes; path.cost['read_file_calls'] += result.file_reads
        try:
            if result.truncated or not result.outcome.ok: raise ContextualMemoryError('incomplete release selection')
            current = self._current()
            if current is None: raise ContextualMemoryError('no committed producer release')
            pointer, m = current
            if m['producer_release'] != path.encoder.release():
                raise ContextualMemoryError('stale producer release corpus')
            records = []
            payloads = []
            for r in result.records:
                entry = m['records'].get(r.record_id)
                if entry is None: raise ContextualMemoryError('unrebuilt source in release selection')
                span = CanonicalSpan(**entry['span']); pin = entry['source_pin']
                self._validate_source_pin(pin)
                records.append({'record_id': r.record_id, 'text': span.text, 'values': None,
                    'source_id': span.source_id, 'source_epoch': span.source_epoch,
                    'source_provenance': entry['source_provenance'], 'value_producer_id': 'contextual_byte_transformer_v2',
                    'value_producer_version': m['producer_release'], 'scope_key': scope.scope_key,
                    'text_sha256': pin['text_sha256'], 'supersedes': span.supersedes,
                    'cluster_id': pin['cluster_id'], 'cluster_generation': pin['generation'], 'revision': pin['revision']})
                payloads.append({'record_id': r.record_id,
                    'path': str((self.root/f"{m['generation']:08d}"/entry['value_file']).relative_to(self.store.root)),
                    'sha256': entry['value_sha256'], 'bytes': entry['value_bytes']})
            if not records: raise ContextualMemoryError('no eligible release sources')
            # Release artifacts are disk-backed, regardless of E's placement.
            # Route these through the same bounded pending API, never a hidden
            # synchronous cold payload read at a decoder injection site.
            return path.begin_release_read(self.store, scope, records, payloads,
                                            pointer, wait_seconds=wait_seconds)
        finally:
            self.store.release_selection(result.selection_id)
