"""Representation-bound incremental SGDN-2 state and explicit rebuild semantics."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import torch

from ..architecture.contracts import FastMemoryRead, fingerprint
from ..memory_system import EligibilityScope, MemoryRecord
from ..neural import FastState, SGDN2Memory
from .fast_memory import FastMemoryConfig, save_fast_depth_image, load_fast_depth_image
from .memory_representation import (FACT_REPRESENTATION, RAW_REPRESENTATION, PERSISTENT_FORMAT,
                                    encoding_dependencies, encoding_producer, parse_fact_source,
                                    representation_configuration)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in '0123456789abcdef' for char in value))


def read_persistent_manifest(path):
    """Authenticate the wrapper manifest; callers must also verify producer pins."""
    value = json.loads((Path(path) / 'persistent_manifest.json').read_text())
    fields = {'format', 'representation', 'representation_config', 'dependencies', 'engine_config',
              'model_version', 'producer', 'events', 'semantics', 'migration',
              'depth_manifest_sha256', 'source_binding_sha256', 'manifest_digest'}
    if (set(value) != fields or value['format'] != PERSISTENT_FORMAT
            or value['representation_config'] != representation_configuration(value['representation'])
            or type(value['representation_config']['version']) is not int
            or value['semantics'] not in {'incremental_history', 'history', 'current_view'}):
        raise ValueError('Persistent representation metadata is incompatible')
    body = {key: item for key, item in value.items() if key != 'manifest_digest'}
    if value['manifest_digest'] != fingerprint(body):
        raise ValueError('Persistent manifest digest differs')
    if _sha(Path(path) / 'manifest.json') != value['depth_manifest_sha256']:
        raise ValueError('Persistent depth manifest bytes differ')
    binding = Path(path) / 'source_binding.json'
    if ((_sha(binding) if binding.is_file() else None) != value['source_binding_sha256']):
        raise ValueError('Persistent source-binding bytes differ')
    return value


class PersistentFastMemory:
    def __init__(self, model, tokenizer, scope, *, training=False,
                 representation=RAW_REPRESENTATION, config=None):
        self.model, self.tokenizer, self.scope, self.training = model, tokenizer, scope, training
        self.representation = representation
        self.representation_config = representation_configuration(representation)
        spec = model.manifest.fast_memory
        if spec.depth_mode != 'shared' or not spec.depth_writers:
            raise ValueError('Persistent state requires shared per-depth SGDN-2')
        self.config = config or FastMemoryConfig(
            heads=spec.heads, key_size=spec.key_size, value_size=spec.value_size,
            update_rule='sgdn2', max_tokens_per_chunk=256, max_image_bytes=1048576)
        if (self.config.heads, self.config.key_size, self.config.value_size, self.config.update_rule) != (
                spec.heads, spec.key_size, spec.value_size, 'sgdn2'):
            raise ValueError('Persistent engine configuration differs from native memory')
        self.primitive = SGDN2Memory(spec.heads, spec.key_size, spec.value_size)
        self.device = model.base.get_input_embeddings().weight.device
        self.state = self.primitive.initial_state(1, device=self.device)
        self.dependencies = encoding_dependencies(model, tokenizer, representation)
        self.model_producer = model.fast_producer_key()
        self._legacy = False
        self.producer = self._producer()
        self.lineage, self.events = [], []
        self.semantics = 'incremental_history'
        self.migration = None
        self.source_binding = None
        self._source_reference = self._source_scope = None

    def _producer(self):
        key = encoding_producer(self.model.fast_producer_key(), self.representation,
                                self.dependencies, self.config)
        return fingerprint({'legacy_read_only': key}) if self._legacy else key

    def _check_producer(self):
        if (self.representation_config != representation_configuration(self.representation)
                or encoding_dependencies(self.model, self.tokenizer, self.representation) != self.dependencies
                or self._producer() != self.producer):
            raise ValueError('State producer/representation changed; re-encode explicitly')

    def _mutable(self):
        if self._legacy:
            raise ValueError('Legacy snapshot is read-only; migrate by explicit source re-encode')
        self._check_producer()

    def _source_owner(self, scope):
        if not isinstance(scope, EligibilityScope):
            raise TypeError('An independently derived source scope is required')
        if (scope.subject_id, scope.branch_id, scope.purpose, scope.mode) != (
                self.scope.subject_id, self.scope.branch_id, self.scope.purpose, self.scope.mode):
            raise PermissionError('Source view scope differs from the persistent state owner')


    def _authorize_source_view(self, reference, view):
        """Prove the consumed lineage is still authorized by this exact view.

        Record-backed events must match an eligible record byte-for-byte, or be
        superseded originals whose eligible correction this state consumed later.
        Unbacked observation events must not repeat revoked source text, and the
        state must already reflect every eligible correction in the view. A
        metadata-only rebind can therefore never re-authorize stale contents:
        only consuming the authoritative change (append) or an explicit verified
        rebuild can. Unbacked events are text-authorized, so revocation inside a
        larger observed page is detected only at that page's own granularity.
        """
        ledger = getattr(reference, 'ledger', None)
        if ledger is None:
            raise TypeError('Source binding requires the authoritative exact reference')
        eligible = {record.record_id: record for record in view.records}
        superseding = {record.supersedes: record for record in view.records
                       if record.supersedes is not None}
        positions = {event_id: index for index, (event_id, _) in enumerate(self.lineage)}
        revoked = set(ledger.revoked_source_hashes())
        for (event_id, source_hash), event in zip(self.lineage, self.events):
            record = eligible.get(event_id)
            if event['record_hash'] is not None:
                if record is not None:
                    if (source_hash != record.source_hash
                            or event['record_hash'] != record.record_hash
                            or event['supersedes'] != record.supersedes
                            or event['correction'] != (record.supersedes is not None)):
                        raise ValueError('Consumed record no longer matches the authoritative view')
                    continue
                correction = superseding.get(event_id)
                if (correction is None
                        or positions.get(correction.record_id, -1) <= positions[event_id]):
                    raise ValueError('Consumed record is absent from the authoritative view; '
                                     'rebuild before rebinding')
                status = ledger.record_status(event_id)
                if (status is None or status['revoked']
                        or (status['source_hash'], status['record_hash'], status['supersedes'])
                        != (source_hash, event['record_hash'], event['supersedes'])):
                    raise ValueError('Consumed superseded record no longer matches the ledger')
            else:
                if source_hash in revoked:
                    raise ValueError('Consumed observation repeats revoked source text; '
                                     'rebuild before rebinding')
        for correction in superseding.values():
            if not any(event_id == correction.record_id
                       or source_hash == correction.source_hash
                       for event_id, source_hash in self.lineage):
                raise ValueError('Authoritative view contains corrections this state never '
                                 'consumed; append and rebind or rebuild')

    def bind_source(self, reference, *, scope):
        """Fetch a fresh authoritative view using the caller's independent scope."""
        from .exact_memory import ExactMemoryReference
        self._mutable()
        self._source_owner(scope)
        if not isinstance(reference, ExactMemoryReference):
            raise TypeError('Source binding requires the authoritative exact reference')
        view = reference.view(scope)
        self._authorize_source_view(reference, view)
        self.source_binding = {'ledger_id': view.ledger_id, 'epoch': view.epoch,
                               'view_digest': view.digest}
        self._source_reference, self._source_scope = reference, scope
        return dict(self.source_binding)

    def _check_source(self):
        if self.source_binding is not None:
            view = self._source_reference.view(self._source_scope)
            if any(self.source_binding[name] != actual for name, actual in (
                    ('ledger_id', view.ledger_id), ('epoch', view.epoch), ('view_digest', view.digest))):
                raise ValueError('Persistent source view is stale; rebuild or append and rebind')

    def _clear_binding(self):
        self.source_binding = None
        self._source_reference = self._source_scope = None

    def _record(self, record):
        if not isinstance(record, MemoryRecord):
            raise TypeError('Rebuild requires immutable source MemoryRecord values')
        if (record.subject_id != self.scope.subject_id or record.purpose != self.scope.purpose
                or record.branch_id not in {'source', self.scope.branch_id}):
            raise PermissionError('Source record is outside the persistent owner scope')
        return record

    def _append(self, event_id, text, *, fact=None, correction=False, record=None):
        self._mutable()
        if self.state.chunk_index >= self.config.max_chunks:
            raise ValueError('Persistent chunk quota exceeded')
        if not isinstance(event_id, str) or not event_id or event_id in {item[0] for item in self.lineage}:
            raise ValueError('Observation event IDs must be fresh')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Observed source text must be nonempty')
        ids = self.tokenizer(text, add_special_tokens=False)['input_ids']
        if not ids or len(ids) > self.config.max_tokens_per_chunk:
            raise ValueError('Observation must fit without silent truncation')
        from .fact_memory import encode_fact_binding
        with torch.set_grad_enabled(self.training):
            proposals = {}
            for layer in self.model.ports.injection_layers:
                write = (encode_fact_binding(self.model, self.tokenizer, fact, layer=layer)
                         if self.representation == FACT_REPRESENTATION else
                         self.model.encode_fast_observation(text, self.tokenizer, layer=layer))
                proposals[layer] = (replace(write, chunk_index=self.state.chunk_index),)
            state = self.primitive.reduce_shared_chunk(self.state, proposals)
        source_hash = _digest_text(text)
        event = {'event_id': event_id, 'source_hash': source_hash, 'correction': correction,
                 'record_hash': record.record_hash if record is not None else None,
                 'supersedes': record.supersedes if record is not None else None}
        self.state = state
        self.lineage.append((event_id, source_hash))
        self.events.append(event)
        self._clear_binding()

    def append(self, event_id, text):
        if self.representation != RAW_REPRESENTATION:
            raise ValueError('Raw append cannot mix with fact-binding representation')
        self._append(event_id, text)

    def append_fact(self, event_id, fact, *, correction=False):
        if self.representation != FACT_REPRESENTATION:
            raise ValueError('Fact append requires explicit fact-binding representation')
        if type(correction) is not bool:
            raise ValueError('Fact correction flag must be boolean')
        from .fact_memory import fact_source_text
        text = fact_source_text(fact, correction=correction)
        self._append(event_id, text, fact=fact, correction=correction)

    def append_record(self, record):
        record = self._record(record)
        if self.representation == FACT_REPRESENTATION:
            fact, correction = parse_fact_source(record.text)
            if correction != (record.supersedes is not None):
                raise ValueError('Fact correction text and supersession metadata differ')
        else:
            fact, correction = None, record.supersedes is not None
        self._append(record.record_id, record.text, fact=fact, correction=correction, record=record)

    def rebuild(self, records=None, *, semantics, reference=None, source_scope=None):
        """Atomically replay exact ordered history OR rebuild an eligible current view.

        History must match the recorded event order and bytes. A current view is
        fetched from the authoritative ledger; corrections/revocations can change
        its membership and ordering, so it need not equal incremental history.
        Both operations invalidate the old publication binding.
        """
        self._mutable()
        if semantics == 'history':
            if reference is not None or source_scope is not None or records is None:
                raise ValueError('History replay requires only the original ordered source records')
            records = tuple(self._record(record) for record in records)
            if [(record.record_id, record.source_hash) for record in records] != self.lineage:
                raise ValueError('History replay differs in source order, IDs or hashes')
            for event, record in zip(self.events, records):
                if event['correction'] != (record.supersedes is not None):
                    raise ValueError('History replay changed correction semantics')
                if event['record_hash'] is not None and (
                        event['record_hash'] != record.record_hash or event['supersedes'] != record.supersedes):
                    raise ValueError('History replay changed record or correction provenance')
        elif semantics == 'current_view':
            from .exact_memory import ExactMemoryReference
            if records is not None or not isinstance(reference, ExactMemoryReference):
                raise ValueError('Current-view rebuild requires an authoritative reference, not supplied records')
            self._source_owner(source_scope)
            records = reference.view(source_scope).records
        else:
            raise ValueError('Rebuild semantics must be history or current_view')
        fresh = PersistentFastMemory(self.model, self.tokenizer, self.scope, training=self.training,
                                     representation=self.representation, config=self.config)
        for record in records:
            fresh.append_record(record)
        fresh.read()  # Validate before replacing the last valid state.
        self.state, self.lineage, self.events = fresh.state, fresh.lineage, fresh.events
        self.semantics = semantics
        self._clear_binding()
        return self.read()

    @property
    def cache_key(self):
        return fingerprint({'state': self.read().cache_key, 'scope': self.scope.to_dict(),
                            'lineage': self.lineage, 'events': self.events,
                            'source_binding': self.source_binding})

    def read(self):
        self._check_producer()
        self._check_source()
        weights = self.state.weights if self.training else self.state.weights.detach()
        digest = ('functional-training-state' if self.training else
                  hashlib.sha256(weights.float().cpu().numpy().tobytes()).hexdigest())
        return FastMemoryRead(weights, self.model_producer, digest, encoding_identity=self.producer)

    def save(self, path, *, require_source_binding=False):
        self._mutable()
        if self.training:
            raise ValueError('Training graphs cannot be persisted')
        if require_source_binding and self.source_binding is None:
            raise ValueError('Source-bound image required')
        read = self.read()
        final = Path(path)
        if final.exists():
            raise FileExistsError('Fast image output already exists')
        temporary = final.with_name(final.name + '.tmp-' + uuid.uuid4().hex)
        try:
            model_version = self.model.fast_version_key()
            producer = encoding_producer(model_version, self.representation, self.dependencies, self.config)
            metadata = save_fast_depth_image(
                temporary, read, config=self.config, scope=self.scope, producer=producer,
                chunk_index=self.state.chunk_index, update_count=self.state.chunk_index,
                cursor=len(self.lineage), lineage=tuple(self.lineage))
            binding_path = temporary / 'source_binding.json'
            if self.source_binding is not None:
                binding = dict(self.source_binding, format='replica.fast_memory.source_binding.v1',
                               lineage_digest=metadata['lineage_digest'])
                binding_path.write_text(json.dumps(binding, indent=2) + '\n')
                metadata = dict(metadata, source_binding=binding)
            persistent = {'format': PERSISTENT_FORMAT, 'representation': self.representation,
                          'representation_config': self.representation_config,
                          'dependencies': self.dependencies, 'engine_config': self.config.to_dict(),
                          'model_version': model_version, 'producer': producer, 'events': self.events,
                          'semantics': self.semantics, 'migration': self.migration,
                          'depth_manifest_sha256': _sha(temporary / 'manifest.json'),
                          'source_binding_sha256': _sha(binding_path) if binding_path.exists() else None}
            persistent['manifest_digest'] = fingerprint(persistent)
            (temporary / 'persistent_manifest.json').write_text(json.dumps(persistent, indent=2) + '\n')
            temporary.rename(final)
            return dict(metadata, persistent=persistent)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    @classmethod
    def load(cls, path, model, tokenizer, scope, *, representation=RAW_REPRESENTATION, config=None,
             expected_view_digest=None, require_source_binding=False, source_reference=None,
             source_scope=None, legacy_representation=None):
        path = Path(path)
        modern = (path / 'persistent_manifest.json').is_file()
        if not modern and legacy_representation is None:
            raise ValueError('Legacy snapshot requires an explicit legacy representation and read-only interpretation')
        if modern and legacy_representation is not None:
            raise ValueError('Legacy interpretation cannot relabel a versioned snapshot')
        representation = representation if modern else legacy_representation
        result = cls(model, tokenizer, scope, representation=representation, config=config)
        model_version = model.fast_version_key()
        persistent = read_persistent_manifest(path) if modern else None
        producer = encoding_producer(model_version, representation, result.dependencies, result.config)
        if modern:
            if (persistent['representation'] != representation or persistent['dependencies'] != result.dependencies
                    or persistent['engine_config'] != result.config.to_dict()
                    or persistent['model_version'] != model_version or persistent['producer'] != producer):
                raise ValueError('Persistent producer, representation or dependency metadata differs')
        read, metadata = load_fast_depth_image(
            path, config=result.config, scope=scope, producer=producer if modern else model_version,
            frame_producer=result.model_producer, device=result.device)
        if (metadata['depth_mode'] != 'shared'
                or not metadata['cursor'] == metadata['chunk_index'] == metadata['update_count'] == len(metadata['lineage'])
                or len({entry[0] for entry in metadata['lineage']}) != len(metadata['lineage'])):
            raise ValueError('Persistent image state/lineage mismatch')
        result.state = FastState(read.weights, metadata['chunk_index'])
        result.lineage = [tuple(item) for item in metadata['lineage']]
        if modern:
            events = persistent['events']
            if not isinstance(events, list) or len(events) != len(result.lineage):
                raise ValueError('Persistent event inventory differs')
            for event, (event_id, source_hash) in zip(events, result.lineage):
                if (not isinstance(event, dict) or set(event) != {
                        'event_id', 'source_hash', 'correction', 'record_hash', 'supersedes'}
                        or (event['event_id'], event['source_hash']) != (event_id, source_hash)
                        or type(event['correction']) is not bool
                        or (event['record_hash'] is not None and not _is_sha256(event['record_hash']))
                        or (event['supersedes'] is not None and (
                            not isinstance(event['supersedes'], str) or not event['supersedes']))
                        or (not event['correction'] and event['supersedes'] is not None)
                        or (event['record_hash'] is not None
                            and event['correction'] != (event['supersedes'] is not None))):
                    raise ValueError('Persistent event lineage or correction metadata differs')
            result.events = events
            result.semantics, result.migration = persistent['semantics'], persistent['migration']
        else:
            result._legacy = True
            result.producer = result._producer()
            result.events = [dict(event_id=event_id, source_hash=source_hash, correction=False,
                                  record_hash=None, supersedes=None) for event_id, source_hash in result.lineage]
            result.migration = {'legacy_depth_manifest_sha256': _sha(path / 'manifest.json'),
                                'legacy_state_digest': read.state_digest,
                                'declared_legacy_representation': representation}
        binding_path = path / 'source_binding.json'
        binding = json.loads(binding_path.read_text()) if binding_path.exists() else None
        if binding is not None and (set(binding) != {'format', 'ledger_id', 'epoch', 'view_digest', 'lineage_digest'}
                or binding['format'] != 'replica.fast_memory.source_binding.v1'
                or binding['lineage_digest'] != metadata['lineage_digest']
                or type(binding['epoch']) is not int or binding['epoch'] < 0
                or not isinstance(binding['ledger_id'], str) or not binding['ledger_id']
                or not _is_sha256(binding['view_digest'])):
            raise ValueError('Fast source-binding metadata differs')
        if require_source_binding and binding is None:
            raise ValueError('Fast source-binding metadata is missing')
        if expected_view_digest is not None and (binding is None or binding['view_digest'] != expected_view_digest):
            raise ValueError('Fast source view is stale')
        if binding is not None and source_reference is not None:
            from .exact_memory import ExactMemoryReference
            if not isinstance(source_reference, ExactMemoryReference):
                raise TypeError('Source-bound load requires the authoritative exact reference')
            result._source_owner(source_scope)
            view = source_reference.view(source_scope)
            if (binding['ledger_id'], binding['epoch'], binding['view_digest']) != (
                    view.ledger_id, view.epoch, view.digest):
                raise ValueError('Fast source view is stale or belongs to another ledger')
            result._authorize_source_view(source_reference, view)
            result.source_binding = binding
            result._source_reference, result._source_scope = source_reference, source_scope
        elif require_source_binding or expected_view_digest is not None:
            raise ValueError('Source-bound load requires an authoritative reference and caller scope')
        # Without a live authority the tensor can be inspected, but its old
        # source binding cannot authorize a new source-bound publication.
        return result

    def migrate_legacy(self, records=None, *, semantics, reference=None, source_scope=None):
        if not self._legacy:
            raise ValueError('Only an explicitly loaded legacy snapshot needs migration')
        self._check_producer()
        fresh = PersistentFastMemory(self.model, self.tokenizer, self.scope, training=self.training,
                                     representation=self.representation, config=self.config)
        fresh.lineage, fresh.events = list(self.lineage), list(self.events)
        if semantics == 'history' and records is not None:
            records = tuple(records)
            fresh.events = [dict(event, correction=record.supersedes is not None)
                            for event, record in zip(fresh.events, records)]
        fresh.rebuild(records, semantics=semantics, reference=reference, source_scope=source_scope)
        fresh.migration = dict(self.migration, operation='explicit_observation_reencode', semantics=semantics)
        return fresh
