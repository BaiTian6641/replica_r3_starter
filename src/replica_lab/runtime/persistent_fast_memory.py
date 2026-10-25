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


PAGE_DEPENDENCY_FORMAT = 'replica.fast_memory.page_dependencies.v1'
_EVENT_FIELDS = {'event_id', 'source_hash', 'correction', 'record_hash', 'supersedes'}
_PAGE_DEPENDENCY_FIELDS = {'record_id', 'record_hash', 'source_hash', 'supersedes'}


def _validate_page_dependencies(value):
    """Authenticate an explicit ordered page -> record dependency list."""
    if not isinstance(value, list):
        raise ValueError('Persistent page dependency metadata differs')
    dependencies, seen = [], set()
    for dependency in value:
        if not isinstance(dependency, dict) or set(dependency) != _PAGE_DEPENDENCY_FIELDS:
            raise ValueError('Persistent page dependency metadata differs')
        record_id = dependency['record_id']
        if (not isinstance(record_id, str) or not record_id or record_id in seen
                or not _is_sha256(dependency['record_hash'])
                or not _is_sha256(dependency['source_hash'])
                or (dependency['supersedes'] is not None and (
                    not isinstance(dependency['supersedes'], str) or not dependency['supersedes']))):
            raise ValueError('Persistent page dependency metadata differs')
        seen.add(record_id)
        dependencies.append({key: dependency[key] for key in
                             ('record_id', 'record_hash', 'source_hash', 'supersedes')})
    return dependencies


def read_persistent_manifest(path):
    """Authenticate the wrapper manifest; callers must also verify producer pins."""
    value = json.loads((Path(path) / 'persistent_manifest.json').read_text())
    fields = {'format', 'representation', 'representation_config', 'dependencies', 'engine_config',
              'model_version', 'producer', 'events', 'semantics', 'migration',
              'depth_manifest_sha256', 'source_binding_sha256', 'manifest_digest'}
    if (set(value) not in (fields, fields | {'page_dependency_format'})
            or value['format'] != PERSISTENT_FORMAT
            or ('page_dependency_format' in value
                and value['page_dependency_format'] != PAGE_DEPENDENCY_FORMAT)
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


    def _authorize_page_dependency(self, dependency, *, index, dep_index, eligible,
                                   superseding, positions, ledger):
        """Prove one explicitly declared page dependency is still authorized."""
        binding = (dependency['source_hash'], dependency['record_hash'],
                   dependency['supersedes'])
        record_id = dependency['record_id']
        record = eligible.get(record_id)
        if record is not None:
            if binding != (record.source_hash, record.record_hash, record.supersedes):
                raise ValueError('Consumed page dependency no longer matches the authoritative view')
            return
        status = ledger.record_status(record_id)
        if status is None:
            raise ValueError('Consumed page dependency is missing from the ledger; '
                             'rebuild before rebinding')
        if status['revoked']:
            raise ValueError('Consumed page dependency was revoked; rebuild before rebinding')
        if binding != (status['source_hash'], status['record_hash'], status['supersedes']):
            raise ValueError('Consumed page dependency changed in the ledger; '
                             'rebuild before rebinding')
        correction = superseding.get(record_id)
        if (correction is None
                or positions.get(correction.record_id, (-1, -1)) <= (index, dep_index)):
            raise ValueError('Consumed page dependency is absent from the authoritative view; '
                             'rebuild before rebinding')

    def _authorize_source_view(self, reference, view):
        """Prove the consumed lineage is still authorized by this exact view.

        Record-backed events must match an eligible record byte-for-byte, or be
        superseded originals whose eligible correction this state consumed later.
        Raw events authorize their exact consumed page text and every explicitly
        declared page -> record dependency. Dependencies are never guessed from
        arbitrary page text: an event whose provenance is unknown (legacy
        snapshots) fails closed here. Dependency revocation, absence or change
        therefore invalidates the page even when the page hash is unchanged. The
        state must also already reflect every eligible correction in the view.
        A metadata-only rebind can never re-authorize stale contents: only
        consuming the authoritative change (append) or an explicit verified
        rebuild can.
        """
        ledger = getattr(reference, 'ledger', None)
        if ledger is None:
            raise TypeError('Source binding requires the authoritative exact reference')
        eligible = {record.record_id: record for record in view.records}
        superseding = {record.supersedes: record for record in view.records
                       if record.supersedes is not None}
        positions = {}
        for index, ((event_id, _), event) in enumerate(zip(self.lineage, self.events)):
            if event['record_hash'] is not None:
                positions[event_id] = max(positions.get(event_id, (-1, -1)), (index, -1))
            for dep_index, dependency in enumerate(event['dependencies'] or ()):
                record_id = dependency['record_id']
                positions[record_id] = max(positions.get(record_id, (-1, -1)),
                                           (index, dep_index))
        revoked = set(ledger.revoked_source_hashes())
        consumed_ids = set()
        consumed_hashes = {source_hash for _, source_hash in self.lineage}
        for index, ((event_id, source_hash), event) in enumerate(zip(self.lineage, self.events)):
            record = eligible.get(event_id)
            if event['record_hash'] is not None:
                consumed_ids.add(event_id)
                if record is not None:
                    if (source_hash != record.source_hash
                            or event['record_hash'] != record.record_hash
                            or event['supersedes'] != record.supersedes
                            or event['correction'] != (record.supersedes is not None)):
                        raise ValueError('Consumed record no longer matches the authoritative view')
                    continue
                correction = superseding.get(event_id)
                if (correction is None
                        or positions.get(correction.record_id, (-1, -1)) <= positions[event_id]):
                    raise ValueError('Consumed record is absent from the authoritative view; '
                                     'rebuild before rebinding')
                status = ledger.record_status(event_id)
                if (status is None or status['revoked']
                        or (status['source_hash'], status['record_hash'], status['supersedes'])
                        != (source_hash, event['record_hash'], event['supersedes'])):
                    raise ValueError('Consumed superseded record no longer matches the ledger')
            else:
                dependencies = event['dependencies']
                if dependencies is None:
                    raise ValueError('Consumed page dependency provenance is absent; '
                                     'rebuild before rebinding')
                if source_hash in revoked:
                    raise ValueError('Consumed observation repeats revoked source text; '
                                     'rebuild before rebinding')
                for dep_index, dependency in enumerate(dependencies):
                    self._authorize_page_dependency(
                        dependency, index=index, dep_index=dep_index, eligible=eligible,
                        superseding=superseding, positions=positions, ledger=ledger)
                    consumed_ids.add(dependency['record_id'])
        for correction in superseding.values():
            if (correction.record_id not in consumed_ids
                    and correction.source_hash not in consumed_hashes):
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

    def _page_dependencies(self, text, records):
        """Derive authenticated dependencies from explicit source records only.

        Omitted records mean provenance is unknown, not empty: such an event can
        be inspected but never source-authorized or persisted. Callers must pass
        an explicit tuple (empty only for a deliberately opaque observation).
        """
        if records is None:
            return None
        if isinstance(records, (str, bytes)):
            raise TypeError('Raw page dependencies require MemoryRecord values')
        lines = text.splitlines()
        dependencies, seen = [], set()
        for record in records:
            record = self._record(record)
            if record.record_id in seen:
                raise ValueError('Raw page dependency IDs must be unique')
            if record.text not in lines:
                raise ValueError('Declared page dependency is not present exactly '
                                 'in the observed page')
            seen.add(record.record_id)
            dependencies.append({'record_id': record.record_id,
                                 'record_hash': record.record_hash,
                                 'source_hash': record.source_hash,
                                 'supersedes': record.supersedes})
        return dependencies

    def _append(self, event_id, text, *, fact=None, correction=False, record=None, records=None):
        self._mutable()
        if self.state.chunk_index >= self.config.max_chunks:
            raise ValueError('Persistent chunk quota exceeded')
        if not isinstance(event_id, str) or not event_id or event_id in {item[0] for item in self.lineage}:
            raise ValueError('Observation event IDs must be fresh')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Observed source text must be nonempty')
        if record is not None and records is not None:
            raise ValueError('A record-backed event cannot also declare page dependencies')
        dependencies = ([] if record is not None or fact is not None
                        else self._page_dependencies(text, records))
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
                 'supersedes': record.supersedes if record is not None else None,
                 'dependencies': dependencies}
        self.state = state
        self.lineage.append((event_id, source_hash))
        self.events.append(event)
        self._clear_binding()

    def append(self, event_id, text, *, records=None):
        """Consume one raw observation page.

        `records` explicitly declares the ordered ledger records contained in
        this page. Their identity and hashes are persisted with the event so a
        later rebind can reject revoked, missing or changed dependencies instead
        of trusting the page text hash. Omitting `records` leaves provenance
        unknown: the event can be inspected but not source-authorized or saved.
        Pass `records=()` only for a deliberately opaque observation with no
        ledger dependencies. Text is never parsed to guess provenance.
        """
        if self.representation != RAW_REPRESENTATION:
            raise ValueError('Raw append cannot mix with fact-binding representation')
        self._append(event_id, text, records=records)

    def append_page(self, event_id, records):
        """Consume a raw page derived exactly from these ordered source records."""
        if self.representation != RAW_REPRESENTATION:
            raise ValueError('Raw append cannot mix with fact-binding representation')
        if isinstance(records, (str, bytes)):
            raise TypeError('Raw page dependencies require MemoryRecord values')
        records = tuple(self._record(record) for record in records)
        if not records:
            raise ValueError('Raw page dependencies must be explicit and nonempty')
        text = '\n'.join(record.text for record in records)
        self._append(event_id, text, records=records)

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
        if any(not isinstance(event.get('dependencies'), list) for event in self.events):
            raise ValueError('Page dependency provenance is unknown; rebuild before saving')
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
                          'model_version': model_version, 'producer': producer,
                          'page_dependency_format': PAGE_DEPENDENCY_FORMAT,
                          'events': self.events,
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
            provenance = 'page_dependency_format' in persistent
            if not isinstance(events, list) or len(events) != len(result.lineage):
                raise ValueError('Persistent event inventory differs')
            normalized = []
            for event, (event_id, source_hash) in zip(events, result.lineage):
                if (not isinstance(event, dict)
                        or set(event) not in (_EVENT_FIELDS,
                                              _EVENT_FIELDS | {'dependencies'})):
                    raise ValueError('Persistent event lineage or correction metadata differs')
                if ((event['event_id'], event['source_hash']) != (event_id, source_hash)
                        or type(event['correction']) is not bool
                        or (event['record_hash'] is not None and not _is_sha256(event['record_hash']))
                        or (event['supersedes'] is not None and (
                            not isinstance(event['supersedes'], str) or not event['supersedes']))
                        or (not event['correction'] and event['supersedes'] is not None)
                        or (event['record_hash'] is not None
                            and event['correction'] != (event['supersedes'] is not None))):
                    raise ValueError('Persistent event lineage or correction metadata differs')
                if provenance != ('dependencies' in event):
                    raise ValueError('Persistent page dependency metadata is incomplete')
                # Older versioned snapshots omit dependencies entirely. They
                # remain inspectable; record-backed events keep full provenance
                # through their record hash, while raw events have unknown
                # provenance and fail closed on any source authorization.
                if provenance:
                    if event['dependencies'] is None:
                        if event['record_hash'] is not None:
                            raise ValueError('Persistent event lineage or correction metadata differs')
                        dependencies = None
                    else:
                        dependencies = _validate_page_dependencies(event['dependencies'])
                        if event['record_hash'] is not None and dependencies:
                            raise ValueError('Persistent event lineage or correction metadata differs')
                elif event['record_hash'] is not None:
                    dependencies = []
                else:
                    dependencies = None
                normalized.append(dict(event, dependencies=dependencies))
            result.events = normalized
            result.semantics, result.migration = persistent['semantics'], persistent['migration']
        else:
            result._legacy = True
            result.producer = result._producer()
            result.events = [dict(event_id=event_id, source_hash=source_hash, correction=False,
                                  record_hash=None, supersedes=None, dependencies=None)
                             for event_id, source_hash in result.lineage]
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
