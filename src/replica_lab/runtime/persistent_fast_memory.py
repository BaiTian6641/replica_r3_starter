"""Incremental shared SGDN-2 state using the production writer and image format."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import torch

from ..architecture.contracts import FastMemoryRead
from ..neural import FastState, SGDN2Memory
from .fast_memory import FastMemoryConfig, save_fast_depth_image, load_fast_depth_image


class PersistentFastMemory:
    def __init__(self, model, tokenizer, scope, *, training=False, source_binding=None):
        self.model, self.tokenizer, self.scope, self.training = model, tokenizer, scope, training
        spec = model.manifest.fast_memory
        if spec.depth_mode != 'shared' or not spec.depth_writers:
            raise ValueError('Persistent diagnostic requires shared per-depth SGDN-2')
        self.config = FastMemoryConfig(heads=spec.heads, key_size=spec.key_size, value_size=spec.value_size,
                                       update_rule='sgdn2', max_tokens_per_chunk=256, max_image_bytes=1048576)
        self.primitive = SGDN2Memory(spec.heads, spec.key_size, spec.value_size)
        self.device = model.base.get_input_embeddings().weight.device
        self.state = self.primitive.initial_state(1, device=self.device)
        self.producer = model.fast_producer_key()
        self.lineage = []
        self.source_binding = source_binding

    def _check_producer(self):
        if self.model.fast_producer_key() != self.producer:
            raise ValueError('State producer changed; reload or re-encode explicitly')

    def bind_source(self, view):
        """Bind the next image to an authenticated exact-ledger current view."""
        for name in ('ledger_id', 'epoch', 'digest'):
            if not hasattr(view, name):
                raise TypeError('Source view must expose ledger_id, epoch and digest')
        if type(view.epoch) is not int or view.epoch < 0:
            raise ValueError('Source-view epoch must be nonnegative')
        self.source_binding = {'ledger_id': view.ledger_id, 'epoch': view.epoch,
                               'view_digest': view.digest}
        return dict(self.source_binding)

    def append(self, event_id, text):
        self._check_producer()
        if self.state.chunk_index >= self.config.max_chunks:
            raise ValueError('Persistent chunk quota exceeded')
        if not isinstance(event_id, str) or not event_id or event_id in {x[0] for x in self.lineage}:
            raise ValueError('Observation event IDs must be fresh')
        ids = self.tokenizer(text, add_special_tokens=False)['input_ids']
        if not ids or len(ids) > self.config.max_tokens_per_chunk:
            raise ValueError('Observation must fit without silent truncation')
        with torch.set_grad_enabled(self.training):
            proposals = {layer: (replace(self.model.encode_fast_observation(text, self.tokenizer, layer=layer),
                                         chunk_index=self.state.chunk_index),)
                         for layer in self.model.ports.injection_layers}
            self.state = self.primitive.reduce_shared_chunk(self.state, proposals)
        self.lineage.append((event_id, hashlib.sha256(text.encode()).hexdigest()))
        # Any successful append invalidates the previous current-view binding.
        self.source_binding = None

    def rebuild(self, records):
        """Replace derived state only from an explicitly supplied current view.

        This is a permitted-source rebuild, not an in-place unlearning claim.
        Construction is atomic with respect to this wrapper's state/lineage.
        """
        fresh = PersistentFastMemory(self.model, self.tokenizer, self.scope,
                                     training=self.training, source_binding=self.source_binding)
        for record in records:
            fresh.append(record.record_id, record.text)
        self.state, self.lineage = fresh.state, fresh.lineage
        # A rebuilt tensor set needs a newly authenticated current-view binding.
        self.source_binding = None
        return self.read()

    def read(self):
        self._check_producer()
        weights = self.state.weights if self.training else self.state.weights.detach()
        digest = ('functional-training-state' if self.training else
                  hashlib.sha256(weights.float().cpu().numpy().tobytes()).hexdigest())
        return FastMemoryRead(weights, self.producer, digest)

    def save(self, path, *, require_source_binding=False):
        if self.training:
            raise ValueError('Training graphs cannot be persisted')
        if require_source_binding and self.source_binding is None:
            raise ValueError('Source-bound image required')
        final = Path(path)
        if final.exists():
            raise FileExistsError('Fast image output already exists')
        temporary = final.with_name(final.name + '.tmp-' + uuid.uuid4().hex)
        try:
            metadata = save_fast_depth_image(
                temporary, self.read(), config=self.config, scope=self.scope,
                producer=self.model.fast_version_key(), chunk_index=self.state.chunk_index,
                update_count=self.state.chunk_index, cursor=len(self.lineage),
                lineage=tuple(self.lineage))
            if self.source_binding is not None:
                binding = dict(self.source_binding)
                binding.update(format='replica.fast_memory.source_binding.v1',
                               lineage_digest=metadata['lineage_digest'])
                (temporary / 'source_binding.json').write_text(json.dumps(binding, indent=2) + '\n')
                metadata = dict(metadata, source_binding=binding)
            temporary.rename(final)
            return metadata
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    @classmethod
    def load(cls, path, model, tokenizer, scope, *, expected_view_digest=None,
             require_source_binding=False):
        result = cls(model, tokenizer, scope)
        read, metadata = load_fast_depth_image(path, config=result.config, scope=scope,
                                               producer=model.fast_version_key(), frame_producer=result.producer,
                                               device=result.device)
        binding_path = Path(path) / 'source_binding.json'
        binding = None
        if binding_path.exists():
            binding = json.loads(binding_path.read_text())
            if (set(binding) != {'format', 'ledger_id', 'epoch', 'view_digest', 'lineage_digest'}
                    or binding['format'] != 'replica.fast_memory.source_binding.v1'
                    or binding['lineage_digest'] != metadata['lineage_digest']
                    or type(binding['epoch']) is not int or binding['epoch'] < 0
                    or not isinstance(binding['ledger_id'], str) or not binding['ledger_id']
                    or not isinstance(binding['view_digest'], str) or len(binding['view_digest']) != 64):
                raise ValueError('Fast source-binding metadata differs')
        elif require_source_binding or expected_view_digest is not None:
            raise ValueError('Fast source-binding metadata is missing')
        if expected_view_digest is not None and (
                binding is None or binding['view_digest'] != expected_view_digest):
            raise ValueError('Fast source view is stale')
        if (metadata['depth_mode'] != 'shared'
                or not metadata['cursor'] == metadata['chunk_index'] == metadata['update_count'] == len(metadata['lineage'])
                or len({entry[0] for entry in metadata['lineage']}) != len(metadata['lineage'])):
            raise ValueError('Persistent image state/lineage mismatch')
        result.state = FastState(read.weights, metadata['chunk_index'])
        result.lineage = [tuple(x) for x in metadata['lineage']]
        result.source_binding = binding
        return result
