"""Reference observation-to-fast-memory bridge (host side only).

The bridge is deliberately simple and untrained: a deterministic token feature
encoder stages one actual observation's keys/values and commits once. It proves
the ownership and scheduling path; it is not a learned semantic encoder and
does not establish delayed-use benefit. Callers pass only actual tool/source
observations, never generated answers or teacher targets.
"""
from __future__ import annotations

import hashlib
import math

import torch

from .fast_memory import FastMemoryCommit, FastMemorySession


class DeterministicTextEncoder:
    """Stable sin/cos token features; no learned parameters or model forward."""
    def __init__(self, tokenizer, *, key_size=64, value_size=64, max_tokens=256):
        if min(key_size, value_size, max_tokens) < 1:
            raise ValueError('positive encoder dimensions required')
        self.tokenizer, self.key_size = tokenizer, key_size
        self.value_size, self.max_tokens = value_size, max_tokens

    def _ids(self, text):
        if not isinstance(text, str) or not text:
            raise ValueError('an observation requires nonempty text')
        ids = self.tokenizer(text, add_special_tokens=False)['input_ids']
        if not ids:
            raise ValueError('an observation produced no tokens')
        return ids[:self.max_tokens]

    def _vectors(self, ids, width, phase):
        rows = []
        for token in ids:
            rows.append([math.sin((token + 1) * (index + 1) * phase)
                         for index in range(width)])
        value = torch.tensor([rows], dtype=torch.float32)
        value = value / value.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return value

    def __call__(self, text):
        ids = self._ids(text)
        keys = self._vectors(ids, self.key_size, 0.017)
        values = self._vectors(ids, self.value_size, 0.013)
        return keys, values


class FastMemoryBridge:
    """Stages/commits actual observations into one owned fast-memory session."""
    def __init__(self, session: FastMemorySession, encoder, *, image_root=None):
        if not isinstance(session, FastMemorySession):
            raise ValueError('bridge requires a FastMemorySession')
        self.session, self.encoder = session, encoder
        self.image_root = image_root
        self.observed = 0
        self.last_commit: FastMemoryCommit | None = None

    @classmethod
    def from_image(cls, image_dir, *, config, scope, producer, encoder,
                   image_root=None):
        """Load one exact owner/scope/producer image into a fresh host bridge.

        The image is a committed fast-state snapshot, not a model checkpoint.
        `observed` is reconstructed from the image update count; no pending
        writes, generated text or cross-branch material are accepted.
        """
        session = FastMemorySession.load(image_dir, config=config, scope=scope,
                                         producer=producer)
        bridge = cls(session, encoder, image_root=image_root)
        bridge.observed = int(session.state.update_count)
        return bridge

    def observe(self, observation_id: str, text: str, *, source_hash: str):
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            raise ValueError('source_hash must be a SHA256')
        keys, values = self.encoder(text)
        changed = self.session.stage_observation(observation_id, keys, values,
                                                 source_hash=source_hash)
        if not changed:
            return None
        receipt = self.session.commit()
        self.last_commit = receipt
        self.observed += 1
        if self.image_root is not None:
            path = self.image_root / f'image-{self.session.state.generation}-{receipt.chunk_index}'
            if not path.exists():
                # FastMemorySession.save creates the directory itself and
                # validates the image quota; do not pre-create it here.
                self.session.save(path)
        return receipt

    def replay(self, observations, *, reason='permitted_source_replay',
               invalidate=True):
        """Reset and replay a bounded sequence of permitted observations once.

        Each item is ``(observation_id, text, source_hash)``. The host supplies
        only still-permitted observed source/tool text, never generated output
        or expected answers. Replay commits one observation at a time; old
        images remain historical and are not treated as current.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('replay reason required')
        if not hasattr(observations, '__iter__'):
            raise ValueError('replay observations must be iterable')
        items = list(observations)
        if len(items) > 512:
            raise ValueError('replay observation bound exceeded')
        if invalidate:
            self.invalidate(reason)
        receipts = []
        for item in items:
            if not isinstance(item, (tuple, list)) or len(item) != 3:
                raise ValueError('each replay item must be (id, text, source_hash)')
            observation_id, text, source_hash = item
            receipt = self.observe(observation_id, text, source_hash=source_hash)
            if receipt is not None:
                receipts.append(receipt)
        return tuple(receipts)

    def snapshot(self):
        return self.session.state.detached()

    def invalidate(self, reason: str):
        self.session.invalidate(reason)
        self.last_commit = None
        return self.session.state
