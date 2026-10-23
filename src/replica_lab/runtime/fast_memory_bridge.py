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
                path.mkdir(parents=True)
                self.session.save(path)
        return receipt

    def snapshot(self):
        return self.session.state.detached()

    def invalidate(self, reason: str):
        self.session.invalidate(reason)
        self.last_commit = None
        return self.session.state
