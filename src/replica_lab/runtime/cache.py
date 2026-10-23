"""C17 bounded, ownership-keyed cache manager (host payloads only).

This module manages immutable host payloads such as validated record-latent
references or context packs. It does **not** create or reuse native KV/recurrent
caches; the reference adapter still re-prefills. Exact key equality is required:
a shape or producer match is insufficient. Source invalidation removes matching
entries; bounds evict oldest entries deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Callable


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be nonempty text')
    return value


@dataclass(frozen=True)
class CacheKey:
    producer: str
    writer: str
    tokenizer: str
    record_id: str
    record_hash: str
    span_hash: str
    subject_id: str
    branch_id: str
    purpose: str
    cutoff: int
    revision: int
    identity_digest: str | None = None

    def __post_init__(self):
        for name in ('producer', 'writer', 'tokenizer', 'record_id', 'record_hash',
                     'span_hash', 'subject_id', 'branch_id', 'purpose'):
            _text(getattr(self, name), name)
        for name in ('record_hash', 'span_hash'):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'{name} must be a lowercase SHA256')
        for name in ('cutoff', 'revision'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be nonnegative')
        if self.identity_digest is not None and (
                not isinstance(self.identity_digest, str) or len(self.identity_digest) != 64):
            raise ValueError('identity_digest must be a SHA256 or None')

    @property
    def digest(self):
        return _digest({'producer': self.producer, 'writer': self.writer,
                        'tokenizer': self.tokenizer, 'record_id': self.record_id,
                        'record_hash': self.record_hash, 'span_hash': self.span_hash,
                        'subject_id': self.subject_id, 'branch_id': self.branch_id,
                        'purpose': self.purpose, 'cutoff': self.cutoff,
                        'revision': self.revision,
                        'identity_digest': self.identity_digest})


@dataclass
class CacheEntry:
    key: CacheKey
    payload: Any
    size_bytes: int
    serial: int

    def __post_init__(self):
        if not isinstance(self.key, CacheKey):
            raise ValueError('cache entry requires CacheKey')
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError('entry size must be nonnegative')


class OwnedCache:
    """Exact-key, bounded, in-process reference cache. No native tensors."""
    def __init__(self, *, max_entries=64, max_bytes=8 * 1024 * 1024):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError('max_entries must be positive')
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('max_bytes must be positive')
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._entries = {}
        self._serial = 0
        self._bytes = 0
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0, 'invalidations': 0}

    def __len__(self):
        return len(self._entries)

    def get(self, key: CacheKey):
        if not isinstance(key, CacheKey):
            raise ValueError('get requires CacheKey')
        entry = self._entries.get(key.digest)
        if entry is None or entry.key != key:
            self.stats['misses'] += 1
            return None
        self.stats['hits'] += 1
        return entry.payload

    def put(self, key: CacheKey, payload: Any, *, size_bytes: int):
        if not isinstance(key, CacheKey):
            raise ValueError('put requires CacheKey')
        if type(size_bytes) is not int or size_bytes < 0:
            raise ValueError('size_bytes must be nonnegative')
        existing = self._entries.get(key.digest)
        if existing is not None and existing.key != key:
            raise ValueError('cache digest collision with different key')
        if existing is not None:
            self._bytes -= existing.size_bytes
        self._serial += 1
        entry = CacheEntry(key, payload, size_bytes, self._serial)
        self._entries[key.digest] = entry
        self._bytes += size_bytes
        while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
            oldest = min(self._entries.values(), key=lambda item: item.serial)
            del self._entries[oldest.key.digest]
            self._bytes -= oldest.size_bytes
            self.stats['evictions'] += 1

    def invalidate(self, predicate: Callable[[CacheKey], bool]):
        if not callable(predicate):
            raise ValueError('invalidate requires a predicate')
        removed = 0
        for digest, entry in list(self._entries.items()):
            if predicate(entry.key):
                del self._entries[digest]
                self._bytes -= entry.size_bytes
                removed += 1
        self.stats['invalidations'] += removed
        return removed

    def invalidate_source(self, record_hash: str):
        if not isinstance(record_hash, str) or len(record_hash) != 64:
            raise ValueError('record_hash must be SHA256')
        return self.invalidate(lambda key: key.record_hash == record_hash)

    def clear(self):
        self._entries.clear()
        self._bytes = 0
