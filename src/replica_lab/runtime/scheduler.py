"""C17 request admission and ownership boundary (single native job).

The scheduler serializes admitted operations and gives each one an immutable
request lease. It is deliberately not a native KV/recurrent cache: the reference
adapter still re-prefills and cached generation remains unsupported. The lease
makes ownership and invalidation explicit so a future scheduler cannot mix task,
branch, identity or evidence epochs.

Storage and tool effects stay in the caller's operation; the scheduler neither
runs tools nor mutates model parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import threading
from contextlib import contextmanager


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or any(c.isspace() for c in value):
        raise ValueError(f'{name} must be a nonempty identifier without whitespace')
    return value


@dataclass(frozen=True)
class RequestOwnership:
    """Identity that must remain pinned for one admitted operation."""
    request_id: str
    task_id: str
    task_digest: str
    revision: int
    evidence_epoch: int
    branch_id: str
    producer: str
    identity_digest: str | None = None

    def __post_init__(self):
        for name in ('request_id', 'task_id', 'task_digest', 'branch_id', 'producer'):
            _text(getattr(self, name), name)
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError('revision must be nonnegative')
        if type(self.evidence_epoch) is not int or self.evidence_epoch < 0:
            raise ValueError('evidence_epoch must be nonnegative')
        if self.identity_digest is not None and (
                not isinstance(self.identity_digest, str) or len(self.identity_digest) != 64):
            raise ValueError('identity_digest must be a SHA256 or None')

    @property
    def key(self):
        return _digest({'request_id': self.request_id, 'task_id': self.task_id,
                        'task_digest': self.task_digest, 'revision': self.revision,
                        'evidence_epoch': self.evidence_epoch,
                        'branch_id': self.branch_id, 'producer': self.producer,
                        'identity_digest': self.identity_digest})


class SchedulerBusy(RuntimeError):
    """A second native operation was attempted while one lease was active."""


@dataclass
class RequestLease:
    ownership: RequestOwnership
    _state: str = 'active'
    _reason: str | None = None

    def assert_current(self):
        if self._state != 'active':
            raise RuntimeError(f'lease is {self._state}: {self._reason or "released"}')

    def invalidate(self, reason='host_invalidation'):
        if self._state == 'active':
            self._state, self._reason = 'invalidated', reason
        return self

    def release(self):
        if self._state == 'active':
            self._state, self._reason = 'released', None
        return self


class RequestScheduler:
    """Single-operation admission; no native cache is created or reused."""
    def __init__(self):
        self._admission = threading.Lock()
        self._state_lock = threading.Lock()
        self._active: RequestLease | None = None
        self._generation = 0
        self._events = []
        # Request IDs are revoked for the scheduler's lifetime. This makes a
        # host invalidation effective even when it arrives before admission.
        self._invalidated_requests: dict[str, str] = {}

    def active(self):
        with self._state_lock:
            return None if self._active is None else self._active.ownership

    def events(self):
        with self._state_lock:
            return tuple(dict(event) for event in self._events)

    def invalidate(self, request_id, reason='host_invalidation'):
        with self._state_lock:
            self._invalidated_requests[request_id] = reason
            if self._active is not None and self._active.ownership.request_id == request_id:
                self._active.invalidate(reason)

    @contextmanager
    def admit(self, ownership: RequestOwnership):
        if not isinstance(ownership, RequestOwnership):
            raise ValueError('admit requires RequestOwnership')
        if not self._admission.acquire(blocking=False):
            raise SchedulerBusy('another native request is active')
        lease = None
        try:
            with self._state_lock:
                revoked = self._invalidated_requests.get(ownership.request_id)
                if revoked is not None:
                    self._events.append({'event': 'refused', 'request_id': ownership.request_id,
                                         'task_digest': ownership.task_digest,
                                         'revision': ownership.revision,
                                         'evidence_epoch': ownership.evidence_epoch,
                                         'generation': self._generation,
                                         'reason': revoked})
                    raise RuntimeError(
                        f'invalidated_before_admission: {ownership.request_id}: {revoked}')
                self._generation += 1
                lease = RequestLease(ownership)
                self._active = lease
                self._events.append({'event': 'admitted', 'request_id': ownership.request_id,
                                     'task_digest': ownership.task_digest,
                                     'revision': ownership.revision,
                                     'evidence_epoch': ownership.evidence_epoch,
                                     'generation': self._generation})
            yield lease
            lease.assert_current()
        finally:
            with self._state_lock:
                if lease is not None:
                    lease.release()
                    self._events.append({'event': 'released', 'request_id': ownership.request_id,
                                         'generation': self._generation,
                                         'state': lease._state})
                self._active = None
            self._admission.release()

    def run(self, ownership, operation, *args, **kwargs):
        with self.admit(ownership) as lease:
            return operation(lease, *args, **kwargs)
