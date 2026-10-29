"""Exact bounded evidence and query-driven Spark spine-plus-memory learning.

``E`` is host-owned, versioned evidence. ``P`` is the retained NativeFusedAdamW
expert bank. Reads are selected by current query/native features, never by gold
record IDs. Clearing E removes source payloads and derived views without
unlearning P or erasing the compact evidence-version lineage that taught it.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, MutableMapping
import copy
import hashlib
import json
from typing import Any, Mapping
import weakref

import torch

from .spine import (BRANCHES, DecisionSnapshot, Packet, SpineCore,
                    SpineError, SpineForwardContext, SpineSpec)
from .spine_checkpoint import load_spine_checkpoint, save_spine_checkpoint
from ..nested_learning import ExpertBank, PlasticLearningController
from ..model_identity import durable_model_identity
from ..optimizers import NativeFusedAdamW

EVIDENCE_CHECKPOINT_FORMAT = "replica.spark.evidence.v2"
LEGACY_EVIDENCE_CHECKPOINT_FORMAT = "replica.spark.evidence.v1"
MEMORY_STATE_FORMAT = "replica.spark.spine.memory.state.v1"
P_QUERY_READ_VERSION = "replica.spark.p_query_read.v1"


def _positive(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise SpineError(f"{name} must be positive")
    return value


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _payload_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(_payload_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_payload_bytes(item) for item in value)
    return 0


def _owned(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return copy.deepcopy(value)


class _BoundedCache(MutableMapping):
    """An ownership-safe LRU with explicit byte/entry/eviction accounting."""

    def __init__(self, *, max_entries: int, max_payload_bytes: int) -> None:
        self.max_entries = _positive(max_entries, "max_entries")
        self.max_payload_bytes = _positive(max_payload_bytes,
                                           "max_payload_bytes")
        self._values: OrderedDict[str, Any] = OrderedDict()
        self._sizes: dict[str, int] = {}
        self._bytes = 0
        self.evictions = 0
        self.invalidations = 0

    def __getitem__(self, key: str) -> Any:
        value = self._values[key]
        self._values.move_to_end(key)
        return _owned(value)

    def __setitem__(self, key: str, value: Any) -> None:
        if not isinstance(key, str) or not key:
            raise SpineError("cache keys must be nonempty strings")
        owned = _owned(value)
        size = _payload_bytes(owned)
        if size > self.max_payload_bytes:
            raise SpineError("cache item exceeds the payload byte bound")
        if key in self._values:
            self.__delitem__(key)
        while self._values and (len(self._values) >= self.max_entries
                                or self._bytes + size > self.max_payload_bytes):
            old_key, _ = self._values.popitem(last=False)
            self._bytes -= self._sizes.pop(old_key)
            self.evictions += 1
        self._values[key] = owned
        self._sizes[key] = size
        self._bytes += size

    def __delitem__(self, key: str) -> None:
        del self._values[key]
        self._bytes -= self._sizes.pop(key)

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def clear(self) -> None:
        if self._values:
            self.invalidations += 1
        self._values.clear()
        self._sizes.clear()
        self._bytes = 0

    @property
    def payload_bytes(self) -> int:
        return self._bytes

    def accounting(self) -> dict[str, int]:
        return {"entries": len(self), "payload_bytes": self.payload_bytes,
                "max_entries": self.max_entries,
                "max_payload_bytes": self.max_payload_bytes,
                "evictions": self.evictions, "invalidations": self.invalidations}


class EvidenceRecord:
    """One exact evidence version with source and validity provenance."""

    def __init__(self, *, record_id: str, scope: str, payload: torch.Tensor,
                 source_epoch: int, available_at: int, valid_from: int,
                 producer_revision: str, representation_version: str,
                 codec_version: str, payload_type: str,
                 source_lineage: Mapping[str, Any], valid_until: int | None = None,
                 supersedes: str | None = None) -> None:
        if not isinstance(record_id, str) or not record_id:
            raise SpineError("evidence record_id is required")
        if not isinstance(scope, str) or not scope:
            raise SpineError("evidence scope is required")
        if not torch.is_tensor(payload) or payload.ndim != 1 \
                or payload.shape[0] != 512 or not torch.is_floating_point(payload):
            raise SpineError("evidence payload must be a width-512 vector")
        if not bool(torch.isfinite(payload).all()):
            raise SpineError("evidence payload is nonfinite")
        for name, value in (("source_epoch", source_epoch),
                            ("available_at", available_at), ("valid_from", valid_from)):
            if type(value) is not int or value < 0:
                raise SpineError(f"{name} must be a nonnegative integer")
        if available_at < valid_from:
            raise SpineError("evidence availability precedes validity start")
        if valid_until is not None and (type(valid_until) is not int or valid_until < 0):
            raise SpineError("valid_until must be a nonnegative integer or null")
        if valid_until is not None and valid_until < available_at:
            raise SpineError("evidence validity ends before availability")
        for name, value in (("producer_revision", producer_revision),
                            ("representation_version", representation_version),
                            ("codec_version", codec_version), ("payload_type", payload_type)):
            if not isinstance(value, str) or not value:
                raise SpineError(f"{name} is required")
        if payload_type not in {"observation", "interpretation", "prediction",
                                "verified_feedback"}:
            raise SpineError("unknown evidence payload type")
        if not isinstance(source_lineage, Mapping) or not source_lineage:
            raise SpineError("evidence source lineage is required")
        if supersedes is not None and (
                not isinstance(supersedes, str) or not supersedes):
            raise SpineError("supersedes must be a nonempty record identity or null")
        self.record_id = record_id
        self.scope = scope
        self.payload = payload.detach().clone()
        self.source_epoch = source_epoch
        self.available_at = available_at
        self.valid_from = valid_from
        self.valid_until = valid_until
        self.producer_revision = producer_revision
        self.representation_version = representation_version
        self.codec_version = codec_version
        self.payload_type = payload_type
        self.source_lineage = copy.deepcopy(dict(source_lineage))
        self.supersedes = supersedes

    @property
    def payload_bytes(self) -> int:
        return int(self.payload.numel() * self.payload.element_size())

    @property
    def metadata_bytes(self) -> int:
        return len(json.dumps(self.identity(), sort_keys=True,
                              default=str).encode("utf-8"))

    @property
    def payload_sha256(self) -> str:
        return _tensor_digest(self.payload)

    @property
    def version(self) -> str:
        body = {"identity": self.identity(), "payload_sha256": self.payload_sha256}
        encoded = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
        return f"{self.record_id}@{hashlib.sha256(encoded).hexdigest()}"

    def identity(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id, "scope": self.scope,
            "source_epoch": self.source_epoch, "available_at": self.available_at,
            "valid_from": self.valid_from, "valid_until": self.valid_until,
            "producer_revision": self.producer_revision,
            "representation_version": self.representation_version,
            "codec_version": self.codec_version, "payload_type": self.payload_type,
            "source_lineage": copy.deepcopy(self.source_lineage),
            "supersedes": self.supersedes,
        }

    def lineage_payload(self) -> dict[str, Any]:
        return {"version": self.version, "identity": self.identity(),
                "payload_sha256": self.payload_sha256}

    def to_payload(self) -> dict[str, Any]:
        return {**self.identity(), "payload": self.payload.detach().clone()}

    def clone(self) -> "EvidenceRecord":
        return copy.deepcopy(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EvidenceRecord":
        required = {"record_id", "scope", "source_epoch", "available_at",
                    "valid_from", "valid_until", "producer_revision",
                    "representation_version", "codec_version", "payload_type",
                    "source_lineage", "supersedes", "payload"}
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise SpineError("evidence record checkpoint keys differ")
        return cls(record_id=payload["record_id"], scope=payload["scope"],
                   payload=payload["payload"], source_epoch=payload["source_epoch"],
                   available_at=payload["available_at"], valid_from=payload["valid_from"],
                   producer_revision=payload["producer_revision"],
                   representation_version=payload["representation_version"],
                   codec_version=payload["codec_version"],
                   payload_type=payload["payload_type"],
                   source_lineage=payload["source_lineage"],
                   valid_until=payload["valid_until"], supersedes=payload["supersedes"])


class EvidenceBank:
    """Bounded E storage with a separately declared bounded semantic read cap."""

    def __init__(self, scope: str, *, max_records: int = 64,
                 max_payload_bytes: int = 1 << 20, max_read_records: int = 8,
                 max_history_records: int = 256,
                 max_history_bytes: int = 4 << 20,
                 max_cached_snapshots: int = 16, max_revocations: int = 256,
                 max_record_metadata_bytes: int = 64 << 10,
                 representation_version: str = "repr-1",
                 codec_version: str = "codec-1") -> None:
        if not isinstance(scope, str) or not scope:
            raise SpineError("evidence bank scope is required")
        for name, value in (
                ("max_records", max_records), ("max_payload_bytes", max_payload_bytes),
                ("max_read_records", max_read_records),
                ("max_history_records", max_history_records),
                ("max_history_bytes", max_history_bytes),
                ("max_cached_snapshots", max_cached_snapshots),
                ("max_revocations", max_revocations),
                ("max_record_metadata_bytes", max_record_metadata_bytes)):
            _positive(value, name)
        if max_history_records < max_records or max_history_bytes < max_payload_bytes:
            raise SpineError("history bounds must contain the active evidence bounds")
        if not isinstance(representation_version, str) or not representation_version \
                or not isinstance(codec_version, str) or not codec_version:
            raise SpineError("evidence representation/codec versions are required")
        self.scope = scope
        self.max_records = max_records
        self.max_payload_bytes = max_payload_bytes
        self.max_read_records = max_read_records
        self.max_history_records = max_history_records
        self.max_history_bytes = max_history_bytes
        self.max_cached_snapshots = max_cached_snapshots
        self.max_revocations = max_revocations
        self.max_record_metadata_bytes = max_record_metadata_bytes
        self.representation_version = representation_version
        self.codec_version = codec_version
        self._active: OrderedDict[str, EvidenceRecord] = OrderedDict()
        self._history: list[EvidenceRecord] = []
        self._revocations: OrderedDict[str, str] = OrderedDict()
        self._revision = 0
        self._payload_bytes = 0
        self._cache: OrderedDict[tuple, Packet] = OrderedDict()
        self._cache_evictions = 0
        self._cache_invalidations = 0
        self._listeners: weakref.WeakSet = weakref.WeakSet()
        self._last_read: dict[str, Any] = {
            "eligible_records": 0, "read_records": 0, "omitted_records": 0,
            "query_driven": False, "cache_hits": 0,
        }

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def pending(self) -> int:
        return len(self._active)

    @property
    def payload_bytes(self) -> int:
        return self._payload_bytes

    @property
    def history_payload_bytes(self) -> int:
        return sum(record.payload_bytes for record in self._history)

    @property
    def history(self) -> tuple[EvidenceRecord, ...]:
        return tuple(record.clone() for record in self._history)

    @property
    def active(self) -> tuple[EvidenceRecord, ...]:
        return tuple(record.clone() for record in self._active.values())

    @property
    def last_read(self) -> dict[str, Any]:
        return copy.deepcopy(self._last_read)

    def add_invalidation_listener(self, listener: Any) -> None:
        if not callable(getattr(listener, "_evidence_invalidated", None)):
            raise SpineError("evidence listener lacks _evidence_invalidated")
        self._listeners.add(listener)
    def _clear_cache(self) -> None:
        if self._cache:
            self._cache_invalidations += 1
        self._cache.clear()


    def accounting(self) -> dict[str, Any]:
        return {
            "active_records": len(self._active), "active_payload_bytes": self._payload_bytes,
            "max_active_records": self.max_records,
            "max_active_payload_bytes": self.max_payload_bytes,
            "history_records": len(self._history),
            "history_payload_bytes": self.history_payload_bytes,
            "max_history_records": self.max_history_records,
            "max_history_payload_bytes": self.max_history_bytes,
            "revocations": len(self._revocations), "max_revocations": self.max_revocations,
            "cached_snapshots": len(self._cache),
            "max_cached_snapshots": self.max_cached_snapshots,
            "cache_evictions": self._cache_evictions,
            "cache_invalidations": self._cache_invalidations,
            "max_read_records": self.max_read_records,
            "last_read": self.last_read,
        }

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "format": EVIDENCE_CHECKPOINT_FORMAT, "scope": self.scope,
            "max_records": self.max_records, "max_payload_bytes": self.max_payload_bytes,
            "max_read_records": self.max_read_records,
            "max_history_records": self.max_history_records,
            "max_history_bytes": self.max_history_bytes,
            "max_cached_snapshots": self.max_cached_snapshots,
            "max_revocations": self.max_revocations,
            "max_record_metadata_bytes": self.max_record_metadata_bytes,
            "representation_version": self.representation_version,
            "codec_version": self.codec_version,
            "revision": self._revision,
            "active": [record.record_id for record in self._active.values()],
            "history": [record.to_payload() for record in self._history],
            "revocations": dict(self._revocations),
            "payload_bytes": self._payload_bytes,
            "history_payload_bytes": self.history_payload_bytes,
        }

    def load_checkpoint(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise SpineError("evidence checkpoint must be a mapping")
        legacy_keys = {"format", "scope", "max_records", "max_payload_bytes",
                       "revision", "active", "history", "revocations", "payload_bytes"}
        if payload.get("format") == LEGACY_EVIDENCE_CHECKPOINT_FORMAT:
            if set(payload) != legacy_keys:
                raise SpineError("legacy evidence checkpoint keys differ")
            if not isinstance(payload["history"], list):
                raise SpineError("legacy evidence checkpoint history must be a list")
            # Explicit v1 -> v2 migration: v1 had no durable read/cache/history
            # controls, so the already-constructed owner's declared bounds apply.
            payload = {**payload, "format": EVIDENCE_CHECKPOINT_FORMAT,
                       "max_read_records": self.max_read_records,
                       "max_history_records": self.max_history_records,
                       "max_history_bytes": self.max_history_bytes,
                       "max_cached_snapshots": self.max_cached_snapshots,
                       "max_revocations": self.max_revocations,
                       "max_record_metadata_bytes": self.max_record_metadata_bytes,
                       "representation_version": self.representation_version,
                       "codec_version": self.codec_version,
                       "history_payload_bytes": sum(
                           EvidenceRecord.from_payload(item).payload_bytes
                           for item in payload["history"])}
        required = {"format", "scope", "max_records", "max_payload_bytes",
                    "max_read_records", "max_history_records", "max_history_bytes",
                    "max_cached_snapshots", "max_revocations",
                    "max_record_metadata_bytes", "representation_version",
                    "codec_version", "revision", "active", "history", "revocations",
                    "payload_bytes", "history_payload_bytes"}
        if set(payload) != required or payload["format"] != EVIDENCE_CHECKPOINT_FORMAT:
            raise SpineError("evidence checkpoint keys/format differ")
        for key in ("scope", "max_records", "max_payload_bytes", "max_read_records",
                    "max_history_records", "max_history_bytes", "max_cached_snapshots",
                    "max_revocations", "max_record_metadata_bytes",
                    "representation_version", "codec_version"):
            if payload[key] != getattr(self, key):
                raise SpineError(f"evidence checkpoint {key} differs")
        if type(payload["revision"]) is not int or payload["revision"] < 0:
            raise SpineError("evidence checkpoint revision is invalid")
        if not isinstance(payload["active"], list) or not isinstance(payload["history"], list):
            raise SpineError("evidence checkpoint records must be lists")
        active_ids = payload["active"]
        if any(not isinstance(record_id, str) or not record_id for record_id in active_ids):
            raise SpineError("evidence checkpoint active identities are invalid")
        if len(set(active_ids)) != len(active_ids):
            raise SpineError("evidence checkpoint has duplicate active identities")
        history = [EvidenceRecord.from_payload(item) for item in payload["history"]]
        history_ids = [record.record_id for record in history]
        if len(set(history_ids)) != len(history_ids):
            raise SpineError("evidence checkpoint has duplicate lineage identities")
        if any(record.scope != self.scope for record in history):
            raise SpineError("evidence checkpoint record scope differs")
        if any(record.representation_version != self.representation_version
               or record.codec_version != self.codec_version for record in history):
            raise SpineError("evidence checkpoint representation/codec differs")
        if any(record.metadata_bytes > self.max_record_metadata_bytes for record in history):
            raise SpineError("evidence checkpoint metadata exceeds its bound")
        history_by_id = {record.record_id: record for record in history}
        if any(record_id not in history_by_id for record_id in active_ids):
            raise SpineError("evidence checkpoint active record lacks lineage")
        active = [history_by_id[record_id] for record_id in active_ids]
        revocations = payload["revocations"]
        if not isinstance(revocations, Mapping) or any(
                not isinstance(key, str) or not key
                or not isinstance(value, str) or not value
                for key, value in revocations.items()):
            raise SpineError("evidence checkpoint revocations are invalid")
        if any(record_id not in history_by_id for record_id in revocations):
            raise SpineError("evidence checkpoint revocation lacks lineage")
        if any(record_id in revocations for record_id in active_ids):
            raise SpineError("evidence checkpoint has an active revoked record")
        payload_bytes = sum(record.payload_bytes for record in active)
        history_bytes = sum(record.payload_bytes for record in history)
        if type(payload["payload_bytes"]) is not int \
                or payload["payload_bytes"] != payload_bytes \
                or type(payload["history_payload_bytes"]) is not int \
                or payload["history_payload_bytes"] != history_bytes:
            raise SpineError("evidence checkpoint byte count differs")
        if len(active) > self.max_records or payload_bytes > self.max_payload_bytes \
                or len(history) > self.max_history_records \
                or history_bytes > self.max_history_bytes \
                or len(revocations) > self.max_revocations:
            raise SpineError("evidence checkpoint exceeds configured limits")
        self._active = OrderedDict((record.record_id, record) for record in active)
        self._history = history
        self._revocations = OrderedDict(revocations)
        self._revision = payload["revision"]
        self._payload_bytes = payload_bytes
        self._clear_cache()
        for listener in list(self._listeners):
            listener._evidence_invalidated()

    def _bump(self) -> None:
        self._revision += 1
        self._clear_cache()
        for listener in list(self._listeners):
            listener._evidence_invalidated()

    def admit(self, record: EvidenceRecord, *, allow_orphan_correction: bool = False) -> bool:
        if not isinstance(record, EvidenceRecord):
            raise SpineError("evidence bank admits EvidenceRecord instances")
        if type(allow_orphan_correction) is not bool:
            raise SpineError("orphan-correction reconciliation flag must be boolean")
        if record.scope != self.scope:
            raise SpineError("evidence scope differs from the bank")
        if record.representation_version != self.representation_version \
                or record.codec_version != self.codec_version:
            raise SpineError("evidence representation/codec version differs from the bank")
        if record.metadata_bytes > self.max_record_metadata_bytes:
            raise SpineError("evidence record metadata capacity exceeded")
        if record.record_id in self._revocations:
            raise SpineError("a revoked record cannot be re-admitted without a new identity")
        current = self._active.get(record.record_id)
        if current is not None and record.supersedes is None:
            if current.identity() == record.identity() and torch.equal(
                    current.payload, record.payload):
                return False
            raise SpineError("duplicate evidence record with changed content")
        if current is not None:
            raise SpineError("correction record identity is already active")
        if current is None and any(previous.record_id == record.record_id
                                   for previous in self._history):
            raise SpineError("evidence record identity was already used")
        parent = None
        if record.supersedes is not None:
            parent = self._active.get(record.supersedes)
            if parent is None and not allow_orphan_correction:
                raise SpineError("correction parent is not active")
            if parent is not None and parent.record_id == record.record_id:
                raise SpineError("correction requires a new record identity")
            if parent is not None and record.source_epoch <= parent.source_epoch:
                raise SpineError("correction source epoch must increase")
            if parent is not None and record.available_at < parent.available_at:
                raise SpineError("correction cannot be available before its parent")
        active_count = len(self._active) - (1 if parent is not None else 0)
        if active_count >= self.max_records:
            raise SpineError("evidence record capacity exceeded")
        removed_bytes = parent.payload_bytes if parent is not None else 0
        next_bytes = self._payload_bytes - removed_bytes + record.payload_bytes
        if next_bytes > self.max_payload_bytes:
            raise SpineError("evidence payload byte capacity exceeded")
        if len(self._history) >= self.max_history_records:
            raise SpineError("evidence history record capacity exceeded")
        if self.history_payload_bytes + record.payload_bytes > self.max_history_bytes:
            raise SpineError("evidence history payload byte capacity exceeded")
        if parent is not None:
            del self._active[parent.record_id]
        self._active[record.record_id] = record.clone()
        self._history.append(record.clone())
        self._payload_bytes = next_bytes
        self._bump()
        return True

    def revoke(self, record_id: str, *, reason: str) -> None:
        if record_id not in self._active:
            raise SpineError("cannot revoke an unknown or already inactive record")
        if not isinstance(reason, str) or not reason:
            raise SpineError("revocation reason is required")
        if len(self._revocations) >= self.max_revocations:
            raise SpineError("evidence revocation capacity exceeded")
        record = self._active[record_id]
        self._revocations[record_id] = reason
        self._payload_bytes -= record.payload_bytes
        del self._active[record_id]
        self._bump()

    def clear(self) -> None:
        """Clear all E payloads/tombstones; compact P teaching lineage is external."""
        self._active.clear()
        self._history.clear()
        self._revocations.clear()
        self._payload_bytes = 0
        self._bump()

    @staticmethod
    def _query_tensor(query_features: torch.Tensor | None,
                      query_count: int) -> torch.Tensor | None:
        if query_features is None:
            return None
        if not torch.is_tensor(query_features) or not torch.is_floating_point(query_features):
            raise SpineError("query_features must be a floating tensor")
        if query_features.ndim == 2:
            query_features = query_features.unsqueeze(0)
        if query_features.ndim != 3 or query_features.shape[0] != 1 \
                or query_features.shape[1] != query_count \
                or query_features.shape[2] != 512:
            raise SpineError("E query_features must have shape [1,Q,512] or [Q,512]")
        if not bool(torch.isfinite(query_features).all()):
            raise SpineError("query_features contains nonfinite values")
        return query_features.detach().to(dtype=torch.float32).clone()

    def snapshot(self, *, query_time: int, query_count: int,
                 decision: DecisionSnapshot,
                 query_features: torch.Tensor | None = None) -> Packet:
        if not isinstance(decision, DecisionSnapshot):
            raise SpineError("evidence snapshot requires a DecisionSnapshot")
        if type(query_time) is not int or query_time < 0 \
                or type(query_count) is not int or query_count <= 0:
            raise SpineError("query time/count is invalid")
        if decision.scope != self.scope:
            raise SpineError("decision scope differs from evidence bank")
        if decision.state_revision != self.revision:
            raise SpineError("decision used a stale evidence revision")
        query = self._query_tensor(query_features, query_count)
        query_digest = None if query is None else _tensor_digest(query)
        key = (query_time, query_count, decision.decision_id, self.revision, query_digest)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._last_read["cache_hits"] += 1
            return self._cache[key].clone()
        eligible = [record for record in self._active.values()
                    if record.valid_from <= query_time
                    and record.available_at <= query_time
                    and (record.valid_until is None or query_time <= record.valid_until)]
        eligible.sort(key=lambda record: (record.available_at, record.source_epoch,
                                          record.record_id))
        if query is not None and eligible:
            question = query[0].mean(dim=0)
            question = torch.nn.functional.normalize(question, dim=0)
            matrix = torch.stack([record.payload.to(dtype=torch.float32)
                                  for record in eligible])
            matrix = torch.nn.functional.normalize(matrix, dim=1)
            scores = matrix @ question
            order = sorted(range(len(eligible)),
                           key=lambda index: (-float(scores[index]), index))
        else:
            order = list(range(len(eligible)))
        selected_indices = order[:self.max_read_records]
        selected = [eligible[index] for index in selected_indices]
        selected_set = set(selected_indices)
        omitted = [record for index, record in enumerate(eligible)
                   if index not in selected_set]
        if selected:
            payload = torch.stack([record.payload for record in selected]).unsqueeze(0)
            availability = torch.ones(1, query_count, len(selected), dtype=torch.bool)
        else:
            payload = torch.zeros(1, 0, 512)
            availability = torch.zeros(1, query_count, 0, dtype=torch.bool)
        omitted_lineage = [{
            "record_id": record.record_id, "version": record.version,
            "source_epoch": record.source_epoch, "available_at": record.available_at,
            "valid_from": record.valid_from, "valid_until": record.valid_until,
            "producer_revision": record.producer_revision,
            "representation_version": record.representation_version,
            "codec_version": record.codec_version, "supersedes": record.supersedes,
            "source_lineage": copy.deepcopy(record.source_lineage),
        } for record in omitted]
        lineage = {
            "record_ids": [record.record_id for record in selected],
            "record_versions": [record.version for record in selected],
            "eligible_record_count": len(eligible), "read_record_count": len(selected),
            "stored_record_count": len(self._active),
            "read_budget": self.max_read_records,
            "omitted_record_count": len(omitted),
            "omitted_record_ids": [record.record_id for record in omitted],
            "omitted_records": omitted_lineage,
            "history_versions": len(self._history), "query_driven": query is not None,
            "query_sha256": query_digest,
        }
        packet = Packet(
            schema="replica.spark.packet.v1", producer_revision="evidence-bank-v2",
            representation_version=self.representation_version,
            codec_version=self.codec_version, scope=self.scope, stream_id="memory",
            event_id="memory-snapshot", sequence=query_time,
            observation_start=query_time, observation_end=query_time,
            available_at=query_time, state_revision=self.revision,
            payload_type="observation", source_lineage=lineage,
            payload=payload, availability=availability)
        self._cache[key] = packet
        while len(self._cache) > self.max_cached_snapshots:
            self._cache.popitem(last=False)
            self._cache_evictions += 1
        self._last_read = {
            "eligible_records": len(eligible), "read_records": len(selected),
            "omitted_records": len(omitted), "query_driven": query is not None,
            "cache_hits": self._last_read["cache_hits"],
        }
        return packet.clone()


class SpineMemoryExperiment:
    """Bounded ordered observe/read/teach/commit learner with durable lineage."""

    def __init__(self, *, scope: str = "task", spec: SpineSpec,
                 bank_experts: int = 2, bank_hidden_width: int = 64,
                 seed: int = 0, core: SpineCore | None = None,
                 evidence: EvidenceBank | None = None, read_records: int | None = None,
                 max_journal_records: int = 64, max_taught_evidence: int = 256,
                 max_cached_snapshots: int = 16,
                 max_transient_entries: int = 64,
                 max_transient_payload_bytes: int = 1 << 20,
                 codec_version: str = "codec-1",
                 representation_version: str = "repr-1") -> None:
        if core is not None and (not isinstance(core, SpineCore) or core.spec != spec):
            raise SpineError("supplied core does not match the memory experiment spec")
        if read_records is None:
            read_records = (evidence.max_read_records if evidence is not None
                            else min(8, spec.slots))
        _positive(read_records, "read_records")
        if read_records > spec.slots:
            raise SpineError("read_records exceeds the declared spine slot codec cap")
        for name, value in (("max_journal_records", max_journal_records),
                            ("max_taught_evidence", max_taught_evidence),
                            ("max_cached_snapshots", max_cached_snapshots),
                            ("max_transient_entries", max_transient_entries),
                            ("max_transient_payload_bytes", max_transient_payload_bytes)):
            _positive(value, name)
        self.scope = scope
        self.read_records = read_records
        self.max_journal_records = max_journal_records
        self.max_taught_evidence = max_taught_evidence
        self.core = (SpineCore(spec, codec_version=codec_version,
                               representation_version=representation_version)
                     if core is None else core)
        if self.core.codec_version != codec_version \
                or self.core.representation_version != representation_version:
            raise SpineError("supplied core representation/codec versions differ")
        if evidence is None:
            evidence = EvidenceBank(
                scope, max_read_records=read_records,
                max_cached_snapshots=max_cached_snapshots,
                representation_version=representation_version,
                codec_version=codec_version)
        if evidence.scope != scope or evidence.max_read_records != read_records \
                or evidence.representation_version != representation_version \
                or evidence.codec_version != codec_version:
            raise SpineError("supplied evidence bank identity/read budget differs")
        self.evidence = evidence
        self.evidence.add_invalidation_listener(self)
        self.bank = ExpertBank(expert_count=bank_experts, width=256,
                               hidden_width=bank_hidden_width, seed=seed,
                               scope="spine_memory")
        self.controller = PlasticLearningController(self.bank, scope=scope)
        parameters = list(self.core.parameters())
        self.core_optimizer = NativeFusedAdamW([{
            "params": parameters, "lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8,
            "weight_decay": 0.0, "algorithm": "adamw_fused", "fused": True,
            "group_name": "spine_core", "scope": f"{scope}/spine",
            "param_names": [name for name, _ in self.core.named_parameters()],
        }])
        self.arm = "arm-0"
        self.phase = "observation"
        self.counters: dict[str, int] = {
            "observations": 0, "snapshots": 0, "learning_events": 0,
            "learning_transactions": 0, "retried_learning_events": 0,
            "optimizer_steps": 0, "backward_calls": 0, "forward_calls": 0,
            "fit_evaluation_forward_calls": 0, "codec_calls": 0,
            "p_only_clears": 0, "read_calls": 0, "p_read_calls": 0,
            "e_read_calls": 0, "omitted_read_records": 0,
            "changed_tensors": 0, "invalidations": 0,
            "dropped_journal_records": 0, "dropped_taught_lineages": 0,
        }
        self.completed_evaluations: list[dict[str, Any]] = []
        self.learning_journal: list[dict[str, Any]] = []
        self.taught_evidence: OrderedDict[str, dict[str, Any]] = OrderedDict()
        cache = lambda: _BoundedCache(
            max_entries=max_transient_entries,
            max_payload_bytes=max_transient_payload_bytes)
        self.source_cache = cache()
        self.provider_cache = cache()
        self.source_kv = cache()
        self.fast_state = cache()
        self.workspace_state = cache()
        self._snapshot_context: SpineForwardContext | None = None
        self._contexts: weakref.WeakSet[SpineForwardContext] = weakref.WeakSet()
        self.max_live_contexts = max_transient_entries
        self._state_generation = 0
        self._lineage_complete = True

    def _evidence_invalidated(self) -> None:
        self.invalidate(reason="evidence_revision")

    def invalidate(self, *, reason: str = "parameter_or_representation_change") -> None:
        """Invalidate every public derived view after any owned state change."""
        if not isinstance(reason, str) or not reason:
            raise SpineError("invalidation reason is required")
        for context in list(self._contexts):
            context.invalidate()
        self._contexts.clear()
        self._snapshot_context = None
        for cache in (self.source_cache, self.provider_cache, self.source_kv,
                      self.fast_state, self.workspace_state):
            cache.clear()
        self.evidence._clear_cache()
        self._state_generation += 1
        self.counters["invalidations"] += 1

    def _invalidate_snapshot(self) -> None:
        """Backward-compatible alias for complete public invalidation."""
        self.invalidate(reason="spine_state_change")

    def observe(self, record: EvidenceRecord, *, reconcile: bool = False) -> bool:
        admitted = self.evidence.admit(record, allow_orphan_correction=reconcile)
        if admitted:
            self.counters["observations"] += 1
            self.phase = "observation"
        return admitted

    def revoke(self, record_id: str, *, reason: str) -> None:
        self.evidence.revoke(record_id, reason=reason)
        self.phase = "observation"

    def _decision(self, query_time: int, phase: str) -> DecisionSnapshot:
        return DecisionSnapshot(
            decision_id=f"{self.arm}:{query_time}:{phase}", scope=self.scope,
            stream_id="memory", state_revision=self.evidence.revision,
            arm=self.arm, event=f"event-{query_time}", phase=phase)

    def _register(self, context: SpineForwardContext, phase: str) -> SpineForwardContext:
        if len(self._contexts) >= self.max_live_contexts:
            raise SpineError("live decision context capacity exceeded")
        self._snapshot_context = context
        self._contexts.add(context)
        self.counters["snapshots"] += 1
        self.counters["codec_calls"] += context.charges["codec_calls"]
        self.phase = phase
        return context

    def _query_features(self, features: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(features) or not torch.is_floating_point(features):
            raise SpineError("query_features must be a floating tensor")
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim != 3 or features.shape[-1] != 512 \
                or features.shape[0] <= 0 or features.shape[1] <= 0:
            raise SpineError("query_features must have shape [B,Q,512] or [Q,512]")
        if not bool(torch.isfinite(features).all()):
            raise SpineError("query_features contains nonfinite values")
        device = self.core.memory_projection.weight.device
        return features.detach().to(device=device, dtype=torch.float32).clone()

    def _native_hidden(self, hidden: torch.Tensor | None,
                       query: torch.Tensor) -> torch.Tensor | None:
        if hidden is None:
            return None
        if not torch.is_tensor(hidden) or not torch.is_floating_point(hidden) \
                or hidden.ndim != 3 \
                or tuple(hidden.shape[:2]) != tuple(query.shape[:2]) \
                or hidden.shape[2] != self.core.spec.hidden_size:
            raise SpineError("native_hidden must have shape [B,Q,hidden_size]")
        if not bool(torch.isfinite(hidden).all()):
            raise SpineError("native_hidden contains nonfinite values")
        return hidden.to(device=query.device, dtype=torch.float32)

    def read(self, *, query_features: torch.Tensor,
             native_hidden: torch.Tensor | None = None, query_time: int = 0,
             plastic_expert_id: int | None = None,
             enabled_branches=("memory",), phase: str = "read") -> SpineForwardContext:
        """Read E or retained P using only current query/native hidden features."""
        query = self._query_features(query_features)
        native = self._native_hidden(native_hidden, query)
        decision = self._decision(query_time, phase)
        memory_slots = None
        memory_availability = None
        if plastic_expert_id is not None:
            self.bank._validate_expert(plastic_expert_id)
            projected = self.core.memory_projection(query)
            if native is not None:
                port = self.core.ports["memory"][str(self.core.spec.sites[0])]
                projected = projected + port.query_projection(port.query_norm(native))
            memory_slots = self.bank.expert(plastic_expert_id)(projected).unsqueeze(2)
            memory_availability = torch.ones(
                memory_slots.shape[:3], dtype=torch.bool, device=query.device)
            packet = Packet(
                schema="replica.spark.packet.v1", producer_revision="p-query-reader-v1",
                representation_version=self.core.representation_version,
                codec_version=self.core.codec_version, scope=self.scope,
                stream_id="memory", event_id="p-query", sequence=query_time,
                observation_start=query_time, observation_end=query_time,
                available_at=query_time, state_revision=self.evidence.revision,
                payload_type="observation",
                source_lineage={"source": "current_query",
                                "query_read_version": P_QUERY_READ_VERSION,
                                "query_sha256": _tensor_digest(query),
                                "evidence_record_ids": [],
                                "p_expert_id": plastic_expert_id},
                payload=torch.zeros(query.shape[0], 0, 512, device=query.device),
                availability=torch.zeros(query.shape[0], query.shape[1], 0,
                                         dtype=torch.bool, device=query.device))
            source = "P"
            self.counters["p_read_calls"] += 1
        else:
            packet = self.evidence.snapshot(
                query_time=query_time, query_count=query.shape[1], decision=decision,
                query_features=query)
            source = "E"
            self.counters["e_read_calls"] += 1
            self.counters["omitted_read_records"] += int(
                packet.source_lineage["omitted_record_count"])
        context = SpineForwardContext(
            self.core, packet, decision, enabled_branches=tuple(enabled_branches),
            memory_slots=memory_slots, memory_source=source,
            memory_availability=memory_availability)
        self.counters["read_calls"] += 1
        return self._register(context, phase)

    def snapshot(self, *, query_time: int, query_count: int,
                 phase: str = "read", plastic_expert_id: int | None = None,
                 enabled_branches=BRANCHES,
                 query_features: torch.Tensor | None = None,
                 native_hidden: torch.Tensor | None = None) -> SpineForwardContext:
        if query_features is not None or native_hidden is not None:
            if query_features is None:
                raise SpineError("native_hidden requires query_features")
            return self.read(query_features=query_features, native_hidden=native_hidden,
                             query_time=query_time, plastic_expert_id=plastic_expert_id,
                             enabled_branches=enabled_branches, phase=phase)
        decision = self._decision(query_time, phase)
        packet = self.evidence.snapshot(
            query_time=query_time, query_count=query_count, decision=decision)
        self.counters["e_read_calls"] += 1
        self.counters["read_calls"] += 1
        self.counters["omitted_read_records"] += int(
            packet.source_lineage["omitted_record_count"])
        memory_slots = None
        if plastic_expert_id is not None and "memory" in tuple(enabled_branches):
            if packet.slot_count == 0:
                raise SpineError(
                    "P-only snapshot requires query_features; use read(...)")
            memory_slots = self.bank.expert(plastic_expert_id)(
                self.core.encode(packet)["memory"])
            self.counters["codec_calls"] += packet.query_count
        context = SpineForwardContext(
            self.core, packet, decision, enabled_branches=enabled_branches,
            memory_slots=memory_slots,
            memory_source="P" if memory_slots is not None else "E")
        return self._register(context, phase)

    @staticmethod
    def _loss(expert: Any, features: torch.Tensor, target: torch.Tensor) -> float:
        with torch.no_grad():
            return float(0.5 * (expert(features) - target).pow(2).sum())

    @staticmethod
    def _parameters(expert: Any) -> dict[str, torch.Tensor]:
        return {name: parameter.detach().clone()
                for name, parameter in expert.named_parameters()}

    def _append_bounded(self, target: list[dict[str, Any]], row: dict[str, Any],
                        counter: str) -> None:
        if len(target) >= self.max_journal_records:
            target.pop(0)
            self.counters[counter] += 1
        target.append(copy.deepcopy(row))

    def _remember_teaching(self, records: Iterable[EvidenceRecord]) -> list[str]:
        versions = []
        for record in records:
            version = record.version
            versions.append(version)
            if version not in self.taught_evidence \
                    and len(self.taught_evidence) >= self.max_taught_evidence:
                self.taught_evidence.popitem(last=False)
                self.counters["dropped_taught_lineages"] += 1
            self.taught_evidence[version] = record.lineage_payload()
            self.taught_evidence.move_to_end(version)
        return versions

    def learn(self, *, event_id: str, features: torch.Tensor,
              target: torch.Tensor, expert_id: int = 0, steps: int = 1,
              teaching_evidence: Iterable[EvidenceRecord] | None = None) -> dict[str, Any]:
        records = tuple(self.evidence.active if teaching_evidence is None
                        else teaching_evidence)
        active_versions = {record.version for record in self.evidence.active}
        if any(not isinstance(record, EvidenceRecord) or record.version not in active_versions
               for record in records):
            raise SpineError("teaching_evidence must name exact currently active versions")
        expert = self.bank.expert(expert_id)
        before = self._parameters(expert)
        outcome = self.controller.apply_teaching_transaction(
            event_id=event_id, expert_id=expert_id, features=features,
            target=target, steps=steps)
        retried = outcome.retried
        if retried:
            self.counters["retried_learning_events"] += 1
            evidence = {
                "declared_updates": outcome.steps, "actual_updates": 0,
                "execution_complete": True, "parameter_update": False,
                "optimizer_steps": 0, "backward_calls": 0, "forward_calls": 0,
                "fit_evaluation_forward_calls": 0,
                "changed_trainable_names": [], "changed_tensor_count": 0,
                "train_fit": {"evaluated": False, "loss_before": None,
                              "loss_after": None, "improved": None},
                "transfer": {"evaluated": False, "improved": None},
                "retention": {"evaluated": False, "retained": None},
                "useful_learning": False, "learning_success": False,
                "status": "retried", "retried": True,
            }
            self.phase = "commit"
            return {"outcome": {**outcome.to_dict(), "retried": True},
                    "learning_evidence": evidence}
        actual_steps = outcome.steps
        after = self._parameters(expert)
        changed = [f"expert_{expert_id}.{name}" for name in before
                   if not torch.equal(before[name], after[name])]
        loss_after = self._loss(expert, features, target)
        loss_before = float(outcome.loss)
        fit_improved = loss_after < loss_before
        parameter_update = bool(changed)
        useful_learning = parameter_update and fit_improved
        self.counters["learning_transactions"] += 1
        self.counters["learning_events"] += int(parameter_update)
        self.counters["optimizer_steps"] += actual_steps
        self.counters["backward_calls"] += actual_steps
        self.counters["forward_calls"] += actual_steps
        self.counters["fit_evaluation_forward_calls"] += 1
        self.counters["changed_tensors"] += len(changed)
        train_fit = {"evaluated": True, "loss_before": loss_before,
                     "loss_after": loss_after, "improved": fit_improved}
        evidence = {
            "declared_updates": outcome.steps, "actual_updates": actual_steps,
            "execution_complete": True, "parameter_update": parameter_update,
            "optimizer_steps": actual_steps, "backward_calls": actual_steps,
            "forward_calls": actual_steps, "fit_evaluation_forward_calls": 1,
            "changed_trainable_names": changed, "changed_tensor_count": len(changed),
            "train_fit": train_fit,
            "transfer": {"evaluated": False, "improved": None},
            "retention": {"evaluated": False, "retained": None},
            "useful_learning": useful_learning, "learning_success": useful_learning,
            "status": ("passed" if useful_learning else
                       "no_op" if not parameter_update else "fit_not_improved"),
            "retried": False,
        }
        row = {"event_id": event_id, "expert_id": expert_id,
               "outcome": outcome.to_dict(), "learning_evidence": copy.deepcopy(evidence)}
        self._append_bounded(self.completed_evaluations, row, "dropped_journal_records")
        versions = self._remember_teaching(records) if parameter_update else []
        journal_row = {
            "event_id": event_id, "expert_id": expert_id, "steps": actual_steps,
            "controller_revision": outcome.revision,
            "evidence_versions": versions, "parameter_update": parameter_update,
            "changed_trainable_names": changed, "train_fit": train_fit,
            "useful_learning": useful_learning,
            "replay": {"event_id": event_id, "expert_id": expert_id,
                       "steps": steps, "features_sha256": _tensor_digest(features),
                       "target_sha256": _tensor_digest(target)},
        }
        self._append_bounded(self.learning_journal, journal_row,
                             "dropped_journal_records")
        self.invalidate(reason="plastic_parameter_change")
        self.phase = "commit"
        return {"outcome": {**outcome.to_dict(), "retried": False},
                "learning_evidence": evidence}

    def clear_for_p_only(self) -> None:
        """Remove E/S/C/F/source state while retaining P and compact lineage."""
        self.evidence.clear()
        self.invalidate(reason="p_only_clear")
        self.counters["p_only_clears"] += 1
        self.phase = "p_only_evaluation"

    def accounting(self) -> dict[str, Any]:
        return {
            "evidence": self.evidence.accounting(),
            "history": {"records": len(self.evidence._history),
                        "bound": self.evidence.max_history_records},
            "journal": {"records": len(self.learning_journal),
                        "completed_evaluations": len(self.completed_evaluations),
                        "bound": self.max_journal_records,
                        "dropped": self.counters["dropped_journal_records"]},
            "replay": {"records": len(self.learning_journal),
                       "bound": self.max_journal_records},
            "teaching_lineage": {"records": len(self.taught_evidence),
                                 "bound": self.max_taught_evidence,
                                 "dropped": self.counters["dropped_taught_lineages"]},
            "caches": {name: cache.accounting() for name, cache in (
                ("source_cache", self.source_cache),
                ("provider_cache", self.provider_cache),
                ("source_kv", self.source_kv), ("fast_state", self.fast_state),
                ("workspace_state", self.workspace_state))},
            "transient": {"live_contexts": len(self._contexts),
                          "max_live_contexts": self.max_live_contexts,
                          "snapshot_contexts": int(self._snapshot_context is not None),
                          "state_generation": self._state_generation},
            "metadata": {"evidence_record_bound":
                         self.evidence.max_record_metadata_bytes},
        }

    def memory_checkpoint_payload(self) -> dict[str, Any]:
        return {
            "format": MEMORY_STATE_FORMAT, "scope": self.scope,
            "codec_version": self.core.codec_version,
            "representation_version": self.core.representation_version,
            "query_read_version": P_QUERY_READ_VERSION,
            "core_spec": self.core.spec.to_dict(), "read_records": self.read_records,
            "max_journal_records": self.max_journal_records,
            "max_taught_evidence": self.max_taught_evidence,
            "learning_journal": copy.deepcopy(self.learning_journal),
            "taught_evidence": [copy.deepcopy(row)
                                for row in self.taught_evidence.values()],
            "state_generation": self._state_generation,
            "lineage_complete": self._lineage_complete,
            "rebuild": {"core_spec": self.core.spec.to_dict(),
                        "bank_experts": self.bank.expert_count,
                        "bank_width": self.bank.width,
                        "bank_hidden_width": self.bank.hidden_width,
                        "bank_seed": self.bank.seed,
                        "derived_caches_required": False,
                        "source_payloads_required_for_p": False},
        }

    def load_memory_checkpoint(self, payload: Mapping[str, Any]) -> None:
        required = {"format", "scope", "codec_version", "representation_version",
                    "query_read_version", "core_spec", "read_records",
                    "max_journal_records", "max_taught_evidence", "learning_journal",
                    "taught_evidence", "state_generation", "lineage_complete", "rebuild"}
        if not isinstance(payload, Mapping) or set(payload) != required \
                or payload["format"] != MEMORY_STATE_FORMAT:
            raise SpineError("memory state checkpoint keys/format differ")
        expected = {"scope": self.scope, "codec_version": self.core.codec_version,
                    "representation_version": self.core.representation_version,
                    "query_read_version": P_QUERY_READ_VERSION,
                    "core_spec": self.core.spec.to_dict(),
                    "read_records": self.read_records,
                    "max_journal_records": self.max_journal_records,
                    "max_taught_evidence": self.max_taught_evidence}
        for key, value in expected.items():
            if payload[key] != value:
                raise SpineError(f"memory state checkpoint {key} differs")
        if type(payload["lineage_complete"]) is not bool \
                or type(payload["state_generation"]) is not int \
                or payload["state_generation"] < 0:
            raise SpineError("memory state checkpoint lineage/generation is invalid")
        if not isinstance(payload["learning_journal"], list) \
                or len(payload["learning_journal"]) > self.max_journal_records \
                or any(not isinstance(row, dict) for row in payload["learning_journal"]):
            raise SpineError("memory state learning journal is invalid")
        if not isinstance(payload["taught_evidence"], list) \
                or len(payload["taught_evidence"]) > self.max_taught_evidence:
            raise SpineError("memory state teaching lineage is invalid")
        taught = payload["taught_evidence"]
        versions = []
        for row in taught:
            if not isinstance(row, Mapping) or set(row) != {
                    "version", "identity", "payload_sha256"} \
                    or not isinstance(row["version"], str) or not row["version"] \
                    or not isinstance(row["identity"], Mapping) \
                    or not isinstance(row["payload_sha256"], str) \
                    or len(row["payload_sha256"]) != 64:
                raise SpineError("memory state teaching lineage row is invalid")
            identity = row["identity"]
            identity_keys = {"record_id", "scope", "source_epoch", "available_at",
                             "valid_from", "valid_until", "producer_revision",
                             "representation_version", "codec_version",
                             "payload_type", "source_lineage", "supersedes"}
            if set(identity) != identity_keys \
                    or identity["scope"] != self.scope \
                    or identity["representation_version"] != self.core.representation_version \
                    or identity["codec_version"] != self.core.codec_version:
                raise SpineError("memory state teaching lineage identity differs")
            body = {"identity": identity,
                    "payload_sha256": row["payload_sha256"]}
            digest = hashlib.sha256(json.dumps(
                body, sort_keys=True, default=str).encode("utf-8")).hexdigest()
            if row["version"] != f"{identity['record_id']}@{digest}":
                raise SpineError("memory state teaching lineage version differs")
            versions.append(row["version"])
        if len(set(versions)) != len(versions):
            raise SpineError("memory state teaching lineage has duplicates")
        rebuild = payload["rebuild"]
        rebuild_keys = {"core_spec", "bank_experts", "bank_width",
                        "bank_hidden_width", "bank_seed",
                        "derived_caches_required", "source_payloads_required_for_p"}
        if not isinstance(rebuild, Mapping) or set(rebuild) != rebuild_keys \
                or rebuild.get("core_spec") != self.core.spec.to_dict() \
                or rebuild.get("bank_experts") != self.bank.expert_count \
                or rebuild.get("bank_width") != self.bank.width \
                or rebuild.get("bank_hidden_width") != self.bank.hidden_width \
                or rebuild.get("bank_seed") != self.bank.seed \
                or rebuild.get("derived_caches_required") is not False \
                or rebuild.get("source_payloads_required_for_p") is not False:
            raise SpineError("memory state rebuild contract differs")
        self.learning_journal = copy.deepcopy(payload["learning_journal"])
        self.taught_evidence = OrderedDict(
            (row["version"], copy.deepcopy(dict(row))) for row in taught)
        self._state_generation = payload["state_generation"]
        self._lineage_complete = payload["lineage_complete"]

    def validate_completed_evaluations(self, rows: Any) -> None:
        if not isinstance(rows, list) or len(rows) > self.max_journal_records:
            raise SpineError("checkpoint completed evaluations exceed journal bound")

    def migrate_counters(self, counters: Any, checkpoint_version: int) -> dict[str, int]:
        if not isinstance(counters, Mapping) or checkpoint_version not in (1, 2):
            raise SpineError("memory checkpoint counters/version are invalid")
        if not set(counters).issubset(self.counters):
            raise SpineError("memory checkpoint counters contain unknown names")
        if checkpoint_version == 2 and set(counters) != set(self.counters):
            raise SpineError("v2 memory checkpoint counter inventory differs")
        migrated = {name: 0 for name in self.counters}
        migrated.update(counters)
        return migrated

    def checkpoint(self, root, relative: str, *, run_contract: Mapping[str, Any],
                   meter: Any = None) -> dict:
        return save_spine_checkpoint(
            root, relative, self.core, self.core_optimizer,
            arm=self.arm, event=self._snapshot_context.snapshot.event
            if self._snapshot_context is not None else "event-0",
            phase=self.phase, completed_evaluations=self.completed_evaluations,
            counters=self.counters, run_contract=run_contract, meter=meter,
            plastic=self.controller.checkpoint_payload(),
            evidence=self.evidence.checkpoint_payload(),
            memory=self.memory_checkpoint_payload())

    def restore(self, root, relative: str, *, run_contract: Mapping[str, Any]) -> dict:
        evaluations_before = copy.deepcopy(self.completed_evaluations)
        counters_before = dict(self.counters)
        arm_before, phase_before = self.arm, self.phase
        try:
            metadata = load_spine_checkpoint(
                root, relative, self.core, self.core_optimizer,
                run_contract=run_contract, plastic=self.controller,
                evidence=self.evidence, memory=self)
            self.arm = metadata["arm"]
            self.phase = metadata["phase"]
            self.counters = dict(metadata["counters"])
            self.completed_evaluations = copy.deepcopy(metadata["completed_evaluations"])
            self.invalidate(reason="checkpoint_restore")
        except BaseException:
            self.completed_evaluations = evaluations_before
            self.counters = counters_before
            self.arm, self.phase = arm_before, phase_before
            self.invalidate(reason="checkpoint_restore_rollback")
            raise
        return metadata

    def model_identity(self) -> str:
        return durable_model_identity(self.core)


__all__ = ["EVIDENCE_CHECKPOINT_FORMAT", "MEMORY_STATE_FORMAT",
           "P_QUERY_READ_VERSION", "EvidenceRecord", "EvidenceBank",
           "SpineMemoryExperiment"]
