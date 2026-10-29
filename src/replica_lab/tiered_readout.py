"""Scoped memory readout over the durable tiered memory store.

This module is the readout boundary that a neural query-construction path can
call.  It is deliberately separate from :mod:`replica_lab.tiered_memory`: that
module owns durable admission, selection, cold recall and accounting, while
this module decides how one scoped read is assembled, cached, revalidated and
reported.

What a read actually does
-------------------------
``read_scoped_memory(store, query_text, scope, budget)`` performs, in order:

1. A scope-filtered selection with :meth:`TieredMemoryStore.select`.  The SQL
   scope filter runs before any semantic key is read inside the store; this
   module never widens it.  Selection order is preserved exactly.
2. For each selected record, a real payload acquisition:
   * a bounded in-process payload cache keyed by
     ``(scope, record, revision, representation_version, cluster generation,
     canonical text digest)``;
   * otherwise the store's verified payload reader
     (:meth:`TieredMemoryStore.read_record`) for warm records, or a real
     ``begin_recall`` -> ``wait_recall`` -> ``resume_recall`` ticket for cold
     records (never implicit cold I/O);
   * otherwise, when selection already admitted the payload, the bytes the
     store actually read during selection.
3. Post-acquisition revalidation against the live SQLite index: scope,
   current status (revocation/supersession), revision, cluster generation,
   representation version, retrieval-key identity and ordered-value identity,
   plus digest checks on the payload that will be returned.  A stale or
   revoked record is never served; ``new_decision_boundary`` is set when a
   read is invalidated or a representation changed under the read.
4. Presentation under independent budgets for canonical text bytes and value
   slots.  Budget exhaustion is reported as an explicit ``incomplete`` record;
   nothing is fabricated or silently truncated.

What is *not* claimed
---------------------
* Retrieval keys (512-wide signed unigram/bigram hashes) are separate from the
  ordered readable value slots.  Keys exist only to order candidates and are
  labelled synthetic controls.  They are not a neural encoding of the text.
* Canonical text is returned by reference (exact Unicode string plus digest and
  span offsets).  No semantic-compression claim is made.
* Each record also carries exact restoration provenance read from the store
  row: ``observed_at``, ``event_at``, ``valid_from``, ``valid_until`` (the
  ``valid_to`` column), ``source_lineage`` and its decoded JSON provenance,
  producer/representation/codec versions only when the stored provenance
  actually records them, and a ``durable_ref`` of exact text/key/value files,
  digests and cluster manifest.  ``available_at`` is set only when the decoded
  provenance identifies the persistent-learner mapping
  (``available_at_mapped``); otherwise it is null and ``observed_at`` plus the
  mapping label are returned rather than a substituted timestamp.  Missing
  ``payload_type``/``codec_version``/``supersedes`` stay null with explicit
  availability flags; nothing is guessed for neural-path restoration.
* Values are producer output copied from the store's ordered slots.  They are
  never derived from hashes here, and are not claimed to reside in GPU memory.
* The cache is a small in-process copy with a byte bound.  It does not enlarge
  any tier, is not pinned, and is not store-accounted.  Nothing on the hot,
  warm or cold tier is resized.
* Timings are wall-clock measurements of work actually performed.  There is no
  artificial delay and no page-cache-residency measurement.

Known integration limit
-----------------------
``TieredMemoryStore.select`` reads candidate retrieval keys and the payload
bytes it admits (warm canonical text and ordered values) even when the caller
only needs ordering and metadata. This readout reports exact
``selection_file_reads``/``selection_io_bytes`` from the store, including
candidate keys, admitted payloads and exact fallback reads; the admitted
subset remains separately visible. There is currently no ordering-only
selection call, so a warm read that avoids readout-level payload I/O still pays
the store's internal selection reads. The payload cache removes repeated
*readout-level* reads and all repeated cold reads.

Threading / lifetime
--------------------
One cache is kept per :class:`TieredMemoryStore` instance and is guarded by a
lock.  The store itself remains single-connection/single-thread as documented
by :mod:`replica_lab.tiered_memory`.  Selection handles are released before the
call returns; a pending cold ticket, when the wait budget is exhausted, is
returned explicitly and can be completed with
:func:`resume_scoped_readout`, which performs real ``resume_recall`` I/O.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields as dataclass_fields
import hashlib
import json
import math
import threading
import time
import weakref

from replica_lab.tiered_memory import (
    DEFAULT_COLD_IO_BUDGET_BYTES,
    EVIDENCE_RECORDS_MAX,
    EVIDENCE_VALUE_SLOTS_MAX,
    EXACT_SPAN_BUDGET_BYTES,
    EXACT_TEXT_BUDGET_BYTES,
    KEY_WIDTH,
    MAX_EXACT_SPANS,
    VALUE_SLOTS_PER_RECORD,
    VALUE_WIDTH,
    MemoryScope,
    Outcome,
    RetrievalBudget,
    TieredMemoryStore,
    pack_values,
)

__all__ = [
    "READOUT_FORMAT",
    "DEFAULT_READOUT_CACHE_BYTES",
    "RETRIEVAL_KEY_PRODUCER_ID",
    "RETRIEVAL_KEY_PRODUCER_VERSION",
    "ReadoutBudget",
    "SourceSpan",
    "RetrievalKeyReference",
    "ReadoutValueSet",
    "ScopedMemoryRecord",
    "PendingColdReadout",
    "ReadoutMetrics",
    "merge_readout_metrics",
    "ScopedMemoryReadout",
    "read_scoped_memory",
    "resume_scoped_readout",
    "resume_scoped_memory",
    "readout_contract",
    "readout_cache_state",
    "clear_readout_cache",
]

READOUT_FORMAT = "replica.tiered_readout.v1"
DEFAULT_READOUT_CACHE_BYTES = 2 * 1024**2
RETRIEVAL_KEY_PRODUCER_ID = "deterministic_unigram_bigram"
RETRIEVAL_KEY_PRODUCER_VERSION = "v1"

_READY = "ok"
_OK_STATUSES = ("ok", "degraded")
_FAILURE_PRECEDENCE = (
    "denied", "invalidated", "stale", "in_progress",
    "incomplete", "failed", "unsupported",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _perf_ns() -> int:
    return time.perf_counter_ns()


def _nonnegative_int(name: str, value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _coerce_budget(budget: object) -> "ReadoutBudget":
    if budget is None:
        return ReadoutBudget()
    if isinstance(budget, ReadoutBudget):
        return budget
    if isinstance(budget, RetrievalBudget):
        return ReadoutBudget.from_retrieval_budget(budget)
    raise TypeError("budget must be a ReadoutBudget, a RetrievalBudget or None")


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ReadoutBudget:
    """Independent caps for one scoped read.

    The selection caps mirror :class:`RetrievalBudget` and are enforced by the
    store.  ``max_exact_text_bytes`` and ``max_value_slots`` additionally cap
    what this readout returns.  ``cache_bytes`` bounds only the in-process
    payload copy; it never enlarges a logical tier.
    """

    max_records: int = EVIDENCE_RECORDS_MAX
    max_value_slots: int = EVIDENCE_VALUE_SLOTS_MAX
    max_exact_text_bytes: int = EXACT_TEXT_BUDGET_BYTES
    max_span_bytes: int = EXACT_SPAN_BUDGET_BYTES
    max_spans: int = MAX_EXACT_SPANS
    max_cold_bytes: int = DEFAULT_COLD_IO_BUDGET_BYTES
    cold_wait_seconds: float = 2.0
    cache_bytes: int = DEFAULT_READOUT_CACHE_BYTES
    allow_expanded_context: bool = False

    def __post_init__(self) -> None:
        for name in ("max_records", "max_value_slots", "max_exact_text_bytes",
                     "max_span_bytes", "max_spans", "max_cold_bytes",
                     "cache_bytes"):
            _nonnegative_int(name, getattr(self, name))
        if not math.isfinite(self.cold_wait_seconds) or self.cold_wait_seconds < 0:
            raise ValueError("cold_wait_seconds must be finite and nonnegative")
        if type(self.allow_expanded_context) is not bool:
            raise ValueError("allow_expanded_context must be boolean")
        # Reuse the store's cap checks (raises BudgetError on over-wide calls).
        RetrievalBudget(
            max_records=self.max_records,
            max_value_slots=self.max_value_slots,
            max_exact_text_bytes=self.max_exact_text_bytes,
            max_span_bytes=self.max_span_bytes,
            max_spans=self.max_spans,
            max_cold_bytes=self.max_cold_bytes,
            cold_wait_seconds=self.cold_wait_seconds,
            allow_expanded_context=self.allow_expanded_context,
        )

    def retrieval_budget(self) -> RetrievalBudget:
        return RetrievalBudget(
            max_records=self.max_records,
            max_value_slots=self.max_value_slots,
            max_exact_text_bytes=self.max_exact_text_bytes,
            max_span_bytes=self.max_span_bytes,
            max_spans=self.max_spans,
            max_cold_bytes=self.max_cold_bytes,
            cold_wait_seconds=self.cold_wait_seconds,
            allow_expanded_context=self.allow_expanded_context,
        )

    @classmethod
    def from_retrieval_budget(cls, budget: RetrievalBudget, *,
                              cache_bytes: int | None = None) -> "ReadoutBudget":
        if not isinstance(budget, RetrievalBudget):
            raise TypeError("budget must be a RetrievalBudget")
        return cls(
            max_records=budget.max_records,
            max_value_slots=budget.max_value_slots,
            max_exact_text_bytes=budget.max_exact_text_bytes,
            max_span_bytes=budget.max_span_bytes,
            max_spans=budget.max_spans,
            max_cold_bytes=budget.max_cold_bytes,
            cold_wait_seconds=budget.cold_wait_seconds,
            allow_expanded_context=budget.allow_expanded_context,
            cache_bytes=(DEFAULT_READOUT_CACHE_BYTES if cache_bytes is None
                         else cache_bytes),
        )

    def document(self) -> dict:
        return {
            "max_records": self.max_records,
            "max_value_slots": self.max_value_slots,
            "max_exact_text_bytes": self.max_exact_text_bytes,
            "max_span_bytes": self.max_span_bytes,
            "max_spans": self.max_spans,
            "max_cold_bytes": self.max_cold_bytes,
            "cold_wait_seconds": self.cold_wait_seconds,
            "cache_bytes": self.cache_bytes,
            "allow_expanded_context": self.allow_expanded_context,
            "cross_class_borrowing": False,
            "cache_is_logical_tier_copy": False,
        }


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Exact source span in canonical text (code-point offsets, exclusive end)."""

    record_id: str
    start: int
    end: int
    sha256: str
    source_text_sha256: str
    text: str
    phrase: str | None = None
    exclusive_end: bool = True
    canonical_reference: bool = False
    query_reference: bool = False

    def document(self) -> dict:
        return {
            "record_id": self.record_id,
            "start": self.start,
            "end": self.end,
            "exclusive_end": self.exclusive_end,
            "sha256": self.sha256,
            "source_text_sha256": self.source_text_sha256,
            "text": self.text,
            "text_bytes": len(self.text.encode("utf-8")),
            "phrase": self.phrase,
            "canonical_reference": self.canonical_reference,
            "query_reference": self.query_reference,
        }


@dataclass(frozen=True, slots=True)
class RetrievalKeyReference:
    """Identity of the retrieval key, kept apart from ordered values.

    Retrieval keys are deterministic hash-derived selection controls.  They are
    labelled as synthetic controls and are never presented as neural memory.
    """

    record_id: str
    sha256: str
    width: int
    producer_id: str
    producer_version: str
    status: str
    producer_identity_available: bool
    used_for: str = "selection_ordering_only"
    is_synthetic_control: bool = True
    is_neural_memory: bool = False

    def document(self) -> dict:
        return {
            "record_id": self.record_id,
            "sha256": self.sha256,
            "width": self.width,
            "producer_id": self.producer_id,
            "producer_version": self.producer_version,
            "status": self.status,
            "producer_identity_available": self.producer_identity_available,
            "used_for": self.used_for,
            "is_synthetic_control": self.is_synthetic_control,
            "is_neural_memory": self.is_neural_memory,
        }


@dataclass(frozen=True, slots=True)
class ReadoutValueSet:
    """Identity of the ordered readable values returned for one record."""

    record_id: str
    slots: int
    width: int
    dtype: str
    producer_id: str
    producer_version: str
    sha256: tuple[str, ...]
    all_current: bool
    producer_identity_available: bool
    ordered_slots: bool = True
    is_hash_derived: bool = False
    is_neural_memory: bool = False

    def document(self) -> dict:
        return {
            "record_id": self.record_id,
            "slots": self.slots,
            "width": self.width,
            "dtype": self.dtype,
            "producer_id": self.producer_id,
            "producer_version": self.producer_version,
            "sha256": self.sha256,
            "all_current": self.all_current,
            "producer_identity_available": self.producer_identity_available,
            "ordered_slots": self.ordered_slots,
            "is_hash_derived": self.is_hash_derived,
            "is_neural_memory": self.is_neural_memory,
        }


@dataclass(frozen=True, slots=True)
class ScopedMemoryRecord:
    """One ordered, scoped evidence record returned by the readout."""

    rank: int
    record_id: str
    cluster_id: str
    cluster_generation: int | None
    revision: int | None
    score: float | None
    status: str
    reason: str
    retryable: bool
    placement: str
    payload_source: str
    text: str | None
    text_sha256: str
    text_bytes: int
    source_text_bytes: int
    canonical_span: SourceSpan | None
    source_spans: tuple[SourceSpan, ...]
    retrieval_key: RetrievalKeyReference
    values: tuple[tuple[float, ...], ...] | None
    value_metadata: ReadoutValueSet | None
    checks: tuple[str, ...]
    source_id: str
    source_epoch: int | None
    observed_at: int | None
    valid_to: int | None
    verification_state: str
    critical: bool
    decision_boundary: bool
    # --- canonical restoration provenance (exact store values) --------------
    available_at: int | None = None
    available_at_mapped: bool = False
    available_at_source: str = "records.observed_at"
    available_at_mapping: str = "observed_at_only"
    event_at: int | None = None
    valid_from: int | None = None
    valid_until: int | None = None
    source_lineage: tuple[str, ...] = ()
    lineage_predecessor: str | None = None
    source_provenance: Mapping[str, object] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)
    producer_revision: str | None = None
    producer_revision_available: bool = False
    representation_version: str | None = None
    representation_version_available: bool = False
    codec_version: str | None = None
    codec_version_available: bool = False
    record_representation_version: str = ""
    payload_type: str | None = None
    payload_type_available: bool = False
    supersedes: str | None = None
    supersedes_available: bool = False
    superseded_by: str | None = None
    record_hash: str = ""
    scope_key: str = ""
    kind: str = ""
    authenticity: str = ""
    importance: int | None = None
    created_at_ns: int | None = None
    updated_at_ns: int | None = None
    key_producer_id: str = ""
    key_producer_version: str = ""
    value_producer_id: str = ""
    value_producer_version: str = ""
    value_producer_consistent: bool = False
    durable_ref: Mapping[str, object] | None = None

    @property
    def ready(self) -> bool:
        return self.status == "ok"

    @property
    def value_slots(self) -> int:
        return 0 if self.values is None else len(self.values)

    def document(self) -> dict:
        return {
            "rank": self.rank,
            "record_id": self.record_id,
            "cluster_id": self.cluster_id,
            "cluster_generation": self.cluster_generation,
            "revision": self.revision,
            "score": self.score,
            "status": self.status,
            "reason": self.reason,
            "retryable": self.retryable,
            "placement": self.placement,
            "payload_source": self.payload_source,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "text_bytes": self.text_bytes,
            "source_text_bytes": self.source_text_bytes,
            "canonical_span": (None if self.canonical_span is None
                               else self.canonical_span.document()),
            "source_spans": tuple(span.document() for span in self.source_spans),
            "retrieval_key": self.retrieval_key.document(),
            "values": (None if self.values is None
                       else [list(slot) for slot in self.values]),
            "value_slots": self.value_slots,
            "value_metadata": (None if self.value_metadata is None
                               else self.value_metadata.document()),
            "values_are_producer_output": self.value_metadata is not None,
            "values_hash_derived": False,
            "checks": self.checks,
            "source_id": self.source_id,
            "source_epoch": self.source_epoch,
            "observed_at": self.observed_at,
            "valid_to": self.valid_to,
            "verification_state": self.verification_state,
            "critical": self.critical,
            "decision_boundary": self.decision_boundary,
            "canonical_text_reference": self.canonical_span is not None,
            "semantic_compression_claimed": False,
            "available_at": self.available_at,
            "available_at_mapped": self.available_at_mapped,
            "available_at_source": self.available_at_source,
            "available_at_mapping": self.available_at_mapping,
            "event_at": self.event_at,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "source_lineage": self.source_lineage,
            "lineage_predecessor": self.lineage_predecessor,
            "source_provenance": dict(self.source_provenance),
            "provenance": dict(self.provenance),
            "producer_revision": self.producer_revision,
            "producer_revision_available": self.producer_revision_available,
            "representation_version": self.representation_version,
            "representation_version_available": (
                self.representation_version_available),
            "codec_version": self.codec_version,
            "codec_version_available": self.codec_version_available,
            "record_representation_version": self.record_representation_version,
            "payload_type": self.payload_type,
            "payload_type_available": self.payload_type_available,
            "supersedes": self.supersedes,
            "supersedes_available": self.supersedes_available,
            "superseded_by": self.superseded_by,
            "record_hash": self.record_hash,
            "scope_key": self.scope_key,
            "kind": self.kind,
            "authenticity": self.authenticity,
            "importance": self.importance,
            "created_at_ns": self.created_at_ns,
            "updated_at_ns": self.updated_at_ns,
            "key_producer_id": self.key_producer_id,
            "key_producer_version": self.key_producer_version,
            "value_producer_id": self.value_producer_id,
            "value_producer_version": self.value_producer_version,
            "value_producer_consistent": self.value_producer_consistent,
            "durable_ref": (None if self.durable_ref is None
                            else dict(self.durable_ref)),
        }


@dataclass(frozen=True, slots=True)
class PendingColdReadout:
    """A cold ticket left pending because the bounded wait expired."""

    ticket_id: str
    record_id: str
    rank: int
    reason: str
    required_bytes: int
    max_bytes: int
    retryable: bool = True

    def document(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "record_id": self.record_id,
            "rank": self.rank,
            "reason": self.reason,
            "required_bytes": self.required_bytes,
            "max_bytes": self.max_bytes,
            "retryable": self.retryable,
            "resume_with": "resume_scoped_readout",
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ReadoutMetrics:
    cache_hits: int
    cache_misses: int
    cache_hit_rejected: int
    cache_inserts: int
    cache_evictions: int
    cache_skipped: int
    cache_entries: int
    cache_bytes: int
    cache_bound_bytes: int
    selection_admitted_text_bytes: int
    selection_admitted_value_slots: int
    selection_admitted_payload_bytes: int
    selection_admitted_file_reads: int
    selection_ns: int
    payload_read_calls: int
    payload_file_reads: int
    payload_io_bytes: int
    payload_ns: int
    cold_tickets: int
    cold_waits: int
    cold_resumes: int
    cold_io_bytes: int
    cold_file_reads: int
    cold_revalidation_checks: int
    cold_in_progress: int
    cold_ns: int
    revalidations: int
    revalidation_failures: int
    revalidation_ns: int
    records_returned: int
    records_ready: int
    records_degraded: int
    records_incomplete: int
    records_denied: int
    records_invalidated: int
    records_stale: int
    records_in_progress: int
    total_file_reads: int
    total_io_bytes: int
    total_ns: int
    timings_are_wall_clock: bool = True
    artificial_delay: bool = False
    larger_bank_allocated: bool = False
    gpu_resident: bool = False
    page_cache_residency_claimed: bool = False
    semantic_compression_claimed: bool = False
    cache_is_in_process_copy: bool = True
    selection_file_reads: int = 0
    selection_io_bytes: int = 0
    selection_calls: int = 0
    io_provenance: Mapping[str, str] = field(default_factory=dict)

    def document(self) -> dict:
        return {
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rejected": self.cache_hit_rejected,
                "inserts": self.cache_inserts,
                "evictions": self.cache_evictions,
                "skipped": self.cache_skipped,
                "entries": self.cache_entries,
                "bytes": self.cache_bytes,
                "bound_bytes": self.cache_bound_bytes,
                "in_process_copy": self.cache_is_in_process_copy,
            },
            "selection": {
                "admitted_text_bytes": self.selection_admitted_text_bytes,
                "admitted_value_slots": self.selection_admitted_value_slots,
                "admitted_payload_bytes": self.selection_admitted_payload_bytes,
                "admitted_file_reads": self.selection_admitted_file_reads,
                "file_reads": self.selection_file_reads,
                "io_bytes": self.selection_io_bytes,
                "calls": self.selection_calls,
                "ns": self.selection_ns,
            },
            "payload_reads": {
                "calls": self.payload_read_calls,
                "file_reads": self.payload_file_reads,
                "io_bytes": self.payload_io_bytes,
                "ns": self.payload_ns,
            },
            "cold_recall": {
                "tickets": self.cold_tickets,
                "waits": self.cold_waits,
                "resumes": self.cold_resumes,
                "io_bytes": self.cold_io_bytes,
                "file_reads": self.cold_file_reads,
                "revalidation_checks": self.cold_revalidation_checks,
                "in_progress": self.cold_in_progress,
                "ns": self.cold_ns,
            },
            "revalidation": {
                "checks": self.revalidations,
                "failures": self.revalidation_failures,
                "ns": self.revalidation_ns,
            },
            "totals": {
                "file_reads": self.total_file_reads,
                "io_bytes": self.total_io_bytes,
                "ns": self.total_ns,
            },
            "records": {
                "returned": self.records_returned,
                "ready": self.records_ready,
                "degraded": self.records_degraded,
                "incomplete": self.records_incomplete,
                "denied": self.records_denied,
                "invalidated": self.records_invalidated,
                "stale": self.records_stale,
                "in_progress": self.records_in_progress,
            },
            "timings_are_wall_clock": self.timings_are_wall_clock,
            "artificial_delay": self.artificial_delay,
            "larger_bank_allocated": self.larger_bank_allocated,
            "gpu_resident": self.gpu_resident,
            "page_cache_residency_claimed": self.page_cache_residency_claimed,
            "semantic_compression_claimed": self.semantic_compression_claimed,
            "io_provenance": dict(self.io_provenance),
        }


# ---------------------------------------------------------------------------
# Top-level result contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ScopedMemoryReadout:
    """Authoritative result of one scoped read (or one cold-ticket resume)."""

    format: str
    query_text: str
    scope: MemoryScope
    status: str
    outcome: Outcome
    new_decision_boundary: bool
    records: tuple[ScopedMemoryRecord, ...]
    metrics: ReadoutMetrics
    selection_id: str
    epoch: int
    candidate_count: int
    truncated: bool
    fallback_mode: str
    representation_stale: bool
    generation_pins: tuple[tuple[str, int], ...]
    pending_cold: tuple[PendingColdReadout, ...]
    incomplete_record_ids: tuple[str, ...]
    denied_record_ids: tuple[str, ...]
    invalidated_record_ids: tuple[str, ...]
    resumed_ticket_id: str | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in _OK_STATUSES

    @property
    def retryable(self) -> bool:
        return self.outcome.retryable

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(record.record_id for record in self.records)

    @property
    def texts(self) -> tuple[str | None, ...]:
        return tuple(record.text for record in self.records)

    @property
    def scores(self) -> tuple[float | None, ...]:
        return tuple(record.score for record in self.records)

    def record(self, record_id: str) -> ScopedMemoryRecord | None:
        for item in self.records:
            if item.record_id == record_id:
                return item
        return None

    def document(self) -> dict:
        return {
            "format": self.format,
            "query_text": self.query_text,
            "scope": self.scope.document(),
            "status": self.status,
            "outcome": self.outcome.document(),
            "new_decision_boundary": self.new_decision_boundary,
            "record_ids": self.record_ids,
            "records": tuple(record.document() for record in self.records),
            "selection_id": self.selection_id,
            "epoch": self.epoch,
            "candidate_count": self.candidate_count,
            "truncated": self.truncated,
            "fallback_mode": self.fallback_mode,
            "representation_stale": self.representation_stale,
            "generation_pins": self.generation_pins,
            "pending_cold": tuple(item.document() for item in self.pending_cold),
            "incomplete_record_ids": self.incomplete_record_ids,
            "denied_record_ids": self.denied_record_ids,
            "invalidated_record_ids": self.invalidated_record_ids,
            "resumed_ticket_id": self.resumed_ticket_id,
            "metrics": self.metrics.document(),
            "diagnostics": dict(self.diagnostics),
        }


# ---------------------------------------------------------------------------
# In-process bounded payload cache
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _CacheEntry:
    text: str
    text_sha256: str
    values: tuple[tuple[float, ...], ...]
    value_sha256: tuple[str, ...]
    bytes: int
    placement: str
    inserted_at_ns: int


class _StoreCache:
    __slots__ = ("entries", "total_bytes", "bound_bytes", "lock")

    def __init__(self) -> None:
        self.entries: OrderedDict[tuple, _CacheEntry] = OrderedDict()
        self.total_bytes = 0
        self.bound_bytes = 0
        self.lock = threading.Lock()


_CACHES: "weakref.WeakKeyDictionary[TieredMemoryStore, _StoreCache]" = (
    weakref.WeakKeyDictionary())
_CACHES_LOCK = threading.Lock()


def _store_cache(store: TieredMemoryStore) -> _StoreCache:
    with _CACHES_LOCK:
        cache = _CACHES.get(store)
        if cache is None:
            cache = _StoreCache()
            _CACHES[store] = cache
        return cache


def _cache_key(pin: Mapping[str, object]) -> tuple:
    return (
        pin["scope_key"],
        pin["record_id"],
        pin["revision"],
        pin["representation_version"],
        pin["cluster_generation"],
        pin["text_sha256"],
    )


def _cache_get(cache: _StoreCache, key: tuple) -> _CacheEntry | None:
    with cache.lock:
        entry = cache.entries.get(key)
        if entry is not None:
            cache.entries.move_to_end(key)
        return entry


def _cache_drop(cache: _StoreCache, key: tuple) -> None:
    with cache.lock:
        entry = cache.entries.pop(key, None)
        if entry is not None:
            cache.total_bytes -= entry.bytes


def _cache_trim(cache: _StoreCache, bound: int, stats: "_Stats") -> None:
    with cache.lock:
        cache.bound_bytes = bound
        while cache.total_bytes > bound and cache.entries:
            _, evicted = cache.entries.popitem(last=False)
            cache.total_bytes -= evicted.bytes
            stats.cache_evictions += 1


def _cache_put(cache: _StoreCache, key: tuple, entry: _CacheEntry,
               bound: int, stats: "_Stats") -> None:
    if entry.bytes > bound:
        stats.cache_skipped += 1
        return
    with cache.lock:
        cache.bound_bytes = bound
        existing = cache.entries.pop(key, None)
        if existing is not None:
            cache.total_bytes -= existing.bytes
        cache.entries[key] = entry
        cache.total_bytes += entry.bytes
        stats.cache_inserts += 1
        while cache.total_bytes > bound and cache.entries:
            _, evicted = cache.entries.popitem(last=False)
            cache.total_bytes -= evicted.bytes
            stats.cache_evictions += 1


def readout_cache_state(store: TieredMemoryStore) -> dict:
    """Inspect the in-process cache without reading payloads."""
    if not isinstance(store, TieredMemoryStore):
        raise TypeError("store must be a TieredMemoryStore")
    cache = _store_cache(store)
    with cache.lock:
        return {
            "entries": len(cache.entries),
            "bytes": cache.total_bytes,
            "bound_bytes": cache.bound_bytes,
            "record_ids": tuple(sorted({key[1] for key in cache.entries})),
        }


def clear_readout_cache(store: TieredMemoryStore | None = None) -> None:
    """Drop cached payload copies (all stores when ``store`` is omitted)."""
    if store is None:
        with _CACHES_LOCK:
            stores = list(_CACHES.keys())
        for item in stores:
            _clear_one(item)
        return
    if not isinstance(store, TieredMemoryStore):
        raise TypeError("store must be a TieredMemoryStore")
    _clear_one(store)


def _clear_one(store: TieredMemoryStore) -> None:
    cache = _store_cache(store)
    with cache.lock:
        cache.entries.clear()
        cache.total_bytes = 0


# ---------------------------------------------------------------------------
# Live index inspection and revalidation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _Live:
    record: Mapping[str, object]
    key: Mapping[str, object] | None
    values: tuple[Mapping[str, object], ...]
    generation: int | None


@dataclass(frozen=True, slots=True)
class _Verdict:
    status: str
    reason: str
    checks: tuple[str, ...]
    decision_boundary: bool

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _load_live(store: TieredMemoryStore, record_id: str) -> _Live | None:
    connection = store.connection
    row = connection.execute(
        "SELECT * FROM records WHERE record_id=?", (record_id,)).fetchone()
    if row is None:
        return None
    key = connection.execute(
        "SELECT * FROM record_keys WHERE record_id=?", (record_id,)).fetchone()
    values = connection.execute(
        "SELECT * FROM record_values WHERE record_id=? ORDER BY slot_index",
        (record_id,)).fetchall()
    cluster = connection.execute(
        "SELECT current_generation FROM clusters WHERE cluster_id=?",
        (row["cluster_id"],)).fetchone()
    return _Live(
        record={name: row[name] for name in row.keys()},
        key=None if key is None else {name: key[name] for name in key.keys()},
        values=tuple({name: value[name] for name in value.keys()}
                     for value in values),
        generation=None if cluster is None else int(cluster["current_generation"]),
    )


def _pin_from_selected(scope_key: str, selected: object) -> dict:
    values = selected.values if selected.values_available else ()
    payloads = pack_values(values) if values else ()
    return {
        "record_id": selected.record_id,
        "scope_key": scope_key,
        "revision": selected.revision,
        "representation_version": selected.representation_version,
        "cluster_generation": selected.cluster_generation,
        "cluster_id": selected.cluster_id,
        "text_sha256": selected.text_sha256,
        "text_bytes": selected.text_bytes,
        "key_sha256": selected.key_sha256,
        "value_slots": len(payloads),
        "value_sha256": tuple(_sha256_bytes(item) for item in payloads),
        "placement": selected.placement,
    }


def _pin_from_live(scope_key: str, live: _Live) -> dict:
    row = live.record
    key = live.key
    return {
        "record_id": str(row["record_id"]),
        "scope_key": scope_key,
        "revision": int(row["revision"]),
        "representation_version": str(row["representation_version"]),
        "cluster_generation": live.generation,
        "cluster_id": str(row["cluster_id"]),
        "text_sha256": str(row["text_sha256"]),
        "text_bytes": int(row["text_bytes"]),
        "key_sha256": "" if key is None else str(key["key_sha256"]),
        "value_slots": len(live.values),
        "value_sha256": tuple(str(value["value_sha256"])
                              for value in live.values),
        "placement": str(row["placement"]),
    }


def _pin_from_ticket(ticket_pin: Mapping[str, object], scope_key: str) -> dict:
    values = tuple(ticket_pin.get("values", ()))
    return {
        "record_id": str(ticket_pin["record_id"]),
        "scope_key": scope_key,
        "revision": int(ticket_pin["revision"]),
        "representation_version": str(ticket_pin["representation_version"]),
        "cluster_generation": int(ticket_pin["cluster_generation"]),
        "cluster_id": str(ticket_pin["cluster_id"]),
        "text_sha256": str(ticket_pin["text_sha256"]),
        "text_bytes": int(ticket_pin["text_bytes"]),
        "key_sha256": str(ticket_pin.get("key_sha256", "")),
        "value_slots": len(values),
        "value_sha256": tuple(str(value["sha256"]) for value in values),
        "placement": str(ticket_pin.get("placement", "cold")),
    }


def _payload_value_sha(values: Sequence[Sequence[float]]) -> tuple[str, ...]:
    return tuple(_sha256_bytes(payload) for payload in pack_values(values))


def _revalidate(store: TieredMemoryStore, pin: Mapping[str, object],
                scope_key: str, live: _Live | None, *,
                payload_text: str | None = None,
                payload_values: Sequence[Sequence[float]] | None = None,
                payload_value_sha: Sequence[str] | None = None,
                scope: MemoryScope | None = None) -> _Verdict:
    """Compare a pre-read pin, post-read index state and the held payload."""
    checks: list[str] = []
    record_id = str(pin["record_id"])
    if live is None:
        return _Verdict("denied", "record_missing_after_io", tuple(checks), False)
    row = live.record
    if str(row["scope_key"]) != scope_key or str(row["scope_key"]) != str(
            pin["scope_key"]):
        return _Verdict("denied", "cross_scope_after_io", tuple(checks), False)
    checks.append(f"scope:{record_id}")
    if str(row["status"]) != "current":
        return _Verdict("invalidated",
                        "record_revoked_or_superseded_during_readout",
                        tuple(checks), True)
    checks.append(f"status:{record_id}")
    if str(row["verification_state"]) not in {
            "unverified", "verified_source", "observed_tool", "verified_human"}:
        return _Verdict("invalidated", "verification_state_changed_during_readout",
                        tuple(checks), True)
    checks.append(f"verification:{record_id}")
    if scope is not None:
        cutoff = scope.cutoff
        valid_from = row.get("valid_from")
        valid_to = row.get("valid_to")
        if (int(row["observed_at"]) > cutoff
                or (valid_from is not None and int(valid_from) > cutoff)
                or (valid_to is not None and int(valid_to) <= cutoff)):
            return _Verdict("invalidated", "record_outside_time_scope_after_io",
                            tuple(checks), True)
        checks.append(f"time_scope:{record_id}")
    if int(row["revision"]) != int(pin["revision"]):
        return _Verdict("invalidated", "record_revision_changed_during_readout",
                        tuple(checks), True)
    checks.append(f"revision:{record_id}")
    if (live.generation is not None
            and pin.get("cluster_generation") is not None
            and int(live.generation) != int(pin["cluster_generation"])):
        return _Verdict("stale", "cluster_generation_changed_during_readout",
                        tuple(checks), True)
    checks.append(f"generation:{record_id}")
    if str(row["representation_version"]) != str(
            pin["representation_version"]):
        return _Verdict("stale",
                        "representation_version_changed_during_readout",
                        tuple(checks), True)
    checks.append(f"representation:{record_id}")
    if str(row["text_sha256"]) != str(pin["text_sha256"]):
        return _Verdict("stale", "canonical_text_changed_during_readout",
                        tuple(checks), True)
    pinned_key_sha = str(pin["key_sha256"])
    if pinned_key_sha:
        key = live.key
        if (key is None or str(key["status"]) != "current"
                or str(key["key_sha256"]) != pinned_key_sha):
            return _Verdict("stale", "retrieval_key_stale_during_readout",
                            tuple(checks), True)
        checks.append(f"key:{record_id}")
    pinned_slots = int(pin["value_slots"])
    if pinned_slots:
        values = live.values
        if len(values) != pinned_slots:
            return _Verdict("stale",
                            "value_representation_changed_during_readout",
                            tuple(checks), True)
        pinned_sha = tuple(str(item) for item in pin["value_sha256"])
        for offset, (value, sha) in enumerate(zip(values, pinned_sha)):
            if (str(value["status"]) != "current"
                    or str(value["value_sha256"]) != sha
                    or int(value["slot_index"]) != offset):
                return _Verdict("stale",
                                "value_representation_stale_during_readout",
                                tuple(checks), True)
        checks.append(f"values:{record_id}")
    if payload_text is not None:
        if _sha256_text(payload_text) != str(pin["text_sha256"]):
            return _Verdict("failed", "payload_digest_mismatch",
                            tuple(checks), False)
        checks.append(f"text_payload:{record_id}")
    if payload_values is not None:
        held_sha = (tuple(payload_value_sha) if payload_value_sha is not None
                    else _payload_value_sha(payload_values))
        pinned_sha = tuple(str(item) for item in pin["value_sha256"])
        if pinned_sha and (len(held_sha) != len(pinned_sha)
                           or any(left != right for left, right
                                  in zip(held_sha, pinned_sha))):
            return _Verdict("failed", "payload_digest_mismatch",
                            tuple(checks), False)
        checks.append(f"values_payload:{record_id}")
    return _Verdict("ok", "post_io_revalidation_passed", tuple(checks), False)


# ---------------------------------------------------------------------------
# Mutable per-call counters
# ---------------------------------------------------------------------------

@dataclass
class _Stats:
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rejected: int = 0
    cache_inserts: int = 0
    cache_evictions: int = 0
    cache_skipped: int = 0
    selection_admitted_text_bytes: int = 0
    selection_admitted_value_slots: int = 0
    selection_admitted_payload_bytes: int = 0
    selection_admitted_file_reads: int = 0
    selection_file_reads: int = 0
    selection_io_bytes: int = 0
    selection_calls: int = 0
    selection_ns: int = 0
    payload_read_calls: int = 0
    payload_file_reads: int = 0
    payload_io_bytes: int = 0
    payload_ns: int = 0
    cold_tickets: int = 0
    cold_waits: int = 0
    cold_resumes: int = 0
    cold_io_bytes: int = 0
    cold_file_reads: int = 0
    cold_revalidation_checks: int = 0
    cold_in_progress: int = 0
    cold_ns: int = 0
    revalidations: int = 0
    revalidation_failures: int = 0
    revalidation_ns: int = 0


# ---------------------------------------------------------------------------
# Acquisition results
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _Acquired:
    status: str
    reason: str
    retryable: bool
    boundary: bool
    text: str | None
    values: tuple[tuple[float, ...], ...] | None
    text_sha256: str
    value_sha256: tuple[str, ...]
    payload_source: str
    checks: tuple[str, ...]
    io_bytes: int
    file_reads: int
    pending: PendingColdReadout | None = None


def _acquire_direct(store: TieredMemoryStore, record_id: str,
                    scope: MemoryScope, budget: ReadoutBudget,
                    stats: _Stats) -> _Acquired:
    started = _perf_ns()
    result = store.read_record(
        record_id, scope=scope, budget=budget.retrieval_budget())
    stats.payload_read_calls += 1
    stats.payload_io_bytes += int(result.io_bytes)
    stats.payload_file_reads += int(result.file_reads)
    acquired: _Acquired
    if result.status == "ready" and result.records:
        recalled = result.records[0]
        acquired = _Acquired(
            status="ok", reason="direct_payload_ready", retryable=False,
            boundary=bool(result.new_decision_boundary),
            text=recalled.text, values=recalled.values,
            text_sha256=recalled.text_sha256,
            value_sha256=tuple(recalled.value_sha256),
            payload_source="read_record", checks=tuple(result.checks),
            io_bytes=int(result.io_bytes),
            file_reads=int(result.file_reads))
    else:
        acquired = _Acquired(
            status=result.status, reason=result.outcome.reason,
            retryable=bool(result.outcome.retryable),
            boundary=bool(result.new_decision_boundary),
            text=None, values=None, text_sha256="", value_sha256=(),
            payload_source="read_record", checks=tuple(result.checks),
            io_bytes=int(result.io_bytes), file_reads=int(result.file_reads))
    stats.payload_ns += _perf_ns() - started
    return acquired


def _acquire_cold(store: TieredMemoryStore, selection_id: str,
                  selected: object, scope: MemoryScope,
                  budget: ReadoutBudget, stats: _Stats,
                  rank: int) -> _Acquired:
    started = _perf_ns()
    ticket = store.begin_recall(
        selection_id, record_ids=(selected.record_id,),
        budget=budget.retrieval_budget())
    stats.cold_tickets += 1
    if ticket.status == "unsupported":
        stats.cold_ns += _perf_ns() - started
        return _Acquired(
            status="unsupported", reason=ticket.outcome.reason,
            retryable=False, boundary=False, text=None, values=None,
            text_sha256="", value_sha256=(), payload_source="cold_recall",
            checks=(), io_bytes=0, file_reads=0)
    waited = store.wait_recall(ticket.ticket_id, budget.cold_wait_seconds)
    stats.cold_waits += 1
    if waited.status == "in_progress":
        stats.cold_in_progress += 1
        stats.cold_ns += _perf_ns() - started
        pending = PendingColdReadout(
            ticket_id=ticket.ticket_id, record_id=selected.record_id,
            rank=rank, reason=waited.outcome.reason,
            required_bytes=int(ticket.required_bytes),
            max_bytes=int(ticket.max_bytes))
        return _Acquired(
            status="in_progress", reason="cold_recall_wait_in_progress",
            retryable=True, boundary=False, text=None, values=None,
            text_sha256="", value_sha256=(), payload_source="cold_recall",
            checks=(), io_bytes=0, file_reads=0, pending=pending)
    stats.cold_resumes += 1
    stats.cold_io_bytes += int(waited.io_bytes)
    stats.cold_revalidation_checks += len(waited.checks)
    stats.cold_file_reads += int(waited.file_reads)
    if waited.status == "ready" and waited.records:
        recalled = waited.records[0]
        acquired = _Acquired(
            status="ok", reason="cold_payload_ready", retryable=False,
            boundary=bool(waited.new_decision_boundary),
            text=recalled.text, values=recalled.values,
            text_sha256=recalled.text_sha256,
            value_sha256=tuple(recalled.value_sha256),
            payload_source="cold_recall", checks=tuple(waited.checks),
            io_bytes=int(waited.io_bytes),
            file_reads=int(waited.file_reads))
    else:
        acquired = _Acquired(
            status=waited.status, reason=waited.outcome.reason,
            retryable=bool(waited.outcome.retryable)
            or waited.status == "incomplete",
            boundary=bool(waited.new_decision_boundary),
            text=None, values=None, text_sha256="", value_sha256=(),
            payload_source="cold_recall", checks=tuple(waited.checks),
            io_bytes=int(waited.io_bytes), file_reads=int(waited.file_reads))
    stats.cold_ns += _perf_ns() - started
    return acquired


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

def _retrieval_key_reference(pin: Mapping[str, object],
                             live: _Live | None) -> RetrievalKeyReference:
    key = live.key if live is not None else None
    record_id = str(pin["record_id"])
    if key is not None:
        return RetrievalKeyReference(
            record_id=record_id,
            sha256=str(key["key_sha256"]),
            width=int(key["key_width"]),
            producer_id=str(key["producer_id"]),
            producer_version=str(key["producer_version"]),
            status=str(key["status"]),
            producer_identity_available=bool(str(key["producer_id"])),
        )
    return RetrievalKeyReference(
        record_id=record_id,
        sha256=str(pin.get("key_sha256", "")),
        width=KEY_WIDTH,
        producer_id="",
        producer_version="",
        status="missing",
        producer_identity_available=False,
    )


def _value_metadata(pin: Mapping[str, object], live: _Live | None,
                    values: Sequence[Sequence[float]] | None
                    ) -> ReadoutValueSet | None:
    if values is None or len(values) == 0:
        return None
    rows = live.values if live is not None else ()
    if rows:
        width = int(rows[0]["value_width"])
        dtype = str(rows[0]["dtype"])
        producer_id = str(rows[0]["producer_id"])
        producer_version = str(rows[0]["producer_version"])
        sha = tuple(str(row["value_sha256"]) for row in rows)
        all_current = all(str(row["status"]) == "current" for row in rows)
        available = bool(producer_id)
    else:
        width = len(values[0]) if values else 0
        dtype = "unknown"
        producer_id = ""
        producer_version = ""
        sha = _payload_value_sha(values)
        all_current = False
        available = False
    return ReadoutValueSet(
        record_id=str(pin["record_id"]),
        slots=len(values), width=width, dtype=dtype,
        producer_id=producer_id, producer_version=producer_version,
        sha256=sha, all_current=all_current,
        producer_identity_available=available)


def _spans_for(record_id: str, text: str | None, text_sha256: str,
               query_text: str, selection_spans: Sequence[object]
               ) -> tuple[SourceSpan | None, tuple[SourceSpan, ...]]:
    canonical: SourceSpan | None = None
    extra: list[SourceSpan] = []
    if text is not None:
        canonical = SourceSpan(
            record_id=record_id, start=0, end=len(text),
            sha256=_sha256_text(text), source_text_sha256=text_sha256,
            text=text, canonical_reference=True)
        if query_text:
            start = text.find(query_text)
            if start >= 0:
                end = start + len(query_text)
                extra.append(SourceSpan(
                    record_id=record_id, start=start, end=end,
                    sha256=_sha256_text(text[start:end]),
                    source_text_sha256=text_sha256,
                    text=text[start:end], phrase=query_text,
                    query_reference=True))
    for hit in selection_spans:
        extra.append(SourceSpan(
            record_id=hit.record_id, start=hit.start, end=hit.end,
            sha256=hit.start_sha256, source_text_sha256=hit.source_hash,
            text=hit.text, phrase=hit.phrase))
    return canonical, tuple(extra)


def _metadata_value(row: Mapping[str, object], name: str,
                    fallback: object) -> object:
    value = row.get(name, fallback)
    return fallback if value is None else value


_PAYLOAD_TYPES = frozenset({
    "observation", "interpretation", "prediction", "verified_feedback"})


def _string_or_none(mapping: Mapping[str, object], name: str) -> str | None:
    value = mapping.get(name)
    if isinstance(value, str) and value:
        return value
    return None


def _lineage_and_provenance(row: Mapping[str, object]
                            ) -> tuple[tuple[str, ...],
                                       Mapping[str, object] | None, bool]:
    """Decode only the lineage object belonging to the current record."""
    try:
        items = json.loads(str(row.get("source_lineage", "[]")))
    except (TypeError, ValueError):
        return (), None, False
    if not isinstance(items, list):
        return (), None, False
    lineage = tuple(item for item in items if isinstance(item, str))
    if len(lineage) != len(items):
        return lineage, None, False
    record_id = str(row.get("record_id", ""))
    for item in reversed(lineage):
        try:
            candidate = json.loads(item)
        except (TypeError, ValueError):
            continue
        if isinstance(candidate, dict) and candidate.get("record_id") == record_id:
            return lineage, dict(candidate), True
    return lineage, None, False


def _manifest_reference(store: TieredMemoryStore, live: _Live
                        ) -> dict | None:
    if live.generation is None:
        return None
    row = store.connection.execute(
        "SELECT manifest_path, manifest_sha256 FROM cluster_generations "
        "WHERE cluster_id=? AND generation=?",
        (str(live.record["cluster_id"]), int(live.generation))).fetchone()
    if row is None:
        return None
    relative = str(row["manifest_path"])
    return {
        "relative_path": relative,
        "path": str(store.root / relative),
        "sha256": str(row["manifest_sha256"]),
        "generation": int(live.generation),
    }


def _durable_reference(store: TieredMemoryStore, live: _Live,
                       recorded: Mapping[str, object] | None
                       ) -> Mapping[str, object]:
    """Exact file/digest reference for rebuilding this record without guessing."""
    row = live.record
    text_relative = str(row["text_path"])

    def absolute(relative: object) -> str | None:
        if not isinstance(relative, str) or not relative:
            return None
        return str(store.root / relative)

    key = live.key
    key_ref = None if key is None else {
        "relative_path": str(key["key_path"]),
        "path": absolute(key["key_path"]),
        "sha256": str(key["key_sha256"]),
        "bytes": int(key["key_bytes"]),
        "width": int(key["key_width"]),
        "producer_id": str(key["producer_id"]),
        "producer_version": str(key["producer_version"]),
        "status": str(key["status"]),
    }
    value_refs = tuple({
        "slot": int(value["slot_index"]),
        "relative_path": str(value["value_path"]),
        "path": absolute(value["value_path"]),
        "sha256": str(value["value_sha256"]),
        "bytes": int(value["value_bytes"]),
        "width": int(value["value_width"]),
        "dtype": str(value["dtype"]),
        "producer_id": str(value["producer_id"]),
        "producer_version": str(value["producer_version"]),
        "status": str(value["status"]),
    } for value in live.values)
    return {
        "locator": str(store.root / text_relative),
        "relative_locator": text_relative,
        "text_sha256": str(row["text_sha256"]),
        "store_root": str(store.root),
        "record_id": str(row["record_id"]),
        "scope_key": str(row["scope_key"]),
        "cluster_id": str(row["cluster_id"]),
        "cluster_generation": live.generation,
        "record_hash": str(row["record_hash"]),
        "text": {
            "relative_path": text_relative,
            "path": str(store.root / text_relative),
            "sha256": str(row["text_sha256"]),
            "bytes": int(row["text_bytes"]),
        },
        "key": key_ref,
        "values": value_refs,
        "cluster_manifest": _manifest_reference(store, live),
        "recorded_durable_ref": None if recorded is None else dict(recorded),
    }


def _restore_fields(store: TieredMemoryStore | None, live: _Live | None
                    ) -> dict:
    """Exact restoration metadata taken from the store row, never inferred."""
    if store is None or live is None:
        return {}
    row = live.record
    lineage, parsed, parsed_ok = _lineage_and_provenance(row)
    provenance_map: Mapping[str, object] = parsed if parsed is not None else {}
    observed_at = int(row["observed_at"])
    available_mapped = bool(parsed_ok)
    available_mapping = (
        "persistent_learner_runner.available_at_to_observed_at"
        if available_mapped else "observed_at_only")
    available_at = observed_at if available_mapped else None
    producer_revision = _string_or_none(provenance_map, "producer_revision")
    representation_version = _string_or_none(
        provenance_map, "representation_version")
    codec_version = _string_or_none(provenance_map, "codec_version")
    raw_payload_type = provenance_map.get("payload_type")
    payload_type = (raw_payload_type
                    if isinstance(raw_payload_type, str)
                    and raw_payload_type in _PAYLOAD_TYPES else None)
    raw_supersedes = provenance_map.get("supersedes")
    supersedes = (raw_supersedes
                  if isinstance(raw_supersedes, str) and raw_supersedes
                  else None)
    if str(row["kind"]) == "correction" and lineage:
        predecessor = lineage[-1]
        if predecessor != str(row["record_id"]):
            supersedes = predecessor
    recorded_durable = provenance_map.get("durable_ref")
    if not isinstance(recorded_durable, Mapping):
        recorded_durable = None
    key = live.key
    value_producers = {
        (str(value["producer_id"]), str(value["producer_version"]))
        for value in live.values}
    value_producer_consistent = bool(live.values) and len(value_producers) == 1
    if value_producer_consistent:
        value_producer_id, value_producer_version = next(iter(value_producers))
    else:
        value_producer_id = value_producer_version = ""
    valid_from = (None if row.get("valid_from") is None
                  else int(row["valid_from"]))
    valid_until = (None if row.get("valid_to") is None
                   else int(row["valid_to"]))
    durable = _durable_reference(store, live, recorded_durable)
    provenance = {
        "record_id": str(row["record_id"]),
        "scope_key": str(row["scope_key"]),
        "cluster_id": str(row["cluster_id"]),
        "cluster_generation": live.generation,
        "record_hash": str(row["record_hash"]),
        "kind": str(row["kind"]),
        "status": str(row["status"]),
        "source_id": str(row["source_id"]),
        "source_epoch": int(row["source_epoch"]),
        "source_lineage": list(lineage),
        "lineage_predecessor": (lineage[-1] if len(lineage) > 1 else None),
        "source_provenance": (dict(parsed) if parsed is not None else None),
        "authenticity": str(row["authenticity"]),
        "verification_state": str(row["verification_state"]),
        "importance": int(row["importance"]),
        "critical": bool(row["critical"]),
        "event_at": int(row["event_at"]),
        "observed_at": observed_at,
        "available_at": available_at,
        "available_at_mapping": available_mapping,
        "valid_from": valid_from,
        "valid_until": valid_until,
        "created_at_ns": int(row["created_at_ns"]),
        "updated_at_ns": int(row["updated_at_ns"]),
        "superseded_by": row.get("superseded_by"),
        "text_sha256": str(row["text_sha256"]),
        "text_bytes": int(row["text_bytes"]),
        "placement": str(row["placement"]),
        "producer_revision": producer_revision,
        "representation_version": representation_version,
        "codec_version": codec_version,
        "payload_type": payload_type,
        "supersedes": supersedes,
        "supersedes_available": supersedes is not None,
    }
    return {
        "available_at": available_at,
        "available_at_mapped": available_mapped,
        "available_at_source": ("records.observed_at" if available_mapped
                                else "not_recorded"),
        "available_at_mapping": available_mapping,
        "event_at": int(row["event_at"]),
        "valid_from": valid_from,
        "valid_until": valid_until,
        "source_lineage": lineage,
        "lineage_predecessor": (lineage[-1] if len(lineage) > 1 else None),
        "source_provenance": dict(parsed) if parsed is not None else {},
        "provenance": provenance,
        "producer_revision": producer_revision,
        "producer_revision_available": producer_revision is not None,
        "representation_version": representation_version,
        "representation_version_available": representation_version is not None,
        "codec_version": codec_version,
        "codec_version_available": codec_version is not None,
        "record_representation_version": str(row["representation_version"]),
        "payload_type": payload_type,
        "payload_type_available": payload_type is not None,
        "supersedes": supersedes,
        "supersedes_available": supersedes is not None,
        "superseded_by": row.get("superseded_by"),
        "record_hash": str(row["record_hash"]),
        "scope_key": str(row["scope_key"]),
        "kind": str(row["kind"]),
        "authenticity": str(row["authenticity"]),
        "importance": int(row["importance"]),
        "created_at_ns": int(row["created_at_ns"]),
        "updated_at_ns": int(row["updated_at_ns"]),
        "key_producer_id": "" if key is None else str(key["producer_id"]),
        "key_producer_version": ("" if key is None
                                 else str(key["producer_version"])),
        "value_producer_id": value_producer_id,
        "value_producer_version": value_producer_version,
        "value_producer_consistent": value_producer_consistent,
        "durable_ref": durable,
    }


def _build_record(*, rank: int, pin: Mapping[str, object], live: _Live | None,
                  selected: object | None, status: str, reason: str,
                  retryable: bool, payload_source: str, text: str | None,
                  values: tuple[tuple[float, ...], ...] | None,
                  text_sha256: str, checks: Sequence[str], query_text: str,
                  selection_spans: Sequence[object],
                  decision_boundary: bool,
                  store: TieredMemoryStore | None = None
                  ) -> ScopedMemoryRecord:
    row = live.record if live is not None else {}
    record_id = str(pin["record_id"])
    canonical, extra = _spans_for(
        record_id, text, text_sha256 or str(pin["text_sha256"]),
        query_text, selection_spans)
    fallback_source_id = getattr(selected, "source_id", "")
    fallback_epoch = getattr(selected, "source_epoch", None)
    fallback_observed = getattr(selected, "observed_at", None)
    fallback_valid_to = getattr(selected, "valid_to", None)
    fallback_verification = getattr(selected, "verification_state", "unverified")
    fallback_critical = getattr(selected, "critical", False)
    score = None if selected is None else float(selected.score)
    return ScopedMemoryRecord(
        rank=rank,
        record_id=record_id,
        cluster_id=str(_metadata_value(row, "cluster_id",
                                       pin["cluster_id"])),
        cluster_generation=pin["cluster_generation"],
        revision=int(_metadata_value(row, "revision", pin["revision"])),
        score=score,
        status=status,
        reason=reason,
        retryable=retryable,
        placement=str(pin.get("placement", "unknown")),
        payload_source=payload_source,
        text=text,
        text_sha256=(text_sha256 or str(pin["text_sha256"])),
        text_bytes=(0 if text is None else len(text.encode("utf-8"))),
        source_text_bytes=int(pin.get("text_bytes", 0)),
        canonical_span=canonical,
        source_spans=extra,
        retrieval_key=_retrieval_key_reference(pin, live),
        values=values,
        value_metadata=_value_metadata(pin, live, values),
        checks=tuple(checks),
        source_id=str(_metadata_value(row, "source_id", fallback_source_id)),
        source_epoch=(_metadata_value(row, "source_epoch", fallback_epoch)
                      if row or fallback_epoch is not None else None),
        observed_at=(_metadata_value(row, "observed_at", fallback_observed)
                     if row or fallback_observed is not None else None),
        valid_to=(row.get("valid_to") if row else fallback_valid_to),
        verification_state=str(_metadata_value(
            row, "verification_state", fallback_verification)),
        critical=bool(_metadata_value(row, "critical", fallback_critical)),
        decision_boundary=decision_boundary,
        **_restore_fields(store, live),
    )


# ---------------------------------------------------------------------------
# Presentation budgets
# ---------------------------------------------------------------------------

def _present(text: str | None,
             values: tuple[tuple[float, ...], ...] | None,
             text_available: int, slots_available: int
             ) -> tuple[str | None, tuple[tuple[float, ...], ...] | None,
                        int, int, str, str, bool]:
    """Apply independent text/slot budgets; return possibly reduced payload."""
    chosen_text: str | None = None
    used_text = 0
    chosen_values: tuple[tuple[float, ...], ...] | None = None
    used_slots = 0
    status = _READY
    reason = "readout_ready"
    retryable = False
    if values is not None and len(values) == 0:
        values = None
    if text is not None:
        size = len(text.encode("utf-8"))
        if size <= text_available:
            chosen_text = text
            used_text = size
        else:
            status = "incomplete"
            reason = "canonical_text_budget_exhausted"
            retryable = True
    if values is not None:
        count = len(values)
        if count <= slots_available:
            chosen_values = values
            used_slots = count
        else:
            status = "incomplete"
            reason = "value_slot_budget_exhausted"
            retryable = True
    return (chosen_text, chosen_values, used_text, used_slots,
            status, reason, retryable)


def _record_presented(*, store: TieredMemoryStore,
                      rank: int, pin: Mapping[str, object],
                      live: _Live | None, selected: object | None,
                      acquired: _Acquired, text_available: int,
                      slots_available: int, query_text: str,
                      selection_spans: Sequence[object],
                      expect_values: bool = False
                      ) -> tuple[ScopedMemoryRecord, int, int]:
    if acquired.status not in _OK_STATUSES:
        record = _build_record(
            store=store,
            rank=rank, pin=pin, live=live, selected=selected,
            status=acquired.status, reason=acquired.reason,
            retryable=acquired.retryable,
            payload_source=acquired.payload_source,
            text=None, values=None, text_sha256=acquired.text_sha256,
            checks=acquired.checks, query_text=query_text,
            selection_spans=selection_spans,
            decision_boundary=acquired.boundary)
        return record, 0, 0
    text, values, used_text, used_slots, status, reason, retryable = (
        _present(acquired.text, acquired.values, text_available,
                 slots_available))
    if acquired.status == "degraded" and status == _READY:
        status, reason = "degraded", acquired.reason
    if status == _READY and text is None and int(pin.get("text_bytes", 0)) > 0:
        status = "incomplete"
        reason = ("canonical_text_budget_exhausted" if text_available <= 0
                  else "canonical_text_not_admitted")
        retryable = True
    if status == _READY and values is None and expect_values and slots_available <= 0:
        status = "incomplete"
        reason = "value_slot_budget_exhausted"
        retryable = True
    record = _build_record(
        store=store,
        rank=rank, pin=pin, live=live, selected=selected,
        status=status, reason=reason, retryable=retryable,
        payload_source=acquired.payload_source, text=text, values=values,
        text_sha256=acquired.text_sha256, checks=acquired.checks,
        query_text=query_text, selection_spans=selection_spans,
        decision_boundary=acquired.boundary)
    return record, used_text, used_slots


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(records: Sequence[ScopedMemoryRecord],
               selection_status: str, selection_reason: str,
               truncated: bool, representation_stale: bool,
               pending: Sequence[PendingColdReadout]) -> tuple[str, str, bool]:
    if not records:
        status = selection_status if selection_status else "incomplete"
        reason = selection_reason or "no_records_selected"
        return status, reason, False
    statuses = [record.status for record in records]
    if all(status in _OK_STATUSES for status in statuses):
        if any(status == "degraded" for status in statuses):
            return "degraded", "readout_degraded", False
        if truncated:
            return "degraded", "selection_truncated", False
        if representation_stale:
            return "degraded", "some_representations_stale", False
        return "ok", "readout_ready", False
    ready = sum(1 for status in statuses if status in _OK_STATUSES)
    failures = [status for status in statuses if status not in _OK_STATUSES]
    if ready:
        return "incomplete", "partial_scoped_readout", True
    for candidate in _FAILURE_PRECEDENCE:
        if candidate in failures:
            reason = next((record.reason for record in records
                           if record.status == candidate),
                          f"readout_{candidate}")
            return candidate, reason, candidate in ("incomplete", "in_progress")
    return "failed", "readout_failed", False


def _finish_metrics(stats: _Stats, cache: _StoreCache,
                    records: Sequence[ScopedMemoryRecord],
                    total_ns: int) -> ReadoutMetrics:
    with cache.lock:
        entries = len(cache.entries)
        cache_bytes = cache.total_bytes
        cache_bound = cache.bound_bytes
    status_counts = {
        name: sum(1 for record in records if record.status == name)
        for name in ("ok", "degraded", "incomplete", "denied",
                     "invalidated", "stale", "in_progress")
    }
    total_file_reads = (stats.selection_file_reads
                        + stats.payload_file_reads + stats.cold_file_reads)
    total_io_bytes = (stats.selection_io_bytes
                      + stats.payload_io_bytes + stats.cold_io_bytes)
    return ReadoutMetrics(
        cache_hits=stats.cache_hits,
        cache_misses=stats.cache_misses,
        cache_hit_rejected=stats.cache_hit_rejected,
        cache_inserts=stats.cache_inserts,
        cache_evictions=stats.cache_evictions,
        cache_skipped=stats.cache_skipped,
        cache_entries=entries,
        cache_bytes=cache_bytes,
        cache_bound_bytes=cache_bound,
        selection_admitted_text_bytes=stats.selection_admitted_text_bytes,
        selection_admitted_value_slots=stats.selection_admitted_value_slots,
        selection_admitted_payload_bytes=stats.selection_admitted_payload_bytes,
        selection_admitted_file_reads=stats.selection_admitted_file_reads,
        selection_file_reads=stats.selection_file_reads,
        selection_io_bytes=stats.selection_io_bytes,
        selection_calls=stats.selection_calls,
        selection_ns=stats.selection_ns,
        payload_read_calls=stats.payload_read_calls,
        payload_file_reads=stats.payload_file_reads,
        payload_io_bytes=stats.payload_io_bytes,
        payload_ns=stats.payload_ns,
        cold_tickets=stats.cold_tickets,
        cold_waits=stats.cold_waits,
        cold_resumes=stats.cold_resumes,
        cold_io_bytes=stats.cold_io_bytes,
        cold_file_reads=stats.cold_file_reads,
        cold_revalidation_checks=stats.cold_revalidation_checks,
        cold_in_progress=stats.cold_in_progress,
        cold_ns=stats.cold_ns,
        revalidations=stats.revalidations,
        revalidation_failures=stats.revalidation_failures,
        revalidation_ns=stats.revalidation_ns,
        records_returned=len(records),
        records_ready=status_counts["ok"],
        records_degraded=status_counts["degraded"],
        records_incomplete=status_counts["incomplete"],
        records_denied=status_counts["denied"],
        records_invalidated=status_counts["invalidated"],
        records_stale=status_counts["stale"],
        records_in_progress=status_counts["in_progress"],
        total_file_reads=total_file_reads,
        total_io_bytes=total_io_bytes,
        total_ns=total_ns,
        io_provenance={
            "selection": (
                "exact verified-read accounting from TieredMemoryStore.select: "
                "candidate keys, admitted payloads and exact fallback"),
            "payload_reads": (
                "exact RecallResult.io_bytes and file_reads from "
                "TieredMemoryStore.read_record"),
            "cold_recall": (
                "exact RecallResult.io_bytes and file_reads from "
                "wait_recall/resume_recall, including rejected reads"),
            "file_read_counts": (
                "actual read_bytes calls, including payloads later rejected "
                "by digest, scope or generation validation"),
            "artificial_delay": "none",
        },
    )

def merge_readout_metrics(metrics: Sequence[ReadoutMetrics]) -> ReadoutMetrics:
    """Sum actual work across readout attempts, retaining final state gauges."""
    items = tuple(metrics)
    if not items:
        raise ValueError("at least one readout metric set is required")
    final = items[-1]
    gauges = {
        "cache_entries", "cache_bytes", "cache_bound_bytes", "cold_in_progress",
        "records_returned", "records_ready", "records_degraded",
        "records_incomplete", "records_denied", "records_invalidated",
        "records_stale", "records_in_progress", "timings_are_wall_clock",
        "artificial_delay", "larger_bank_allocated", "gpu_resident",
        "page_cache_residency_claimed", "semantic_compression_claimed",
        "cache_is_in_process_copy", "io_provenance",
    }
    values = {}
    for descriptor in dataclass_fields(ReadoutMetrics):
        name = descriptor.name
        if name in gauges:
            values[name] = getattr(final, name)
        else:
            values[name] = sum(getattr(item, name) for item in items)
    return ReadoutMetrics(**values)


def _diagnostics(budget: ReadoutBudget,
                 selection_engine: str = "TieredMemoryStore.select") -> dict:
    return {
        "selection_engine": selection_engine,
        "selection_payloads_reused_when_admitted": True,
        "canonical_text_reference": True,
        "semantic_compression_claimed": False,
        "retrieval_key_separate_from_values": True,
        "hash_derived_retrieval_keys_are_synthetic_controls": True,
        "values_are_producer_output_not_hashed": True,
        "values_synthesized": False,
        "neural_memory_claimed": False,
        "cold_wait_resume": "begin_recall -> wait_recall -> resume_recall",
        "post_io_revalidation": (
            "scope,time_scope,verification,status,revision,cluster_generation,"
            "representation_version,retrieval_key,ordered_values,payload_digests"),
        "page_cache_residency_measured": False,
        "gpu_residency_measured": False,
        "larger_bank_allocated": False,
        "budget": budget.document(),
    }


def _empty_readout(*, store: TieredMemoryStore, query_text: str,
                   scope: MemoryScope, status: str, reason: str,
                   retryable: bool, stats: _Stats, total_ns: int,
                   resumed_ticket_id: str | None = None,
                   detail: Mapping[str, object] | None = None) -> ScopedMemoryReadout:
    cache = _store_cache(store)
    metrics = _finish_metrics(stats, cache, (), total_ns)
    outcome = Outcome(status=status, reason=reason,
                      detail=dict(detail or {}), retryable=retryable)
    return ScopedMemoryReadout(
        format=READOUT_FORMAT,
        query_text=query_text,
        scope=scope,
        status=status,
        outcome=outcome,
        new_decision_boundary=False,
        records=(),
        metrics=metrics,
        selection_id="",
        epoch=store.epoch,
        candidate_count=0,
        truncated=False,
        fallback_mode="none",
        representation_stale=False,
        generation_pins=(),
        pending_cold=(),
        incomplete_record_ids=(),
        denied_record_ids=(),
        invalidated_record_ids=(),
        resumed_ticket_id=resumed_ticket_id,
        diagnostics={"readout": "empty", "reason": reason},
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def read_scoped_memory(store: TieredMemoryStore, query_text: str,
                       scope: MemoryScope,
                       budget: ReadoutBudget | RetrievalBudget | None = None,
                       selection_mode: str = "semantic") -> ScopedMemoryReadout:
    """Read scoped evidence for neural query construction.

    Returns ordered records with canonical text references, retrieval-key
    identity kept separate from ordered values, real I/O and cache metrics, and
    an explicit status/outcome.  Never widens ``scope`` and never performs
    implicit cold I/O beyond the bounded wait budget.
    """
    if not isinstance(store, TieredMemoryStore):
        raise TypeError("store must be a TieredMemoryStore")
    if not isinstance(query_text, str):
        raise ValueError("query_text must be a string")
    if not isinstance(scope, MemoryScope):
        raise TypeError("scope must be a MemoryScope")
    if selection_mode not in {"semantic", "all_eligible"}:
        raise ValueError("selection_mode must be semantic or all_eligible")
    budget = _coerce_budget(budget)
    total_started = _perf_ns()
    stats = _Stats()
    cache = _store_cache(store)
    _cache_trim(cache, budget.cache_bytes, stats)

    selection_started = _perf_ns()
    selection = store.select(
        query_text, scope=scope, budget=budget.retrieval_budget(),
        selection_mode=selection_mode)
    stats.selection_calls = 1
    stats.selection_file_reads = int(selection.file_reads)
    stats.selection_io_bytes = int(selection.io_bytes)
    stats.selection_ns = _perf_ns() - selection_started
    stats.selection_admitted_text_bytes = int(selection.used_text_bytes)
    stats.selection_admitted_value_slots = int(selection.used_value_slots)
    stats.selection_admitted_payload_bytes = (
        int(selection.used_text_bytes)
        + int(selection.used_value_slots) * VALUE_WIDTH * 4)
    stats.selection_admitted_file_reads = (
        sum(1 for record in selection.records
            if record.placement == "warm" and record.text_included)
        + int(selection.used_value_slots))

    spans_by_record: dict[str, list[object]] = {}
    for hit in selection.exact_spans:
        spans_by_record.setdefault(hit.record_id, []).append(hit)

    records: list[ScopedMemoryRecord] = []
    pending: list[PendingColdReadout] = []
    text_available = budget.max_exact_text_bytes
    slots_available = budget.max_value_slots
    try:
        for rank, selected in enumerate(selection.records):
            selection_spans = tuple(spans_by_record.get(selected.record_id, ()))
            live = _load_live(store, selected.record_id)
            if live is None:
                pin = _pin_from_selected(scope.scope_key, selected)
                record = _build_record(
                    store=store,
                    rank=rank, pin=pin, live=None, selected=selected,
                    status="denied", reason="record_missing_before_read",
                    retryable=False, payload_source="none", text=None,
                    values=None, text_sha256=pin["text_sha256"], checks=(),
                    query_text=query_text, selection_spans=selection_spans,
                    decision_boundary=False)
                records.append(record)
                continue
            if str(live.record["scope_key"]) != scope.scope_key:
                pin = _pin_from_live(scope.scope_key, live)
                record = _build_record(
                    store=store,
                    rank=rank, pin=pin, live=live, selected=selected,
                    status="denied", reason="cross_scope_before_read",
                    retryable=False, payload_source="none", text=None,
                    values=None, text_sha256=pin["text_sha256"], checks=(),
                    query_text=query_text, selection_spans=selection_spans,
                    decision_boundary=False)
                records.append(record)
                continue
            if str(live.record["status"]) != "current":
                pin = _pin_from_live(scope.scope_key, live)
                record = _build_record(
                    store=store,
                    rank=rank, pin=pin, live=live, selected=selected,
                    status="invalidated",
                    reason="record_not_current_before_read", retryable=False,
                    payload_source="none", text=None, values=None,
                    text_sha256=pin["text_sha256"], checks=(),
                    query_text=query_text, selection_spans=selection_spans,
                    decision_boundary=True)
                records.append(record)
                continue

            pin = _pin_from_live(scope.scope_key, live)
            key = _cache_key(pin)
            entry = _cache_get(cache, key)
            acquired: _Acquired | None = None
            if entry is not None and len(entry.values) != int(pin["value_slots"]):
                stats.cache_hit_rejected += 1
                _cache_drop(cache, key)
                entry = None
            if entry is not None:
                verdict = _revalidate(
                    store, pin, scope.scope_key, live,
                    payload_text=entry.text, payload_values=entry.values,
                    payload_value_sha=entry.value_sha256, scope=scope)
                stats.revalidations += 1
                if not verdict.ok:
                    stats.revalidation_failures += 1
                if verdict.ok:
                    stats.cache_hits += 1
                    acquired = _Acquired(
                        status="ok", reason="cached_payload_ready",
                        retryable=False, boundary=verdict.decision_boundary,
                        text=entry.text,
                        values=entry.values,
                        text_sha256=entry.text_sha256,
                        value_sha256=entry.value_sha256,
                        payload_source="cache", checks=verdict.checks,
                        io_bytes=0, file_reads=0)
                else:
                    stats.cache_hit_rejected += 1
                    _cache_drop(cache, key)
            if acquired is None:
                stats.cache_misses += 1
                placement = str(live.record["placement"])
                if placement == "cold":
                    acquired = _acquire_cold(
                        store, selection.selection_id, selected, scope,
                        budget, stats, rank)
                else:
                    has_text = selected.text is not None
                    has_values = (selected.values_available
                                  and bool(selected.values))
                    if has_text or (has_values and text_available <= 0):
                        values = (selected.values
                                  if selected.values_available else None)
                        acquired = _Acquired(
                            status=("degraded" if (has_text and values is None)
                                    else "ok"),
                            reason=("value_representation_unavailable"
                                    if (has_text and values is None)
                                    else "selection_payload_ready"),
                            retryable=False, boundary=False,
                            text=selected.text, values=values,
                            text_sha256=selected.text_sha256,
                            value_sha256=(_payload_value_sha(values)
                                          if values else ()),
                            payload_source="selection", checks=(),
                            io_bytes=0, file_reads=0)
                    else:
                        acquired = _acquire_direct(
                            store, selected.record_id, scope, budget, stats)
                if acquired.status in _OK_STATUSES:
                    post = _load_live(store, selected.record_id)
                    revalidation_started = _perf_ns()
                    verdict = _revalidate(
                        store, pin, scope.scope_key, post,
                        payload_text=acquired.text,
                        payload_values=acquired.values,
                        payload_value_sha=acquired.value_sha256, scope=scope)
                    stats.revalidation_ns += _perf_ns() - revalidation_started
                    stats.revalidations += 1
                    if not verdict.ok:
                        stats.revalidation_failures += 1
                        acquired = _Acquired(
                            status=verdict.status, reason=verdict.reason,
                            retryable=False,
                            boundary=verdict.decision_boundary,
                            text=None, values=None, text_sha256="",
                            value_sha256=(), payload_source=acquired.payload_source,
                            checks=verdict.checks,
                            io_bytes=acquired.io_bytes,
                            file_reads=acquired.file_reads)
                    else:
                        live = post or live
                        if (acquired.text is not None
                                and acquired.values is not None
                                and len(acquired.values) == VALUE_SLOTS_PER_RECORD):
                            payload_bytes = (
                                len(acquired.text.encode("utf-8"))
                                + len(acquired.values) * VALUE_WIDTH * 4)
                            _cache_put(
                                cache, key,
                                _CacheEntry(
                                    text=acquired.text,
                                    text_sha256=acquired.text_sha256,
                                    values=acquired.values,
                                    value_sha256=acquired.value_sha256,
                                    bytes=payload_bytes,
                                    placement=str(live.record["placement"]),
                                    inserted_at_ns=time.time_ns()),
                                budget.cache_bytes, stats)
            record, used_text, used_slots = _record_presented(
                store=store,
                rank=rank, pin=pin, live=live, selected=selected,
                acquired=acquired, text_available=text_available,
                slots_available=slots_available, query_text=query_text,
                selection_spans=selection_spans,
                expect_values=bool(live.values))
            if record.payload_source == "cold_recall" and (
                    acquired.pending is not None):
                pending.append(acquired.pending)
            records.append(record)
            text_available = max(0, text_available - used_text)
            slots_available = max(0, slots_available - used_slots)
    finally:
        store.release_selection(selection.selection_id)

    status, reason, retryable = _aggregate(
        records, selection.outcome.status, selection.outcome.reason,
        bool(selection.truncated), bool(selection.representation_stale),
        pending)
    incomplete_ids = tuple(record.record_id for record in records
                           if record.status == "incomplete")
    denied_ids = tuple(record.record_id for record in records
                       if record.status == "denied")
    invalidated_ids = tuple(record.record_id for record in records
                            if record.status in ("invalidated", "stale"))
    boundary = any(record.decision_boundary for record in records)
    detail = {
        "candidate_count": int(selection.candidate_count),
        "semantic_candidate_count": int(selection.candidate_count),
        "selection_mode": selection_mode,
        "truncated": bool(selection.truncated),
        "fallback_mode": selection.fallback_mode,
        "representation_stale": bool(selection.representation_stale),
        "selection_status": selection.outcome.status,
        "selection_reason": selection.outcome.reason,
        "record_statuses": {record.record_id: record.status
                            for record in records},
        "pending_ticket_ids": tuple(item.ticket_id for item in pending),
        "incomplete_record_ids": incomplete_ids,
        "denied_record_ids": denied_ids,
        "invalidated_record_ids": invalidated_ids,
        "budget": budget.document(),
    }
    outcome = Outcome(status=status, reason=reason, detail=detail,
                      retryable=bool(retryable or pending))
    metrics = _finish_metrics(
        stats, cache, records, _perf_ns() - total_started)
    return ScopedMemoryReadout(
        format=READOUT_FORMAT,
        query_text=query_text,
        scope=scope,
        status=status,
        outcome=outcome,
        new_decision_boundary=boundary,
        records=tuple(records),
        metrics=metrics,
        selection_id=selection.selection_id,
        epoch=store.epoch,
        candidate_count=int(selection.candidate_count),
        truncated=bool(selection.truncated),
        fallback_mode=selection.fallback_mode,
        representation_stale=bool(selection.representation_stale),
        generation_pins=selection.generation_pins,
        pending_cold=tuple(pending),
        incomplete_record_ids=incomplete_ids,
        denied_record_ids=denied_ids,
        invalidated_record_ids=invalidated_ids,
        diagnostics={**_diagnostics(budget), "selection_mode": selection_mode},
    )


def resume_scoped_readout(store: TieredMemoryStore, ticket_id: str,
                          *, scope: MemoryScope,
                          budget: ReadoutBudget | RetrievalBudget | None = None
                          ) -> ScopedMemoryReadout:
    """Complete a pending cold ticket with real ``resume_recall`` I/O.

    The ticket must have been produced by :func:`read_scoped_memory` with an
    exhausted wait budget.  Scope is checked before any I/O.
    """
    if not isinstance(store, TieredMemoryStore):
        raise TypeError("store must be a TieredMemoryStore")
    if not isinstance(ticket_id, str) or not ticket_id:
        raise ValueError("ticket_id must be a nonempty string")
    if not isinstance(scope, MemoryScope):
        raise TypeError("scope must be a MemoryScope")
    budget = _coerce_budget(budget)
    total_started = _perf_ns()
    stats = _Stats()
    cache = _store_cache(store)
    _cache_trim(cache, budget.cache_bytes, stats)
    ticket = store.ticket(ticket_id)
    if ticket is None:
        return _empty_readout(
            store=store, query_text="", scope=scope,
            status="denied", reason="recall_ticket_unknown", retryable=False,
            stats=stats, total_ns=_perf_ns() - total_started,
            resumed_ticket_id=ticket_id)
    if ticket.scope != scope:
        return _empty_readout(
            store=store, query_text="", scope=scope,
            status="denied", reason="cross_scope_before_resume",
            retryable=False, stats=stats,
            total_ns=_perf_ns() - total_started,
            resumed_ticket_id=ticket_id,
            detail={"ticket_scope_key": ticket.scope.scope_key,
                    "requested_scope_key": scope.scope_key,
                    "ticket_scope": ticket.scope.document(),
                    "requested_scope": scope.document()})
    pins = tuple(ticket.pinned)
    if not pins:
        return _empty_readout(
            store=store, query_text="", scope=scope,
            status="unsupported", reason="recall_ticket_has_no_pins",
            retryable=False, stats=stats,
            total_ns=_perf_ns() - total_started,
            resumed_ticket_id=ticket_id)

    resume_started = _perf_ns()
    replay = not ticket.pending
    result = store.resume_recall(ticket_id)
    stats.cold_resumes += 1
    if not replay:
        stats.cold_io_bytes += int(result.io_bytes)
        stats.cold_file_reads += int(result.file_reads)
        stats.cold_revalidation_checks += len(result.checks)
    stats.cold_ns += _perf_ns() - resume_started
    recalled = {record.record_id: record for record in result.records}

    records: list[ScopedMemoryRecord] = []
    text_available = budget.max_exact_text_bytes
    slots_available = budget.max_value_slots
    for rank, ticket_pin in enumerate(pins):
        pin = _pin_from_ticket(ticket_pin, scope.scope_key)
        live = _load_live(store, str(pin["record_id"]))
        if result.status != "ready" or str(pin["record_id"]) not in recalled:
            record = _build_record(
                store=store,
                rank=rank, pin=pin, live=live, selected=None,
                status=result.status, reason=result.outcome.reason,
                retryable=bool(result.outcome.retryable)
                or result.status == "incomplete",
                payload_source="cold_recall", text=None, values=None,
                text_sha256=str(pin["text_sha256"]), checks=result.checks,
                query_text="", selection_spans=(),
                decision_boundary=bool(result.new_decision_boundary))
            records.append(record)
            continue
        acquired = recalled[str(pin["record_id"])]
        acquired_values = acquired.values if acquired.values else None
        post = live
        revalidation_started = _perf_ns()
        verdict = _revalidate(
            store, pin, scope.scope_key, post,
            payload_text=acquired.text, payload_values=acquired_values,
            payload_value_sha=tuple(acquired.value_sha256), scope=scope)
        stats.revalidation_ns += _perf_ns() - revalidation_started
        stats.revalidations += 1
        if not verdict.ok:
            stats.revalidation_failures += 1
            record = _build_record(
                store=store,
                rank=rank, pin=pin, live=post, selected=None,
                status=verdict.status, reason=verdict.reason,
                retryable=False, payload_source="cold_recall", text=None,
                values=None, text_sha256="", checks=verdict.checks,
                query_text="", selection_spans=(),
                decision_boundary=True)
            records.append(record)
            continue
        payload = _Acquired(
            status="ok", reason="cold_resume_ready", retryable=False,
            boundary=bool(verdict.decision_boundary), text=acquired.text,
            values=acquired_values, text_sha256=acquired.text_sha256,
            value_sha256=tuple(acquired.value_sha256),
            payload_source="cold_recall", checks=verdict.checks,
            io_bytes=int(result.io_bytes),
            file_reads=int(result.file_reads))
        record, used_text, used_slots = _record_presented(
            store=store,
            rank=rank, pin=pin, live=post, selected=None, acquired=payload,
            text_available=text_available, slots_available=slots_available,
            query_text="", selection_spans=(),
            expect_values=bool(post.values) if post is not None else False)
        if (payload.text is not None and payload.values is not None
                and len(payload.values) == VALUE_SLOTS_PER_RECORD):
            payload_bytes = (len(payload.text.encode("utf-8"))
                             + len(payload.values) * VALUE_WIDTH * 4)
            _cache_put(
                cache, _cache_key(pin),
                _CacheEntry(
                    text=payload.text, text_sha256=payload.text_sha256,
                    values=payload.values, value_sha256=payload.value_sha256,
                    bytes=payload_bytes, placement="cold",
                    inserted_at_ns=time.time_ns()),
                budget.cache_bytes, stats)
        records.append(record)
        text_available = max(0, text_available - used_text)
        slots_available = max(0, slots_available - used_slots)

    status, reason, retryable = _aggregate(
        records, result.status, result.outcome.reason,
        False, False, ())
    boundary = any(record.decision_boundary for record in records)
    incomplete_ids = tuple(record.record_id for record in records
                           if record.status == "incomplete")
    denied_ids = tuple(record.record_id for record in records
                       if record.status == "denied")
    invalidated_ids = tuple(record.record_id for record in records
                            if record.status in ("invalidated", "stale"))
    detail = {
        "ticket_status": result.status,
        "ticket_reason": result.outcome.reason,
        "record_statuses": {record.record_id: record.status
                            for record in records},
        "incomplete_record_ids": incomplete_ids,
        "denied_record_ids": denied_ids,
        "invalidated_record_ids": invalidated_ids,
        "budget": budget.document(),
    }
    outcome = Outcome(status=status, reason=reason, detail=detail,
                      retryable=bool(retryable or result.outcome.retryable))
    metrics = _finish_metrics(
        stats, cache, records, _perf_ns() - total_started)
    diagnostics = _diagnostics(budget)
    diagnostics["resumed_ticket_id"] = ticket_id
    diagnostics["cold_wait_resume"] = "resume_recall"
    return ScopedMemoryReadout(
        format=READOUT_FORMAT,
        query_text="",
        scope=scope,
        status=status,
        outcome=outcome,
        new_decision_boundary=boundary,
        records=tuple(records),
        metrics=metrics,
        selection_id=result.selection_id,
        epoch=store.epoch,
        candidate_count=0,
        truncated=False,
        fallback_mode="none",
        representation_stale=False,
        generation_pins=(),
        pending_cold=(),
        incomplete_record_ids=incomplete_ids,
        denied_record_ids=denied_ids,
        invalidated_record_ids=invalidated_ids,
        resumed_ticket_id=ticket_id,
        diagnostics=diagnostics,
    )


# A convenience name matching the read/resume pairing; both entry points return
# the same authoritative contract.
resume_scoped_memory = resume_scoped_readout


def readout_contract() -> dict:
    """Static description of this readout boundary's claims and limits."""
    return {
        "format": READOUT_FORMAT,
        "entry_points": {
            "read": (
                "read_scoped_memory(store, query_text, scope, budget=None, "
                "selection_mode='semantic')"),
            "resume": "resume_scoped_readout(store, ticket_id, scope=scope)",
        },
        "result_fields": (
            "format, query_text, scope, status, outcome, new_decision_boundary, "
            "records, metrics, selection_id, epoch, candidate_count, truncated, "
            "fallback_mode, representation_stale, generation_pins, pending_cold, "
            "incomplete_record_ids, denied_record_ids, invalidated_record_ids, "
            "resumed_ticket_id, diagnostics"),
        "record_fields": (
            "rank, record_id, cluster_id, cluster_generation, revision, score, "
            "status, reason, retryable, placement, payload_source, text, "
            "text_sha256, text_bytes, source_text_bytes, canonical_span, "
            "source_spans, retrieval_key, values, value_metadata, checks, "
            "source_id, source_epoch, observed_at, valid_to, verification_state, "
            "critical, decision_boundary, available_at, available_at_mapped, "
            "available_at_source, available_at_mapping, event_at, valid_from, "
            "valid_until, source_lineage, lineage_predecessor, source_provenance, provenance, "
            "producer_revision, representation_version, codec_version, "
            "producer_revision_available, representation_version_available, "
            "codec_version_available, record_representation_version, payload_type, "
            "payload_type_available, supersedes, supersedes_available, "
            "superseded_by, record_hash, scope_key, kind, authenticity, importance, "
            "created_at_ns, updated_at_ns, key_producer_id, key_producer_version, "
            "value_producer_id, value_producer_version, value_producer_consistent, "
            "durable_ref"),
        "cache": (
            "bounded in-process LRU keyed by scope, record, revision, "
            "representation_version, cluster generation and canonical text digest"),
        "cold": (
            "begin_recall -> bounded wait_recall -> resume_recall; pending tickets "
            "are explicit and complete via resume_scoped_readout"),
        "revalidation": (
            "live index checks after acquisition: scope, time scope, verification, "
            "status, revision, cluster generation, representation version, "
            "retrieval key, ordered values and payload digests"),
        "claims": {
            "canonical_text_reference": True,
            "retrieval_keys_are_synthetic_controls": True,
            "retrieval_keys_are_neural_memory": False,
            "values_are_producer_output": True,
            "values_hash_derived": False,
            "values_synthesized": False,
            "artificial_delay": False,
            "page_cache_residency_measured": False,
            "gpu_residency_measured": False,
            "larger_bank_allocated": False,
            "semantic_compression_claimed": False,
        },
        "restoration": {
            "available_at": (
                "records.observed_at only when decoded current-record provenance "
                "identifies the persistent learner mapping (available_at_mapped); "
                "otherwise null, with observed_at and available_at_mapping exposed"),
            "valid_from": "records.valid_from",
            "valid_until": "records.valid_to",
            "source_lineage": "records.source_lineage (exact stored string list)",
            "source_provenance": (
                "decoded lineage JSON object whose record_id matches the current "
                "record; correction lineage may also contain an explicit predecessor"),
            "durable_ref": (
                "exact store_root-relative text/key/value paths, digests, cluster "
                "manifest, and any durable_ref embedded in source provenance"),
            "versions": (
                "producer/representation/codec versions only when present in "
                "current source provenance; tiered key/value producer identities and "
                "record_representation_version are exposed separately"),
            "never_approximated": (
                "missing payload_type/codec_version and current provenance are null "
                "with explicit availability flags; correction supersedes comes only "
                "from an explicit predecessor link"),
        },
        "limits": {
            "selection_reads_payloads_internally": True,
            "selection_file_reads_and_io_bytes_are_exact": True,
            "selection_admitted_file_reads_are_derived_subset": True,
            "cache_is_process_local_and_not_tier_accounted": True,
            "exact_phrase_span_fallback": (
                "not used by this entry point; spans are canonical-text and "
                "exact query-string references"),
        },
    }
