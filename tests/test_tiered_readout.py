"""Tests for the scoped memory readout boundary.

These tests use small real SQLite stores and real payload files under pytest's
temporary directory.  They exercise warm selection reuse, real cold
begin/wait/resume I/O, post-I/O revalidation, bounded in-process caching,
budget exhaustion and cache pressure.  They do not load Spark, allocate an
expanded expert bank, run a GPU or qualify native capability.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from replica_lab.tiered_memory import (
    KEY_WIDTH,
    VALUE_SLOTS_PER_RECORD,
    VALUE_WIDTH,
    BudgetError,
    EvidenceRecordDraft,
    MemoryScope,
    RetrievalBudget,
    TierCeilings,
    TieredMemoryStore,
)
from replica_lab.tiered_readout import (
    READOUT_FORMAT,
    ReadoutBudget,
    clear_readout_cache,
    read_scoped_memory,
    readout_cache_state,
    readout_contract,
    resume_scoped_readout,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def sample_values(seed: int) -> tuple[tuple[float, ...], ...]:
    rows = []
    for slot in range(VALUE_SLOTS_PER_RECORD):
        row = []
        for index in range(VALUE_WIDTH):
            numerator = ((seed * 29 + slot * 31 + index * 17) % 101) - 50
            row.append(numerator / 50.0)
        rows.append(tuple(row))
    return tuple(rows)


def scope(subject: str = "user_a", *, as_of: int = 1000) -> MemoryScope:
    return MemoryScope(subject_id=subject, branch_id="main",
                       purpose="assistant", as_of=as_of)


def make_draft(record_id: str, text: str, *, seed: int = 1,
               include_values: bool = True,
               source_id: str = "source-1",
               source_epoch: int = 1,
               observed_at: int = 100,
               event_at: int | None = None,
               valid_from: int | None = None,
               valid_to: int | None = None,
               source_lineage: tuple[str, ...] = (),
               authenticity: str = "unverified",
               verification_state: str = "unverified",
               importance: int = 0) -> EvidenceRecordDraft:
    values = sample_values(seed) if include_values else None
    return EvidenceRecordDraft(
        record_id=record_id, text=text, source_id=source_id,
        source_epoch=source_epoch, observed_at=observed_at,
        event_at=event_at, valid_from=valid_from, valid_to=valid_to,
        authenticity=authenticity, verification_state=verification_state,
        importance=importance, source_lineage=source_lineage,
        values=values,
        value_producer_id="test-values" if values is not None else None,
        value_producer_version="test-values-v1" if values is not None else None,
        key_producer_version="v1",
    )


def policy(**overrides) -> TierCeilings:
    base = {
        "hot_bytes": 32 * 1024**2,
        "warm_bytes": 32 * 1024**2,
        "cold_bytes": 32 * 1024**2,
        "identity_reserved_bytes": 64 * 1024,
        "replay_reserved_bytes": 256 * 1024,
        "minimum_volume_free_bytes": 0,
        "workspace_quota_bytes": 1 * 1024**2,
        "artifact_budget_bytes": 1 * 1024**2,
        "frequent_reserve_fraction": 0.5,
        "frequent_min_accesses": 2,
    }
    base.update(overrides)
    return TierCeilings(**base)


def open_store(tmp_path: Path, *, name: str = "store") -> TieredMemoryStore:
    return TieredMemoryStore(
        tmp_path / name,
        profile_id="P1",
        ceilings=policy(),
        enforce_volume_free=False,
    )


class MissingRestoreField(RuntimeError):
    """A neural-path record cannot be rebuilt without guessing a field."""


class NeuralTextPathStub:
    """Minimal stand-in for SparkTextMemoryPath evidence, keyed by record_id.

    It accepts only an identity that carries the exact restoration fields, so a
    readout result can be checked for whether it represents a record absent
    from the neural path and whether it can be re-observed without guessing.
    """

    REQUIRED = ("record_id", "scope", "source_epoch", "available_at",
                "valid_from", "producer_revision", "representation_version",
                "codec_version", "payload_type", "source_lineage")

    def __init__(self) -> None:
        self.records: dict[str, dict] = {}

    def observe(self, identity: dict) -> None:
        missing = [name for name in self.REQUIRED if identity.get(name) is None]
        if missing:
            raise MissingRestoreField(
                "restoration requires: " + ", ".join(missing))
        if not isinstance(identity["source_lineage"], dict) \
                or not identity["source_lineage"]:
            raise MissingRestoreField("restoration requires source_lineage")
        self.records[identity["record_id"]] = dict(identity)

    def contains(self, record_id: str) -> bool:
        return record_id in self.records

    def remove(self, record_id: str) -> None:
        self.records.pop(record_id, None)


def _identity_from_readout(record) -> dict:
    """Rebuild a Spark-shaped evidence identity from exact readout fields."""
    missing = [name for name, value in (
        ("codec_version", record.codec_version),
        ("producer_revision", record.producer_revision),
        ("representation_version", record.representation_version),
        ("payload_type", record.payload_type),
        ("available_at", record.available_at),
        ("valid_from", record.valid_from),
    ) if value is None]
    if missing:
        raise MissingRestoreField(
            "restoration requires: " + ", ".join(missing))
    if not record.source_provenance:
        raise MissingRestoreField("restoration requires source_lineage")
    return {
        "record_id": record.record_id,
        "scope": record.scope_key,
        "source_epoch": record.source_epoch,
        "available_at": record.available_at,
        "valid_from": record.valid_from,
        "valid_until": record.valid_until,
        "producer_revision": record.producer_revision,
        "representation_version": record.representation_version,
        "codec_version": record.codec_version,
        "payload_type": record.payload_type,
        "source_lineage": dict(record.source_provenance),
        "supersedes": record.supersedes,
        "durable_ref": (None if record.durable_ref is None
                        else dict(record.durable_ref)),
    }


VALUE_BYTES = VALUE_WIDTH * 4


def expected_payload_bytes(text: str) -> int:
    return len(text.encode("utf-8")) + VALUE_SLOTS_PER_RECORD * VALUE_BYTES


# ---------------------------------------------------------------------------
# Warm reads: order, spans, identity and budgeted payloads
# ---------------------------------------------------------------------------

def test_warm_readout_preserves_order_spans_and_producer_identity(
        tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        user = scope()
        texts = {
            "w1": "alpha launch code red",
            "w2": "alpha launch code blue",
            "w3": "beta launch code green",
        }
        store.admit_cluster(
            "warm", scope=user,
            records=[make_draft(rid, text, seed=index + 1)
                     for index, (rid, text) in enumerate(texts.items())])
        query = "alpha launch code"

        direct = store.select(query, scope=user, budget=RetrievalBudget())
        store.release_selection(direct.selection_id)

        readout = read_scoped_memory(store, query, user)
        assert readout.format == READOUT_FORMAT
        assert readout.status == "ok"
        assert readout.outcome.reason == "readout_ready"
        assert readout.record_ids == direct.record_ids
        assert readout.new_decision_boundary is False

        for rank, (record, selected) in enumerate(
                zip(readout.records, direct.records)):
            assert record.rank == rank
            assert record.record_id == selected.record_id
            assert record.text == texts[record.record_id]
            assert record.payload_source == "selection"
            assert record.cluster_id == selected.cluster_id
            assert record.cluster_generation == selected.cluster_generation
            assert record.revision == selected.revision
            assert record.source_id == "source-1"
            assert record.source_epoch == 1
            assert record.text_sha256 == hashlib.sha256(
                record.text.encode("utf-8")).hexdigest()
            assert record.text_bytes == len(record.text.encode("utf-8"))

            # Canonical text is returned by reference, not compressed.
            span = record.canonical_span
            assert span is not None
            assert (span.start, span.end) == (0, len(record.text))
            assert span.exclusive_end is True
            assert span.canonical_reference is True
            assert span.sha256 == hashlib.sha256(
                span.text.encode("utf-8")).hexdigest()
            assert span.source_text_sha256 == record.text_sha256

            # Retrieval keys stay separate from ordered values.
            assert record.retrieval_key.width == KEY_WIDTH
            assert len(record.retrieval_key.sha256) == 64
            assert record.retrieval_key.is_synthetic_control is True
            assert record.retrieval_key.is_neural_memory is False
            assert record.retrieval_key.producer_id == (
                "deterministic_unigram_bigram")
            assert record.retrieval_key.producer_version == "v1"
            assert record.values is not None
            assert len(record.values) == VALUE_SLOTS_PER_RECORD
            assert len(record.values[0]) == VALUE_WIDTH
            assert record.value_metadata is not None
            assert record.value_metadata.producer_id == "test-values"
            assert record.value_metadata.producer_version == "test-values-v1"
            assert record.value_metadata.is_hash_derived is False
            assert record.value_metadata.is_neural_memory is False
            assert record.document()["values_hash_derived"] is False
            assert record.values == selected.values

            if query in record.text:
                spans = [item for item in record.source_spans
                         if item.query_reference]
                assert spans
                start = record.text.find(query)
                assert (spans[0].start, spans[0].end) == (
                    start, start + len(query))
                assert spans[0].text == query

        metrics = readout.metrics
        assert metrics.selection_admitted_payload_bytes > 0
        assert metrics.selection_admitted_file_reads > 0
        assert metrics.payload_read_calls == 0
        assert metrics.payload_io_bytes == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == len(readout.records)
        assert metrics.cache_inserts == len(readout.records)
        assert metrics.total_io_bytes == metrics.selection_admitted_payload_bytes
        assert metrics.total_file_reads == metrics.selection_admitted_file_reads
        assert metrics.total_ns > 0
        assert metrics.selection_ns >= 0
        assert metrics.artificial_delay is False
        assert metrics.gpu_resident is False
        assert metrics.larger_bank_allocated is False
        assert metrics.semantic_compression_claimed is False

        # The selection handle and its hot charge are released.
        hot = store.usage()["tiers"]["hot"]
        assert hot["by_category"].get("selection", 0) == 0

        # The in-process cache is bounded and holds the full payloads.
        state = readout_cache_state(store)
        assert state["entries"] == len(readout.records)
        assert state["bytes"] <= state["bound_bytes"]

        json.dumps(readout.document())  # one authoritative JSON contract

        assert readout_contract()["claims"]["gpu_residency_measured"] is False
        assert readout_contract()["claims"]["semantic_compression_claimed"] is False


def test_warm_cache_hits_and_explicit_text_budget_exhaustion(
        tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        user = scope()
        texts = {
            "w1": "gamma shared readout alpha",
            "w2": "gamma shared readout beta",
        }
        store.admit_cluster(
            "warm", scope=user,
            records=[make_draft(rid, text, seed=index + 4)
                     for index, (rid, text) in enumerate(texts.items())])
        query = "gamma shared readout"

        first = read_scoped_memory(store, query, user)
        assert first.status == "ok"
        assert first.metrics.cache_inserts == len(first.records)

        second = read_scoped_memory(store, query, user)
        assert second.record_ids == first.record_ids
        assert second.status == "ok"
        assert second.metrics.cache_hits == len(second.records)
        assert second.metrics.cache_misses == 0
        assert second.metrics.payload_read_calls == 0
        assert second.metrics.payload_io_bytes == 0
        assert all(record.payload_source == "cache"
                   for record in second.records)
        assert second.texts == first.texts

        # With no canonical-text budget the payload is still ordered and valued,
        # but text omission is explicit rather than silently fabricated.
        zero_text = read_scoped_memory(
            store, query, user, ReadoutBudget(max_exact_text_bytes=0))
        assert zero_text.status == "incomplete"
        assert zero_text.metrics.cache_hits == len(zero_text.records)
        assert zero_text.metrics.payload_io_bytes == 0
        assert zero_text.incomplete_record_ids == zero_text.record_ids
        for record in zero_text.records:
            assert record.text is None
            assert record.canonical_span is None
            assert record.status == "incomplete"
            assert record.reason == "canonical_text_budget_exhausted"
            assert record.values is not None
            assert len(record.values) == VALUE_SLOTS_PER_RECORD


def test_direct_readout_payload_reads_are_measured_and_cached(
        tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        user = scope()
        texts = {
            "d1": "delta direct payload one",
            "d2": "delta direct payload two",
        }
        store.admit_cluster(
            "warm", scope=user,
            records=[make_draft(rid, text, seed=index + 7)
                     for index, (rid, text) in enumerate(texts.items())])
        query = "delta direct payload"
        budget = ReadoutBudget(max_exact_text_bytes=1)

        first = read_scoped_memory(store, query, user, budget)
        assert first.status == "incomplete"
        assert first.metrics.payload_read_calls == len(first.records)
        assert first.metrics.payload_file_reads == (
            len(first.records) * (1 + VALUE_SLOTS_PER_RECORD))
        assert first.metrics.payload_io_bytes == sum(
            expected_payload_bytes(text) for text in texts.values())
        assert first.metrics.cache_inserts == len(first.records)
        for record in first.records:
            assert record.payload_source == "read_record"
            assert record.text is None
            assert record.reason == "canonical_text_budget_exhausted"
            assert record.values is not None

        second = read_scoped_memory(store, query, user, budget)
        assert second.metrics.cache_hits == len(second.records)
        assert second.metrics.payload_read_calls == 0
        assert second.metrics.payload_io_bytes == 0
        assert all(record.payload_source == "cache"
                   for record in second.records)

        assert readout_cache_state(store)["entries"] == len(first.records)
        clear_readout_cache(store)
        assert readout_cache_state(store)["entries"] == 0


# ---------------------------------------------------------------------------
# Cold begin/wait/resume and post-I/O revalidation
# ---------------------------------------------------------------------------

def test_cold_recall_is_real_and_cached_second_read_has_no_cold_io(
        tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        user = scope()
        text = "alphaunique cold payload"
        store.admit_cluster(
            "cold", scope=user, placement="cold",
            records=[make_draft("c1", text, seed=11),
                     make_draft("c2", "betaunique cold payload", seed=12)])
        budget = ReadoutBudget(max_records=1)
        query = "alphaunique"

        first = read_scoped_memory(store, query, user, budget)
        assert first.status == "ok"
        record = first.records[0]
        assert record.record_id == "c1"
        assert record.payload_source == "cold_recall"
        assert record.text == text
        assert record.values is not None
        assert len(record.values) == VALUE_SLOTS_PER_RECORD
        metrics = first.metrics
        assert metrics.cold_tickets == 1
        assert metrics.cold_waits == 1
        assert metrics.cold_resumes == 1
        assert metrics.cold_in_progress == 0
        assert metrics.cold_io_bytes == expected_payload_bytes(text)
        assert metrics.cold_file_reads == 1 + VALUE_SLOTS_PER_RECORD
        assert metrics.cold_revalidation_checks > 0
        assert metrics.payload_io_bytes == 0
        assert metrics.total_io_bytes == (
            metrics.selection_admitted_payload_bytes
            + metrics.cold_io_bytes)
        assert first.new_decision_boundary is False
        assert first.pending_cold == ()

        second = read_scoped_memory(store, query, user, budget)
        assert second.status == "ok"
        assert second.texts == first.texts
        assert second.records[0].payload_source == "cache"
        assert second.metrics.cache_hits == 1
        assert second.metrics.cold_tickets == 0
        assert second.metrics.cold_io_bytes == 0
        assert second.metrics.payload_io_bytes == 0


def test_zero_wait_leaves_explicit_pending_ticket_and_resume_completes(
        tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        user = scope()
        text = "gammaunique cold payload"
        store.admit_cluster(
            "cold", scope=user, placement="cold",
            records=[make_draft("g1", text, seed=13)])
        budget = ReadoutBudget(max_records=1, cold_wait_seconds=0.0)

        pending = read_scoped_memory(store, "gammaunique cold payload",
                                      user, budget)
        assert pending.status == "in_progress"
        assert pending.retryable is True
        assert len(pending.pending_cold) == 1
        entry = pending.pending_cold[0]
        assert entry.record_id == "g1"
        assert entry.ticket_id
        assert pending.records[0].status == "in_progress"
        assert pending.records[0].text is None
        assert pending.metrics.cold_in_progress == 1
        assert pending.metrics.cold_io_bytes == 0
        assert pending.metrics.cold_resumes == 0

        resumed = resume_scoped_readout(
            store, entry.ticket_id, scope=user, budget=budget)
        assert resumed.status == "ok"
        assert resumed.resumed_ticket_id == entry.ticket_id
        assert resumed.records[0].text == text
        assert resumed.metrics.cold_resumes == 1
        assert resumed.metrics.cold_io_bytes == expected_payload_bytes(text)
        assert resumed.metrics.cold_file_reads == (
            1 + VALUE_SLOTS_PER_RECORD)
        assert resumed.new_decision_boundary is False


def test_cold_resume_after_revocation_refuses_payload_and_sets_boundary(
        tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        user = scope()
        text = "deltaunique cold payload"
        store.admit_cluster(
            "cold", scope=user, placement="cold",
            records=[make_draft("r1", text, seed=14)])
        budget = ReadoutBudget(max_records=1, cold_wait_seconds=0.0)
        pending = read_scoped_memory(store, "deltaunique cold payload",
                                      user, budget)
        ticket_id = pending.pending_cold[0].ticket_id

        revocation = store.revoke_record(
            "r1", reason="user_retraction", scope=user)
        resumed = resume_scoped_readout(
            store, ticket_id, scope=user, budget=budget)

        assert resumed.status in ("invalidated", "denied")
        assert resumed.new_decision_boundary is True
        assert resumed.records[0].status in ("invalidated", "denied")
        assert resumed.records[0].text is None
        assert resumed.records[0].values is None
        assert "r1" in resumed.invalidated_record_ids
        # The payload was read before validation, then refused.
        assert resumed.metrics.cold_io_bytes > 0
        assert resumed.epoch >= revocation.epoch


# ---------------------------------------------------------------------------
# Scope isolation and representation invalidation
# ---------------------------------------------------------------------------

def test_cross_scope_reads_and_resumes_are_explicitly_denied(
        tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        owner = scope()
        intruder = scope("someone_else")
        store.admit_cluster(
            "warm", scope=owner,
            records=[make_draft("x1", "epsilon private payload", seed=15)])
        store.admit_cluster(
            "cold", scope=owner, placement="cold",
            records=[make_draft("x2", "zeta private cold payload", seed=16)])

        warm_read = read_scoped_memory(store, "epsilon private payload",
                                        intruder)
        assert warm_read.records == ()
        assert warm_read.candidate_count == 0
        assert warm_read.status == "incomplete"
        assert warm_read.outcome.reason == "no_semantic_candidate"
        assert warm_read.metrics.total_io_bytes == 0
        assert warm_read.metrics.payload_read_calls == 0
        assert warm_read.metrics.cold_tickets == 0

        pending = read_scoped_memory(
            store, "zeta private cold payload", owner,
            ReadoutBudget(max_records=1, cold_wait_seconds=0.0))
        ticket_id = pending.pending_cold[0].ticket_id
        denied = resume_scoped_readout(store, ticket_id, scope=intruder)
        assert denied.status == "denied"
        assert denied.outcome.reason == "cross_scope_before_resume"
        assert denied.records == ()
        assert denied.metrics.cold_io_bytes == 0
        assert denied.metrics.cold_resumes == 0


def test_representation_invalidation_prevents_stale_cache_reuse(
        tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        user = scope()
        text = "eta representation payload"
        store.admit_cluster(
            "warm", scope=user,
            records=[make_draft("v1", text, seed=17)])
        first = read_scoped_memory(store, "eta representation payload", user)
        assert first.status == "ok"
        assert readout_cache_state(store)["entries"] == 1

        invalidation = store.invalidate_representations(
            "v1", reason="encoder_version_superseded")
        assert invalidation.reencode_required is True
        assert invalidation.stale_keys >= 1

        second = read_scoped_memory(store, "eta representation payload", user)
        assert second.representation_stale is True
        assert second.records == ()
        assert second.status == "incomplete"
        assert second.outcome.reason == "no_semantic_candidate"
        assert second.metrics.cache_hits == 0
        assert second.metrics.payload_io_bytes == 0
        assert second.metrics.total_io_bytes == 0
        # The cache was not consulted for a record the store no longer selects.
        assert readout_cache_state(store)["entries"] == 1


# ---------------------------------------------------------------------------
# Bounded cache pressure and budget exhaustion
# ---------------------------------------------------------------------------

def test_cache_pressure_evicts_and_stays_bounded_without_losing_evidence(
        tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        user = scope()
        records = [
            make_draft(f"p{index}", f"pressure payload record {name}", seed=20 + index)
            for index, name in enumerate(("one", "two", "three", "four"))
        ]
        store.admit_cluster("cold", scope=user, placement="cold",
                            records=records)
        query = "pressure payload record"
        budget = ReadoutBudget(max_records=4, cache_bytes=5000)

        first = read_scoped_memory(store, query, user, budget)
        assert first.status == "ok"
        assert len(first.records) == 4
        assert first.metrics.cold_tickets == 4
        assert first.metrics.cache_inserts == 4
        assert first.metrics.cache_evictions >= 3
        assert first.metrics.cache_bytes <= 5000
        assert first.metrics.cache_entries <= 1
        assert all(record.text for record in first.records)

        second = read_scoped_memory(store, query, user, budget)
        assert second.status == "ok"
        assert second.texts == first.texts
        assert second.metrics.cache_hits <= first.metrics.cache_entries
        assert second.metrics.cold_tickets == (
            4 - second.metrics.cache_hits)
        assert second.metrics.cold_io_bytes > 0
        assert second.metrics.cache_bytes <= 5000

        # The bound is a real in-process byte bound, not a tier allocation.
        state = readout_cache_state(store)
        assert state["bytes"] <= 5000


def test_cold_budget_denial_is_explicit_and_never_fabricates_payload(
        tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        user = scope()
        store.admit_cluster(
            "cold", scope=user, placement="cold",
            records=[make_draft("b1", "thetaunique cold payload", seed=30)])
        budget = ReadoutBudget(max_records=1, max_cold_bytes=10)

        result = read_scoped_memory(store, "thetaunique cold payload",
                                    user, budget)
        assert result.status == "incomplete"
        record = result.records[0]
        assert record.record_id == "b1"
        assert record.status == "incomplete"
        assert record.reason == "cold_io_budget_exceeded"
        assert record.retryable is True
        assert record.text is None
        assert record.values is None
        assert record.value_metadata is None
        assert record.payload_source == "cold_recall"
        assert record.retrieval_key.is_synthetic_control is True
        assert result.incomplete_record_ids == ("b1",)
        assert result.pending_cold == ()
        assert result.metrics.cold_tickets == 1
        assert result.metrics.cold_io_bytes == 0
        assert result.metrics.cold_file_reads == 0

        with pytest.raises(BudgetError):
            ReadoutBudget(max_records=9)
        with pytest.raises(BudgetError):
            ReadoutBudget(max_value_slots=33)
        with pytest.raises(ValueError):
            ReadoutBudget(cache_bytes=-1)


def test_truncated_selection_is_reported_as_degraded(tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        user = scope()
        store.admit_cluster(
            "warm", scope=user,
            records=[make_draft("t1", "iota shared payload one", seed=31),
                     make_draft("t2", "iota shared payload two", seed=32)])
        result = read_scoped_memory(
            store, "iota shared payload", user,
            ReadoutBudget(max_records=1))
        assert result.status == "degraded"
        assert result.truncated is True
        assert result.outcome.reason == "selection_truncated"
        assert len(result.records) == 1
        assert result.records[0].status == "ok"
        assert result.candidate_count == 2


# ---------------------------------------------------------------------------
# Canonical restoration evidence for records missing from the neural path
# ---------------------------------------------------------------------------

def test_readout_exposes_exact_restore_provenance_and_durable_ref(
        tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        user = scope(as_of=1500)
        text = "lambdarestore exact canonical payload"
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        provenance = {
            "fixture_format": "test.fixture.v1",
            "record_id": "restore1",
            "role": "fit",
            "source": "owned_test",
            "text_sha256": digest,
            "world_id": "world_a",
            "producer_revision": "text-producer-v1",
            "representation_version": "unicode-token-hash-v1",
            "codec_version": "spine-ingestion-codec-v1",
            "payload_type": "observation",
            "canonical_text_retained": True,
            "durable_ref": {
                "locator": "data/test/observations.jsonl#restore1",
                "text_sha256": digest,
            },
        }
        lineage = json.dumps(provenance, sort_keys=True)
        store.admit_cluster(
            "restore", scope=user,
            records=[make_draft(
                "restore1", text, seed=40, source_id="owned_test",
                source_epoch=7, observed_at=1200, event_at=1100,
                valid_from=1000, valid_to=2000, source_lineage=(lineage,))])

        readout = read_scoped_memory(store, "lambdarestore canonical", user)
        assert readout.status == "ok"
        record = readout.records[0]

        # Timestamps are the exact stored columns.  available_at is the
        # observed_at column, labelled with the learner write-path mapping.
        assert record.observed_at == 1200
        assert record.available_at == 1200
        assert record.available_at_mapped is True
        assert record.available_at_source == "records.observed_at"
        assert record.available_at_mapping == (
            "persistent_learner_runner.available_at_to_observed_at")
        assert record.event_at == 1100
        assert record.valid_from == 1000
        assert record.valid_until == 2000
        assert record.valid_to == 2000

        # Exact lineage and decoded provenance, including versions that were
        # actually recorded.  No value is inferred when absent.
        assert record.source_lineage == (lineage,)
        assert record.source_provenance == provenance
        assert record.producer_revision == "text-producer-v1"
        assert record.producer_revision_available is True
        assert record.representation_version == "unicode-token-hash-v1"
        assert record.representation_version_available is True
        assert record.codec_version == "spine-ingestion-codec-v1"
        assert record.codec_version_available is True
        assert record.payload_type == "observation"
        assert record.payload_type_available is True
        assert record.supersedes is None
        assert record.supersedes_available is False
        assert record.superseded_by is None

        row = store.connection.execute(
            "SELECT * FROM records WHERE record_id='restore1'").fetchone()
        assert record.record_representation_version == row["representation_version"]
        assert record.record_representation_version.startswith("rep_")
        assert record.record_representation_version != (
            record.representation_version)
        assert record.record_hash == row["record_hash"]
        assert record.scope_key == user.scope_key
        assert record.kind == "source"
        assert record.authenticity == "unverified"
        assert record.importance == 0
        assert isinstance(record.created_at_ns, int)
        assert isinstance(record.updated_at_ns, int)

        # Tiered key/value producer identities stay separate from the
        # neural-path producer/representation/codec versions above.
        assert record.key_producer_id == "deterministic_unigram_bigram"
        assert record.key_producer_version == "v1"
        assert record.value_producer_id == "test-values"
        assert record.value_producer_version == "test-values-v1"
        assert record.value_producer_consistent is True

        provenance_map = record.provenance
        assert provenance_map["source_id"] == "owned_test"
        assert provenance_map["source_epoch"] == 7
        assert provenance_map["available_at"] == 1200
        assert provenance_map["observed_at"] == 1200
        assert provenance_map["event_at"] == 1100
        assert provenance_map["valid_from"] == 1000
        assert provenance_map["valid_until"] == 2000
        assert provenance_map["record_hash"] == row["record_hash"]
        assert provenance_map["scope_key"] == user.scope_key
        assert provenance_map["source_lineage"] == [lineage]
        assert provenance_map["source_provenance"] == provenance
        assert provenance_map["superseded_by"] is None
        assert isinstance(provenance_map["created_at_ns"], int)
        assert isinstance(provenance_map["updated_at_ns"], int)

        durable = record.durable_ref
        assert durable is not None
        assert durable["text_sha256"] == digest
        assert durable["relative_locator"] == (
            "storage/warm/records/restore1/text.bin")
        assert durable["locator"] == str(
            store.root / "storage/warm/records/restore1/text.bin")
        assert Path(durable["locator"]).read_text(encoding="utf-8") == text
        assert durable["store_root"] == str(store.root)
        assert durable["record_id"] == "restore1"
        assert durable["scope_key"] == user.scope_key
        assert durable["cluster_generation"] == record.cluster_generation
        assert durable["record_hash"] == row["record_hash"]
        assert durable["text"]["sha256"] == digest
        assert durable["text"]["bytes"] == len(text.encode("utf-8"))
        assert durable["key"]["width"] == KEY_WIDTH
        assert len(Path(durable["key"]["path"]).read_bytes()) == KEY_WIDTH
        assert hashlib.sha256(
            Path(durable["key"]["path"]).read_bytes()).hexdigest() == (
                durable["key"]["sha256"])
        assert len(durable["values"]) == VALUE_SLOTS_PER_RECORD
        for slot, value_ref in enumerate(durable["values"]):
            assert value_ref["slot"] == slot
            assert Path(value_ref["path"]).is_file()
            assert hashlib.sha256(
                Path(value_ref["path"]).read_bytes()).hexdigest() == (
                    value_ref["sha256"])
        manifest = durable["cluster_manifest"]
        assert manifest is not None
        assert manifest["generation"] == record.cluster_generation
        assert Path(manifest["path"]).is_file()
        assert len(manifest["sha256"]) == 64
        assert durable["recorded_durable_ref"] == provenance["durable_ref"]

        json.dumps(record.document())
        contract = readout_contract()["restoration"]
        assert "observed_at" in contract["available_at"]
        assert "never_approximated" in contract


def test_missing_neural_path_record_is_restorable_without_guessing(
        tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        user = scope(as_of=1500)
        text_a = "omicronrestore exact canonical payload"
        text_b = "pirestore exact canonical payload"
        digest_a = hashlib.sha256(text_a.encode("utf-8")).hexdigest()
        digest_b = hashlib.sha256(text_b.encode("utf-8")).hexdigest()
        provenance_a = {
            "fixture_format": "test.fixture.v1", "record_id": "restore2",
            "role": "fit", "source": "owned_test", "text_sha256": digest_a,
            "world_id": "world_a", "producer_revision": "text-producer-v1",
            "representation_version": "unicode-token-hash-v1",
            "codec_version": "spine-ingestion-codec-v1",
            "payload_type": "observation",
        }
        provenance_b = {
            "fixture_format": "test.fixture.v1", "record_id": "restore3",
            "role": "fit", "source": "owned_test", "text_sha256": digest_b,
            "world_id": "world_b",
        }
        store.admit_cluster(
            "restore", scope=user,
            records=[
                make_draft(
                    "restore2", text_a, seed=41, source_id="owned_test",
                    source_epoch=7, observed_at=1200, event_at=1100,
                    valid_from=1000, valid_to=2000,
                    source_lineage=(json.dumps(provenance_a, sort_keys=True),)),
                make_draft(
                    "restore3", text_b, seed=42, source_id="owned_test",
                    source_epoch=7, observed_at=1200, event_at=1100,
                    valid_from=1000, valid_to=2000,
                    source_lineage=(json.dumps(provenance_b, sort_keys=True),)),
            ])

        # The neural path does not contain the record the readout selected.
        path = NeuralTextPathStub()
        assert path.contains("restore2") is False
        readout = read_scoped_memory(store, "omicronrestore", user)
        record = readout.records[0]
        assert record.record_id == "restore2"

        identity = _identity_from_readout(record)
        assert identity["available_at"] == 1200
        assert identity["valid_from"] == 1000
        assert identity["valid_until"] == 2000
        assert identity["source_epoch"] == 7
        assert identity["source_lineage"] == provenance_a
        assert identity["durable_ref"] == dict(record.durable_ref)

        path.observe(identity)
        assert path.contains("restore2") is True
        assert path.records["restore2"] == identity

        # Deleting it again does not lose the canonical restoration evidence:
        # the same readout result reconstructs the exact identity.
        path.remove("restore2")
        assert path.contains("restore2") is False
        path.observe(_identity_from_readout(record))
        assert path.records["restore2"] == identity

        # Absent producer/representation/codec/payload provenance is exposed as
        # null with explicit flags, and restoration refuses to guess it.
        partial = read_scoped_memory(store, "pirestore", user).records[0]
        assert partial.record_id == "restore3"
        assert partial.available_at == 1200
        assert partial.source_lineage == (
            json.dumps(provenance_b, sort_keys=True),)
        assert partial.source_provenance == provenance_b
        assert partial.producer_revision is None
        assert partial.producer_revision_available is False
        assert partial.representation_version is None
        assert partial.representation_version_available is False
        assert partial.codec_version is None
        assert partial.codec_version_available is False
        assert partial.payload_type is None
        assert partial.payload_type_available is False
        assert partial.supersedes is None
        assert partial.supersedes_available is False
        assert partial.durable_ref is not None
        with pytest.raises(MissingRestoreField) as failure:
            _identity_from_readout(partial)
        assert "codec_version" in str(failure.value)
        assert "restore3" not in path.records


def test_correction_lineage_keeps_restoration_fields_exact(
        tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        user = scope(as_of=1500)
        old_text = "oldnurestore canonical payload"
        new_text = "correctednurestore canonical payload"
        old_digest = hashlib.sha256(old_text.encode("utf-8")).hexdigest()
        new_digest = hashlib.sha256(new_text.encode("utf-8")).hexdigest()
        provenance = {
            "fixture_format": "test.fixture.v1", "record_id": "nurestore-v1",
            "source": "owned_test", "text_sha256": old_digest,
            "world_id": "world_c", "producer_revision": "text-producer-v1",
            "representation_version": "unicode-token-hash-v1",
            "codec_version": "spine-ingestion-codec-v1",
            "payload_type": "observation",
        }
        store.admit_cluster(
            "corrections", scope=user,
            records=[make_draft(
                "nurestore-v1", old_text, seed=50, source_id="owned_test",
                source_epoch=7, observed_at=1000, event_at=1000,
                valid_from=1000, valid_to=2000,
                source_lineage=(json.dumps(provenance, sort_keys=True),))])
        correction = store.correct_record(
            "nurestore-v1", correction_id="nurestore-v2", text=new_text,
            source_id="owned_test", source_epoch=8, observed_at=1200,
            event_at=1100, valid_from=1100, valid_to=2000,
            values=sample_values(51), value_producer_id="test-values",
            value_producer_version="test-values-v1", scope=user)
        assert correction.new_record_id == "nurestore-v2"

        record = read_scoped_memory(
            store, "correctednurestore", user).records[0]
        assert record.record_id == "nurestore-v2"
        assert record.kind == "correction"
        assert record.available_at == 1200
        assert record.available_at_mapped is True
        assert record.event_at == 1100
        assert record.valid_from == 1100
        assert record.valid_until == 2000

        # The store appends the corrected predecessor to source_lineage; the
        # first entry is the original provenance object.  Both are exact.
        assert len(record.source_lineage) == 2
        assert record.source_lineage[1] == "nurestore-v1"
        assert record.lineage_predecessor == "nurestore-v1"
        assert record.source_provenance == provenance
        assert record.provenance["lineage_predecessor"] == "nurestore-v1"
        assert record.producer_revision == "text-producer-v1"
        assert record.representation_version == "unicode-token-hash-v1"
        assert record.codec_version == "spine-ingestion-codec-v1"

        # Supersedes is not a store column, so it stays null and flagged; the
        # exact predecessor remains available through source_lineage.
        assert record.supersedes is None
        assert record.supersedes_available is False

        durable = record.durable_ref
        assert durable is not None
        assert durable["text_sha256"] == new_digest
        assert Path(durable["locator"]).read_text(encoding="utf-8") == new_text
        assert durable["locator"] != str(
            store.root / "storage/warm/records/nurestore-v1/text.bin")
