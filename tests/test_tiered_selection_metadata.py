"""Metadata-only selection: same ordering/pins, no admitted payload reads."""
from __future__ import annotations

from pathlib import Path

import pytest

from replica_lab.tiered_memory import VALUE_SLOTS_PER_RECORD
from test_tiered_memory import make_draft, open_store, scope


def usage_category(store, category: str) -> int:
    return int(store.usage()["tiers"]["hot"]["by_category"].get(category, 0))


def test_metadata_only_matches_order_and_pins_without_payload_reads(
        tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        user_scope = scope()
        store.admit_cluster(
            "metadata", scope=user_scope,
            records=[make_draft("r1", "quokka alpha launch", seed=51),
                     make_draft("r2", "quokka beta launch", seed=52),
                     make_draft("r3", "unrelated ordinary text", seed=53)])
        full = store.select("quokka", scope=user_scope)
        try:
            assert full.record_ids == ("r1", "r2")
            assert full.used_text_bytes > 0
            assert full.used_value_slots > 0
            assert all(record.text_included for record in full.records)
            charged = usage_category(store, "selection")
            assert charged > 0
        finally:
            store.release_selection(full.selection_id)
        baseline = usage_category(store, "selection")

        metadata = store.select(
            "quokka", scope=user_scope, admit_payload=False)
        try:
            assert metadata.record_ids == full.record_ids
            assert metadata.scores == full.scores
            assert metadata.generation_pins == full.generation_pins
            assert metadata.fallback_mode == "semantic"
            assert metadata.used_text_bytes == 0
            assert metadata.used_value_slots == 0
            assert metadata.requires_cold_recall is False
            assert metadata.file_reads < full.file_reads
            assert metadata.io_bytes < full.io_bytes
            for expected, actual in zip(full.records, metadata.records):
                assert actual.text is None
                assert actual.text_included is False
                assert actual.values == ()
                assert actual.values_available is False
                assert actual.value_slots == 0
                assert actual.value_slot_count == 0
                assert actual.text_sha256 == expected.text_sha256
                assert actual.text_bytes == expected.text_bytes
                assert actual.key_sha256 == expected.key_sha256
                assert actual.cluster_id == expected.cluster_id
                assert actual.cluster_generation == expected.cluster_generation
                assert actual.revision == expected.revision
                assert actual.source_id == expected.source_id
                assert actual.source_epoch == expected.source_epoch
                assert actual.verification_state == expected.verification_state
                assert actual.observed_at == expected.observed_at
                assert actual.valid_to == expected.valid_to
                assert actual.critical == expected.critical
                assert actual.score == expected.score
            assert usage_category(store, "selection") == baseline
        finally:
            store.release_selection(metadata.selection_id)

        with pytest.raises(ValueError, match="admit_payload"):
            store.select("quokka", scope=user_scope, admit_payload="no")


def test_metadata_only_exact_fallback_reads_only_exact_text(
        tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        user_scope = scope()
        store.admit_cluster(
            "warm-exact", scope=user_scope,
            records=[make_draft(
                "w1", "The launch code is ALPHA-7; keep it.", seed=54)])
        store.admit_cluster(
            "cold-exact", scope=user_scope, placement="cold",
            records=[make_draft("c1", "cold secret payload", seed=55)])

        key_only = store.select(
            "cold secret", scope=user_scope, admit_payload=False,
            exact_phrases=())
        try:
            assert set(key_only.record_ids) == {"c1", "w1"}
            assert key_only.record_ids[0] == "c1"
            assert key_only.cold_record_ids == ("c1",)
            assert key_only.requires_cold_recall is True
            key_reads = key_only.file_reads
            key_bytes = key_only.io_bytes
            assert key_reads == 2  # two candidate retrieval keys, no text/values
            assert key_only.used_text_bytes == 0
            assert key_only.used_value_slots == 0
        finally:
            store.release_selection(key_only.selection_id)

        exact = store.select(
            "cold secret", scope=user_scope, admit_payload=False,
            exact_phrases=("ALPHA-7",))
        try:
            assert set(exact.record_ids) == {"c1", "w1"}
            assert exact.cold_record_ids == ("c1",)
            assert exact.requires_cold_recall is True
            assert all(record.text is None for record in exact.records)
            assert all(record.values_available is False
                       for record in exact.records)
            assert exact.used_text_bytes == 0
            assert exact.used_value_slots == 0
            assert len(exact.exact_spans) == 1
            span = exact.exact_spans[0]
            assert span.record_id == "w1"
            assert span.phrase == "ALPHA-7"
            assert span.text == "ALPHA-7"
            assert exact.used_span_bytes == len("ALPHA-7")
            # One exact text read beyond the two key reads.
            assert exact.file_reads == key_reads + 1
            assert exact.io_bytes > key_bytes
            assert exact.io_bytes - key_bytes == len(
                "The launch code is ALPHA-7; keep it.")
        finally:
            store.release_selection(exact.selection_id)


def test_admit_payload_true_preserves_admitted_payload_behaviour(
        tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        user_scope = scope()
        store.admit_cluster(
            "default-payload", scope=user_scope,
            records=[make_draft("r1", "quokka launch", seed=56)])
        default = store.select("quokka", scope=user_scope)
        try:
            explicit = store.select(
                "quokka", scope=user_scope, admit_payload=True)
            try:
                assert default.records[0].text == "quokka launch"
                assert default.records[0].text_included is True
                assert default.records[0].values_available is True
                assert default.records[0].value_slots == VALUE_SLOTS_PER_RECORD
                assert default.used_text_bytes == len(b"quokka launch")
                assert default.used_value_slots == VALUE_SLOTS_PER_RECORD
                assert explicit.record_ids == default.record_ids
                assert explicit.used_text_bytes == default.used_text_bytes
                assert explicit.used_value_slots == default.used_value_slots
            finally:
                store.release_selection(explicit.selection_id)
        finally:
            store.release_selection(default.selection_id)
