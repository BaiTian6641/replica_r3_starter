"""Hot/warm/cold validity, recovery and accounting CPU contracts."""
from __future__ import annotations

import pytest

from replica_lab.successors.tiers import (CapacityExceeded, DirtyState,
                                          ProjectionCache, ReadContract,
                                          ReplayEligibility, TieredBlobStore,
                                          TierLimits)

LIMITS = TierLimits(hot_bytes=2048, warm_bytes=4096, cold_bytes=8192,
                    identity_reservation_bytes=128,
                    replay_reservation_bytes=128)
CONTRACT = ReadContract("producer-v1", "rep-v1", "a" * 64)


def make_store(root, *, delay=0.0, limits=LIMITS) -> TieredBlobStore:
    return TieredBlobStore(root, limits=limits, hot_capacity_bytes=512,
                           io_delay_seconds=delay)


def publish(store, record_id="rec-a", payload=b"payload-" * 8, **changes):
    values = dict(generation=1, kind="evidence", producer="producer-v1",
                  representation="rep-v1", scope_digest="a" * 64)
    values.update(changes)
    return store.publish(record_id, payload, **values)


def test_tier_transitions_preserve_complete_payload(tmp_path):
    store = make_store(tmp_path)
    manifest = publish(store)
    hot = store.read("rec-a", contract=CONTRACT)
    assert hot.status == "ready" and hot.tier == "hot"
    store.demote("rec-a")
    cold = make_store(tmp_path).read("rec-a", contract=CONTRACT, expected_generation=1)
    assert cold.status == "ready" and cold.tier == "cold"
    assert cold.payload == b"payload-" * 8
    cold_store = make_store(tmp_path)
    cold_store.promote("rec-a")
    warm = make_store(tmp_path).read("rec-a", contract=CONTRACT, expected_generation=1)
    assert warm.status == "ready" and warm.tier == "warm"
    assert warm.manifest["payload_sha256"] == manifest["payload_sha256"]


def test_correction_during_cold_io_rejects_stale_ticket(tmp_path):
    store = make_store(tmp_path, delay=0.01)
    publish(store)
    store.demote("rec-a")
    reader = make_store(tmp_path, delay=0.01)
    pending = reader.read("rec-a", contract=CONTRACT, expected_generation=1, budget_seconds=0.0001)
    assert pending.status == "pending" and pending.ticket_id
    assert pending.payload is None
    reader.invalidate("rec-a", reason="correction", revoked=True)
    resumed = reader.resume(pending.ticket_id)
    assert resumed.status == "rejected"
    assert resumed.payload is None
    assert reader.read("rec-a", contract=CONTRACT).status == "ineligible"


def test_missing_corrupt_and_generation_mismatch_are_explicit(tmp_path):
    store = make_store(tmp_path)
    assert store.read("absent", contract=CONTRACT).status == "unavailable"
    publish(store, "rec-b", b"good-bytes")
    store.demote("rec-b")
    (tmp_path / "cold" / "rec-b" / "payload.bin").write_bytes(b"corrupt")
    corrupt = make_store(tmp_path).read("rec-b", contract=CONTRACT)
    assert corrupt.status == "corrupt" and corrupt.payload is None
    publish(store, "rec-c", b"current")
    mismatch = store.read("rec-c", contract=CONTRACT, expected_generation=7)
    assert mismatch.status == "rejected" and mismatch.payload is None


def test_recovery_quarantines_interrupted_move_and_keeps_healthy_state(tmp_path):
    store = make_store(tmp_path)
    publish(store, "healthy", b"healthy")
    (tmp_path / "staging" / "interrupted").mkdir(parents=True)
    (tmp_path / "staging" / "interrupted" / "payload.bin").write_bytes(b"partial")
    report = make_store(tmp_path).recover()
    assert report["status"] == "degraded"
    assert "interrupted" in report["quarantined_staging"]
    assert report["healthy_records"] == 1
    assert make_store(tmp_path).read("healthy", contract=CONTRACT).status == "ready"


def test_p_state_requires_coherent_publication_before_eviction(tmp_path):
    store = make_store(tmp_path)
    store.publish_p("expert", weights=b"w" * 16, moments=b"m" * 16, clock=4,
                    encoder="encoder-v1", lineage=("ep-1",), dirty=True)
    with pytest.raises(DirtyState):
        store.evict_p("expert")
    store.publish_p("expert", weights=b"w" * 16, moments=b"m" * 16, clock=4,
                    encoder="encoder-v1", lineage=("ep-1",))
    assert store.evict_p("expert") == "warm"
    state = store.read_p("expert")
    assert state["clock"] == 4 and state["weights"] == b"w" * 16
    assert state["moments"] == b"m" * 16
    assert state["lineage"] == ["ep-1"]


def test_projection_cache_identity_includes_eligibility_revision(tmp_path):
    cache = ProjectionCache()
    key = cache.key("rec-a", generation=1, producer="p", representation="r",
                    reader_release="rel", site=3, eligibility_revision=0)
    cache.put(key, "value")
    assert cache.get(key) == "value"
    assert cache.get(cache.key("rec-a", generation=1, producer="p",
                               representation="r", reader_release="rel", site=3,
                               eligibility_revision=1)) is None
    store = make_store(tmp_path)
    store.projections = cache
    store.replay.register("rec-a", generation=1, producer="p")
    store.invalidate("rec-a", reason="correction")
    assert len(cache) == 0
    assert not store.replay.eligible("rec-a", generation=1, producer="p")


def test_canonical_fallback_is_eligible_and_charged(tmp_path):
    store = make_store(tmp_path)
    publish(store, "canonical", b"exact canonical text", kind="canonical",
            producer="host", representation="text", scope_digest="d" * 64)
    result = store.read("absent", contract=CONTRACT, fallback_id="canonical",
                        fallback_contract=ReadContract("host", "text", "d" * 64),
                        fallback_tokens=12)
    assert result.status == "fallback" and result.record_id == "canonical"
    assert result.fallback_tokens == 12
    assert store.metrics_document()["fallback_tokens"] == 12
    store.invalidate("canonical", reason="revoked", revoked=True)
    refused = store.read("absent", contract=CONTRACT, fallback_id="canonical",
                         fallback_contract=ReadContract("host", "text", "d" * 64),
                         fallback_tokens=12)
    assert refused.status == "unavailable"


def test_authority_callback_controls_selection_before_and_after_io(tmp_path):
    store = make_store(tmp_path, delay=0.005)
    publish(store, "rec-a")
    store.demote("rec-a")
    reader = make_store(tmp_path, delay=0.005)

    def ineligible(record_id):
        return {"eligible": False, "reason": "correction"}

    assert reader.read("rec-a", authority=ineligible).status == "ineligible"

    def wrong_generation(record_id):
        return {"eligible": True, "generation": 99, "eligibility_revision": 0}

    assert reader.read("rec-a", authority=wrong_generation,
                       expected_generation=1).status == "rejected"

    def eligible(record_id):
        return {"eligible": True, "generation": 1, "eligibility_revision": 0,
                "producer": "producer-v1", "representation": "rep-v1",
                "scope_digest": "a" * 64, "cutoff": 10 ** 9}

    assert reader.read("rec-a", authority=eligible,
                       expected_generation=1).status == "ready"


def test_capacity_accounting_is_actual_and_not_double_counted(tmp_path):
    limits = TierLimits(hot_bytes=1024, warm_bytes=2048, cold_bytes=4096,
                        identity_reservation_bytes=16,
                        replay_reservation_bytes=16)
    store = make_store(tmp_path, limits=limits)
    store.publish("small", b"x" * 16, generation=1, kind="evidence",
                  producer="p", representation="r", scope_digest="c" * 64)
    usage = store.usage()
    assert usage["hot"]["records"] == 1
    assert usage["warm"]["bytes"] >= 16 + len(
        (tmp_path / "warm" / "small" / "manifest.json").read_text())
    with pytest.raises(CapacityExceeded):
        store.publish("large", b"y" * 10_000, generation=2, kind="evidence",
                      producer="p", representation="r", scope_digest="c" * 64)
        store.usage()
