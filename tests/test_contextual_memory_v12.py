"""v12 durable memory: authority, cold tickets, correction, release and recovery."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import threading
import time

import pytest
import torch

from replica_lab.contextual_memory import ContextualMemoryError
from replica_lab.contextual_memory_v5 import ContextualReadFailure, PendingContextualRead
from replica_lab.contextual_release_v12 import ProducerReleaseCorpus
from replica_lab.contextual_v12_fixture import (
    build_v12_stack, canonical_span, open_v12_store, scope, spans_for)
from replica_lab.tiered_memory import MemoryScope
from v12_fixtures import run_child, step, token_inputs


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make(**kwargs):
    kwargs.setdefault("layers", 36)
    kwargs.setdefault("sites", (3, 7))
    return build_v12_stack(**kwargs)


def source(text="Badge means red.", record_id="record0", *, available=1, valid_until=None,
           epoch=1):
    return canonical_span(text, record_id=record_id, source_epoch=epoch,
                          available_at=available, valid_until=valid_until)[0]


def sc(as_of=1000):
    return scope(as_of=as_of)


def cold_pair(path, store, *, prefix="record"):
    for index in range(2):
        path.admit(store, source(f"Badge shared marker {index}",
                                 record_id=f"{prefix}{index}"), scope=sc(), placement="cold")


def test_scope_time_and_wrong_scope_are_refused_before_consumption(tmp_path):
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    with pytest.raises(ContextualMemoryError):
        path.admit(store, source(available=2000), scope=sc())
    with pytest.raises(ContextualMemoryError):
        path.admit(store, source(valid_until=5), scope=sc(as_of=10))
    snapshot = MemoryScope("user_a", "main", "engineering", 20, mode="source_snapshot",
                           snapshot_at=10)
    with pytest.raises(ContextualMemoryError):
        path.training_values((source(available=15),), scope=snapshot)
    path.admit(store, source(), scope=sc())
    with pytest.raises(ContextualMemoryError):
        path.read(store, "badge", scope=scope("user_b"))
    for bad in (-1, 6, True):
        with pytest.raises(ContextualMemoryError):
            path.read(store, "badge", scope=sc(), wait_seconds=bad)
    store.close()


def test_hot_and_cold_transitions_remain_authoritative(tmp_path):
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    path.admit(store, source(), scope=sc())
    cached = store.cache_record("record0.span0")
    assert cached.ok
    hot = path.read(store, "badge", scope=sc())
    assert len(hot.spans) == 1 and hot.slots.shape == (4, 256)
    assert path.validate(hot, sc())
    migrated = store.migrate_records(["record0.span0"], to_tier="cold")
    assert migrated.ok
    pending = path.read(store, "badge", scope=sc())
    assert isinstance(pending, PendingContextualRead)
    cold = path.resume(store, pending, scope=sc(), wait_seconds=1)
    assert len(cold.spans) == 1 and cold.read_costs["cold_completed_files"] == 5
    store.close()


def test_warm_read_matches_live_training_values_and_stales_after_update(tmp_path):
    adapter, layout, provider, path = make()
    ids, mask, labels = token_inputs()
    store = open_v12_store(tmp_path)
    path.admit(store, source(), scope=sc())
    durable = path.read(store, "badge", scope=sc())
    fresh = path.training_values(durable.spans, scope=sc())
    with torch.no_grad():
        assert torch.equal(path.forward(ids, mask, fresh, scope=sc()),
                           path.forward(ids, mask, durable, scope=sc()))
    step(path, adapter, durable.spans, ids=ids, labels=labels, sc=sc())
    with pytest.raises(ContextualMemoryError, match="stale producer release"):
        path.read(store, "badge", scope=sc())
    relearned = path.training_from_stored(durable, scope=sc())
    assert relearned.training and relearned.slots.requires_grad
    store.close()


def test_cold_pending_resume_costs_and_no_partial_values(tmp_path):
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    cold_pair(path, store)
    pending = path.read(store, "badge shared", scope=sc())
    assert isinstance(pending, PendingContextualRead)
    assert not hasattr(pending, "slots")
    with pytest.raises(ContextualMemoryError):
        path.validate(pending, sc())
    batch = path.resume(store, pending, scope=sc(), wait_seconds=1)
    assert len(batch.spans) == 2 and batch.slots.shape == (8, 256)
    assert batch.read_costs["cold_completed_files"] == 10
    assert batch.read_costs["read_file_calls"] >= 12
    assert batch.read_costs["cold_completed_bytes"] == batch.read_costs["cold_reserved_bytes"]
    path.close()
    store.close()


@pytest.mark.parametrize("mutation", ["correction", "revocation", "corrupt", "missing"])
def test_invalidated_or_unavailable_cold_payload_never_becomes_partial(tmp_path, mutation):
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    cold_pair(path, store)
    pending = path.read(store, "badge", scope=sc())
    if mutation == "correction":
        path.correct(store, "record0.span0",
                     replace(source("Badge changed", "new", available=2), source_epoch=2,
                             supersedes="record0.span0"), scope=sc())
    elif mutation == "revocation":
        store.revoke_record("record0.span0", reason="withdrawn", scope=sc())
    else:
        row = store.connection.execute(
            "SELECT text_path FROM records WHERE record_id=?", ("record0.span0",)).fetchone()
        path_file = store.root / row["text_path"]
        if mutation == "corrupt":
            path_file.write_text("damaged")
        else:
            path_file.unlink()
    with pytest.raises(ContextualReadFailure) as error:
        path.resume(store, pending, scope=sc(), wait_seconds=1)
    assert error.value.costs["cold_reserved_bytes"] > 0
    assert all(name != "record0.span0" for name, _ in provider._cache.values())
    path.close()
    store.close()


@pytest.mark.parametrize("change", ["correction", "revocation"])
def test_bounded_wait_and_authority_change_during_actual_io(tmp_path, monkeypatch, change):
    import replica_lab.contextual_memory_v12 as module
    original = module._cold_payloads
    started, release = threading.Event(), threading.Event()

    def delayed(*args):
        started.set()
        release.wait(5)
        return original(*args)

    monkeypatch.setattr(module, "_cold_payloads", delayed)
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    cold_pair(path, store)
    pending = path.read(store, "badge", scope=sc())
    began = time.monotonic()
    pending = path.resume(store, pending, scope=sc(), wait_seconds=0.01)
    assert isinstance(pending, PendingContextualRead)
    assert time.monotonic() - began < 1.0 and started.is_set()
    if change == "revocation":
        store.revoke_record("record1.span0", reason="withdrawn_during_io", scope=sc())
    else:
        path.correct(store, "record1.span0",
                     replace(source("Badge changed", "new", available=2), source_epoch=2,
                             supersedes="record1.span0"), scope=sc())
    release.set()
    with pytest.raises(ContextualReadFailure):
        path.resume(store, pending, scope=sc(), wait_seconds=1)
    path.close()
    store.close()


def test_pending_ticket_continues_in_a_fresh_process(tmp_path):
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    cold_pair(path, store)
    pending = path.read(store, "badge shared", scope=sc())
    path.close()
    store.close()
    (tmp_path / "pending.json").write_text(json.dumps(asdict(pending)))
    code = '''
import json, sys, torch
from pathlib import Path
torch.set_num_threads(1)
from replica_lab.contextual_memory_v5 import PendingContextualRead
from replica_lab.contextual_v12_fixture import build_v12_stack, open_v12_store, scope
root = Path(sys.argv[1])
path = build_v12_stack(layers=36, sites=(3, 7))[3]
store = open_v12_store(root)
pending = PendingContextualRead(**json.loads((root / "pending.json").read_text()))
batch = path.resume(store, pending, scope=scope(), wait_seconds=1)
assert len(batch.spans) == 2 and batch.slots.shape == (8, 256)
root.joinpath("child_ok.json").write_text("{}")
path.close(); store.close()
'''
    result = run_child(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "child_ok.json").is_file()


def test_recover_uncertain_dispatch_keeps_reservation_and_bounds_retries(tmp_path):
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    cold_pair(path, store)
    pending = path.read(store, "badge", scope=sc())
    descriptor_path = path._ticket_path(store, pending.ticket_id)
    doc = json.loads(descriptor_path.read_text())
    required = sum(t["required_bytes"] for t in doc["tickets"])
    doc["attempts"] = 1
    doc["costs"]["cold_reserved_bytes"] = required
    doc["costs"]["cold_reserved_files"] = 10
    path._persist(store, pending.ticket_id, doc)
    path.close()
    store.close()
    store = open_v12_store(tmp_path)
    recovered_path = make()[3]
    recovered = recovered_path.recover_pending(store, pending.ticket_id, scope=sc())
    batch = recovered_path.resume(store, recovered, scope=sc(), wait_seconds=1)
    assert batch.read_costs["cold_reserved_bytes"] == 2 * required
    assert batch.read_costs["cold_completed_bytes"] == required
    assert batch.read_costs["ticket_write_bytes_reserved"] > 0
    with pytest.raises(ContextualReadFailure):
        recovered_path.recover_pending(store, pending.ticket_id, scope=sc())
    recovered_path.close()
    store.close()


def test_quarantined_record_is_unavailable_not_partial(tmp_path):
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    path.admit(store, source(), scope=sc())
    with store._transaction():
        store.connection.execute(
            "UPDATE records SET status='quarantined' WHERE record_id=?", ("record0.span0",))
    with pytest.raises(ContextualReadFailure):
        path.read(store, "badge", scope=sc())
    store.close()


def test_release_publication_roundtrip_stale_producer_and_reload(tmp_path):
    adapter, layout, provider, path = make()
    ids, mask, labels = token_inputs()
    store = open_v12_store(tmp_path)
    path.admit(store, source(), scope=sc())
    corpus = ProducerReleaseCorpus(store, sc())
    published = corpus.rebuild(path)
    assert published["generation"] == 1
    path.release_corpus = corpus
    first = path.read(store, "badge", scope=sc(), wait_seconds=1)
    durable = path.read(store, "badge", scope=sc(), wait_seconds=1)
    assert torch.equal(first.slots, durable.slots)
    step(path, adapter, first.spans, ids=ids, labels=labels, sc=sc())
    step(path, adapter, first.spans, ids=ids, labels=labels, sc=sc())
    with pytest.raises(ContextualMemoryError):
        path.read(store, "badge", scope=sc())
    with pytest.raises(ContextualMemoryError):
        path.validate(first, sc())
    second = corpus.rebuild(path)
    assert second["generation"] > published["generation"]
    assert (corpus.root / "00000001" / "manifest.json").is_file()
    after = path.read(store, "badge", scope=sc(), wait_seconds=1)
    assert not torch.equal(first.slots, after.slots)
    path.save(tmp_path / "checkpoint")
    path.close()
    store.close()
    store = open_v12_store(tmp_path)
    reloaded = make()[3]
    reloaded.load(tmp_path / "checkpoint")
    reloaded.release_corpus = ProducerReleaseCorpus(store, sc())
    restored = reloaded.read(store, "badge", scope=sc(), wait_seconds=1)
    assert torch.equal(after.slots, restored.slots)
    reloaded.close()
    store.close()


def test_release_interrupted_publication_missing_and_mixed_authority(tmp_path):
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    path.admit(store, source(), scope=sc())
    corpus = ProducerReleaseCorpus(store, sc())

    def crash():
        raise RuntimeError("interrupted before publication")

    with pytest.raises(RuntimeError):
        corpus.rebuild(path, before_publish=crash)
    assert ProducerReleaseCorpus(store, sc())._current() is None
    manifest = corpus.rebuild(path)
    assert manifest["generation"] == 2
    manifest_path = corpus.root / "00000002" / "manifest.json"
    document = json.loads(manifest_path.read_text())
    document["records"]["record0.span0"]["producer_release"] = "wrong"
    document["record_digest"] = hashlib.sha256(json.dumps(
        document["records"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest_path.write_text(json.dumps(document))
    pointer = corpus.root / "current.json"
    record = json.loads(pointer.read_text())
    record["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    pointer.write_text(json.dumps(record))
    with pytest.raises(ContextualMemoryError, match="mixed"):
        corpus._current()
    manifest_path.unlink()
    with pytest.raises(ContextualMemoryError):
        corpus._current()
    store.close()


def test_first_release_interruption_and_lost_pointer_fail_closed(tmp_path):
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    path.admit(store, source(), scope=sc())
    corpus = ProducerReleaseCorpus(store, sc())

    def crash():
        raise RuntimeError("interrupted before publication")

    with pytest.raises(RuntimeError):
        corpus.rebuild(path, before_publish=crash)
    corpus = ProducerReleaseCorpus(store, sc())
    assert corpus._current() is None
    assert corpus.rebuild(path)["generation"] == 2
    (corpus.root / "current.json").unlink()
    with pytest.raises(ContextualMemoryError):
        corpus._current()
    with pytest.raises(ContextualMemoryError):
        ProducerReleaseCorpus(store, sc())
    store.close()


def test_release_publication_rejects_corrected_source_but_keeps_generation(tmp_path):
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    path.admit(store, source(), scope=sc())
    corpus = ProducerReleaseCorpus(store, sc())
    corpus.rebuild(path)

    def correct():
        path.correct(store, "record0.span0",
                     replace(source("Badge blue", "new", available=2), source_epoch=2,
                             supersedes="record0.span0"), scope=sc())

    with pytest.raises(ContextualMemoryError, match="invalidated"):
        corpus.rebuild(path, before_publish=correct)
    assert corpus._current()[1]["generation"] == 1
    rebuilt = corpus.rebuild(path)
    assert set(rebuilt["records"]) == {"new.span0"}
    store.close()


def test_release_pending_invalidated_by_publication_or_revocation(tmp_path):
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    path.admit(store, source(), scope=sc())
    corpus = ProducerReleaseCorpus(store, sc())
    corpus.rebuild(path)
    path.release_corpus = corpus
    pending = path.read(store, "badge", scope=sc())
    assert isinstance(pending, PendingContextualRead)
    corpus.rebuild(path)
    with pytest.raises(ContextualReadFailure, match="publication"):
        path.resume(store, pending, scope=sc(), wait_seconds=1)
    pending = path.read(store, "badge", scope=sc())
    store.revoke_record("record0.span0", reason="withdrawn", scope=sc())
    with pytest.raises(ContextualReadFailure):
        path.resume(store, pending, scope=sc(), wait_seconds=1)
    store.close()


def test_release_payload_corruption_refuses_only_the_damaged_record(tmp_path):
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    left = source("alpha marker one", "alpha")
    right = source("beta marker two", "beta")
    path.admit(store, left, scope=sc())
    path.admit(store, right, scope=sc())
    corpus = ProducerReleaseCorpus(store, sc())
    corpus.rebuild(path)
    manifest = corpus._current()[1]
    entry = manifest["records"]["alpha.span0"]
    damaged = corpus.root / f"{manifest['generation']:08d}" / entry["value_file"]
    payload = bytearray(damaged.read_bytes())
    payload[-1] ^= 1
    damaged.write_bytes(bytes(payload))
    with pytest.raises(ContextualMemoryError, match="digest"):
        corpus.values(entry["source_pin"], left, version=manifest["producer_release"],
                      scope=sc(), source_identity=path.source_identity)
    beta_entry = manifest["records"]["beta.span0"]
    values, release_pin = corpus.values(
        beta_entry["source_pin"], right, version=manifest["producer_release"],
        scope=sc(), source_identity=path.source_identity)
    assert values.shape == (4, 256) and release_pin["producer_generation"] == manifest["generation"]
    store.close()


@pytest.mark.parametrize("damage", ["representation", "codec", "producer", "scope"])
def test_stale_batch_identity_is_refused_before_consumption(tmp_path, damage):
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    path.admit(store, source(), scope=sc())
    batch = path.read(store, "badge", scope=sc())
    mutated = {
        "representation": replace(batch, representation="other"),
        "codec": replace(batch, codec="other"),
        "producer": replace(batch, producer_version="v12-stale"),
        "scope": replace(batch, scope_key="sc_stale"),
    }[damage]
    with pytest.raises(ContextualMemoryError):
        path.validate(mutated, sc())
    store.close()


def test_correction_evicts_only_its_feature_cache_and_keeps_healthy_records(tmp_path):
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    left, right = source("Badge red", "left"), source("Badge blue", "right")
    path.admit(store, left, scope=sc())
    path.admit(store, right, scope=sc())
    provider.features(left)
    provider.features(right)
    assert provider.cost["cache_bytes"] > 0
    path.correct(store, "left.span0",
                 replace(source("Badge green", "new", available=2), source_epoch=2,
                         supersedes="left.span0"), scope=sc())
    cached_names = {name for name, _ in provider._cache.values()}
    assert "left.span0" not in cached_names
    assert "right.span0" in cached_names
    assert provider.cost["cache_bytes"] > 0
    batch = path.read(store, "Badge", scope=sc())
    assert {span.record_id for span in batch.spans} == {"new.span0", "right.span0"}
    store.close()


def test_revocation_after_validate_is_refused_and_evicts_cache(tmp_path):
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    path.admit(store, source(), scope=sc())
    batch = path.read(store, "badge", scope=sc())
    provider.features(batch.spans[0])
    store.revoke_record("record0.span0", reason="withdrawn", scope=sc())
    with pytest.raises(ContextualMemoryError):
        path.validate(batch, sc())
    assert all(name != "record0.span0" for name, _ in provider._cache.values())
    with pytest.raises(ContextualMemoryError):
        path.forward(*token_inputs()[:2], batch, scope=sc())
    store.close()


def test_other_v12_feature_contract_batch_is_refused(tmp_path):
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    path.admit(store, source(), scope=sc())
    batch = path.read(store, "badge", scope=sc())
    other = make(tokenizer_mode="words")[3]
    with pytest.raises(ContextualMemoryError):
        other.validate(batch, sc())
    store.close()


def test_close_and_source_free_clear_bounded_caches(tmp_path):
    adapter, layout, provider, path = make()
    span = source()
    provider.features(span)
    assert provider.cost["cache_bytes"] > 0
    path.clear_source_caches()
    assert provider.cost["cache_bytes"] == 0 and provider._cache == {}
    provider.features(span)
    path.close()
    assert provider.cost["cache_bytes"] == 0 and path._pool is None


def test_mixed_release_evicts_only_the_stale_record(tmp_path):
    adapter, layout, provider, path = make()
    ids, mask, labels = token_inputs()
    sc = scope()
    store = open_v12_store(tmp_path)
    left = source("badge red", "a")
    right = source("badge blue", "b")
    path.admit(store, left, scope=sc)
    step(path, adapter, (left,), ids=ids, labels=labels, sc=sc)
    path.admit(store, right, scope=sc)
    provider.features(left)
    provider.features(right)
    with pytest.raises(ContextualMemoryError):
        path.read(store, "badge", scope=sc)
    names = {name for name, _ in provider._cache.values()}
    assert "a.span0" not in names
    assert "b.span0" in names
    assert provider.cost["cache_bytes"] > 0
    store.close()


def test_release_source_execution_identity_is_enforced(tmp_path):
    adapter, layout, provider, path = make()
    sc = scope()
    store = open_v12_store(tmp_path)
    span = source("badge red", "a")
    path.admit(store, span, scope=sc)
    corpus = ProducerReleaseCorpus(store, sc)
    corpus.rebuild(path)
    batch = path.read(store, "badge", scope=sc, wait_seconds=1)
    other = make(source_identity="different_source")[3]
    other.release_corpus = ProducerReleaseCorpus(store, sc)
    with pytest.raises(ContextualMemoryError):
        other.validate(batch, sc)
    with pytest.raises(ContextualMemoryError):
        other.release_corpus.read(other, "badge", scope=sc, wait_seconds=1)
    manifest = corpus._current()[1]
    entry = manifest["records"]["a.span0"]
    with pytest.raises(ContextualMemoryError):
        corpus.values(entry["source_pin"], span, version=manifest["producer_release"],
                      scope=sc, source_identity="different_source")
    other.close()
    path.close()
    store.close()


def test_two_record_live_warm_cold_release_order_and_parity(tmp_path):
    adapter, layout, provider, path = make()
    ids, mask, labels = token_inputs()
    sc = scope()
    left = source("badge red", "a", available=1, epoch=1)
    right = source("badge blue", "b", available=2, epoch=2)
    store = open_v12_store(tmp_path)
    path.admit(store, left, scope=sc)
    path.admit(store, right, scope=sc)
    live = path.training_values((left, right), scope=sc)
    warm = path.read(store, "badge", scope=sc)
    assert [span.record_id for span in warm.spans] == ["a.span0", "b.span0"]
    assert torch.equal(live.slots, warm.slots)
    with torch.no_grad():
        assert torch.equal(path.forward(ids, mask, live, scope=sc),
                           path.forward(ids, mask, warm, scope=sc))

    cold_store = open_v12_store(tmp_path, name="cold_parity")
    path.admit(cold_store, left, scope=sc, placement="cold")
    path.admit(cold_store, right, scope=sc, placement="cold")
    pending = path.read(cold_store, "badge", scope=sc)
    cold = path.resume(cold_store, pending, scope=sc, wait_seconds=1)
    assert [span.record_id for span in cold.spans] == ["a.span0", "b.span0"]
    assert torch.equal(live.slots, cold.slots)
    with torch.no_grad():
        assert torch.equal(path.forward(ids, mask, live, scope=sc),
                           path.forward(ids, mask, cold, scope=sc))

    corpus = ProducerReleaseCorpus(store, sc)
    corpus.rebuild(path)
    path.release_corpus = corpus
    released = path.read(store, "badge", scope=sc, wait_seconds=1)
    assert [span.record_id for span in released.spans] == ["a.span0", "b.span0"]
    assert torch.equal(live.slots, released.slots)
    with torch.no_grad():
        assert torch.equal(path.forward(ids, mask, live, scope=sc),
                           path.forward(ids, mask, released, scope=sc))
    path.release_corpus = None
    cold_store.close()
    store.close()


def test_durable_slot_order_ties_are_deterministic(tmp_path):
    adapter, layout, provider, path = make()
    sc = scope()
    store = open_v12_store(tmp_path)
    path.admit(store, source("badge alpha", "z"), scope=sc)
    path.admit(store, source("badge beta", "a"), scope=sc)
    first = path.read(store, "badge", scope=sc)
    second = path.read(store, "badge", scope=sc)
    assert [span.record_id for span in first.spans] == ["a.span0", "z.span0"]
    assert [span.record_id for span in second.spans] == ["a.span0", "z.span0"]
    assert torch.equal(first.slots, second.slots)
    store.close()


def test_release_write_failure_releases_charge_and_retry_succeeds(tmp_path, monkeypatch):
    from pathlib import Path as PathClass
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    path.admit(store, source("badge red", "a"), scope=sc())
    corpus = ProducerReleaseCorpus(store, sc())
    original = PathClass.open
    state = {"fail": True}

    def failing(self, *args, **kwargs):
        if state["fail"] and self.suffix == ".slots":
            raise OSError("simulated release write failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PathClass, "open", failing)
    before = corpus.cost["write_bytes"]
    with pytest.raises(OSError):
        corpus.rebuild(path)
    assert corpus.cost["write_bytes"] == before
    charged = store.connection.execute(
        "SELECT COUNT(*) AS c FROM tier_usage WHERE component_id LIKE 'release_v12:%'"
    ).fetchone()["c"]
    assert charged == 0
    state["fail"] = False
    published = corpus.rebuild(path)
    assert published["generation"] == 2
    assert corpus.cost["write_bytes"] > 0
    store.close()


def test_semantic_order_beats_record_id_and_reversed_caller_order(tmp_path):
    adapter, layout, provider, path = make()
    ids, mask, labels = token_inputs()
    sc = scope()
    early = source("badge alpha", "z", available=1, epoch=1)
    late = source("badge beta", "a", available=2, epoch=2)
    store = open_v12_store(tmp_path)
    path.admit(store, late, scope=sc)
    path.admit(store, early, scope=sc)
    durable = path.read(store, "badge", scope=sc)
    assert [span.record_id for span in durable.spans] == ["z.span0", "a.span0"]
    live = path.training_values((early, late), scope=sc)
    assert torch.equal(live.slots, durable.slots)
    reversed_live = path.training_values((late, early), scope=sc)
    assert not torch.equal(reversed_live.slots, durable.slots)
    with torch.no_grad():
        assert torch.equal(path.forward(ids, mask, live, scope=sc),
                           path.forward(ids, mask, durable, scope=sc))
    store.close()


def test_post_correction_live_and_durable_parity(tmp_path):
    adapter, layout, provider, path = make()
    ids, mask, labels = token_inputs()
    sc = scope()
    store = open_v12_store(tmp_path)
    first = source("badge alpha", "a", available=1, epoch=1)
    second = source("badge blue", "b", available=2, epoch=2)
    path.admit(store, first, scope=sc)
    path.admit(store, second, scope=sc)
    corrected = replace(source("badge green", "new", available=3, epoch=3),
                        supersedes="a.span0")
    path.correct(store, "a.span0", corrected, scope=sc)
    durable = path.read(store, "badge", scope=sc)
    assert [span.record_id for span in durable.spans] == ["b.span0", "new.span0"]
    live = path.training_values((second, corrected), scope=sc)
    assert torch.equal(live.slots, durable.slots)
    with torch.no_grad():
        assert torch.equal(path.forward(ids, mask, live, scope=sc),
                           path.forward(ids, mask, durable, scope=sc))
    store.close()


def test_release_pointer_write_failure_keeps_previous_generation(tmp_path, monkeypatch):
    import replica_lab.contextual_release_v12 as module
    adapter, layout, provider, path = make()
    sc = scope()
    store = open_v12_store(tmp_path)
    path.admit(store, source("badge red", "a"), scope=sc)
    corpus = ProducerReleaseCorpus(store, sc)
    first = corpus.rebuild(path)
    pointer_path = corpus.root / "current.json"
    before = pointer_path.read_bytes()
    component = "release_pointer:" + sc.scope_key
    old_size = store.connection.execute(
        "SELECT bytes FROM tier_usage WHERE component_id=?", (component,)).fetchone()["bytes"]
    original = module.atomic_json

    def failing(path_value, value):
        if path_value.name.startswith("current.json.pending"):
            raise OSError("simulated pointer staging failure")
        return original(path_value, value)

    monkeypatch.setattr(module, "atomic_json", failing)
    with pytest.raises(OSError):
        corpus.rebuild(path)
    current = corpus._current()
    assert current is not None and current[1]["generation"] == first["generation"]
    assert pointer_path.read_bytes() == before
    assert store.connection.execute(
        "SELECT bytes FROM tier_usage WHERE component_id=?", (component,)
    ).fetchone()["bytes"] == old_size
    assert corpus.cost["publications"] == 1
    store.close()


def test_release_manifest_write_failure_keeps_previous_generation(tmp_path, monkeypatch):
    from pathlib import Path as PathClass
    adapter, layout, provider, path = make()
    sc = scope()
    store = open_v12_store(tmp_path)
    path.admit(store, source("badge red", "a"), scope=sc)
    corpus = ProducerReleaseCorpus(store, sc)
    first = corpus.rebuild(path)
    original = PathClass.open
    state = {"fail": True}

    def failing(self, *args, **kwargs):
        if state["fail"] and self.name == "manifest.json" and self.parent.name.isdigit():
            raise OSError("simulated release manifest failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PathClass, "open", failing)
    with pytest.raises(OSError):
        corpus.rebuild(path)
    state["fail"] = False
    current = corpus._current()
    assert current is not None and current[1]["generation"] == first["generation"]
    assert corpus.cost["publications"] == 1
    store.close()


def test_invalidated_release_evicts_only_its_selection(tmp_path):
    adapter, layout, provider, path = make()
    sc = scope()
    store = open_v12_store(tmp_path)
    first = source("badge alpha", "a", available=1, epoch=1)
    second = source("badge beta", "b", available=2, epoch=2)
    path.admit(store, first, scope=sc)
    path.admit(store, second, scope=sc)
    unrelated = source("unrelated note", "u")
    path.admit(store, unrelated, scope=scope("user_b"))
    corpus = ProducerReleaseCorpus(store, sc)
    corpus.rebuild(path)
    path.release_corpus = corpus
    provider.features(first)
    provider.features(second)
    provider.features(unrelated)
    pending = path.read(store, "badge", scope=sc)
    assert isinstance(pending, PendingContextualRead)
    store.revoke_record("a.span0", reason="rereview", scope=sc)
    with pytest.raises(ContextualReadFailure):
        path.resume(store, pending, scope=sc, wait_seconds=1)
    names = {name for name, _ in provider._cache.values()}
    assert "a.span0" not in names and "b.span0" not in names
    assert "u.span0" in names
    path.release_corpus = None
    store.close()


def test_cold_staging_propagates_process_control_exceptions(tmp_path, monkeypatch):
    import replica_lab.contextual_memory_v12 as module

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(module, "_payload_read_verified", interrupted)
    ticket = {"pinned": [{"record_id": "r", "text_path": "missing",
                          "text_sha256": "0" * 64, "text_bytes": 1, "values": []}]}
    with pytest.raises(KeyboardInterrupt):
        module._cold_payloads(tmp_path, [ticket])


def test_cold_failure_is_typed_and_charged(tmp_path):
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    for index in range(2):
        path.admit(store, source(f"badge marker {index}", f"r{index}"),
                   scope=sc(), placement="cold")
    pending = path.read(store, "badge", scope=sc())
    row = store.connection.execute(
        "SELECT text_path FROM records WHERE record_id=?", ("r0.span0",)).fetchone()
    (store.root / row["text_path"]).write_text("damaged")
    with pytest.raises(ContextualReadFailure) as error:
        path.resume(store, pending, scope=sc(), wait_seconds=1)
    assert error.value.costs["cold_completed_files"] >= 5
    descriptor = json.loads(path._ticket_path(store, pending.ticket_id).read_text())
    assert descriptor["status"] == "failed"
    assert "r0.span0" in descriptor["failed_payload_records"]
    names = {name for name, _ in provider._cache.values()}
    assert "r0.span0" not in names and "r1.span0" in names
    store.close()


def test_release_accounting_failure_does_not_publish(tmp_path, monkeypatch):
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    path.admit(store, source("badge red", "a"), scope=sc())
    corpus = ProducerReleaseCorpus(store, sc())
    first = corpus.rebuild(path)
    pointer_path = corpus.root / "current.json"
    before = pointer_path.read_bytes()
    original = store._set_component_locked

    def failing(component_id, *args, **kwargs):
        if component_id.startswith("release_pointer:"):
            raise OSError("simulated release accounting failure")
        return original(component_id, *args, **kwargs)

    monkeypatch.setattr(store, "_set_component_locked", failing)
    with pytest.raises(OSError):
        corpus.rebuild(path)
    assert pointer_path.read_bytes() == before
    current = corpus._current()
    assert current is not None and current[1]["generation"] == first["generation"]
    assert corpus.cost["publications"] == 1
    store.close()


def test_release_post_commit_failure_does_not_publish(tmp_path, monkeypatch):
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    path.admit(store, source("badge red", "a"), scope=sc())
    corpus = ProducerReleaseCorpus(store, sc())
    first = corpus.rebuild(path)
    pointer_path = corpus.root / "current.json"
    before = pointer_path.read_bytes()

    def failing():
        raise RuntimeError("simulated post-commit cache failure")

    monkeypatch.setattr(provider, "clear", failing)
    with pytest.raises(RuntimeError):
        corpus.rebuild(path)
    assert pointer_path.read_bytes() == before
    current = corpus._current()
    assert current is not None and current[1]["generation"] == first["generation"]
    store.close()


def test_release_pointer_is_published_after_store_accounting(tmp_path, monkeypatch):
    import replica_lab.contextual_release_v12 as module
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    path.admit(store, source("badge red", "a"), scope=sc())
    corpus = ProducerReleaseCorpus(store, sc())
    calls = []
    original_component = store._set_component_locked
    original_publish = corpus._publish_pointer

    def component(component_id, *args, **kwargs):
        if component_id.startswith("release_pointer:"):
            calls.append("accounting")
        return original_component(component_id, *args, **kwargs)

    def publish(pending, previous):
        calls.append("publish")
        return original_publish(pending, previous)

    monkeypatch.setattr(store, "_set_component_locked", component)
    monkeypatch.setattr(corpus, "_publish_pointer", publish)
    corpus.rebuild(path)
    assert calls == ["accounting", "publish"]
    store.close()


def test_release_read_refuses_different_feature_contract(tmp_path):
    adapter, layout, provider, path = make()
    sc = scope()
    store = open_v12_store(tmp_path)
    path.admit(store, source("badge red", "a"), scope=sc)
    corpus = ProducerReleaseCorpus(store, sc)
    corpus.rebuild(path)
    other = make(tokenizer_mode="words", source_identity=path.source_identity)[3]
    other.release_corpus = ProducerReleaseCorpus(store, sc)
    with pytest.raises(ContextualMemoryError):
        other.read(store, "badge", scope=sc, wait_seconds=1)
    other.close()
    path.close()
    store.close()


def test_release_post_replace_fsync_failure_restores_generation_and_charge(tmp_path, monkeypatch):
    adapter, layout, provider, path = make()
    sc = scope()
    store = open_v12_store(tmp_path)
    path.admit(store, source("badge red", "a"), scope=sc)
    corpus = ProducerReleaseCorpus(store, sc)
    first = corpus.rebuild(path)
    pointer_path = corpus.root / "current.json"
    before = pointer_path.read_bytes()
    component = "release_pointer:" + sc.scope_key
    old_size = store.connection.execute(
        "SELECT bytes FROM tier_usage WHERE component_id=?", (component,)).fetchone()["bytes"]

    def failing(directory):
        raise OSError("simulated post-replace directory fsync failure")

    monkeypatch.setattr(corpus, "_fsync_directory", failing)
    with pytest.raises(OSError):
        corpus.rebuild(path)
    assert pointer_path.read_bytes() == before
    current = corpus._current()
    assert current is not None and current[1]["generation"] == first["generation"]
    assert store.connection.execute(
        "SELECT bytes FROM tier_usage WHERE component_id=?", (component,)
    ).fetchone()["bytes"] == old_size
    store.close()


def test_release_pointer_replace_failure_rolls_back_charge(tmp_path, monkeypatch):
    import replica_lab.contextual_release_v12 as module
    adapter, layout, provider, path = make()
    sc = scope()
    store = open_v12_store(tmp_path)
    path.admit(store, source("badge red", "a"), scope=sc)
    corpus = ProducerReleaseCorpus(store, sc)
    first = corpus.rebuild(path)
    pointer_path = corpus.root / "current.json"
    before = pointer_path.read_bytes()
    component = "release_pointer:" + sc.scope_key
    old_size = store.connection.execute(
        "SELECT bytes FROM tier_usage WHERE component_id=?", (component,)).fetchone()["bytes"]
    original = module.os.replace

    def failing(source, destination):
        if Path(destination).name == "current.json":
            raise OSError("simulated pointer replace failure")
        return original(source, destination)

    monkeypatch.setattr(module.os, "replace", failing)
    with pytest.raises(OSError):
        corpus.rebuild(path)
    assert pointer_path.read_bytes() == before
    current = corpus._current()
    assert current is not None and current[1]["generation"] == first["generation"]
    assert store.connection.execute(
        "SELECT bytes FROM tier_usage WHERE component_id=?", (component,)
    ).fetchone()["bytes"] == old_size
    store.close()


def test_release_direct_reads_are_accounted(tmp_path):
    class Recorder:
        def __init__(self):
            self.reserved, self.completed = [], []

        def reserve_read(self, operation, byte_count, file_count=1):
            self.reserved.append((operation, byte_count, file_count))
            return len(self.reserved) - 1

        def complete_read(self, token, *, actual_bytes, status="completed"):
            self.completed.append((token, actual_bytes, status))

    adapter, layout, provider, path = make()
    sc = scope()
    store = open_v12_store(tmp_path)
    span = source("badge red", "a")
    path.admit(store, span, scope=sc)
    recorder = Recorder()
    corpus = ProducerReleaseCorpus(store, sc, read_accounting=recorder)
    published = corpus.rebuild(path)
    assert corpus.cost["artifact_read_bytes"] > 0
    assert corpus.cost["artifact_read_files"] > 0
    assert any(operation == "release_source" for operation, _, _ in recorder.reserved)
    before = corpus.cost["artifact_read_bytes"]
    manifest = corpus._current()[1]
    entry = manifest["records"]["a.span0"]
    values, _ = corpus.values(entry["source_pin"], span, version=manifest["producer_release"],
                              scope=sc, source_identity=path.source_identity)
    assert values.shape == (4, 256)
    assert corpus.cost["artifact_read_bytes"] > before
    assert any(operation == "release_payload" for operation, _, _ in recorder.reserved)
    assert any(operation == "release_authority" for operation, _, _ in recorder.reserved)
    assert all(status == "completed" for _, _, status in recorder.completed)
    store.close()


def test_work_ledger_release_read_accounting_maps_artifact_bytes(tmp_path):
    from replica_lab.contextual_release_v12 import WorkLedgerReleaseReadAccounting

    class Ledger:
        def __init__(self):
            self.reserved, self.finished = [], []

        def reserve(self, stage, operation, **amounts):
            self.reserved.append((stage, operation, dict(amounts)))
            return len(self.reserved) - 1

        def finish(self, token, *, status="completed", actual=None, detail=None):
            self.finished.append((token, status, dict(actual or {})))

    adapter, layout, provider, path = make()
    sc = scope()
    store = open_v12_store(tmp_path)
    path.admit(store, source("badge red", "a"), scope=sc)
    ledger = Ledger()
    corpus = ProducerReleaseCorpus(
        store, sc, read_accounting=WorkLedgerReleaseReadAccounting(ledger, "mechanics"))
    corpus.rebuild(path)
    read_reservations = [amount for _, _, amount in ledger.reserved
                         if "artifact_read_bytes" in amount]
    assert read_reservations and all(
        amount["artifact_read_bytes"] >= 0 for amount in read_reservations)
    assert any("writes" in amount for _, _, amount in ledger.reserved)
    assert ledger.finished and all(status == "completed" for _, status, _ in ledger.finished)
    store.close()


def test_eviction_process_control_exception_propagates(tmp_path, monkeypatch):
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    path.admit(store, source("badge red", "a"), scope=sc(), placement="cold")
    pending = path.read(store, "badge", scope=sc())
    store.revoke_record("a.span0", reason="rereview", scope=sc())

    def interrupted(store_value, doc):
        raise KeyboardInterrupt()

    monkeypatch.setattr(path, "_evict_failed_records", interrupted)
    with pytest.raises(KeyboardInterrupt):
        path.resume(store, pending, scope=sc(), wait_seconds=1)
    store.close()


def test_persist_failure_preserves_typed_failure_and_terminal_receipt(tmp_path, monkeypatch):
    adapter, layout, provider, path = make()
    store = open_v12_store(tmp_path)
    path.admit(store, source("badge red", "a"), scope=sc(), placement="cold")
    pending = path.read(store, "badge", scope=sc())
    store.revoke_record("a.span0", reason="rereview", scope=sc())

    original_persist = path._persist

    def failing(store_value, token, doc):
        if doc.get("status") == "failed":
            raise RuntimeError("simulated charged persistence failure")
        return original_persist(store_value, token, doc)

    monkeypatch.setattr(path, "_persist", failing)
    with pytest.raises(ContextualReadFailure):
        path.resume(store, pending, scope=sc(), wait_seconds=1)
    descriptor = json.loads(path._ticket_path(store, pending.ticket_id).read_text())
    assert descriptor["status"] == "failed"
    store.close()


def test_release_restore_pointer_failure_removes_authority(tmp_path, monkeypatch):
    adapter, layout, provider, path = make()
    sc = scope()
    store = open_v12_store(tmp_path)
    path.admit(store, source("badge red", "a"), scope=sc)
    corpus = ProducerReleaseCorpus(store, sc)
    first = corpus.rebuild(path)
    original_fsync = corpus._fsync_directory
    state = {"calls": 0}

    def failing_fsync(directory):
        state["calls"] += 1
        if state["calls"] == 1:
            raise OSError("simulated post-replace fsync failure")
        return original_fsync(directory)

    def failing_restore(previous):
        raise OSError("simulated pointer restore failure")

    monkeypatch.setattr(corpus, "_fsync_directory", failing_fsync)
    monkeypatch.setattr(corpus, "_restore_pointer", failing_restore)
    with pytest.raises(ContextualMemoryError, match="authority removed"):
        corpus.rebuild(path)
    assert not (corpus.root / "current.json").exists()
    assert corpus.cost["publications"] == 1
    store.close()


def test_release_charge_rollback_failure_is_typed(tmp_path, monkeypatch):
    adapter, layout, provider, path = make()
    sc = scope()
    store = open_v12_store(tmp_path)
    path.admit(store, source("badge red", "a"), scope=sc)
    corpus = ProducerReleaseCorpus(store, sc)
    first = corpus.rebuild(path)
    pointer_path = corpus.root / "current.json"
    before = pointer_path.read_bytes()

    def failing_fsync(directory):
        raise OSError("simulated post-replace fsync failure")

    def failing_charge(previous):
        raise RuntimeError("simulated pointer charge rollback failure")

    monkeypatch.setattr(corpus, "_fsync_directory", failing_fsync)
    monkeypatch.setattr(corpus, "_revert_pointer_charge", failing_charge)
    with pytest.raises(ContextualMemoryError, match="charge rollback failed") as error:
        corpus.rebuild(path)
    assert isinstance(error.value.__cause__, OSError)
    assert pointer_path.read_bytes() == before
    assert corpus._current()[1]["generation"] == first["generation"]
    store.close()


def test_release_origin_creation_refunds_charge_on_failure(tmp_path, monkeypatch):
    import replica_lab.contextual_release_v12 as module
    adapter, layout, provider, path = make()
    sc = scope()
    store = open_v12_store(tmp_path)
    state = {"fail": True}
    original = module.atomic_json

    def failing(path_value, value):
        if state["fail"] and path_value.name == "current.json":
            raise OSError("simulated release origin failure")
        return original(path_value, value)

    monkeypatch.setattr(module, "atomic_json", failing)
    with pytest.raises(OSError):
        ProducerReleaseCorpus(store, sc)
    assert not (store.root / "artifacts" / "contextual_v12" / "releases"
                / sc.scope_key / "current.json").exists()
    charged = store.connection.execute(
        "SELECT COUNT(*) AS c FROM tier_usage WHERE component_id=?",
        ("release_pointer:" + sc.scope_key,)).fetchone()["c"]
    assert charged == 0
    state["fail"] = False
    corpus = ProducerReleaseCorpus(store, sc)
    assert corpus._current() is None
    assert (corpus.root / "current.json").is_file()
    store.close()


@pytest.mark.parametrize("surface", ["temp_open", "file_fsync", "replace", "parent_fsync"])
def test_release_origin_real_atomic_failure_surfaces(tmp_path, monkeypatch, surface):
    import replica_lab.contextual_release_v12 as module
    adapter, layout, provider, path = make()
    sc = scope()
    store = open_v12_store(tmp_path)
    root = store.root / "artifacts" / "contextual_v12" / "releases" / sc.scope_key
    pointer = root / "current.json"
    component = "release_pointer:" + sc.scope_key
    state = {"fail": True, "fsync_calls": 0}
    if surface == "temp_open":
        original = Path.open

        def failing(self, *args, **kwargs):
            if state["fail"] and self.name.startswith("current.json.") and self.name.endswith(".tmp"):
                raise OSError("simulated origin temp open failure")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", failing)
    elif surface in ("file_fsync", "parent_fsync"):
        original = module.os.fsync
        target = 1 if surface == "file_fsync" else 2

        def failing(fd):
            state["fsync_calls"] += 1
            if state["fail"] and state["fsync_calls"] == target:
                raise OSError(f"simulated origin fsync failure {target}")
            return original(fd)

        monkeypatch.setattr(module.os, "fsync", failing)
    else:
        original = module.os.replace

        def failing(source, destination):
            if state["fail"] and Path(destination).name == "current.json":
                raise OSError("simulated origin replace failure")
            return original(source, destination)

        monkeypatch.setattr(module.os, "replace", failing)
    with pytest.raises(OSError):
        ProducerReleaseCorpus(store, sc)
    assert not pointer.exists()
    assert store.connection.execute(
        "SELECT COUNT(*) AS c FROM tier_usage WHERE component_id=?", (component,)
    ).fetchone()["c"] == 0
    assert not list(root.glob("current.json.*"))
    state["fail"] = False
    corpus = ProducerReleaseCorpus(store, sc)
    assert pointer.is_file()
    assert corpus._current() is None
    store.close()


def test_release_write_charge_rollback_failure_preserves_original(tmp_path, monkeypatch):
    adapter, layout, provider, path = make()
    sc = scope()
    store = open_v12_store(tmp_path)
    path.admit(store, source("badge red", "a"), scope=sc)
    corpus = ProducerReleaseCorpus(store, sc)
    original_open = Path.open
    original_release = store._release

    def failing_open(self, *args, **kwargs):
        if self.suffix == ".slots":
            raise OSError("simulated immutable write failure")
        return original_open(self, *args, **kwargs)

    def failing_release(component_id):
        raise RuntimeError("simulated release charge rollback failure")

    monkeypatch.setattr(Path, "open", failing_open)
    monkeypatch.setattr(store, "_release", failing_release)
    with pytest.raises(OSError) as error:
        corpus.rebuild(path)
    assert any("charge rollback failed" in note for note in getattr(error.value, "__notes__", []))
    store.close()


def test_work_ledger_release_write_accounting_maps_origin(tmp_path):
    from replica_lab.contextual_release_v12 import WorkLedgerReleaseReadAccounting

    class Ledger:
        def __init__(self):
            self.reserved, self.finished = [], []

        def reserve(self, stage, operation, **amounts):
            self.reserved.append((stage, operation, dict(amounts)))
            return len(self.reserved) - 1

        def finish(self, token, *, status="completed", actual=None, detail=None):
            self.finished.append((token, status, dict(actual or {})))

    adapter, layout, provider, path = make()
    sc = scope()
    store = open_v12_store(tmp_path)
    ledger = Ledger()
    corpus = ProducerReleaseCorpus(
        store, sc, read_accounting=WorkLedgerReleaseReadAccounting(ledger, "mechanics"))
    assert any(operation == "release_origin_and_lock" and amounts.get("writes") == 1
               and amounts.get("artifact_bytes", 0) > 0
               for _, operation, amounts in ledger.reserved)
    assert any(status == "completed" for _, status, _ in ledger.finished)
    store.close()


def test_release_accounting_multiplicity_is_exposed(tmp_path):
    adapter, layout, provider, path = make()
    sc = scope()
    store = open_v12_store(tmp_path)
    span = source("badge red", "a")
    path.admit(store, span, scope=sc)
    corpus = ProducerReleaseCorpus(store, sc)
    published = corpus.rebuild(path)
    counts = corpus.accounting_multiplicity()
    assert counts["release_origin_and_lock"] == {"read": 0, "write": 1}
    assert counts["release_pointer_previous"]["read"] == 1
    assert counts["release_manifest_digest"]["read"] == 1
    assert counts["release_source"]["read"] >= 1
    assert counts["release_value"]["write"] >= 1
    assert counts["release_manifest"]["write"] == 1
    assert counts["release_pointer"]["write"] == 1
    before_authority = counts["release_authority"]["read"]
    entry = corpus._current()[1]["records"]["a.span0"]
    corpus.values(entry["source_pin"], span, version=published["producer_release"],
                  scope=sc, source_identity=path.source_identity)
    counts = corpus.accounting_multiplicity()
    assert counts["release_authority"]["read"] > before_authority
    assert counts["release_payload"]["read"] == 1
    store.close()


def test_release_restore_verification_read_is_accounted(tmp_path, monkeypatch):
    class Recorder:
        def __init__(self):
            self.reserved = []

        def reserve_read(self, operation, byte_count, file_count=1):
            self.reserved.append(operation)
            return len(self.reserved) - 1

        def complete_read(self, token, *, actual_bytes, status="completed"):
            return True

        def reserve_write(self, operation, byte_count, file_count=1):
            self.reserved.append(operation)
            return len(self.reserved) - 1

        def complete_write(self, token, *, actual_bytes, status="completed"):
            return True

    adapter, layout, provider, path = make()
    sc = scope()
    store = open_v12_store(tmp_path)
    path.admit(store, source("badge red", "a"), scope=sc)
    recorder = Recorder()
    corpus = ProducerReleaseCorpus(store, sc, read_accounting=recorder)
    first = corpus.rebuild(path)
    pointer_path = corpus.root / "current.json"
    before = pointer_path.read_bytes()

    def failing(directory):
        raise OSError("simulated post-replace fsync failure")

    monkeypatch.setattr(corpus, "_fsync_directory", failing)
    with pytest.raises(OSError):
        corpus.rebuild(path)
    assert pointer_path.read_bytes() == before
    assert corpus._current()[1]["generation"] == first["generation"]
    assert "release_pointer_restore_check" in recorder.reserved
    store.close()


def test_release_forward_validation_reads_scale_with_active_sites(tmp_path):
    adapter, layout, provider, path = make(sites=(3, 7))
    ids, mask, labels = token_inputs()
    sc = scope()
    store = open_v12_store(tmp_path)
    path.admit(store, source("badge red", "a"), scope=sc)
    path.admit(store, source("badge blue", "b"), scope=sc)
    corpus = ProducerReleaseCorpus(store, sc)
    corpus.rebuild(path)
    path.release_corpus = corpus
    released = path.read(store, "badge", scope=sc, wait_seconds=1)
    before = corpus.accounting_multiplicity()["release_authority"]["read"]
    with torch.no_grad():
        path.forward(ids, mask, released, scope=sc)
    after = corpus.accounting_multiplicity()["release_authority"]["read"]
    assert after - before >= 2 * len(layout.sites)
    path.release_corpus = None
    store.close()


def test_release_origin_write_reservation_refusal_leaves_no_charge_or_root(tmp_path):
    from replica_lab.contextual_release_v12 import WorkLedgerReleaseReadAccounting

    class RefusingLedger:
        def reserve(self, stage, operation, **amounts):
            raise ContextualMemoryError("simulated reservation refusal")

        def finish(self, token, *, status="completed", actual=None, detail=None):
            return True

    adapter, layout, provider, path = make()
    sc = scope()
    store = open_v12_store(tmp_path)
    root = store.root / "artifacts" / "contextual_v12" / "releases" / sc.scope_key
    with pytest.raises(ContextualMemoryError, match="reservation refusal"):
        ProducerReleaseCorpus(
            store, sc, read_accounting=WorkLedgerReleaseReadAccounting(
                RefusingLedger(), "mechanics"))
    assert not root.exists()
    assert store.connection.execute(
        "SELECT COUNT(*) AS c FROM tier_usage WHERE component_id=?",
        ("release_pointer:" + sc.scope_key,)).fetchone()["c"] == 0
    store.close()
