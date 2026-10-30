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
