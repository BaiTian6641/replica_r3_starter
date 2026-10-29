"""Real contextual producer/store + 36-block CPU fixture; no capability claim."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from replica_lab.architecture.intrinsic_memory_layout_v4 import spark_layout
from replica_lab.contextual_memory import ContextualMemoryError, canonical_spans
from replica_lab.contextual_memory_layout_v4 import ContextualMemoryLayoutV4, FORMAT
from replica_lab.contextual_spark_fixture import build_contextual_adapter, ids_and_mask, labels_from_ids
from replica_lab.optimizers import NativeFusedAdamW
from test_tiered_memory import open_store, scope


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make(arm="all"):
    torch.manual_seed(17)
    adapter, *_ = build_contextual_adapter(seed=17, layers=36, spine_sites=(3, 7),
                                           spine_slots=8, memory_size=256)
    path = ContextualMemoryLayoutV4(adapter, spark_layout(arm, hidden_size=8),
                                   source_identity="random_cpu_fixture_seed17", lr=0.001)
    adapter.extension.spine = None
    return path


def spans(text="Crimson badge means ready.", name="rule", available=10):
    return canonical_spans(text, record_id=name, source_id="verified-source",
                           source_epoch=1, available_at=available)


def forward(path, batch=None, **kwargs):
    ids, mask = ids_and_mask(4)
    if batch is None:
        batch = path.training_values(spans(), scope=scope())
    return path.forward(ids, mask, batch, scope=scope(), **kwargs)


def step(path, batch=None):
    ids, mask = ids_and_mask(4)
    path.optimizer.zero_grad(set_to_none=True)
    live = path.training_values(spans(), scope=scope()) if batch is None else path.training_from_stored(batch, scope=scope())
    live.slots.retain_grad()
    logits = path.forward(ids, mask, live, scope=scope())
    F.cross_entropy(logits.float().flatten(0, 1), labels_from_ids(ids).flatten()).backward()
    assert all(torch.isfinite(p.grad).all() for group in path.optimizer.param_groups
               for p in group["params"] if p.grad is not None)
    gradients = sum(float(p.grad.abs().sum()) for p in path.encoder.parameters() if p.grad is not None)
    path.optimizer.step()
    return gradients, live.slots.grad


@pytest.mark.parametrize("arm", ["two", "full_attention_nine", "all"])
def test_complete_gradient_path_and_exact_optimizer_owner(arm):
    path = make(arm)
    first, slot_first = step(path)
    assert first == 0 and slot_first is not None and slot_first.abs().sum() == 0
    later, slot_later = step(path)
    assert later > 0 and slot_later.abs().sum() > 0
    assert all(port.query_projection.weight.grad.abs().sum() > 0
               for port in path.reader.ports.values())
    assert all(port.output_projection.weight.grad.abs().sum() > 0
               for port in path.reader.ports.values())
    # In the two-site arm, gradients from the response cross frozen blocks 8..35.
    assert path.adapter.base_unchanged()
    assert all(p.grad is None for p in path.adapter.base.parameters())
    assert isinstance(path.optimizer, NativeFusedAdamW)
    intended = {id(p) for module in path.modules().values() for p in module.parameters()}
    owned = [id(p) for group in path.optimizer.param_groups for p in group["params"]]
    assert len(owned) == len(set(owned)) and set(owned) == intended
    assert path.reader._optimizer_owner is path.optimizer
    with pytest.raises(ValueError, match="optimizer owner"):
        path.reader.make_optimizer()


@pytest.mark.parametrize("arm,count", [("two", 2), ("full_attention_nine", 9), ("all", 36)])
def test_one_store_read_one_snapshot_no_layer_writes_or_value_copies(tmp_path, monkeypatch, arm, count):
    path = make(arm)
    store = open_store(tmp_path)
    assert path.admit(store, spans()[0], scope=scope()).outcome.ok
    batch = path.read(store, "Crimson badge", scope=scope())
    assert not batch.slots.requires_grad
    addresses, calls = [], []
    handles = [port.register_forward_hook(lambda module, args, result:
                (addresses.append(args[1].data_ptr()), calls.append(module.site)) and None)
               for port in path.reader.ports.values()]
    before = dict(path.cost)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(path.encoder, "forward", lambda *args: pytest.fail("durable inference re-encoded text"))
            forward(path, batch)
        assert len(calls) == count and len(set(addresses)) == 1
        assert path.cost["retrieval_calls"] == before["retrieval_calls"] == 1
        assert path.cost["admission_calls"] == before["admission_calls"] == 1
        assert path.cost["correction_calls"] == 0
        assert path.cost["snapshot_copies"] - before["snapshot_copies"] == 1
        assert path.cost["snapshot_bytes"] - before["snapshot_bytes"] == 4*4*256*4
        assert path.cost["value_slots_read"] - before["value_slots_read"] == 4
        assert len(path.optimizer.state) == 0
    finally:
        for handle in handles:
            handle.remove()
        store.close()


def test_disabled_branches_skip_compute_even_with_invalid_batch():
    path = make()
    batch = path.training_values(spans(), scope=scope())
    corrupt = replace(batch, scope_key="wrong")
    ids, mask = ids_and_mask(4)
    baseline = path.forward(ids, mask, None, scope=scope())
    before = dict(path.cost)
    assert torch.equal(baseline, path.forward(ids, mask, corrupt, scope=scope(), enabled=False))
    assert torch.equal(baseline, path.forward(ids, mask, corrupt, scope=scope(), active_sites=()))
    assert path.cost == before
    calls = []
    handles = [port.register_forward_hook(lambda module, args, output: calls.append(module.site))
               for port in path.reader.ports.values()]
    try:
        forward(path, batch, active_sites=(0, 35))
        assert calls == [0, 35]
    finally:
        for handle in handles:
            handle.remove()


def test_controls_change_logits_and_exact_fallback_is_charged(tmp_path):
    path = make()
    step(path); step(path)
    correct = path.training_values(spans(), scope=scope())
    wrong = path.training_values(spans("Silver badge means closed."), scope=scope())
    shuffled = replace(correct, slots=correct.slots.flip(0))
    logits = forward(path, correct)
    assert not torch.equal(logits, forward(path, wrong))
    assert not torch.equal(logits, forward(path, shuffled))
    assert not torch.equal(logits, forward(path, correct, enabled=False))
    text = path.exact_fallback(correct, scope=scope(), tokenizer=lambda value: value.split())
    assert path.cost["fallback_calls"] == 1
    assert path.cost["fallback_bytes"] == len(text.encode())
    assert path.cost["fallback_tokens"] == len(text.split())


@pytest.mark.parametrize("change", ["correction", "revocation"])
def test_shared_snapshot_revalidates_before_next_reader_after_authority_changes(tmp_path, change):
    path = make()
    store = open_store(tmp_path)
    original = spans()[0]
    path.admit(store, original, scope=scope())
    batch = path.read(store, "badge", scope=scope())
    ids, _ = ids_and_mask(4)
    snapshot = path.snapshot(ids, batch, scope=scope())
    hidden = torch.zeros(1, 4, 8)
    snapshot.delta(0, hidden)
    if change == "correction":
        correction = replace(spans("Crimson badge means closed.", name="correction", available=11)[0],
                             source_epoch=2, supersedes=original.record_id)
        assert path.correct(store, original.record_id, correction, scope=scope()).outcome.ok
    else:
        store.revoke_record(original.record_id, reason="withdrawn", scope=scope())
    with pytest.raises(ContextualMemoryError, match="invalidated"):
        snapshot.delta(35, hidden)
    with pytest.raises(ContextualMemoryError, match="invalidated"):
        path.training_from_stored(batch, scope=scope())
    # No persistent projected K/V exists to bypass the authority check.
    assert not any("cache" in name for name in vars(snapshot))
    store.close()


def test_training_uses_current_live_producer_with_pinned_canonical_text(tmp_path):
    path = make("two")
    store = open_store(tmp_path)
    path.admit(store, spans()[0], scope=scope())
    batch = path.read(store, "badge", scope=scope())
    step(path, batch); step(path, batch)
    assert batch.producer_version != path.encoder.release()
    with pytest.raises(ContextualMemoryError, match="stale producer"):
        forward(path, batch)
    gradient, _ = step(path, batch)
    assert gradient > 0  # fitting re-encodes current phi, not stale serialized slots
    with pytest.raises(ContextualMemoryError, match="payload differs"):
        path.training_from_stored(replace(batch, slots=batch.slots.flip(0)), scope=scope())
    store.close()


def assert_tree_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_complete_checkpoint_fresh_process_store_read_and_actual_continuation(tmp_path):
    path = make()
    step(path); step(path)
    store_path = tmp_path / "store"
    store = open_store(store_path)
    path.admit(store, spans()[0], scope=scope())
    batch = path.read(store, "badge", scope=scope())
    checkpoint = tmp_path / "extension"
    manifest = path.save(checkpoint)
    assert manifest["format"] == FORMAT and manifest["contains_base_weights"] is False
    assert set(file.name for file in checkpoint.iterdir()) == {"manifest.json", "extension.safetensors", "optimizer.pt"}
    expected_logits = forward(path, batch).detach()
    before_release = path.encoder.release()
    step(path, batch)
    assert before_release != path.encoder.release()
    expected_weights = {k: v.clone() for k, v in path._tensors().items()}
    expected_optimizer = path.optimizer.state_dict()
    store.close()
    code = '''
import sys, torch
from pathlib import Path
from test_contextual_memory_layout_v4 import make, step, forward, open_store, scope
torch.set_num_threads(1)
path = make()
path.load(sys.argv[1])
store = open_store(Path(sys.argv[2]))
batch = path.read(store, "badge", scope=scope())
logits = forward(path, batch).detach()
step(path, batch)
assert path.adapter.base_unchanged()
path.assert_optimizer_ownership()
torch.save({"logits": logits, "weights": path._tensors(), "optimizer": path.optimizer.state_dict()}, sys.argv[3])
store.close()
'''
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "continued.pt"
    result = subprocess.run([sys.executable, "-c", code, str(checkpoint), str(store_path), str(output)],
        cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src") + os.pathsep + str(root / "tests"),
                       "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    actual = torch.load(output, weights_only=True)
    assert torch.equal(actual["logits"], expected_logits)
    assert_tree_equal(actual["weights"], expected_weights)
    assert_tree_equal(actual["optimizer"], expected_optimizer)
    assert path.adapter.base_unchanged()


@pytest.mark.parametrize("damage", ["source", "layout", "producer", "representation", "codec", "format",
                                   "missing_weights", "optimizer_owner", "optimizer_moment", "optimizer_lr", "optimizer_alias"])
def test_complete_checkpoint_refuses_damage_before_mutation(tmp_path, damage):
    path = make("two")
    step(path)
    checkpoint = tmp_path / "checkpoint"
    path.save(checkpoint)
    before = {k: v.clone() for k, v in path._tensors().items()}
    old_state = path.optimizer.state
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    if damage == "source": manifest["source_identity"] = "different"
    if damage == "layout": manifest["layout"]["sites"] = [0, 35]
    if damage == "producer": manifest["producer_release"] = "old"
    if damage in {"representation", "codec", "format"}: manifest[damage] = "wrong"
    if damage == "missing_weights": (checkpoint / "extension.safetensors").unlink()
    if damage.startswith("optimizer_"):
        state = torch.load(checkpoint / "optimizer.pt", weights_only=True)
        if damage == "optimizer_owner": state["param_groups"][0]["param_names"].reverse()
        if damage == "optimizer_alias":
            identifiers = state["param_groups"][0]["params"]
            removed = identifiers[1]
            identifiers[1] = identifiers[0]
            state["state"].pop(removed, None)
        if damage == "optimizer_moment": next(iter(state["state"].values()))["exp_avg"].fill_(float("nan"))
        if damage == "optimizer_lr": state["param_groups"][0]["lr"] = float("nan")
        torch.save(state, checkpoint / "optimizer.pt")
        manifest["optimizer_sha256"] = hashlib.sha256((checkpoint / "optimizer.pt").read_bytes()).hexdigest()
    (checkpoint / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        path.load(checkpoint)
    assert_tree_equal(path._tensors(), before)
    assert path.optimizer.state is old_state


def test_freeze_checkpoint_retains_optimizer_and_requires_explicit_resume(tmp_path):
    path = make("two")
    step(path)
    path.freeze_readout()
    path.save(tmp_path / "frozen")
    new = make("two")
    new.load(tmp_path / "frozen")
    assert not any(p.requires_grad for module in new.modules().values() for p in module.parameters())
    with pytest.raises(ValueError, match="trainable FP32"):
        new.optimizer.step()
    assert_tree_equal(path.optimizer.state_dict(), new.optimizer.state_dict())
    new.resume_training()
    step(new)
    assert new.adapter.base_unchanged()


def test_pinned_source_text_and_time_cannot_be_substituted(tmp_path):
    path = make("two")
    store = open_store(tmp_path)
    path.admit(store, spans()[0], scope=scope())
    batch = path.read(store, "badge", scope=scope())
    for span in (replace(batch.spans[0], text="Silver! badge means ready."),
                 replace(batch.spans[0], available_at=0)):
        with pytest.raises(ContextualMemoryError, match="provenance"):
            path.training_from_stored(replace(batch, spans=(span,)), scope=scope())
    store.close()


def test_tier_movement_invalidates_hot_snapshot_then_reacquires_exact_values(tmp_path):
    path = make("two")
    store = open_store(tmp_path)
    span = spans()[0]
    path.admit(store, span, scope=scope())
    original = path.read(store, "badge", scope=scope())
    for tier in ("cold", "warm"):
        assert store.migrate_records([span.record_id], to_tier=tier).ok
        with pytest.raises(ContextualMemoryError, match="invalidated"):
            forward(path, original)
        moved = path.read(store, "badge", scope=scope())
        assert torch.equal(moved.slots, original.slots)
        assert moved.spans == original.spans
        assert torch.isfinite(forward(path, moved)).all()
        original = moved
    store.close()


def test_site_zero_only_preserves_response_gradient_through_all_later_frozen_blocks():
    path = make()
    ids, mask = ids_and_mask(4)
    for update in range(2):
        path.optimizer.zero_grad(set_to_none=True)
        batch = path.training_values(spans(), scope=scope())
        logits = path.forward(ids, mask, batch, scope=scope(), active_sites=(0,))
        F.cross_entropy(logits.float().flatten(0, 1), labels_from_ids(ids).flatten()).backward()
        assert path.reader.ports["0"].output_projection.weight.grad.abs().sum() > 0
        assert all(port.output_projection.weight.grad is None
                   for site, port in path.reader.ports.items() if site != "0")
        if update:
            assert path.reader.ports["0"].query_projection.weight.grad.abs().sum() > 0
            assert sum(float(p.grad.abs().sum()) for p in path.encoder.parameters() if p.grad is not None) > 0
        path.optimizer.step()
    assert path.adapter.base_unchanged()


def test_cold_corruption_after_store_restart_is_detectable(tmp_path):
    path = make("two")
    store = open_store(tmp_path)
    span = spans()[0]
    path.admit(store, span, scope=scope())
    assert store.migrate_records([span.record_id], to_tier="cold").ok
    relative = store.connection.execute("SELECT text_path FROM records WHERE record_id=?", (span.record_id,)).fetchone()[0]
    damaged = store.root / relative
    store.close()
    damaged.write_bytes(b"corrupt source")
    reopened = open_store(tmp_path)
    with pytest.raises(ContextualMemoryError):
        path.read(reopened, "badge", scope=scope())
    reopened.close()


def test_same_initial_producer_and_common_readers_across_density_arms():
    two, nine, all_layers = make("two"), make("full_attention_nine"), make("all")
    for other in (nine, all_layers):
        assert_tree_equal(two.encoder.state_dict(), other.encoder.state_dict())
        assert_tree_equal(two.order.state_dict(), other.order.state_dict())
        for site in ("3", "7"):
            assert_tree_equal(two.reader.ports[site].state_dict(), other.reader.ports[site].state_dict())


def test_protocol_code_bindings_and_declared_boundaries():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs/contextual_memory_layer_training_protocol_v4.json").read_text())
    assert protocol["checkpoint_format"] == FORMAT
    assert protocol["architecture"]["placement"] == "block_input_query_to_post_block_residual"
    assert protocol["ownership"]["persistent_P_updates"] == 0
    assert protocol["CPU_diagnostic"]["maximum_total_updates_all_arms"] == 12
    assert protocol["architecture"]["projection_cache"].startswith("none_")
    for filename, digest in protocol["code_bindings"].items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == digest, filename


def test_training_and_checkpoint_phase_boundaries_are_explicit(tmp_path):
    path = make("two")
    live = path.training_values(spans(), scope=scope())
    with pytest.raises(ContextualMemoryError, match="detached durable"):
        path.training_from_stored(live, scope=scope())
    path.save(tmp_path / "checkpoint")
    path.freeze_readout()
    old_state = path.optimizer.state
    with pytest.raises(ContextualMemoryError, match="explicitly resumed"):
        path.load(tmp_path / "checkpoint")
    assert path.optimizer.state is old_state
    path.resume_training()
    path.load(tmp_path / "checkpoint")
    path.order.weight.requires_grad_(False)
    with pytest.raises(ContextualMemoryError, match="partial phi freezing"):
        path.assert_optimizer_ownership()
