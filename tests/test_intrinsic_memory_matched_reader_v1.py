"""Fixed-parameter access control on a random 36-block BF16 CPU fixture."""
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

from replica_lab.architecture.spine import SpinePort
from replica_lab.contextual_memory import ContextualMemoryError
from replica_lab.contextual_memory_v5 import ContextualMemoryV5
from replica_lab.contextual_spark_fixture import build_contextual_adapter, ids_and_mask, labels_from_ids
from replica_lab.intrinsic_memory_matched_reader_v1 import (
    ContextualMemoryMatchedV1, MatchedParameterReader, matched_layout, SCOPE,
)
from replica_lab.optimizers import NativeFusedAdamW
from test_contextual_memory_layout_v4 import assert_tree_equal
from test_contextual_memory_v5 import spans
from test_tiered_memory import open_store, scope

ARMS = ("two", "full_attention_nine", "all")


@pytest.mark.parametrize("arm", ARMS)
def test_declared_control_matches_installed_parameter_and_access_contract(arm):
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root/"configs/intrinsic_memory_matched_reader_v1.json").read_text())
    layout = matched_layout(arm)
    assert list(layout.sites) == protocol["arms"][arm]["active_sites"]
    assert layout.reader_parameter_count() == protocol["arms"][arm]["unique_native_reader_parameters"]
    assert protocol["native_launch_enabled"] is False and protocol["P_updates"] == 0
    assert protocol["unique_reader_groups"] == 2
    assert protocol["sharing"] == "group=(layer//4)%2"
    assert protocol["placement"] == layout.placement


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make(arm="all"):
    adapter, *_ = build_contextual_adapter(seed=17, layers=36, spine_sites=(3, 7),
                                           spine_slots=8, memory_size=256)
    path = ContextualMemoryMatchedV1(adapter, matched_layout(arm, hidden_size=8),
                                    source_identity="random_fixture17", lr=0.001)
    adapter.extension.spine = None
    return path


def forward(path, batch=None, **kwargs):
    ids, mask = ids_and_mask(4)
    batch = path.training_values(spans(), scope=scope()) if batch is None else batch
    return path.forward(ids, mask, batch, scope=scope(), **kwargs)


def step(path, batch=None):
    ids, mask = ids_and_mask(4)
    path.optimizer.zero_grad(set_to_none=True)
    live = (path.training_values(spans(), scope=scope()) if batch is None
            else path.training_from_stored(batch, scope=scope()))
    live.slots.retain_grad()
    logits = path.forward(ids, mask, live, scope=scope())
    F.cross_entropy(logits.flatten(0, 1), labels_from_ids(ids).flatten()).backward()
    assert all(torch.isfinite(p.grad).all() for m in path.modules().values()
               for p in m.parameters() if p.grad is not None)
    encoder_gradient = sum(float(p.grad.abs().sum()) for p in path.encoder.parameters() if p.grad is not None)
    path.optimizer.step()
    return encoder_gradient, live.slots.grad


@pytest.mark.parametrize("arm", ARMS)
def test_native_parameter_count_is_exactly_two_registered_groups_without_allocating_base(arm):
    layout = matched_layout(arm)
    with torch.device("meta"):
        reader = MatchedParameterReader(layout, [SpinePort(layout, "memory", s) for s in (3, 7)])
    assert sum(p.numel() for p in reader.parameters()) == layout.reader_parameter_count() == 2 * 1447937
    assert len(reader.groups) == 2
    assert all(name.startswith("groups.") for name in reader.state_dict())
    assert len({id(port) for port in reader.ports.values()}) == 2


def test_two_site_control_has_identical_initialization_logits_and_updates_to_independent_v5():
    from test_contextual_memory_v5 import make as reference_make
    reference, shared = reference_make(), make("two")
    for name in ("encoder", "order"):
        assert_tree_equal(getattr(reference, name).state_dict(), getattr(shared, name).state_dict())
    for site, group in ((3, 0), (7, 1)):
        assert_tree_equal(reference.reader.ports[str(site)].state_dict(), shared.reader.groups[str(group)].state_dict())
    for _ in range(3):
        assert torch.equal(forward(reference), forward(shared))
        step(reference); step(shared)
    assert torch.equal(forward(reference), forward(shared))
    assert reference.adapter.base_unchanged() and shared.adapter.base_unchanged()


@pytest.mark.parametrize("arm", ARMS)
def test_response_reaches_every_active_site_and_producer_with_one_optimizer(arm):
    path = make(arm)
    first, first_slot = step(path)
    assert first == 0 and first_slot.abs().sum() == 0
    activations, addresses = [], []
    def retain(module, args, output):
        output.retain_grad()
        activations.append(output)
        addresses.append(args[1].data_ptr())
    hooks = [module.register_forward_hook(retain) for module in path.reader.groups.values()]
    try:
        later, slots = step(path)
    finally:
        for hook in hooks: hook.remove()
    assert later > 0 and slots.abs().sum() > 0
    assert len(activations) == len(path.spec.sites) and len(set(addresses)) == 1
    # Inspect each invocation, not just the accumulated shared parameter gradient.
    assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in activations)
    assert all(p.grad is None for p in path.adapter.base.parameters()) and path.adapter.base_unchanged()
    assert isinstance(path.optimizer, NativeFusedAdamW)
    names = path.assert_optimizer_ownership()
    owned = [id(p) for group in path.optimizer.param_groups for p in group["params"]]
    assert len(owned) == len(set(owned))
    assert all(group["scope"] == SCOPE for group in path.optimizer.param_groups)
    assert all(name.startswith("reader.groups.") for name in names["reader"])
    with pytest.raises(ValueError, match="optimizer owner"):
        path.reader.make_optimizer()


@pytest.mark.parametrize("arm", ARMS)
def test_one_store_read_one_snapshot_and_no_density_dependent_write(tmp_path, arm):
    path, store = make(arm), open_store(tmp_path)
    try:
        assert path.admit(store, spans()[0], scope=scope()).outcome.ok
        batch = path.read(store, "Badge", scope=scope())
        before = dict(path.cost)
        forward(path, batch)
        assert path.cost["retrieval_calls"] == before["retrieval_calls"] == 1
        assert path.cost["admission_calls"] == before["admission_calls"] == 1
        assert path.cost["durable_artifact_growth_bytes"] == before["durable_artifact_growth_bytes"]
        assert path.cost["snapshot_copies"] - before["snapshot_copies"] == 1
        assert path.cost["snapshot_bytes"] - before["snapshot_bytes"] == 1 * 4 * 8 * 256 * 4
        assert path.cost["producer_calls"] == before["producer_calls"]
        assert not path.optimizer.state
    finally:
        path.close(); store.close()


def test_disabled_branches_skip_all_reader_and_producer_computation(monkeypatch):
    path = make()
    batch = path.training_values(spans(), scope=scope())
    corrupt = replace(batch, scope_key="different")
    ids, mask = ids_and_mask(4)
    baseline = path.forward(ids, mask, None, scope=scope())
    before = dict(path.cost)
    monkeypatch.setattr(path.encoder, "forward", lambda *args: pytest.fail("disabled producer"))
    for group in path.reader.groups.values():
        monkeypatch.setattr(group, "forward", lambda *args: pytest.fail("disabled reader"))
    assert torch.equal(baseline, path.forward(ids, mask, corrupt, scope=scope(), enabled=False))
    assert torch.equal(baseline, path.forward(ids, mask, corrupt, scope=scope(), active_sites=()))
    assert path.cost == before


def test_active_mask_dispatches_only_selected_sites():
    path = make()
    calls = []
    hooks = [port.register_forward_hook(lambda m, a, o: calls.append(m.site))
             for port in path.reader.groups.values()]
    try:
        forward(path, active_sites=(0, 35))
        assert calls == [3, 3]  # two calls, both to canonical group 0
    finally:
        for hook in hooks: hook.remove()


@pytest.mark.parametrize("change", ("correction", "revocation"))
def test_shared_snapshot_refuses_stale_authority_before_next_active_read(tmp_path, change):
    path, store = make(), open_store(tmp_path)
    try:
        original = spans()[0]
        path.admit(store, original, scope=scope())
        batch = path.read(store, "Badge", scope=scope())
        ids, _ = ids_and_mask(4)
        snapshot = path.snapshot(ids, batch, scope=scope())
        snapshot.delta(0, torch.zeros(1, 4, 8))
        if change == "correction":
            corrected = replace(spans("Badge means blue.", name="new", available=6)[0],
                                source_epoch=2, supersedes=original.record_id)
            assert path.correct(store, original.record_id, corrected, scope=scope()).outcome.ok
        else:
            store.revoke_record(original.record_id, reason="withdrawn", scope=scope())
        with pytest.raises(ContextualMemoryError, match="invalidated"):
            snapshot.delta(35, torch.zeros(1, 4, 8))
        assert not any("cache" in name for name in vars(snapshot))
    finally:
        path.close(); store.close()


@pytest.mark.parametrize("arm", ARMS)
def test_complete_checkpoint_fresh_process_exact_logits_weights_and_actual_continuation(tmp_path, arm):
    path = make(arm)
    step(path); step(path)
    store_path = tmp_path / "store"
    store = open_store(store_path)
    path.admit(store, spans()[0], scope=scope())
    batch = path.read(store, "Badge", scope=scope())
    checkpoint = tmp_path / "extension"
    path.save(checkpoint)
    before = forward(path, batch).detach()
    old_release = path.encoder.release()
    step(path, batch)
    assert path.encoder.release() != old_release
    expected = {"logits": before, "weights": {k: v.clone() for k, v in path._tensors().items()},
                "optimizer": path.optimizer.state_dict()}
    store.close(); path.close()
    child = '''
import sys, torch
from pathlib import Path
from test_intrinsic_memory_matched_reader_v1 import make, step, forward, open_store, scope
torch.set_num_threads(1)
p=make(sys.argv[1]);p.load(sys.argv[2]);s=open_store(Path(sys.argv[3]))
b=p.read(s,'Badge',scope=scope());logits=forward(p,b).detach();step(p,b)
assert p.adapter.base_unchanged();p.assert_optimizer_ownership()
torch.save({'logits':logits,'weights':p._tensors(),'optimizer':p.optimizer.state_dict()},sys.argv[4])
s.close();p.close()
'''
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "continued.pt"
    result = subprocess.run([sys.executable, "-c", child, arm, str(checkpoint), str(store_path), str(output)],
        cwd=root, env={**os.environ, "PYTHONPATH": str(root/"src")+os.pathsep+str(root/"tests"),
                       "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert_tree_equal(torch.load(output, weights_only=True), expected)


@pytest.mark.parametrize("damage", ("mapping", "source", "scope", "alias", "format", "missing_weights"))
def test_incompatible_complete_checkpoint_refused_before_mutation(tmp_path, damage):
    path = make("two"); step(path); path.save(tmp_path/"checkpoint")
    other = make("two")
    before = {k: v.clone() for k, v in other._tensors().items()}
    root = tmp_path/"checkpoint"
    outer_file = root/"manifest.json"; outer = json.loads(outer_file.read_text())
    middle_file = root/"contextual_v5/manifest.json"; middle = json.loads(middle_file.read_text())
    inner_file = root/"contextual_v5/components/manifest.json"; inner = json.loads(inner_file.read_text())
    if damage == "mapping": outer["layout"]["site_to_group"]["3"] = 1
    if damage == "format": outer["format"] = "legacy_independent"
    if damage == "source": inner["source_identity"] = "substituted"
    if damage == "missing_weights": (inner_file.parent/"extension.safetensors").unlink()
    if damage in {"scope", "alias"}:
        p = inner_file.parent/"optimizer.pt"
        state = torch.load(p, weights_only=True)
        if damage == "scope": state["param_groups"][-1]["scope"] = "wrong"
        else: state["param_groups"][-1]["params"][1] = state["param_groups"][-1]["params"][0]
        torch.save(state, p)
        inner["optimizer_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    inner_file.write_text(json.dumps(inner))
    middle["components_manifest_sha256"] = hashlib.sha256(inner_file.read_bytes()).hexdigest()
    middle_file.write_text(json.dumps(middle))
    outer["contextual_manifest_sha256"] = hashlib.sha256(middle_file.read_bytes()).hexdigest()
    outer_file.write_text(json.dumps(outer))
    with pytest.raises(ValueError): other.load(root)
    assert_tree_equal(other._tensors(), before)
    assert not other.optimizer.state


def test_wrong_arm_and_independent_format_cannot_load_shared_checkpoint(tmp_path):
    path = make("two"); path.save(tmp_path/"checkpoint")
    with pytest.raises(ContextualMemoryError): make("all").load(tmp_path/"checkpoint")
    from test_contextual_memory_v5 import make as independent
    with pytest.raises(ContextualMemoryError): independent().load(tmp_path/"checkpoint")
    with pytest.raises(ContextualMemoryError): path.reader.save_checkpoint(tmp_path/"reader_only")


def test_stale_values_refused_while_canonical_source_retains_live_gradient(tmp_path):
    path, store = make(), open_store(tmp_path)
    try:
        path.admit(store, spans()[0], scope=scope())
        batch = path.read(store, "Badge", scope=scope())
        step(path, batch); step(path, batch)
        with pytest.raises(ContextualMemoryError, match="stale producer"):
            forward(path, batch)
        gradient, slot_gradient = step(path, batch)
        assert gradient > 0 and slot_gradient.abs().sum() > 0
    finally:
        path.close(); store.close()


def test_frozen_complete_checkpoint_preserves_sharing_and_requires_explicit_resume(tmp_path):
    path = make(); step(path); path.freeze_readout(); path.save(tmp_path/"frozen")
    other = make(); other.load(tmp_path/"frozen")
    assert not any(p.requires_grad for m in other.modules().values() for p in m.parameters())
    assert len(other.reader.groups) == 2
    other.resume_training()
    step(other)
    assert other.adapter.base_unchanged()
