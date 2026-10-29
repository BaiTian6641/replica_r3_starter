"""CPU geometry/ownership evidence; no native Spark or learned-utility claim."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from replica_lab.architecture.intrinsic_memory_layout_v4 import (
    ConfigurableIntrinsicMemoryReader, IntrinsicMemoryLayout, spark_layout,
)
from replica_lab.contextual_spark_fixture import build_contextual_adapter, ids_and_mask

IDENTITY = dict(source_identity="random_cpu_fixture_17", producer_identity="tensor_fixture_v1",
                representation_identity="contextual_tensor_slots_v1",
                codec_identity="ordered_FP32_width256_fixture")


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make(arm="all"):
    adapter, *_ = build_contextual_adapter(seed=17, layers=36, spine_sites=(3, 7),
                                           spine_slots=8, memory_size=256)
    reader = ConfigurableIntrinsicMemoryReader.install(adapter, spark_layout(arm, hidden_size=8))
    adapter.extension.spine = None
    reader.make_optimizer(lr=1e-3)
    return adapter, reader


def values():
    return torch.randn(1, 4, 8, 256, generator=torch.Generator().manual_seed(71))


def forward(adapter, reader, source, **kwargs):
    ids, mask = ids_and_mask(4)
    context = reader.snapshot(source, torch.ones(source.shape[:3], dtype=torch.bool),
                              validate=lambda: None, **kwargs)
    return adapter.forward_text(ids, mask, use_projector=False, use_ports=False,
                                intrinsic_memory=context)


def step(adapter, reader):
    source = values().requires_grad_()
    reader._optimizer_owner.zero_grad(set_to_none=True)
    logits = forward(adapter, reader, source)
    ids, _ = ids_and_mask(4)
    F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids.reshape(-1)).backward()
    assert all(torch.isfinite(p.grad).all() for p in reader.parameters() if p.grad is not None)
    reader._optimizer_owner.step()
    return source.grad


@pytest.mark.parametrize("arm,count", [("two", 2), ("full_attention_nine", 9), ("all", 36)])
def test_native_geometry_parameter_arithmetic_without_allocating_native_weights(arm, count):
    layout = spark_layout(arm)
    with torch.device("meta"):
        reader = ConfigurableIntrinsicMemoryReader(layout)
    assert len(reader.ports) == count
    assert sum(p.numel() for p in reader.parameters()) == layout.reader_parameter_count() == count * 1447937
    assert len({id(p) for p in reader.parameters()}) == len(list(reader.parameters()))


@pytest.mark.parametrize("arm", ["two", "full_attention_nine", "all"])
def test_cpu_36_block_zero_effect_and_disabled_parity_without_external_spine(arm):
    adapter, reader = make(arm)
    ids, mask = ids_and_mask(4)
    baseline = adapter.forward_text(ids, mask, use_projector=False, use_ports=False)
    source = values()
    old = source.clone()
    assert torch.equal(baseline, forward(adapter, reader, source))
    assert torch.equal(source, old)
    assert adapter.base_unchanged()
    assert {id(p) for p in adapter.parameters() if p.requires_grad} == {id(p) for p in reader.parameters()}
    with pytest.raises(ValueError, match="optimizer owner"):
        reader.make_optimizer()


def test_all_layers_receive_gradients_and_values_remain_differentiable():
    adapter, reader = make()
    first = step(adapter, reader)
    assert first is not None and first.abs().sum() == 0  # closed output projections
    later = step(adapter, reader)
    assert later is not None and later.abs().sum() > 0
    assert all(port.query_projection.weight.grad.abs().sum() > 0 for port in reader.ports.values())
    assert all(port.output_projection.weight.grad.abs().sum() > 0 for port in reader.ports.values())
    assert adapter.base_unchanged()


def test_active_layer_mask_skips_readers_and_does_not_multiply_state_or_updates():
    adapter, reader = make()
    calls = []
    slot_storage = []
    def observe(module, args, result):
        calls.append(module.site)
        slot_storage.append(args[1].data_ptr())
    handles = [p.register_forward_hook(observe)
               for p in reader.ports.values()]
    try:
        source = values()
        before = source.clone()
        forward(adapter, reader, source, active_sites=(0, 35)).sum().backward()
        assert calls == [0, 35]
        assert len(set(slot_storage)) == 1
        assert torch.equal(source, before)
        for site, port in reader.ports.items():
            assert (port.output_projection.weight.grad is not None) == (int(site) in (0, 35))
        assert len(reader._optimizer_owner.state) == 0  # forward/backward never step
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("sites", [(), (7, 3), (3, 3), (-1, 7), (3, 36), (True, 7)])
def test_invalid_sites_rejected(sites):
    with pytest.raises(ValueError, match="layout"):
        IntrinsicMemoryLayout(8, sites=sites)


def test_geometry_and_existing_owner_refused_before_mutation():
    adapter, *_ = build_contextual_adapter(seed=17, layers=8, spine_sites=(3, 7))
    ports = adapter.extension.spine.ports["memory"]
    with pytest.raises(ValueError, match="geometry"):
        ConfigurableIntrinsicMemoryReader.install(adapter, spark_layout("all", hidden_size=8))
    assert adapter.extension.spine.ports["memory"] is ports
    layout = IntrinsicMemoryLayout(8, decoder_layers=8)
    ConfigurableIntrinsicMemoryReader.install(adapter, layout)
    with pytest.raises(ValueError, match="already has an owner"):
        ConfigurableIntrinsicMemoryReader.install(adapter, layout)


def test_shared_snapshot_revalidates_correction_and_rejects_undeclared_site():
    _, reader = make()
    valid = True
    def validate():
        if not valid:
            raise ValueError("corrected source revision")
    v = values()
    snap = reader.snapshot(v, torch.ones(v.shape[:3], dtype=torch.bool), validate=validate,
                           active_sites=(0, 35))
    with pytest.raises(ValueError, match="disabled"):
        snap.delta(1, torch.zeros(1, 4, 8))
    valid = False
    with pytest.raises(ValueError, match="corrected"):
        snap.delta(0, torch.zeros(1, 4, 8))


def test_extension_only_checkpoint_and_actual_fresh_process_optimizer_continuation(tmp_path):
    adapter, reader = make()
    step(adapter, reader); step(adapter, reader)
    path = tmp_path / "reader"
    reader.save_checkpoint(path, **IDENTITY)
    expected_logits = forward(adapter, reader, values()).detach()
    step(adapter, reader)
    code = """
import sys, torch
from pathlib import Path
from test_intrinsic_memory_layout_v4 import make, values, forward, step, IDENTITY
torch.set_num_threads(1)
adapter, reader = make()
reader.load_checkpoint(Path(sys.argv[1]), **IDENTITY)
logits = forward(adapter, reader, values()).detach()
step(adapter, reader)
torch.save({'logits': logits, 'weights': reader.state_dict(),
            'optimizer': reader._optimizer_owner.state_dict()}, sys.argv[2])
assert adapter.base_unchanged()
"""
    root = Path(__file__).resolve().parents[1]
    child = subprocess.run([sys.executable, "-c", code, str(path), str(tmp_path / "continued.pt")],
                           cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")+os.pathsep+str(root / "tests"),
                                          "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stderr
    result = torch.load(tmp_path / "continued.pt", weights_only=True)
    assert torch.equal(result["logits"], expected_logits)
    assert all(torch.equal(v, result["weights"][k]) for k, v in reader.state_dict().items())
    expected = reader._optimizer_owner.state_dict()
    assert result["optimizer"]["param_groups"] == expected["param_groups"]
    for key, state in expected["state"].items():
        for name, value in state.items():
            assert torch.equal(value, result["optimizer"]["state"][key][name]) if torch.is_tensor(value) else value == result["optimizer"]["state"][key][name]
    assert set(p.name for p in path.iterdir()) == {"manifest.json", "reader.safetensors", "optimizer.pt"}


@pytest.mark.parametrize("damage", ["layout", "producer", "representation", "legacy_format", "missing_payload",
                                   "corrupt_optimizer", "optimizer_moment", "optimizer_owner"])
def test_checkpoint_identity_and_payload_refusal_before_reader_mutation(tmp_path, damage):
    adapter, reader = make("two")
    step(adapter, reader)
    path = tmp_path / "reader"
    reader.save_checkpoint(path, **IDENTITY)
    before = {k: v.clone() for k, v in reader.state_dict().items()}
    manifest = json.loads((path / "manifest.json").read_text())
    if damage == "layout": manifest["layout"]["sites"] = [0, 35]
    if damage == "producer": manifest["producer_identity"] = "stale"
    if damage == "representation": manifest["representation_identity"] = "wrong_semantics"
    if damage == "legacy_format": manifest["format"] = "replica.intrinsic_memory.implementation.v3"
    if damage == "missing_payload": (path / "reader.safetensors").unlink()
    if damage == "corrupt_optimizer": (path / "optimizer.pt").write_bytes(b"bad")
    if damage in {"optimizer_moment", "optimizer_owner"}:
        state = torch.load(path / "optimizer.pt", weights_only=True)
        if damage == "optimizer_moment":
            next(iter(state["state"].values()))["exp_avg"] = torch.tensor(float("nan"))
        else:
            state["param_groups"][0]["param_names"].reverse()
        torch.save(state, path / "optimizer.pt")
        manifest["optimizer_sha256"] = hashlib.sha256((path / "optimizer.pt").read_bytes()).hexdigest()
    (path / "manifest.json").write_text(json.dumps(manifest))
    previous_state = reader._optimizer_owner.state
    with pytest.raises(ValueError):
        reader.load_checkpoint(path, **IDENTITY)
    assert all(torch.equal(value, reader.state_dict()[name]) for name, value in before.items())
    assert reader._optimizer_owner.state is previous_state


def test_reader_only_checkpoint_refuses_stale_optimizer_state(tmp_path):
    adapter, reader = make("two")
    path = tmp_path / "reader"
    untrained = ConfigurableIntrinsicMemoryReader(reader.spec)
    untrained.save_checkpoint(path, **IDENTITY)
    step(adapter, reader)
    with pytest.raises(ValueError, match="unrelated optimizer"):
        reader.load_checkpoint(path, **IDENTITY)
