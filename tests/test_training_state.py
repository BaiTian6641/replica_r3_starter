"""Durable fine-tuning restart tests with real mixed Muon/Adam updates."""

import copy
import random
import sys
import types

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from replica_lab.optimizers import NativeFusedAdamW, build_optimizer
from replica_lab.training_state import (
    capture_training_fingerprint,
    compare_training_fingerprints,
    fingerprint_buffers,
    load_runtime_state,
    materialize_resume_model,
    save_runtime_state,
    summarize_training_norms,
    tensor_error_norms,
    tensor_fingerprint,
)


CONTRACT = {
    "source": {"revision": "tiny-fixture-v1", "dtype": "float32"},
    "tokenizer": "test-tokens-v1",
    "data": {"sha256": "test-dataset-hash", "sampler": "balanced-v1"},
    "training_regime": "full",
    "optimizer": {"name": "muon8bit", "lr": 0.003, "block_size": 16},
    "schedule": {"kind": "linear", "total_steps": 8},
}


class InstrumentedModel(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 4, device=device) * 0.1)
        self.register_buffer("persistent_scale", torch.ones(4, device=device))
        self.register_buffer("rotary_inv_freq", torch.arange(1, 5, dtype=torch.float32,
                                                            device=device),
                             persistent=False)

    def forward(self, inputs):
        return inputs @ self.weight * self.persistent_scale


class MixedModel(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(3, 7, device=device) * 0.1)
        self.bias = nn.Parameter(torch.zeros(3, device=device))

    def optimizer_groups(self, base_lr, new_lr):
        return [{"params": list(self.parameters()), "lr": base_lr, "group_name": "original"}]

    def forward(self, inputs):
        return F.dropout(inputs, p=0.25, training=True) @ self.matrix.T + self.bias


def make_optimizer(model):
    return build_optimizer(model, lr_base=0.003, lr_muon_base=0.003,
                           block_size=16, weight_decay=0.01)


def seed(device="cpu"):
    random.seed(991)
    torch.manual_seed(991)
    if device == "mps":
        torch.mps.manual_seed(991)


def updates(model, optimizer, start, stop, data_state, schedule_state):
    losses = []
    device = model.matrix.device
    for index in range(start, stop):
        for group, initial_lr in zip(optimizer.param_groups, schedule_state["initial_group_lrs"]):
            group["lr"] = initial_lr * (1 - index / schedule_state["total_steps"])
        inputs = torch.randn(5, 7, device=device)
        targets = torch.tensor([random.randrange(3) for _ in range(5)], device=device)
        loss = F.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.detach().cpu().item())
        data_state["cursor"] += 5
        schedule_state["completed_updates"] = index + 1
    return losses


def assert_nested_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            assert_nested_equal(first, second)
    else:
        assert actual == expected


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(),
                                    reason="MPS unavailable in this execution context"))])
def test_interrupted_training_matches_uninterrupted_with_rng_batches(tmp_path, device):
    seed(device)
    model = MixedModel(device)
    optimizer = make_optimizer(model)
    data = {"cursor": 0, "epoch": 0, "order": [0, 1, 2]}
    schedule = {"total_steps": 8, "completed_updates": 0,
                "initial_group_lrs": [group["lr"] for group in optimizer.param_groups]}
    updates(model, optimizer, 0, 3, data, schedule)
    model_path, runtime_path = tmp_path / "model.pt", tmp_path / "runtime.pt"
    torch.save(model.state_dict(), model_path)
    metadata = save_runtime_state(runtime_path, optimizer, step=3, data_state=data,
                                  schedule_state=schedule, run_contract=CONTRACT)
    expected_losses = updates(model, optimizer, 3, 8, data, schedule)
    expected_python = random.random()
    expected_cpu = torch.rand(5)
    expected_device = torch.rand(5, device=device)

    # Construction and unrelated work must not leak their RNG changes into resume.
    random.seed(19)
    torch.manual_seed(19)
    restored = MixedModel(device)
    restored.load_state_dict(torch.load(model_path, weights_only=True))
    restored_optimizer = make_optimizer(restored)
    resumed = load_runtime_state(runtime_path, restored_optimizer, run_contract=CONTRACT)
    assert resumed == metadata
    assert resumed["data_state"]["cursor"] == 15
    assert resumed["schedule_state"]["completed_updates"] == 3
    actual_losses = updates(restored, restored_optimizer, resumed["step"], 8,
                            resumed["data_state"], resumed["schedule_state"])
    assert actual_losses == expected_losses
    assert_nested_equal(restored.state_dict(), model.state_dict())
    assert_nested_equal(restored_optimizer.state_dict(), optimizer.state_dict())
    assert resumed["data_state"] == data
    assert resumed["schedule_state"] == schedule
    assert restored_optimizer.state[restored.matrix]["momentum_q8"].dtype == torch.int8
    assert restored_optimizer.state[restored.bias]["exp_avg"].dtype == torch.float32
    assert random.random() == expected_python
    torch.testing.assert_close(torch.rand(5), expected_cpu, rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(5, device=device), expected_device, rtol=0, atol=0)


def test_contract_is_canonical_and_mismatch_cannot_change_optimizer_or_rng(tmp_path):
    seed()
    model = MixedModel()
    optimizer = make_optimizer(model)
    path = tmp_path / "runtime.pt"
    saved = save_runtime_state(path, optimizer, step=0, data_state={"cursor": 0},
                               run_contract=CONTRACT)
    reordered = dict(reversed(list(CONTRACT.items())))
    assert load_runtime_state(path, optimizer, run_contract=reordered) == saved
    old_optimizer = copy.deepcopy(optimizer.state_dict())
    old_python, old_cpu = random.getstate(), torch.get_rng_state().clone()
    changed = copy.deepcopy(CONTRACT)
    changed["data"]["sha256"] = "different-data"
    with pytest.raises(ValueError, match="run_contract differs"):
        load_runtime_state(path, optimizer, run_contract=changed)
    assert_nested_equal(optimizer.state_dict(), old_optimizer)
    assert random.getstate() == old_python
    assert torch.equal(torch.get_rng_state(), old_cpu)


def test_atomic_failed_save_preserves_existing_file_and_removes_temp(tmp_path, monkeypatch):
    model = MixedModel()
    optimizer = make_optimizer(model)
    path = tmp_path / "nested" / "runtime.pt"
    save_runtime_state(path, optimizer, step=0, data_state={}, run_contract=CONTRACT)
    original = path.read_bytes()

    def fail(payload, stream):
        stream.write(b"partial checkpoint")
        raise OSError("simulated full disk")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="simulated full disk"):
        save_runtime_state(path, optimizer, step=1, data_state={}, run_contract=CONTRACT)
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("extra", [
    {"step": -1}, {"step": True}, {"data_state": [1]},
    {"data_state": {"bad": float("nan")}}, {"data_state": {1: "bad-key"}},
    {"schedule_state": {"bad": object()}}, {"run_contract": {}},
])
def test_rejects_invalid_metadata_before_writing(tmp_path, extra):
    optimizer = make_optimizer(MixedModel())
    arguments = {"step": 0, "data_state": {}, "run_contract": CONTRACT, **extra}
    path = tmp_path / "runtime.pt"
    with pytest.raises(ValueError):
        save_runtime_state(path, optimizer, **arguments)
    assert not path.exists()


def test_corrupt_contract_hash_is_rejected(tmp_path):
    optimizer = make_optimizer(MixedModel())
    path = tmp_path / "runtime.pt"
    save_runtime_state(path, optimizer, step=0, data_state={}, run_contract=CONTRACT)
    payload = torch.load(path, weights_only=True)
    payload["metadata"]["contract_sha256"] = "tampered"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="hash is inconsistent"):
        load_runtime_state(path, optimizer, run_contract=CONTRACT)


def test_missing_device_rng_requires_explicit_migration(tmp_path, monkeypatch):
    optimizer = make_optimizer(MixedModel())
    path = tmp_path / "runtime.pt"
    save_runtime_state(path, optimizer, step=0, data_state={}, run_contract=CONTRACT)
    payload = torch.load(path, weights_only=True)
    payload["rng"]["torch_mps"] = torch.zeros(44, dtype=torch.uint8)
    torch.save(payload, path)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="requires MPS RNG"):
        load_runtime_state(path, optimizer, run_contract=CONTRACT)
    assert load_runtime_state(path, optimizer, run_contract=CONTRACT,
                              strict_rng=False)["step"] == 0


def test_restore_rng_can_be_disabled_without_changing_global_rng(tmp_path):
    seed()
    optimizer = make_optimizer(MixedModel())
    path = tmp_path / "runtime.pt"
    save_runtime_state(path, optimizer, step=0, data_state={}, run_contract=CONTRACT)
    random.random()
    torch.rand(5)
    old_python, old_cpu = random.getstate(), torch.get_rng_state().clone()
    load_runtime_state(path, optimizer, run_contract=CONTRACT, restore_rng=False)
    assert random.getstate() == old_python
    assert torch.equal(torch.get_rng_state(), old_cpu)


def test_loader_uses_cpu_storage_and_weights_only(tmp_path, monkeypatch):
    optimizer = make_optimizer(MixedModel())
    path = tmp_path / "runtime.pt"
    save_runtime_state(path, optimizer, step=0, data_state={}, run_contract=CONTRACT)
    original_load = torch.load
    calls = []

    def checked_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    load_runtime_state(path, optimizer, run_contract=CONTRACT)
    assert calls == [{"map_location": "cpu", "weights_only": True, "mmap": True}]


def test_tensor_fingerprint_records_values_storage_and_error_norms():
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3) ** 2
    fingerprint = tensor_fingerprint(tensor)
    assert fingerprint["shape"] == [2, 3]
    assert fingerprint["dtype"] == "torch.float32"
    assert fingerprint["device"] == "cpu"
    assert fingerprint["numel"] == 6
    assert fingerprint["storage_bytes"] >= tensor.numel() * tensor.element_size()
    assert fingerprint["stride"] == [3, 1]
    assert fingerprint["is_contiguous"] is True
    assert fingerprint["max_abs"] == 25.0
    assert fingerprint["l2_norm"] == pytest.approx(tensor.double().square().sum().sqrt())
    assert fingerprint["finite"] is True
    assert fingerprint["value_sha256"] == tensor_fingerprint(tensor)["value_sha256"]
    chunked = tensor_fingerprint(tensor, chunk_elements=2)
    assert chunked["value_sha256"] == fingerprint["value_sha256"]
    assert chunked["l2_norm"] == pytest.approx(fingerprint["l2_norm"])
    assert chunked["max_abs"] == pytest.approx(fingerprint["max_abs"])

    exact = tensor_error_norms(tensor, tensor.clone())
    assert exact["exact_equal"] is True
    assert exact["max_abs_error"] == 0.0
    changed = tensor.clone()
    changed[0, 0] += 0.5
    error = tensor_error_norms(tensor, changed)
    assert error["exact_equal"] is False
    assert error["max_abs_error"] == pytest.approx(0.5)
    assert error["l1_error"] == pytest.approx(0.5)
    assert error["rms_error"] == pytest.approx(0.5 / torch.tensor(6.0).sqrt())


def test_checkpoint_metadata_fingerprints_rng_without_changing_it(tmp_path):
    seed()
    optimizer = make_optimizer(MixedModel())
    path = tmp_path / "runtime.pt"
    metadata = save_runtime_state(path, optimizer, step=0, data_state={},
                                  run_contract=CONTRACT)
    assert "rng_fingerprint" in metadata
    assert metadata["rng_fingerprint"]["devices"]
    restored = load_runtime_state(path, optimizer, run_contract=CONTRACT)
    assert restored["rng_fingerprint"] == metadata["rng_fingerprint"]


def test_boundary_fingerprint_round_trip_localizes_restored_state(tmp_path):
    seed()
    model = MixedModel()
    optimizer = make_optimizer(model)
    data = {"cursor": 0, "epoch": 0, "order": [0, 1, 2]}
    schedule = {"total_steps": 8, "completed_updates": 0,
                "initial_group_lrs": [group["lr"] for group in optimizer.param_groups]}
    updates(model, optimizer, 0, 1, data, schedule)
    boundary = capture_training_fingerprint(
        model, optimizer, schedule_state=schedule, data_state=data)
    model_path, runtime_path = tmp_path / "model.pt", tmp_path / "runtime.pt"
    torch.save(model.state_dict(), model_path)
    save_runtime_state(runtime_path, optimizer, step=1, data_state=data,
                       schedule_state=schedule, run_contract=CONTRACT)

    restored = MixedModel()
    restored.load_state_dict(torch.load(model_path, weights_only=True))
    restored.train()
    restored_optimizer = make_optimizer(restored)
    load_runtime_state(runtime_path, restored_optimizer, run_contract=CONTRACT)
    restored_boundary = capture_training_fingerprint(
        restored, restored_optimizer, schedule_state=schedule, data_state=data)
    comparison = compare_training_fingerprints(boundary, restored_boundary)
    assert comparison["matched"], comparison["earliest_divergence"]
    assert comparison["difference_count"] == 0

    restored.matrix.data.add_(0.25)
    mismatched = compare_training_fingerprints(
        boundary,
        capture_training_fingerprint(
            restored, restored_optimizer, schedule_state=schedule, data_state=data))
    assert not mismatched["matched"]
    assert mismatched["earliest_divergence"]["section"] == "parameters"
    assert mismatched["earliest_divergence"]["name"] == "matrix"
    assert "value_sha256" in mismatched["earliest_divergence"]["fields"]
    assert mismatched["numerical"]["max_l2_norm_delta"] > 0.0


def test_fingerprint_localizes_buffers_train_mode_cursor_rng_and_moments():
    model = InstrumentedModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(2, 3)
    model(inputs).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    schedule = {"completed_steps": 1, "base_rates": [1e-3]}
    data = {"cursor": 3, "next_step": 1}
    saved_rng = (random.getstate(), torch.get_rng_state().clone())
    baseline = capture_training_fingerprint(
        model, optimizer, schedule_state=schedule, data_state=data)

    assert compare_training_fingerprints(baseline, baseline)["matched"]

    buffers = fingerprint_buffers(model)
    assert buffers["buffers"]["rotary_inv_freq"]["persistent"] is False
    assert buffers["buffers"]["persistent_scale"]["persistent"] is True
    assert buffers["nonpersistent_names"] == ["rotary_inv_freq"]

    saved_rotary = model.rotary_inv_freq.detach().clone()
    with torch.no_grad():
        model.rotary_inv_freq.add_(1.0)
    changed = compare_training_fingerprints(
        baseline, capture_training_fingerprint(
            model, optimizer, schedule_state=schedule, data_state=data))
    assert changed["earliest_divergence"]["section"] == "buffers"
    assert changed["earliest_divergence"]["name"] == "rotary_inv_freq"
    with torch.no_grad():
        model.rotary_inv_freq.copy_(saved_rotary)

    model.eval()
    changed = compare_training_fingerprints(
        baseline, capture_training_fingerprint(
            model, optimizer, schedule_state=schedule, data_state=data))
    assert changed["earliest_divergence"]["section"] == "train_mode"
    model.train()

    changed = compare_training_fingerprints(
        baseline, capture_training_fingerprint(
            model, optimizer, schedule_state=schedule,
            data_state={"cursor": 4, "next_step": 1}))
    assert changed["earliest_divergence"]["section"] == "data_state"
    assert changed["earliest_divergence"]["name"] == "cursor"

    changed = compare_training_fingerprints(
        baseline, capture_training_fingerprint(
            model, optimizer, schedule_state={"completed_steps": 2,
                                              "base_rates": [1e-3]},
            data_state=data))
    assert changed["earliest_divergence"]["section"] == "schedule_state"
    assert changed["earliest_divergence"]["name"] == "completed_steps"

    random.seed(1234)
    changed = compare_training_fingerprints(
        baseline, capture_training_fingerprint(
            model, optimizer, schedule_state=schedule, data_state=data))
    assert changed["earliest_divergence"]["section"] == "rng"
    random.setstate(saved_rng[0])
    torch.set_rng_state(saved_rng[1])

    moment = next(value for state in optimizer.state.values()
                  for value in state.values()
                  if isinstance(value, torch.Tensor) and value.numel() > 1
                  and value is not optimizer.state[model.weight]["step"])
    with torch.no_grad():
        moment.add_(0.5)
    changed = compare_training_fingerprints(
        baseline, capture_training_fingerprint(
            model, optimizer, schedule_state=schedule, data_state=data))
    assert changed["earliest_divergence"]["section"] == "optimizer"
    assert changed["numerical"]["max_l2_norm_delta"] > 0.0


def test_update_norm_summary_covers_parameters_gradients_and_moments():
    model = MixedModel()
    optimizer = make_optimizer(model)
    inputs = torch.randn(5, 7)
    targets = torch.tensor([0, 1, 2, 0, 1])
    loss = F.cross_entropy(model(inputs), targets)
    loss.backward()
    optimizer.step()
    summary = summarize_training_norms(model, optimizer)
    assert summary["format"] == "replica.training_norms.v1"
    assert summary["parameters"]["tensor_count"] == 2
    assert summary["parameters"]["parameter_count"] == 24
    assert summary["parameters"]["l2_norm"] > 0.0
    assert summary["gradients"]["missing_count"] == 0
    assert set(summary["gradients"]["per_tensor"]) == {"matrix", "bias"}
    assert summary["optimizer"]["state_tensors"] > 0
    assert summary["optimizer"]["counters"]["bias.step"] == 1.0
    assert summary == summarize_training_norms(model, optimizer)
    optimizer.zero_grad(set_to_none=True)

    previous = summary["parameters"]["per_tensor"]["matrix"]["l2_norm"]
    with torch.no_grad():
        model.matrix.add_(0.5)
    updated = summarize_training_norms(model, optimizer)
    assert updated["parameters"]["per_tensor"]["matrix"]["l2_norm"] > previous


def _native_fused(parameter, *, name="weight", lr=1e-3):
    return NativeFusedAdamW([{
        "params": [parameter],
        "param_names": [name],
        "scope": "full",
        "group_name": "full_adamw_fused",
        "algorithm": "adamw_fused",
        "lr": lr,
        "weight_decay": 0.0,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }])


def _native_fused_with_state(parameter, *, name="weight"):
    optimizer = _native_fused(parameter, name=name)
    exp_avg = torch.full_like(parameter, 0.25)
    exp_avg_sq = torch.full_like(parameter, 0.01)
    with torch.no_grad():
        exp_avg.view(-1)[0] = -0.5
        exp_avg_sq.view(-1)[0] = 0.25
    optimizer.state[parameter] = {
        "step": torch.tensor(1.0),
        "exp_avg": exp_avg,
        "exp_avg_sq": exp_avg_sq,
    }
    return optimizer


REQUIRES_MPS = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS unavailable in this execution context")


def test_load_runtime_state_propagates_target_device_to_loader_and_optimizer(tmp_path, monkeypatch):
    optimizer = make_optimizer(MixedModel())
    path = tmp_path / "runtime.pt"
    save_runtime_state(path, optimizer, step=0, data_state={}, run_contract=CONTRACT)
    load_calls, state_calls = [], []
    original_load = torch.load
    original_load_state = type(optimizer).load_state_dict

    def checked_load(*args, **kwargs):
        load_calls.append(dict(kwargs))
        return original_load(*args, **kwargs)

    def checked_load_state(self, state_dict, target_device=None):
        state_calls.append(target_device)
        return original_load_state(self, state_dict, target_device=target_device)

    monkeypatch.setattr(torch, "load", checked_load)
    monkeypatch.setattr(type(optimizer), "load_state_dict", checked_load_state)
    result = load_runtime_state(path, optimizer, run_contract=CONTRACT,
                                restore_rng=False, target_device="cpu")
    assert result["step"] == 0
    assert load_calls == [{"map_location": "cpu", "weights_only": True, "mmap": True}]
    assert state_calls == ["cpu"]


@REQUIRES_MPS
def test_native_fused_restore_moves_moments_to_target_device_only(tmp_path, monkeypatch):
    source_parameter = nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    source = _native_fused_with_state(source_parameter)
    path = tmp_path / "runtime.pt"
    save_runtime_state(path, source, step=1, data_state={"next_step": 1},
                       run_contract=CONTRACT)
    expected = {key: value.detach().clone()
                for key, value in source.state[source_parameter].items()}

    target_parameter = nn.Parameter(torch.zeros(3))
    target = _native_fused(target_parameter)
    state_calls, load_calls = [], []
    original_load_state = NativeFusedAdamW.load_state_dict
    original_load = torch.load

    def checked_load_state(self, state_dict, target_device=None):
        state_calls.append(target_device)
        return original_load_state(self, state_dict, target_device=target_device)

    def checked_load(*args, **kwargs):
        load_calls.append(kwargs.get("map_location"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(NativeFusedAdamW, "load_state_dict", checked_load_state)
    monkeypatch.setattr(torch, "load", checked_load)
    result = load_runtime_state(path, target, run_contract=CONTRACT, target_device="mps")
    assert result["step"] == 1
    assert state_calls == ["mps"]
    assert load_calls == ["mps"]
    assert target_parameter.device.type == "cpu"
    state = target.state[target_parameter]
    assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
    for key, value in expected.items():
        assert state[key].device.type == "mps"
        assert state[key].dtype == value.dtype
        assert torch.equal(state[key].cpu(), value)
        assert tensor_fingerprint(state[key].cpu()) == tensor_fingerprint(value)


def test_failed_direct_restore_leaves_existing_optimizer_state_unchanged(tmp_path):
    source_parameter = nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    source = _native_fused_with_state(source_parameter)
    path = tmp_path / "runtime.pt"
    save_runtime_state(path, source, step=1, data_state={}, run_contract=CONTRACT)

    target_parameter = nn.Parameter(torch.tensor([4.0, 5.0, 6.0]))
    target = _native_fused(target_parameter, name="other")
    target.state[target_parameter] = {
        "step": torch.tensor(3.0),
        "exp_avg": torch.full_like(target_parameter, -1.0),
        "exp_avg_sq": torch.full_like(target_parameter, 0.5),
    }
    before_state = {key: value.clone()
                    for key, value in target.state[target_parameter].items()}
    before_groups = [{key: value for key, value in group.items() if key != "params"}
                     for group in target.param_groups]
    with pytest.raises(ValueError, match="param_names differs"):
        load_runtime_state(path, target, run_contract=CONTRACT, target_device="cpu")
    state = target.state[target_parameter]
    for key, value in before_state.items():
        assert torch.equal(state[key], value)
    assert [{key: value for key, value in group.items() if key != "params"}
            for group in target.param_groups] == before_groups


@REQUIRES_MPS
def test_direct_restore_preserves_boundary_fingerprint_of_staged_path(tmp_path):
    schedule = {"completed_steps": 1}
    data = {"batch_plan_sha256": "plan", "next_step": 1}
    model_staged = InstrumentedModel()
    runtime_path = tmp_path / "runtime.pt"
    save_runtime_state(runtime_path, _native_fused_with_state(model_staged.weight),
                       step=1, data_state=data, schedule_state=schedule,
                       run_contract=CONTRACT)
    model_direct = copy.deepcopy(model_staged)

    staged_optimizer = _native_fused(model_staged.weight)
    load_runtime_state(runtime_path, staged_optimizer, run_contract=CONTRACT,
                       target_device="cpu", restore_rng=False)
    for state in staged_optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to("mps")
    direct_optimizer = _native_fused(model_direct.weight)
    load_runtime_state(runtime_path, direct_optimizer, run_contract=CONTRACT,
                       target_device="mps", restore_rng=False)

    reference = capture_training_fingerprint(
        model_staged, staged_optimizer, schedule_state=schedule, data_state=data,
        include_rng=False)
    actual = capture_training_fingerprint(
        model_direct, direct_optimizer, schedule_state=schedule, data_state=data,
        include_rng=False)
    comparison = compare_training_fingerprints(reference, actual)
    assert comparison["matched"] is True, comparison["earliest_divergence"]
    assert all(value.device.type == "mps"
               for state in direct_optimizer.state.values()
               for value in state.values() if isinstance(value, torch.Tensor))


class _FakeResumeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.weight = nn.Parameter(torch.zeros(2, 3))


def _install_fake_native_sources(monkeypatch):
    transformers = types.ModuleType("transformers")

    class _Config:
        pass

    transformers.AutoConfig = types.SimpleNamespace(
        from_pretrained=lambda path: _Config())
    transformers.AutoModelForCausalLM = types.SimpleNamespace(
        from_config=lambda config, attn_implementation: _FakeResumeBase())
    safetensors = types.ModuleType("safetensors")
    safetensors_torch = types.ModuleType("safetensors.torch")

    def load_file(path, device=None):
        return {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3).to(device)}

    safetensors_torch.load_file = load_file
    safetensors.torch = safetensors_torch
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)


def _fake_native_optimizer(model):
    return _native_fused(model.weight)


def _write_fake_native_resume(tmp_path, *, contract=CONTRACT):
    base_path = tmp_path / "base"
    base_path.mkdir()
    (base_path / "model.safetensors").touch()
    source_parameter = nn.Parameter(torch.tensor(
        [[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]]))
    source = _native_fused_with_state(source_parameter)
    runtime_path = tmp_path / "runtime.pt"
    save_runtime_state(runtime_path, source, step=1, data_state={"next_step": 1},
                       schedule_state={"completed_steps": 1}, run_contract=contract)
    expected = {key: value.detach().clone()
                for key, value in source.state[source_parameter].items()}
    return base_path, runtime_path, expected


@REQUIRES_MPS
def test_materialize_native_resume_restores_optimizer_directly_on_device(tmp_path, monkeypatch):
    _install_fake_native_sources(monkeypatch)
    base_path, runtime_path, expected = _write_fake_native_resume(tmp_path)
    load_calls, state_calls = [], []
    original_load = torch.load
    original_load_state = NativeFusedAdamW.load_state_dict

    def checked_load(*args, **kwargs):
        load_calls.append(kwargs.get("map_location"))
        return original_load(*args, **kwargs)

    def checked_load_state(self, state_dict, target_device=None):
        state_calls.append(target_device)
        return original_load_state(self, state_dict, target_device=target_device)

    def reject_migration(*args, **kwargs):
        raise AssertionError("native resume must not migrate optimizer state manually")

    monkeypatch.setattr(torch, "load", checked_load)
    monkeypatch.setattr(NativeFusedAdamW, "load_state_dict", checked_load_state)
    monkeypatch.setattr("replica_lab.training_state.optimizer_state_to", reject_migration)
    accounting = {}
    base, model, optimizer, restored, base_rates = materialize_resume_model(
        base_path,
        adapter_factory=lambda materialized: materialized,
        optimizer_factory=_fake_native_optimizer,
        runtime_path=runtime_path, run_contract=CONTRACT,
        device="mps", accounting=accounting)

    assert model is base
    assert model.weight.device.type == "mps"
    assert load_calls == ["mps"]
    assert state_calls == ["mps"]
    assert accounting["path"] == "native"
    assert accounting["device"] == "mps"
    assert accounting["full_cpu_model_copy"] is False
    assert accounting["model"]["checkpoint_tensors_adopted_by_model"] is True
    optimizer_accounting = accounting["optimizer"]
    assert optimizer_accounting["restore_strategy"] == "direct_target_device"
    assert optimizer_accounting["target_device"] == "mps"
    assert optimizer_accounting["restored_device"] == "mps"
    assert optimizer_accounting["staged_device"] is None
    assert optimizer_accounting["staged"] == {"tensor_count": 0, "bytes": 0,
                                              "bytes_by_dtype": {}}
    assert optimizer_accounting["staged_cpu_optimizer_bytes"] == 0
    assert optimizer_accounting["migration"] is None
    assert optimizer_accounting["restored"] == {
        "tensor_count": 3, "bytes": 52, "bytes_by_dtype": {"torch.float32": 52}}
    resident = accounting["resident_estimate"]
    assert resident["optimizer_restore_strategy"] == "direct_target_device"
    assert resident["optimizer_target_device"] == "mps"
    assert resident["staged_cpu_optimizer_bytes"] == 0
    assert resident["restored_optimizer_bytes"] == 52
    assert "optimizer_restore_start" in accounting["stages"]
    assert "optimizer_runtime_restored_direct" in accounting["stages"]
    assert "optimizer_runtime_staged_on_cpu" not in accounting["stages"]
    assert "optimizer_migrated" not in accounting["stages"]
    assert "swap_used_bytes_start" in accounting
    assert "swap_growth_bytes" in accounting
    assert restored["step"] == 1
    assert restored["schedule_state"] == {"completed_steps": 1}
    assert base_rates == [1e-3]
    state = optimizer.state[model.weight]
    for key, value in expected.items():
        assert state[key].device.type == "mps"
        assert torch.equal(state[key].cpu(), value)
        assert tensor_fingerprint(state[key].cpu()) == tensor_fingerprint(value)


@REQUIRES_MPS
def test_materialize_native_resume_rejects_contract_mismatch_before_optimizer_accounting(
        tmp_path, monkeypatch):
    _install_fake_native_sources(monkeypatch)
    base_path, runtime_path, _ = _write_fake_native_resume(tmp_path)
    accounting = {}
    with pytest.raises(ValueError, match="run_contract differs"):
        materialize_resume_model(
            base_path,
            adapter_factory=lambda materialized: materialized,
            optimizer_factory=_fake_native_optimizer,
            runtime_path=runtime_path,
            run_contract={**CONTRACT, "training_regime": "modules_only"},
            device="mps", accounting=accounting)
    assert "optimizer" not in accounting
    assert "resident_estimate" not in accounting
