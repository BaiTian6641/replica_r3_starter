"""Durable fine-tuning restart tests with real mixed Muon/Adam updates."""

import copy
import random

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from replica_lab.optimizers import build_optimizer
from replica_lab.training_state import load_runtime_state, save_runtime_state


CONTRACT = {
    "source": {"revision": "tiny-fixture-v1", "dtype": "float32"},
    "tokenizer": "test-tokens-v1",
    "data": {"sha256": "test-dataset-hash", "sampler": "balanced-v1"},
    "training_regime": "full",
    "optimizer": {"name": "muon8bit", "lr": 0.003, "block_size": 16},
    "schedule": {"kind": "linear", "total_steps": 8},
}


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
