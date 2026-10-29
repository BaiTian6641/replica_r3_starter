"""Portable device-placement tests; no checkpoint weights or MPS execution."""
from pathlib import Path
import inspect
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from replica_lab.spark_device_placement_v2 import DevicePlacementError, place_frozen_spark_base
from replica_lab.architecture.spark2_5 import load_spark2_5_config
from replica_lab.contextual_spark_fixture import ContextualSparkCausalLM, contextual_config_raw


class TinyTied(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4, dtype=torch.bfloat16)
        self.output = nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
        self.output.weight = self.embedding.weight
        self.register_buffer("scale", torch.ones(4, dtype=torch.float32))
        self.register_buffer("counter", torch.zeros((), dtype=torch.int64))

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.output


def test_cpu_placement_preserves_mixed_dtypes_bytes_ties_buffers_and_freezes():
    base = TinyTied()
    before = {k: v.clone() for k, v in base.state_dict().items()}
    receipt = place_frozen_spark_base(base, device="cpu")
    assert receipt["content_sha256_before"] == receipt["content_sha256_after"]
    assert receipt["all_original_parameters_frozen"]
    assert receipt["tied_output_preserved"]
    assert receipt["unique_tensor_bytes"] == 8 * 4 * 2 + 4 * 4 + 8
    assert receipt["bytes_hashed"] == 2 * receipt["unique_tensor_bytes"]
    assert base.embedding.weight is base.output.weight
    assert all(not p.requires_grad for p in base.parameters())
    for name, value in base.state_dict().items():
        assert value.dtype == before[name].dtype
        assert torch.equal(value, before[name])


def test_real_tiny_spark_fixture_preserves_logits_original_bytes_and_tied_path():
    base = ContextualSparkCausalLM(load_spark2_5_config(contextual_config_raw()))
    ids = torch.tensor([[1, 2, 3]])
    expected = base(input_ids=ids).logits.detach().clone()
    receipt = place_frozen_spark_base(base, device="cpu")
    actual = base(input_ids=ids).logits
    assert torch.equal(expected, actual)
    assert receipt["status"] == "verified"
    assert base.get_input_embeddings().weight is base.get_output_embeddings().weight


@pytest.mark.parametrize("damage", ["untied", "meta_buffer", "noncontiguous", "missing_output"])
def test_incompatible_loaded_model_refused_before_placement(damage):
    base = TinyTied()
    if damage == "untied":
        base.output.weight = nn.Parameter(base.output.weight.detach().clone())
    elif damage == "meta_buffer":
        base.scale = torch.empty(4, device="meta")
    elif damage == "noncontiguous":
        base.register_buffer("view", torch.ones(3, 4).t())
    elif damage == "missing_output":
        base.get_output_embeddings = lambda: None
    base.to = lambda **kwargs: pytest.fail("placement happened before admission")
    with pytest.raises(DevicePlacementError):
        place_frozen_spark_base(base, device="cpu")


@pytest.mark.parametrize("damage", ["cast", "content", "untie", "move_buffer", "unfreeze", "new_owner"])
def test_placement_mutation_or_incomplete_device_move_is_refused(damage):
    base = TinyTied()
    def broken_to(**kwargs):
        assert kwargs == {"device": torch.device("cpu")}
        if damage == "cast": base.float()
        elif damage == "content": base.embedding.weight.data[0, 0] += 1
        elif damage == "untie": base.output.weight = nn.Parameter(base.output.weight.clone(), requires_grad=False)
        elif damage == "move_buffer": base.scale = torch.empty(4, device="meta")
        elif damage == "unfreeze": base.requires_grad_(True)
        elif damage == "new_owner": return TinyTied()
        return base
    base.to = broken_to
    with pytest.raises(DevicePlacementError):
        place_frozen_spark_base(base, device="cpu")


@pytest.mark.parametrize("device", ["cuda", "meta", "mps:1", "cpu:1"])
def test_unsupported_target_refused_without_device_access(device):
    with pytest.raises(DevicePlacementError):
        place_frozen_spark_base(TinyTied(), device=device)


def test_hashing_uses_bounded_chunks(monkeypatch):
    import replica_lab.spark_device_placement_v2 as placement
    monkeypatch.setattr(placement, "HASH_CHUNK_BYTES", 8)
    base = TinyTied()
    receipt = place_frozen_spark_base(base, device="cpu")
    assert receipt["hash_chunk_bytes_max"] == 8
    assert receipt["content_sha256_before"] == receipt["content_sha256_after"]


def test_new_runner_places_base_before_adapter_and_preserves_sealed_loader():
    import run_spark_intrinsic_readout_placement_v2 as revised
    import run_spark_persistent_learner_derivative_bf16 as sealed
    source = inspect.getsource(revised.build_native_derivative_backend)
    assert source.index("place_frozen_spark_base") < source.index("adapter = Spark2_5Adapter")
    assert inspect.getsource(revised._load_derivative_model) == inspect.getsource(sealed._load_derivative_model)
    assert revised.RUN_FORMAT == sealed.RUN_FORMAT
    assert revised.P1_SPINE_SITES == sealed.P1_SPINE_SITES == (3, 7)
    assert "_readout_restart_command" in inspect.getsource(revised._native_impl)
    assert "restart_device_placement.json" in inspect.getsource(revised._native_impl)


def test_fresh_process_selects_repaired_runner_for_both_restart_paths(tmp_path):
    code = '''
import json
from pathlib import Path
from types import SimpleNamespace
import run_spark_intrinsic_readout_placement_v2 as r
p=SimpleNamespace(root=Path("/synthetic"),path=Path("/synthetic/config.json"))
print(json.dumps([r._readout_restart_command(p,Path("out"),Path("data"),Path("auth")),
 r._restart_command(p,Path("out"),Path("data"),Path("auth"),Path("export"))]))
'''
    child = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    commands = json.loads(child.stdout)
    for command in commands:
        assert command[0] == sys.executable
        assert command[1] == "/synthetic/scripts/run_spark_intrinsic_readout_placement_v2.py"
        assert command[command.index("--device") + 1] == "mps"
    assert commands[0][commands[0].index("--stage") + 1] == "readout_restart"


def test_new_launcher_keeps_qualified_interpreter_and_guards():
    import launch_spark_intrinsic_readout_placement_v2 as launcher
    packet = {"output_root": "out", "protocol": {"path": "config.json"},
              "admission_receipt": "admission.json", "guard_report": "guard.json",
              "prepared_root": "data"}
    preflight, guard = launcher.commands(packet, Path("auth.json"), root=Path("/synthetic"))
    assert preflight[0] == guard[0] == sys.executable
    assert preflight[1].endswith("preflight_spark_intrinsic_readout_placement_v2.py")
    assert guard[1].endswith("run_capacity_preflight_spark_placement_v2.py")
    assert guard[guard.index("--entry-point") + 1] == "run_spark_intrinsic_readout_placement_v2.py"
    for key, value in [("--driver-limit-gib", "50.0"), ("--swap-growth-limit-mib", "512.0"),
                       ("--minimum-free-percent", "15.0"), ("--stage", "readout")]:
        assert guard[guard.index(key) + 1] == value


def test_consumed_environment_packet_cannot_authorize_new_launcher(tmp_path):
    import launch_spark_intrinsic_readout_environment_v1 as old
    import launch_spark_intrinsic_readout_placement_v2 as new
    assert old.digest(old.__file__) != new.digest(new.__file__)
    packet = {"execution_environment": {"path": "env.json", "sha256": "0" * 64},
              "launcher": {"sha256": old.digest(old.__file__)}}
    path = tmp_path / "packet.json"
    path.write_text(json.dumps(packet))
    (tmp_path / "env.json").write_text("{}")
    with pytest.raises(ValueError, match="reviewed environment or launcher changed"):
        new.run(path, launch=False, root=tmp_path)


def test_entire_launcher_checks_environment_twice_and_runs_bound_guard(tmp_path, monkeypatch):
    import launch_spark_intrinsic_readout_placement_v2 as launcher
    (tmp_path / "env.json").write_text("{}")
    (tmp_path / "protocol.json").write_text("{}")
    packet = {"execution_environment": {"path": "env.json", "sha256": launcher.digest(tmp_path / "env.json")},
              "launcher": {"sha256": launcher.digest(launcher.__file__)},
              "protocol": {"path": "protocol.json", "sha256": launcher.digest(tmp_path / "protocol.json")},
              "output_root": "out", "guard_report": "guard.json", "admission_receipt": "admission.json",
              "prepared_root": "data"}
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet))
    auth = {"execution_environment_sha256": packet["execution_environment"]["sha256"],
            "launcher_sha256": packet["launcher"]["sha256"],
            "approved_packet_sha256": launcher.digest(packet_path),
            "output_root": str(tmp_path / "out"), "guard_report": str(tmp_path / "guard.json")}
    auth_path = tmp_path / "synthetic_auth.json"
    auth_path.write_text(json.dumps(auth))
    calls, checks = [], []
    def environment(*args, **kwargs):
        checks.append(True)
        return {"status": "synthetic_environment_verified"}
    def launched(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(launcher, "validate_environment", environment)
    monkeypatch.setattr(launcher.subprocess, "run", launched)
    assert launcher.run(packet_path, auth_path, launch=True, root=tmp_path) == 0
    assert len(checks) == 2 and len(calls) == 2
    assert calls[0][0][1].endswith("preflight_spark_intrinsic_readout_placement_v2.py")
    assert calls[1][0][1].endswith("run_capacity_preflight_spark_placement_v2.py")
    assert "run_spark_intrinsic_readout_placement_v2.py" in calls[1][0]
    assert calls[1][1]["env"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"


def test_guard_only_adds_new_entrypoint_and_preflight_reports_it():
    root = Path(__file__).resolve().parents[1]
    old = (root / "scripts/run_capacity_preflight.py").read_text()
    new = (root / "scripts/run_capacity_preflight_spark_placement_v2.py").read_text()
    assert new.replace('                                                 "run_spark_intrinsic_readout_placement_v2.py",\n', "").replace('("scripts/run_capacity_preflight_spark_placement_v2.py", "src/replica_lab/capacity_guard.py")', '("scripts/run_capacity_preflight.py", "src/replica_lab/capacity_guard.py")') == old
    import preflight_spark_intrinsic_readout_placement_v2 as preflight
    config = {"launch": {"runner": "scripts/run_spark_intrinsic_readout_placement_v2.py"},
              "dataset": {"root": "data/synthetic"},
              "work_and_cost": {"guards": {"driver_limit_bytes": 50 * 1024**3,
                  "sustained_swap_growth_limit_bytes": 512 * 1024**2,
                  "minimum_host_free_fraction": 0.15, "sustained_host_samples": 3}}}
    command = preflight.guarded_command(config, Path("out"), Path("auth"))
    assert "run_capacity_preflight_spark_placement_v2.py" in command
    assert ".venv-spark-bf16/bin/python" in command
    assert ".venv/bin/python" not in command
    assert "run_spark_intrinsic_readout_placement_v2.py" in command
