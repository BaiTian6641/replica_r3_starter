"""Portable environment admission tests; no checkpoint, torch or model import."""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_launcher = Path(__file__).resolve().parents[1] / "scripts/launch_spark_intrinsic_readout_environment_v1.py"
_spec = importlib.util.spec_from_file_location("spark_environment_admission_fixture", _launcher)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


@pytest.fixture
def fixture(tmp_path):
    prefix = tmp_path / "env"
    module = prefix / "lib/transformers/__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("# synthetic executable dependency\n")
    lock = tmp_path / "lock.txt"
    lock.write_text("transformers==4.57.1\ntorch==2.14.1\n")
    manifest = {"format": gate.FORMAT, "environment_directory": "env", "python": "3.12.14",
                "lock": {"path": "lock.txt", "sha256": gate.digest(lock)},
                "packages": {"transformers": "4.57.1", "torch": "2.14.1"},
                "module_origins": {"transformers": "lib/transformers/__init__.py"},
                "critical_files": {"lib/transformers/__init__.py": gate.digest(module)}}
    kwargs = {"root": tmp_path, "prefix": prefix, "python": "3.12.14",
              "version": lambda name: manifest["packages"][name],
              "module_spec": lambda name: SimpleNamespace(origin=str(module))}
    return manifest, kwargs, module


def test_correct_environment_without_model_import(fixture):
    manifest, kwargs, _ = fixture
    result = gate.validate_environment(manifest, **kwargs)
    assert result["status"] == "environment_verified"
    assert result["model_imported"] is False
    assert result["native_execution"] is False


@pytest.mark.parametrize("damage", ["prefix", "python", "version", "missing_package",
                                   "module_shadow", "bytes", "missing_file", "lock", "inventory"])
def test_wrong_environment_and_changed_executable_dependencies_refused(fixture, damage):
    manifest, kwargs, module = fixture
    if damage == "prefix": kwargs["prefix"] = kwargs["root"] / "other_env"
    if damage == "python": kwargs["python"] = "3.13.0"
    if damage == "version": kwargs["version"] = lambda name: "5.20.2"
    if damage == "missing_package":
        def missing(name): raise importlib.metadata.PackageNotFoundError(name)
        kwargs["version"] = missing
    if damage == "module_shadow": kwargs["module_spec"] = lambda name: SimpleNamespace(origin="/elsewhere/module.py")
    if damage == "bytes": module.write_text("# changed implementation\n")
    if damage == "missing_file": module.unlink()
    if damage == "lock": (kwargs["root"] / "lock.txt").write_text("transformers==5.20.2\n")
    if damage == "inventory": manifest["packages"]["unlocked"] = "1"
    with pytest.raises(ValueError):
        gate.validate_environment(manifest, **kwargs)


def packet_fixture(tmp_path):
    manifest = tmp_path / "environment.json"
    manifest.write_text("{}")
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    packet = {"execution_environment": {"path": "environment.json", "sha256": gate.digest(manifest)},
              "launcher": {"sha256": gate.digest(Path(gate.__file__))},
              "protocol": {"path": "protocol.json", "sha256": gate.digest(protocol)},
              "output_root": "output", "guard_report": "guard.json", "admission_receipt": "admission.json",
              "prepared_root": "prepared"}
    path = tmp_path / "packet.json"
    path.write_text(json.dumps(packet))
    auth = {"execution_environment_sha256": gate.digest(manifest),
            "launcher_sha256": gate.digest(Path(gate.__file__)),
            "approved_packet_sha256": gate.digest(path), "output_root": str(tmp_path / "output"),
            "guard_report": str(tmp_path / "guard.json")}
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps(auth))
    return packet, path, auth_path, auth, manifest


@pytest.mark.parametrize("damage", ["missing_env_binding", "wrong_launcher", "wrong_packet", "wrong_output", "old_output", "old_guard", "protocol"])
def test_receipt_does_not_authorize_changed_environment_or_retry(tmp_path, damage):
    packet, path, _, auth, manifest = packet_fixture(tmp_path)
    if damage == "missing_env_binding": auth.pop("execution_environment_sha256")
    if damage == "wrong_launcher": auth["launcher_sha256"] = "0" * 64
    if damage == "wrong_packet": auth["approved_packet_sha256"] = "0" * 64
    if damage == "wrong_output": auth["output_root"] = str(tmp_path / "other")
    if damage == "old_output": (tmp_path / "output").mkdir()
    if damage == "old_guard": (tmp_path / "guard.json").write_text("{}")
    if damage == "protocol": (tmp_path / "protocol.json").write_text('{"modified":true}')
    with pytest.raises(ValueError):
        gate.validate_launch(packet, auth, packet_path=path, manifest_path=manifest, root=tmp_path)


def test_wrong_runtime_stops_before_preflight_or_guard(tmp_path, monkeypatch):
    _, path, auth, _, _ = packet_fixture(tmp_path)
    def refuse(*args, **kwargs): raise ValueError("wrong interpreter environment")
    def forbidden(*args, **kwargs): pytest.fail("subprocess launched before environment admission")
    monkeypatch.setattr(gate, "validate_environment", refuse)
    monkeypatch.setattr(gate.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="wrong interpreter"):
        gate.run(path, auth, launch=True, root=tmp_path)


def test_valid_bound_launch_uses_same_interpreter_and_stops_if_source_preflight_fails(tmp_path, monkeypatch):
    packet, path, auth, _, _ = packet_fixture(tmp_path)
    calls = []
    monkeypatch.setattr(gate, "validate_environment", lambda *args, **kwargs: {"status": "environment_verified"})
    monkeypatch.setattr(gate.sys, "executable", "/fixture/env/bin/python")
    def preflight_fails(command, **kwargs):
        calls.append((command, kwargs));return SimpleNamespace(returncode=2)
    monkeypatch.setattr(gate.subprocess, "run", preflight_fails)
    assert gate.run(path, auth, launch=True, root=tmp_path) == 2
    assert len(calls) == 1 and calls[0][0][0] == "/fixture/env/bin/python"
    assert "preflight_spark_persistent_learner_derivative_bf16.py" in calls[0][0][1]
    assert calls[0][1]["env"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"


def test_environment_rechecked_and_guard_keeps_resource_limits(tmp_path, monkeypatch):
    packet, path, auth, _, _ = packet_fixture(tmp_path)
    calls, checks = [], []
    def environment(*args, **kwargs): checks.append(True);return {"status": "environment_verified"}
    def success(command, **kwargs): calls.append(command);return SimpleNamespace(returncode=0)
    monkeypatch.setattr(gate, "validate_environment", environment)
    monkeypatch.setattr(gate.subprocess, "run", success)
    assert gate.run(path, auth, launch=True, root=tmp_path) == 0
    assert len(checks) == 2 and len(calls) == 2
    assert calls[0][0] == calls[1][0] == gate.sys.executable
    guard = calls[1]
    for flag, value in [("--driver-limit-gib", "50.0"), ("--swap-growth-limit-mib", "512.0"),
                        ("--minimum-free-percent", "15.0"), ("--sustained-host-samples", "3"), ("--stage", "readout")]:
        assert guard[guard.index(flag)+1] == value


def test_review_never_launches_and_missing_authorization_refuses(tmp_path, monkeypatch):
    _, path, _, _, _ = packet_fixture(tmp_path)
    monkeypatch.setattr(gate, "validate_environment", lambda *args, **kwargs: {"status": "environment_verified"})
    monkeypatch.setattr(gate.subprocess, "run", lambda *args, **kwargs: pytest.fail("review launched a subprocess"))
    assert gate.run(path, root=tmp_path) == 0
    with pytest.raises(ValueError, match="human authorization"):
        gate.run(path, launch=True, root=tmp_path)
