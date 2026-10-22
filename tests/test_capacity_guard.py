import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from replica_lab.capacity_guard import CapacityGuard, CapacityPolicy, parse_host_readings


def baseline():
    return {"swap_used_bytes": 100, "memory_free_percent": 80}


def test_host_units_and_missing_readings():
    assert parse_host_readings("used = 1.5G", "System-wide memory free percentage: 61%") == {
        "swap_used_bytes": 1610612736, "memory_free_percent": 61.0}
    with pytest.raises(ValueError):
        parse_host_readings("unavailable", "")


def test_swap_requires_consecutive_growth_and_resets_after_recovery():
    guard = CapacityGuard(CapacityPolicy(1000, 50, 15, 3), baseline())
    for swap in [150, 151, 149, 150, 150]:
        assert guard.observe_host({"swap_used_bytes": swap, "memory_free_percent": 80}) is None
    assert guard.observe_host({"swap_used_bytes": 150, "memory_free_percent": 80}) == "sustained_host_swap_growth"
    assert guard.peak_swap_growth_bytes == 51


def test_host_pressure_and_driver_have_separate_thresholds():
    guard = CapacityGuard(CapacityPolicy(1000, 50, 15, 2), baseline())
    assert guard.observe_host({"swap_used_bytes": 100, "memory_free_percent": 15}) is None
    assert guard.observe_host({"swap_used_bytes": 100, "memory_free_percent": 14}) == "sustained_low_host_free_percentage"
    guard = CapacityGuard(CapacityPolicy(1000, 50, 15, 2), baseline())
    assert guard.observe_profile({"type": "phase", "memory_sample": {"mps_driver_bytes": 999}}) is None
    assert guard.observe_profile({"type": "sample", "mps_driver_bytes": 1000}) == "sampled_mps_driver_limit"
    assert guard.report()["peak_driver_bytes"] == 1000


def test_guard_rejects_invalid_telemetry_and_initial_pressure():
    assert CapacityGuard(CapacityPolicy(), {"swap_used_bytes": 0, "memory_free_percent": 15}).stop_reason
    with pytest.raises(ValueError):
        CapacityGuard(CapacityPolicy(), {"swap_used_bytes": None, "memory_free_percent": 50})
    with pytest.raises(ValueError):
        CapacityPolicy(driver_limit_bytes=-1)
    with pytest.raises(ValueError):
        CapacityGuard(CapacityPolicy(), baseline()).observe_profile({"mps_driver_bytes": True})


def test_watchdog_argument_binding_rejects_other_backend_and_duplicate_output():
    path = Path(__file__).parents[1] / "scripts/run_capacity_preflight.py"
    spec = importlib.util.spec_from_file_location("capacity_preflight", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    args, output = script.training_arguments(["--", "--device", "mps", "--output", "runs/test"])
    assert args[0] == "--device" and output.name == "test"
    for arguments in (["--device", "cpu", "--output", "out"],
                      ["--device", "mps", "--output", "a", "--output", "b"]):
        with pytest.raises(ValueError):
            script.training_arguments(arguments)


def load_script():
    path = Path(__file__).parents[1] / "scripts/run_capacity_preflight.py"
    spec = importlib.util.spec_from_file_location("capacity_failure_test", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_unavailable_host_telemetry_preserves_failure_and_never_launches(monkeypatch, tmp_path):
    script = load_script()
    report = tmp_path / "failure.json"
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "argv", ["guard", "--report", str(report), "--",
                                    "--device", "mps", "--output", str(tmp_path / "run")])
    def unavailable():
        raise RuntimeError("host telemetry denied")
    def forbidden(*args, **kwargs):
        raise AssertionError("Child must not start without host baseline")
    monkeypatch.setattr(script, "host_reading", unavailable)
    monkeypatch.setattr(script.subprocess, "Popen", forbidden)
    with pytest.raises(RuntimeError, match="telemetry denied"):
        script.main()
    result = json.loads(report.read_text())
    assert result["status"] == "guard_or_child_failed"
    assert "child_exit_code" not in result


def test_guard_interrupt_reaps_a_live_child():
    script = load_script()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        script.stop_child(child)
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_relative_output_is_bound_to_child_workspace(monkeypatch, tmp_path):
    script = load_script()
    monkeypatch.chdir(tmp_path)
    _, output = script.training_arguments(["--device", "mps", "--output", "runs/test"])
    assert output == (script.ROOT / "runs/test").resolve()


def test_final_profile_write_at_child_exit_cannot_hide_a_driver_breach(monkeypatch, tmp_path):
    script = load_script()
    report, output = tmp_path / "guard.json", tmp_path / "run"
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "argv", ["guard", "--report", str(report), "--",
                                    "--device", "mps", "--output", str(output)])
    monkeypatch.setattr(script, "host_reading", baseline)
    class ExitsAfterFinalWrite:
        returncode = 0
        def __init__(self, *args, **kwargs):
            output.mkdir()
            self.profile = output / "profile_samples.jsonl"
            self.profile.write_text(json.dumps({"mps_driver_bytes": 1024**3}) + "\n")
            self.written = False
        def poll(self):
            if not self.written:
                with self.profile.open("a") as stream:
                    stream.write(json.dumps({"mps_driver_bytes": 41 * 1024**3}) + "\n")
                self.written = True
            return 0
        def wait(self, **kwargs):
            return 0
    monkeypatch.setattr(script.subprocess, "Popen", ExitsAfterFinalWrite)
    assert script.main() == 1
    result = json.loads(report.read_text())
    assert result["status"] == "stopped_or_failed"
    assert result["guard"]["peak_driver_bytes"] == 41 * 1024**3
    assert result["guard"]["stop_reason"] == "sampled_mps_driver_limit"
    assert result["child_exit_code"] == 0
