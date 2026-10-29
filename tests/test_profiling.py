"""Profiler units and failure behavior without a real accelerator allocation."""
import json
import threading
from types import SimpleNamespace

import pytest

from replica_lab import profiling
from replica_lab.profiling import RunProfiler, _parse_swapusage, _ru_maxrss_bytes


def fake_commands(args):
    if args[0] == "/bin/ps":
        return "1234"
    if args[0] == "/usr/sbin/sysctl":
        return "total = 1024.00M  used = 12.50M  free = 1011.50M  (encrypted)"
    if args[0] == "/usr/bin/memory_pressure":
        return "System-wide memory free percentage: 74%"
    raise AssertionError(args)


def prepare(profiler, monkeypatch, platform="darwin"):
    monkeypatch.setattr(profiling.sys, "platform", platform)
    monkeypatch.setattr(profiling.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=9876543))
    monkeypatch.setattr(profiler, "_command", fake_commands)
    return profiler


def fake_torch_mps(mps, available=True):
    return SimpleNamespace(mps=mps, backends=SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: available)))


def test_rss_platform_units_and_swap_parser():
    assert _ru_maxrss_bytes(7, "darwin") == 7
    assert _ru_maxrss_bytes(7, "linux") == 7 * 1024
    assert _parse_swapusage("total = 1.00G used = 1.50M free = 2K") == {
        "total_bytes": 1024**3, "used_bytes": int(1.5 * 1024**2), "free_bytes": 2048}
    assert _parse_swapusage("unavailable") == {}


def test_cpu_context_writes_samples_summary_host_state_and_phases(tmp_path, monkeypatch):
    profiler = prepare(RunProfiler("cpu", tmp_path, sample_interval=10), monkeypatch)
    with profiler as active:
        assert active is profiler
        event = profiler.mark("model_loaded", parameters=42)
        assert event["fields"] == {"parameters": 42}
        assert event["elapsed_seconds"] >= 0
    report = profiler.stop()
    assert report is profiler.stop()
    assert report["sample_count"] == 3
    assert report["memory"] == {"rss_peak_process_lifetime_bytes": 9876543,
                                 "rss_peak_sampled_bytes": 1234 * 1024}
    assert report["host_start"]["swapusage"]["used_bytes"] == int(12.5 * 1024**2)
    assert report["host_end"]["memory_free_percent"] == 74
    assert report["phases"][0]["phase"] == "model_loaded"
    assert report["errors"] == []
    assert not profiler._thread.is_alive()
    assert profiler._stream.closed
    assert json.loads((tmp_path / "profile_summary.json").read_text()) == report
    events = [json.loads(line) for line in (tmp_path / "profile_samples.jsonl").read_text().splitlines()]
    assert [event["type"] for event in events] == [
        "host_start", "sample", "sample", "phase", "sample", "host_end"]


def test_supervised_restart_appends_without_losing_parent_samples(tmp_path, monkeypatch):
    with prepare(RunProfiler("cpu", tmp_path, sample_interval=10), monkeypatch):
        pass
    previous = (tmp_path / "profile_samples.jsonl").read_bytes()
    with prepare(RunProfiler("cpu", tmp_path, sample_interval=10, append=True), monkeypatch):
        pass
    combined = (tmp_path / "profile_samples.jsonl").read_bytes()
    assert combined.startswith(previous) and len(combined) > len(previous)
    assert sum(json.loads(line)["type"] == "host_start" for line in combined.splitlines()) == 2


def test_mps_sampled_peaks_and_main_thread_synchronization(tmp_path, monkeypatch):
    calls = []
    allocated = iter([100, 200, 50])
    driver = iter([300, 400, 350])
    mps = SimpleNamespace(
        current_allocated_memory=lambda: next(allocated),
        driver_allocated_memory=lambda: next(driver),
        recommended_max_memory=lambda: 500,
        synchronize=lambda: calls.append(threading.current_thread().name))
    profiler = prepare(RunProfiler("mps", tmp_path, sample_interval=10), monkeypatch)
    profiler._torch = fake_torch_mps(mps)
    profiler.start()
    assert calls == []
    profiler.mark("forward_complete")
    report = profiler.stop()
    assert report["memory"]["mps_allocated_peak_sampled_bytes"] == 200
    assert report["memory"]["mps_driver_peak_sampled_bytes"] == 400
    assert report["mps_recommended_max_memory_bytes"] == 500
    assert calls == [threading.current_thread().name] * 2
    assert not any("physical" in key for key in report["memory"])
    assert any("overlap" in note for note in report["notes"])


def test_cuda_uses_reset_actual_peak_api_and_device(tmp_path, monkeypatch):
    calls = []

    def record(name, result=None):
        def method(device):
            calls.append((name, device))
            return result
        return method

    cuda = SimpleNamespace(**{name: record(name, value) for name, value in {
        "reset_peak_memory_stats": None, "synchronize": None,
        "memory_allocated": 100, "memory_reserved": 200,
        "max_memory_allocated": 400, "max_memory_reserved": 500}.items()})
    cuda.is_available = lambda: True
    profiler = prepare(RunProfiler("cuda:1", tmp_path, sample_interval=10), monkeypatch,
                       platform="linux")
    profiler._torch = SimpleNamespace(cuda=cuda)
    with profiler:
        profiler.mark("step")
    assert calls[0] == ("reset_peak_memory_stats", "cuda:1")
    assert all(device == "cuda:1" for _, device in calls)
    report = profiler.stop()
    assert report["memory"]["cuda_allocated_peak_bytes"] == 400
    assert report["memory"]["cuda_reserved_peak_bytes"] == 500
    assert report["memory"]["rss_peak_process_lifetime_bytes"] == 9876543 * 1024


def test_backend_and_rss_failures_are_telemetry_and_preserve_training_exception(tmp_path, monkeypatch):
    def fail():
        raise RuntimeError("counter unavailable")

    profiler = prepare(RunProfiler("mps", tmp_path, sample_interval=10), monkeypatch)
    profiler._torch = fake_torch_mps(SimpleNamespace(
        current_allocated_memory=fail, driver_allocated_memory=fail,
        recommended_max_memory=fail, synchronize=fail))
    monkeypatch.setattr(profiler, "_command", lambda args: fail())
    with pytest.raises(ValueError, match="training failure"):
        with profiler:
            profiler.mark("before_failure")
            raise ValueError("training failure")
    report = profiler.stop()
    operations = {error["operation"] for error in report["errors"]}
    assert {"mps_allocated_bytes", "mps_driver_bytes", "ps_current_rss",
            "mps_synchronize", "sysctl_vm_swapusage", "memory_pressure_query"} <= operations
    assert all(error["count"] >= 1 for error in report["errors"])
    assert report["sample_count"] == 3
    assert not profiler._thread.is_alive()


def test_unavailable_mps_skips_native_apis_that_can_crash_before_initialization(tmp_path, monkeypatch):
    profiler = prepare(RunProfiler("mps", tmp_path, sample_interval=10), monkeypatch)
    # No native methods exist: no call should even try to access one.
    profiler._torch = fake_torch_mps(SimpleNamespace(), available=False)
    with profiler:
        profiler.mark("cpu_loading")
    report = profiler.stop()
    assert {error["operation"] for error in report["errors"]} == {"mps_unavailable"}
    assert "mps_allocated_peak_sampled_bytes" not in report["memory"]


def test_rss_is_throttled_but_final_sample_is_fresh(tmp_path, monkeypatch):
    profiler = prepare(RunProfiler("cpu", tmp_path, sample_interval=10), monkeypatch)
    calls = []

    def command(args):
        calls.append(args)
        return fake_commands(args)

    monkeypatch.setattr(profiler, "_command", command)
    profiler.start()
    profiler.mark("one")
    profiler.mark("two")
    assert sum(args[0] == "/bin/ps" for args in calls) == 1
    profiler.stop()
    assert sum(args[0] == "/bin/ps" for args in calls) == 2


def test_background_sampler_stops_promptly_without_device_synchronization(tmp_path, monkeypatch):
    profiler = prepare(RunProfiler("cpu", tmp_path, sample_interval=0.005), monkeypatch)
    sampled = threading.Event()
    original = profiler._sample

    def sample(**kwargs):
        result = original(**kwargs)
        if threading.current_thread().name == "run-memory-profiler":
            sampled.set()
        return result

    monkeypatch.setattr(profiler, "_sample", sample)
    with profiler:
        assert sampled.wait(timeout=2)
    assert not profiler._thread.is_alive()
    assert profiler.stop()["sample_count"] >= 3


def test_invalid_interval_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        RunProfiler("cpu", tmp_path, sample_interval=0)
