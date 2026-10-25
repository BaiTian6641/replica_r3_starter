"""Offline evaluator routing and guard exit-race tests; no child/model is launched."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("generation_guard", ROOT / "scripts/run_generation_preflight.py")
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


def test_evaluator_arguments_keep_payload_and_require_mps(tmp_path):
    arguments = ["--device", "mps", "--output", str(tmp_path / "eval"), "--split", "train", "--max-examples", "8"]
    routed, output = GUARD.evaluation_arguments(["--", *arguments])
    assert routed == arguments and output == tmp_path / "eval"


@pytest.mark.parametrize("arguments", [
    ["--device", "cpu", "--output", "runs/eval"],
    ["--device", "mps"],
    ["--device", "mps", "--output"],
    ["--device", "mps", "--output", "--split", "train"],
    ["--device", "mps", "--output", "a", "--output", "b"],
    ["--device", "mps", "--device", "mps", "--output", "a"],
])
def test_invalid_device_or_output_arguments_rejected(arguments):
    with pytest.raises(ValueError):
        GUARD.evaluation_arguments(arguments)


def test_child_environment_is_offline_without_mutating_parent():
    parent = {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0", "HF_DATASETS_OFFLINE": "0", "OTHER": "kept"}
    child = GUARD.child_environment(parent)
    assert all(child[key] == "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"))
    assert child["PYTORCH_ENABLE_MPS_FALLBACK"] == "0" and child["OTHER"] == "kept"
    assert parent["HF_HUB_OFFLINE"] == "0" and "PYTORCH_ENABLE_MPS_FALLBACK" not in parent
    with pytest.raises(ValueError, match="CPU fallback is forbidden"):
        GUARD.child_environment({"PYTORCH_ENABLE_MPS_FALLBACK": "1"})


@pytest.mark.parametrize("existing", ["report.json", "output", "report.child.log", "report.host.jsonl"])
def test_existing_evidence_or_output_is_never_reused(tmp_path, existing):
    target = tmp_path / existing
    if existing == "output":
        target.mkdir()  # Even an empty existing output is rejected.
    else:
        target.write_text("preserved")
    with pytest.raises(ValueError, match="fresh report"):
        GUARD.fresh_paths(tmp_path / "report.json", tmp_path / "output")
    assert target.exists()


def test_guard_evidence_cannot_make_evaluator_output_nonempty(tmp_path):
    with pytest.raises(ValueError, match="outside the evaluator output"):
        GUARD.fresh_paths(tmp_path / "output/report.json", tmp_path / "output")
    broken = tmp_path / "broken-output"
    broken.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="symlink"):
        GUARD.evaluation_arguments(["--device", "mps", "--output", str(broken)])


@pytest.mark.parametrize("limit_gib,final_event,expected_status,expected_reason", [
    (None, {"type": "sample", "mps_driver_bytes": 2048}, "completed_within_declared_sampled_limits", None),
    (None, {"type": "sample", "mps_driver_bytes": 40 * 1024**3}, "completed_within_declared_sampled_limits", None),
    (None, {"type": "sample", "mps_driver_bytes": 50 * 1024**3 - 1}, "completed_within_declared_sampled_limits", None),
    (None, {"type": "sample", "mps_driver_bytes": 50 * 1024**3}, "stopped_or_failed", "sampled_mps_driver_limit"),
    (40, {"type": "sample", "mps_driver_bytes": 40 * 1024**3}, "stopped_or_failed", "sampled_mps_driver_limit"),
    (None, "incomplete", "stopped_or_failed", "incomplete_final_profile_record"),
])
def test_mocked_evaluator_routes_offline_and_drains_late_exit_samples(
        tmp_path, monkeypatch, limit_gib, final_event, expected_status, expected_reason):
    output, report = tmp_path / "generation", tmp_path / "guard.json"
    monkeypatch.setattr(sys, "platform", "darwin")
    limit_args = [] if limit_gib is None else ["--driver-limit-gib", str(limit_gib)]
    monkeypatch.setattr(sys, "argv", ["run_generation_preflight.py", *limit_args, "--report", str(report), "--",
                                      "--device", "mps", "--output", str(output), "--split", "train",
                                      "--max-examples", "8", "--max-new-tokens", "256"])
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(GUARD, "host_reading", lambda: {"swap_used_bytes": 0, "memory_free_percent": 90.0})
    captured = {}

    class Child:
        returncode = None

        def __init__(self, command, **kwargs):
            captured.update(command=command, **kwargs)
            output.mkdir()
            self.profile = output / "profile_samples.jsonl"
            self.profile.write_text(json.dumps({"type": "sample", "mps_driver_bytes": 1024}) + "\n")

        def poll(self):
            if self.returncode is None:
                # Simulate the race where final profiler bytes appear only
                # after the wrapper's preceding read observes child exit.
                with self.profile.open("a") as stream:
                    stream.write('{"type":' if final_event == "incomplete" else json.dumps(final_event) + "\n")
                self.returncode = 0
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(GUARD.subprocess, "Popen", Child)
    result = GUARD.main()
    saved = json.loads(report.read_text())
    assert saved["status"] == expected_status and saved["guard"]["stop_reason"] == expected_reason
    assert saved["guard"]["policy"]["driver_limit_bytes"] == (50 if limit_gib is None else limit_gib) * 1024**3
    assert result == (0 if expected_reason is None else 1)
    assert captured["command"][:2] == [sys.executable, str(ROOT / "scripts/evaluate_text.py")]
    assert "scripts/train_text.py" not in " ".join(captured["command"])
    assert captured["env"]["HF_HUB_OFFLINE"] == "1" and captured["env"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    assert saved["guard"]["host_samples"] == 1
    assert saved["guard"]["driver_samples"] == (1 if final_event == "incomplete" else 2)
    assert Path(saved["host_samples"]).is_file() and Path(saved["child_log"]).is_file()
