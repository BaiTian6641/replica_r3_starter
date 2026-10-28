"""Harmless execution of the emitted DC-R1 v3 capacity-guard command."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from replica_lab import dc_r1_v3_runner as runner
import dc_r1_v3_fixtures as fixture
import preflight_dc_r1_native_value_v3_20261028 as preflight


def _emitted_run(tmp_path: Path, mode: str) -> tuple[dict, Path, Path]:
    output = tmp_path / "native01"
    authorization = fixture.authorization(tmp_path, output)
    context, _ = runner.verify_authorization(
        config_path=fixture.CONFIG, prepared=fixture.PREPARED,
        authorization_path=authorization, output=output,
        require_head=False, commit_verifier=fixture.commit_verifier,
        register=False)
    context.authorization["path"] = str(authorization)
    command = preflight._command(context, "dev", tmp_path / "preflight_receipt.json")
    root = tmp_path / "guard_root"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts/run_capacity_preflight.py", root / "scripts/run_capacity_preflight.py")
    shutil.copy(ROOT / "tests/fixtures/dc_r1_guard_entry.py",
                root / "scripts/run_dc_r1_native_value_v3_20261028.py")
    (root / "src/replica_lab").mkdir(parents=True)
    shutil.copy(ROOT / "src/replica_lab/capacity_guard.py",
                root / "src/replica_lab/capacity_guard.py")
    shutil.copy(ROOT / "src/replica_lab/__init__.py",
                root / "src/replica_lab/__init__.py")
    parts = shlex.split(command)[1:]
    parts[0] = sys.executable
    separator = parts.index("--")
    parts[separator:separator] = ["--telemetry-timeout-seconds", "0.2"]
    parts.extend(["--mode", mode])
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        parts, cwd=root, env=environment, text=True, capture_output=True,
        timeout=20, check=False)
    report_path = tmp_path / "guard_dev01.json"
    assert report_path.is_file(), completed.stdout + completed.stderr
    report = json.loads(report_path.read_text())
    return report, output, completed


def test_emitted_guard_command_completes_with_root_telemetry(tmp_path):
    report, output, completed = _emitted_run(tmp_path, "success")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert report["status"] == "completed_within_declared_sampled_limits"
    assert report["guard"]["driver_samples"] == 1
    assert (output / "profile_samples.jsonl").is_file()
    assert not (output / "dev/profile_samples.jsonl").exists()


@pytest.mark.parametrize("mode", ["missing", "stale"])
def test_emitted_guard_command_stops_missing_or_stale_telemetry(tmp_path, mode):
    report, output, completed = _emitted_run(tmp_path, mode)
    assert completed.returncode == 1
    assert report["status"] == "stopped_or_failed"
    assert report["guard"]["stop_reason"] == "mps_driver_telemetry_stale_or_missing"
    assert report["guard"]["driver_samples"] == (0 if mode == "missing" else 1)
    shutdown = json.loads((output / "shutdown.json").read_text())
    assert shutdown["signal"] in (2, 15)
    assert (tmp_path / "guard_dev01.child.log").is_file()
    assert (tmp_path / "guard_dev01.host.jsonl").is_file()
