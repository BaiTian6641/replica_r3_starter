"""Retired MiniCPM launch quarantine tests; historical replay stays available."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from dc_protocol_common_20261027 import (  # noqa: E402
    ProtocolError, read_json, seal_record, sha256_file)
import dc_r1_native_value_v2_20261027 as protocol  # noqa: E402
from replica_lab import dc_r1_native_runner as legacy_runner  # noqa: E402
from replica_lab.runner_registry import (  # noqa: E402
    RetiredRunnerError, require_not_retired, retirement_for)
import run_dc_r1_native_value_v2_20261027 as legacy_cli  # noqa: E402


CONFIG = ROOT / "configs/dc_r1_native_value_protocol_v2.json"
PREPARED = ROOT / "data/dc_r1_native_value_v2_20261027"


def _authorization(tmp_path: Path) -> Path:
    body = {
        "format": legacy_runner.AUTHORIZATION_FORMAT,
        "protocol_id": legacy_runner.PROTOCOL_ID,
        "config_sha256": sha256_file(CONFIG),
        "prepared_manifest_sha256": read_json(PREPARED / "manifest.json")["manifest_sha256"],
        "authorized_by": "retirement-fixture",
        "native_execution_allowed": True,
        "single_attempt": True,
        "restart_allowance": 1,
    }
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(seal_record(body), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def test_legacy_preflight_is_retired_not_launch_admitted(tmp_path):
    report = protocol.preflight(
        launch=True, authorization=_authorization(tmp_path),
        native_runner=ROOT / "scripts/run_dc_r1_native_value_v2_20261027.py")
    assert report["status"] == "retired_minicpm_only"
    assert report["next_command"] is None
    assert report["retirement"]["replacement"] == (
        "scripts/run_dc_r1_native_value_v3_20261028.py")
    assert report["native_execution"] is False


def test_retirement_cannot_be_bypassed_with_absolute_or_normalized_paths():
    for path in (ROOT / "scripts/run_dc_r1_native_value_v2_20261027.py",
                 "scripts/./run_dc_r1_native_value_v2_20261027.py"):
        with pytest.raises(RetiredRunnerError):
            require_not_retired(path)


def test_direct_native_api_refuses_retired_runner_before_authorization(tmp_path):
    with pytest.raises(RetiredRunnerError):
        legacy_runner.run_stage(
            "dev", config_path=tmp_path / "missing-config",
            prepared=tmp_path / "missing-prepared",
            authorization_path=tmp_path / "missing-authorization",
            output=tmp_path / "native")


def test_historical_replay_is_explicit_and_authorization_remains_bound(tmp_path):
    authorization = _authorization(tmp_path)
    report = protocol.preflight(
        launch=True, authorization=authorization,
        native_runner=ROOT / "scripts/run_dc_r1_native_value_v2_20261027.py",
        historical_replay=True)
    assert report["status"] == "historical_replay_admitted"
    assert "--historical-replay" in report["next_command"]
    assert report["native_execution"] is False
    require_not_retired(
        "scripts/run_dc_r1_native_value_v2_20261027.py", historical_replay=True)
    tampered = read_json(authorization)
    tampered["receipt_sha256"] = "0" * 64
    authorization.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ProtocolError, match="authorization"):
        protocol.preflight(
            launch=True, authorization=authorization,
            native_runner=ROOT / "scripts/run_dc_r1_native_value_v2_20261027.py",
            historical_replay=True)


def test_cli_marks_explicit_historical_replay(tmp_path, monkeypatch, capsys):
    calls = []

    def replay(stage, **kwargs):
        calls.append((stage, kwargs))
        return {"status": "passed", "admitted": False,
                "historical_replay": kwargs["historical_replay"]}

    monkeypatch.setattr(legacy_cli, "run_stage", replay)
    code = legacy_cli.main([
        "--stage", "dev", "--device", "mps",
        "--output", str(tmp_path / "native"),
        "--authorization", str(_authorization(tmp_path)),
        "--historical-replay"])
    assert code == 4
    assert calls[0][1]["historical_replay"] is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["execution_mode"] == "historical_replay"
    assert payload["historical_replay"] is True


def test_retired_cli_refuses_before_output_or_backend(tmp_path, capsys):
    output = tmp_path / "native"
    code = legacy_cli.main([
        "--stage", "dev", "--device", "mps", "--output", str(output),
        "--authorization", str(_authorization(tmp_path))])
    assert code == 2
    assert not output.exists()
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["check"] == "retired_runner"
    assert refusal["replacement"] == "scripts/run_dc_r1_native_value_v3_20261028.py"


def test_replacement_is_not_retired():
    assert retirement_for("scripts/run_dc_r1_native_value_v3_20261028.py") is None
    require_not_retired("scripts/run_dc_r1_native_value_v3_20261028.py")
