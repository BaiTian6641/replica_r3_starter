"""Protocol/preflight contracts; no native model or authorization is created."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from preflight_spark_text_memory_v1 import (
    CONFIG_PATH, ProtocolError, derive_work, preflight, read_json, sha256_file,
    validate_config,
)


def _head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                          capture_output=True, check=True).stdout.strip()


def _valid_launch_inputs(tmp_path: Path):
    config = copy.deepcopy(read_json(CONFIG_PATH))
    implementation = validate_config(config)["implementation_sha256"]
    config["launch"]["protocol_commit"] = _head()
    config["launch"]["implementation_sha256"] = implementation
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"format": "fixture", "rows": []}) + "\n",
                            encoding="utf-8")
    config["dataset"]["manifest"] = str(manifest_path)
    config["dataset"]["manifest_sha256"] = sha256_file(manifest_path)
    backend_path = tmp_path / "backend.json"
    backend = dict(config["launch"]["required_backend_evidence"])
    backend_path.write_text(json.dumps(backend, indent=2) + "\n", encoding="utf-8")
    config["launch"]["backend_evidence_path"] = str(backend_path)
    config_path = tmp_path / "protocol.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    output = tmp_path / "native"
    authorization = {
        "authorization_id": "protocol-test-auth",
        "protocol_id": "spark_text_memory_v1",
        "protocol_sha256": sha256_file(config_path),
        "protocol_commit": _head(),
        "implementation_sha256": implementation,
        "prepared_manifest_sha256": config["dataset"]["manifest_sha256"],
        "source_identity": "official_fp8",
        "execution_format": "official_native_fp8",
        "execution_backend": config["launch"]["required_execution_binding"]["execution_backend"],
        "backend_evidence_sha256": sha256_file(backend_path),
        "attempt_id": "native01",
        "output_root": str(output),
        "issued_by": "protocol-test",
        "expires_at_utc": "2099-01-01T00:00:00+00:00",
    }
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(authorization, indent=2) + "\n",
                                  encoding="utf-8")
    return config_path, authorization_path, output, backend_path


def test_review_and_derived_work_are_explicit_and_non_native():
    report = preflight()
    assert report["status"] == "review_ready_preparation_only"
    assert report["native_execution"] is False
    assert report["launch_allowed"] is False
    assert report["native_command"] is None
    assert "external_authorization_receipt" in report["missing_bindings"]
    assert "native_fp8_backend_evidence" in report["missing_bindings"]
    work = derive_work(read_json(CONFIG_PATH))
    assert work["generation_requests_max"] == 111
    assert work["autoregressive_forwards_max"] == 3552
    assert work["query_sensing_forwards_max"] == 3560
    assert work["native_query_passes_max"] == 3560
    assert work["forward_calls_max"] == 7120
    assert work["processed_text_positions_max"] == 14683680
    assert work["plastic_forward_calls_max"] == 4
    assert work["plastic_backward_calls_max"] == 4
    assert work["optimizer_steps_max"] == 8
    assert work["backward_processed_positions_max"] == 8192
    assert work["plastic_feature_vectors_max"] == 8192
    assert work["plastic_backward_feature_vectors_max"] == 8192
    assert work["checkpoint_writes_max"] == 5
    assert work["verification_calls_max"] == 112


def test_launch_requires_external_authorization():
    with pytest.raises(ProtocolError) as error:
        preflight(launch=True)
    assert error.value.check == "authorization"
    assert "external authorization" in error.value.detail


def test_valid_authorization_constructs_guarded_command_without_native_work(tmp_path):
    config_path, authorization, output, _ = _valid_launch_inputs(tmp_path)
    report = preflight(config_path, authorization=authorization,
                       output=output, launch=True)
    assert report["status"] == "launch_admitted"
    assert report["launch_allowed"] is True
    assert report["native_execution"] is False
    assert "run_capacity_preflight.py" in report["native_command"]
    assert "run_spark_text_memory_v1.py" in report["native_command"]
    assert "--device mps" in report["native_command"]
    assert not output.exists()


def test_wrong_execution_format_is_rejected_before_launch(tmp_path):
    config_path, authorization, output, _ = _valid_launch_inputs(tmp_path)
    payload = read_json(authorization)
    payload["execution_format"] = "official_bf16"
    authorization.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ProtocolError) as error:
        preflight(config_path, authorization=authorization,
                  output=output, launch=True)
    assert error.value.check == "authorization"
    assert "execution binding execution_format" in error.value.detail


def test_stale_implementation_binding_is_rejected(tmp_path):
    config_path, authorization, output, _ = _valid_launch_inputs(tmp_path)
    config = read_json(config_path)
    first = next(iter(config["launch"]["implementation_sha256"]))
    config["launch"]["implementation_sha256"][first] = "0" * 64
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    authorization_payload = read_json(authorization)
    authorization_payload["protocol_sha256"] = sha256_file(config_path)
    authorization.write_text(json.dumps(authorization_payload) + "\n", encoding="utf-8")
    with pytest.raises(ProtocolError) as error:
        preflight(config_path, authorization=authorization,
                  output=output, launch=True)
    assert error.value.check == "authorization"
    assert "config implementation binding differs" in error.value.detail


def test_missing_backend_evidence_is_rejected(tmp_path):
    config_path, authorization, output, backend_path = _valid_launch_inputs(tmp_path)
    backend_path.unlink()
    with pytest.raises(ProtocolError) as error:
        preflight(config_path, authorization=authorization,
                  output=output, launch=True)
    assert error.value.check == "authorization"
    assert "backend evidence is missing" in error.value.detail
