"""Portable protocol and access-boundary tests for persistent learner v2."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import preflight_spark_persistent_learner_v2 as preflight_module
from preflight_spark_persistent_learner_v2 import ProtocolError, preflight, validate_config
from replica_lab.persistent_learner_runner import PreparedDataset, Protocol, Refused

CONFIG_PATH = ROOT / "configs/spark_persistent_learner_protocol_v2.json"
import run_capacity_preflight
HEAD = "a" * 40

from replica_lab.persistent_learner_runner import PreparedDataset, Protocol, Refused, run_protocol
def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def tracked_metadata_mocks(monkeypatch):
    """Metadata-only mocks; never authenticate a native source artifact."""
    source = read_json(CONFIG_PATH)["source_identities"]["official_fp8"]

    def validate(config, root, *, full_weight_hash):
        return {
            "source_identity": "official_fp8",
            "source_root": str(root / "tracked-source-metadata"),
            "config_sha256": source["config_sha256"],
            "weight_path": str(root / "tracked-source-metadata/model.safetensors"),
            "weight_sha256_checked": bool(full_weight_hash),
            "execution_format": source["execution_format"],
            "required_backend": source["required_backend"],
            "tokenizer_checked": True,
        }

    monkeypatch.setattr(preflight_module, "_source_validation", validate)
    monkeypatch.setattr(preflight_module, "_current_head", lambda root: HEAD)
    real_run = subprocess.run

    def run(args, **kwargs):
        if args[:2] in (["git", "merge-base"], ["git", "diff"]):
            return subprocess.CompletedProcess(args, 0)
        return real_run(args, **kwargs)

    monkeypatch.setattr(preflight_module.subprocess, "run", run)


def test_review_derives_real_profile_and_control_budgets_without_native_work():
    report = preflight()
    assert report["status"] == "review_ready_preparation_only"
    assert report["native_execution"] is False
    assert report["model_loaded"] is False
    assert report["source"]["source_identity"] == "official_fp8"
    assert report["source"]["execution_format"] == "official_native_fp8"
    assert report["work_units"]["forward_calls_max"] == 8921
    assert report["work_units"]["processed_text_positions_max"] == 18408096
    assert report["work_units"]["verification_calls_max"] == 155
    assert report["work_units"]["competence_cases"] == 8
    assert report["work_units"]["optimizer_steps_max"] == 8
    assert report["work_units"]["response_port_optimizer_steps_max"] == 4
    assert report["work_units"]["selected_P_optimizer_steps_max"] == 4
    assert "native_fp8_backend_evidence" in report["missing_bindings"]
    assert "external_authorization_receipt" in report["missing_bindings"]
    assert not any("materialize" in item for item in report["remaining_prerequisites"])
    config = read_json(CONFIG_PATH)
    assert set(config["launch"]["implementation_files"]) == set(report["implementation_sha256"])
    assert config["launch"]["implementation_sha256"] == report["implementation_sha256"]
    assert {
        "src/replica_lab/architecture/ports.py",
        "src/replica_lab/model_identity.py",
        "src/replica_lab/training_state.py",
        "scripts/preflight_spark_text_memory_v1.py",
    } <= set(config["launch"]["implementation_files"])
    assert report["work_units"]["readout_selection_calls_max"] == 296
    assert report["work_units"]["readout_file_reads_max"] == 66304
    assert report["work_units"]["payload_file_reads_max"] == 23680
    assert report["work_units"]["readout_cache_bytes_max"] == 2097152
    assert report["work_units"]["readout_result_bytes_max"] == 73728


def test_independent_fixture_is_hash_bound_and_labels_stay_behind_fit_receipt():
    protocol = Protocol.load(CONFIG_PATH)
    dataset = PreparedDataset.load(
        ROOT / "data/spark_persistent_learner_v2",
        expected_sha256=protocol.config["dataset"]["manifest_sha256"],
        protocol_sha256=protocol.sha256)
    assert len(dataset.observations()) == 13
    assert len(dataset.fit_events()) == 4
    with pytest.raises(Refused, match="development_labels_closed"):
        dataset.development_cases()
    receipt = {
        "format": "replica.spark_persistent_learner.fit_receipt.v2",
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "prepared_manifest_sha256": dataset.manifest_sha256,
        "status": "fit_executed", "execution_complete": True,
        "competence": __import__("test_baseline_competence").passing_evidence(),
    }
    dataset.open_development(receipt, protocol=protocol)
    cases = dataset.development_cases()
    assert len(cases) == 12
    assert {case["expected"] for case in cases} >= {"ready", "unknown", "closed", "standby"}
    assert dataset.manifest["sealed_final"] == {"materialized": False, "opened": False, "path": None}


def _valid_launch_inputs(tmp_path: Path):
    config = copy.deepcopy(read_json(CONFIG_PATH))
    config["launch"]["protocol_commit"] = HEAD
    config["launch"]["implementation_sha256"] = validate_config(config)["implementation_sha256"]
    backend_path = tmp_path / "backend.json"
    backend_path.write_text(json.dumps(config["launch"]["required_backend_evidence"]), encoding="utf-8")
    config["launch"]["backend_evidence_path"] = str(backend_path)
    config_path = tmp_path / "protocol.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output = tmp_path / "native01"
    receipt = {
        "authorization_id": "auth-v2-test", "attempt_id": "attempt-v2-test",
        "protocol_id": config["protocol_id"], "protocol_sha256": sha256_file(config_path),
        "protocol_commit": HEAD, "implementation_sha256": config["launch"]["implementation_sha256"],
        "prepared_manifest_sha256": config["dataset"]["manifest_sha256"],
        "source_identity": "official_fp8", "execution_format": "official_native_fp8",
        "execution_backend": config["launch"]["required_execution_binding"]["execution_backend"],
        "backend_evidence_sha256": sha256_file(backend_path), "output_root": str(output),
        "issued_by": "protocol-test", "expires_at_utc": (
            datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
    }
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(receipt), encoding="utf-8")
    return config_path, authorization_path, output, backend_path


def test_valid_authorization_constructs_guarded_v2_command_without_native_work(tmp_path: Path):
    config_path, authorization, output, _ = _valid_launch_inputs(tmp_path)
    report = preflight(config_path, authorization=authorization, output=output, launch=True)
    assert report["launch_allowed"] is True
    assert report["weight_sha256_checked"] is True
    assert "run_spark_persistent_learner_v2.py" in report["native_command"]
    assert "--device mps" in report["native_command"]
    assert "official_bf16" not in report["native_command"]


def test_wrong_execution_backend_is_rejected_before_launch(tmp_path: Path):
    config_path, authorization, output, _ = _valid_launch_inputs(tmp_path)
    receipt = read_json(authorization)
    receipt["execution_backend"] = "bf16_substitute"
    authorization.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ProtocolError, match="execution binding execution_backend"):
        preflight(config_path, authorization=authorization, output=output, launch=True)


def test_missing_backend_evidence_is_rejected_before_launch(tmp_path: Path):
    config_path, authorization, output, backend_path = _valid_launch_inputs(tmp_path)
    backend_path.unlink()
    with pytest.raises(ProtocolError, match="native FP8 backend evidence is missing"):
        preflight(config_path, authorization=authorization, output=output, launch=True)


def test_restart_output_reuse_requires_explicit_authenticated_stage(tmp_path: Path):
    config_path, authorization, output, _ = _valid_launch_inputs(tmp_path)
    output.mkdir()
    with pytest.raises(ProtocolError, match="native output must be fresh"):
        preflight(config_path, authorization=authorization, output=output, launch=True)
    config = read_json(config_path)
    receipt = preflight_module._validate_authorization(
        config, authorization, output, ROOT,
        config["launch"]["implementation_sha256"], config_path=config_path,
        allow_existing_output=True)
    assert receipt["protocol_id"] == "spark_persistent_learner_v2"


def test_native_execution_requires_watchdog_and_restart_is_a_continuation(tmp_path: Path, monkeypatch):
    for name in ("REPLICA_CAPACITY_GUARD", "REPLICA_NATIVE_SUPERVISED_CHILD"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(Refused, match="capacity_guard"):
        run_protocol(CONFIG_PATH, ROOT / "data/spark_persistent_learner_v2",
                     tmp_path / "native", backend_kind="native", device="mps")
    assert run_capacity_preflight.continuation_requested(["--stage", "restart"]) is True
    assert run_capacity_preflight.continuation_requested(["--stage", "development"]) is True
    assert run_capacity_preflight.continuation_requested(["--stage", "fit"]) is False


def test_tiered_dependency_change_invalidates_authorization(tmp_path: Path):
    config_path, authorization, output, _ = _valid_launch_inputs(tmp_path)
    config = read_json(config_path)
    assert "src/replica_lab/tiered_memory.py" in config["launch"]["implementation_files"]
    assert "src/replica_lab/tiered_readout.py" in config["launch"]["implementation_files"]
    receipt = read_json(authorization)
    receipt["implementation_sha256"]["src/replica_lab/tiered_memory.py"] = "0" * 64
    authorization.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ProtocolError, match="implementation hashes differ"):
        preflight(config_path, authorization=authorization, output=output, launch=True)
