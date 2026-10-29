"""DC-R1 v3 staged admission, cumulative accounting, and recovery tests."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import shutil
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from dc_protocol_common_20261027 import ProtocolError, read_json
from replica_lab import dc_r1_v3_runner as runner
from replica_lab import dc_r1_v3_scoring as scoring
from replica_lab import artifact_publication as publication
import dc_r1_v3_fixtures as fixture
import preflight_dc_r1_native_value_v3_20261028 as preflight
import run_capacity_preflight as capacity


def _dev(tmp_path, *, native=True, backend=None, fail_after=None,
         immediate_eos=False):
    output = tmp_path / "native01"
    authorization = fixture.authorization(tmp_path, output)
    backend = backend or fixture.FakeBackend(native=native, fail_after=fail_after,
                                               immediate_eos=immediate_eos)
    if native:
        def factory(context, stage):
            return backend
        original = runner.build_native_backend
        runner.build_native_backend = factory
        try:
            result = runner.run_dev(
                config_path=fixture.CONFIG, prepared=fixture.PREPARED,
                authorization_path=authorization, output=output,
                require_head=False, commit_verifier=fixture.commit_verifier)
        finally:
            runner.build_native_backend = original
    else:
        result = runner.run_dev(
            config_path=fixture.CONFIG, prepared=fixture.PREPARED,
            authorization_path=authorization, output=output, backend=backend,
            require_head=False, commit_verifier=fixture.commit_verifier)
    return output, authorization, backend, result


def _final(output, authorization, backend, monkeypatch):
    def factory(context, stage):
        return backend

    def decoder(context, injected, ledger):
        ledger.reserve({"tokenizer_loads": 1})
        ledger.charge({"tokenizer_loads": 1})
        return backend.decode_fn

    monkeypatch.setattr(runner, "build_native_backend", factory)
    monkeypatch.setattr(runner, "_load_final_decoder", decoder)
    return runner.run_sealed_final(
        config_path=fixture.CONFIG, prepared=fixture.PREPARED,
        authorization_path=authorization, output=output,
        require_head=False, commit_verifier=fixture.commit_verifier)


def test_native_dev_and_final_share_attempt_and_cumulative_ledger(tmp_path, monkeypatch):
    output, authorization, dev_backend, dev_result = _dev(tmp_path)
    assert dev_result["admitted"] is True
    dev_report = read_json(output / "dev/score_report.json")
    assert dev_report["format"] == scoring.SCORE_FORMAT
    assert dev_report["native_execution_admissible"] is True
    assert dev_report["physical_work"]["totals"]["generation_requests"] == 180
    assert dev_report["physical_work"]["totals"]["forward_calls"] == 360

    final_backend = fixture.FakeBackend(native=True)
    final_result = _final(output, authorization, final_backend, monkeypatch)
    assert final_result["admitted"] is True
    final_report = read_json(output / "sealed_final/score_report.json")
    assert final_report["final_labels_opened"] is True
    assert final_report["final_admission"]["native_development_evidence"] is True
    assert final_report["attempt_id"] == dev_report["attempt_id"]
    ledger = read_json(output / "work_ledger.json")
    assert ledger["attempt_id"] == dev_report["attempt_id"]
    assert ledger["cumulative"]["model_loads"] == 2
    assert ledger["cumulative"]["tokenizer_loads"] == 3
    assert ledger["cumulative"]["generation_requests"] == 240
    assert ledger["cumulative"]["forward_calls"] == 480
    assert ledger["cumulative"]["generated_tokens"] == 480
    assert ledger["cumulative"]["processed_positions"] == 720
    assert ledger["recovery"] == {}
    assert final_backend.closed is True


def test_cpu_dev_is_not_native_and_cannot_open_final(tmp_path):
    output, authorization, backend, result = _dev(tmp_path, native=False)
    assert result["admitted"] is False
    report = read_json(output / "dev/score_report.json")
    assert report["evidence_class"] == "cpu_fixture"
    assert report["development_evidence_admissible"] is True
    assert report["native_execution_admissible"] is False
    final_backend = fixture.FakeBackend(native=False)
    with pytest.raises(ProtocolError, match="native evidence"):
        runner.run_sealed_final(
            config_path=fixture.CONFIG, prepared=fixture.PREPARED,
            authorization_path=authorization, output=output,
            backend=final_backend, require_head=False,
            commit_verifier=fixture.commit_verifier)
    assert final_backend.calls == []
    assert not (output / "sealed_final/score_report.json").exists()


def test_immediate_eos_is_retained_empty_failure_with_cost(tmp_path):
    output, _, _, result = _dev(tmp_path, immediate_eos=True)
    assert result["admitted"] is False
    report = read_json(output / "dev/score_report.json")
    assert report["physical_work"]["totals"]["forward_calls"] == 180
    assert report["physical_work"]["totals"]["generated_tokens"] == 180
    assert report["physical_work"]["totals"]["generation_decode_forwards"] == 0
    first = report["per_case_outputs"][0]["layers"][0]
    assert first["raw_generated_ids"] == [1]
    assert first["terminal_stop_ids"] == [1]
    assert first["generated_ids"] == []
    assert first["generated_text"] == ""
    assert first["empty_generation"] is True
    assert first["correct"] is False
    assert first["completion_valid"] is True


def test_interrupted_work_and_recovery_replay_are_separate(tmp_path):
    output = tmp_path / "native01"
    authorization = fixture.authorization(tmp_path, output)
    first = fixture.FakeBackend(native=True, fail_after=4)
    original = runner.build_native_backend
    runner.build_native_backend = lambda context, stage: first
    try:
        with pytest.raises(RuntimeError, match="injected interruption"):
            runner.run_dev(
                config_path=fixture.CONFIG, prepared=fixture.PREPARED,
                authorization_path=authorization, output=output,
                require_head=False, commit_verifier=fixture.commit_verifier)
    finally:
        runner.build_native_backend = original
    failed = read_json(output / "dev/journal.json")
    assert failed["status"] == "failed"
    assert len(failed["trajectories"]) == 9
    ledger = read_json(output / "work_ledger.json")
    assert ledger["failed"]["failed_forward_calls"] == 1
    assert ledger["failed"]["failed_processed_positions"] == 1

    resumed = fixture.FakeBackend(native=True)
    runner.build_native_backend = lambda context, stage: resumed
    try:
        result = runner.run_dev(
            config_path=fixture.CONFIG, prepared=fixture.PREPARED,
            authorization_path=authorization, output=output, resume=True,
            require_head=False, commit_verifier=fixture.commit_verifier)
    finally:
        runner.build_native_backend = original
    assert result["admitted"] is True
    ledger = read_json(output / "work_ledger.json")
    assert ledger["cumulative"]["restarts"] == 1
    assert ledger["cumulative"]["model_loads"] == 2
    assert ledger["cumulative"]["generation_requests"] == 179
    assert ledger["recovery"]["recovery_generation_requests"] == 2
    assert ledger["recovery"]["recovery_forward_calls"] == 4
    assert ledger["recovery"]["recovery_generated_tokens"] == 4
    assert ledger["recovery"]["recovery_processed_positions"] == 6
    assert ledger["recovery"]["recovery_layer_readout_streams"] == 2


def test_single_attempt_registry_rejects_another_output(tmp_path):
    output = tmp_path / "one"
    authorization = fixture.authorization(tmp_path, output)
    context, _ = runner.verify_authorization(
        config_path=fixture.CONFIG, prepared=fixture.PREPARED,
        authorization_path=authorization, output=output,
        require_head=False, commit_verifier=fixture.commit_verifier)
    registry = runner.AttemptRegistry(authorization)
    with pytest.raises(ProtocolError, match="single_attempt"):
        registry.register(replace(context, output=tmp_path / "two"))


def test_wall_limit_is_enforced_before_backend_work(tmp_path):
    output = tmp_path / "native01"
    authorization = fixture.authorization(tmp_path, output)
    values = iter((0.0, 2000.0))
    with pytest.raises(ProtocolError, match="wall_time"):
        runner.run_dev(
            config_path=fixture.CONFIG, prepared=fixture.PREPARED,
            authorization_path=authorization, output=output,
            require_head=False, commit_verifier=fixture.commit_verifier,
            clock=lambda: next(values))


def test_preflight_emits_distinct_dev_and_sealed_final_guard_commands(tmp_path):
    output = tmp_path / "native01"
    authorization = fixture.authorization(tmp_path, output)
    report = preflight.preflight(
        fixture.CONFIG, fixture.PREPARED, launch=True,
        authorization=authorization, output=output, require_head=False,
        commit_verifier=fixture.commit_verifier)
    assert report["status"] == "launch_admitted"
    assert "run_dc_r1_native_value_v3_20261028.py" in report["native_command"]
    assert "guard_dev01.json" in report["native_command"]
    assert "guard_sealed_final01.json" in report["sealed_final_command"]
    assert "--continue-existing-output" in report["sealed_final_command"]
    assert capacity.continuation_requested(["--stage", "sealed_final"]) is True
    assert report["native_execution"] is False


def test_final_rejects_tampered_development_artifact(tmp_path, monkeypatch):
    output, authorization, _, _ = _dev(tmp_path)
    path = output / "dev/predictions_all.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["generated_text"] = "tampered"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    backend = fixture.FakeBackend(native=True)
    with pytest.raises(ProtocolError):
        _final(output, authorization, backend, monkeypatch)
    assert backend.calls == []
    assert not (output / "sealed_final/score_report.json").exists()


def test_final_rejects_attempt_mismatch(tmp_path, monkeypatch):
    output, authorization, _, _ = _dev(tmp_path)
    path = output / "dev/journal.json"
    journal = read_json(path)
    journal["attempt_id"] = "different-attempt"
    path.write_text(json.dumps(journal))
    backend = fixture.FakeBackend(native=True)
    with pytest.raises(ProtocolError, match="development attempt differs"):
        _final(output, authorization, backend, monkeypatch)
    assert backend.calls == []


def test_final_rejects_model_identity_mismatch(tmp_path, monkeypatch):
    output, authorization, _, _ = _dev(tmp_path)
    path = output / "work_ledger.json"
    ledger = read_json(path)
    ledger["model_identity"] = "0" * 64
    path.write_text(json.dumps(ledger))
    backend = fixture.FakeBackend(native=True)
    with pytest.raises(ProtocolError, match="model identity differs"):
        _final(output, authorization, backend, monkeypatch)
    assert backend.calls == []


def test_live_artifact_cap_is_enforced_during_ledger_creation(tmp_path):
    output = tmp_path / "native01"
    authorization = fixture.authorization(tmp_path, output)
    context, _ = runner.verify_authorization(
        config_path=fixture.CONFIG, prepared=fixture.PREPARED,
        authorization_path=authorization, output=output,
        require_head=False, commit_verifier=fixture.commit_verifier,
        register=False)
    context.config["budget"]["physical_limits"]["max_artifact_bytes"] = 1
    with pytest.raises(publication.PublicationOverflow):
        runner.WorkLedger(context)
    assert (output / "artifact_overflow.json").is_file()


def test_corrupted_prepared_manifest_is_rejected_before_authorization(tmp_path):
    prepared = tmp_path / "prepared"
    shutil.copytree(fixture.PREPARED, prepared)
    path = prepared / "manifest.json"
    manifest = read_json(path)
    manifest["manifest_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    authorization = fixture.authorization(tmp_path, tmp_path / "native01")
    with pytest.raises(ProtocolError, match="manifest self digest"):
        runner.verify_authorization(
            config_path=fixture.CONFIG, prepared=prepared,
            authorization_path=authorization, output=tmp_path / "native01",
            require_head=False, commit_verifier=fixture.commit_verifier,
            register=False)


def test_missing_publication_record_is_republished_without_replaying(tmp_path):
    output, authorization, _, _ = _dev(tmp_path)
    (output / "publication_journal.json").unlink()
    resumed = fixture.FakeBackend(native=True)
    original = runner.build_native_backend
    runner.build_native_backend = lambda context, stage: resumed
    try:
        result = runner.run_dev(
            config_path=fixture.CONFIG, prepared=fixture.PREPARED,
            authorization_path=authorization, output=output, resume=True,
            require_head=False, commit_verifier=fixture.commit_verifier)
    finally:
        runner.build_native_backend = original
    assert result.get("already_complete", False) is False
    assert resumed.calls == []
    assert (output / "publication_journal.json").is_file()
    journal = read_json(output / "dev/journal.json")
    assert journal["stage_complete"] is True
    assert journal["publication_stage_done"] is True


def test_publication_recovery_replays_only_unfinished_streams(tmp_path):
    output, authorization, _, _ = _dev(tmp_path)
    (output / "publication_journal.json").unlink()
    path = output / "dev/journal.json"
    journal = read_json(path)
    condition = runner.CONDITIONS[0]
    missing = journal["predictions"][condition].pop()
    journal.update({
        "stage_complete": False,
        "publication_stage_done": False,
        "status": "in_progress",
        "result": None,
    })
    path.write_text(json.dumps(journal, indent=2, sort_keys=True) + "\n")
    resumed = fixture.FakeBackend(native=True)
    original = runner.build_native_backend
    runner.build_native_backend = lambda context, stage: resumed
    try:
        result = runner.run_dev(
            config_path=fixture.CONFIG, prepared=fixture.PREPARED,
            authorization_path=authorization, output=output, resume=True,
            require_head=False, commit_verifier=fixture.commit_verifier)
    finally:
        runner.build_native_backend = original
    assert result["status"] == "native_compressed_channel_blocked"
    assert read_json(output / "dev" / "journal.json")["publication_stage_done"] is True
    assert len(resumed.calls) == len(runner.LAYERS)
    assert {call[1] for call in resumed.calls} == {missing["case_id"]}
    assert {call[0] for call in resumed.calls} == {condition}


def test_resume_rejects_publication_from_another_attempt(tmp_path):
    output, authorization, _, _ = _dev(tmp_path)
    path = output / "publication_journal.json"
    payload = read_json(path)
    payload["identity"]["attempt_id"] = "another-attempt"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(publication.PublicationError, match="identity differs"):
        runner.run_dev(
            config_path=fixture.CONFIG, prepared=fixture.PREPARED,
            authorization_path=authorization, output=output, resume=True,
            backend=fixture.FakeBackend(native=True), require_head=False,
            commit_verifier=fixture.commit_verifier)
