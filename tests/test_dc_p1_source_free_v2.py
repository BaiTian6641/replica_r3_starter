"""Source-free v2 authorization, executable fit, and recovery tests."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from dc_protocol_common_20261027 import read_json, seal_record, sha256_file
from replica_lab.dc_p1_source_free_native import (
    SourceFreeAttemptRegistry, run_source_free_fit, verify_source_free_authorization)
from replica_lab.dc_p1_source_free_runner import run_fit_diagnostic
from replica_lab.dc_p1_tiny_learning import make_tiny_learning_view
from replica_lab.dc_p1_learning_audit import _teaching_and_cases
import preflight_dc_p1_source_free_feature_alignment_v2 as preflight

CONFIG = ROOT / "configs/dc_p1_source_free_feature_alignment_v2.json"
PREPARED = ROOT / "data/dc_p1_native_response_v2_20261027"
COMMIT = "a" * 40


def _authorization(tmp_path: Path, output: Path) -> Path:
    manifest = read_json(PREPARED / "manifest.json")
    body = {
        "format": "replica.dc_protocol.authorization.v2",
        "protocol_id": "dc_p1_source_free_feature_alignment_v2",
        "config_sha256": sha256_file(CONFIG),
        "prepared_manifest_sha256": manifest["manifest_sha256"],
        "protocol_commit": COMMIT,
        "attempt_id": "source-free-v2-attempt-1",
        "output_root": str(output.resolve()),
        "authorized_by": "cpu-orchestration-fixture",
        "native_execution_allowed": True,
        "single_attempt": True,
        "max_optimizer_steps": 18,
        "max_generation_requests": 48,
        "restart_allowance": 1,
    }
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(seal_record(body), indent=2, sort_keys=True) + "\n")
    return path


def _commit_verifier(commit, relative):
    if commit != COMMIT:
        return None
    return (ROOT / relative).read_bytes()


def _view():
    view = make_tiny_learning_view()
    view.assert_backbone_unchanged = lambda: {"backbone_unchanged": True}
    view.close = lambda: None
    return view


def test_source_free_authorization_and_launch_packet(tmp_path):
    output = tmp_path / "native01"
    authorization = _authorization(tmp_path, output)
    context, derived = verify_source_free_authorization(
        config_path=CONFIG, prepared=PREPARED,
        authorization_path=authorization, output=output,
        require_head=False, commit_verifier=_commit_verifier)
    assert context.attempt_id == "source-free-v2-attempt-1"
    assert derived["optimizer_steps"] == 18
    assert derived["generation_requests"] == 48
    report = preflight.preflight(
        CONFIG, PREPARED, authorization=authorization, output=output,
        require_head=False, commit_verifier=_commit_verifier, launch=True)
    assert report["status"] == "launch_admitted"
    assert report["native_execution"] is False
    assert report["native_command"] == report["next_eligible_command"]
    assert "run_dc_p1_source_free_feature_alignment_v2.py" in report["native_command"]
    assert report["attempt_id"] == context.attempt_id


def test_source_free_fit_interruption_resume_and_cost_reconciliation(tmp_path):
    teaching, cases = _teaching_and_cases()
    output = tmp_path / "fit"
    with pytest.raises(RuntimeError, match="injected fit interruption"):
        run_fit_diagnostic(
            _view(), teaching, cases, output, interrupt_after=2,
            prepared=PREPARED)
    interrupted = read_json(output / "fit_state.json")
    assert interrupted["status"] == "interrupted"
    state = run_fit_diagnostic(
        _view(), teaching, cases, output, resume=True, prepared=PREPARED)
    counters = state["counters"]
    assert state["status"] == "passed_fit_learnability_diagnostic"
    assert state["arms"]["treatment"]["events_completed"] == list(range(6))
    assert counters["training_forwards"] == 36
    assert counters["source_free_query_forwards"] == 18
    assert counters["backward_calls"] == 18
    assert counters["optimizer_steps"] == 18
    assert counters["checkpoint_writes"] == 19
    assert counters["generation_requests"] == 48
    assert counters["baseline_generation_requests"] == 24
    assert counters["restarts"] == 1
    assert counters["model_loads"] == 2
    assert counters["tokenizer_loads"] == 2
    assert counters["recovery_generation_requests"] == 1
    assert counters["recovery_forward_calls"] > 0
    assert counters["recovery_processed_positions"] > 0
    assert counters["recovery_training_forwards"] == 0
    assert counters["recovery_backward_calls"] == 0
    assert counters["recovery_optimizer_steps"] == 0
    assert counters["forward_calls"] == 228
    assert state["artifact"]["includes_temporary_files"] is True
    assert state["behavior"]["no_update"]["mean_target_probability_delta"] == 0.0
    assert state["behavior"]["reset_P"]["mean_target_probability_delta"] == 0.0
    assert state["generalization_qualified"] is False


def test_native_wrapper_uses_the_same_fit_path(tmp_path, monkeypatch):
    output = tmp_path / "native01"
    authorization = _authorization(tmp_path, output)
    view = _view()
    monkeypatch.setattr(
        "replica_lab.dc_p1_source_free_native.NativeP1Backend",
        lambda context, stage: view)
    result = run_source_free_fit(
        config_path=CONFIG, prepared=PREPARED,
        authorization_path=authorization, output=output,
        require_head=False, commit_verifier=_commit_verifier)
    assert result["evidence_class"] == "native_mps"
    assert result["native_execution_admissible"] is True
    assert result["backbone"] == {"backbone_unchanged": True}
    assert result["fit_is_learnability_only"] is True
    assert result["development_or_final_access"] is False


def test_source_free_single_attempt_registry_rejects_output_change(tmp_path):
    output = tmp_path / "one"
    authorization = _authorization(tmp_path, output)
    context, _ = verify_source_free_authorization(
        config_path=CONFIG, prepared=PREPARED,
        authorization_path=authorization, output=output,
        require_head=False, commit_verifier=_commit_verifier)
    with pytest.raises(Exception, match="single_attempt"):
        SourceFreeAttemptRegistry(authorization).register(
            replace(context, output=tmp_path / "two")
        )
