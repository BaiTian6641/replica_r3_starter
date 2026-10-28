"""Committed milestone readiness and missing-authorization tests."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import preflight_bounded_milestone_20261028 as milestone  # noqa: E402


def test_readiness_binds_commit_budgets_and_missing_receipt_fields():
    commit = "a" * 40
    checked = []

    def verifier(received_commit, relative):
        assert received_commit == commit
        checked.append(relative)
        return (ROOT / relative).read_bytes()

    report = milestone.build_report(commit, commit_verifier=verifier)
    assert report["status"] == "review_ready_native_blocked_missing_authorization"
    assert report["protocol_commit"] == commit
    assert report["native_execution"] is False
    assert report["native_jobs"] == 0
    assert report["model_loaded"] is False
    assert report["tokenizer_loaded"] is False
    assert report["evaluation_labels_read"] is False
    assert report["capability_qualified"] is False

    r1 = report["dc_r1"]
    assert r1["workload"]["generation_requests"] == 240
    assert r1["workload"]["forward_calls_max"] == 1920
    assert r1["authorized_run_ceiling"]["artifact_bytes_max"] == 33554432
    assert r1["native_command"] is None
    assert r1["authorization_receipt_present"] is False
    assert r1["missing_receipt_fields"]

    alternative = report["dc_p1_source_free_alternative"]
    assert alternative["work_units"]["optimizer_steps"] == 18
    assert alternative["work_units"]["forward_calls"] == 424
    assert alternative["artifact_bytes_max"] == 33554432
    assert alternative["runner_status"] == "not_implemented"
    assert alternative["native_command"] is None
    assert alternative["audit"]["gradient_ratio"] == 1.5
    assert alternative["audit"]["generated_changed_token_count"] == 0
    assert alternative["audit"]["capability_qualified"] is False
    assert alternative["missing_receipt_fields"]

    spark = report["spark_and_data_preparation"]
    assert spark["verification"]["passed"] == 721
    assert spark["final_product"]["passed"] == 500
    assert spark["caption_partition"]["counts"]["admitted"] == 9
    assert spark["caption_partition"]["counts"]["exclusions"] == 21
    assert spark["caption_partition"]["counts"]["exposed"] == 8
    assert spark["native_gates"]["summary"]["unresolved"] == 10
    assert spark["rsi_or_personality_executed"] is False

    assert set(checked) == set(milestone.COMMIT_FILES)
    assert milestone.R1_RUNNER in report["commit_bindings"]
    assert "scripts/preflight_bounded_milestone_20261028.py" in checked
