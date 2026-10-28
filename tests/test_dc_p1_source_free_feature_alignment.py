"""Source-free feature-alignment protocol and readiness tests."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dc_protocol_common_20261027 import ProtocolError, read_json  # noqa: E402
import preflight_dc_p1_source_free_feature_alignment_v1 as preflight  # noqa: E402
from replica_lab.dc_p1_learning_audit import run_cpu_audit  # noqa: E402


def test_source_free_alternative_protocol_and_readiness(tmp_path):
    config = read_json(preflight.CONFIG_PATH)
    derived = preflight.validate_config(config)
    assert derived["optimizer_steps"] == 18
    assert derived["source_free_query_forwards"] == 18
    assert derived["generation_requests"] == 24
    assert derived["checkpoint_writes"] == 19
    assert derived["forward_calls"] == 424
    assert derived["processed_text_positions"] == 18860
    assert derived["artifact_bytes"] == 33554432
    prepared = preflight.validate_prepared(preflight.PREPARED_DIR, config)
    assert prepared == {"teaching_rows": 6, "fit_cases": 6}
    report = preflight.preflight(
        preflight.CONFIG_PATH, preflight.PREPARED_DIR,
        tmp_path / "review.json")
    assert report["status"] == "review_ready_preparation_only"
    assert report["native_execution"] is False
    assert report["model_loaded"] is False
    assert report["tokenizer_loaded"] is False
    assert report["labels_read"] is False
    assert report["launch_allowed"] is False
    assert report["runner_status"] == "not_implemented"
    assert report["native_command"] is None
    assert report["authorization_receipt_present"] is False
    assert report["prepared_validation"] == prepared
    assert (tmp_path / "review.json").is_file()
    assert set(report["missing_receipt_fields"]) == set(
        config["launch"]["missing_receipt_fields"])
    assert config["objective"]["matched_trained_weight"] == 0.0
    assert config["objective"]["span_specific_supervision"] is False
    assert config["objective"]["memory_repair_claimed"] is False
    assert config["prepared"]["development_labels"] == "closed"
    assert config["prepared"]["sealed_final_labels"] == "closed"


def test_source_free_alternative_budget_drift_is_refused():
    config = deepcopy(read_json(preflight.CONFIG_PATH))
    config["budget"]["physical_limits"]["max_training_forwards"] += 1
    with pytest.raises(ProtocolError, match="budget"):
        preflight.validate_config(config)


def test_learning_audit_justifies_source_free_alternative():
    audit = run_cpu_audit()
    config = read_json(preflight.CONFIG_PATH)
    assert audit["alternative_justified"] is True
    assert audit["alternative_protocol"] == config["protocol_id"]
