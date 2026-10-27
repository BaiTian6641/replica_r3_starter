"""Focused CPU tests for the H1 evidence replay validator.

The integration tests replay the real, already-produced H1 artifacts and do
not run a model or a native job.  They skip with the exact artifact list when
the ignored local evidence is absent.  The unit tests use temporary synthetic
fixtures and always run.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "validate_memory_bank_readout",
    ROOT / "scripts/validate_memory_bank_readout.py")
validator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validator)

FIRST = ROOT / "runs/memory_bank_readout_20261026/first02"
RESTART = ROOT / "runs/memory_bank_readout_20261026/restart02"
SUMMARY = ROOT / "reports/memory_bank_readout_20261026/summary.json"
CONFIG = ROOT / "configs/memory_bank_readout_v2.json"
PREPARED = ROOT / "data/memory_causal_fresh_20261026/prepared04"
ROLES = ROOT / "data/memory_causal_fresh_20261026/prepared04_roles"

H1_INPUTS = (
    FIRST / "report.json",
    RESTART / "report.json",
    SUMMARY,
    CONFIG,
    PREPARED / "manifest.json",
    ROLES / "manifest.json",
)
requires_h1 = pytest.mark.skipif(
    not all(path.exists() for path in H1_INPUTS),
    reason=("missing local H1 evidence — copy the immutable first02/restart02 "
            "reports, summary.json, prepared04 and prepared04_roles from the "
            "provisioned native-evidence workspace"))


def corrected_summary(tmp_path: Path, name: str = "summary.json",
                      mutate=None) -> Path:
    """A summary copy with the independently replayed dev count (152)."""
    document = json.loads(SUMMARY.read_text())
    document["verification"]["independent_dev_generations_validated"] = 152
    if mutate is not None:
        mutate(document)
    path = tmp_path / name
    path.write_text(json.dumps(document, indent=2) + "\n")
    return path


def test_manifest_entry_rejects_tampered_hash_and_rows(tmp_path):
    directory = tmp_path / "raw"
    directory.mkdir()
    path = directory / "decisions_dev.jsonl"
    path.write_text(
        json.dumps({"decision_id": "decision-a"}) + "\n"
        + json.dumps({"decision_id": "decision-b"}) + "\n")
    entry = {
        "filename": path.name,
        "rows": 2,
        "sha256": validator.sha256_file(path),
        "first_write": True,
        "immutable": True,
    }
    assert validator.validate_manifest_entry(entry, directory, "raw.dev") == {
        "filename": path.name, "rows": 2, "sha256": entry["sha256"]}

    bad_hash = dict(entry, sha256="0" * 64)
    with pytest.raises(validator.ReplayError) as error:
        validator.validate_manifest_entry(bad_hash, directory, "raw.dev")
    assert error.value.check == "raw.dev.sha256"

    path.write_text(path.read_text() + json.dumps({"decision_id": "c"}) + "\n")
    with pytest.raises(validator.ReplayError) as error:
        validator.validate_manifest_entry(entry, directory, "raw.dev")
    assert error.value.check == "raw.dev.rows"


def test_decision_identity_formula_and_uniqueness_are_enforced():
    row = {
        "format": validator.DECISIONS_FORMAT,
        "episode_id": "epi-e-test",
        "state": "initial",
        "phase": "first_process",
        "query_id": "p0a",
    }
    row["decision_id"] = validator.decision_identity(
        row["episode_id"], row["state"], row["phase"], row["query_id"])
    assert validator.validate_decision_identities([row]) == {
        "rows": 1, "unique_ids": 1, "unique_keys": 1}

    bad = dict(row, decision_id="decision-0000000000000000")
    with pytest.raises(validator.ReplayError) as error:
        validator.validate_decision_identities([bad])
    assert error.value.check == "decisions.decision_id"

    with pytest.raises(validator.ReplayError) as error:
        validator.validate_decision_identities([row, dict(row)])
    assert "repeated decision key" in error.value.detail


@requires_h1
@pytest.mark.integration
def test_classification_memory_disabled_and_zero_update_tampering():
    report = json.loads((FIRST / "report.json").read_text())
    config = json.loads(CONFIG.read_text())
    validator.validate_classification(report)
    validator.validate_memory_disabled(report, config)
    validator.validate_zero_updates(report, config)

    memory = deepcopy(report)
    memory["protocol"]["generation_contract"]["record_memory"] = True
    with pytest.raises(validator.ReplayError) as error:
        validator.validate_memory_disabled(memory, config)
    assert error.value.check == "report.protocol.generation_contract"

    updates = deepcopy(report)
    updates["resource_usage"]["optimizer_updates"] = 1
    with pytest.raises(validator.ReplayError) as error:
        validator.validate_zero_updates(updates, config)
    assert error.value.check == "report.resource_usage.optimizer_updates"

    classification = deepcopy(report)
    classification["classification"]["persistent_learning"] = True
    with pytest.raises(validator.ReplayError) as error:
        validator.validate_classification(classification)
    assert error.value.check == "report.classification"


@requires_h1
@pytest.mark.integration
def test_replayed_gate_mismatch_is_rejected():
    report = json.loads((FIRST / "report.json").read_text())
    gate = report["gating"]["first_process"]["gate"]
    validator.compare_gate(gate, deepcopy(gate), "gating.first_process")

    tampered = deepcopy(gate)
    tampered["checks"]["exact_answer"]["count"] = 71
    with pytest.raises(validator.ReplayError) as error:
        validator.compare_gate(tampered, gate, "gating.first_process")
    assert error.value.check == "gating.first_process"


@requires_h1
@pytest.mark.integration
def test_full_replay_passes_corrected_summary_without_heldout_labels(
        tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import replica_lab.memory_bank_readout_v2 as readout

    opened = []
    original = readout._read_role_json

    def spy(path):
        opened.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(readout, "_read_role_json", spy)
    summary = corrected_summary(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    assert validator.main([
        "--summary", str(summary),
        "--receipt", str(receipt_path),
    ]) == 0

    receipt = json.loads(receipt_path.read_text())
    assert receipt["passed"] is True
    assert receipt["native_execution"] is False
    assert receipt["heldout_labels_read"] is False
    assert receipt["heldout_rescored"] is False
    assert receipt["evidence"]["dev_generation_rows"] == 152
    assert receipt["evidence"]["heldout_generation_rows"] == 152
    assert receipt["evidence"]["decision_rows"] == 456
    assert receipt["evidence"]["literal_gate"]["passed"] is True
    assert receipt["evidence"]["state_gate"]["passed"] is True
    assert receipt["evidence"]["accounting_gate"]["passed"] is True
    assert receipt["evidence"]["resource_totals"] == {
        "gradients": 0, "optimizer_updates": 0, "checkpoints_written": 0,
        "base_exports": 0, "extension_exports": 0, "downloads": 0,
    }
    assert opened == ["labels_dev.json"]
    assert all("heldout" not in name for name in opened)

    # The receipt carries no wall-clock timestamp and is byte-deterministic.
    second = tmp_path / "receipt-again.json"
    assert validator.main([
        "--summary", str(summary),
        "--receipt", str(second),
    ]) == 0
    assert second.read_bytes() == receipt_path.read_bytes()


@requires_h1
@pytest.mark.integration
def test_full_replay_rejects_summary_tampering(tmp_path):
    pytest.importorskip("torch")

    def dev_count_232(document):
        document["verification"]["independent_dev_generations_validated"] = 232

    def wrong_metric(document):
        document["metrics"]["first_exact_answers"] = [71, 72]

    def wrong_gate(document):
        document["outcome"]["literal_dev"] = "failed"

    def wrong_hash_claim(document):
        document["verification"]["raw_hashes_match"] = False

    cases = [
        ("dev-count", dev_count_232,
         "summary.verification.independent_dev_generations_validated"),
        ("metric", wrong_metric, "summary.metrics.first_exact_answers"),
        ("gate", wrong_gate, "summary.outcome"),
        ("hash-claim", wrong_hash_claim,
         "summary.verification.raw_hashes_match"),
    ]
    for name, mutate, expected_check in cases:
        summary = corrected_summary(tmp_path, f"summary-{name}.json", mutate)
        receipt_path = tmp_path / f"receipt-{name}.json"
        assert validator.main([
            "--summary", str(summary),
            "--receipt", str(receipt_path),
        ]) == 1
        receipt = json.loads(receipt_path.read_text())
        assert receipt["passed"] is False
        assert receipt["check"] == expected_check
        if name == "dev-count":
            assert receipt["evidence"]["dev_generation_rows"] == 152
            assert "232" in receipt["detail"]


@requires_h1
@pytest.mark.integration
def test_receipt_cannot_be_written_inside_raw_output_directory(tmp_path):
    pytest.importorskip("torch")
    summary = corrected_summary(tmp_path)
    assert validator.main([
        "--summary", str(summary),
        "--receipt", str(FIRST / "receipt.json"),
    ]) == 2
    assert not (FIRST / "receipt.json").exists()
