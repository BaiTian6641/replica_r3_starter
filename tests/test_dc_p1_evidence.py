"""Independent DC-P1 artifact and score-recompute admission tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from dc_protocol_common_20261027 import ProtocolError, read_json  # noqa: E402
from replica_lab import dc_p1_evidence as evidence  # noqa: E402
from replica_lab import dc_staged_evaluation as staged  # noqa: E402
import test_dc_p1_native_runner_v2 as fixture  # noqa: E402


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                            for row in rows), encoding="utf-8")


def _fit(tmp_path: Path):
    output = tmp_path / "native01"
    authorization = fixture._authorization(tmp_path)
    result = fixture._run_stage(
        "fit", output, fixture._controlled_backend(), authorization)
    assert result["status"] == "passed"
    return output


def _kwargs(output: Path, *, require_publication: bool = True) -> dict:
    config = read_json(fixture.CONFIG)
    return {
        "output_root": output,
        "prepared": fixture.PREPARED,
        "config": config,
        "config_path": fixture.CONFIG,
        "manifest": read_json(fixture.PREPARED / "manifest.json"),
        "protocol_commit": fixture.COMMIT,
        "binding": config["tokenizer_binding"],
        "decode_fn": fixture._decode,
        "expected_evidence_class": "cpu_fixture",
        "require_publication": require_publication,
    }


def test_fit_evidence_validates_artifacts_and_recompute(tmp_path):
    output = _fit(tmp_path)
    result = evidence.verify_stage_evidence(stage="fit", **_kwargs(output))
    assert result["recomputed"] == result["report"]
    assert result["published_artifacts_verified"] > 0
    assert set(result["predictions"]) == set(fixture.runner.ARMS)


def test_evidence_rejects_missing_prediction_artifact(tmp_path):
    output = _fit(tmp_path)
    (output / "fit" / "predictions_treatment.jsonl").unlink()
    with pytest.raises(ProtocolError, match="missing evidence artifact"):
        evidence.verify_stage_evidence(stage="fit", **_kwargs(output))


def test_evidence_rejects_resealed_contradictory_metrics(tmp_path):
    output = _fit(tmp_path)
    report_path = output / "fit" / "fit_report.json"
    report = read_json(report_path)
    report["arms"]["treatment"]["exact_accuracy"] = 0.0
    report["gates"]["treatment_fit_exact"]["value"] = 0.0
    report["gates"]["treatment_fit_exact"]["passed"] = True
    _write_json(report_path, report)
    config = read_json(fixture.CONFIG)
    expectation = evidence.receipt_expectation(
        config=config, config_path=fixture.CONFIG,
        manifest=read_json(fixture.PREPARED / "manifest.json"),
        binding=config["tokenizer_binding"], protocol_commit=fixture.COMMIT)
    receipt = read_json(output / "fit" / "receipt.json")
    _write_json(output / "fit" / "receipt.json", staged.build_stage_receipt(
        report=report, expectation=expectation,
        attempt_id=receipt["attempt_id"]))
    with pytest.raises(ProtocolError, match="score_recompute"):
        evidence.verify_stage_evidence(stage="fit", **_kwargs(output))


def test_evidence_rejects_false_decoded_output(tmp_path):
    output = _fit(tmp_path)
    path = output / "fit" / "predictions_treatment.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["generated_text"] = "not-the-token-text"
    _write_jsonl(path, rows)
    with pytest.raises(ProtocolError, match="decoded ids do not reproduce"):
        evidence.verify_stage_evidence(
            stage="fit", **_kwargs(output, require_publication=False))


def test_evidence_rejects_zero_report_digest(tmp_path):
    output = _fit(tmp_path)
    path = output / "fit" / "receipt.json"
    receipt = read_json(path)
    receipt["score_report_sha256"] = "0" * 64
    _write_json(path, staged.seal_record(receipt))
    with pytest.raises(ProtocolError, match="actual score report"):
        evidence.verify_stage_evidence(stage="fit", **_kwargs(output))


def test_publication_evidence_rejects_zero_artifact_digest(tmp_path):
    output = _fit(tmp_path)
    path = output / "publication_journal.json"
    journal = read_json(path)
    for entry in journal["stages"]:
        if entry["stage"] == "fit":
            for artifact in entry["artifacts"]:
                if artifact["path"] == "fit/fit_report.json":
                    artifact["sha256"] = "0" * 64
    _write_json(path, journal)
    with pytest.raises(ProtocolError, match="publication artifacts changed"):
        evidence.verify_stage_evidence(stage="fit", **_kwargs(output))
