"""End-to-end standalone DC-P1 scoring CLI admission and label-access tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from dc_protocol_common_20261027 import read_json, read_jsonl  # noqa: E402
from replica_lab import dc_p1_evidence as evidence  # noqa: E402
from replica_lab import dc_staged_evaluation as staged  # noqa: E402
import score_dc_p1_generated_v2_20261027 as cli  # noqa: E402
import test_dc_p1_native_runner_v2 as fixture  # noqa: E402


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                            for row in rows), encoding="utf-8")


def _production_evidence(tmp_path: Path, monkeypatch) -> Path:
    output = tmp_path / "native01"
    authorization = fixture._authorization(tmp_path)

    def factory(context, stage):
        return fixture._production_backend(wrong_primary_for_arms={
            "matched_trained", "no_update", "reset_P"})

    monkeypatch.setattr(fixture.runner, "build_native_backend", factory)
    for stage in ("fit", "development", "final"):
        result = fixture.runner.run_stage(
            stage, config_path=fixture.CONFIG, prepared=fixture.PREPARED,
            authorization_path=authorization, protocol_commit=fixture.COMMIT,
            output=output, backend=None, require_head=False,
            commit_verifier=fixture._commit_verifier)
        assert result["status"] == "passed"
        assert result["admitted"] is True
    return output


def _cpu_evidence(tmp_path: Path) -> Path:
    output = tmp_path / "native01"
    authorization = fixture._authorization(tmp_path)
    for stage in ("fit", "development", "final"):
        result = fixture._run_stage(
            stage, output, fixture._controlled_backend(), authorization)
        assert result["status"] == "passed"
        assert result["admitted"] is False
    return output


def _combined(output: Path, stage: str) -> Path:
    rows = []
    for arm in fixture.runner.ARMS if stage != "final" else ("treatment",):
        rows.extend(read_jsonl(output / stage / f"predictions_{arm}.jsonl"))
    path = output / f"combined_{stage}.jsonl"
    _write_jsonl(path, rows)
    return path


def _args(output: Path, stage: str, receipt: Path) -> list[str]:
    report_name = {
        "fit": "fit_report.json",
        "development": "development_report.json",
        "final": "final_report.json",
    }[stage]
    report = read_json(output / stage / report_name)
    audit = output / f"scoring_{stage}_objective_audit.json"
    _write_json(audit, report["objective_audit"])
    args = [
        "--stage", stage,
        "--predictions", str(_combined(output, stage)),
        "--config", str(fixture.CONFIG),
        "--prepared", str(fixture.PREPARED),
        "--protocol-commit", fixture.COMMIT,
        "--attempt-id", read_json(output / stage / "receipt.json")["attempt_id"],
        "--objective-audit", str(audit),
        "--evidence-root", str(output),
        "--receipt", str(receipt),
    ]
    if stage in ("development", "final"):
        args += ["--fit-receipt", str(output / "fit" / "receipt.json")]
    if stage == "final":
        args += ["--development-receipt",
                 str(output / "development" / "receipt.json")]
    return args


def _label_spy(monkeypatch) -> list[str]:
    calls = []
    original = evidence.read_jsonl

    def spy(path):
        calls.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(evidence, "read_jsonl", spy)
    return calls


def _parsed_label_events(payload: dict) -> list[str]:
    return [event["detail"] for event in payload["access"]["events"]
            if event["access"] == "parse" and event["granted"]]


def test_scoring_cli_valid_prerequisites_and_real_label_spies(
        tmp_path, monkeypatch, capsys):
    output = _production_evidence(tmp_path, monkeypatch)
    calls = _label_spy(monkeypatch)
    development_receipt = tmp_path / "scored_development" / "receipt.json"
    assert cli.main(_args(output, "development", development_receipt)) == 0
    development_payload = json.loads(capsys.readouterr().out)
    assert "refused" not in development_payload["admission"]
    assert "labels:sealed_labels_dev.jsonl" in _parsed_label_events(
        development_payload)
    assert "sealed_labels_dev.jsonl" in calls
    assert development_receipt.is_file()
    assert development_receipt.with_suffix(".report.json").is_file()

    final_receipt = tmp_path / "scored_final" / "receipt.json"
    assert cli.main(_args(output, "final", final_receipt)) == 0
    final_payload = json.loads(capsys.readouterr().out)
    assert "refused" not in final_payload["admission"]
    events = _parsed_label_events(final_payload)
    assert "labels:sealed_final_labels_fit.jsonl" in events
    assert "labels:sealed_labels_final.jsonl" in events
    assert "sealed_final_labels_fit.jsonl" in calls
    assert "sealed_labels_final.jsonl" in calls
    assert final_receipt.is_file()
    assert final_receipt.with_suffix(".report.json").is_file()


def test_scoring_cli_missing_prerequisites_never_reads_labels(
        tmp_path, monkeypatch, capsys):
    output = _production_evidence(tmp_path, monkeypatch)
    calls = _label_spy(monkeypatch)
    development_args = [item for item in _args(
        output, "development", tmp_path / "missing_dev.json")
        if item not in {"--fit-receipt", str(output / "fit" / "receipt.json")}]
    assert cli.main(development_args) == 2
    assert calls == []
    capsys.readouterr()

    final_args = [item for item in _args(
        output, "final", tmp_path / "missing_final.json")
        if item not in {"--development-receipt",
                        str(output / "development" / "receipt.json")}]
    assert cli.main(final_args) == 2
    assert calls == []
    capsys.readouterr()


def test_scoring_cli_resealed_contradiction_blocks_development_labels(
        tmp_path, monkeypatch, capsys):
    output = _production_evidence(tmp_path, monkeypatch)
    calls = _label_spy(monkeypatch)
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
    assert cli.main(_args(
        output, "development", tmp_path / "blocked_dev.json")) == 2
    assert "sealed_labels_dev.jsonl" not in calls
    capsys.readouterr()


def test_scoring_cli_false_decode_blocks_development_labels(
        tmp_path, monkeypatch, capsys):
    output = _production_evidence(tmp_path, monkeypatch)
    calls = _label_spy(monkeypatch)
    path = output / "fit" / "predictions_treatment.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["generated_text"] = "not-the-token-text"
    _write_jsonl(path, rows)
    assert cli.main(_args(
        output, "development", tmp_path / "blocked_decode.json")) == 2
    assert "sealed_labels_dev.jsonl" not in calls
    capsys.readouterr()


def test_scoring_cli_cpu_prerequisites_stay_closed(
        tmp_path, monkeypatch, capsys):
    output = _cpu_evidence(tmp_path)
    calls = _label_spy(monkeypatch)
    assert cli.main(_args(
        output, "development", tmp_path / "blocked_cpu.json")) == 2
    assert "sealed_labels_dev.jsonl" not in calls
    capsys.readouterr()


def test_scoring_cli_final_reused_receipt_is_refused_before_labels(
        tmp_path, monkeypatch, capsys):
    output = _production_evidence(tmp_path, monkeypatch)
    receipt = tmp_path / "one_shot" / "receipt.json"
    args = _args(output, "final", receipt)
    assert cli.main(args) == 0
    capsys.readouterr()
    calls = _label_spy(monkeypatch)
    assert cli.main(args) == 2
    assert calls == []
    capsys.readouterr()
