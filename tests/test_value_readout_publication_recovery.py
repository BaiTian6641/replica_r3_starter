"""Fault-injection contracts for recoverable final artifact publication.

CPU-only: real value-readout training/evaluation with injected publication
failures. No native model, tokenizer, network or learned-capability claim.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from replica_lab import artifact_publication as publication
import replica_lab.value_readout_abstention as runner_module
from test_value_readout_abstention_runner import CONFIG_PATH, PREPARED


class InjectedPublication(RuntimeError):
    pass


@pytest.fixture(scope="module")
def fit_baseline(tmp_path_factory):
    output = tmp_path_factory.mktemp("publication_baseline") / "fit"
    result = runner_module.run_fit(PREPARED, CONFIG_PATH, output)
    assert result["status"] == "passed"
    return output, result


@pytest.fixture(scope="module")
def eval_baseline(fit_baseline, tmp_path_factory):
    fit_root, _ = fit_baseline
    root = tmp_path_factory.mktemp("stage_baseline")
    dev = runner_module.run_development(PREPARED, CONFIG_PATH, fit_root,
                                        root / "dev")
    heldout = runner_module.run_heldout(PREPARED, CONFIG_PATH, fit_root,
                                        root / "dev", root / "heldout")
    controls = runner_module.run_controls(PREPARED, CONFIG_PATH, fit_root,
                                          root / "controls", split="dev")
    return {"fit": fit_root, "dev": root / "dev", "heldout": root / "heldout",
            "controls": root / "controls", "dev_result": dev,
            "heldout_result": heldout, "controls_result": controls}


def _fail_json(monkeypatch, filename):
    original = runner_module._write_json_atomic

    def fail(path, *args, **kwargs):
        if Path(path).name == filename:
            raise InjectedPublication(filename)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(runner_module, "_write_json_atomic", fail)
    return original


def _assert_no_optimizer_replay(output, baseline, *, compare_report=True,
                                compare_receipt=True):
    evidence = json.loads((output / "resume_evidence.json").read_text())
    assert evidence["optimizer_steps_by_process"] == [72, 0]
    for key in ("optimizer_steps", "backward_calls", "training_forward_calls"):
        assert evidence["cumulative_work"][key] == 72
        assert evidence["processes"][1]["work"][key] == 0
    relatives = [
        "fit/checkpoint.pt", "fit/fit_predictions.jsonl",
        "fit/control_shuffled_values.pt", "fit/control_wrong_world_values.pt"]
    if compare_report:
        relatives.append("fit/fit_report.json")
    if compare_receipt:
        relatives.append("fit_receipt.json")
    for relative in relatives:
        assert runner_module.sha256_file(output / relative) == \
            runner_module.sha256_file(baseline / relative), relative
    return evidence


@pytest.mark.parametrize("stage", [
    "fit_report.json", "fit_receipt.json", "resume_evidence.json",
    "artifact_budget",
])
def test_final_publication_failures_recover_without_optimizer_replay(
        fit_baseline, tmp_path, monkeypatch, stage):
    baseline, _ = fit_baseline
    output = tmp_path / stage.replace(".", "_")
    if stage == "artifact_budget":
        original = runner_module.enforce_artifact_budget

        def fail(*args, **kwargs):
            raise InjectedPublication(stage)

        monkeypatch.setattr(runner_module, "enforce_artifact_budget", fail)
    else:
        original = _fail_json(monkeypatch, stage)
    with pytest.raises(InjectedPublication):
        runner_module.run_fit(PREPARED, CONFIG_PATH, output)
    if stage == "artifact_budget":
        monkeypatch.setattr(runner_module, "enforce_artifact_budget", original)
    else:
        monkeypatch.setattr(runner_module, "_write_json_atomic", original)
    # Recovery remains supported until every final stage and measurement pass.
    assert (output / "resume.pt").is_file()
    result = runner_module.run_fit(PREPARED, CONFIG_PATH, output, resume=True)
    assert result["status"] == "passed"
    evidence = _assert_no_optimizer_replay(output, baseline)
    assert evidence["cumulative_work"]["evaluation_forward_calls"] == 135
    assert not (output / "resume.pt").exists()


def test_checkpoint_failure_reuses_control_payloads_and_counts_real_writes(
        fit_baseline, tmp_path, monkeypatch):
    baseline, _ = fit_baseline
    output = tmp_path / "checkpoint_failure"
    original = runner_module.publication.write_torch

    def fail(root, relative, *args, **kwargs):
        if relative == "fit/checkpoint.pt":
            raise InjectedPublication(relative)
        return original(root, relative, *args, **kwargs)

    monkeypatch.setattr(runner_module.publication, "write_torch", fail)
    with pytest.raises(InjectedPublication):
        runner_module.run_fit(PREPARED, CONFIG_PATH, output)
    monkeypatch.setattr(runner_module.publication, "write_torch", original)
    assert (output / "resume.pt").is_file()
    controls = [output / "fit" / f"control_{arm}.pt"
                for arm in ("shuffled_values", "wrong_world_values")]
    mtimes = [path.stat().st_mtime_ns for path in controls]
    result = runner_module.run_fit(PREPARED, CONFIG_PATH, output, resume=True)
    assert result["status"] == "passed"
    assert [path.stat().st_mtime_ns for path in controls] == mtimes
    evidence = _assert_no_optimizer_replay(
        output, baseline, compare_report=False, compare_receipt=False)
    # Replaying valid pre-checkpoint predictions is real recovery work and is
    # reported. Optimizer/backward work and valid checkpoint writes are not
    # repeated.
    assert evidence["cumulative_work"]["evaluation_forward_calls"] == 216
    assert result["physical_work"]["checkpoint_writes"] == 3
    assert result["physical_work"]["evaluation_forward_calls"] == 216


@pytest.mark.parametrize("stage", ["dev", "heldout", "controls"])
def test_partial_stage_report_write_recovers_with_identical_evidence(
        eval_baseline, tmp_path, monkeypatch, stage):
    output = tmp_path / stage
    filename = {
        "dev": "dev_report.json",
        "heldout": "heldout_report.json",
        "controls": "dev_control_report.json",
    }[stage]
    original = _fail_json(monkeypatch, filename)
    call = {
        "dev": lambda: runner_module.run_development(
            PREPARED, CONFIG_PATH, eval_baseline["fit"], output),
        "heldout": lambda: runner_module.run_heldout(
            PREPARED, CONFIG_PATH, eval_baseline["fit"],
            eval_baseline["dev"], output),
        "controls": lambda: runner_module.run_controls(
            PREPARED, CONFIG_PATH, eval_baseline["fit"], output, split="dev"),
    }[stage]
    with pytest.raises(InjectedPublication):
        call()
    monkeypatch.setattr(runner_module, "_write_json_atomic", original)
    predictions = {
        "dev": "dev/dev_predictions.jsonl",
        "heldout": "heldout/heldout_predictions.jsonl",
        "controls": "controls/dev_control_predictions.jsonl",
    }[stage]
    assert (output / predictions).is_file()
    assert not (output / filename.replace("dev_control_", "controls/dev_")
                if stage == "controls" else output / stage / filename).exists()
    result = call()
    assert result["status"] == "passed"
    assert runner_module.sha256_file(output / predictions) == \
        runner_module.sha256_file(eval_baseline[stage] / predictions)
    report_relative = {
        "dev": "dev/dev_report.json",
        "heldout": "heldout/heldout_report.json",
        "controls": "controls/dev_control_report.json",
    }[stage]
    assert runner_module.sha256_file(output / report_relative) == \
        runner_module.sha256_file(eval_baseline[stage] / report_relative)
    if stage != "controls":
        receipt = output / f"{stage}_receipt.json"
        assert runner_module.sha256_file(receipt) == \
            runner_module.sha256_file(eval_baseline[stage] / receipt)


def test_tampered_partial_stage_predictions_are_not_overwritten(
        eval_baseline, tmp_path, monkeypatch):
    output = tmp_path / "dev"
    original = _fail_json(monkeypatch, "dev_report.json")
    with pytest.raises(InjectedPublication):
        runner_module.run_development(PREPARED, CONFIG_PATH,
                                      eval_baseline["fit"], output)
    monkeypatch.setattr(runner_module, "_write_json_atomic", original)
    path = output / "dev" / "dev_predictions.jsonl"
    before = path.read_text()
    rows = [json.loads(line) for line in before.splitlines()]
    rows[0]["predicted_token"] = "tampered"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(runner_module.ReadoutError,
                       match="existing stage predictions differ"):
        runner_module.run_development(PREPARED, CONFIG_PATH,
                                      eval_baseline["fit"], output)
    assert path.read_text() != before


def test_atomic_reuse_live_meter_and_actual_write_callback(tmp_path):
    root = tmp_path / "publication"
    meter = publication.LiveByteMeter(root)
    first = publication.write_bytes(root, "small.json", b"abc", meter=meter)
    assert first.bytes == 3
    mtime = (root / "small.json").stat().st_mtime_ns
    publication.write_bytes(root, "small.json", b"abc", meter=meter)
    assert (root / "small.json").stat().st_mtime_ns == mtime
    assert meter.observations == 2
    assert ".small.json.tmp" in meter.report()["peak_live_files"]

    writes = []
    payload = {"tensor": torch.tensor([1, 2, 3])}
    publication.write_torch(root, "state.pt", payload, meter=meter,
                            on_write=lambda: writes.append(1))
    checkpoint_mtime = (root / "state.pt").stat().st_mtime_ns
    publication.write_torch(root, "state.pt", payload, meter=meter,
                            on_write=lambda: writes.append(1))
    assert writes == [1]
    assert (root / "state.pt").stat().st_mtime_ns == checkpoint_mtime


def test_fit_live_peak_and_overflow_keep_recovery_until_measurement(
        fit_baseline, tmp_path):
    output, _ = fit_baseline
    summary = json.loads((output / "artifact_budget.json").read_text())
    assert summary["cap_bytes"] == 33_554_432
    assert summary["retained_bytes_before_release"] >= summary["measured_bytes"]
    assert summary["live_peak"]["includes_temporary_files"] is True
    assert summary["live_peak"]["includes_recovery_artifacts"] is True
    assert "resume.pt" in summary["live_peak"]["peak_live_files"]
    assert "fit/checkpoint.pt" in summary["live_peak"]["peak_live_files"]
    assert not (output / "resume.pt").exists()

    root = tmp_path / "deferred_overflow"
    root.mkdir()
    recovery = root / "resume.pt"
    recovery.write_bytes(b"recovery" * 512)
    config = json.loads(CONFIG_PATH.read_text())
    config["budget"]["authorized_run_ceiling"]["artifact_bytes_max"] = 1024
    with pytest.raises(runner_module.ArtifactBudgetExceeded):
        runner_module.enforce_artifact_budget(
            root, config, context="overflow_recovery",
            release_after_measurement=[recovery])
    assert recovery.is_file()
    config["budget"]["authorized_run_ceiling"]["artifact_bytes_max"] = 1_000_000
    report = runner_module.enforce_artifact_budget(
        root, config, context="overflow_recovery",
        release_after_measurement=[recovery])
    assert not recovery.exists()
    assert report["retained_bytes_before_release"] >= report["measured_bytes"]
    assert any(row["path"] == "resume.pt"
               for row in report["released_recovery_artifacts"])
