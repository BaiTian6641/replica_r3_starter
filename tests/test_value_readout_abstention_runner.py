"""CPU integration contracts for the value-readout runner.

These tests exercise real training, optimizer accounting, checkpointing,
interruption/resume and the seven control views.  They never load a native
model, tokenizer or download.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "value_readout_abstention_protocol_v1.json"
PREPARED = ROOT / "data" / "value_readout_abstention_20261027" / "fixture02"

from replica_lab.value_readout_abstention import (  # noqa: E402
    ArtifactBudgetExceeded, RunInterrupted, enforce_artifact_budget,
    run_controls, run_development, run_fit, run_heldout, sha256_file,
)
import replica_lab.value_readout_abstention as runner_module  # noqa: E402


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _report(root: Path) -> dict:
    return json.loads((root / "fit" / "fit_report.json").read_text(encoding="utf-8"))


def _payload(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


@pytest.fixture(scope="module")
def full_runs(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("runner_integration")
    fit_root = root / "fit_run"
    fit = run_fit(PREPARED, CONFIG_PATH, fit_root)
    assert fit["status"] == "passed"
    dev_root = root / "dev_run"
    dev = run_development(PREPARED, CONFIG_PATH, fit_root, dev_root)
    assert dev["status"] == "passed"
    heldout_root = root / "heldout_run"
    heldout = run_heldout(PREPARED, CONFIG_PATH, fit_root, dev_root,
                          heldout_root)
    assert heldout["status"] == "passed"
    controls_root = root / "controls_run"
    controls = run_controls(PREPARED, CONFIG_PATH, fit_root, controls_root,
                            split="dev")
    assert controls["status"] == "passed"
    return {"fit": fit, "fit_root": fit_root, "dev": dev, "dev_root": dev_root,
            "heldout": heldout, "heldout_root": heldout_root,
            "controls": controls, "controls_root": controls_root}


def test_runner_accounts_three_independent_24_update_arms(full_runs):
    report = _report(full_runs["fit_root"])
    assert report["completed_steps"] == 24
    assert report["event_clock_steps"] == 36
    assert report["events_processed"] == 36
    assert len(report["update_records"]) == 24
    assert [row["step"] for row in report["update_records"]] == list(range(24))
    assert [row["clock_step"] for row in report["update_records"]] == sorted(
        row["clock_step"] for row in report["update_records"])
    assert report["optimizer_meta"]["step_counts"] == {
        "selected_expert.gate": 24,
        "selected_expert.up": 24,
        "selected_expert.down": 24,
    }
    assert report["optimizer_meta"]["param_names"] == [
        "selected_expert.gate", "selected_expert.up", "selected_expert.down"]
    treatment = _payload(full_runs["fit_root"] / "fit" / "checkpoint.pt")
    assert treatment["arm_id"] == "learned_value_readout"
    assert treatment["phase"] == "fit_treatment"
    assert treatment["completed_steps"] == 24
    assert treatment["state_sha256"] == report["resulting_state_sha256"]
    assert treatment["parent_state_sha256"] == report["parent_state_sha256"]
    assert treatment["state_sha256"] != treatment["parent_state_sha256"]
    assert treatment["fixed_state_sha256"] != {}
    assert treatment["classification"]["native_model"] is False
    control_hashes = set()
    for arm_id in ("shuffled_values", "wrong_world_values"):
        payload = _payload(
            full_runs["fit_root"] / "fit" / f"control_{arm_id}.pt")
        assert payload["arm_id"] == arm_id
        assert payload["phase"] == "control_fit"
        assert payload["completed_steps"] == 24
        assert payload["state_sha256"] != treatment["state_sha256"]
        control_hashes.add(payload["state_sha256"])
    assert len(control_hashes) == 2


def test_runner_predictions_are_per_case_and_immutable(full_runs):
    report = _report(full_runs["fit_root"])
    rows = _rows(full_runs["fit_root"] / "fit" / "fit_predictions.jsonl")
    assert len(rows) == 27
    assert len({row["case_id"] for row in rows}) == 27
    assert all(row["arm_id"] == "learned_value_readout" for row in rows)
    assert all(row["model_state_sha256"] == report["resulting_state_sha256"]
               for row in rows)
    assert all(len(row["memory_state_sha256"]) == 64 for row in rows)
    forbidden = {
        "target", "expected_value", "expected_status", "label_source_epoch",
        "authority", "source_record_id", "stale_record_id", "stale_value",
        "rejected_world_id", "rejected_record_id",
        "rejected_record_source_epoch",
    }
    assert all(not (forbidden & set(row)) for row in rows)
    assert all(row["split"] == "fit" for row in rows)
    assert {row["case_type"] for row in rows} == {
        "future_answer", "correction", "unaffected_retention",
        "unknown_abstention", "wrong_world_abstention"}
    dev_rows = _rows(full_runs["dev_root"] / "dev" / "dev_predictions.jsonl")
    heldout_rows = _rows(
        full_runs["heldout_root"] / "heldout" / "heldout_predictions.jsonl")
    assert len(dev_rows) == 10 and len(heldout_rows) == 10
    assert all(row["split"] == "dev" for row in dev_rows)
    assert all(row["split"] == "heldout" for row in heldout_rows)
    assert full_runs["dev"]["report"]["optimizer_steps"] == 0
    assert full_runs["heldout"]["report"]["optimizer_steps"] == 0


def test_seven_controls_are_distinct_and_gate_as_declared(full_runs):
    report = json.loads(
        (full_runs["controls_root"] / "controls" / "dev_control_report.json")
        .read_text(encoding="utf-8"))
    arms = report["arms"]
    assert list(arms) == [
        "learned_value_readout", "exact_context",
        "selected_record_literal_readout", "memory_disabled", "memory_reset",
        "shuffled_values", "wrong_world_values"]
    assert report["distinct_arms"] is True
    assert report["treatment_required_to_beat_reference"] is False
    for arm_id in ("memory_disabled", "memory_reset"):
        assert arms[arm_id]["non_abstain_rate"] == 0.0
        assert arms[arm_id]["gate_passed"] is True
    for arm_id in ("shuffled_values", "wrong_world_values"):
        assert arms[arm_id]["max_any_case_accuracy"] <= 0.5
        assert arms[arm_id]["gate_passed"] is True
    assert arms["exact_context"]["quality_metric"] == "case_accuracy"
    assert arms["exact_context"]["cost_metric"] == "context_tokens_per_query"
    assert arms["exact_context"]["context_tokens_per_query"] > 0
    assert arms["selected_record_literal_readout"]["diagnostic_only"] is True
    assert "gate_passed" not in arms["learned_value_readout"]
    assert "gate_passed" not in arms["exact_context"]


def test_interruption_and_resume_are_bit_identical(full_runs, tmp_path):
    baseline = _report(full_runs["fit_root"])
    target = tmp_path / "interrupted"
    with pytest.raises(RunInterrupted) as interrupted:
        run_fit(PREPARED, CONFIG_PATH, target, interrupt_after=5)
    assert interrupted.value.completed_updates == 5
    assert (target / "resume.pt").is_file()
    resumed = run_fit(PREPARED, CONFIG_PATH, target, resume=True)
    report = resumed["fit_report"]
    assert report["completed_steps"] == 24
    assert report["resulting_state_sha256"] == baseline["resulting_state_sha256"]
    assert report["update_records"] == baseline["update_records"]
    for relative in (
            "fit/checkpoint.pt", "fit/fit_predictions.jsonl",
            "fit/fit_report.json", "fit_receipt.json",
            "fit/control_shuffled_values.pt",
            "fit/control_wrong_world_values.pt",
            "fit/control_shuffled_values_predictions.jsonl",
            "fit/control_wrong_world_values_predictions.jsonl",
            "artifact_budget.json"):
        assert sha256_file(target / relative) == sha256_file(
            full_runs["fit_root"] / relative), relative
    for arm_id in ("shuffled_values", "wrong_world_values"):
        resumed_payload = _payload(
            target / "fit" / f"control_{arm_id}.pt")
        baseline_payload = _payload(
            full_runs["fit_root"] / "fit" / f"control_{arm_id}.pt")
        assert resumed_payload["state_sha256"] == \
            baseline_payload["state_sha256"]
        assert resumed_payload["update_records"] == \
            baseline_payload["update_records"]


def test_fresh_process_resume_matches_uninterrupted(full_runs, tmp_path):
    output = tmp_path / "fresh_process"
    command = [sys.executable,
               str(ROOT / "scripts" / "run_value_readout_abstention_v1.py"),
               "fit", "--output", str(output), "--interrupt-after", "4"]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                           timeout=180,
                           env={"PYTHONDONTWRITEBYTECODE": "1",
                                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"})
    assert first.returncode == 3, first.stderr
    assert json.loads(first.stdout)["status"] == "interrupted"
    second = subprocess.run(
        command[:-2] + ["--resume"], cwd=ROOT, capture_output=True,
        text=True, timeout=180,
        env={"PYTHONDONTWRITEBYTECODE": "1",
             "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"})
    assert second.returncode == 0, second.stderr
    report = _report(output)
    baseline = _report(full_runs["fit_root"])
    assert report["resulting_state_sha256"] == baseline["resulting_state_sha256"]
    assert report["update_records"] == baseline["update_records"]
    for relative in ("fit/checkpoint.pt", "fit/fit_predictions.jsonl",
                     "fit/fit_report.json", "fit_receipt.json",
                     "artifact_budget.json"):
        assert sha256_file(output / relative) == sha256_file(
            full_runs["fit_root"] / relative), relative
    assert json.loads((output / "fit_receipt.json").read_text(
        encoding="utf-8")) == json.loads(
            (full_runs["fit_root"] / "fit_receipt.json").read_text(
                encoding="utf-8"))


def test_runner_reads_no_evaluation_labels_during_fit(monkeypatch, tmp_path):
    calls: list[str] = []
    original = runner_module.read_jsonl

    def spy(path):
        calls.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(runner_module, "read_jsonl", spy)
    run_fit(PREPARED, CONFIG_PATH, tmp_path / "fit")
    assert "labels_dev.jsonl" not in calls
    assert "labels_heldout.jsonl" not in calls
    assert "cases_train.jsonl" in calls
    assert "cases_train_replay.jsonl" in calls


def test_fit_receipt_binds_only_treatment_artifacts(full_runs):
    receipt = json.loads(
        (full_runs["fit_root"] / "fit_receipt.json").read_text(encoding="utf-8"))
    control_hashes = {
        sha256_file(full_runs["fit_root"] / "fit" / f"control_{arm_id}.pt")
        for arm_id in ("shuffled_values", "wrong_world_values")}
    assert receipt["artifacts"]["checkpoint_sha256"] not in control_hashes
    assert receipt["created_by"] == "scripts/run_value_readout_abstention_v1.py"
    assert receipt["optimizer_steps"] == 24
    assert receipt["heldout_opened"] is False
    assert receipt["dev_labels_read"] is False
    assert receipt["fit_gates"]["local_associative_loss_gate"] is False
    assert receipt["status"] == "passed"


def test_fit_artifact_budget_documents_and_enforces_amended_cap(full_runs):
    summary = json.loads(
        (full_runs["fit_root"] / "artifact_budget.json").read_text(
            encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    declared = config["budget"]["authorized_run_ceiling"]
    assert summary["status"] == "within_budget"
    assert summary["cap_bytes"] == 33554432
    assert summary["cap_bytes"] == declared["artifact_bytes_max"]
    assert summary["old_cap_bytes"] == 16777216
    assert summary["old_cap_underprovisioned"] is True
    assert summary["resume_complete_checkpoints"] == 3
    assert summary["measured_bytes"] > summary["old_cap_bytes"]
    assert summary["measured_bytes"] <= summary["cap_bytes"]
    assert summary["measured_bytes"] == sum(
        path.stat().st_size for path in full_runs["fit_root"].rglob("*")
        if path.is_file())
    assert declared["artifact_budget"]["observed_total_bytes"] \
        == 29107600
    assert declared["artifact_budget"]["headroom_bytes"] == \
        33554432 - 29107600
    checkpoint_sizes = [row["bytes"] for row in summary["files"]
                        if row["path"].endswith(".pt")]
    assert len(checkpoint_sizes) == 3
    assert all(size > 9_000_000 for size in checkpoint_sizes)
    assert sum(checkpoint_sizes) > summary["old_cap_bytes"]
    report = _report(full_runs["fit_root"])
    assert report["artifact_budget"]["cap_bytes"] == 33554432
    assert report["artifact_budget"]["old_cap_underprovisioned"] is True


def test_too_small_artifact_cap_is_rejected_and_overflow_preserved(tmp_path):
    root = tmp_path / "overflow"
    root.mkdir()
    payload = root / "checkpoint-like.bin"
    payload.write_bytes(b"x" * 4096)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["budget"]["authorized_run_ceiling"]["artifact_bytes_max"] = 1024
    with pytest.raises(ArtifactBudgetExceeded) as exceeded:
        enforce_artifact_budget(root, config, context="regression")
    assert exceeded.value.cap_bytes == 1024
    assert exceeded.value.measured_bytes > 1024
    overflow_path = root / "artifact_budget_overflow.json"
    assert overflow_path.is_file()
    overflow = json.loads(overflow_path.read_text(encoding="utf-8"))
    assert overflow["status"] == "overflow"
    assert overflow["cap_bytes"] == 1024
    assert overflow["preserved"] is True
    assert overflow["deleted_artifacts"] == []
    assert payload.is_file()
    assert payload.stat().st_size == 4096
    summary = json.loads((root / "artifact_budget.json").read_text(
        encoding="utf-8"))
    assert summary["status"] == "overflow"
