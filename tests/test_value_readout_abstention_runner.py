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
    enforce_resume_artifact_budget, run_controls, run_development, run_fit,
    run_heldout, sha256_file,
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
    assert report["physical_work"] == {
        "scope": "fit_treatment_and_two_control_arms_plus_zero_update_baselines",
        "optimizer_steps": 72,
        "backward_calls": 72,
        "training_forward_calls": 72,
        "evaluation_forward_calls": 135,
        "retention_diagnostic_forwards": 15,
        "training_write_events": 27,
        "evaluation_write_events": 45,
        "checkpoint_writes": 3,
        "matches_global_logical_updates": True,
        "process_history": "resume_evidence.json",
    }
    baselines = report["zero_update_baselines"]
    assert baselines["parent"]["optimizer_steps"] == 0
    assert baselines["parent"]["case_accuracy"] == 1.0
    assert baselines["parent"]["all_declared_gates_passed"] is True
    assert baselines["reset"]["case_accuracy"] == pytest.approx(4 / 9)
    assert baselines["reset"]["all_declared_gates_passed"] is False
    assert baselines["integration_vs_learning"][
        "learning_improvement_claimed"] is False
    assert baselines["integration_vs_learning"][
        "parent_already_passes_all_fit_gates"] is True
    evidence = json.loads(
        (full_runs["fit_root"] / "resume_evidence.json").read_text(
            encoding="utf-8"))
    assert evidence["process_count"] == 1
    assert evidence["optimizer_steps_by_process"] == [72]
    assert evidence["cumulative_work"]["optimizer_steps"] == 72
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


def _assert_resume_equivalence(target: Path, baseline_root: Path) -> dict:
    baseline = _report(baseline_root)
    report = _report(target)
    assert report["completed_steps"] == 24
    assert report["resulting_state_sha256"] == baseline["resulting_state_sha256"]
    assert report["update_records"] == baseline["update_records"]
    # The report and receipt bind model work only; process history lives in the
    # separate resume evidence, so resumed and uninterrupted reports agree.
    for relative in (
            "fit/checkpoint.pt", "fit/fit_predictions.jsonl",
            "fit/fit_report.json", "fit_receipt.json",
            "fit/baseline_parent_predictions.jsonl",
            "fit/baseline_reset_predictions.jsonl",
            "fit/control_shuffled_values.pt",
            "fit/control_wrong_world_values.pt",
            "fit/control_shuffled_values_predictions.jsonl",
            "fit/control_wrong_world_values_predictions.jsonl"):
        assert sha256_file(target / relative) == sha256_file(
            baseline_root / relative), relative
    for arm_id in ("shuffled_values", "wrong_world_values"):
        resumed_payload = _payload(target / "fit" / f"control_{arm_id}.pt")
        baseline_payload = _payload(
            baseline_root / "fit" / f"control_{arm_id}.pt")
        assert resumed_payload["state_sha256"] == \
            baseline_payload["state_sha256"]
        assert resumed_payload["update_records"] == \
            baseline_payload["update_records"]
    return json.loads((target / "resume_evidence.json").read_text(
        encoding="utf-8"))


def test_interruption_and_resume_are_bit_identical(full_runs, tmp_path):
    target = tmp_path / "interrupted"
    with pytest.raises(RunInterrupted) as interrupted:
        run_fit(PREPARED, CONFIG_PATH, target, interrupt_after=5)
    assert interrupted.value.completed_updates == 5
    assert (target / "resume.pt").is_file()
    state = _payload(target / "resume.pt")
    assert state["format"] == runner_module.RESUME_FORMAT
    assert state["cursor"]["next_arm_index"] == 0
    assert state["cursor"]["completed_arm_ids"] == []
    assert state["cursor"]["active_arm_id"] == "learned_value_readout"
    assert state["processes"][-1]["status"] == "interrupted"
    assert state["cumulative_work"]["optimizer_steps"] == 5
    resumed = run_fit(PREPARED, CONFIG_PATH, target, resume=True)
    evidence = _assert_resume_equivalence(target, full_runs["fit_root"])
    assert resumed["physical_work"]["optimizer_steps"] == 72
    assert evidence["optimizer_steps_by_process"] == [5, 67]
    assert evidence["cumulative_work"]["optimizer_steps"] == 72
    assert evidence["completed_arms_never_retrained"] is True


@pytest.mark.parametrize("cut", [24, 25, 29, 48, 49])
def test_resume_at_arm_boundaries_never_retrains(full_runs, tmp_path, cut):
    target = tmp_path / f"boundary_{cut}"
    with pytest.raises(RunInterrupted) as interrupted:
        run_fit(PREPARED, CONFIG_PATH, target, interrupt_after=cut)
    assert interrupted.value.completed_updates == cut
    state = _payload(target / "resume.pt")
    completed = state["cursor"]["completed_arm_ids"]
    assert completed == list(runner_module.TRAINED_ARMS)[:cut // 24]
    resumed = run_fit(PREPARED, CONFIG_PATH, target, resume=True)
    evidence = _assert_resume_equivalence(target, full_runs["fit_root"])
    assert evidence["optimizer_steps_by_process"] == [cut, 72 - cut]
    assert evidence["cumulative_work"]["optimizer_steps"] == 72
    assert resumed["physical_work"]["optimizer_steps"] == 72
    assert evidence["completed_arms_never_retrained"] is True


def test_multiple_restarts_accumulate_physical_work_without_replay(
        full_runs, tmp_path):
    target = tmp_path / "multiple"
    with pytest.raises(RunInterrupted):
        run_fit(PREPARED, CONFIG_PATH, target, interrupt_after=24)
    with pytest.raises(RunInterrupted):
        run_fit(PREPARED, CONFIG_PATH, target, resume=True,
                interrupt_after=49)
    resumed = run_fit(PREPARED, CONFIG_PATH, target, resume=True)
    evidence = _assert_resume_equivalence(target, full_runs["fit_root"])
    assert evidence["optimizer_steps_by_process"] == [24, 25, 23]
    assert evidence["process_count"] == 3
    assert evidence["restart_count"] == 2
    assert sum(evidence["optimizer_steps_by_process"]) == 72
    assert evidence["cumulative_work"]["backward_calls"] == 72
    assert evidence["cumulative_work"]["training_forward_calls"] == 72
    assert resumed["physical_work"]["training_write_events"] == 27


def test_resume_rejects_changed_input_or_config_identity(tmp_path):
    target = tmp_path / "identity"
    with pytest.raises(RunInterrupted):
        run_fit(PREPARED, CONFIG_PATH, target, interrupt_after=5)
    state = _payload(target / "resume.pt")
    state["identities"]["event_sequence_sha256"] = "0" * 64
    torch.save(state, target / "resume.pt")
    with pytest.raises(runner_module.ReadoutError):
        run_fit(PREPARED, CONFIG_PATH, target, resume=True)


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
                     "fit/control_shuffled_values.pt",
                     "fit/control_wrong_world_values.pt"):
        assert sha256_file(output / relative) == sha256_file(
            full_runs["fit_root"] / relative), relative
    assert json.loads((output / "fit_receipt.json").read_text(
        encoding="utf-8")) == json.loads(
            (full_runs["fit_root"] / "fit_receipt.json").read_text(
                encoding="utf-8"))
    evidence = json.loads((output / "resume_evidence.json").read_text(
        encoding="utf-8"))
    assert evidence["optimizer_steps_by_process"] == [4, 68]
    assert evidence["process_count"] == 2
    assert evidence["cumulative_work"]["optimizer_steps"] == 72


def test_all_cli_interruption_and_resume_preserve_cursor(full_runs, tmp_path):
    root = tmp_path / "all_cli"
    env = {"PYTHONDONTWRITEBYTECODE": "1",
           "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
    command = [sys.executable,
               str(ROOT / "scripts" / "run_value_readout_abstention_v1.py"),
               "all", "--output", str(root)]
    first = subprocess.run(command + ["--interrupt-after", "29"], cwd=ROOT,
                           capture_output=True, text=True, timeout=180,
                           env=env)
    assert first.returncode == 3, first.stderr
    payload = json.loads(first.stdout)
    assert payload["status"] == "interrupted"
    assert payload["completed_updates"] == 29
    resume_path = Path(payload["resume"])
    assert resume_path == root / "fit" / "resume.pt"
    assert resume_path.is_file()
    state = _payload(resume_path)
    assert state["cursor"]["completed_arm_ids"] == [
        "learned_value_readout"]
    assert state["cursor"]["next_arm_index"] == 1
    assert state["cursor"]["active_arm_id"] == "shuffled_values"
    assert state["cursor"]["global_optimizer_steps"] == 29

    second = subprocess.run(command + ["--resume"], cwd=ROOT,
                            capture_output=True, text=True, timeout=180,
                            env=env)
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["status"] == "completed"
    assert not resume_path.exists()
    assert not (root / "fit" / "resume_arms").exists()
    for relative, baseline_key, baseline_relative in (
            ("fit/fit/checkpoint.pt", "fit_root", "fit/checkpoint.pt"),
            ("fit/fit/fit_predictions.jsonl", "fit_root",
             "fit/fit_predictions.jsonl"),
            ("fit/fit/fit_report.json", "fit_root", "fit/fit_report.json"),
            ("fit/fit_receipt.json", "fit_root", "fit_receipt.json"),
            ("dev_run/dev_receipt.json", "dev_root", "dev_receipt.json"),
            ("heldout_run/heldout_receipt.json", "heldout_root",
             "heldout_receipt.json"),
            ("controls_run/controls/dev_control_report.json", "controls_root",
             "controls/dev_control_report.json")):
        assert sha256_file(root / relative) == sha256_file(
            full_runs[baseline_key] / baseline_relative), relative
    evidence = json.loads((root / "fit" / "resume_evidence.json").read_text(
        encoding="utf-8"))
    assert evidence["optimizer_steps_by_process"] == [29, 43]
    assert evidence["completed_arms_never_retrained"] is True


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
    assert summary["resume_artifacts_included"] is True
    assert "resume_evidence.json" in summary["resume_artifact_paths"]
    assert summary["resume_artifact_bytes"] > 0
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


def test_resume_artifacts_are_in_budget_and_overflow_preserves_them(tmp_path):
    root = tmp_path / "resume_budget"
    with pytest.raises(RunInterrupted):
        run_fit(PREPARED, CONFIG_PATH, root, interrupt_after=49)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    declared_cap = config["budget"]["authorized_run_ceiling"][
        "artifact_bytes_max"]
    resume_files = [root / "resume.pt"] + sorted(
        (root / "resume_arms").glob("*.pt"))
    assert resume_files and all(path.is_file() for path in resume_files)
    measured = enforce_resume_artifact_budget(root, config,
                                              context="resume_regression")
    assert measured["cap_bytes"] == declared_cap
    assert measured["measured_bytes"] <= declared_cap
    assert measured["resume_artifact_bytes"] == sum(
        path.stat().st_size for path in resume_files)
    assert set(measured["resume_artifact_paths"]) >= {
        "resume.pt", "resume_arms/learned_value_readout.pt",
        "resume_arms/shuffled_values.pt"}

    tight = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    tight["budget"]["authorized_run_ceiling"]["artifact_bytes_max"] = 1024
    with pytest.raises(ArtifactBudgetExceeded) as exceeded:
        enforce_resume_artifact_budget(root, tight, context="resume_overflow")
    assert exceeded.value.cap_bytes == 1024
    assert exceeded.value.measured_bytes > 1024
    overflow = json.loads((root / "artifact_budget_overflow.json").read_text(
        encoding="utf-8"))
    assert overflow["context"] == "resume_overflow"
    assert overflow["preserved"] is True
    assert set(overflow["resume_artifacts_within_measurement"]) >= {
        "resume.pt", "resume_arms/learned_value_readout.pt",
        "resume_arms/shuffled_values.pt"}
    assert overflow["deleted_artifacts"] == []
    assert all(path.is_file() for path in resume_files)
    # The declared 32 MiB cap was never changed by the overflow.
    assert config["budget"]["authorized_run_ceiling"][
        "artifact_bytes_max"] == 33554432
