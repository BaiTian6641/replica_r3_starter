"""CPU-only contracts for the bounded vision learning curve.

These tests exercise the predeclared five-readout config, reused historical
pins, readout prefix/digest/weight/generation failures and the T1/T2/T3 curve
decision.  They never load a model, run a native forward, step an optimizer or
read an evaluation split.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from replica_lab import vision_learnability_curve as curve
from replica_lab import vision_learnability_diagnostic as learn
from conftest import require_local_artifacts


ROOT = Path(__file__).resolve().parents[1]
CURVE_CONFIG = ROOT / "configs/vision_learnability_curve_v1.json"
GRADIENT_CONFIG = ROOT / "configs/vision_gradient_learnability_v1.json"
ABLATION_RECEIPT = (ROOT / "runs/vision_mean_only_ablation_v1_20261026_attempt09/"
                           "report.json")
TRAINING = (ROOT / "data/vision_balanced_alignment_v2_20261026/prepared04/"
                   "training/alignment.jsonl")
BATCHES = (ROOT / "data/vision_balanced_alignment_v2_20261026/prepared04/"
                  "training/batches.jsonl")
CAPTION_LABELS = (ROOT / "data/vision_balanced_alignment_v2_20261026/"
                         "prepared04/training/train_caption_labels.jsonl")


def _module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner_module():
    return _module("vision_learnability_curve_runner",
                   "scripts/run_vision_learnability_curve.py")


def _gradient_runner_module():
    return _module("vision_learnability_gradient_runner_for_curve",
                   "scripts/run_vision_learnability_diagnostic.py")


def _sha(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Config and historical pins
# ---------------------------------------------------------------------------


def test_curve_config_is_predeclared_with_five_prefixes_and_reused_pins():
    config = curve.load_config(CURVE_CONFIG)
    gradient = learn.load_config(GRADIENT_CONFIG)
    assert config["mode"] == curve.MODE == "curve"
    assert config["readouts"] == curve.READOUT_CONTRACT
    assert config["readouts"]["prefixes"] == [0, 1, 2, 4, 7]
    assert config["readouts"]["after_updates"] == [1, 2, 4, 7]
    assert config["access"]["advance_after"] == "first_readout"
    assert config["decision"]["precedence"] == list(curve.STATUSES)
    assert config["decision"]["trailing_slope"] == "descriptive_only"
    for key in ("prepared", "source", "precondition", "schedule",
                "initialization", "optimizer", "probes", "gates",
                "prompt_contract", "controls", "stop_tokens",
                "target_eos_token", "answer_max_new_tokens",
                "caption_max_new_tokens", "claims"):
        assert config[key] == gradient[key], key
    assert config["v0_gate"] == {
        "planning_authorized_on_pass": False,
        "run_v0": False,
        "requires_separate_predeclared_job": True,
        "heldout_sealed": True,
    }
    assert gradient["v0_gate"]["planning_authorized_on_pass"] is True
    for relative in curve.MIGRATION_PATHS:
        if relative in {"src/replica_lab/architecture/contracts.py",
                        "src/replica_lab/architecture/minicpm5.py",
                        "src/replica_lab/memory_system/session.py"}:
            assert config["implementation_pins"]["migrations"][relative][
                "current_sha256"] == _sha(relative)
            assert config["implementation_pins"]["migrations"][relative][
                "current_sha256"] != gradient["implementation_pins"][
                    "migrations"][relative]["current_sha256"]
        else:
            assert config["implementation_pins"]["migrations"][relative] == (
                gradient["implementation_pins"]["migrations"][relative])
    assert set(config["implementation_pins"]["migrations"]) == set(
        curve.MIGRATION_PATHS)
    assert config["implementation_pins"]["new_files"][
        "src/replica_lab/vision_learnability_diagnostic.py"] == (
            gradient["implementation_pins"]["new_files"][
                "src/replica_lab/vision_learnability_diagnostic.py"])
    assert config["implementation_pins"]["new_files"][
        "scripts/run_vision_learnability_diagnostic.py"] == (
            gradient["implementation_pins"]["new_files"][
                "scripts/run_vision_learnability_diagnostic.py"])
    for relative in curve.NEW_FILE_PINS:
        assert config["implementation_pins"]["new_files"][relative] == _sha(
            relative)
    assert set(config["implementation_pins"]["new_files"]) == set(
        curve.NEW_FILE_PINS)
    for key, value in gradient["limits"].items():
        assert config["limits"][key] == value
    assert config["limits"]["max_readouts"] == 5
    assert config["limits"]["max_scoring_generations"] == 490
    assert config["limits"]["max_teacher_forced_cells"] == 490
    assert config["limits"]["max_scoring_forwards"] == 1470
    assert config["limits"]["readout_weight_writes"] == 0


@pytest.mark.parametrize("mutation, message", [
    (lambda config: config["access"].update(
        {"advance_after": "updates_complete"}), "advance"),
    (lambda config: config["readouts"].update(
        {"prefixes": [0, 1, 2, 4]}), "Readout contract"),
    (lambda config: config["readouts"].update(
        {"after_updates": [1, 2, 4]}), "Readout contract"),
    (lambda config: config["decision"].update(
        {"t1_relative_reduction_min": 0.2}), "Decision threshold"),
    (lambda config: config["limits"].update(
        {"max_readouts": 4}), "limits"),
    (lambda config: config["limits"].update(
        {"max_scoring_forwards": 1469}), "limits"),
    (lambda config: config["implementation_pins"]["migrations"][
        "scripts/run_vision_curriculum.py"].update(
            {"parent_sha256": "0" * 64}), "Migration parent"),
    (lambda config: config["implementation_pins"]["new_files"].pop(
        "src/replica_lab/vision_learnability_curve.py"),
     "New-file"),
    (lambda config: config.update({"mode": "gradient"}),
     "Unknown curve mode"),
])
def test_curve_config_rejects_protocol_or_pin_drift(tmp_path, mutation,
                                                    message):
    config = json.loads(CURVE_CONFIG.read_text())
    mutation(config)
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(config))
    with pytest.raises(learn.LearnabilityDiagnosticError, match=message):
        curve.load_config(path)


def test_curve_cli_defaults_to_the_curve_mode_and_mps_device():
    parser = _runner_module().build_parser()
    args = parser.parse_args([
        "--config", "configs/vision_learnability_curve_v1.json",
        "--output", "runs/example",
        "--model-path", "checkpoints/minicpm5_source",
        "--model-lock", "configs/minicpm5_sealed_lock.local.json",
    ])
    assert args.mode == "curve"
    assert args.device == "mps"
    assert args.output == Path("runs/example")


@require_local_artifacts(
    "checkpoints/minicpm5_source",
    "checkpoints/siglip2_base_patch16_224_source",
    "data/vision_balanced_alignment_v2_20261026/prepared04",
    "runs/vision_balanced_alignment_v2_20261026/train04/alignment/extension",
    "runs/vision_collapse_diagnostic_v1_20261026_attempt02",
    "runs/vision_mean_only_ablation_v1_20261026_attempt09",
)
def test_curve_pins_verify_and_historical_gradient_migration_failure_is_preserved():
    curve_runner = _runner_module()
    gradient_runner = _gradient_runner_module()
    config = curve.load_config(CURVE_CONFIG)
    gradient = learn.load_config(GRADIENT_CONFIG)
    pins = curve_runner.verify_pins(config)
    assert pins["pins"]["mode"] == "curve"
    assert set(pins["pins"]["implementation"]["new_files"]) == set(
        curve.NEW_FILE_PINS)
    with pytest.raises(learn.LearnabilityDiagnosticError,
                       match="Migrated implementation bytes differ"):
        gradient_runner.verify_pins(gradient)
    for relative in (
            "src/replica_lab/vision_learnability_diagnostic.py",
            "scripts/run_vision_learnability_diagnostic.py"):
        assert gradient["implementation_pins"]["new_files"][
            relative] == _sha(relative)
    drifted = copy.deepcopy(config)
    drifted["implementation_pins"]["new_files"][
        "src/replica_lab/vision_learnability_curve.py"] = "0" * 64
    with pytest.raises(learn.LearnabilityDiagnosticError,
                       match="New implementation bytes differ"):
        curve_runner.verify_pins(drifted)


# ---------------------------------------------------------------------------
# Readout invariants
# ---------------------------------------------------------------------------


def _digest(prefix):
    return hashlib.sha256(f"projector:{prefix}".encode()).hexdigest()


def _readout(prefix, *, digest=None, generation_ok=True,
             weight_writes=0, state_changed=False):
    digest = _digest(prefix) if digest is None else digest
    state_sha = f"state:{prefix}"
    if state_changed:
        after_sha = state_sha + ":changed"
    else:
        after_sha = state_sha
    before = {
        "sha256": state_sha,
        "state_entries": 0 if prefix == 0 else 6,
        "steps": [] if prefix == 0 else [prefix],
        "max_step": prefix,
        "empty": prefix == 0,
    }
    after = {
        **before,
        "sha256": after_sha,
    }
    return {
        "format": curve.READOUT_FORMAT,
        "mode": curve.MODE,
        "prefix": prefix,
        "optimizer_steps": prefix,
        "optimizer_state_before": before,
        "optimizer_state_after": after,
        "optimizer_state_unchanged": not state_changed,
        "optimizer_state_empty": prefix == 0,
        "projector_state_sha256_before": digest,
        "projector_state_sha256_after": digest,
        "projector_state_sha256": digest,
        "projector_unchanged_by_readout": True,
        "projector_digest_matches_initialization": prefix == 0,
        "weight_change_names": [],
        "readout_weight_writes": weight_writes,
        "base_stamp_before": "sealed-base",
        "base_stamp_after": "sealed-base",
        "base_stamp_unchanged": True,
        "adapter_mode_after": "train",
        "adapter_train_restored": True,
        "counts": {
            "records": 104,
            "available": 98,
            "placeholders": 6,
            "generations": 98,
            "prefill_forwards": 98,
            "teacher_forced_cells": 98,
            "scoring_forwards": 294,
        },
        "generation": {
            "ok": generation_ok,
            "violations": [] if generation_ok else ["empty_ids"],
            "records": 104,
            "available": 98,
            "unavailable": 6,
        },
        "scores": {
            "answer": {
                "answer_prefill_gold_accuracy": 0.0,
                "answer_correct_accuracy": 0.0,
            },
            "caption": {"caption_token_f1": 0.1},
        },
        "teacher_forced_aggregates": {
            "answer": {
                "correct": {"mean": 4.0, "cases": 14},
                "noimage": {"mean": 4.5, "cases": 14},
            },
            "caption": {
                "correct": {"mean": 5.0, "cases": 12},
                "noimage": {"mean": 5.5, "cases": 12},
            },
        },
        "metrics": {"answer_prefill_gold_rank_mean": 10.0},
        "gates": (None if prefix != 7 else {
            "all_declared_gates_passed": False,
            "endpoint_context_only": True,
            "curve_decision_input": False,
        }),
    }


def _valid_readouts():
    return [_readout(prefix) for prefix in curve.PREFIXES]


def test_curve_readouts_accept_the_exact_five_prefix_contract():
    report = curve.validate_curve_readouts(
        _valid_readouts(), init_hash=_digest(0))
    assert report["ok"] is True
    assert report["readouts"] == 5
    assert report["total_generations"] == 490
    assert report["total_teacher_forced_cells"] == 490
    assert report["total_scoring_forwards"] == 1470
    assert report["endpoint_gate_reports"] == 1


@pytest.mark.parametrize("mutation, violation", [
    (lambda readouts: readouts[2].update({"prefix": 3}),
     "prefix:2:3!=2"),
    (lambda readouts: readouts[1].update(
        {"projector_state_sha256": _digest(0),
         "projector_state_sha256_before": _digest(0),
         "projector_state_sha256_after": _digest(0),
         "projector_digest_matches_initialization": False}),
     "prefix:1:projector_digest_repeated"),
    (lambda readouts: readouts[2].update(
        {"readout_weight_writes": 1}), "prefix:2:readout_weight_write"),
    (lambda readouts: readouts[3].update(
        {"weight_change_names": ["vision_projector.0.weight"]}),
     "prefix:4:readout_weight_write"),
    (lambda readouts: readouts[4].update(
        {"generation": {"ok": False, "violations": ["empty_ids"],
                        "records": 104, "available": 98,
                        "unavailable": 6}}),
     "prefix:7:generation"),
    (lambda readouts: readouts[4].update({"gates": None}),
     "prefix:7:endpoint_gates"),
    (lambda readouts: readouts[1].update(
        {"optimizer_state_unchanged": False}),
     "prefix:1:optimizer_state_changed"),
    (lambda readouts: readouts[0].update(
        {"optimizer_state_empty": False}), "prefix:0:optimizer_state_not_empty"),
    (lambda readouts: readouts[2].update(
        {"optimizer_steps": 1}), "prefix:2:optimizer_steps:1"),
    (lambda readouts: readouts[3].update(
        {"adapter_mode_after": "eval", "adapter_train_restored": False}),
     "prefix:4:adapter_not_train"),
])
def test_curve_readouts_reject_prefix_digest_weight_and_generation_drift(
        mutation, violation):
    readouts = _valid_readouts()
    mutation(readouts)
    report = curve.validate_curve_readouts(
        readouts, init_hash=_digest(0))
    assert report["ok"] is False
    assert violation in report["violations"]


def test_curve_readouts_reject_a_digest_that_never_changes_after_zero():
    readouts = _valid_readouts()
    for position, readout in enumerate(readouts):
        if position == 0:
            continue
        digest = _digest(0)
        readout.update({
            "projector_state_sha256": digest,
            "projector_state_sha256_before": digest,
            "projector_state_sha256_after": digest,
            "projector_digest_matches_initialization": False,
        })
    report = curve.validate_curve_readouts(
        readouts, init_hash=_digest(0))
    assert report["ok"] is False
    assert any("projector_digest" in item
               for item in report["violations"])


# ---------------------------------------------------------------------------
# T1/T2/T3 trajectory and decision
# ---------------------------------------------------------------------------


def _trajectory_readout(prefix, *, answer_loss, caption_loss,
                        prefill=0.0, correct=0.0, f1=0.1, rank=10.0):
    return {
        "prefix": prefix,
        "scores": {
            "answer": {
                "answer_prefill_gold_accuracy": prefill,
                "answer_correct_accuracy": correct,
            },
            "caption": {"caption_token_f1": f1},
        },
        "teacher_forced_aggregates": {
            "answer": {
                "correct": {"mean": answer_loss, "cases": 14},
                "noimage": {"mean": answer_loss + 0.5, "cases": 14},
            },
            "caption": {
                "correct": {"mean": caption_loss, "cases": 12},
                "noimage": {"mean": caption_loss + 0.5, "cases": 12},
            },
        },
        "metrics": {"answer_prefill_gold_rank_mean": rank},
    }


def _trajectory_readouts(*, answer_losses=None, caption_losses=None,
                         prefill=None, correct=None, f1=None, rank=None):
    answer_losses = answer_losses or [4.0, 3.0, 2.5, 2.2, 2.0]
    caption_losses = caption_losses or [5.0, 4.0, 3.5, 3.2, 3.0]
    prefill = prefill or [0.0] * 5
    correct = correct or [0.0] * 5
    f1 = f1 or [0.1] * 5
    rank = rank or [10.0, 9.0, 8.0, 7.0, 4.0]
    return [
        _trajectory_readout(
            prefix, answer_loss=answer_losses[index],
            caption_loss=caption_losses[index],
            prefill=prefill[index], correct=correct[index],
            f1=f1[index], rank=rank[index])
        for index, prefix in enumerate(curve.PREFIXES)
    ]


def test_curve_trajectory_reports_t1_t2_t3_and_no_conversion_inputs():
    readouts = _trajectory_readouts()
    trajectory = curve.curve_trajectory_report(readouts)
    assert trajectory["ok"] is True
    assert trajectory["t1"]["passed"] is True
    assert trajectory["t1"]["roles"]["answer"]["relative_reduction"] == 0.5
    assert trajectory["t1"]["roles"]["caption"]["relative_reduction"] == 0.4
    assert trajectory["t2"]["passed"] is True
    assert trajectory["t2"]["steps"] == 4
    assert trajectory["teacher_forced_loss"]["answer"] == [
        4.0, 3.0, 2.5, 2.2, 2.0]
    assert trajectory["teacher_forced_loss"]["caption"] == [
        5.0, 4.0, 3.5, 3.2, 3.0]
    assert trajectory["t2"]["roles"]["answer"]["series"][-1] == {
        "prefix": 7, "loss": 2.0}
    assert trajectory["t3"]["passed"] is True
    assert trajectory["t3"]["criteria"][
        "answer_prefill_gold_rank_mean_ratio"]["ratio"] == pytest.approx(0.4)
    assert trajectory["trailing_slope"]["descriptive_only"] is True
    assert trajectory["trailing_slope"]["answer"] == (
        2.0 - 2.2) / (7 - 4)
    assert set(trajectory["t3"]["criteria"]) == {
        "answer_prefill_gold_accuracy_from_zero",
        "answer_correct_accuracy",
        "caption_token_f1_gain",
        "answer_prefill_gold_rank_mean_ratio",
    }


@pytest.mark.parametrize("criterion", [
    "prefill",
    "correct",
    "caption_f1",
    "rank",
])
def test_curve_t3_conversion_accepts_each_predeclared_criterion(criterion):
    prefill = [0.0] * 5
    correct = [0.0] * 5
    f1 = [0.1] * 5
    rank = [10.0] * 5
    if criterion == "prefill":
        prefill[0] = 0.0
        prefill[-1] = 1.0 / 14.0
    elif criterion == "correct":
        correct[-1] = 1.0 / 14.0
    elif criterion == "caption_f1":
        f1[-1] = 0.2
    else:
        rank[-1] = 5.0
    trajectory = curve.curve_trajectory_report(_trajectory_readouts(
        prefill=prefill, correct=correct, f1=f1, rank=rank))
    assert trajectory["t3"]["passed"] is True
    decision = curve.curve_decision(
        engineering_report={"ok": True},
        trajectory_report=trajectory)
    assert decision["status"] == curve.STATUS_CONVERSION


def test_curve_decision_precedence_and_terminal_authorizations():
    conversion = curve.curve_trajectory_report(_trajectory_readouts())
    observed = curve.curve_decision(
        engineering_report={"ok": True}, trajectory_report=conversion)
    assert observed["status"] == curve.STATUS_CONVERSION
    optimization = curve.curve_trajectory_report(_trajectory_readouts(
        prefill=[0.0] * 5, correct=[0.0] * 5,
        f1=[0.1] * 5, rank=[10.0] * 5))
    assert optimization["t1"]["passed"] is True
    assert optimization["t2"]["passed"] is True
    assert optimization["t3"]["passed"] is False
    only = curve.curve_decision(
        engineering_report={"ok": True}, trajectory_report=optimization)
    assert only["status"] == curve.STATUS_OPTIMIZATION_ONLY
    no_progress = curve.curve_trajectory_report(_trajectory_readouts(
        answer_losses=[4.0] * 5, caption_losses=[5.0] * 5))
    assert no_progress["t1"]["passed"] is False
    stalled = curve.curve_decision(
        engineering_report={"ok": True}, trajectory_report=no_progress)
    assert stalled["status"] == curve.STATUS_NO_PROGRESS
    failed = curve.curve_decision(
        engineering_report={"ok": False, "violations": ["readout"]},
        trajectory_report=conversion)
    assert failed["status"] == curve.STATUS_ENGINEERING_FAILURE
    for decision in (observed, only, stalled, failed):
        assert decision["authorizes_v0"] is False
        assert decision["v0_authorized"] is False
        assert decision["v0_planning_authorized"] is False
        assert decision["v0_run"] is False
        assert decision["dev_authorized"] is False
        assert decision["dev_accessed"] is False
        assert decision["heldout_authorized"] is False
        assert decision["heldout_accessed"] is False
        assert decision["promotion"] is False
        assert decision["promotion_authorized"] is False
        assert decision["retry_authorized"] is False
        assert decision["no_automatic_retry"] is True
        assert decision["trailing_slope_used_in_decision"] is False
        assert decision["endpoint_gates_used_in_decision"] is False


def test_curve_t1_requires_both_roles_and_t2_uses_the_declared_tolerance():
    only_answer = curve.curve_trajectory_report(_trajectory_readouts(
        answer_losses=[4.0, 3.0, 2.5, 2.2, 2.0],
        caption_losses=[5.0] * 5))
    assert only_answer["t1"]["roles"]["answer"]["passed"] is True
    assert only_answer["t1"]["roles"]["caption"]["passed"] is False
    assert only_answer["t1"]["passed"] is False
    within = curve.curve_trajectory_report(_trajectory_readouts(
        answer_losses=[4.0, 4.0005, 4.0004, 4.0003, 4.0002],
        caption_losses=[5.0, 5.0005, 5.0004, 5.0003, 5.0002]))
    assert within["t2"]["passed"] is True
    beyond = curve.curve_trajectory_report(_trajectory_readouts(
        answer_losses=[4.0, 4.002, 4.001, 4.0005, 4.0004],
        caption_losses=[5.0, 5.002, 5.001, 5.0005, 5.0004]))
    assert beyond["t2"]["passed"] is False
    assert beyond["t2"]["roles"]["answer"]["steps"][0][
        "non_increasing"] is False


def test_curve_engineering_aggregate_surfaces_readout_and_cap_failures():
    readouts = _valid_readouts()
    readouts.append(_readout(3))
    engineering = curve.curve_engineering_report(
        trace=[], probes=[], limits=curve.load_config(CURVE_CONFIG)["limits"],
        schedule_report={}, generation_report={"ok": True, "violations": []},
        fit_coverage={}, access_report={"ok": True, "violations": []},
        precondition={"ok": True, "violations": []}, readouts=readouts,
        init_hash=_digest(0), updates_completed=7, optimizer_steps=7,
        update_backward_calls=28, probe_backward_calls=2,
        readout_weight_writes=0)
    assert engineering["ok"] is False
    assert any("readout" in item or "gradient" in item
               for item in engineering["violations"])


def test_curve_engineering_aggregate_accepts_exact_readouts(monkeypatch):
    monkeypatch.setattr(
        curve.learn, "gradient_engineering_report",
        lambda **kwargs: {"ok": True, "violations": [], "caps": {}})
    engineering = curve.curve_engineering_report(
        trace=[], probes=[], limits=curve.load_config(CURVE_CONFIG)["limits"],
        schedule_report={"ok": True}, generation_report={"ok": True},
        fit_coverage={}, access_report={"ok": True, "violations": []},
        precondition={"ok": True, "violations": []},
        readouts=_valid_readouts(), init_hash=_digest(0),
        updates_completed=7, optimizer_steps=7,
        update_backward_calls=28, probe_backward_calls=2,
        readout_weight_writes=0)
    assert engineering["ok"] is True
    assert engineering["caps"]["readouts"] == 5
    assert engineering["caps"]["scoring_generations"] == 490
    assert engineering["caps"]["teacher_forced_cells"] == 490
    assert engineering["caps"]["scoring_forwards"] == 1470
    assert engineering["trajectory"]["ok"] is True


def test_curve_stop_report_is_labelled_curve_not_gradient():
    report = curve.curve_stop_conditions_report(
        wall_seconds=1.0, wall_limit_seconds=900,
        artifact_bytes=10, artifact_limit_bytes=67108864,
        updates=7, optimizer_steps=7, update_backward_calls=28,
        probe_backward_calls=2, model_loads=1, extension_loads=0,
        extension_writes=0, checkpoint_writes=0, downloads=0,
        cpu_fallback=False, sealed_evaluation_access=0,
        guard_stop_reason=None, readouts=5, scoring_generations=490,
        teacher_forced_cells=490, scoring_forwards=1470,
        readout_weight_writes=0)
    assert report["mode"] == "curve"
    assert report["format"] == curve.STOP_FORMAT
    assert report["exact_readouts"] is True
    assert report["exact_scoring_generations"] is True
    assert report["exact_teacher_forced_cells"] is True
    assert report["exact_scoring_forwards"] is True
    curve.require_curve_stop_conditions(report)
    over = curve.curve_stop_conditions_report(
        wall_seconds=1.0, wall_limit_seconds=900,
        artifact_bytes=10, artifact_limit_bytes=67108864,
        updates=7, optimizer_steps=7, update_backward_calls=28,
        probe_backward_calls=2, model_loads=1, extension_loads=0,
        extension_writes=0, checkpoint_writes=0, downloads=0,
        cpu_fallback=False, sealed_evaluation_access=0,
        guard_stop_reason=None, readouts=6, scoring_generations=490,
        teacher_forced_cells=490, scoring_forwards=1470,
        readout_weight_writes=0)
    assert "max_readouts_exceeded" in over["violations"]
    with pytest.raises(learn.LearnabilityDiagnosticError,
                       match="stop condition"):
        curve.require_curve_stop_conditions(over)


@require_local_artifacts(
    "data/vision_balanced_alignment_v2_20261026/prepared04",
    "runs/vision_mean_only_ablation_v1_20261026_attempt09",
)
def test_optional_real_curve_artifacts_are_absent_or_parseable():
    """The optional native run is deliberately not executed here."""
    if not ABLATION_RECEIPT.exists() or not TRAINING.exists():
        pytest.skip("real curve artifacts are absent")
    receipt = json.loads(ABLATION_RECEIPT.read_text())
    assert receipt["mode"] == "ablation"
    assert receipt["decision"]["decision"] == learn.DECISION_OBSERVED
    assert BATCHES.exists() and CAPTION_LABELS.exists()
