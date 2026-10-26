"""CPU-only contracts for the vision-learnability diagnostics.

These tests exercise the mean-only ablation math and invariants, baseline
identity precedence, the paired decision rule, the seven-update prefix audit,
gradient ownership/finiteness/cap evidence, single-hypothesis generation
evidence, exact per-role gate separation and train-only access order.  They
never load a model, run a native forward, execute an optimizer step or read an
evaluation split.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from replica_lab import vision_learnability_diagnostic as learn
from replica_lab.vision_collapse_diagnostic import control_plan
from conftest import require_local_artifacts


ROOT = Path(__file__).resolve().parents[1]
ABLATION_CONFIG = ROOT / "configs/vision_mean_only_ablation_v1.json"
GRADIENT_CONFIG = ROOT / "configs/vision_gradient_learnability_v1.json"
ABLATION_RECEIPT = (ROOT / "runs/vision_mean_only_ablation_v1_20261026_attempt09/"
                           "report.json")
TRAINING = (ROOT / "data/vision_balanced_alignment_v2_20261026/prepared04/"
                   "training/alignment.jsonl")
BATCHES = (ROOT / "data/vision_balanced_alignment_v2_20261026/prepared04/"
                  "training/batches.jsonl")
CAPTION_LABELS = (ROOT / "data/vision_balanced_alignment_v2_20261026/"
                         "prepared04/training/train_caption_labels.jsonl")


def _rows():
    return [json.loads(line) for line in TRAINING.read_text().splitlines()
            if line.strip()]


def _batches():
    return [json.loads(line) for line in BATCHES.read_text().splitlines()
            if line.strip()]


def _labels():
    return [json.loads(line) for line in CAPTION_LABELS.read_text().splitlines()
            if line.strip()]


def _runner_module():
    spec = importlib.util.spec_from_file_location(
        "vision_learnability_runner",
        ROOT / "scripts/run_vision_learnability_diagnostic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Mean-only projected-patch ablation
# ---------------------------------------------------------------------------


def _ablation_input():
    projected = torch.tensor([
        [[1.0, 0.0], [3.0, 0.0], [5.0, 0.0], [0.0, 0.0],
         [0.0, 0.0], [0.0, 0.0]],
        [[2.0, 4.0], [4.0, 4.0], [0.0, 0.0], [0.0, 0.0],
         [0.0, 0.0], [0.0, 0.0]],
    ])
    valid = torch.tensor([[True, True, True, False, False, False],
                          [True, True, False, False, False, False]])
    return projected, valid


def test_mean_only_replacement_preserves_mean_and_zeros_spatial_variance():
    projected, valid = _ablation_input()
    original = projected.clone()
    patched, receipts = learn.mean_only_projected_patches(projected, valid)
    assert torch.equal(projected, original)
    assert patched.shape == projected.shape
    assert torch.equal(patched[0, :3], torch.tensor([[3.0, 0.0]] * 3))
    assert torch.equal(patched[1, :2], torch.tensor([[3.0, 4.0]] * 2))
    assert torch.equal(patched[0, 3:], torch.zeros(3, 2))
    assert all(item["within_tolerance"] for item in receipts)
    assert receipts[0]["valid_tokens"] == 3
    assert receipts[0]["max_abs_deviation_from_mean"] == 0.0
    assert receipts[0]["spatial_variance"] == 0.0
    assert receipts[1]["valid_tokens"] == 2


def test_mean_only_rejects_a_nonzero_invalid_position():
    projected, valid = _ablation_input()
    projected = projected.clone()
    projected[0, 5, 0] = 1.0
    _, receipts = learn.mean_only_projected_patches(projected, valid)
    assert receipts[0]["invalid_positions_zero"] is False
    assert receipts[0]["within_tolerance"] is False


class _Visual:
    def __init__(self, features, valid, sha="a" * 64):
        self.features = features
        self.valid = valid
        self.image_sha256 = sha
        self.producer = "producer"
        self.task_id = "task"

    def assert_immutable(self):
        return None


class _Adapter:
    def __init__(self, projected):
        self.projected = projected
        self.calls = 0

    def project_visual(self, visual):
        self.calls += 1
        return self.projected


def test_mean_only_wrapper_installs_restores_and_never_mutates_visual():
    projected, valid = _ablation_input()
    visual = _Visual(torch.randn_like(projected[0]), valid[0])
    adapter = _Adapter(projected[0])
    before = adapter.__dict__.copy()
    with learn.mean_only_projector(adapter) as wrapper:
        out = adapter.project_visual(visual)
        assert out.shape == projected[0].shape
        summary = wrapper.summary()
        assert summary["installed"] is True
        assert summary["unique_images"] == 1
        assert wrapper.calls == 1
    wrapper.assert_restored()
    assert adapter.calls == 1
    assert "project_visual" not in adapter.__dict__
    assert set(before) == set(adapter.__dict__)
    final = wrapper.summary()
    assert final["installed"] is False
    assert final["all_within_tolerance"] is True
    receipts = learn.verify_ablation_receipts(
        final, expected_image_shas=[visual.image_sha256],
        expected_valid_tokens=3)
    assert receipts["ok"] is True
    wrong = learn.verify_ablation_receipts(
        final, expected_image_shas=["b" * 64], expected_valid_tokens=3)
    assert wrong["ok"] is False


def _image_stats(mean_patch, *, spatial):
    return {
        "tokens": 196,
        "projected_rms": 2.0,
        "projected_max": 3.0,
        "mean_vector_rms": 2.0,
        "spatial_variance": spatial,
        "mean_patch": list(mean_patch),
    }


def _feature_summary(spatial):
    return {
        "pca_content": {"ratio": 0.5, "top_singular_squared": 10.0,
                        "total_singular_squared": 20.0},
        "sibling_cosine_distance": {"mean": 0.1},
        "cross_pair_cosine_distance": {"mean": 0.2},
        "projected_mean_vector_rms_mean": 2.0,
        "within_image_spatial_variance_mean": spatial,
        "encoder_feature_matrix_sha256": "e" * 64,
        "projected_feature_matrix_sha256": "p" * 64,
    }


def test_ablation_stage_invariants_require_collapse_and_mean_preservation():
    full = {"c1": _image_stats([1.0, 2.0], spatial=0.75)}
    collapsed = {"c1": _image_stats([1.0, 2.0], spatial=0.0)}
    stats = learn.verify_mean_only_image_stats(full, collapsed)
    assert stats["ok"] is True
    shifted = {"c1": _image_stats([1.1, 2.0], spatial=0.0)}
    wrong = learn.verify_mean_only_image_stats(full, shifted)
    assert wrong["ok"] is False
    assert wrong["violations"] == ["c1"]
    same = learn.verify_mean_preservation(
        _feature_summary(0.75), _feature_summary(0.0))
    assert same["ok"] is True
    drift = _feature_summary(0.0)
    assert learn.verify_mean_preservation(
        _feature_summary(0.75), drift)["ok"] is True
    drift["projected_mean_vector_rms_mean"] = 2.1
    assert learn.verify_mean_preservation(
        _feature_summary(0.75), drift)["ok"] is False


# ---------------------------------------------------------------------------
# Baseline identity and decision precedence
# ---------------------------------------------------------------------------


def _identity_record(case_id, control, *, available=True, gap=1.0):
    record = {
        "case_id": case_id,
        "control": control,
        "kind": "answer",
        "available": available,
        "generation": {"generated_ids": [5, 1], "generated_text": "yes"}
        if available else {},
        "first_token": {
            "gold_token": 5,
            "opposite_token": 7,
            "argmax_token": 5,
            "gold_rank": 1,
            "gold_logit": 3.0,
            "opposite_logit": 3.0 - gap,
            "gap": gap,
        } if available else {},
        "positions": {"mean_nll": 0.5, "loss_tokens": 2}
        if available else {},
    }
    return record


def _stage_payload(*, gap=1.0):
    records = []
    placeholder_cases = {f"c{index}" for index in range(6)}
    for index in range(26):
        case_id = f"c{index}"
        for control in learn.CONTROL_NAMES:
            available = not (control == "sibling"
                             and case_id in placeholder_cases)
            records.append(_identity_record(case_id, control,
                                            available=available, gap=gap))
    cells = []
    for index in range(72):
        cells.append({
            "image_role": "own",
            "target_role": f"t{index}",
            "image_case_id": f"i{index}",
            "target_case_id": f"c{index}",
            "first_token": {
                "gold_token": 5, "opposite_token": 7,
                "argmax_token": 5, "gold_rank": 1,
                "gold_logit": 3.0, "opposite_logit": 2.0, "gap": 1.0,
            },
            "positions": {"mean_nll": 0.5, "loss_tokens": 2},
        })
    return {
        "records": records,
        "cells": cells,
        "teacher_forced_cells": 98,
        "fit_coverage": {"all_rows_checked": True, "checked_rows": 26},
    }


def _baseline_rows(stage=learn.BASELINE_STAGE, gap=1.0):
    rows = []
    placeholder_cases = {f"c{index}" for index in range(6)}
    for index in range(26):
        case_id = f"c{index}"
        for control in learn.CONTROL_NAMES:
            available = not (control == "sibling"
                             and case_id in placeholder_cases)
            rows.append({"stage": stage,
                         **_identity_record(case_id, control,
                                            available=available, gap=gap)})
    return rows


def _baseline_cells(stage=learn.BASELINE_STAGE):
    cells = []
    for index in range(72):
        cells.append({
            "stage": stage,
            "image_role": "own",
            "target_role": f"t{index}",
            "image_case_id": f"i{index}",
            "target_case_id": f"c{index}",
            "first_token": {
                "gold_token": 5, "opposite_token": 7,
                "argmax_token": 5, "gold_rank": 1,
                "gold_logit": 3.0, "opposite_logit": 2.0, "gap": 1.0,
            },
            "positions": {"mean_nll": 0.5, "loss_tokens": 2},
        })
    return cells


def test_baseline_identity_matches_exact_fields_or_reports_mismatch():
    stage = _stage_payload()
    report = learn.baseline_identity_report(
        full_stage=stage, baseline_rows=_baseline_rows(),
        baseline_cells=_baseline_cells())
    assert report["match"] is True
    assert report["checks"]["available_records"] == 98
    assert report["checks"]["teacher_forced_cells"] == 98
    drift = _stage_payload(gap=1.01)
    mismatch = learn.baseline_identity_report(
        full_stage=drift, baseline_rows=_baseline_rows(),
        baseline_cells=_baseline_cells())
    assert mismatch["match"] is False
    assert any(item["field"] == "first_token.gap"
               for item in mismatch["mismatches"])


def _decision_config():
    return json.loads(ABLATION_CONFIG.read_text())["decision"]


def test_decision_precedence_is_baseline_then_observed_then_not_established():
    observed = {"changed_pair_rows": 5, "answer_gap_reduction_rows": 0}
    below = {"changed_pair_rows": 4, "answer_gap_reduction_rows": 2}
    config = _decision_config()
    mismatch = learn.ablation_decision(
        baseline_match=False, paired=observed, decision=config)
    assert mismatch["decision"] == learn.DECISION_BASELINE_MISMATCH
    assert mismatch["gradient_diagnostic_authorized"] is False
    assert mismatch["v0_run"] is False
    hit = learn.ablation_decision(
        baseline_match=True, paired=observed, decision=config)
    assert hit["decision"] == learn.DECISION_OBSERVED
    assert hit["gradient_diagnostic_authorized"] is True
    assert hit["v0_authorized"] is False
    gap_hit = learn.ablation_decision(
        baseline_match=True,
        paired={"changed_pair_rows": 0, "answer_gap_reduction_rows": 3},
        decision=config)
    assert gap_hit["decision"] == learn.DECISION_OBSERVED
    miss = learn.ablation_decision(
        baseline_match=True, paired=below, decision=config)
    assert miss["decision"] == learn.DECISION_NOT_ESTABLISHED
    assert miss["sequence"] == "stopped"
    assert miss["gradient_diagnostic_authorized"] is False


def _pa_ablation_records():
    full = []
    mean = []
    for index in range(26):
        case_id = f"c{index}"
        for control in learn.CONTROL_NAMES:
            full.append({
                "case_id": case_id, "control": control, "kind": "answer",
                "available": True,
                "generation": {"generated_ids": [5, 1]},
                "first_token": {"gap": 2.0},
            })
            changed = control == "sibling" and index < 5
            mean.append({
                "case_id": case_id, "control": control, "kind": "answer",
                "available": True,
                "generation": {"generated_ids": [6, 1] if changed else [5, 1]},
                "first_token": {"gap": 1.0 if index < 3 else 2.0},
            })
    return full, mean


def test_paired_ablation_counts_changed_rows_and_gap_reductions():
    full, mean = _pa_ablation_records()
    report = learn.paired_ablation_report(full_records=full,
                                          mean_records=mean)
    assert report["changed_sibling_rows"] == 5
    assert report["changed_pair_rows"] == 5
    assert report["answer_gap_rows"] == 26
    assert report["answer_gap_reduction_rows"] == 3
    assert report["answer_gap_sign_flip_rows"] == 0
    assert report["per_control"]["sibling"]["changed"] == 5


# ---------------------------------------------------------------------------
# Seven-update full-coverage prefix
# ---------------------------------------------------------------------------


def test_seven_update_prefix_is_the_smallest_full_coverage_prefix():
    report = learn.prefix_schedule_report(_batches())
    assert report["updates"] == 7
    assert report["slots"] == 28
    assert report["unique_rows"] == 26
    assert report["unique_answer_rows"] == 14
    assert report["unique_caption_rows"] == 12
    assert report["first_full_coverage_update"] == 7
    assert sorted(report["repeated_example_ids"]) == [
        "caption:vba2-w014-p0-a", "caption:vba2-w014-p0-b"]
    learn.require_prefix_schedule(report)


def test_prefix_rejects_a_wrong_role_mix_or_incomplete_coverage():
    batches = _batches()
    mutated = copy.deepcopy(batches[:7])
    answer_index = next(index for index, example_id
                        in enumerate(mutated[0]["example_ids"])
                        if example_id.startswith("answer:"))
    caption_index = next(index for index, example_id
                         in enumerate(mutated[1]["example_ids"])
                         if example_id.startswith("caption:"))
    answer_id = mutated[0]["example_ids"][answer_index]
    caption_id = mutated[1]["example_ids"][caption_index]
    mutated[0]["example_ids"][answer_index] = caption_id
    mutated[1]["example_ids"][caption_index] = answer_id
    for batch, answers, captions in ((mutated[0], 1, 3),
                                     (mutated[1], 3, 1)):
        batch["answer_examples"] = answers
        batch["caption_examples"] = captions
    with pytest.raises(learn.LearnabilityDiagnosticError,
                       match="role mix|repeats|full-coverage"):
        learn.prefix_schedule_report(mutated)
    short = _batches()[:6]
    with pytest.raises(learn.LearnabilityDiagnosticError,
                       match="covers|short|full-coverage|unique"):
        learn.prefix_schedule_report(short)


# ---------------------------------------------------------------------------
# Gradient ownership, probes and caps
# ---------------------------------------------------------------------------


def _trace_record(index, **overrides):
    tensors = list(learn.PROJECTOR_TENSORS)
    changed = [tensors[(index - 1) % len(tensors)]]
    record = {
        "update": index,
        "roles": {"answer": 2, "caption": 2},
        "example_ids": [f"a{index}", f"b{index}", f"c{index}", f"d{index}"],
        "case_ids": [f"ca{index}", f"cb{index}", f"cc{index}", f"cd{index}"],
        "role_losses": {"answer": [1.0, 1.1], "caption": [1.2, 1.3]},
        "loss": 1.15,
        "grad_norm": 1.0,
        "projector_update_norm": 0.01,
        "nonprojector_gradient_names": [],
        "base_stamp_unchanged": True,
        "gradient_audit": {
            "original": {"present": [], "nonzero": [], "nonfinite": []},
            "new": {"present": tensors, "nonzero": tensors, "nonfinite": []},
        },
        "changed_projector_tensors": changed,
        "backward_calls": 4,
        "fit_contract_checked": True,
        "fit_contract_rows": [{"contract_valid": True}] * 4,
        "response_only_labels": True,
    }
    record.update(overrides)
    return record


def _limits():
    return json.loads(GRADIENT_CONFIG.read_text())["limits"]


def test_gradient_trace_requires_six_tensor_updates_and_exact_caps():
    trace = [_trace_record(index) for index in range(1, 8)]
    report = learn.validate_gradient_trace(trace, limits=_limits())
    assert report["ok"] is True
    assert report["update_backward_calls"] == 28
    assert len(report["changed_tensor_counts"]) == 6
    stuck = [_trace_record(index, changed_projector_tensors=[
        learn.PROJECTOR_TENSORS[0]]) for index in range(1, 8)]
    stuck_report = learn.validate_gradient_trace(stuck, limits=_limits())
    assert stuck_report["ok"] is False
    assert any("never_changed" in name
               for name in stuck_report["violations"])
    nonprojector = [_trace_record(index) for index in range(1, 8)]
    nonprojector[2]["nonprojector_gradient_names"] = ["base.weight"]
    assert learn.validate_gradient_trace(
        nonprojector, limits=_limits())["ok"] is False
    zero_update = [_trace_record(index) for index in range(1, 8)]
    zero_update[4]["projector_update_norm"] = 0.0
    assert learn.validate_gradient_trace(
        zero_update, limits=_limits())["ok"] is False
    missing = [_trace_record(index) for index in range(1, 8)]
    missing[0]["gradient_audit"]["new"]["nonzero"] = \
        learn.PROJECTOR_TENSORS[:-1]
    assert learn.validate_gradient_trace(
        missing, limits=_limits())["ok"] is False
    with pytest.raises(learn.LearnabilityDiagnosticError,
                       match="role mix"):
        learn.require_update_role_mix(
            _trace_record(1, roles={"answer": 1, "caption": 3}))


def _probe(role, **overrides):
    record = {
        "role": role,
        "backward_only": True,
        "probe_applied_update": False,
        "optimizer_steps": 0,
        "weight_change_names": [],
        "projector_state_sha256_before": "s" * 64,
        "projector_state_sha256_after": "s" * 64,
        "optimizer_state_empty": True,
        "loss": 1.0,
        "projector_gradient_names": list(learn.PROJECTOR_TENSORS),
        "projector_gradients_nonzero_names": list(learn.PROJECTOR_TENSORS),
        "projector_gradients_finite": True,
        "nonprojector_gradient_names": [],
        "backward_calls": 1,
    }
    record.update(overrides)
    return record


def test_backward_probes_require_answer_and_caption_ownership():
    probes = [_probe("answer"), _probe("caption")]
    report = learn.validate_backward_probes(probes)
    assert report["ok"] is True
    assert report["backward_calls"] == 2
    applied = [_probe("answer"), _probe("caption",
                                        probe_applied_update=True)]
    assert learn.validate_backward_probes(applied)["ok"] is False
    missing = [_probe("answer"), _probe(
        "caption", projector_gradients_nonzero_names=
        learn.PROJECTOR_TENSORS[:-1])]
    assert learn.validate_backward_probes(missing)["ok"] is False
    with pytest.raises(learn.LearnabilityDiagnosticError):
        learn.validate_backward_probes([_probe("answer")])


def test_gradient_caps_are_exact_and_sealed_from_writes():
    kwargs = dict(
        wall_seconds=10.0, wall_limit_seconds=900.0, artifact_bytes=100,
        artifact_limit_bytes=1024, updates=7, optimizer_steps=7,
        update_backward_calls=28, probe_backward_calls=2, model_loads=1,
        extension_loads=0, extension_writes=0, checkpoint_writes=0,
        downloads=0, cpu_fallback=False, sealed_evaluation_access=0,
        guard_stop_reason=None)
    report = learn.gradient_stop_conditions_report(**kwargs)
    assert report["violations"] == []
    learn.require_gradient_stop_conditions(report)
    too_many = learn.gradient_stop_conditions_report(
        **{**kwargs, "update_backward_calls": 29})
    assert too_many["violations"]
    with pytest.raises(learn.LearnabilityDiagnosticError):
        learn.require_gradient_stop_conditions(too_many)
    wrote = learn.gradient_stop_conditions_report(
        **{**kwargs, "extension_writes": 1})
    assert wrote["violations"] == ["extension_writes"]
    sealed = learn.gradient_stop_conditions_report(
        **{**kwargs, "sealed_evaluation_access": 1})
    assert sealed["violations"] == ["sealed_evaluation_access"]


# ---------------------------------------------------------------------------
# Single-hypothesis generation evidence
# ---------------------------------------------------------------------------


def _generation_rows():
    rows = []
    placeholder_cases = {f"c{index}" for index in range(6)}
    for index in range(26):
        case_id = f"c{index}"
        for control in learn.CONTROL_NAMES:
            available = not (control == "sibling"
                             and case_id in placeholder_cases)
            rows.append({
                "case_id": case_id,
                "control": control,
                "kind": "answer",
                "available": available,
                "generated_ids": [5, 1] if available else [],
                "generated_text": "yes" if available else None,
                "prefill_argmax_matches_first_token": available,
                "hypothesis_count": 1,
                "stop_tokens": [1, 130073] if available else [],
            })
    return rows


def test_generation_evidence_requires_one_matching_greedy_hypothesis():
    report = learn.generation_hypothesis_report(_generation_rows())
    assert report["ok"] is True
    assert report["available"] == 98
    assert report["prefill_agreement"] == 98
    broken = _generation_rows()
    broken[0]["prefill_argmax_matches_first_token"] = False
    broken[1]["hypothesis_count"] = 2
    broken[2]["generated_ids"] = []
    assert learn.generation_hypothesis_report(broken)["ok"] is False
    budget = _generation_rows()
    budget[0]["generated_ids"] = [5] * 9 + [1]
    assert learn.generation_hypothesis_report(budget)["ok"] is False


def test_terminal_stop_only_generation_is_not_empty_output():
    rows = _generation_rows()
    rows[0]["generated_ids"] = [1]
    rows[0]["generated_text"] = ""
    assert learn.generation_hypothesis_report(rows)["ok"] is True
    rows[0]["generated_ids"] = []
    assert learn.generation_hypothesis_report(rows)["ok"] is False
    rows[0]["generated_ids"] = [5]
    assert learn.generation_hypothesis_report(rows)["ok"] is False


# ---------------------------------------------------------------------------
# Train-fit scoring and exact per-role gates
# ---------------------------------------------------------------------------


def _fit_records():
    rows = _rows()
    plan = control_plan(rows)
    rows_by_case = {row["case_id"]: row for row in rows}
    labels = {row["case_id"]: row for row in _labels()}
    records = []
    answer_variants = 0
    for row in rows:
        case_id = row["case_id"]
        if row["kind"] == "answer":
            response = row["response"]
            variant = answer_variants % 3
            answer_variants += 1
            generated = [response, response + ".", " " + response][variant]
            records.append({
                "case_id": case_id, "control": "correct",
                "available": True,
                "generated_ids": [5, 1], "generated_text": generated,
                "prefill": {"argmax_is_gold": True},
                "kind": "answer",
            })
            if plan[case_id]["sibling_available"]:
                sibling = rows_by_case[plan[case_id]["sibling_case_id"]]
                records.append({
                    "case_id": case_id, "control": "sibling",
                    "available": True,
                    "generated_ids": [6, 1],
                    "generated_text": ("no" if row["response"] == "yes"
                                       else "yes"),
                    "prefill": {"argmax_is_gold": True},
                    "kind": "answer",
                })
                records[-1]["generated_text"] = sibling["response"]
            for control in ("shuffled", "noimage"):
                records.append({
                    "case_id": case_id, "control": control,
                    "available": True,
                    "generated_ids": [7, 1], "generated_text": "purple",
                    "prefill": {"argmax_is_gold": True},
                    "kind": "answer",
                })
        else:
            target = labels[case_id]["answer"]
            for control in ("correct", "sibling", "shuffled", "noimage"):
                if control == "sibling" and not plan[case_id][
                        "sibling_available"]:
                    records.append({
                        "case_id": case_id, "control": control,
                        "available": False, "generated_ids": [],
                        "generated_text": None, "kind": "caption",
                        "target_text": labels[case_id]["answer"],
                    })
                    continue
                if control == "correct":
                    generated = target
                    loss = 1.0
                elif control == "sibling":
                    donor = labels[plan[case_id]["sibling_case_id"]]
                    generated = donor["answer"]
                    loss = 1.2
                elif control == "shuffled":
                    donor_id = plan[case_id]["shuffled_case_id"]
                    generated = (labels[donor_id]["answer"]
                                 if donor_id in labels else "nothing")
                    loss = 1.2
                else:
                    generated = "nothing"
                    loss = 1.5
                ids = {"correct": [9, 1], "sibling": [10, 1],
                       "shuffled": [11, 1], "noimage": [12, 1]}[control]
                records.append({
                    "case_id": case_id, "control": control,
                    "available": True,
                    "generated_ids": ids,
                    "generated_text": generated,
                    "prefill_argmax_matches_first_token": True,
                    "teacher_forced_loss": loss,
                    "target_text": target,
                    "kind": "caption",
                })
    return records, plan, rows_by_case, labels


def test_train_fit_scoring_and_exact_per_role_gate_separation():
    records, plan, rows_by_case, labels = _fit_records()
    scores = learn.score_train_fit(
        records=records, plan=plan, rows_by_case=rows_by_case,
        caption_labels=labels)
    answer = scores["answer"]
    caption = scores["caption"]
    gates = learn.train_fit_gate_report(
        answer=answer, caption=caption, gates=learn.GATE_THRESHOLDS)
    assert gates["answer_gates_passed"] is True
    assert gates["caption_gates_passed"] is True
    assert gates["all_declared_gates_passed"] is True
    assert set(gates["gate_failures"]) == set()
    assert answer["answer_correct_accuracy"] == pytest.approx(1.0)
    assert answer["answer_correct_unique_texts"] >= 3
    assert caption["caption_semantic_f1"] == pytest.approx(1.0)
    assert caption["caption_sibling_original_semantic_drop"] >= 0.15
    assert caption["caption_shuffled_original_semantic_drop"] >= 0.15
    assert caption["caption_dependence_f1_margin"] >= 0.0
    assert caption["caption_noimage_semantic_f1"] <= 0.25
    assert caption["caption_noimage_loss_margin"] == pytest.approx(0.5)
    assert caption["caption_semantic_f1"] >= 0.5

    broken = dict(caption)
    broken["caption_semantic_f1"] = 0.0
    separated = learn.train_fit_gate_report(
        answer=answer, caption=broken, gates=learn.GATE_THRESHOLDS)
    assert separated["answer_gates_passed"] is True
    assert separated["caption_gates_passed"] is False
    assert separated["all_declared_gates_passed"] is False
    assert "caption_semantic_f1" in separated["caption_gate_failures"]
    assert separated["answer_gate_failures"] == []
    assert not set(separated["answer_gate_failures"]) & set(
        separated["caption_gate_failures"])

    unknown_gate = dict(learn.GATE_THRESHOLDS)
    unknown_gate["caption_unknown_gate"] = 0.0
    with pytest.raises(learn.LearnabilityDiagnosticError,
                       match="threshold"):
        learn.train_fit_gate_report(
            answer=answer, caption=caption, gates=unknown_gate)


# ---------------------------------------------------------------------------
# Train-only access order
# ---------------------------------------------------------------------------


def test_access_order_stops_before_dev_and_rejects_sealed_paths():
    order = learn.train_only_access_order()
    order.access("train")
    ok = learn.training_only_access_report(
        order.snapshot(),
        [str(TRAINING), str(BATCHES), str(CAPTION_LABELS)])
    assert ok["ok"] is True
    order.advance("train_fit")
    order.access("train_fit")
    assert learn.training_only_access_report(order.snapshot(), [])["ok"] is True
    with pytest.raises(Exception):
        order.advance("dev")
    assert order.snapshot()["phase"] == "train_fit"
    sealed = learn.training_only_access_report(
        order.snapshot(),
        [str(ROOT / "data/vision_balanced_alignment_v2_20261026/"
                  "prepared04/evaluation/dev_answer_cases.jsonl")])
    assert sealed["ok"] is False
    assert any("sealed_path" in item for item in sealed["violations"])
    wrong_phase = learn.training_only_access_report(
        {"phase": "dev", "order": ["train", "train_fit", "dev", "audit"],
         "events": [{"role": "dev"}]}, [])
    assert wrong_phase["ok"] is False


# ---------------------------------------------------------------------------
# Precondition and engineering aggregate
# ---------------------------------------------------------------------------


def _precondition_receipt(decision=learn.DECISION_OBSERVED):
    return {
        "format": learn.REPORT_FORMAT,
        "mode": "ablation",
        "output": "runs/example",
        "config_sha256": "c" * 64,
        "decision": {
            "decision": decision,
            "gradient_diagnostic_authorized":
                decision == learn.DECISION_OBSERVED,
            "v0_run": False,
        },
        "baseline_identity": {"match": True},
        "ablation_invariants": {"ok": True},
        "stop_conditions": {
            "no_gradients": True,
            "no_optimizer_steps": True,
            "no_sealed_evaluation_access": True,
        },
    }


def test_gradient_precondition_requires_the_observed_ablation_decision():
    report = learn.ablation_precondition_report(_precondition_receipt())
    assert report["ok"] is True
    refused = learn.ablation_precondition_report(
        _precondition_receipt(learn.DECISION_NOT_ESTABLISHED))
    assert refused["ok"] is False
    assert any("decision" in item for item in refused["violations"])
    config_hash = learn.ablation_precondition_report(
        _precondition_receipt(), config_sha256="d" * 64)
    assert config_hash["ok"] is False


def test_engineering_aggregate_reports_every_failure_component():
    trace = [_trace_record(index) for index in range(1, 8)]
    probes = [_probe("answer"), _probe("caption")]
    schedule = learn.prefix_schedule_report(_batches())
    generation = learn.generation_hypothesis_report(_generation_rows())
    coverage = {"all_rows_checked": True, "checked_rows": 26,
                "per_kind": {"answer": 14, "caption": 12}}
    access = {"ok": True, "violations": []}
    precondition = learn.ablation_precondition_report(_precondition_receipt())
    report = learn.gradient_engineering_report(
        trace=trace, probes=probes, limits=_limits(),
        schedule_report=schedule, generation_report=generation,
        fit_coverage=coverage, access_report=access,
        precondition=precondition, updates_completed=7,
        optimizer_steps=7, update_backward_calls=28,
        probe_backward_calls=2)
    assert report["ok"] is True
    assert report["caps"]["total_backward_calls"] == 30
    broken = learn.gradient_engineering_report(
        trace=trace, probes=probes, limits=_limits(),
        schedule_report=schedule, generation_report=generation,
        fit_coverage={**coverage, "checked_rows": 25},
        access_report=access, precondition=precondition,
        updates_completed=7, optimizer_steps=7,
        update_backward_calls=28, probe_backward_calls=2)
    assert broken["ok"] is False
    assert "fit_coverage_incomplete" in broken["violations"]


def test_gradient_disposition_never_runs_v0_and_keeps_failures_terminal():
    precondition = learn.ablation_precondition_report(_precondition_receipt())
    passing_gates = {"all_declared_gates_passed": True,
                     "gate_failures": [], "answer_gate_failures": [],
                     "caption_gate_failures": []}
    engineering = {"ok": True}
    passed = learn.gradient_decision(
        precondition_report=precondition, engineering_report=engineering,
        gate_report=passing_gates)
    assert passed["status"] == "learnability_pass"
    assert passed["v0_planning_authorized"] is True
    assert passed["v0_run"] is False
    failed_gates = {"all_declared_gates_passed": False,
                    "gate_failures": ["caption_semantic_f1"],
                    "answer_gate_failures": [],
                    "caption_gate_failures": ["caption_semantic_f1"]}
    stopped = learn.gradient_decision(
        precondition_report=precondition, engineering_report=engineering,
        gate_report=failed_gates)
    assert stopped["status"] == "stopped_before_dev_heldout_v0"
    assert stopped["v0_planning_authorized"] is False
    refused = learn.gradient_decision(
        precondition_report={"ok": False},
        engineering_report=engineering, gate_report=passing_gates)
    assert refused["status"] == "refused_ablation_precondition"


# ---------------------------------------------------------------------------
# Config declarations
# ---------------------------------------------------------------------------


def test_both_configs_declare_the_exact_protocol():
    ablation = learn.load_config(ABLATION_CONFIG)
    gradient = learn.load_config(GRADIENT_CONFIG)
    assert ablation["mode"] == "ablation"
    assert [stage["name"] for stage in ablation["stages"]] == [
        learn.FULL_STAGE, learn.MEAN_STAGE]
    assert ablation["limits"]["updates"] == 0
    assert ablation["limits"]["gradients"] == 0
    assert ablation["policy"]["no_optimizer"] is True
    assert gradient["mode"] == "gradient"
    assert gradient["schedule"]["prefix_updates"] == 7
    assert gradient["limits"]["max_total_backward_calls"] == 30
    assert gradient["gates"] == learn.GATE_THRESHOLDS
    assert gradient["v0_gate"]["run_v0"] is False
    assert gradient["access"]["order"] == ["train", "train_fit"]
    assert gradient["precondition"]["required_decision"] == (
        learn.DECISION_OBSERVED)


@pytest.mark.parametrize("mutation, message", [
    (lambda config: config.update({"stop_tokens": [1]}), "Stop tokens"),
    (lambda config: config["ablation"].update({"global_dataset_mean": True}),
     "Ablation declaration"),
    (lambda config: config["decision"]["observed_if"].update(
        {"changed_pair_rows_min": 1}), "Observed thresholds"),
    (lambda config: config["policy"].update({"no_backward": False}),
     "Ablation policy"),
])
def test_ablation_config_rejects_protocol_drift(tmp_path, mutation, message):
    config = json.loads(ABLATION_CONFIG.read_text())
    mutation(config)
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(config))
    with pytest.raises(learn.LearnabilityDiagnosticError, match=message):
        learn.load_config(path)


@pytest.mark.parametrize("mutation, message", [
    (lambda config: config["schedule"].update({"prefix_updates": 8}),
     "prefix"),
    (lambda config: config["gates"].update(
        {"answer_correct_accuracy": 0.1}), "gates"),
    (lambda config: config["policy"].update({"no_dev_access": False}),
     "policy"),
    (lambda config: config["limits"].update(
        {"max_total_backward_calls": 31}), "limits"),
    (lambda config: config["access"].update({"order": ["train", "dev"]}),
     "access"),
])
def test_gradient_config_rejects_protocol_drift(tmp_path, mutation, message):
    config = json.loads(GRADIENT_CONFIG.read_text())
    mutation(config)
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(config))
    with pytest.raises(learn.LearnabilityDiagnosticError, match=message):
        learn.load_config(path)


def test_config_rejects_a_gradient_field_in_ablation(tmp_path):
    config = json.loads(ABLATION_CONFIG.read_text())
    config["optimizer"] = {}
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(config))
    with pytest.raises(learn.LearnabilityDiagnosticError, match="fields differ"):
        learn.load_config(path)


@require_local_artifacts(
    "checkpoints/minicpm5_source",
    "checkpoints/siglip2_base_patch16_224_source",
    "data/vision_balanced_alignment_v2_20261026/prepared04",
    "runs/vision_balanced_alignment_v2_20261026/train04/alignment/extension",
    "runs/vision_collapse_diagnostic_v1_20261026_attempt02",
    "runs/vision_mean_only_ablation_v1_20261026_attempt09",
)
def test_static_pins_verify_and_historical_ablation_refuses_drift():
    module = _runner_module()
    ablation = learn.load_config(ABLATION_CONFIG)
    gradient = learn.load_config(GRADIENT_CONFIG)
    receipt = json.loads(ABLATION_RECEIPT.read_text())
    runner = "scripts/run_vision_learnability_diagnostic.py"
    historical_pin = receipt["pins"]["implementation"]["new_files"][runner]
    assert ablation["implementation_pins"]["new_files"][runner] == (
        historical_pin["pinned_sha256"])
    assert module._sha(ABLATION_CONFIG) == receipt["config_sha256"]
    assert gradient["precondition"]["config_sha256"] == receipt["config_sha256"]
    with pytest.raises(learn.LearnabilityDiagnosticError,
                       match="New implementation bytes differ"):
        module.verify_pins(ablation)
    gradient_pins = module.verify_pins(gradient)
    assert gradient_pins["pins"]["source"]["model_revision"] == (
        "f97400052a43d642bbc6e9975e2397e3ae6a6b52")
    assert gradient_pins["pins"]["schedule"]["source_sha256"] == (
        "08484a4dbc000488c94c54510fad655c06251598a87294837e24d80cad09433a")


@require_local_artifacts(
    "checkpoints/minicpm5_source",
    "checkpoints/siglip2_base_patch16_224_source",
    "data/vision_balanced_alignment_v2_20261026/prepared04",
    "runs/vision_balanced_alignment_v2_20261026/train04/alignment/extension",
    "runs/vision_collapse_diagnostic_v1_20261026_attempt02",
)
def test_new_file_pin_drift_is_rejected(tmp_path):
    module = _runner_module()
    config = learn.load_config(ABLATION_CONFIG)
    config["implementation_pins"]["new_files"][
        "src/replica_lab/vision_learnability_diagnostic.py"] = "0" * 64
    with pytest.raises(learn.LearnabilityDiagnosticError,
                       match="New implementation bytes differ"):
        module.verify_pins(config)
