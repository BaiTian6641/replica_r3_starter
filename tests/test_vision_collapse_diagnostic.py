"""CPU-only contracts for the training-only vision-collapse diagnostic.

These tests exercise the reusable token/logit/feature math, the declared stop
set, the pair/control plan over the real sealed training rows, the fit-contract
coverage audit and the declared stop conditions.  They never load a model, run
a native forward, execute an optimizer step or touch an evaluation split.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace
import time

import pytest

from replica_lab import vision_collapse_diagnostic as diagnostic
from conftest import require_local_artifacts

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_SNAPSHOT = (
    ROOT / "reports/vision_historical_source_20261027/source_snapshot.json")
CURRENT_RECEIPT = ROOT / "configs/vision_current_source_receipt_v1.json"
CONFIG = ROOT / "configs/vision_collapse_diagnostic_v1.json"
TRAINING = (ROOT / "data/vision_balanced_alignment_v2_20261026/prepared04/"
                   "training/alignment.jsonl")


def _rows():
    return [json.loads(line) for line in TRAINING.read_text().splitlines()
            if line.strip()]


def _uniform(width):
    return [0.0] * width


def _spiked(width, **spikes):
    row = _uniform(width)
    for token, value in spikes.items():
        row[int(token)] = float(value)
    return row


# ---------------------------------------------------------------------------
# Per-position logit math
# ---------------------------------------------------------------------------


def test_position_nll_uses_one_shifted_supervised_position_per_target():
    width = 13
    labels = [-100, -100, 10, 11]
    logits = [_uniform(width), _spiked(width, **{"10": 5.0}),
              _spiked(width, **{"11": 4.0}), _uniform(width)]
    result = diagnostic.position_nll(logits, labels, 2)
    first = result["positions"][0]
    second = result["positions"][1]
    expected_first = math.log(sum(math.exp(v) for v in logits[1])) - 5.0
    expected_second = math.log(sum(math.exp(v) for v in logits[2])) - 4.0
    assert first["target_position"] == 2 and first["predicted_from"] == 1
    assert second["target_position"] == 3 and second["predicted_from"] == 2
    assert first["nll"] == pytest.approx(expected_first)
    assert second["nll"] == pytest.approx(expected_second)
    assert result["label_shift"] == 1
    assert result["label_positions"] == [2, 3]
    assert result["predictor_positions"] == [1, 2]
    assert result["supervised_label_tokens"] == 2
    assert result["shifted_supervision_tokens"] == 2
    assert result["loss_tokens"] == 2
    assert result["prompt_masked_tokens"] == 2
    assert result["prompt_fully_masked"] is True
    assert result["first_token_nll"] == pytest.approx(expected_first)
    assert result["continuation_mean_nll"] == pytest.approx(expected_second)
    assert result["top1_matches"] == 2
    with pytest.raises(diagnostic.DiagnosticError):
        diagnostic.position_nll(logits, labels, 2, label_shift=2)
    with pytest.raises(diagnostic.DiagnosticError):
        diagnostic.position_nll(logits[:2], labels, 2)


def test_rank_topk_entropy_and_first_token_analysis():
    width = 6
    row = _spiked(width, **{"3": 2.0, "1": 3.0})
    assert diagnostic.logit_of(row, 3) == 2.0
    assert diagnostic.rank_of(row, 3) == 2
    tied = [0.0, 0.0, 1.0, 1.0]
    assert [item["token"] for item in diagnostic.top_k(tied, 3)] == [2, 3, 0]
    assert diagnostic.entropy([0.0, 0.0, 0.0, 0.0]) == pytest.approx(math.log(4))
    analysis = diagnostic.first_token_analysis(
        _spiked(width, **{"4": 5.0, "2": 3.0}), 4, opposite_token=2)
    assert analysis["gold_rank"] == 1
    assert analysis["opposite_rank"] == 2
    assert analysis["gap"] == pytest.approx(2.0)
    assert analysis["argmax_is_gold"] is True
    assert analysis["top_k"][0] == {"token": 4, "logit": 5.0}
    assert analysis["opposite_is_gold"] is False
    absent = diagnostic.first_token_analysis(_spiked(width, **{"4": 5.0}), 4)
    assert absent["opposite_token"] is None and absent["gap"] is None


def test_eos_pair_analysis_reports_both_stop_logits():
    row = _spiked(8, **{"1": 2.0, "3": 4.0})
    analysis = diagnostic.eos_pair_analysis(row, (1, 3))
    assert analysis["stop_tokens"] == [1, 3]
    assert analysis["stop_1_logit"] == 2.0
    assert analysis["stop_3_logit"] == 4.0
    assert analysis["stop_gap"] == pytest.approx(-2.0)
    assert analysis["preferred_stop_token"] == 3
    assert analysis["argmax_token"] == 3
    assert analysis["argmax_is_stop"] is True


# ---------------------------------------------------------------------------
# Declared stop set and generation evidence
# ---------------------------------------------------------------------------


def test_stop_set_uses_the_native_config_not_only_the_tokenizer_eos():
    assert diagnostic.EXPECTED_STOP_TOKENS == (1, 130073)
    stops, source = diagnostic.resolve_stop_tokens([1, 130073], 1)
    assert stops == (1, 130073) and source == "model_config"
    fallback, fallback_source = diagnostic.resolve_stop_tokens(None, 130073)
    assert fallback == (130073,) and fallback_source == "tokenizer_fallback"
    with pytest.raises(diagnostic.DiagnosticError):
        diagnostic.resolve_stop_tokens(None, None)
    with pytest.raises(diagnostic.DiagnosticError):
        diagnostic.resolve_stop_tokens([1, "x"], 1)


def test_stop_evidence_accepts_either_declared_stop_token():
    stops = diagnostic.EXPECTED_STOP_TOKENS
    im_end = diagnostic.stop_evidence([42, 130073], stops)
    assert im_end["ended_by_eos"] is True
    assert im_end["stop_token"] == 130073
    assert im_end["stop_reason"] == "stop_token"
    slash = diagnostic.stop_evidence([42, 1], stops)
    assert slash["ended_by_eos"] is True and slash["stop_token"] == 1
    budget = diagnostic.stop_evidence([42, 43], stops)
    assert budget["ended_by_eos"] is False
    assert budget["stop_token"] is None
    assert budget["stop_reason"] == "token_budget"
    late_loop = diagnostic.stop_evidence([1, 42], stops)
    assert late_loop["ended_by_eos"] is False
    assert late_loop["stop_token_before_final"] == 0
    with pytest.raises(diagnostic.DiagnosticError):
        diagnostic.stop_evidence([], stops)


def test_generation_evidence_records_prefill_agreement_and_budget():
    evidence = diagnostic.generation_evidence(
        [42, 130073], "answer", stops=diagnostic.EXPECTED_STOP_TOKENS,
        prefill_argmax=42, max_new_tokens=8)
    assert evidence["prefill_argmax_matches_first_token"] is True
    assert evidence["ended_by_eos"] is True
    assert evidence["hit_token_budget"] is False
    capped = diagnostic.generation_evidence(
        [7, 8], "capped", stops=diagnostic.EXPECTED_STOP_TOKENS,
        prefill_argmax=9, max_new_tokens=2)
    assert capped["prefill_argmax_matches_first_token"] is False
    assert capped["hit_token_budget"] is True
    with pytest.raises(diagnostic.DiagnosticError):
        diagnostic.generation_evidence([7], "x", stops=(),
                                       prefill_argmax=7, max_new_tokens=1)


# ---------------------------------------------------------------------------
# Prompt/label/EOS fit contract
# ---------------------------------------------------------------------------


def _fit_row(**overrides):
    payload = {
        "example_id": "answer:case-a",
        "case_id": "case-a",
        "kind": "answer",
        "prompt_tokens": 3,
        "input_ids": [10, 11, 12, 20, 21, 1],
        "labels": [-100, -100, -100, 20, 21, 1],
        "eos_token": 1,
        "stop_tokens": diagnostic.EXPECTED_STOP_TOKENS,
        "pad_token": 1,
        "question": "Is the circle red?",
        "caption_template": "Describe the image briefly.",
        "max_sequence_tokens": 288,
    }
    payload.update(overrides)
    return diagnostic.fit_contract_row_report(**payload)


def test_fit_contract_row_names_supervision_without_ambiguity():
    row = _fit_row()
    assert row["prompt_tokens"] == 3
    assert row["response_tokens"] == 3
    assert row["supervised_label_tokens"] == 3
    assert row["shifted_supervision_tokens"] == 3
    assert row["loss_tokens"] == 3
    assert row["label_shift"] == 1
    assert row["label_positions"] == [3, 5]
    assert row["predictor_positions"] == [2, 4]
    assert row["prompt_masked_tokens"] == 3
    assert row["response_eos_token"] == 1
    assert row["stop_tokens"] == [1, 130073]
    assert row["eos_in_stop_set"] is True
    assert row["pad_positions_before_eos"] == []
    assert row["contract_valid"] is True
    caption = _fit_row(example_id="caption:case-b", case_id="case-b",
                       kind="caption",
                       question="Describe the image briefly.")
    assert caption["template_checked"] is True
    assert caption["kind"] == "caption"


@pytest.mark.parametrize("overrides, message", [
    ({"labels": [5, -100, -100, 20, 21, 1]}, "Prompt labels"),
    ({"labels": [-100, -100, -100, 20, 21, -100]}, "masked or empty"),
    ({"labels": [-100, -100, -100, 20, 21, 22],
      "input_ids": [10, 11, 12, 20, 21, 22]}, "target EOS"),
    ({"input_ids": [10, 11, 12, 20, 21, 22],
      "labels": [-100, -100, -100, 20, 21, 22]}, "target EOS"),
    ({"eos_token": 9}, "target EOS"),
    ({"kind": "caption", "question": "Describe it."}, "Caption prompt differs"),
    ({"input_ids": [10, 11, 12, 20, 1, 1],
      "labels": [-100, -100, -100, 20, 1, 1]}, "pad token"),
    ({"max_sequence_tokens": 5}, "sequence budget"),
])
def test_fit_contract_rejects_each_contract_break(overrides, message):
    with pytest.raises(diagnostic.DiagnosticError, match=message):
        _fit_row(**overrides)


# ---------------------------------------------------------------------------
# Fit coverage over every row, captions included
# ---------------------------------------------------------------------------


def test_fit_coverage_checks_every_training_row_including_captions():
    rows = _rows()
    retained = []
    for row in rows:
        retained.append(diagnostic.fit_contract_row_report(
            example_id=row["example_id"], case_id=row["case_id"],
            kind=row["kind"], prompt_tokens=3,
            input_ids=[10, 11, 12, 20, 21, 1],
            labels=[-100, -100, -100, 20, 21, 1], eos_token=1,
            stop_tokens=diagnostic.EXPECTED_STOP_TOKENS, pad_token=1,
            question=row["question"],
            caption_template="Describe the image briefly.",
            max_sequence_tokens=288))
    expected = [row["example_id"] for row in rows]
    coverage = diagnostic.fit_coverage(retained, expected)
    assert coverage["expected_rows"] == coverage["checked_rows"] == 26
    assert coverage["retained_row_reports"] == 26
    assert coverage["per_kind"] == {"answer": 14, "caption": 12}
    assert coverage["all_rows_checked"] is True
    diagnostic.require_fit_coverage(
        coverage, expected_kinds={"answer": 14, "caption": 12})
    # A missing caption row fails coverage even though every answer was checked.
    caption_free = [row for row in retained if row["kind"] != "caption"
                    or row["example_id"] != rows[14]["example_id"]]
    missing = diagnostic.fit_coverage(caption_free, expected)
    assert missing["missing"] == [rows[14]["example_id"]]
    assert missing["all_rows_checked"] is False
    with pytest.raises(diagnostic.DiagnosticError, match="coverage is incomplete"):
        diagnostic.require_fit_coverage(
            missing, expected_kinds={"answer": 14, "caption": 12})
    # Repeated retention reports stay visible but do not duplicate coverage.
    repeated = diagnostic.fit_coverage(retained + retained[:2], expected)
    assert repeated["checked_rows"] == 26
    assert repeated["repeated_example_ids"] == sorted(
        [retained[0]["example_id"], retained[1]["example_id"]])
    assert repeated["all_rows_checked"] is True
    # Unexpected rows are reported, not silently accepted.
    stray = retained + [diagnostic.fit_contract_row_report(
        example_id="answer:stray", case_id="stray", kind="answer",
        prompt_tokens=3, input_ids=[10, 11, 12, 20, 21, 1],
        labels=[-100, -100, -100, 20, 21, 1], eos_token=1,
        stop_tokens=diagnostic.EXPECTED_STOP_TOKENS, pad_token=1,
        question="q", caption_template="Describe the image briefly.",
        max_sequence_tokens=288)]
    unexpected = diagnostic.fit_coverage(stray, expected)
    assert unexpected["unexpected"] == ["answer:stray"]
    assert unexpected["all_rows_checked"] is False


# ---------------------------------------------------------------------------
# Real pair layout and matched controls
# ---------------------------------------------------------------------------


def test_pair_layout_pins_five_answer_and_seven_caption_pairs():
    rows = _rows()
    layout = diagnostic.pair_layout(rows)
    assert layout["opposite_answer_pairs"] == 5
    assert layout["caption_sibling_pairs"] == 7
    assert layout["answer_pair_count"] == 9
    assert layout["caption_pair_count"] == 7
    assert layout["complete_answer_pairs"] == 5
    assert layout["complete_caption_pairs"] == 5
    assert layout["singleton_answer_pair_count"] == 4
    assert layout["singleton_answer_pairs"] == [
        "vba2-w037-p0", "vba2-w038-p0", "vba2-w039-p0", "vba2-w060-p0"]
    assert layout["incomplete_caption_pairs"] == [
        "vba2-w017-p0", "vba2-w018-p0"]
    groups = diagnostic.pair_groups(rows)
    assert sum(group["complete"] for group in groups["answer"]) == 5
    assert sum(group["complete"] for group in groups["caption"]) == 5


def test_control_plan_is_deterministic_and_target_text_blind():
    rows = _rows()
    plan = diagnostic.control_plan(rows)
    assert len(plan) == 26
    by_case = {row["case_id"]: row for row in rows}
    for case_id, entry in plan.items():
        images = [by_case[entry["correct_case_id"]]["image_sha256"]]
        if entry["sibling_available"]:
            images.append(by_case[entry["sibling_case_id"]]["image_sha256"])
        images.append(by_case[entry["shuffled_case_id"]]["image_sha256"])
        assert len(set(images)) == len(images)
        assert entry["shuffled_pair_id"] != entry["pair_id"]
        assert entry["noimage"] is None
    unavailable = sorted(case_id for case_id, entry in plan.items()
                         if not entry["sibling_available"])
    # Four singleton answer rows and two singleton caption rows have no
    # sibling row in the sealed training payload; the plan reports them as
    # unavailable instead of inventing an image.
    assert unavailable == [
        "vba2-w017-p0-b", "vba2-w018-p0-a", "vba2-w037-p0-b",
        "vba2-w038-p0-b", "vba2-w039-p0-b", "vba2-w060-p0-a"]
    mutated = copy.deepcopy(rows)
    for row in mutated:
        row["response"] = "ZZZ"
        row["question"] = "ZZZ"
    assert diagnostic.control_plan(mutated) == plan


def test_comparison_cells_keep_image_and_target_roles_distinct():
    rows = _rows()
    groups = diagnostic.pair_groups(rows)
    plan = diagnostic.control_plan(rows)
    answer = next(group for group in groups["answer"]
                  if group["pair_id"] == "vba2-w000-p0")
    cells = diagnostic.comparison_cells(answer, plan)
    assert len(cells) == 6
    assert {cell["image_role"] for cell in cells} == {
        "own", "changed_sibling", "cross_pair_shuffled"}
    assert {cell["target_role"] for cell in cells} == {"own", "opposite"}
    pairs = {(cell["image_role"], cell["target_role"]) for cell in cells}
    assert len(pairs) == 6
    incomplete = next(group for group in groups["caption"]
                      if group["pair_id"] == "vba2-w017-p0")
    partial = diagnostic.comparison_cells(incomplete, plan)
    assert len(partial) == 2
    assert {cell["target_role"] for cell in partial} == {"own"}
    assert {cell["image_role"] for cell in partial} == {
        "own", "cross_pair_shuffled"}


# ---------------------------------------------------------------------------
# Text and feature ceilings
# ---------------------------------------------------------------------------


def test_text_ceiling_reports_first_token_rank_and_accuracy():
    width = 4
    rows = []
    for index in range(14):
        if index < 10:
            logits = _spiked(width, **{"2": 5.0})
        else:
            logits = _spiked(width, **{"1": 5.0})
        rows.append({"case_id": f"case-{index}", "gold_token": 2,
                     "logits": logits})
    ceiling = diagnostic.first_token_ceiling(rows, expected_cases=14)
    assert ceiling["cases"] == 14
    assert ceiling["first_token_correct"] == 10
    assert ceiling["first_token_accuracy"] == pytest.approx(10 / 14)
    assert ceiling["mean_rank"] == pytest.approx((10 + 8) / 14)
    assert ceiling["min_rank"] == 1 and ceiling["max_rank"] == 2
    assert ceiling["median_rank"] == pytest.approx(1.0)
    summary = diagnostic.first_token_ceiling_summary(
        [{"case_id": row["case_id"], "gold_token": 2,
          "gold_rank": 1, "argmax_token": 2} for row in rows],
        expected_cases=14)
    assert summary["first_token_accuracy"] == 1.0
    with pytest.raises(diagnostic.DiagnosticError):
        diagnostic.first_token_ceiling(rows, expected_cases=13)


def test_feature_ceiling_is_leave_one_pair_out_and_deterministic():
    pairs = [{"pair_id": f"p{index}", "a_case_id": f"a{index}",
              "b_case_id": f"b{index}"} for index in range(5)]
    means = {}
    for index in range(5):
        means[f"a{index}"] = [-1.0, 0.1 * index, 0.0]
        means[f"b{index}"] = [1.0, 0.1 * index, 0.0]
    ceiling = diagnostic.feature_ceiling(pairs, means, expected_pairs=5)
    assert ceiling["pairs"] == 5
    assert ceiling["feature_width"] == 3
    assert ceiling["accuracy"] == 1.0
    assert ceiling["correct"] == 5
    for fold in ceiling["folds"]:
        assert fold["held_out_pair"] not in fold["training_pairs"]
        assert len(fold["training_pairs"]) == 4
        assert fold["gap"] > 0
    again = diagnostic.feature_ceiling(pairs, means, expected_pairs=5)
    assert again == ceiling
    with pytest.raises(diagnostic.DiagnosticError):
        diagnostic.feature_ceiling(pairs, means, expected_pairs=4)
    degenerate = {key: [0.0, 0.0, 0.0] for key in means}
    flat = diagnostic.feature_ceiling(pairs, degenerate, expected_pairs=5)
    assert flat["correct"] == 0 and flat["accuracy"] == 0.0


def test_pca_content_ratio_recovers_rank_one_structure():
    rank_one = [[float(index), 0.0, 0.0] for index in range(5)]
    result = diagnostic.pca_content_ratio(rank_one)
    assert result["ratio"] == pytest.approx(1.0, abs=1e-6)
    assert result["top_singular_squared"] <= result[
        "total_singular_squared"] + 1e-9
    assert result["converged"] is True
    assert result["budget_exhausted"] is False
    assert result["delta"] <= result["tolerance"]
    assert result == diagnostic.pca_content_ratio(rank_one)
    identical = [[1.0, 2.0, 3.0]] * 4
    flat = diagnostic.pca_content_ratio(identical)
    assert flat["ratio"] is None and flat["total_singular_squared"] == 0.0
    assert flat["converged"] is False and flat["budget_exhausted"] is False


def test_pca_distinguishes_tolerance_convergence_from_budget_exhaustion():
    rank_one = [[float(index), 0.0, 0.0] for index in range(5)]
    exhausted = diagnostic.pca_content_ratio(rank_one, iterations=2)
    assert exhausted["converged"] is False
    assert exhausted["budget_exhausted"] is True
    assert exhausted["iterations"] == 2
    assert exhausted["delta"] > exhausted["tolerance"]
    assert exhausted["ratio"] == pytest.approx(
        exhausted["top_singular_squared"]
        / exhausted["total_singular_squared"])
    converged = diagnostic.pca_content_ratio(rank_one, iterations=3)
    assert converged["converged"] is True
    assert converged["budget_exhausted"] is False
    assert converged["iterations"] == 3
    assert converged["delta"] <= converged["tolerance"]
    assert converged["ratio"] == pytest.approx(exhausted["ratio"])
    one_iteration = diagnostic.pca_content_ratio(rank_one, iterations=1)
    assert one_iteration["budget_exhausted"] is True
    assert one_iteration["converged"] is False
    assert one_iteration["delta"] is None


def test_feature_summary_records_fully_collapsed_projections():
    plan = {
        "c0": {"sibling_available": True, "sibling_case_id": "c1",
               "shuffled_case_id": "c1"},
        "c1": {"sibling_available": True, "sibling_case_id": "c0",
               "shuffled_case_id": "c0"},
    }
    vectors = {"c0": [0.0, 0.0, 0.0], "c1": [0.0, 0.0, 0.0]}
    image_stats = {
        case_id: {
            "image_case_id": case_id,
            "image_sha256": (case_id * 64)[:64],
            "tokens": 196,
            "projected_rms": 0.0,
            "projected_max": 0.0,
            "mean_vector_rms": 0.0,
            "spatial_variance": 0.0,
            "mean_patch": vector,
        }
        for case_id, vector in vectors.items()
    }
    summary = diagnostic.feature_summary(
        "fresh_init_seed29", image_stats, vectors, plan, [2.0, 2.0])
    assert summary["projected_rms_mean"] == 0.0
    assert summary["rms_projected_mean_over_projected"] is None
    assert summary["undefined_projected_ratios"] == 2
    assert summary["sibling_cosine_distance"]["count"] == 0
    assert summary["skipped_zero_norm_distances"] == 4
    assert summary["pca_content"]["ratio"] is None
    assert summary["projected_mean_over_text_rms"] == 0.0


def test_feature_summary_reports_scale_separation_and_pca():
    plan = {
        "c0": {"sibling_available": True, "sibling_case_id": "c1",
               "shuffled_case_id": "c2"},
        "c1": {"sibling_available": True, "sibling_case_id": "c0",
               "shuffled_case_id": "c3"},
        "c2": {"sibling_available": False, "sibling_case_id": None,
               "shuffled_case_id": "c0"},
        "c3": {"sibling_available": False, "sibling_case_id": None,
               "shuffled_case_id": "c1"},
    }
    vectors = {"c0": [1.0, 0.0, 0.0], "c1": [0.0, 1.0, 0.0],
               "c2": [0.5, 0.5, 0.0], "c3": [0.2, 0.0, 0.8]}
    image_stats = {
        case_id: {
            "image_case_id": case_id,
            "image_sha256": (case_id * 64)[:64],
            "tokens": 196,
            "projected_rms": 2.0,
            "projected_max": 5.0,
            "mean_vector_rms": 1.0,
            "spatial_variance": 0.25,
            "mean_patch": vector,
        }
        for case_id, vector in vectors.items()
    }
    summary = diagnostic.feature_summary(
        "stage", image_stats, vectors, plan, [3.0, 3.0, 3.0, 3.0])
    assert summary["images"] == 4
    assert summary["projected_rms_mean"] == pytest.approx(2.0)
    assert summary["projected_mean_vector_rms_mean"] == pytest.approx(1.0)
    assert summary["rms_projected_mean_over_projected"] == pytest.approx(0.5)
    assert summary["within_image_spatial_variance_mean"] == pytest.approx(0.25)
    assert summary["sibling_cosine_distance"]["count"] == 2
    assert summary["cross_pair_cosine_distance"]["count"] == 4
    assert 0.0 <= summary["pca_content"]["ratio"] <= 1.0
    assert summary["text_embedding_rms"]["mean"] == pytest.approx(3.0)
    assert summary["projected_mean_over_text_rms"] == pytest.approx(1 / 3)
    assert diagnostic.variance_across([[1.0, 2.0], [3.0, 4.0]]) == pytest.approx(1.0)
    assert diagnostic.cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Declared stop conditions and static pins
# ---------------------------------------------------------------------------


def _stop_report(**overrides):
    payload = {
        "wall_seconds": 100.0, "wall_limit_seconds": 1200.0,
        "artifact_bytes": 1024, "artifact_limit_bytes": 134217728,
        "updates": 0, "optimizer_steps": 0, "gradients_computed": 0,
        "checkpoint_writes": 0, "extension_writes": 0, "model_loads": 1,
        "extension_loads": 2, "downloads": 0, "cpu_fallback": False,
        "sealed_evaluation_access": 0, "guard_stop_reason": None,
    }
    payload.update(overrides)
    return diagnostic.stop_conditions_report(**payload)


def test_stop_conditions_report_declares_and_enforces_resource_policy():
    report = _stop_report()
    assert report["violations"] == []
    assert report["within_declared_limits"] is True
    assert report["no_training_updates"] and report["no_optimizer_steps"]
    assert report["no_gradients"] and report["no_extension_writes"]
    assert report["single_model_load"] and report["no_cpu_fallback"]
    diagnostic.require_stop_conditions(report)
    over_budget = _stop_report(artifact_bytes=134217729)
    assert over_budget["violations"] == ["artifact_bytes_exceeded"]
    assert over_budget["within_declared_limits"] is False
    with pytest.raises(diagnostic.DiagnosticError, match="stop condition"):
        diagnostic.require_stop_conditions(over_budget)
    updated = _stop_report(updates=1, optimizer_steps=1,
                           gradients_computed=1)
    assert updated["violations"] == []
    with pytest.raises(diagnostic.DiagnosticError, match="no_training_updates"):
        diagnostic.require_stop_conditions(updated)
    slow = _stop_report(wall_seconds=1201.0)
    assert slow["violations"] == ["wall_seconds_exceeded"]
    with pytest.raises(diagnostic.DiagnosticError, match="stop condition"):
        diagnostic.require_stop_conditions(slow)


def test_constant_patch_stats_are_exact_and_spatially_flat():
    import torch
    module = _runner_module()
    evaluator = object.__new__(module.StageEvaluator)
    constant = torch.tensor([[1.25, -2.5], [1.25, -2.5]], dtype=torch.bfloat16)
    visual = SimpleNamespace(
        features=constant,
        valid=torch.ones(2, dtype=torch.bool),
        image_sha256="a" * 64,
    )
    evaluator.adapter = SimpleNamespace(project_visual=lambda value: value.features)
    evaluator.visuals = {("case", "case"): visual}
    evaluator.image_stats = {}
    evaluator._torch = torch
    stats = evaluator._projected_stats("case")
    assert stats["mean_patch"] == [1.25, -2.5]
    assert stats["spatial_variance"] == 0.0
    assert stats["mean_vector_rms"] == pytest.approx(math.sqrt(3.90625), rel=1e-4)


def test_claims_block_is_first_order_only():
    claims = diagnostic.claims_block()
    assert claims["first_order_evidence_only"] is True
    assert claims["promotion"] is False
    assert claims["learned_quality"] is False
    assert claims["perception_quality"] is False
    assert claims["semantic_quality"] is False


def _runner_module():
    spec = importlib.util.spec_from_file_location(
        "vision_collapse_runner",
        ROOT / "scripts/run_vision_collapse_diagnostic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@require_local_artifacts(
    "checkpoints/minicpm5_source",
    "checkpoints/siglip2_base_patch16_224_source",
    "data/vision_balanced_alignment_v2_20261026/prepared04",
    "runs/vision_balanced_alignment_v2_20261026/train04/alignment/extension",
)
def test_config_and_static_pins_verify_without_a_model():
    module = _runner_module()
    config = module.load_config(CONFIG)
    assert config["limits"]["updates"] == 0
    assert config["policy"]["sealed_evaluation_access"] == 0
    pins = module.verify_pins(config, source_root=HISTORICAL_SNAPSHOT,
                              mode="historical")
    assert pins["pins"]["implementation"]["source_mode"] == (
        "historical_snapshot")
    assert pins["pins"]["implementation"]["source_snapshot_sha256"] == (
        module._sha(HISTORICAL_SNAPSHOT))
    assert pins["pins"]["prepared"]["training_examples"] == 26
    assert pins["pins"]["prepared"]["answer_examples"] == 14
    assert pins["pins"]["prepared"]["caption_examples"] == 12
    assert pins["pins"]["extension"]["base_saved"] is False
    assert pins["pins"]["extension"]["weights_bytes"] == 65688472
    assert pins["pins"]["implementation"]["enforced"] is True
    assert pins["pins"]["implementation"]["mode"] == "explicit_read_only_migration"
    files = pins["pins"]["implementation"]["files"]
    migrated = {
        "scripts/run_vision_curriculum.py",
        "scripts/run_vision_balanced_alignment.py",
        "src/replica_lab/architecture/contracts.py",
        "src/replica_lab/architecture/minicpm5.py",
        "src/replica_lab/memory_system/session.py"}
    assert migrated <= set(files)
    for relative, entry in files.items():
        assert len(entry["pinned_sha256"]) == len(entry["actual_sha256"]) == 64
        assert entry["enforced"] is True
        assert entry["migrated"] is (relative in migrated)
        assert entry["matches_prepared_pin"] is (relative not in migrated)
        if relative in migrated:
            migration = entry["migration"]
            assert migration["parent_sha256"] == entry["pinned_sha256"]
            assert migration["current_sha256"] == entry["actual_sha256"]
            assert migration["reason"]
        else:
            assert entry["migration"] is None
    rows, layout = module.load_training_rows(config, pins)
    assert len(rows) == 26
    assert layout["opposite_answer_pairs"] == 5
    assert layout["caption_sibling_pairs"] == 7
    assert all(Path(row["image_path"]).is_file() for row in rows)


@require_local_artifacts(
    "checkpoints/minicpm5_source",
    "checkpoints/siglip2_base_patch16_224_source",
    "data/vision_balanced_alignment_v2_20261026/prepared04",
    "runs/vision_balanced_alignment_v2_20261026/train04/alignment/extension",
)
def test_explicit_migration_rejects_wrong_current_hash():
    module = _runner_module()
    config = module.load_config(CONFIG)
    migration = config["implementation_pins"]["migrations"][
        "scripts/run_vision_curriculum.py"]
    migration["current_sha256"] = "0" * 64
    with pytest.raises(diagnostic.DiagnosticError,
                       match="Migrated implementation bytes differ"):
        module.verify_pins(config, source_root=HISTORICAL_SNAPSHOT)


@require_local_artifacts(
    "checkpoints/minicpm5_source",
    "checkpoints/siglip2_base_patch16_224_source",
    "data/vision_balanced_alignment_v2_20261026/prepared04",
    "runs/vision_balanced_alignment_v2_20261026/train04/alignment/extension",
)
def test_current_receipt_verifies_live_bytes_and_fails_closed(tmp_path):
    module = _runner_module()
    config = module.load_config(CONFIG)
    pins = module.verify_pins(config, mode="current",
                              current_receipt=CURRENT_RECEIPT)
    implementation = pins["pins"]["implementation"]
    assert implementation["source_mode"] == "current_receipt"
    assert implementation["current_receipt_sha256"] == (
        module._sha(CURRENT_RECEIPT))
    assert set(implementation["new_files"]) == set()
    with pytest.raises(diagnostic.DiagnosticError,
                       match="reviewed current-source receipt"):
        module.verify_pins(config, mode="current")

    receipt = json.loads(CURRENT_RECEIPT.read_text())
    relative = sorted(receipt["new_files"])[0]
    receipt["new_files"][relative]["pinned_sha256"] = "0" * 64
    tampered = tmp_path / "tampered_receipt.json"
    tampered.write_text(json.dumps(receipt))
    with pytest.raises(diagnostic.DiagnosticError,
                       match="receipt digest differs"):
        module.verify_pins(config, mode="current",
                           current_receipt=tampered)

    receipt = json.loads(CURRENT_RECEIPT.read_text())
    receipt["new_files"][relative]["pinned_sha256"] = "0" * 64
    receipt["receipt_digest"] = module.provenance.receipt_digest(receipt)
    tampered.write_text(json.dumps(receipt))
    with pytest.raises(diagnostic.DiagnosticError,
                       match="Current implementation bytes differ"):
        module.verify_pins(config, mode="current",
                           current_receipt=tampered)

    receipt = json.loads(CURRENT_RECEIPT.read_text())
    stale = sorted(receipt["migrations"])[0]
    receipt["migrations"][stale]["current_sha256"] = "1" * 64
    receipt["receipt_digest"] = module.provenance.receipt_digest(receipt)
    tampered.write_text(json.dumps(receipt))
    with pytest.raises(diagnostic.DiagnosticError,
                       match="current pin is stale"):
        module.verify_pins(config, mode="current",
                           current_receipt=tampered)

    unrelated = copy.deepcopy(config)
    unrelated["name"] = "vision_collapse_diagnostic_unreviewed"
    with pytest.raises(diagnostic.DiagnosticError,
                       match="not reviewed for config"):
        module.verify_pins(unrelated, mode="current",
                           current_receipt=CURRENT_RECEIPT)


@require_local_artifacts(
    "checkpoints/minicpm5_source",
    "checkpoints/siglip2_base_patch16_224_source",
    "data/vision_balanced_alignment_v2_20261026/prepared04",
    "runs/vision_balanced_alignment_v2_20261026/train04/alignment/extension",
)
def test_historical_snapshot_tampering_fails_closed(tmp_path):
    module = _runner_module()
    config = module.load_config(CONFIG)
    source = ROOT / "reports/vision_historical_source_20261027"

    def copied(name):
        target = tmp_path / name
        shutil.copytree(source, target)
        return target

    target = copied("size")
    manifest = json.loads((target / "source_snapshot.json").read_text())
    manifest["entries"][0]["bytes"] += 1
    (target / "source_snapshot.json").write_text(json.dumps(manifest))
    with pytest.raises(diagnostic.DiagnosticError,
                       match="byte size differs"):
        module.verify_pins(config, source_root=target)

    target = copied("path")
    manifest = json.loads((target / "source_snapshot.json").read_text())
    manifest["entries"][0]["path"] = "../escape.py"
    (target / "source_snapshot.json").write_text(json.dumps(manifest))
    with pytest.raises(diagnostic.DiagnosticError, match="unsafe"):
        module.verify_pins(config, source_root=target)

    target = copied("missing")
    manifest = json.loads((target / "source_snapshot.json").read_text())
    (target / manifest["entries"][0]["blob"]).unlink()
    with pytest.raises(diagnostic.DiagnosticError, match="blob is missing"):
        module.verify_pins(config, source_root=target)

    target = copied("blob")
    manifest = json.loads((target / "source_snapshot.json").read_text())
    blob = target / manifest["entries"][0]["blob"]
    value = bytearray(blob.read_bytes())
    value[0] ^= 0xFF
    blob.write_bytes(bytes(value))
    with pytest.raises(diagnostic.DiagnosticError,
                       match="snapshot bytes differ"):
        module.verify_pins(config, source_root=target)


def test_source_provenance_files_are_self_contained_and_reviewed():
    module = _runner_module()
    snapshot = module.provenance.load_source_snapshot(HISTORICAL_SNAPSHOT)
    historical = snapshot["by_path"][
        "src/replica_lab/architecture/contracts.py"]
    assert len(historical) >= 1
    assert "d882f03cb6f7b0c52dcf39b83b296d8602e34e02bfa95741b2f97b7b654d67a9" \
        in historical
    assert module._sha(ROOT / "src/replica_lab/architecture/contracts.py") != \
        "d882f03cb6f7b0c52dcf39b83b296d8602e34e02bfa95741b2f97b7b654d67a9"
    receipt = module.provenance.load_current_receipt(CURRENT_RECEIPT)
    assert receipt["prepared"]["protocol_sha256"] == (
        "3bf636f00dce4e73168eb5f4b2c2454560276b32dfb1122b91d58ffcd44c2d77")
    assert receipt["configs"]["vision_collapse_diagnostic_v1"] == (
        module.provenance.canonical_digest(module.load_config(CONFIG)))
    for relative, entry in receipt["new_files"].items():
        assert module._sha(ROOT / relative) == entry["pinned_sha256"]


def test_collapse_cli_forwards_source_provenance_flags():
    module = _runner_module()
    args = module.build_parser().parse_args([
        "--config", "configs/vision_collapse_diagnostic_v1.json",
        "--output", "runs/example", "--model-path", "checkpoints/minicpm5_source",
        "--model-lock", "configs/minicpm5_sealed_lock.local.json",
        "--mode", "current", "--current-receipt", "configs/receipt.json",
        "--source-root", "reports/snapshot.json", "--verify-only",
    ])
    assert args.mode == "current"
    assert args.verify_only is True
    assert args.current_receipt == Path("configs/receipt.json")
    assert args.source_root == Path("reports/snapshot.json")


def test_config_rejects_eos_or_write_policy_drift(tmp_path):
    module = _runner_module()
    config = json.loads(CONFIG.read_text())
    drifted = copy.deepcopy(config)
    drifted["stop_tokens"] = [1]
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(drifted))
    with pytest.raises(diagnostic.DiagnosticError, match="stop tokens"):
        module.load_config(path)
    writeful = copy.deepcopy(config)
    writeful["limits"]["updates"] = 1
    path.write_text(json.dumps(writeful))
    with pytest.raises(diagnostic.DiagnosticError, match="forbid"):
        module.load_config(path)


def test_training_rows_reject_sealed_evaluation_paths():
    module = _runner_module()
    config = module.load_config(CONFIG)
    rows = _rows()
    prepared = ROOT / config["prepared"]["root"]
    mutated = copy.deepcopy(rows)
    mutated[0]["image"] = "evaluation/dev_answer_cases.jsonl"
    with pytest.raises(diagnostic.DiagnosticError, match="outside the training"):
        module.validate_rows(mutated, config, prepared)
    mutated = copy.deepcopy(rows)
    mutated[0]["image_sha256"] = "0" * 64
    with pytest.raises(diagnostic.DiagnosticError, match="image bytes"):
        module.validate_rows(mutated, config, prepared)
    mutated = copy.deepcopy(rows)
    mutated[0]["response"] = ""
    with pytest.raises(diagnostic.DiagnosticError, match="response is missing"):
        module.validate_rows(mutated, config, prepared)


# ---------------------------------------------------------------------------
# Reporting corrections from the independent review
# ---------------------------------------------------------------------------


def _summary_row(control, index, *, available=True, rank=1, nll=1.0,
                 ids=(1, 2), text="a"):
    record = {"control": control, "case_id": f"{control}-{index}",
              "available": available}
    if available:
        record["first_token"] = {"gold_rank": rank}
        record["positions"] = {"mean_nll": nll}
        record["generation"] = {"generated_ids": list(ids),
                                "generated_text": text,
                                "ended_by_eos": True}
    return record


def test_control_summary_publishes_declared_and_scored_denominators():
    records = []
    for index in range(20):
        records.append(_summary_row(
            "sibling", index, rank=1 if index < 3 else 2,
            nll=float(index + 1), ids=(index % 3,), text=f"t{index % 3}"))
    for index in range(20, 26):
        records.append(_summary_row("sibling", index, available=False))
    for index in range(26):
        records.append(_summary_row(
            "correct", index, rank=2, nll=2.0, ids=(7,), text="c"))
    summary = diagnostic.control_summary(records)
    sibling = summary["sibling"]
    assert "rows" not in sibling
    assert sibling["declared_rows"] == 26
    assert sibling["scored_rows"] == 20
    assert sibling["first_token_rows"] == 20
    assert sibling["nll_rows"] == 20
    assert sibling["generation_rows"] == 20
    assert sibling["first_token_gold_accuracy"] == pytest.approx(3 / 20)
    assert sibling["first_token_mean_gold_rank"] == pytest.approx(
        (3 + 17 * 2) / 20)
    assert sibling["response_mean_nll"] == pytest.approx(10.5)
    assert sibling["unique_generated_ids"] == 3
    assert sibling["unique_generated_texts"] == 3
    assert sibling["generation_ended_by_stop"] == 20
    assert "scored_rows" in sibling["metric_scope"]
    correct = summary["correct"]
    assert correct["declared_rows"] == 26
    assert correct["scored_rows"] == 26
    assert correct["generation_rows"] == 26
    assert correct["unique_generated_ids"] == 1


def test_control_summary_requires_explicit_unavailable_placeholders():
    rows = [
        _summary_row("sibling", 0, available=False),
        {"control": "sibling", "case_id": "unmarked"},
    ]
    with pytest.raises(diagnostic.DiagnosticError,
                       match="available:false"):
        diagnostic.control_summary(rows)


def test_failure_receipt_counts_only_completed_loads(tmp_path):
    module = _runner_module()
    accounting = module.LoadAccounting()
    limits = {"max_seconds": 1200.0, "artifact_bytes": 134217728}

    def boom():
        raise RuntimeError("injected load failure")

    def receipt():
        return module._stop_report(
            started=time.monotonic(), limits=limits, guard=None,
            accounting=accounting, artifact_bytes=0)

    with pytest.raises(RuntimeError, match="injected"):
        module._accounted_load(boom, accounting, "model")
    assert accounting.counts() == {"model_loads": 0, "extension_loads": 0}
    before_model = receipt()
    assert (before_model["model_loads"],
            before_model["extension_loads"]) == (0, 0)
    assert before_model["single_model_load"] is False

    module._accounted_load(
        lambda: ("adapter", "tokenizer", "lock"), accounting, "model")
    with pytest.raises(RuntimeError, match="injected"):
        module._accounted_load(boom, accounting, "extension")
    assert accounting.counts() == {"model_loads": 1, "extension_loads": 0}
    failed_first_extension = receipt()
    assert (failed_first_extension["model_loads"],
            failed_first_extension["extension_loads"]) == (1, 0)

    module._accounted_load(lambda: None, accounting, "extension")
    stage_a_failure = receipt()
    assert (stage_a_failure["model_loads"],
            stage_a_failure["extension_loads"]) == (1, 1)
    assert stage_a_failure["extension_loads"] != 2

    module._accounted_load(lambda: None, accounting, "extension")
    success = receipt()
    assert (success["model_loads"], success["extension_loads"]) == (1, 2)
    assert success["single_model_load"] is True
    with pytest.raises(diagnostic.DiagnosticError, match="Unknown load kind"):
        module._accounted_load(lambda: None, accounting, "optimizer")


def test_artifact_manifest_reports_listed_and_final_byte_totals(tmp_path):
    module = _runner_module()
    (tmp_path / "a.bin").write_bytes(b"a" * 100)
    (tmp_path / "report.json").write_text("{\"ok\": true}\n")
    (tmp_path / "manifest.json").write_text("stale")
    listed = sum(path.stat().st_size for path in tmp_path.iterdir()
                 if path.name != "manifest.json")
    manifest, payload = module._artifact_manifest(
        tmp_path, limit_bytes=4096,
        generated_at_utc="2026-10-26T00:00:00+00:00")
    assert manifest["total_bytes"] == listed
    assert manifest["listed_files"] == 2
    assert manifest["files_exclude"] == "manifest.json"
    assert manifest["manifest_bytes"] == len(payload.encode())
    assert manifest["final_artifact_bytes"] == (
        listed + manifest["manifest_bytes"])
    assert "including itself" not in manifest["scope"].lower()
    assert "excluding manifest.json" in manifest["scope"]
    (tmp_path / "manifest.json").write_text(payload)
    final = sum(path.stat().st_size for path in tmp_path.iterdir())
    assert final == manifest["final_artifact_bytes"]


def test_artifact_scope_wording_names_both_byte_totals():
    module = _runner_module()
    artifact_scope = module.ARTIFACT_SCOPE
    manifest_scope = module.MANIFEST_SCOPE
    for scope in (artifact_scope, manifest_scope):
        assert "including itself" not in scope.lower()
        assert "excluding manifest.json" in scope
        assert "final_artifact_bytes" in scope
    assert "artifact_bytes_estimate_including_report" in artifact_scope
    assert "artifact_bytes" in artifact_scope
