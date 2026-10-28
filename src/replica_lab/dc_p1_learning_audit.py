"""Real tiny-model DC-P1 learning-hypothesis and source-free audit."""
from __future__ import annotations

from pathlib import Path
import tempfile
import sys
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from dc_protocol_common_20261027 import read_jsonl
from replica_lab.dc_p1_source_free_runner import (
    behavior_delta, measure_behavior, run_fit_diagnostic, source_free_train_step)
from replica_lab.dc_p1_tiny_learning import make_tiny_learning_view
from replica_lab.dc_response_objective import (
    CLAIM_GLOBAL_RESCALE, answer_span_objective, build_response_masks)

PREPARED = Path(__file__).resolve().parents[2] / "data/dc_p1_native_response_v2_20261027"


def global_ce_gradient_probe() -> dict[str, Any]:
    """Preserve the valid coincident-mask/global-rescale finding only."""
    base = torch.tensor([[0.2, -0.4, 0.6, 0.1], [0.3, 0.2, -0.5, 0.4]])
    perturbation = torch.tensor([1.0, -2.0, 0.5, 3.0])
    gradients = {}
    for weight in (0.0, 0.5):
        scale = torch.tensor(0.0, requires_grad=True)
        masks = build_response_masks(
            input_ids=[7, 0, 1], response_ids=[0, 1], answer_span=(0, 0), eos_id=1)
        objective = answer_span_objective(
            base + scale * perturbation,
            torch.tensor([0, 1], dtype=torch.long), masks,
            answer_span_ce_weight=weight)
        objective["loss"].backward()
        gradients[weight] = scale.grad.detach().clone()
    ratio = gradients[0.5] / gradients[0.0]
    return {
        "claim": CLAIM_GLOBAL_RESCALE,
        "gradient_ratio": ratio.item(),
        "matched_gradient": gradients[0.0].item(),
        "treatment_gradient": gradients[0.5].item(),
        "masks_coincide": True,
    }


def _teaching_and_cases() -> tuple[list[dict], list[dict]]:
    from preflight_dc_p1_source_free_feature_alignment_v1 import (
        CONFIG_PATH, validate_prepared)
    from dc_protocol_common_20261027 import read_json
    validate_prepared(PREPARED, read_json(CONFIG_PATH))
    teaching = [row for row in read_jsonl(PREPARED / "teaching.jsonl")
                if row["source_group"] in {"p1v2src_dev_00", "p1v2src_dev_01"}]
    labels = {row["case_id"]: row for row in read_jsonl(PREPARED / "sealed_labels_fit.jsonl")}
    cases = []
    for row in read_jsonl(PREPARED / "fit_cases.jsonl"):
        if row["source_group"] in {"p1v2src_dev_00", "p1v2src_dev_01"}:
            cases.append({**row, "target": labels[row["case_id"]]["target"]})
    return teaching, cases


def _feature_shift(view, row: Mapping[str, Any]) -> dict[str, float]:
    source_ids = view._render(f"{row['source']}\n{row['prompt']}")
    query_ids = view._render(str(row["prompt"]))
    source_hidden = view._native_forward(source_ids)[1]
    query_hidden = view._native_forward(query_ids)[1]
    source_feature = view._features(source_hidden, len(source_ids))
    query_feature = view._features(query_hidden, len(query_ids))
    return {
        "source_free_feature_shift_l2": float((source_feature - query_feature).norm()),
        "source_conditioned_feature_norm": float(source_feature.norm()),
        "source_free_feature_norm": float(query_feature.norm()),
        "source_conditioned_prompt_tokens": len(source_ids),
        "source_free_prompt_tokens": len(query_ids),
    }


def _optimizer_checks(view, row, cases) -> dict:
    before = measure_behavior(view, cases[:1], arm="optimizer_probe")
    result = source_free_train_step(
        view, row, source_free_weight=0.25,
        gradient_amplification=100_000_000)
    after = measure_behavior(view, cases[:1], arm="optimizer_probe")
    treatment = {
        **result,
        "behavior": behavior_delta(before, after),
    }
    large_view = make_tiny_learning_view(lr=0.01, epsilon=0.01)
    large_before = measure_behavior(large_view, cases[:1], arm="large_epsilon")
    large_result = source_free_train_step(
        large_view, row, source_free_weight=0.25)
    large_after = measure_behavior(large_view, cases[:1], arm="large_epsilon")
    large = {
        **large_result,
        "behavior": behavior_delta(large_before, large_after),
    }
    return {"treatment": treatment, "large_epsilon": large}


def run_cpu_audit(output: str | Path | None = None) -> dict[str, Any]:
    teaching, cases = _teaching_and_cases()
    if output is None:
        with tempfile.TemporaryDirectory() as directory:
            return run_cpu_audit(Path(directory) / "fit")
    output = Path(output)
    view = make_tiny_learning_view(
        phrases=[row["target"] for row in teaching] + ["UNKNOWN", "wrong"],
        texts=[row["source"] + "\n" + row["prompt"] for row in teaching])
    feature = _feature_shift(view, teaching[0])
    state = run_fit_diagnostic(view, teaching, cases, output, prepared=PREPARED)
    checks = _optimizer_checks(view, teaching[0], cases)
    treatment_change = state["behavior"]["treatment"]
    justified = bool(
        global_ce_gradient_probe()["gradient_ratio"] == 1.5
        and feature["source_free_feature_shift_l2"] > 0.0
        and checks["treatment"]["raw_gradient_norm"] > 0.0
        and (abs(treatment_change["mean_target_probability_delta"]) > 0.0
             or abs(treatment_change["mean_target_margin_delta"]) > 0.0))
    return {
        "format": "replica.dc_p1_learning_audit.v2",
        "status": "diagnostic_only_no_native_learning_claim",
        "gradient": global_ce_gradient_probe(),
        "feature": feature,
        "optimizer_checks": checks,
        "arms": state["behavior"],
        "fit_counters": state["counters"],
        "artifact": state["artifact"],
        "fit_status": state["status"],
        "fit_is_learnability_only": True,
        "alternative_justified": justified,
        "alternative_protocol": (
            "dc_p1_source_free_feature_alignment_v2" if justified else None),
        "native_execution": False,
        "sealed_minicpm_loaded": False,
        "tiny_model_fixture": True,
        "development_or_final_access": False,
        "generalization_qualified": False,
        "final_labels_read": False,
        "output": str(output),
    }
