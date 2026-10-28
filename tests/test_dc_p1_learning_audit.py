"""Real tiny-Llama learning audit invariants and control evidence."""
from __future__ import annotations

import pytest

from replica_lab.dc_p1_learning_audit import run_cpu_audit


def test_real_learning_audit_measures_objective_optimizer_features_and_controls():
    result = run_cpu_audit()
    assert result["gradient"]["claim"] == "global_ce_rescale"
    assert result["gradient"]["gradient_ratio"] == pytest.approx(1.5)
    assert result["gradient"]["masks_coincide"] is True
    assert result["feature"]["source_free_feature_shift_l2"] > 0.0

    treatment = result["optimizer_checks"]["treatment"]
    assert treatment["optimizer_class"] == "NativeFusedAdamW"
    assert treatment["gradient_ownership"] == {
        "gate_up_down": True, "frozen_backbone": True, "finite": True}
    assert treatment["raw_gradient_norm"] > 1.0
    assert 0.0 < treatment["clip_scale"] < 1.0
    assert treatment["clipped_gradient_norm"] == pytest.approx(1.0)
    assert treatment["physical_work"]["training_forwards"] == 2
    assert treatment["physical_work"]["source_free_query_forwards"] == 1
    assert treatment["behavior"]["generated_tokens"] > 0
    assert "mean_target_probability_delta" in treatment["behavior"]
    large = result["optimizer_checks"]["large_epsilon"]
    assert large["optimizer_class"] == "NativeFusedAdamW"
    assert large["epsilon"] == 0.01
    assert large["behavior"]["generated_tokens"] > 0

    counters = result["fit_counters"]
    assert counters["training_forwards"] == 36
    assert counters["source_free_query_forwards"] == 18
    assert counters["backward_calls"] == 18
    assert counters["optimizer_steps"] == 18
    assert counters["checkpoint_writes"] == 19
    assert counters["artifact_checkpoint_writes"] == 1
    assert counters["generation_requests"] == 48
    assert counters["baseline_generation_requests"] == 24
    assert counters["forward_calls"] == 228
    assert counters["processed_text_positions"] > 0
    assert result["artifact"]["includes_temporary_files"] is True

    assert set(result["arms"]) == {"treatment", "matched_trained", "no_update", "reset_P"}
    for arm in ("treatment", "matched_trained"):
        assert result["arms"][arm]["generated_tokens"] > 0
        assert result["arms"][arm]["rows"]
    assert result["arms"]["no_update"]["mean_target_probability_delta"] == 0.0
    assert result["arms"]["no_update"]["mean_target_margin_delta"] == 0.0
    assert result["arms"]["reset_P"]["mean_target_probability_delta"] == 0.0
    assert result["arms"]["reset_P"]["mean_target_margin_delta"] == 0.0

    assert result["fit_status"] == "passed_fit_learnability_diagnostic"
    assert result["fit_is_learnability_only"] is True
    assert result["alternative_justified"] is True
    assert result["alternative_protocol"] == "dc_p1_source_free_feature_alignment_v2"
    assert result["native_execution"] is False
    assert result["sealed_minicpm_loaded"] is False
    assert result["tiny_model_fixture"] is True
    assert result["development_or_final_access"] is False
    assert result["generalization_qualified"] is False
    assert result["final_labels_read"] is False
