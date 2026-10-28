"""DC-P1 learning-hypothesis audit invariants using the real CPU paths."""
from __future__ import annotations

import pytest

from replica_lab.dc_p1_learning_audit import run_cpu_audit


def test_learning_audit_measures_global_rescale_optimizer_and_feature_shift():
    result = run_cpu_audit()
    assert result["gradient"]["claim"] == "global_ce_rescale"
    assert result["gradient"]["gradient_ratio"] == pytest.approx(1.5)
    assert result["gradient"]["masks_coincide"] is True

    control, treatment, large_epsilon, clipped = result["updates"]
    assert treatment["loss"] == pytest.approx(control["loss"] * 1.5)
    assert treatment["max_logit_delta"] > 0.0
    assert large_epsilon["max_logit_delta"] == 0.0
    assert clipped["raw_gradient_norm"] > 1.0
    assert 0.0 < clipped["clip_scale"] < 1.0
    assert clipped["clipped_gradient_norm"] == pytest.approx(1.0)
    assert clipped["optimizer_class"] == "NativeFusedAdamW"

    assert result["feature"]["source_free_feature_shift_l2"] > 0.0
    assert result["selected_expert_max_logit_delta"] > 0.0
    assert result["alternative_justified"] is True
    assert result["alternative_protocol"] == "dc_p1_source_free_feature_alignment_v1"
    assert result["native_execution"] is False
    assert result["model_loaded"] is False
    assert result["final_labels_read"] is False
