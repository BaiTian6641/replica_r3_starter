from __future__ import annotations

from pathlib import Path
import torch

from replica_lab.intrinsic_memory_cpu_fixture_v18 import (
    fixture_rows,
    make_engine,
)
from replica_lab.spark_semi_cognition_v22 import (
    EQUAL_INFO_CONDITIONS_V22,
    SEMI_COGNITION_PROTOCOL_V22,
    run_spark_semi_cognition_v22_pipeline,
)


def test_spark_semi_cognition_v22_pipeline_cpu(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    try:
        gen = torch.Generator(device="cpu").manual_seed(17)
        with torch.no_grad():
            for port in engine.path.reader.ports.values():
                port.output_projection.weight.normal_(0.0, 0.5, generator=gen)
                port.value_projection.weight.normal_(0.0, 0.5, generator=gen)
                port.gate_logit.fill_(2.0)
        for group in engine.path.optimizer.param_groups:
            group["lr"] = 2e-2

        raw_rows = fixture_rows()
        rows = []
        for idx in range(len(raw_rows)):
            r = dict(raw_rows[idx ^ 1])
            r["world_id"] = raw_rows[idx]["world_id"]
            r["entity"] = f"Badge-{r['family']}"
            r["text"] = f"{r['text']} Tone-{r['answer']}."
            rows.append(r)
        report = run_spark_semi_cognition_v22_pipeline(
            engine,
            rows,
            wake_steps=32,
            sleep_steps=16,
            core_steps=30,
            head_steps=50,
            sleep_lr=2e-2,
            top_k=8,
            require_competence=False,
        )
        assert report["protocol"] == SEMI_COGNITION_PROTOCOL_V22

        # Stage V22-A checks
        v22a = report["v22a_two_pass_cognitive_core"]
        assert v22a["step0_zero_shock_max_abs_diff"] == 0.0
        assert v22a["parameter_matching"]["within_5_percent_budget"] is True
        assert v22a["phi_unchanged_during_core_training"] is True
        for cond in EQUAL_INFO_CONDITIONS_V22:
            assert cond in v22a["conditions"]
        assert v22a["comparisons"]["full_minus_core_disabled_margin"] > 0.0
        assert v22a["comparisons"]["full_minus_workspace_zero_margin"] > 0.0
        assert v22a["comparisons"]["full_minus_workspace_shuffled_margin"] > 0.0
        assert v22a["comparisons"]["full_minus_ff_core_margin"] > 0.0
        assert v22a["v22a_gate_passed"] is True

        # Stage V22-B checks
        v22b = report["v22b_calibrated_monitor_heads"]
        assert v22b["zero_core_gradient_leakage"] is True
        assert v22b["head_gradients_finite_nonzero"] is True
        assert v22b["post_calibration_brier"] < v22b["base_rate_brier"]
        assert v22b["pooled_outcome_head_brier"] < v22b["base_rate_brier"]
        assert v22b["post_calibration_ece"] <= 0.10
        assert v22b["action_kind_routing_accuracy"] == 1.0
        assert v22b["v22b_gate_passed"] is True

        # Stage V22-C checks
        v22c = report["v22c_live_titans_and_paraphrase_p1"]
        s2b = v22c["s2b_live_256d_titans"]
        assert s2b["d_mem"] == 256
        assert s2b["committed_chunks"] == 4
        assert s2b["intra_chunk_prefix_equality"] is True
        assert s2b["recompute_parity"] is True
        assert s2b["watermark_idempotence_verified"] is True
        s3_s5 = v22c["s3_s5_paraphrase_robust_source_free"]
        assert s3_s5["paraphrase_anchor_max_abs_diff"] <= 1e-6
        assert s3_s5["stop_gradient_verified"] is True
        assert len(s3_s5["paraphrase_families"]) == 3
        assert v22c["v22c_gate_passed"] is True

        # Stage V22-D checks
        v22d = report["v22d_reset_dissociation_and_rs4"]
        dissoc = v22d["reset_dissociation_6_conditions"]
        assert dissoc["e1_task_reset_dissociated"] is True
        assert dissoc["e2_identity_null_dissociated"] is True
        assert dissoc["e3_floor_and_commutativity_verified"] is True
        assert dissoc["commutativity_max_abs_diff"] == 0.0
        assert dissoc["both_reset_max_abs_diff"] == 0.0
        rs4 = v22d["rs4_two_cycle_self_improvement"]
        assert rs4["cycle1_director_status"] == "proposed"
        assert rs4["cycle2_director_status"] == "proposed"
        assert rs4["inter_family_task_workspace_reset"] is True
        assert rs4["phi_unchanged_during_cycle2"] is True
        assert rs4["negative_control_rejected"] is True
        assert rs4["positive_cycle2_promotion_receipt"]["promoted"] is True
        assert v22d["v22d_gate_passed"] is True
        assert report["all_v22_gates_passed"] is True
    finally:
        engine.path.close()
