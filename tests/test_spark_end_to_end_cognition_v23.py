from __future__ import annotations

from pathlib import Path
import torch

from replica_lab.intrinsic_memory_cpu_fixture_v18 import (
    fixture_rows,
    make_engine,
)
from replica_lab.spark_end_to_end_cognition_v23 import (
    END_TO_END_COGNITION_PROTOCOL_V23,
    REPLICA_TEST_SUBJECT_V23,
    run_spark_end_to_end_cognition_v23_pipeline,
)
from replica_lab.spark_semi_cognition_v22 import EQUAL_INFO_CONDITIONS_V22


def test_spark_end_to_end_cognition_v23_pipeline_cpu(tmp_path: Path) -> None:
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
            r["question"] = f"What is the recorded color of Badge-{r['family']}?"
            r["text"] = f"{r['text']} Tone-{r['answer']}."
            rows.append(r)

        report = run_spark_end_to_end_cognition_v23_pipeline(
            engine,
            rows,
            wake_steps=32,
            sleep_steps=16,
            core_steps=32,
            head_steps=60,
            sleep_lr=2e-2,
            top_k=8,
            require_competence=False,
        )
        assert report["protocol"] == END_TO_END_COGNITION_PROTOCOL_V23

        # Verify work ledger counters for retrievals, generated_tokens, and restart_restores
        snap = engine.ledger.snapshot(cached=True)
        assert snap["reserved"]["retrievals"] >= 8
        assert snap["reserved"]["generated_tokens"] >= 12
        assert snap["reserved"]["restart_restores"] >= 1

        # Stage V23-A checks
        v23a = report["v23a_end_to_end_cognitive_core"]
        assert v23a["step0_zero_shock_max_abs_diff"] == 0.0
        grad_ver = v23a["end_to_end_gradient_verification"]
        assert grad_ver["action_pass_grad_reaches_core"] is True
        assert grad_ver["action_pass_grad_reaches_sense_proj"] is True
        assert grad_ver["action_pass_grad_reaches_bridge"] is True
        assert grad_ver["base_backbone_grads_absent"] is True
        assert grad_ver["zero_synthetic_48_markers"] is True
        assert v23a["parameter_matching"]["within_5_percent_budget"] is True
        assert v23a["phi_unchanged_during_core_training"] is True
        for cond in EQUAL_INFO_CONDITIONS_V22:
            assert cond in v23a["conditions"]
        for w_info in v23a["conditions"]["full"]["worlds"]:
            assert w_info["span_probabilities"][w_info["mode_index"]] > 0.90
        assert v23a["comparisons"]["full_minus_core_disabled_margin"] > 0.0
        assert v23a["comparisons"]["full_minus_workspace_zero_margin"] > 0.0
        assert v23a["comparisons"]["full_minus_workspace_shuffled_margin"] > 0.0
        assert v23a["comparisons"]["full_minus_ff_core_margin"] > 0.0
        cl = v23a["closed_loop_k0_to_k1_controller"]
        assert cl["retrievals_executed"] == 8
        for ep in cl["episodes"]:
            assert ep["k1_span_probabilities"][ep["world_index"] % 2] > 0.90
        assert v23a["v23a_gate_passed"] is True

        # Stage V23-B checks
        v23b = report["v23b_live_trajectory_heads_and_compilation"]
        assert v23b["zero_core_gradient_leakage"] is True
        assert v23b["head_gradients_finite_nonzero"] is True
        assert v23b["zero_synthetic_slice_offsets"] is True
        assert v23b["evaluated_on_disjoint_diagnostic_worlds"] is True
        assert v23b["post_calibration_brier"] < v23b["base_rate_brier"]
        assert v23b["pooled_outcome_head_brier"] < v23b["base_rate_brier"]
        assert v23b["post_calibration_ece"] <= 0.10
        assert v23b["action_kind_routing_accuracy"] == 1.0
        c12 = v23b["c12_compiled_procedure"]
        assert c12["evaluation_accepted"] is True
        assert c12["all_independent_verifiers_passed"] is True
        assert c12["stale_source_tamper_rejected"] is True
        assert c12["reloaded_from_manifest_consistent"] is True
        assert c12["heldout_restart_execution_succeeded"] is True
        assert v23b["v23b_gate_passed"] is True

        # Stage V23-C checks
        v23c = report["v23c_content_keyed_titans_and_untagged_paraphrase"]
        s2b = v23c["s2b_content_keyed_256d_titans"]
        assert s2b["d_mem"] == 256
        assert s2b["committed_chunks"] == 4
        assert s2b["content_keyed_from_prompt_anchors"] is True
        assert s2b["zero_oracle_target_norm_at_inference"] is True
        assert s2b["intra_chunk_prefix_equality"] is True
        assert s2b["recompute_parity"] is True
        assert s2b["watermark_idempotence_verified"] is True
        s3_s5 = v23c["s3_s5_untagged_salience_paraphrase_source_free"]
        assert s3_s5["zero_entity_string_lookup_at_inference"] is True
        assert s3_s5["evaluated_directly_on_paraphrased_encodings"] is True
        assert s3_s5["paraphrase_anchor_max_abs_diff"] <= 1e-6
        assert s3_s5["stop_gradient_verified"] is True
        assert len(s3_s5["paraphrase_families"]) == 3
        assert v23c["v23c_gate_passed"] is True

        # Stage V23-D checks
        v23d = report["v23d_replica_identity_dissociation_decoding_and_rs4"]
        rep_id = v23d["testing_usage_replica_identity"]
        assert rep_id["subject_id"] == REPLICA_TEST_SUBJECT_V23
        assert rep_id["provenance"] == "fictional"
        assert rep_id["personal_fidelity_claim"] is False
        assert rep_id["temporal_supersession_verified"] is True
        assert rep_id["episode_revocation_quarantine_verified"] is True
        assert rep_id["source_vs_branch_experience_separated"] is True
        assert rep_id["store_reload_digest_match"] is True

        dissoc = v23d["reset_dissociation_6_conditions_live_logits"]
        assert dissoc["e1_task_reset_dissociated"] is True
        assert dissoc["e2_identity_null_dissociated"] is True
        assert dissoc["e3_floor_and_commutativity_verified"] is True
        assert dissoc["identity_never_overrides_facts"] is True
        assert dissoc["commutativity_max_abs_diff"] == 0.0
        assert dissoc["both_reset_max_abs_diff"] == 0.0

        gen_rep = v23d["autoregressive_multitoken_generation"]
        assert gen_rep["total_generated_tokens"] > 0
        assert gen_rep["action_pass_all_terminated"] is True
        assert gen_rep["hope_e0_all_terminated"] is True
        assert gen_rep["unknown_query_routed_to_null_bank"] is True
        assert gen_rep["false_recall"] is False

        rs4 = v23d["rs4_two_cycle_parametric_self_improvement"]
        assert rs4["zero_python_id_routing"] is True
        assert rs4["cycle1_director_status"] == "proposed"
        assert rs4["cycle2_director_status"] == "proposed"
        assert rs4["inter_family_task_workspace_reset"] is True
        assert rs4["phi_unchanged_during_cycle2"] is True
        assert rs4["negative_control_rejected"] is True
        assert rs4["positive_cycle2_promotion_receipt"]["promoted"] is True

        ckpt = v23d["unified_checkpoint_roundtrip"]
        assert ckpt["unified_checkpoint_roundtrip_verified"] is True
        assert ckpt["action_pass_reload_max_abs_diff"] == 0.0
        assert ckpt["hope_reload_max_abs_diff"] == 0.0

        assert v23d["v23d_gate_passed"] is True
        assert report["complete_restoration_verified"] is True
        assert report["all_v23_gates_passed"] is True
    finally:
        engine.path.close()
