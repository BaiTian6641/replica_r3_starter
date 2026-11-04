from __future__ import annotations

from pathlib import Path
import torch

from replica_lab.intrinsic_memory_cpu_fixture_v18 import (
    fixture_rows,
    make_engine,
)
from replica_lab.spark_de_scaffolded_cognition_v24 import (
    DE_SCAFFOLDED_COGNITION_PROTOCOL_V24,
    REPLICA_TEST_SUBJECT_V24,
    run_spark_de_scaffolded_cognition_v24_pipeline,
)


def test_spark_de_scaffolded_cognition_v24_pipeline_cpu(tmp_path: Path) -> None:
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

        report = run_spark_de_scaffolded_cognition_v24_pipeline(
            engine,
            rows,
            wake_steps=32,
            sleep_steps=16,
            core_steps=36,
            head_steps=60,
            sleep_lr=2e-2,
            top_k=8,
            require_competence=False,
        )
        assert report["protocol"] == DE_SCAFFOLDED_COGNITION_PROTOCOL_V24

        snap = engine.ledger.snapshot(cached=True)
        assert snap["reserved"]["generated_tokens"] >= 12
        assert snap["reserved"]["restart_restores"] >= 1

        # Stage V24-A checks (Scaffolds 1 & 2 eliminated)
        v24a = report["v24a_de_scaffolded_cognitive_core"]
        assert v24a["phi_frozen_unchanged"] is True
        assert v24a["core_weights_updated"] is True
        assert v24a["ridge_override_used"] is False
        assert v24a["bridge_out_zeroed"] is False
        assert v24a["bridge_out_weight_l2_norm"] > 0.0
        assert v24a["mean_residual_delta_l2_norm"] > 0.0
        assert v24a["caller_authority_vec_used"] is False
        assert v24a["candidate_order_permutation_invariant_8_of_8"] is True
        assert v24a["full_condition"]["complete_correct_count"] == 8
        assert v24a["swapped_candidate_order_eval"]["complete_correct_count"] == 8
        assert (
            v24a["disjoint_diagnostic_zero_shot_transfer"][
                "complete_correct_count"
            ]
            == 8
        )
        assert v24a["all_ablations_causally_degrade"] is True
        cl = v24a["closed_loop_task_runtime_controller"]
        assert cl["uses_task_runtime_decide"] is True
        assert (
            cl["consolidated_e0_early_stop_correct"]
            == cl["consolidated_e0_total"]
        )
        assert cl["contradictory_k0_to_k1_escalated_correct"] == 8
        assert v24a["v24a_gate_passed"] is True

        # Stage V24-B checks (Scaffold 3 eliminated)
        v24b = report["v24b_unsuppressed_workspace_heads_and_c12_c13"]
        assert v24b["workspace_weights_suppressed"] is False
        assert v24b["workspace_weight_l2_norm"] > 0.50
        assert v24b["workspace_causal_reliance_delta"] >= 0.25
        assert (
            v24b["post_calibration_heldout_full_workspace"]["success_accuracy"]
            >= 0.85
        )
        assert (
            v24b["post_calibration_heldout_full_workspace"]["action_accuracy"]
            >= 0.85
        )
        ans_h = v24b["post_generation_answer_correctness_head"]
        assert ans_h["accuracy"] >= 0.85
        assert ans_h["ece"] <= 0.15
        c13 = v24b["c13_learning_director_proposal_selector"]
        assert c13["ranking_correct"] is True
        c12 = v24b["c12_live_task_runtime_skill_compilation"]
        assert c12["generated_via_task_runtime_decide"] is True
        assert c12["compiled_candidates_count"] >= 1
        assert c12["activated_skill_status"] == "active"
        assert v24b["v24b_gate_passed"] is True

        # Stage V24-C checks (Scaffolds 4 & 5 eliminated)
        v24c = report["v24c_online_surprise_titans_and_single_pass_salience"]
        s2b = v24c["s2b_online_surprise_titans"]
        assert s2b["batch_qr_used"] is False
        assert s2b["online_per_chunk_keys"] is True
        assert s2b["active_surprise_momentum_eta_gt_0"] is True
        assert s2b["active_adaptive_forgetting_alpha_gt_0"] is True
        assert s2b["stream_chunks_observed"] == 6
        assert s2b["post_stream_complete_correct"] == 4
        assert s2b["source_invalidation_verified"] is True

        s3_s5 = v24c["s3_s5_single_pass_salience_paraphrase"]
        assert s3_s5["disc_token_whitelist_used"] is False
        assert s3_s5["second_isolated_forward_pass_used"] is False
        assert s3_s5["s3_fit_epoch1_paraphrase"]["accuracy"] == 1.0
        assert s3_s5["s5_hope_fit_active_paraphrase"]["accuracy"] == 1.0
        assert (
            s3_s5["s3_disjoint_diagnostic_unseen_entities_paraphrase"][
                "accuracy"
            ]
            == 1.0
        )
        assert (
            s3_s5["s3_fit_epoch1_paraphrase"]["min_anchor_cosine_similarity"]
            >= 0.90
        )
        assert (
            s3_s5["s3_disjoint_diagnostic_unseen_entities_paraphrase"][
                "min_anchor_cosine_similarity"
            ]
            >= 0.90
        )
        assert v24c["v24c_gate_passed"] is True

        # Stage V24-D checks (Scaffolds 6, 7 & 8 eliminated)
        v24d = report["v24d_live_replica_decoding_and_two_bank_rs4"]
        rep_id = v24d["testing_usage_replica_identity"]
        assert rep_id["subject_id"] == REPLICA_TEST_SUBJECT_V24
        assert rep_id["provenance"] == "fictional"
        assert rep_id["personal_fidelity_claim"] is False
        assert rep_id["episodes_count"] == 8
        assert rep_id["principles_count"] == 5
        assert rep_id["snapshots_count"] == 3
        assert rep_id["proposed_principle_rejected_from_snapshot"] is True
        assert rep_id["temporal_supersession_verified"] is True
        assert rep_id["episode_revocation_quarantine_verified"] is True
        assert rep_id["source_vs_branch_experience_separated"] is True
        assert rep_id["store_reload_digest_match"] is True
        assert rep_id["zero_static_fidelity_strings"] is True
        assert (
            rep_id["arm_composite_scores"]["neural_identity_reader"]
            > rep_id["arm_composite_scores"]["generic"]
        )

        dissoc = v24d["reset_dissociation_6_conditions_live_logits"]
        assert dissoc["e1_task_reset_dissociated"] is True
        assert dissoc["e2_identity_null_dissociated"] is True
        assert dissoc["e3_floor_and_commutativity_verified"] is True
        assert dissoc["identity_never_overrides_facts"] is True
        assert dissoc["identity_reader_zeroed_before_export"] is False
        assert dissoc["identity_reader_weight_l2_norm"] > 0.0
        assert dissoc["commutativity_max_abs_diff"] == 0.0
        assert dissoc["both_reset_max_abs_diff"] == 0.0

        gen_rep = v24d["autoregressive_multitoken_generation"]
        assert gen_rep["total_generated_tokens"] > 0
        assert gen_rep["action_pass_all_terminated"] is True
        assert gen_rep["hope_e0_all_terminated"] is True
        assert gen_rep["unknown_query_routed_to_null_bank"] is True
        assert gen_rep["false_recall"] is False

        rs4 = v24d["rs4_two_bank_consolidated_self_improvement"]
        assert rs4["total_hope_banks"] == 2
        assert rs4["third_adapter_spawned"] is False
        assert rs4["raw_anchor_key_lists_used"] is False
        assert rs4["cycle1_director_status"] == "proposed"
        assert rs4["cycle2_director_status"] == "proposed"
        assert rs4["inter_family_task_workspace_reset"] is True
        assert rs4["phi_unchanged_during_cycle2"] is True
        assert rs4["negative_control_rejected"] is True
        assert rs4["positive_cycle2_promotion_receipt"]["promoted"] is True

        ckpt = v24d["unified_checkpoint_roundtrip"]
        assert ckpt["unified_checkpoint_roundtrip_verified"] is True
        assert ckpt["action_pass_reload_max_abs_diff"] == 0.0
        assert ckpt["hope_reload_max_abs_diff"] == 0.0
        assert ckpt["identity_preference_reload_max_abs_diff"] == 0.0
        assert ckpt["identity_reader_weight_l2_norm_in_checkpoint"] > 0.0
        assert ckpt["bridge_out_weight_l2_norm_in_checkpoint"] > 0.0

        assert v24d["v24d_gate_passed"] is True
        assert report["complete_restoration_verified"] is True
        assert report["all_v24_gates_passed"] is True
    finally:
        engine.path.close()
