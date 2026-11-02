"""CPU verification tests for V19 alignment, backbone matrix, S3 P1, and S5 continuum learning."""
from __future__ import annotations

import pytest
import torch

from replica_lab.contextual_memory import ContextualMemoryError, ContextualMemoryPath
from replica_lab.contextual_spark_fixture import build_contextual_adapter
from replica_lab.contextual_v12_fixture import canonical_span, open_v12_store, scope
from replica_lab.intrinsic_memory_alignment_v19 import (
    ALIGNMENT_PROTOCOL,
    alignment_protocol_contract,
    evaluate_exact_text_competence,
    run_alignment_comparison,
    teacher_topk_targets,
    validate_alignment_pairs_v19,
)
from replica_lab.intrinsic_memory_backbone_learning_v19 import (
    BACKBONE_MATRIX_FORMAT,
    HiddenStateConditionedP,
    INTEGRATED_PROGRESSION_FORMAT,
    S3_COMPARISON_FORMAT,
    S5_CONTINUUM_FORMAT,
    profile_memory_backbone_matrix,
    run_integrated_spark_learning_progression_v19,
    run_s3_contextual_p_comparison,
    run_s5_continuum_vs_single_clock_comparison,
)
from replica_lab.intrinsic_memory_cpu_fixture_v18 import fixture_rows, make_engine
from replica_lab.intrinsic_memory_measurement_v18 import capture_state, verify_state


def _build_qualified_cpu_readout(seed: int = 17) -> ContextualMemoryPath:
    adapter, *_ = build_contextual_adapter(
        seed=seed, layers=8, spine_sites=(3, 7), spine_slots=8, memory_size=256
    )
    readout = ContextualMemoryPath(adapter, lr=1e-3)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for port in readout.reader.ports.values():
            port.output_projection.weight.normal_(0.0, 0.5, generator=gen)
            port.value_projection.weight.normal_(0.0, 0.5, generator=gen)
            port.gate_logit.fill_(2.0)
    return readout


def test_matched_ce_vs_topk_kl_alignment_and_probe_separation(tmp_path):
    contract = alignment_protocol_contract(top_k=8, steps=4, paired_joint=True)
    assert contract["format"] == ALIGNMENT_PROTOCOL
    assert contract["sites"] == [3, 7]
    assert contract["intrinsic_reader_key_mode"] == "content"
    with pytest.raises(ValueError, match="top_k"):
        alignment_protocol_contract(top_k=1)
    with pytest.raises(ValueError, match="steps"):
        alignment_protocol_contract(steps=0)

    engine = make_engine(tmp_path)
    try:
        initial = capture_state(engine.path)
        rows = fixture_rows()
        teacher = teacher_topk_targets(engine, rows[0], top_k=8)
        assert teacher["top_k"] == 8
        assert teacher["teacher_log_probs"].shape[1] == 8
        assert torch.allclose(
            teacher["teacher_log_probs"].exp().sum(dim=-1),
            torch.ones(teacher["supervised_tokens"]),
            atol=1e-5,
        )

        result = run_alignment_comparison(
            engine, rows, top_k=8, steps=4, paired_joint=True
        )
        verify_state(engine.path, initial)
        assert set(result["arms"]) == {"direct_answer_plus_eos_ce", "top_8_teacher_kl"}
        for arm_name, arm_data in result["arms"].items():
            assert len(arm_data["updates"]) == 4
            for step in arm_data["updates"]:
                assert step["per_module_gradients"]["producer"]["finite"] is True
                assert step["per_module_gradients"]["reader"]["finite"] is True
                assert step["actual_adamw_displacement"]["producer"]["l2"] > 0.0
                assert step["actual_adamw_displacement"]["reader"]["l2"] > 0.0
            final = arm_data["final"]
            assert 0.0 <= final["source_feature_probe_accuracy"] <= 1.0
            assert 0.0 <= final["slot_probe_accuracy"] <= 1.0
            assert 0.0 <= final["read_probe_accuracy"] <= 1.0
            assert final["diagnostic_regime"] in {
                "compression_bottleneck",
                "decoder_interface_bottleneck",
                "aligned_contextual_recall",
            }
            assert len(arm_data["probes"]["slots"]) == 8
    finally:
        engine.path.close()


def test_memory_backbone_matrix_sites_regimes_and_placements():
    matrix = profile_memory_backbone_matrix()
    assert matrix["format"] == BACKBONE_MATRIX_FORMAT
    assert len(matrix["cells"]) == 12
    assert matrix["matched_reader_parameter_parity"] is True
    assert matrix["all_shared_snapshots_verified"] is True
    assert matrix["all_sites_invoked_and_finite"] is True

    by_key = {
        (c["arm"], c["regime"], c["placement"]): c for c in matrix["cells"]
    }
    assert by_key[("two", "independent", "block_input_query_to_post_block_residual")]["site_count"] == 2
    assert by_key[("full_attention_nine", "matched", "block_input_query_to_pre_mlp_residual")]["site_count"] == 9
    assert by_key[("all", "matched", "block_input_query_to_post_block_residual")]["site_count"] == 36
    # Verify independent reader scales linearly with site count while matched stays fixed
    ind_two = by_key[("two", "independent", "block_input_query_to_post_block_residual")]["reader_trainable_parameters"]
    ind_nine = by_key[("full_attention_nine", "independent", "block_input_query_to_post_block_residual")]["reader_trainable_parameters"]
    matched_all = by_key[("all", "matched", "block_input_query_to_post_block_residual")]["reader_trainable_parameters"]
    assert ind_nine > ind_two
    assert matched_all == ind_two


def test_s3_hidden_state_conditioned_p_vs_fixed_question_slot_control():
    comparison = run_s3_contextual_p_comparison(
        _build_qualified_cpu_readout, steps=6, seed=17
    )
    assert comparison["format"] == S3_COMPARISON_FORMAT
    assert comparison["both_arms_improved_under_source_free_eval"] is True
    assert comparison["both_arms_quarantine_and_rollback_verified"] is True

    hidden_arm = comparison["arms"]["hidden_state_conditioned_p"]
    assert hidden_arm["read_mode"] == "hidden_state_conditioned_causal_anchor_P"
    for step in hidden_arm["steps"]:
        assert step["anchor_stop_gradient_verified"] is True
        assert step["phi_and_p0_zero_grad_verified"] is True
        assert step["E"] == 0 and step["S"] == 0 and step["F"] == 0 and step["C"] == 0
        assert step["answer_bearing_I"] == 0

    # Verify HiddenStateConditionedP rejects external evidence during source-free eval
    readout = _build_qualified_cpu_readout()
    learner = HiddenStateConditionedP(
        readout,
        {
            "baseline_competence": True,
            "numerical_qualification": True,
            "learning_utility": True,
            "fresh_process_continuation": True,
        },
        engineering_fixture=True,
    )
    ids = torch.tensor([[3, 5, 7, 9]], dtype=torch.long)
    mask = torch.ones_like(ids)
    with pytest.raises(ContextualMemoryError, match="source-free P evaluation"):
        learner.forward(
            ids,
            mask,
            "What color is the badge?",
            scope=scope("s3"),
            evidence="forbidden_external_evidence",
        )


def test_s5_continuum_nested_clock_vs_single_clock_and_tier_independence(tmp_path):
    result = run_s5_continuum_vs_single_clock_comparison(
        width=32, rank=8, episodes=3, chunks_per_episode=2, seed=17
    )
    assert result["format"] == S5_CONTINUUM_FORMAT
    assert result["equal_trainable_parameters"] is True
    assert result["timescale_separation_verified"] is True
    assert result["tier_movement_advances_learning_clocks"] is False
    assert result["continuum_nested_clock"]["fast_clock_steps"] == 6
    assert result["continuum_nested_clock"]["slow_clock_steps"] == 3
    assert result["matched_single_clock_control"]["single_clock_steps"] == 6

    # Verify storage-tier placement/residency transitions in TieredMemoryStore
    # never advance learning clocks on the engine path
    engine = make_engine(tmp_path)
    store = open_v12_store(tmp_path / "store")
    sc = scope("tier_vs_clock")
    try:
        before_state = capture_state(engine.path)
        for placement in ("warm", "cold"):
            span = canonical_span(
                f"Placement {placement} record.", record_id=f"rec_{placement}"
            )[0]
            engine.path.admit(store, span, scope=sc, placement=placement)
        # Warm read populates hot cache without advancing learning clocks
        engine.path.read(store, "Placement warm", scope=sc)
        verify_state(engine.path, before_state)
    finally:
        store.close()
        engine.path.close()


def test_v19_alignment_disjoint_diagnostic_and_shuffled_controls(tmp_path):
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    fit_rows = json.loads(
        (root / "data/intrinsic_memory_alignment_v19/fit.json").read_text()
    )["rows"]
    diag_rows = json.loads(
        (root / "data/intrinsic_memory_alignment_v19/diagnostic_fit.json").read_text()
    )["rows"]
    validate_alignment_pairs_v19(fit_rows)
    validate_alignment_pairs_v19(diag_rows)

    engine = make_engine(tmp_path)
    try:
        comp = evaluate_exact_text_competence(engine, fit_rows)
        assert len(comp["worlds"]) == 8
        assert "competent" in comp["summary"]

        result = run_alignment_comparison(
            engine,
            fit_rows,
            diagnostic_rows=diag_rows,
            top_k=8,
            steps=2,
            paired_joint=True,
        )
        assert "exact_text_competence" in result
        assert result["exact_text_competence"]["disjoint_diagnostic"] is not None
        for arm_name in ("direct_answer_plus_eos_ce", "top_8_teacher_kl"):
            arm = result["arms"][arm_name]
            assert "disjoint_diagnostic" in arm
            assert "slot_shuffled_probe_accuracy" in arm["final"]
            assert "slot_probe_minus_shuffled_gap" in arm["final"]
            assert "slots_shuffled" in arm["probes"]
            assert len(arm["disjoint_diagnostic"]["final_worlds"]) == 8
        assert "ce_minus_kl_pair_minimum_margin" in result["comparison"]
        assert "ce_primary_support_passed" in result["comparison"]
    finally:
        engine.path.close()


def test_integrated_spark_learning_progression_s2a_s2b_s3_s4_s5(tmp_path):
    report = run_integrated_spark_learning_progression_v19(tmp_path, seed=17)
    assert report["format"] == INTEGRATED_PROGRESSION_FORMAT
    assert report["all_stages_verified"] is True
    assert report["s2a_formation"]["delayed_visibility_verified"] is True
    assert report["s2a_formation"]["provisional_excluded_from_evidence"] is True
    assert report["s2a_formation"]["duplicate_idempotent"] is True
    assert report["s2a_formation"]["conflict_rejected"] is True
    assert report["tiered_residency"]["hot_read_ok"] is True
    assert report["tiered_residency"]["cold_read_ok"] is True
    assert report["tiered_residency"]["warm_read_ok"] is True
    assert (
        report["tiered_residency"]["superseded_record_ineligible_after_correction"]
        is True
    )
    assert report["s2b_fast_associative_chunks"]["recompute_matches_proposed"] is True
    assert (
        report["s2b_fast_associative_chunks"]["duplicate_barrier_idempotent"] is True
    )
    assert report["s2b_fast_associative_chunks"]["prefix_overlap_refused"] is True
    assert (
        report["s2b_fast_associative_chunks"]["causal_chunk_visibility_verified"]
        is True
    )
    assert report["s3_persistent_plastic_p1"]["source_free_improved"] is True
    assert (
        report["s3_persistent_plastic_p1"]["anchor_stop_gradient_verified"] is True
    )
    assert (
        report["s3_persistent_plastic_p1"]["phi_and_p0_zero_grad_verified"] is True
    )
    assert report["s4_correction_retention_rollback"]["superseded_excluded"] is True
    assert report["s4_correction_retention_rollback"]["revoked_excluded"] is True
    assert report["s4_correction_retention_rollback"]["stale_chunk_rejected"] is True
    assert (
        report["s4_correction_retention_rollback"][
            "p1_quarantined_on_lineage_invalidation"
        ]
        is True
    )
    assert (
        report["s4_correction_retention_rollback"][
            "authenticated_rollback_restores_initial_ce"
        ]
        is True
    )
    assert report["s5_continuum_vs_single_clock"]["timescale_separation_verified"] is True

