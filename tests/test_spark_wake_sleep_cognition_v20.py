from __future__ import annotations

from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F

from replica_lab.contextual_memory import ContextualMemoryPath
from replica_lab.contextual_spark_fixture import (
    build_contextual_adapter,
    ids_and_mask,
    labels_from_ids,
)
from replica_lab.intrinsic_memory_backbone_learning_v19 import (
    HiddenStateConditionedP,
)
from replica_lab.runtime.director import OutcomeSummary
from replica_lab.runtime.task_graph import (
    Budget,
    Constraint,
    ExactTaskState,
    Obligation,
    Subgoal,
    SuccessCondition,
)
from replica_lab.spark_wake_sleep_cognition_v20 import (
    CA1EligibilityGateV20,
    CapabilityModelV20,
    CognitiveCoreV20,
    GatedSelfImprovementVerifierV20,
    SleepConsolidatorV20,
    SleepEpisodeCandidateV20,
    TitansChunkSessionV20,
    evaluate_stage2_9site_backbone_v20,
    functional_titans_inner_update,
    run_master_spark_v20_pipeline,
)
from replica_lab.successors.chunks import ChunkStore, tensor_digest
from replica_lab.successors.formation import ProvisionalNote
from replica_lab.tiered_memory import MemoryScope


def test_stage2_9site_backbone_and_pretrained_parity() -> None:
    report = evaluate_stage2_9site_backbone_v20(seed=17)
    assert report["nine_sites_verified"] is True
    assert report["nine_full_attention_sites"] == [3, 7, 11, 15, 19, 23, 27, 31, 35]
    assert report["all_pretrained_parity_exact"] is True
    assert report["matched_9site_equals_2site_reader_params"] is True
    assert report["all_cells_finite_and_shared_snapshot"] is True
    assert len(report["cells"]) == 8


def test_titans_surprise_momentum_forgetting_chunk_session() -> None:
    torch.manual_seed(17)
    with tempfile.TemporaryDirectory() as tmp:
        with ChunkStore(Path(tmp) / "chunks", subject_id="v20-test") as store:
            snap0 = store.open_sequence("F", "main", "seq-1", initial_prefix=(1, 2))
            init_state = torch.zeros(1, 16, 16, dtype=torch.float32)
            session = TitansChunkSessionV20(
                store,
                snap0,
                init_state,
                committed_tokens=(1, 2),
                eta=0.85,
                theta=0.5,
                alpha=0.05,
            )
            k0 = torch.randn(1, 2, 16)
            v0 = torch.randn(1, 2, 16)
            session.begin_chunk((3, 4))
            read_before = session.read_chunk_start(k0)
            proposed0, metrics0 = session.propose_titans(k0, v0)
            read_during = session.read_chunk_start(k0)
            assert torch.allclose(read_before, read_during)
            assert metrics0["surprise_magnitude"] > 0.0
            assert 0.0 < metrics0["surprise_gate"] < 1.0
            assert metrics0["momentum_l2"] > 0.0

            recomp0, recomp_digest = session.recompute_titans(k0, v0)
            assert recomp_digest == tensor_digest(proposed0)
            commit0 = session.commit()
            assert commit0.status == "committed"

            # Next chunk sees committed state and carries momentum
            session.begin_chunk((5, 6))
            read_c1 = session.read_chunk_start(k0)
            assert not torch.allclose(read_before, read_c1)
            session.propose_titans(torch.randn(1, 2, 16), torch.randn(1, 2, 16))
            # Invalidate uncommitted chunk restores committed state and momentum
            session.invalidate()
            assert tensor_digest(session.state) == tensor_digest(proposed0)


def test_stage3_cognitive_core_and_capability_model_routing() -> None:
    torch.manual_seed(17)
    task_state = ExactTaskState(
        task_id="t-1",
        subject="spark-v20",
        branch="main",
        purpose="test",
        source_epoch=1,
        goal="Verify relay channel",
        success_conditions=(
            SuccessCondition(
                condition_id="sc-1",
                description="Answer accurately",
            ),
        ),
        constraints=(
            Constraint(
                constraint_id="c-1",
                text="Identity read-only",
            ),
        ),
        subgoals=(
            Subgoal(
                subgoal_id="sg-1",
                description="Check E2",
                state="active",
            ),
        ),
        claims=(),
        evidence=(),
        obligations=(
            Obligation(
                obligation_id="ob-1",
                description="Verify receipt",
                state="unresolved",
            ),
        ),
        actions=(),
        outcomes=(),
        notes=(),
        budget=Budget(
            max_decisions=4,
            max_tool_calls=2,
            max_prompt_tokens=1024,
            max_generated_tokens=128,
        ),
        binding={"model": "spark-v20"},
        allowed_tools=("read_file",),
    )
    core = CognitiveCoreV20(slot_dim=64, num_slots=8, seed=17)
    cap = CapabilityModelV20(
        slot_dim=64, abstain_threshold=0.55, sleep_episode_threshold=2, seed=17
    )
    sense = torch.randn(1, 64)
    mem = torch.randn(1, 64)
    identity = torch.randn(1, 64, requires_grad=True)
    slots, id_readonly = core(sense, mem, task_state, identity, recurrent_steps=2)
    assert slots.shape == (1, 8, 64)
    assert id_readonly is True

    known = cap.evaluate_and_select(
        slots,
        retrieval_margin=1.8,
        evidence_verified=True,
        eligible_wake_episodes=1,
        identity_read_only=id_readonly,
    )
    unknown = cap.evaluate_and_select(
        slots,
        retrieval_margin=-1.5,
        evidence_verified=False,
        eligible_wake_episodes=1,
        identity_read_only=id_readonly,
    )
    sleep_trig = cap.evaluate_and_select(
        slots,
        retrieval_margin=1.8,
        evidence_verified=True,
        eligible_wake_episodes=2,
        identity_read_only=id_readonly,
    )
    assert known.selected_action == "answer_direct"
    assert known.should_abstain is False
    assert unknown.should_abstain is True
    assert unknown.selected_action in ("abstain_unknown", "retrieve_warm_cold")
    assert known.p_correct > unknown.p_correct
    assert sleep_trig.should_trigger_sleep is True
    assert sleep_trig.selected_action == "trigger_sleep_consolidation"


def test_stage4_and_5_sleep_consolidation_and_rollback() -> None:
    torch.manual_seed(17)
    adapter, *_ = build_contextual_adapter(
        seed=17, layers=8, spine_sites=(3, 7), spine_slots=8, memory_size=256
    )
    readout = ContextualMemoryPath(adapter, lr=1e-3)
    gen = torch.Generator(device="cpu").manual_seed(17)
    with torch.no_grad():
        for port in readout.reader.ports.values():
            port.output_projection.weight.normal_(0.0, 0.5, generator=gen)
            port.value_projection.weight.normal_(0.0, 0.5, generator=gen)
            port.gate_logit.fill_(2.0)
    qual = {
        "baseline_competence": True,
        "numerical_qualification": True,
        "learning_utility": True,
        "fresh_process_continuation": True,
    }
    learner = HiddenStateConditionedP(
        readout, qual, anchor_layer=3, engineering_fixture=True, seed=17, lr=0.05
    )
    scope = MemoryScope("v20-sleep", "main", "test", 100)
    ids, mask = ids_and_mask(6)
    labels = labels_from_ids(ids)
    vocab_size = int(readout.adapter.config.geometry.vocab_size)
    teacher_logits = F.one_hot(labels.clamp_min(0), num_classes=vocab_size).float() * 8.0

    note = ProvisionalNote(
        note_id="prov-note-1",
        text="speculative note",
        origin="model_generated",
        interaction_id="i-1",
    )
    candidates = [
        SleepEpisodeCandidateV20(
            record_id="rec-verified-1",
            text="Verified fact.",
            question="Question 1?",
            ids=ids,
            mask=mask,
            labels=labels,
            teacher_Reference_logits=teacher_logits,
            verified=True,
            is_provisional_note=False,
            surprise_score=0.8,
            importance=10,
        ),
        SleepEpisodeCandidateV20(
            record_id="prov-note-1",
            text="speculative note",
            question="Speculative?",
            ids=ids,
            mask=mask,
            labels=labels,
            teacher_Reference_logits=teacher_logits,
            verified=False,
            is_provisional_note=True,
            surprise_score=0.95,
            importance=10,
        ),
    ]
    consolidator = SleepConsolidatorV20(learner, top_k=16, kl_weight=0.5)
    report = consolidator.consolidate_sleep_window(
        candidates,
        replay_anchors=[],
        provisional_notes=[note],
        scope=scope,
        steps=5,
    )
    assert report["status"] == "consolidated"
    assert report["gate_report"]["zero_provisional_leakage"] is True
    assert report["gate_report"]["rejected_provisional_count"] == 1
    assert report["wake_ce_improvement"] > 0.0
    assert report["wake_kl_improvement"] > 0.0
    assert "rec-verified-1" in learner.lineage

    # S4 revocation quarantine + pre-sleep rollback
    learner.invalidate("rec-verified-1")
    assert learner.quarantined is True
    rb = consolidator.rollback_to_pre_sleep(authorization="test-rollback-auth")
    assert rb["rolled_back"] is True
    assert learner.quarantined is False
    after_rb = consolidator.evaluate_episode_loss(candidates[0], scope=scope)
    assert abs(after_rb["ce"] - report["pre_wake_ce"]) < 1e-6


def test_stage6_gated_rsi_verifier_c0_to_c5() -> None:
    verifier = GatedSelfImprovementVerifierV20()
    summary = OutcomeSummary(
        task_family="recall",
        failure_class="memory",
        attempts=6,
        successes=3,
        source_event_ids=("ev-1", "ev-2", "ev-3"),
        role="train",
        verified_receipts=1,
        cost_units=10.0,
    )
    blocked = verifier.evaluate_candidate(
        summary,
        stages_1_to_5_qualified=False,
        authority_and_identity_intact=True,
        numeric_and_restoration_exact=True,
        old_task_retention_ratio=0.99,
        unseen_task_transfer_delta=0.1,
        calibration_ece_delta=-0.01,
        matched_cost_ratio=1.0,
    )
    assert blocked["promoted"] is False

    promoted = verifier.evaluate_candidate(
        summary,
        stages_1_to_5_qualified=True,
        authority_and_identity_intact=True,
        numeric_and_restoration_exact=True,
        old_task_retention_ratio=0.99,
        unseen_task_transfer_delta=0.1,
        calibration_ece_delta=-0.01,
        matched_cost_ratio=1.0,
    )
    assert promoted["all_gates_passed"] is True
    assert promoted["promoted"] is True


def test_master_spark_v20_end_to_end_pipeline() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        report = run_master_spark_v20_pipeline(Path(tmp), seed=17)
    assert report["all_categories_and_stages_cpu_verified"] is True


def test_stage1_v19_and_stage2_v20_packet_bindings() -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    if str(root / "scripts") not in sys.path:
        sys.path.insert(0, str(root / "scripts"))
    from run_spark_alignment_v19 import validate_v19_bindings

    p1, proto1, rec1 = validate_v19_bindings(
        root, root / "reports/spark_alignment_v19/native_packet.json"
    )
    assert p1["arm"] == "two"
    assert proto1["sites"] == [3, 7]
    assert (root / p1["output_root"] / "result.json").is_file()

    p2, proto2, rec2 = validate_v19_bindings(
        root, root / "reports/spark_backbone_9site_v20/native_packet.json"
    )
    assert p2["arm"] == "full_attention_nine"
    assert p2["reader_regime"] == "matched"
    assert proto2["sites"] == [3, 7, 11, 15, 19, 23, 27, 31, 35]
    assert proto2["stage1_v19_predecessor_outcome"]["ce_teacher_forced_complete_correct"] == 7
    assert proto2["stage1_v19_predecessor_outcome"]["ce_known_known_pair_both_correct_count"] == 3
    assert (root / p2["output_root"] / "result.json").is_file()

    p2b, proto2b, _rec2b = validate_v19_bindings(
        root,
        root
        / "reports/spark_backbone_9site_v20/native_packet_independent_attempt02.json",
    )
    assert p2b["arm"] == "full_attention_nine"
    assert p2b["reader_regime"] == "independent"
    assert proto2b["sites"] == [3, 7, 11, 15, 19, 23, 27, 31, 35]
    assert proto2b["intrinsic_placement"] == "block_input_query_to_post_block_residual"
    assert proto2b["zero_shock_step0_parity_with_v16_2site"] is True
    res2b_path = root / p2b["output_root"] / "result.json"
    assert res2b_path.is_file()
    import json

    res2b = json.loads(res2b_path.read_text())
    assert res2b["ce_final"]["teacher_forced_complete_correct"] == 8
    assert res2b["ce_final"]["known_known_pair_both_correct_count"] == 4
    assert res2b["ce_final"]["guard_count"] == 0
    assert res2b["ce_final"]["diagnostic_regime"] == "aligned_contextual_recall"

    p2c, proto2c, _rec2c = validate_v19_bindings(
        root,
        root
        / "reports/spark_backbone_9site_v20/native_packet_independent_attempt03.json",
    )
    assert p2c["arm"] == "full_attention_nine"
    assert p2c["reader_regime"] == "independent"
    assert (
        proto2c["stage2b_attempt02_predecessor_outcome"]["result_sha256"]
        == "4057f19165576cc4c62d562c93a676509f52ce27f1434b798aad8907fd1758d1"
    )

    from run_spark_wake_sleep_v21 import validate_v21_bindings

    p21, proto21, _rec21 = validate_v21_bindings(
        root, root / "reports/spark_wake_sleep_v21/native_packet.json"
    )
    assert p21["arm"] == "full_attention_nine"
    assert p21["experiment_version"] == 21
    assert proto21["training"]["wake_steps"] == 128
    assert proto21["training"]["sleep_steps"] == 32


def test_stage3_to_6_wake_sleep_continuum_pipeline_v21_cpu(tmp_path) -> None:
    from replica_lab.intrinsic_memory_cpu_fixture_v18 import (
        fixture_rows,
        make_engine,
    )
    from replica_lab.spark_native_wake_sleep_v21 import (
        WAKE_SLEEP_PROTOCOL_V21,
        run_wake_sleep_continuum_pipeline_v21,
    )

    engine = make_engine(tmp_path)
    try:
        rows = fixture_rows()
        report = run_wake_sleep_continuum_pipeline_v21(
            engine,
            rows,
            wake_steps=8,
            sleep_steps=8,
            sleep_lr=1e-2,
            top_k=8,
            require_competence=False,
        )
        assert report["protocol"] == WAKE_SLEEP_PROTOCOL_V21
        assert report["stage3_semi_cognition"]["calibrated_abstention_on_unknown"] is True
        assert report["stage3_semi_cognition"]["ca1_gate_report"]["zero_provisional_leakage"] is True
        s3 = report["stage4_sleep_s3_source_free_p1"]
        assert s3["stop_gradient_verified"] is True
        assert s3["phi_unchanged_during_sleep"] is True
        assert s3["p1_minus_no_memory_margin"] > 0.0
        assert s3["p1_minus_untouched_p1_margin"] > 0.0
        s4 = report["stage5_s4_revocation_and_rollback"]
        assert s4["quarantine_activated"] is True
        assert s4["quarantine_blocked_contaminated_inference"] is True
        assert s4["rollback_receipt"]["exact_match"] is True
        assert s4["s4_gate_passed"] is True
        s5 = report["stage5_s5_hope_continuum"]
        assert s5["hope_minus_single_clock_mean_margin"] >= 0.0
        assert len(report["state_restoration"]) == 2
        assert all(all(c["equal"].values()) for c in report["state_restoration"])
    finally:
        engine.path.close()




