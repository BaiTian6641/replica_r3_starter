"""Authority-specific invalidation and wrapped cold-read regressions; CPU only."""
from dataclasses import replace

import pytest

from replica_lab.contextual_memory import AuthorityRejection
from replica_lab.contextual_memory_v5 import AuthorityReadFailure, ContextualReadFailure
from replica_lab.intrinsic_memory_acceptance_v13 import InvalidationContractError
from replica_lab.intrinsic_memory_acceptance_v15 import (
    chronological_correction, complete_response_assessment, expect_invalidation,
    sequence_log_likelihood)
from replica_lab.intrinsic_memory_coverage_v12 import COMPOUND_FAMILIES
import torch
from replica_lab.intrinsic_memory_budget_v12 import BudgetError
import profile_intrinsic_memory_density_v12 as profile


def test_actual_cold_revocation_has_explicit_authority_reason(tmp_path):
    path = profile.build_path("two")
    store = profile.open_store(tmp_path / "store")
    try:
        path.admit(store, profile.spans()[0], scope=profile.scope(), placement="cold")
        pending = path.read(store, "Badge", scope=profile.scope())
        store.revoke_record("r1.span0", reason="withdrawn", scope=profile.scope())
        outcome = expect_invalidation(
            "revoked_source", lambda: path.resume(store, pending, scope=profile.scope(), wait_seconds=1))
        assert outcome.error_type == "AuthorityReadFailure"
        assert outcome.authority_reason == "revoked_source"
        assert outcome.cause_type == "AuthorityRejection"
    finally:
        store.close()
        path.close()


def test_actual_stale_snapshot_is_positive_authority_evidence(tmp_path):
    path = profile.build_path("two")
    store = profile.open_store(tmp_path / "store")
    try:
        original = profile.spans()[0]
        path.admit(store, original, scope=profile.scope())
        batch = path.read(store, "Badge", scope=profile.scope())
        corrected = replace(profile.spans("Badge blue", name="new", available=2)[0],
                            source_epoch=2, supersedes=original.record_id)
        assert path.correct(store, original.record_id, corrected, scope=profile.scope()).outcome.ok
        outcome = expect_invalidation("stale_snapshot", lambda: path.validate(batch, profile.scope()))
        assert outcome.error_type == "AuthorityRejection"
        assert outcome.authority_reason == "stale_snapshot"
    finally:
        store.close()
        path.close()


@pytest.mark.parametrize("error_type", [OSError, BudgetError, ValueError, RuntimeError])
def test_wrapped_operational_failures_never_count_as_revocation(tmp_path, error_type):
    path = profile.build_path("two")
    store = profile.open_store(tmp_path / "store")
    try:
        path.admit(store, profile.spans()[0], scope=profile.scope(), placement="cold")
        pending = path.read(store, "Badge", scope=profile.scope())
        def interrupted(*args, **kwargs):
            raise error_type("operational failure")
        import replica_lab.contextual_memory_v12 as module
        original = module._payload_read_verified
        module._payload_read_verified = interrupted
        try:
            with pytest.raises(ContextualReadFailure) as caught:
                expect_invalidation(
                    "revoked_source", lambda: path.resume(store, pending, scope=profile.scope(), wait_seconds=1))
            assert isinstance(caught.value.__cause__, error_type)
        finally:
            module._payload_read_verified = original
    finally:
        store.close()
        path.close()


def test_direct_authority_and_unexpected_errors_preserve_contract(tmp_path):
    outcome = expect_invalidation("revoked_source", lambda: (_ for _ in ()).throw(
        AuthorityRejection("revoked_source", "revoked")))
    assert outcome.error_type == "AuthorityRejection"
    with pytest.raises(InvalidationContractError):
        expect_invalidation("revoked_source", lambda: None)
    with pytest.raises(RuntimeError, match="implementation"):
        expect_invalidation("revoked_source", lambda: (_ for _ in ()).throw(
            RuntimeError("implementation")))
    with pytest.raises(InvalidationContractError):
        expect_invalidation("revoked_source", lambda: (_ for _ in ()).throw(
            AuthorityReadFailure("stale_snapshot", "stale", {})))


def compound_row(family, *, unknown=False):
    facts = {"anchor": "A", "target": "B", "answer": "red",
             "anchor_colour": "blue" if family == COMPOUND_FAMILIES[0] else "red",
             "unrecorded_target": unknown}
    from replica_lab.intrinsic_memory_coverage_v12 import _compound_answer
    return {"world_id": "compound", "rule_family": family, "facts": facts,
            "query_entity": "A", "answer": _compound_answer(facts, family, "A"),
            "chronology": 1}


def test_partner_correction_preserves_unrelated_anchor_color():
    corrected = chronological_correction(compound_row(COMPOUND_FAMILIES[0]), "green")
    assert corrected["answer"] == "green"
    assert corrected["facts"]["anchor_colour"] == "blue"
    assert "The recorded color of B is green" in corrected["text"]
    assert corrected["correction"]["rule_family"] == COMPOUND_FAMILIES[0]


def test_same_colour_correction_updates_both_equality_linked_colors():
    corrected = chronological_correction(compound_row(COMPOUND_FAMILIES[1]), "green")
    assert corrected["facts"]["anchor_colour"] == "green"
    assert corrected["facts"]["answer"] == "green"
    assert "The recorded color of B is green" in corrected["text"]


def test_compound_correction_can_record_previously_unknown_target():
    corrected = chronological_correction(compound_row(COMPOUND_FAMILIES[0], unknown=True), "green")
    assert corrected["answer"] == "green" and not corrected["facts"]["unrecorded_target"]
    assert "No color is recorded for B" not in corrected["text"]


def test_complete_response_assessment_uses_answer_sequence_likelihood():
    response = {"raw_answer": "red red", "token_ids": [7, 7, 1],
                "terminated": True, "truncated": False}
    expected = [torch.full((12,), -20.0), torch.full((12,), -20.0), torch.full((12,), -20.0)]
    for step in expected:
        step[7] = 0.0
        step[1] = -1.0
    other = [torch.full((12,), -20.0), torch.full((12,), -20.0), torch.full((12,), -20.0)]
    for step in other:
        step[9] = 0.0
        step[1] = -1.0
    row = {"answer": "red red", "allowed_stop_token_ids": (1,)}
    result = complete_response_assessment(row, response, control="contextual",
                                          sequence_logits=expected, counterfactual_logits=other)
    assert result["complete_correct"] and result["sequence_evidence_complete"]
    assert result["answer_log_likelihood"] > result["counterfactual_log_likelihood"]
    assert result["answer_minus_counterfactual_log_likelihood"] > 0
    with pytest.raises(ValueError, match="exceed available sequence logits"):
        sequence_log_likelihood(expected[:1], response["token_ids"], (1,))
