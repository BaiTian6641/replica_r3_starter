"""C13 deterministic director: bounded proposals, deferral and no authority."""
import pytest

from replica_lab.runtime.director import (DirectorBudget, LearningDirector,
                                          OutcomeSummary)


def outcome(failure_class='retrieval', *, attempts=4, successes=1,
            sources=('ev-1', 'ev-2'), receipts=0, cost=1.0, role='train'):
    return OutcomeSummary('fresh-family', failure_class, attempts, successes,
                          tuple(sources), role=role, verified_receipts=receipts,
                          cost_units=cost)


def test_repeated_retrieval_failure_proposes_only_a_bounded_record_candidate():
    decision = LearningDirector().review([outcome(), outcome(sources=('ev-2', 'ev-3'))],
                                         data_manifest={'train': 'fresh-family'})
    assert decision.status == 'proposed' and decision.proposal is not None
    proposal = decision.proposal
    assert proposal.target_kind == 'record' and proposal.target_id == 'memory-selection'
    assert proposal.source_event_ids == ('ev-1', 'ev-2', 'ev-3')
    assert proposal.optimizer == 'none'
    assert proposal.parent_manifest_id is None
    assert decision.diagnostic['retrieval']['failures'] == 6


def test_procedure_requires_real_receipts_and_neural_requires_labels():
    director = LearningDirector()
    no_receipts = director.review([outcome('procedure', attempts=3, successes=0)])
    assert no_receipts.status == 'deferred' and no_receipts.proposal is None
    with_receipts = director.review([outcome('procedure', attempts=3, successes=0,
                                             receipts=2)])
    assert with_receipts.status == 'proposed'
    assert with_receipts.proposal.target_kind == 'procedure'
    few = director.review([outcome('action', attempts=4, successes=0)])
    assert few.status == 'deferred' and few.reason == 'neural_labels_insufficient'
    enough = director.review([outcome('action', attempts=8, successes=1)])
    assert enough.status == 'proposed'
    assert enough.proposal.target_kind == 'neural'
    assert enough.proposal.optimizer == 'muon8bit+adamw_fp32'


def test_no_repeat_budget_and_audit_are_not_silently_trained():
    director = LearningDirector()
    assert director.review([]).status == 'no_update'
    assert director.review([outcome('retrieval', attempts=1, successes=0)]).status == 'no_update'
    assert director.review([outcome('insufficient_budget')]).status == 'no_update'
    assert director.review([outcome('implementation')]).status == 'no_update'
    constrained = LearningDirector(budget=DirectorBudget(max_cost_units=0.5))
    assert constrained.review([outcome(cost=1.0)]).status == 'deferred'
    audited = director.review([outcome()], data_manifest={'audit': 'forbidden'})
    assert audited.status == 'no_update' and audited.reason == 'audit_or_gold_forbidden'


def test_director_rejects_unverified_roles_and_does_not_publish():
    with pytest.raises(ValueError):
        outcome(role='audit')
    with pytest.raises(ValueError):
        OutcomeSummary('family', 'retrieval', 2, 0, ())  # missing source evidence
    decision = LearningDirector().review([outcome('action', attempts=8, successes=0)])
    assert decision.status == 'proposed'
    # The director has no registry/publisher dependency and exposes no execute API.
    assert not hasattr(LearningDirector, 'publish')
    assert not hasattr(LearningDirector, 'execute')
