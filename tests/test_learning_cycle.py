"""C13 cycle adapter: actual outcome counters become a bounded proposal review."""
from replica_lab.runtime.director import DirectorBudget, LearningDirector
from replica_lab.runtime.learning_cycle import (CycleEvidence, classify_failure,
                                                evidence_from_summary, review_cycle)

def evidence(**changes):
    value=dict(task_id='task',task_family='family',success=False,decisions=9,
      tool_calls=2,prompt_tokens=400,generated_tokens=80,wall_seconds=1.0,
      source_event_ids=('event-1','event-2'),verified_receipts=1,
      malformed_actions=0,false_success_claims=0,repeated_read=True,state_digest='digest')
    value.update(changes);return CycleEvidence(**value)

def test_success_is_no_update_and_repeated_read_maps_to_action():
    assert review_cycle(evidence(success=True)).status=='no_update'
    assert classify_failure(evidence())=='action'
    deferred=review_cycle(evidence(decisions=4),director=LearningDirector(budget=DirectorBudget(max_cost_units=1000)))
    assert deferred.status=='deferred' and deferred.reason=='neural_labels_insufficient'
    proposed=review_cycle(evidence(),director=LearningDirector(budget=DirectorBudget(max_cost_units=1000)))
    assert proposed.status=='proposed' and proposed.proposal.target_kind=='neural'

def test_procedure_failure_can_emit_only_a_proposal():
    decision=review_cycle(evidence(repeated_read=False,tool_calls=3,
                                  source_event_ids=('event-1','event-2')),
                 director=LearningDirector(budget=DirectorBudget(max_cost_units=1000)))
    assert decision.status=='proposed' and decision.proposal is not None
    assert decision.proposal.target_kind=='procedure'

def test_malformed_maps_to_action_and_no_publish_api():
    decision=review_cycle(evidence(decisions=4,malformed_actions=2),director=LearningDirector(budget=DirectorBudget(max_cost_units=1000)))
    assert decision.status=='deferred' and decision.reason=='neural_labels_insufficient'
    assert not hasattr(LearningDirector,'publish')

def test_summary_adapter_uses_actual_counters_only():
    summary={'success':False,'decisions':2,'journal_head':'j1','state_digest':'d1',
      'budget':{'used_tool_calls':1,'used_prompt_tokens':20,'used_generated_tokens':4},
      'trace':[{'action':{'kind':'read'},'decision_wall_seconds':0.1},
               {'action':{'kind':'read'},'decision_wall_seconds':0.2}]}
    value=evidence_from_summary(summary,task_family='f',task_id='t')
    assert value.repeated_read and value.decisions==2 and value.wall_seconds>0
    assert review_cycle(value).status=='deferred'
