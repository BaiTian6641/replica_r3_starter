"""Exact task state authority: goal, claims, obligations, DAG, features, patches."""

import pytest

from replica_lab.architecture.contracts import fingerprint
from replica_lab.runtime.journal import TaskJournal
from replica_lab.runtime.task_graph import (Budget, Claim, ExactTaskState, JournalVerifier,
                                            Obligation, OutcomeRecord, PatchVerifier,
                                            Subgoal, SuccessCondition, TASK_FEATURE_COUNT,
                                            TaskPatch, ActionRecord, apply_patch)

DIGEST = 'a' * 64


def base_state(**changes):
    values = dict(
        task_id='case-1', subject='subject-1', branch='e3', purpose='test', source_epoch=0,
        goal='Make the evaluator pass.', success_conditions=(SuccessCondition('sc-1', 'exit 0'),),
        constraints=(), subgoals=(), claims=(), evidence=(), obligations=(), actions=(),
        outcomes=(), notes=(), budget=Budget(8, 16, 4096, 512), binding={'runtime': 'v1'},
        allowed_tools=('read_file', 'write_file', 'run_test'), revision=0)
    values.update(changes)
    return ExactTaskState(**values)


def patch(op, payload, author='model'):
    return TaskPatch(op, author, payload)


def test_goal_and_success_conditions_are_immutable():
    state = base_state()
    verifier = JournalVerifier([])
    for op, payload in (('set_goal', {'goal': 'other'}),
                        ('set_success_condition', {'condition_id': 'sc-1', 'description': 'x'}),
                        ('edit_constraint', {'constraint_id': 'con-0', 'text': 'x'}),
                        ('set_budget', {'max_decisions': 999}),
                        ('edit_binding', {'runtime': 'evil'})):
        result = apply_patch(state, patch(op, payload), verifier)
        assert result.applied is False
        assert result.reason == 'unknown_op'
        assert result.state is None
    assert state.goal == 'Make the evaluator pass.'
    assert state.digest == base_state().digest


def test_known_claims_require_host_evidence():
    state = base_state()
    verifier = JournalVerifier({DIGEST})
    denied = apply_patch(state, patch('add_claim', {
        'claim_id': 'cl-1', 'text': 'config.json holds a rate', 'status': 'known'}), verifier)
    assert denied.applied is False and denied.reason == 'model_may_not_add_known_claim'

    no_evidence = apply_patch(state, patch('add_claim', {
        'claim_id': 'cl-1', 'text': 'text', 'status': 'known'}, author='host'), verifier)
    assert no_evidence.applied is False
    assert no_evidence.reason == 'known_claim_requires_host_evidence'

    hosted = base_state()
    evidence = apply_patch(hosted, patch('add_evidence', {
        'evidence_id': 'ev-1', 'kind': 'observation', 'digest': DIGEST, 'origin': 'run:test'},
        author='host'), verifier)
    assert evidence.applied is True
    known = apply_patch(evidence.state, patch('add_claim', {
        'claim_id': 'cl-1', 'text': 'config.json holds a rate', 'status': 'known',
        'evidence_ids': ['ev-1']}, author='host'), verifier)
    assert known.applied is True
    assert known.state.claims[0].status == 'known'
    assert known.state.revision == hosted.revision + 2


def test_model_claims_stay_hypothetical_or_worse():
    state = base_state()
    verifier = JournalVerifier(set())
    for status in ('inferred', 'unknown', 'hypothetical'):
        result = apply_patch(state, patch('add_claim', {
            'claim_id': f'cl-{status}', 'text': 'a proposition', 'status': status}), verifier)
        assert result.applied is True
        state = result.state
    assert [c.status for c in state.claims] == ['inferred', 'unknown', 'hypothetical']
    duplicate = apply_patch(state, patch('add_claim', {
        'claim_id': 'cl-inferred', 'text': 'again', 'status': 'inferred'}), verifier)
    assert duplicate.applied is False and duplicate.reason == 'duplicate_claim_id'


def test_obligation_satisfaction_requires_host_receipt():
    state = base_state()
    verifier = JournalVerifier({'f' * 64})
    added = apply_patch(state, patch('add_obligation', {
        'obligation_id': 'ob-1', 'description': 're-run the evaluator after edits'}), verifier)
    assert added.applied is True
    state = added.state
    missing = apply_patch(state, patch('satisfy_obligation', {'obligation_id': 'ob-1'}), verifier)
    assert missing.applied is False and missing.reason == 'obligation_requires_host_receipt'
    unverified = apply_patch(state, patch('satisfy_obligation', {
        'obligation_id': 'ob-1', 'receipt_digest': '0' * 64}), verifier)
    assert unverified.applied is False
    satisfied = apply_patch(state, patch('satisfy_obligation', {
        'obligation_id': 'ob-1', 'receipt_digest': 'f' * 64}, author='host'), verifier)
    assert satisfied.applied is True
    assert satisfied.state.obligations[0].state == 'satisfied'
    again = apply_patch(satisfied.state, patch('satisfy_obligation', {
        'obligation_id': 'ob-1', 'receipt_digest': 'f' * 64}), verifier)
    assert again.applied is False and again.reason == 'obligation_already_satisfied'


def test_add_evidence_is_host_authority():
    state = base_state()
    result = apply_patch(state, patch('add_evidence', {
        'evidence_id': 'ev-1', 'kind': 'receipt', 'digest': DIGEST, 'origin': 'run:test'}), JournalVerifier(set()))
    assert result.applied is False and result.reason == 'host_authority_required'
    bad_digest = apply_patch(state, patch('add_evidence', {
        'evidence_id': 'ev-1', 'kind': 'receipt', 'digest': 'nothex', 'origin': 'run:test'},
        author='host'), JournalVerifier(set()))
    assert bad_digest.applied is False and bad_digest.reason == 'invalid_evidence_digest'


def test_subgoal_dag_rejects_cycles_and_bad_edges():
    state = base_state()
    verifier = JournalVerifier(set())
    first = apply_patch(state, patch('add_subgoal', {
        'subgoal_id': 'sg-a', 'description': 'inspect the config'}), verifier)
    assert first.applied is True
    state = first.state
    second = apply_patch(state, patch('add_subgoal', {
        'subgoal_id': 'sg-b', 'description': 'fix the value', 'depends_on': ['sg-a']}), verifier)
    assert second.applied is True
    state = second.state
    self_dep = apply_patch(state, patch('add_subgoal', {
        'subgoal_id': 'sg-c', 'description': 'x', 'depends_on': ['sg-c']}), verifier)
    assert self_dep.applied is False and self_dep.reason == 'unknown_subgoal_dependency'
    with pytest.raises(ValueError):
        Subgoal('sg-c', 'x', depends_on=('sg-c',))
    unknown = apply_patch(state, patch('add_subgoal', {
        'subgoal_id': 'sg-c', 'description': 'x', 'depends_on': ['missing']}), verifier)
    assert unknown.applied is False and unknown.reason == 'unknown_subgoal_dependency'
    forward = apply_patch(state, patch('add_subgoal', {
        'subgoal_id': 'sg-a2', 'description': 'y', 'depends_on': ['sg-zz']}), verifier)
    assert forward.applied is False
    assert {s.subgoal_id for s in state.subgoals} == {'sg-a', 'sg-b'}


def test_subgoal_state_transitions_are_bounded():
    state = base_state(subgoals=())
    verifier = JournalVerifier(set())
    added = apply_patch(state, patch('add_subgoal', {
        'subgoal_id': 'sg-a', 'description': 'inspect'}), verifier)
    state = added.state
    active = apply_patch(state, patch('set_subgoal_state', {
        'subgoal_id': 'sg-a', 'state': 'active'}), verifier)
    assert active.applied is True
    done = apply_patch(active.state, patch('set_subgoal_state', {
        'subgoal_id': 'sg-a', 'state': 'done'}), verifier)
    assert done.applied is True
    reopened = apply_patch(done.state, patch('set_subgoal_state', {
        'subgoal_id': 'sg-a', 'state': 'open'}), verifier)
    assert reopened.applied is False and reopened.reason == 'invalid_subgoal_transition'
    missing = apply_patch(done.state, patch('set_subgoal_state', {
        'subgoal_id': 'nope', 'state': 'open'}), verifier)
    assert missing.applied is False and missing.reason == 'unknown_subgoal'


def test_feature_layout_is_bounded_and_actual():
    state = base_state()
    features = state.features()
    assert len(features) == TASK_FEATURE_COUNT == 24
    assert all(0.0 <= value <= 1.0 for value in features)
    aged = base_state(revision=40).features()
    assert aged[0] == 1.0
    spent = base_state().spend(decisions=2).features()
    assert spent[16] == pytest.approx(0.25)
    assert spent[23] == pytest.approx(0.75)
    token_used = base_state().spend(prompt_tokens=4096, generated_tokens=512).features()
    assert token_used[18] == pytest.approx(1.0)
    failed = base_state().with_action(ActionRecord(
        action_id='act-1', kind='read', args={}, state='attempted', revision=0,
        idempotency_key='k1')).with_outcome(OutcomeRecord(
            outcome_id='out-1', action_id='act-1', status='failed', actual=True,
            result={}, revision=1))
    assert failed.features()[19] == 1.0
    assert failed.features()[20] == 0.0


def test_task_graph_projection_and_summary():
    state = base_state()
    graph = state.task_graph()
    assert graph.task_id == 'case-1'
    assert graph.goal == state.goal
    assert graph.success_conditions == ('sc-1',)
    assert graph.state_digest == state.digest
    assert graph.budget_remaining == 8
    bounded = state.summary_text(max_chars=80)
    assert len(bounded) <= 80
    assert 'GOAL' in bounded


def test_state_roundtrip_keeps_digest():
    state = base_state().spend(decisions=1)
    restored = ExactTaskState.from_dict(state.to_dict())
    assert restored.digest == state.digest
    assert restored.budget == state.budget


def test_patch_denials_are_recorded_not_silent(tmp_path):
    journal = TaskJournal.open(tmp_path / 'task.sqlite', {
        'task_id': 'case-1', 'subject': 'subject-1', 'branch': 'e3', 'purpose': 'test',
        'source_epoch': 0,
        'goal_digest': fingerprint('Make the evaluator pass.'),
        'component_versions': {'runtime': 'v1'}})
    state = base_state()
    journal.initialize(state)
    result = apply_patch(state, patch('set_goal', {'goal': 'evil'}), JournalVerifier([]))
    assert result.applied is False
    journal.append('patch', {'patch': result.patch.to_dict(),
                             'result': {'applied': result.applied, 'reason': result.reason}})
    snapshot = journal.load()
    patch_events = [event for event in journal.events() if event.kind == 'patch']
    assert len(patch_events) == 1
    assert patch_events[0].payload['result'] == {'applied': False, 'reason': 'unknown_op'}
    assert snapshot.state.digest == state.digest


def test_action_and_outcome_lifecycle():
    state = base_state()
    record = ActionRecord(action_id='act-1', kind='read', args={'path': 'config.json'},
                          state='proposed', revision=0, idempotency_key='k1')
    state = state.with_action(record)
    assert state.actions[0].state == 'proposed'
    state = state.with_action_state('act-1', 'attempted')
    outcome = OutcomeRecord(outcome_id='out-1', action_id='act-1', status='pending_input',
                            actual=True, result={'question': 'which file?'}, revision=state.revision)
    state = state.with_outcome(outcome)
    assert state.actions[0].state == 'attempted'
    assert state.paused() is True
    done = OutcomeRecord(outcome_id='out-2', action_id='act-1', status='completed',
                         actual=True, result={'sha256': DIGEST}, revision=state.revision)
    state = state.with_outcome(done)
    assert state.actions[0].state == 'completed'
    assert state.paused() is False
    stale = OutcomeRecord(outcome_id='out-3', action_id='act-1', status='completed',
                          actual=True, result={}, revision=state.revision + 5)
    with pytest.raises(ValueError):
        state.with_outcome(stale)
    with pytest.raises(ValueError):
        OutcomeRecord(outcome_id='out-4', action_id='act-1', status='completed',
                      actual=False, result={}, revision=state.revision)


def test_budget_is_enforced_before_spend():
    budget = Budget(2, 1, 100, 50)
    assert budget.allows(decisions=1, tool_calls=1) is True
    spent = budget.spend(decisions=2, tool_calls=1, prompt_tokens=100, generated_tokens=50)
    assert spent.allows(decisions=1) is False
    with pytest.raises(ValueError):
        spent.spend(decisions=1)
    with pytest.raises(ValueError):
        Budget(-1, 0, 0, 0)


def test_model_cannot_revise_known_but_host_can_with_new_evidence():
    verifier = JournalVerifier({DIGEST, 'b' * 64})
    state = base_state()
    state = apply_patch(state, patch('add_evidence', {
        'evidence_id': 'ev-1', 'kind': 'observation', 'digest': DIGEST,
        'origin': 'read:first'}, author='host'), verifier).state
    known = apply_patch(state, patch('add_claim', {
        'claim_id': 'cl-known', 'text': 'the rate is wrong', 'status': 'known',
        'evidence_ids': ['ev-1']}, author='host'), verifier)
    assert known.applied is True
    state = known.state
    model = apply_patch(state, patch('revise_claim', {
        'claim_id': 'cl-known', 'text': 'model says otherwise', 'status': 'unknown'}),
        verifier)
    assert model.applied is False and model.reason == 'known_claim_immutable'
    assert state.claim_map()['cl-known'].text == 'the rate is wrong'
    hosted = apply_patch(state, patch('add_evidence', {
        'evidence_id': 'ev-2', 'kind': 'receipt', 'digest': 'b' * 64,
        'origin': 'run:second'}, author='host'), verifier)
    assert hosted.applied is True
    revised = apply_patch(hosted.state, patch('revise_claim', {
        'claim_id': 'cl-known', 'text': 'the rate is still wrong', 'status': 'known',
        'evidence_ids': ['ev-2']}, author='host'), verifier)
    assert revised.applied is True
    claim = revised.state.claim_map()['cl-known']
    assert claim.status == 'known' and claim.evidence_ids == ('ev-2',)
    assert claim.text == 'the rate is still wrong'


def test_revise_note_and_claim_rules():
    state = base_state()
    verifier = JournalVerifier(set())
    noted = apply_patch(state, patch('note', {'text': 'start from config.json'}), verifier)
    assert noted.applied is True and noted.state.notes == ('start from config.json',)
    long_note = apply_patch(noted.state, patch('note', {'text': 'x' * 600}), verifier)
    assert long_note.applied is False and long_note.reason.startswith('invalid_payload')
    claimed = apply_patch(noted.state, patch('add_claim', {
        'claim_id': 'cl-1', 'text': 'the rate is wrong', 'status': 'inferred'}), verifier)
    revised = apply_patch(claimed.state, patch('revise_claim', {
        'claim_id': 'cl-1', 'text': 'the rate must change', 'status': 'unknown'}), verifier)
    assert revised.applied is True
    assert revised.state.claims[0].status == 'unknown'
    immutable = apply_patch(claimed.state, patch('add_claim', {
        'claim_id': 'cl-host', 'text': 'host fact', 'status': 'known',
        'evidence_ids': ['ev-1']}, author='host'), JournalVerifier({DIGEST}))
    assert immutable.applied is False  # evidence ev-1 does not exist in state
