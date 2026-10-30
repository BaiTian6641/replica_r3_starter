"""v12 durable reservation tests: append-only journal, caps, recovery, deadlines.

These tests use synthetic plans only. They do not load weights, start the
native engine or touch any sealed v11 receipt.
"""
import json
import time

import pytest

from replica_lab.intrinsic_memory_budget_v12 import (
    COUNTERS, FORMAT, LEGACY_COUNTERS, BudgetError, WorkLedger, audit_io, sha_json,
    validate_plan)


def plan(**overrides):
    result = {'a': {'wall_seconds': 60, **{k: 10 for k in COUNTERS}},
              'b': {'wall_seconds': 60, **{k: 5 for k in COUNTERS}}}
    for stage, caps in overrides.items():
        result[stage].update(caps)
    return result


def ledger(tmp_path, *, deadline=None):
    now = time.monotonic()
    return WorkLedger.create(tmp_path/'work_ledger.jsonl', identity='packet-sha',
                             plan=plan(), deadline=now+(deadline or 600))


def test_v12_vocabulary_covers_all_v5_counters_and_new_source_work():
    for name in ('forwards', 'backwards', 'updates', 'input_tokens', 'generated_tokens',
                 'model_loads', 'artifact_bytes', 'source_encodes', 'retrievals', 'writes'):
        assert name in COUNTERS
    assert LEGACY_COUNTERS <= set(COUNTERS)
    for name in ('source_feature_forwards', 'tokenizer_calls', 'prompt_tokenizer_calls',
                 'source_token_bytes', 'cache_misses', 'cache_evictions',
                 'cache_recomputations', 'artifact_read_bytes', 'restart_restores'):
        assert name in COUNTERS
    assert len(set(COUNTERS)) == len(COUNTERS)


def test_plan_requires_every_counter_and_positive_wall():
    with pytest.raises(BudgetError): validate_plan({})
    with pytest.raises(BudgetError): validate_plan({'a': {'wall_seconds': 1}})
    incomplete = plan(); del incomplete['a']['cache_misses']
    with pytest.raises(BudgetError): validate_plan(incomplete)
    negative = plan(); negative['a']['writes'] = -1
    with pytest.raises(BudgetError): validate_plan(negative)
    zero_wall = plan(); zero_wall['a']['wall_seconds'] = 0
    with pytest.raises(BudgetError): validate_plan(zero_wall)
    validate_plan(plan())


def test_reservation_precedes_work_and_refuses_over_cap_before_dispatch(tmp_path):
    log = ledger(tmp_path)
    log.enter('a')
    log.reserve('a', 'source_forward', source_feature_forwards=10, cache_misses=10)
    # The cap is consumed by the reservation itself; a second dispatch attempt
    # is refused *before* it could run, which is the whole point of reserving.
    with pytest.raises(BudgetError, match='exceeded before dispatch'):
        log.reserve('a', 'source_forward', source_feature_forwards=1)
    assert log.snapshot()['reserved']['source_feature_forwards'] == 10


def test_failed_work_remains_charged_and_actual_cannot_exceed_reservation(tmp_path):
    log = ledger(tmp_path)
    log.enter('a')
    token = log.reserve('a', 'source_forward', forwards=2, input_tokens=8)
    log.finish(token, status='failed', detail={'error': 'ContextualMemoryError'})
    with pytest.raises(BudgetError, match='already terminal'): log.finish(token)
    overshoot = log.reserve('a', 'source_forward', forwards=1)
    with pytest.raises(BudgetError, match='exceeds reservation'):
        log.finish(overshoot, actual={'forwards': 2})
    log.finish(overshoot, actual={'forwards': 1})
    snapshot = log.snapshot()
    assert snapshot['reserved']['forwards'] == 3
    assert snapshot['events'][0]['status'] == 'failed'
    assert audit_io(log)['failed_or_interrupted_events'] == 1


def test_reservation_outside_active_stage_or_with_unknown_counter_is_refused(tmp_path):
    log = ledger(tmp_path)
    with pytest.raises(BudgetError, match='not active'):
        log.reserve('a', 'source_tokenize', source_encodes=1)
    log.enter('a')
    with pytest.raises(BudgetError, match='invalid reserved counters'):
        log.reserve('a', 'source_tokenize', unknown_counter=1)
    with pytest.raises(BudgetError, match='reservation operation'):
        log.reserve('a', '')


def test_single_active_stage_and_no_stage_reuse(tmp_path):
    log = ledger(tmp_path)
    log.enter('a')
    with pytest.raises(BudgetError, match='another stage is active'): log.enter('b')
    log.leave('a', status='completed')
    with pytest.raises(BudgetError, match='already consumed'): log.enter('a')
    with pytest.raises(BudgetError, match='already closed'): log.leave('a')


def test_stage_with_uncertain_work_cannot_be_closed_without_recovery(tmp_path):
    log = ledger(tmp_path)
    log.enter('a')
    log.reserve('a', 'source_forward', forwards=1)
    with pytest.raises(BudgetError, match='uncertain work'): log.leave('a', status='failed')
    recovered = log.recover_interrupted()
    assert recovered == {'charged_events': [0], 'interrupted_stages': ['a'], 'refunded': False}
    snapshot = log.snapshot()
    assert snapshot['events'][0]['status'] == 'failed'
    assert snapshot['stages']['a']['status'] == 'interrupted'
    assert snapshot['reserved']['forwards'] == 1
    log.enter('b')  # a second stage is available after explicit recovery
    assert 'b' in log.snapshot()['stages']


def test_journal_is_append_only_and_logical_bytes_equal_appended_bytes(tmp_path):
    log = ledger(tmp_path)
    log.enter('a')
    first = log.path.stat().st_size
    token = log.reserve('a', 'source_tokenize', source_encodes=1, input_tokens=8)
    second = log.path.stat().st_size
    log.finish(token, actual={'source_encodes': 1, 'input_tokens': 6})
    third = log.path.stat().st_size
    assert first < second < third
    report = audit_io(log)
    assert report['logical_appended_bytes'] == log.path.stat().st_size
    assert report['physical_write_bytes_measured'] is False
    assert report['append_calls'] == len(log.path.read_text().splitlines())
    # A later append never rewrites an earlier line.
    prefix = log.path.read_bytes()[:third]
    log.leave('a')
    assert log.path.read_bytes()[:third] == prefix


def test_totals_and_deadlines_survive_a_fresh_process_view(tmp_path):
    log = ledger(tmp_path, deadline=300)
    log.enter('a')
    token = log.reserve('a', 'source_forward', forwards=1)
    log.finish(token, actual={'forwards': 1})
    reopened = WorkLedger(log.path, identity='packet-sha')
    snapshot = reopened.snapshot()
    assert snapshot['reserved']['forwards'] == 1
    assert snapshot['format'] == FORMAT
    state = reopened.deadline_state(snapshot)
    assert 0 < state['outer_remaining_seconds'] <= 300
    assert 0 < state['stage_remaining_seconds']['a'] <= 60
    assert state['clock_reanchored'] is False
    # The plan and its digest are immutable once written.
    with pytest.raises(BudgetError, match='identity differs'):
        WorkLedger(log.path, identity='other-packet').snapshot()


def test_tampered_journal_is_refused(tmp_path):
    log = ledger(tmp_path)
    log.enter('a')
    log.reserve('a', 'source_tokenize', source_encodes=1)
    lines = log.path.read_text().splitlines()
    # The third line is the reservation; reassigning it to a stage that exists
    # but is not active must be refused on replay.
    record = json.loads(lines[2]); record['stage'] = 'b'
    log.path.write_text('\n'.join([lines[0], lines[1], json.dumps(record)])+'\n')
    with pytest.raises(BudgetError, match='outside an active stage'):
        WorkLedger(log.path, identity='packet-sha').snapshot()
    # An over-cap reservation written by hand is rejected on replay, not trusted.
    log2 = ledger(tmp_path/'second')
    log2.enter('a')
    plan_line, enter_line = log2.path.read_text().splitlines()[:2]
    forged = {'t': 'reserve', 'id': 0, 'stage': 'a', 'operation': 'x',
              'reserved': {'forwards': 99}, 'started': time.monotonic(), 'pid': 1}
    log2.path.write_text('\n'.join([plan_line, enter_line, json.dumps(forged)])+'\n')
    with pytest.raises(BudgetError, match='exceeded in journal'):
        WorkLedger(log2.path, identity='packet-sha').snapshot()


def test_torn_tail_is_fail_closed(tmp_path):
    log = ledger(tmp_path)
    log.enter('a')
    log.reserve('a', 'source_tokenize', source_encodes=1)
    payload = log.path.read_bytes()
    log.path.write_bytes(payload + b'{"t":"reserve","id":1,')
    reopened = WorkLedger(log.path, identity='packet-sha')
    snapshot = reopened.snapshot()
    assert snapshot['journal']['torn_tail'] is True
    with pytest.raises(BudgetError, match='torn journal tail'):
        reopened.recover_interrupted()


def test_cached_state_cannot_hide_another_process_append(tmp_path):
    first = ledger(tmp_path)
    assert first.snapshot(cached=True)['reserved']['writes'] == 0
    second = WorkLedger(first.path, identity='packet-sha')
    second.enter('a')
    token = second.reserve('a', 'external_writer', writes=1)
    second.finish(token, actual={'writes': 1})
    # The cached reader must observe the append, because the journal size changed.
    assert first.snapshot(cached=True)['reserved']['writes'] == 1
    assert first.snapshot()['reserved']['writes'] == 1


def test_expired_ledger_cannot_be_entered(tmp_path):
    now = time.monotonic()
    log = WorkLedger.create(tmp_path/'work_ledger.jsonl', identity='packet-sha',
                            plan=plan(), deadline=now+0.05)
    time.sleep(0.06)
    with pytest.raises(BudgetError, match='outer deadline exceeded'): log.enter('a')


def test_create_refuses_existing_journal_and_bad_identity(tmp_path):
    ledger(tmp_path)
    with pytest.raises(BudgetError, match='fresh'):
        WorkLedger.create(tmp_path/'work_ledger.jsonl', identity='packet-sha',
                          plan=plan(), deadline=time.monotonic()+60)
    with pytest.raises(BudgetError, match='identity'):
        WorkLedger.create(tmp_path/'other.jsonl', identity='', plan=plan(),
                          deadline=time.monotonic()+60)
    with pytest.raises(BudgetError, match='deadline'):
        WorkLedger.create(tmp_path/'third.jsonl', identity='packet-sha', plan=plan(),
                          deadline=time.monotonic()-1)


def test_plan_digest_and_format_are_bound(tmp_path):
    log = ledger(tmp_path)
    snapshot = log.snapshot()
    assert snapshot['plan_sha256'] == sha_json(snapshot['plan'])
    assert snapshot['format'] == FORMAT
    assert snapshot['reserved_wall_seconds'] == 0
    log.enter('a')
    assert log.snapshot()['reserved_wall_seconds'] == 60
