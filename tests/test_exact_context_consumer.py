"""Caller-bound canonical evidence checks at the real request construction boundary."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from replica_lab.memory_reliability import build_episodes, cases
from replica_lab.runtime.exact_memory import (EXACT_SELECTION_POLICY, ExactMemoryReference,
                                              episode_records)
from test_exact_memory import Tokenizer, _runner


@pytest.fixture
def consumer(tmp_path):
    episode = next(e for e in build_episodes() if e['split'] == 'dev')
    reference = ExactMemoryReference(tmp_path / 'sources.sqlite')
    reference.admit_episode(episode)
    runner = _runner()
    session = SimpleNamespace(tokenizer=Tokenizer(), device=torch.device('cpu'))
    case = cases(episode, 'initial')[0]
    try:
        yield runner, session, reference, episode, case
    finally:
        reference.close()


def consume(parts, context, *, case=None, **kwargs):
    runner, session, reference, _, original = parts
    return runner.request_args(session, case or original, evidence=context,
                               selective=True, exact_mode=True, reference=reference, **kwargs)


def test_consumer_accepts_canonical_relevant_absent_and_conflict(consumer):
    runner, _, reference, episode, case = consumer
    relevant = reference.select(case['query'], runner.case_scope(case))
    assert consume(consumer, relevant)['enabled_ports'] == ()
    general = next(q for q in cases(episode, 'corrected') if q['kind'] == 'general')
    absent = reference.select(general['query'], runner.case_scope(general))
    assert absent.status == 'no_evidence'
    consume(consumer, absent, case=general)
    record = relevant.records[0]
    reference.admit((replace(record, record_id='conflict', text=record.text.replace(
        episode['facts'][0]['value'], 'conflicting')),))
    conflict = reference.select(case['query'], runner.case_scope(case))
    assert conflict.status == 'conflict'
    consume(consumer, conflict)


def test_consumer_rejects_genuine_same_query_foreign_branch(consumer):
    runner, _, reference, episode, case = consumer
    records, _ = episode_records(episode)
    reference.admit(tuple(replace(record, record_id='foreign:' + record.record_id,
                                 branch_id='foreign') for record in records))
    foreign = reference.select(case['query'], replace(runner.case_scope(case), branch_id='foreign'))
    assert foreign.enabled
    with pytest.raises(PermissionError, match='scope'):
        consume(consumer, foreign)


@pytest.mark.parametrize('mutation', ['abstention', 'query_route', 'reason', 'span_text',
                                      'span_hash', 'offset', 'source_hash', 'record',
                                      'duplicate_count', 'boolean_duplicate_count', 'policy', 'limit'])
def test_consumer_rejects_noncanonical_context(consumer, mutation):
    runner, _, reference, _, case = consumer
    context = reference.select(case['query'], runner.case_scope(case))
    supplied_case = case
    if mutation == 'abstention':
        context = replace(context, status='no_evidence', records=(), spans=(), reason='invented')
    elif mutation == 'query_route':
        supplied_case = dict(case, query='What is 2 plus 3?')
        context = replace(context, selection=replace(context.selection, query=supplied_case['query']))
    elif mutation == 'reason':
        context = replace(context, reason='invented relevant reason')
    elif mutation == 'record':
        context = replace(context, records=(replace(context.records[0], text='invented record'),))
    elif mutation in {'duplicate_count', 'boolean_duplicate_count'}:
        context = replace(context, duplicates_removed=True if mutation.startswith('boolean') else 1)
    elif mutation == 'policy':
        context = replace(context, selection_policy='unsupported.v1')
    elif mutation == 'limit':
        context = replace(context, selection_limit=1)
    else:
        field, value = {'span_text': ('text', 'invented'), 'span_hash': ('span_hash', '0' * 64),
                        'offset': ('start', 1), 'source_hash': ('source_hash', '0' * 64)}[mutation]
        context = replace(context, spans=(replace(context.spans[0], **{field: value}),))
    with pytest.raises((ValueError, PermissionError)):
        consume(consumer, context, case=supplied_case)


def test_consumer_binds_limit_policy_and_inventory_order(consumer):
    runner, _, reference, episode, case = consumer
    case = dict(case, query='Project ' + episode['facts'][0]['entity'])
    context = reference.select(case['query'], runner.case_scope(case), limit=1)
    consume(consumer, context, case=case, evidence_limit=1)
    with pytest.raises(ValueError, match='limit'):
        consume(consumer, replace(context, selection_limit=True), case=case, evidence_limit=1)
    with pytest.raises(ValueError, match='limit'):
        consume(consumer, context, case=case)
    with pytest.raises(ValueError, match='policy'):
        consume(consumer, context, case=case, evidence_limit=1, evidence_policy='unsupported.v1')
    context = reference.select(case['query'], runner.case_scope(case))
    assert len(context.records) > 1
    with pytest.raises(ValueError, match='inventory/order'):
        consume(consumer, replace(context, records=context.records[::-1], spans=context.spans[::-1]), case=case)


def test_consumer_rejects_stale_correction_and_revocation(consumer):
    runner, _, reference, episode, case = consumer
    before = reference.select(case['query'], runner.case_scope(case))
    reference.correct_episode(episode)
    with pytest.raises(ValueError, match='stale'):
        consume(consumer, before)
    corrected_case = cases(episode, 'corrected')[0]
    corrected = reference.select(corrected_case['query'], runner.case_scope(corrected_case))
    consume(consumer, corrected, case=corrected_case)
    _, correction = episode_records(episode)
    reference.revoke(correction.supersedes, authorized_subject='memory_reliability',
                     authorized_branch=episode['episode_id'])
    with pytest.raises(ValueError, match='stale'):
        consume(consumer, corrected, case=corrected_case)


def test_consumer_deduplicates_canonically_and_requires_reference(consumer):
    runner, session, reference, _, case = consumer
    context = reference.select(case['query'], runner.case_scope(case))
    reference.admit((replace(context.records[0], record_id='duplicate'),))
    context = reference.select(case['query'], runner.case_scope(case))
    assert context.duplicates_removed == 1
    consume(consumer, context)
    with pytest.raises(ValueError, match='authoritative reference'):
        runner.request_args(session, case, evidence=context)
