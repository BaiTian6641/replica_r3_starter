"""Prospective v4 retention diversity and reference acceptance contracts."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from replica_lab.memory_reliability import GENERAL, GENERAL_V2, audit_dataset
from replica_lab.reliability_scoring import metrics_and_checks
from test_memory_reliability import synthetic_report

ROOT = Path(__file__).resolve().parents[1]
EPISODES = json.loads((ROOT / 'data/memory_reliability_v4/episodes.json').read_text())


def test_v4_has_fresh_families_and_diverse_general_retention():
    audit = audit_dataset(EPISODES, require_novel_values=True,
                          general_oracle='general_v2', min_unique_general=16)
    assert audit['episodes'] == 32 and audit['families'] == 16
    assert audit['unique_general_questions'] == 16
    assert audit['unique_general_questions_by_split'] == {'train': 8, 'dev': 8, 'heldout': 8}
    old_queries = {query for pool in GENERAL.values() for query, _ in pool}
    new_queries = {q['query'] for e in EPISODES for q in e['general_questions']}
    assert len(new_queries) == 24
    assert not new_queries & old_queries
    assert all(len(pool) == 8 for pool in GENERAL_V2.values())


def test_v4_general_oracle_and_diversity_mutations_are_rejected():
    episodes = deepcopy(EPISODES)
    episodes[0]['general_questions'][0]['expected_answer'] = 'wrong'
    with pytest.raises(ValueError, match='General-answer oracle'):
        audit_dataset(episodes, general_oracle='general_v2')
    with pytest.raises(ValueError, match='Insufficient unique'):
        audit_dataset(EPISODES, general_oracle='general_v2', min_unique_general=17)


def test_v4_counterfactual_general_questions_remain_aligned():
    by_family = {}
    for episode in EPISODES:
        by_family.setdefault(episode['family_id'], []).append(
            [(q['query_id'], q['query'], q['expected_answer']) for q in episode['general_questions']])
    assert len(by_family) == 16
    assert all(len(variants) == 2 and variants[0] == variants[1] for variants in by_family.values())


def test_prospective_reference_non_regression_does_not_require_full_context_parity():
    report, decode = synthetic_report(EPISODES)
    fast_rows = [row for row in report['candidate'] if row['arm'] == 'fast']
    report['candidate'].extend(dict(row, arm='selective_context') for row in fast_rows)
    target = next(row for row in report['candidate']
                  if row['arm'] == 'full_context' and row['kind'] == 'fact' and row['split'] == 'dev')
    disabled = next(row for row in report['candidate']
                    if row['arm'] == 'disabled' and row['case_id'] == target['case_id'])
    target['answer'] = 'wrong'
    target['correct'] = False
    target['token_ids'] = disabled['token_ids']
    protocol = {'exact_reference': {'enabled': True}, 'reference_arm': 'selective_context',
                'reference_mode': 'non_regression',
                'evaluation_arms': {'candidate': ['fast', 'disabled', 'shuffled', 'last_chunk',
                                                    'full_context', 'selective_context']}}
    _, checks = metrics_and_checks(report, EPISODES, decode, protocol)
    assert checks['reference_full_context_non_regression']
    assert 'reference_full_context_parity' not in checks
    parity_protocol = dict(protocol, reference_mode='parity')
    _, parity_checks = metrics_and_checks(report, EPISODES, decode, parity_protocol)
    assert not parity_checks['reference_full_context_parity']
