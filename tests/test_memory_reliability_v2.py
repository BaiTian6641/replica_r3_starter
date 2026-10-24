"""Fresh-family dataset and independent scoring contracts."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from replica_lab.memory_reliability import audit_dataset, cases
from replica_lab.reliability_scoring import family_metrics, metrics_and_checks
from test_memory_reliability import synthetic_report

ROOT = Path(__file__).resolve().parents[1]
EPISODES = json.loads((ROOT / 'data/memory_reliability_v2/episodes.json').read_text())


def test_fresh_families_query_forms_and_shortcut_bounds():
    audit = audit_dataset(EPISODES, require_novel_values=True)
    assert audit['episodes'] == 32 and audit['families'] == 16
    assert audit['split_families'] == {'train': 8, 'dev': 4, 'heldout': 4}
    assert set(audit['query_forms']) == {0, 1, 2, 3}
    assert set(audit['query_forms'].values()) == {40}
    assert audit['last_line_accuracy'] <= .25 and audit['first_line_accuracy'] <= .25
    assert audit['query_only_best_accuracy'] == .5
    assert set(audit['heldout_values_absent_from_train_and_dev']) == {
        'crimson', 'gold', 'indigo', 'maroon', 'navy', 'silver'}


def test_fresh_cases_have_family_identity_and_identical_counterfactual_queries():
    by_family = {}
    for episode in EPISODES:
        initial, corrected = cases(episode, 'initial'), cases(episode, 'corrected')
        assert len(initial) == 5 and len(corrected) == 7
        assert all(case['family_id'] == episode['family_id'] for case in initial + corrected)
        by_family.setdefault(episode['family_id'], []).append(
            [(case['query'], case['expected_answer']) for case in initial])
    assert len(by_family) == 16
    for variants in by_family.values():
        assert len(variants) == 2
        assert [query for query, _ in variants[0]] == [query for query, _ in variants[1]]
        assert [answer for _, answer in variants[0]] != [answer for _, answer in variants[1]]


def test_fresh_novel_value_overlap_is_rejected():
    episodes = deepcopy(EPISODES)
    old_value = episodes[-1]['facts'][0]['value']
    new_value = episodes[0]['facts'][0]['value']
    episodes[-1]['facts'][0]['value'] = new_value
    episodes[-1]['observations'] = [page.replace(old_value, new_value)
                                      for page in episodes[-1]['observations']]
    with pytest.raises(ValueError, match='overlap'):
        audit_dataset(episodes, require_novel_values=True)


def test_fresh_query_form_mutation_is_rejected():
    episodes = deepcopy(EPISODES)
    episodes[0]['questions'][0]['query'] = 'Give me the launch code for project Cedar1000.'
    with pytest.raises(ValueError, match='Query text'):
        audit_dataset(episodes)


def test_independent_family_metrics_count_families_not_variants():
    report, decode = synthetic_report(EPISODES)
    families = family_metrics(report, EPISODES, decode)
    assert families['independent_family_counts'] == {'dev': 4, 'heldout': 4}
    assert len(families['by_family']['candidate']['fast']['dev']) == 4
    assert len(families['by_family']['candidate']['fast']['heldout']) == 4
    metrics, checks = metrics_and_checks(report, EPISODES, decode)
    assert all(checks.values())
    assert metrics['candidate']['fast']['dev']['fact'] == 1.0
