import copy
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_spark_intrinsic_supported_values_v8 import CASES, worlds, validate_corpus


def test_fresh_entities_and_supported_answer_vocabulary():
    fit, dev = worlds('fit'), worlds('development')
    validate_corpus(fit)
    validate_corpus(dev)
    names = lambda doc: {name for r in doc['rows'] for name in
                        (r['facts'][0]['entity'], r['query_entity'])}
    assert names(fit).isdisjoint(names(dev))
    from prepare_spark_intrinsic_explicit_evidence_v6 import ENTITIES
    previous = set().union(*map(set, ENTITIES.values()))
    assert (names(fit) | names(dev)).isdisjoint(previous)
    answers = lambda doc: {r['answer'] for r in doc['rows']}
    assert answers(dev) == answers(fit) == {'red', 'blue', 'green', 'gold', 'unknown'}
    assert Counter(r['answer'] for r in fit['rows'])['unknown'] == 2
    assert Counter(r['answer'] for r in dev['rows'])['unknown'] == 1


def test_counterfactual_answers_depend_on_memory_not_question():
    rows = worlds('fit')['rows']
    for i in range(0, 8, 2):
        a, b = rows[i:i+2]
        assert a['question'] == b['question'] and a['text'] != b['text']
        assert a['answer'] != b['answer']
    for row in rows:
        missing = row['query_entity'] not in [f['entity'] for f in row['facts']]
        assert (row['answer'] == 'unknown') == missing
        assert row['answer'] != 'unknown' or 'unknown' not in row['text']


@pytest.mark.parametrize('field', ['answer', 'text', 'question', 'query_entity', 'world_id'])
def test_source_oracle_or_identity_tampering_is_refused(field):
    doc = worlds('fit')
    doc['rows'][0][field] += ' altered'
    with pytest.raises(ValueError):
        validate_corpus(doc)


def test_split_incompleteness_is_refused():
    doc = worlds('development')
    doc['rows'].pop()
    with pytest.raises(ValueError):
        validate_corpus(doc)


def test_native_protocol_keeps_v7_budgets_and_criteria():
    from replica_lab.intrinsic_memory_exposure_v7 import validate_protocol
    old = json.loads((ROOT / 'configs/spark_intrinsic_exposure_curve_v7.json').read_text())
    new = json.loads((ROOT / 'configs/spark_intrinsic_supported_values_v8.json').read_text())
    validate_protocol(new)
    for key in ('training', 'generation', 'competence', 'fit_gate', 'utility',
                'source', 'source_execution', 'environment', 'fit_probe_prefixes'):
        assert old[key] == new[key]
    assert old['data']['calibration'] == new['data']['calibration']
    assert old['data']['development'] != new['data']['development']
    for arm in old['arms']:
        for key in ('budget', 'total_reservations', 'sites', 'resource_limits', 'wall_seconds'):
            assert old['arms'][arm][key] == new['arms'][arm][key]
