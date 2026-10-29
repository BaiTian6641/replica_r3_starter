import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('explicit_v6', ROOT / 'scripts/prepare_spark_intrinsic_explicit_evidence_v6.py')
v6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v6)


@pytest.mark.parametrize('split', ['calibration', 'fit', 'development'])
def test_explicit_facts_and_independent_entity_oracle(split):
    doc = v6.worlds(split)
    v6.validate_corpus(doc)
    for row in doc['rows']:
        # Independently compute the answer from a table, not the expected label
        # or a generated explanation. The phrase binds color, not material.
        table = {f['entity']: f['color'] for f in row['facts']}
        assert row['answer'] == table.get(row['query_entity'], 'unknown')
        assert f"recorded color of {row['facts'][0]['entity']} is" in row['text']
        assert len(row['text'].encode()) <= 256
    if split != 'fit':
        row = doc['rows'][-1]
        assert row['query_entity'] not in row['text']
        assert row['answer'] == 'unknown'


def test_counterfactual_pairs_have_identical_query_and_opposite_evidence():
    rows = v6.worlds('fit')['rows']
    for a, b in zip(rows[::2], rows[1::2]):
        assert a['question'] == b['question']
        assert a['facts'][0]['entity'] == b['facts'][0]['entity']
        assert a['facts'][0]['color'] != b['facts'][0]['color']
        assert a['answer'] != b['answer']


def test_entities_disjoint_across_splits_without_claiming_new_rule_families():
    splits = list(v6.ENTITIES)
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            assert set(v6.ENTITIES[left]).isdisjoint(v6.ENTITIES[right])
    assert all(not v6.worlds(s)['independent_rule_family_claim'] for s in splits)


@pytest.mark.parametrize('mutation', ['wrong_oracle', 'missing_referent', 'material_only', 'partial_worlds'])
def test_ambiguous_or_inconsistent_corpus_refused(mutation):
    doc = copy.deepcopy(v6.worlds('calibration'))
    if mutation == 'wrong_oracle':
        doc['rows'][0]['answer'] = 'silver'
    elif mutation == 'missing_referent':
        doc['rows'][-1]['question'] = 'What color is the missing signal?'
    elif mutation == 'material_only':
        doc['rows'][0]['text'] = 'Runa is copper.'
    else:
        doc['rows'].pop()
    with pytest.raises(ValueError):
        v6.validate_corpus(doc)


def test_preparation_refuses_overwrite(tmp_path):
    (tmp_path / 'configs').mkdir()
    (tmp_path / v6.CONFIG).write_text('sealed')
    with pytest.raises(ValueError, match='no overwrite'):
        v6.prepare(tmp_path)
    assert (tmp_path / v6.CONFIG).read_text() == 'sealed'
