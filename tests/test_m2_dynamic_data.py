"""Fresh M2 data chronology, split and correction checks."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/m2_dynamic'


def _rows(name):
    return [json.loads(line) for line in (DATA / name).read_text().splitlines() if line.strip()]


def test_m2_splits_are_disjoint_and_two_observations_per_task():
    queries = _rows('queries.jsonl')
    groups = {row['source_group'] for row in queries}
    assert len(groups) == len(queries)
    assert {row['split'] for row in queries} == {'train', 'dev', 'heldout', 'retention'}
    for row in queries:
        assert len(row['observations']) >= 1
        assert row['query'] not in '\n'.join(row['observations'])
        assert row['expected_answer'] not in row['query']
        if row['split'] != 'retention':
            assert len(row['observations']) == 2
            assert row['correction_observation'] != row['observations'][0]
            assert row['corrected_answer'] != row['expected_answer']


def test_m2_correction_has_newer_timestamp_and_parent():
    records = _rows('records.jsonl')
    by_id = {row['record_id']: row for row in records}
    for row in records:
        if row['record_id'].endswith('_correction'):
            parent = by_id[row['supersedes']]
            assert row['observed_at'] > parent['observed_at']
            assert row['authenticity'] == 'verified_source'


def test_m2_train_and_heldout_entities_and_values_are_disjoint():
    queries = _rows('queries.jsonl')
    train_names = {row['query'].split('project ')[-1].rstrip('?') for row in queries
                   if row['split'] == 'train'}
    heldout_names = {row['query'].split('project ')[-1].rstrip('?') for row in queries
                     if row['split'] == 'heldout'}
    assert not train_names & heldout_names
    train_values = {row['expected_answer'] for row in queries if row['split'] == 'train'}
    heldout_values = {row['expected_answer'] for row in queries if row['split'] == 'heldout'}
    assert not train_values & heldout_values
