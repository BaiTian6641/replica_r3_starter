"""M2b shared-value chronology and copy-generalization data checks."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/m2_shared'


def _rows(name):
    return [json.loads(line) for line in (DATA / name).read_text().splitlines() if line.strip()]


def test_shared_values_and_new_bindings():
    queries = _rows('queries.jsonl')
    train_values = {row['expected_answer'] for row in queries if row['split'] == 'train'}
    heldout_values = {row['expected_answer'] for row in queries if row['split'] == 'heldout'}
    assert len(train_values) == 6
    assert heldout_values <= train_values
    assert len({row['source_group'] for row in queries}) == len(queries)
    assert len({row['query'] for row in queries}) == len(queries)
    for row in queries:
        if row['split'] != 'retention':
            assert len(row['observations']) == 2
            assert row['corrected_answer'] != row['expected_answer']


def test_correction_is_newer_and_source_separated():
    records = _rows('records.jsonl')
    by_id = {row['record_id']: row for row in records}
    for row in records:
        if row['record_id'].endswith('_correction'):
            parent = by_id[row['supersedes']]
            assert row['observed_at'] > parent['observed_at']
            assert row['authenticity'] == 'verified_source'
