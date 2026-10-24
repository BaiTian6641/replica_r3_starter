"""M3 retention-repair split, lineage and audit-isolation checks."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/m3_repair'


def _rows(name):
    return [json.loads(line) for line in (DATA / name).read_text().splitlines() if line.strip()]


def test_split_roles_and_source_isolation():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    queries = _rows('queries.jsonl')
    assert len(queries) == 58
    roles = {}
    for row in queries:
        roles[row['case_role']] = roles.get(row['case_role'], 0) + 1
    assert roles == manifest['split_roles'] == {
        'delayed_training': 24, 'retention_replay': 8,
        'delayed_development': 4, 'delayed_audit': 8,
        'delayed_regression': 8, 'retention_development': 2, 'retention_audit': 4,
    }
    train = [row for row in queries if row['split'] == 'train']
    assert len(train) == 32
    assert {row['case_role'] for row in train} == {'delayed_training', 'retention_replay'}
    train_groups = {row['source_group'] for row in train}
    for row in queries:
        if row['split'] != 'train':
            assert row['source_group'] not in train_groups
    assert len({row['case_id'] for row in queries}) == len(queries)
    assert len({row['query'] for row in queries}) == len(queries)


def test_diagnostic_replay_and_final_audit_never_train():
    queries = _rows('queries.jsonl')
    assert not any(row['case_id'] in {'m2s_replay_01', 'm2s_replay_02'} for row in queries)
    assert not any(row['expected_answer'] in {'Paris', 'H2O'} and row['case_role'] == 'retention_replay'
                   for row in queries)
    for row in queries:
        if row['case_role'] in {'delayed_development', 'delayed_audit', 'retention_development', 'retention_audit'}:
            assert row['split'] != 'train'
    audit = [row for row in queries if row['case_role'] == 'retention_audit']
    manifest = json.loads((DATA / 'manifest.json').read_text())
    strata = {}
    for row in queries:
        if row['case_role'] == 'delayed_audit':
            strata[row['audit_stratum']] = strata.get(row['audit_stratum'], 0) + 1
    assert strata == manifest['delayed_audit_strata'] == {
        'audit_seen_value': 4, 'audit_novel_value': 2, 'audit_changed_template': 2}
    general_templates = {row['case_role']: row['template_id'] for row in queries
                         if row['case_role'].startswith('retention_')}
    assert general_templates['retention_replay'] == 'declarative_the'
    assert general_templates['retention_development'] == 'fact_prefix'
    assert general_templates['retention_audit'] == 'record_equals'
    assert manifest['holdout_scope'].startswith('Exposed diagnostic')
    assert [row['expected_answer'] for row in audit] == ['Canberra', 'Cl', 'Pacific Ocean', 'William Shakespeare']
    replay = [row for row in queries if row['case_role'] == 'retention_replay']
    assert not ({row['query'] for row in replay} & {row['query'] for row in audit})


def test_delayed_lineage_correction_and_new_bindings():
    queries = _rows('queries.jsonl')
    records = _rows('records.jsonl')
    by_id = {row['record_id']: row for row in records}
    delayed_values = {row['expected_answer'] for row in queries if row['case_role'] == 'delayed_training'}
    for row in queries:
        if (row['case_role'].startswith('delayed_') and row['case_role'] != 'delayed_training'
                and row.get('audit_stratum') != 'audit_novel_value'):
            assert row['expected_answer'] in delayed_values
            correction = by_id[f"{row['case_id']}_correction"]
            parent = by_id[f"{row['case_id']}_relevant"]
            assert correction['supersedes'] == parent['record_id']
            assert correction['observed_at'] > parent['observed_at']
            assert correction['authenticity'] == 'verified_source'
    original = [row for row in queries if row['case_role'] == 'delayed_training']
    regression = [row for row in queries if row['case_role'] == 'delayed_regression']
    assert len(regression) == 8 and all(row['split'] == 'regression' for row in regression)
