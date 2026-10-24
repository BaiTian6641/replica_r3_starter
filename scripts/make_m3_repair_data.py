#!/usr/bin/env python3
"""Build source-separated M3 retention-repair episodes without using diagnostic replay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from make_m2_dynamic_data import observation, record

OUT = ROOT / 'data/m3_repair'
GENERAL_TRAIN = [
    ('capital of Japan', 'Tokyo', 'What is the capital of Japan?'),
    ('chemical symbol for oxygen', 'O', 'What is the chemical symbol for oxygen?'),
    ('largest planet in the Solar System', 'Jupiter', 'What is the largest planet in the Solar System?'),
    ('author of Pride and Prejudice', 'Jane Austen', 'Who is the author of Pride and Prejudice?'),
    ('square root of 144', '12', 'What is the square root of 144?'),
    ('first prime number', '2', 'What is the first prime number?'),
    ('boiling point of water at sea level in Celsius', '100', 'What is the boiling point of water at sea level in Celsius?'),
    ('continent containing Egypt', 'Africa', 'Which continent contains Egypt?'),
]
GENERAL_DEV = [
    ('capital of Canada', 'Ottawa', 'What is the capital of Canada?'),
    ('chemical symbol for sodium', 'Na', 'What is the chemical symbol for sodium?'),
]
GENERAL_AUDIT = [
    ('capital of Australia', 'Canberra', 'What is the capital of Australia?'),
    ('chemical symbol for chlorine', 'Cl', 'What is the chemical symbol for chlorine?'),
    ('largest ocean', 'Pacific Ocean', 'What is the largest ocean?'),
    ('author of Romeo and Juliet', 'William Shakespeare', 'Who is the author of Romeo and Juliet?'),
]
FRESH_DELAYED = [
    ('Nacre', 'deploy region', 'cobalt', 'saffron'),
    ('Oriel', 'launch code', 'saffron', 'teal'),
    ('Pollux', 'backup window', 'teal', 'violet'),
    ('Rill', 'owner', 'violet', 'amber'),
    ('Solen', 'review code', 'amber', 'copper'),
    ('Tarn', 'timeout seconds', 'copper', 'cobalt'),
    ('Ulex', 'retry limit', 'cobalt', 'saffron'),
    ('Vela', 'release label', 'saffron', 'teal'),
]


def general_row(index, split, role, attr, value, question):
    cid = f'm3r_{role}_{index:02d}'
    group = f'm3r_{role}_world_{index:02d}'
    text = f'The {attr} is {value}.'
    query = {'as_of': 4, 'branch_id': cid, 'case_id': cid, 'case_role': role,
             'expected_answer': value, 'expected_record_id': f'{cid}_relevant',
             'observations': [text], 'purpose': 'm2_dynamic', 'query': question,
             'source_group': group, 'split': split, 'subject_id': 'm2_dynamic',
             'world_id': group}
    return query, record(cid, 'relevant', 'm2_dynamic', cid, text, 1, group), {
        'case_id': cid, 'split': split, 'case_role': role, 'source_group': group,
        'attribute': attr, 'value': value, 'corrected_value': None}


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit('Use a fresh data directory')
    OUT.mkdir(parents=True)
    queries, records, cases = [], [], []
    original_queries = [json.loads(line) for line in
                        (ROOT / 'data/m2_shared/queries.jsonl').read_text().splitlines() if line.strip()]
    original_records = [json.loads(line) for line in
                        (ROOT / 'data/m2_shared/records.jsonl').read_text().splitlines() if line.strip()]
    original_by_branch = {}
    for row in original_records:
        original_by_branch.setdefault(row['branch_id'], []).append(row)
    for row in original_queries[:24]:
        queries.append({**row, 'case_role': 'delayed_training'})
        records.extend(original_by_branch[row['branch_id']])
        cases.append({'case_id': row['case_id'], 'split': 'train', 'case_role': 'delayed_training',
                      'source_group': row['source_group'], 'attribute': row['query'],
                      'value': row['expected_answer'], 'corrected_value': row['corrected_answer']})
    for row in original_queries[24:32]:
        queries.append({**row, 'split': 'regression', 'case_role': 'delayed_regression'})
        records.extend(original_by_branch[row['branch_id']])
        cases.append({'case_id': row['case_id'], 'split': 'regression', 'case_role': 'delayed_regression',
                      'source_group': row['source_group'], 'attribute': row['query'],
                      'value': row['expected_answer'], 'corrected_value': row['corrected_answer']})
    for index, (attr, value, question) in enumerate(GENERAL_TRAIN, 1):
        query, record_row, case = general_row(index, 'train', 'retention_replay', attr, value, question)
        queries.append(query); records.append(record_row); cases.append(case)
    for index, (attr, value, question) in enumerate(GENERAL_DEV, 1):
        query, record_row, case = general_row(index, 'retention_dev', 'retention_development', attr, value, question)
        queries.append(query); records.append(record_row); cases.append(case)
    for index, (attr, value, question) in enumerate(GENERAL_AUDIT, 1):
        query, record_row, case = general_row(index, 'retention', 'retention_audit', attr, value, question)
        queries.append(query); records.append(record_row); cases.append(case)
    for offset, (name, attr, value, corrected) in enumerate(FRESH_DELAYED):
        split = 'dev' if offset < 4 else 'heldout'
        role = 'delayed_development' if split == 'dev' else 'delayed_audit'
        cid = f'm3r_{role}_{offset + 1:02d}'
        group = f'm3r_{role}_world_{offset + 1:02d}'
        index = offset + 40
        target = observation(name, attr, value, index, corrected)
        distractor = (f'The review code for project {name} is {corrected}.\n'
                      f'The owner of project {name} is person{index + 20:02d}.\n'
                      f'The backup window for project {name} is {value}.')
        correction = f'Correction: the {attr} for project {name} is {corrected}.'
        queries.append({
            'as_of': 4, 'branch_id': cid, 'case_id': cid, 'case_role': role,
            'expected_answer': value, 'expected_record_id': f'{cid}_relevant',
            'observations': [target, distractor], 'correction_observation': correction,
            'corrected_answer': corrected, 'purpose': 'm2_dynamic',
            'query': f'What is the {attr} for project {name}?', 'source_group': group,
            'split': split, 'subject_id': 'm2_dynamic', 'world_id': group})
        records.extend([
            record(cid, 'relevant', 'm2_dynamic', cid, target, 1, group),
            record(cid, 'distractor', 'm2_dynamic', cid, distractor, 2, group),
            record(cid, 'correction', 'm2_dynamic', cid, correction, 3, group,
                   supersedes=f'{cid}_relevant')])
        cases.append({'case_id': cid, 'split': split, 'case_role': role, 'source_group': group,
                      'attribute': attr, 'value': value, 'corrected_value': corrected})
    manifest = {
        'format': 'replica.m3_repair.dataset.v1',
        'created_at_utc': '2026-10-24',
        'source_revision': 'f97400052a43d642bbc6e9975e2397e3ae6a6b52',
        'source_lineage': {
            'original_delayed_root': 'data/m2_shared',
            'original_queries_sha256': hashlib.sha256((ROOT / 'data/m2_shared/queries.jsonl').read_bytes()).hexdigest(),
            'original_records_sha256': hashlib.sha256((ROOT / 'data/m2_shared/records.jsonl').read_bytes()).hexdigest(),
            'original_replay_excluded': ['m2s_replay_01', 'm2s_replay_02'],
        },
        'delayed_value_set': ['cobalt', 'saffron', 'teal', 'violet', 'amber', 'copper'],
        'split_roles': {
            'delayed_training': 24,
            'retention_replay': 8,
            'delayed_development': 4,
            'delayed_audit': 4,
            'delayed_regression': 8,
            'retention_development': 2,
            'retention_audit': 4,
        },
        'role_rules': 'Only delayed_training and retention_replay may train. Development, regression and final-audit rows never train or select thresholds. The two M3 diagnostic replay cases are excluded everywhere.',
        'cases': cases,
    }
    for name, values in (('queries', queries), ('records', records)):
        (OUT / f'{name}.jsonl').write_text(
            ''.join(json.dumps(row, sort_keys=True) + '\n' for row in values))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'queries': len(queries), 'records': len(records),
                      'queries_sha256': hashlib.sha256((OUT / 'queries.jsonl').read_bytes()).hexdigest(),
                      'records_sha256': hashlib.sha256((OUT / 'records.jsonl').read_bytes()).hexdigest(),
                      'manifest_sha256': hashlib.sha256((OUT / 'manifest.json').read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
