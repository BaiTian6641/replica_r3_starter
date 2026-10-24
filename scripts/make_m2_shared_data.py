#!/usr/bin/env python3
"""Generate M2b fresh shared-value delayed-use data."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from make_m2_dynamic_data import observation, record

OUT = ROOT / 'data/m2_shared'
NAMES = [
    'Alder', 'Basin', 'Cobalt', 'Dell', 'Ember', 'Fern', 'Grove', 'Hearth',
    'Inlet', 'Juniper', 'Kite', 'Lark', 'Meadow', 'Nectar', 'Orchid', 'Pike',
    'Quarry', 'Ridge', 'Sable', 'Talon', 'Umber', 'Vale', 'Wren', 'Yarrow',
    'Zephyr', 'Beryl', 'Cinder', 'Dune', 'Aster', 'Briar', 'Cedar', 'Flint',
]
ATTRIBUTES = [
    'deploy region', 'launch code', 'backup window', 'owner',
    'review code', 'timeout seconds', 'retry limit', 'release label',
]
VALUES = ['cobalt', 'saffron', 'teal', 'violet', 'amber', 'copper']


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit('Use a fresh data directory')
    OUT.mkdir(parents=True)
    queries, records, cases = [], [], []
    for index, name in enumerate(NAMES, 1):
        split = 'train' if index <= 24 else ('dev' if index <= 28 else 'heldout')
        cid = f'm2s_{split}_{index:02d}'
        group = f'm2s_{name.lower()}'
        attr = ATTRIBUTES[(index - 1) % len(ATTRIBUTES)]
        value = VALUES[(index - 1) % len(VALUES)]
        other = VALUES[(index + 1) % len(VALUES)]
        corrected = VALUES[(index + 2) % len(VALUES)]
        target = observation(name, attr, value, index, other)
        distractor = (f'The review code for project {name} is {other}.\n'
                      f'The owner of project {name} is person{index + 20:02d}.\n'
                      f'The backup window for project {name} is {value}.')
        correction = f'Correction: the {attr} for project {name} is {corrected}.'
        queries.append({
            'as_of': 4, 'branch_id': cid, 'case_id': cid,
            'expected_answer': value, 'expected_record_id': f'{cid}_relevant',
            'observations': [target, distractor],
            'correction_observation': correction, 'corrected_answer': corrected,
            'purpose': 'm2_dynamic', 'query': f'What is the {attr} for project {name}?',
            'source_group': group, 'split': split, 'subject_id': 'm2_dynamic',
            'world_id': group,
        })
        records.extend([
            record(cid, 'relevant', 'm2_dynamic', cid, target, 1, group),
            record(cid, 'distractor', 'm2_dynamic', cid, distractor, 2, group),
            record(cid, 'correction', 'm2_dynamic', cid, correction, 3, group,
                   supersedes=f'{cid}_relevant'),
        ])
        cases.append({'case_id': cid, 'split': split, 'source_group': group,
                      'attribute': attr, 'value': value, 'corrected_value': corrected})
    for cid, split, group, attr, value in [
        ('m2s_replay_01', 'retention', 'm2s_replay_a', 'capital of France', 'Paris'),
        ('m2s_replay_02', 'retention', 'm2s_replay_b', 'chemical symbol for water', 'H2O'),
    ]:
        text = f'The {attr} is {value}.'
        queries.append({
            'as_of': 4, 'branch_id': cid, 'case_id': cid,
            'expected_answer': value, 'expected_record_id': f'{cid}_relevant',
            'observations': [text], 'purpose': 'm2_dynamic',
            'query': f'What is the {attr}?', 'source_group': group,
            'split': split, 'subject_id': 'm2_dynamic', 'world_id': group,
        })
        records.append(record(cid, 'relevant', 'm2_dynamic', cid, text, 1, group))
        cases.append({'case_id': cid, 'split': split, 'source_group': group,
                      'attribute': attr, 'value': value, 'corrected_value': None})
    manifest = {
        'format': 'replica.m2_shared.dataset.v1',
        'created_at_utc': '2026-10-24',
        'source_revision': 'f97400052a43d642bbc6e9975e2397e3ae6a6b52',
        'value_set': VALUES,
        'splits': {split: sum(case['split'] == split for case in cases)
                   for split in ('train', 'dev', 'heldout', 'retention')},
        'source_groups': sorted({case['source_group'] for case in cases}),
        'cases': cases,
    }
    (OUT / 'queries.jsonl').write_text('\n'.join(json.dumps(row, sort_keys=True) for row in queries) + '\n')
    (OUT / 'records.jsonl').write_text('\n'.join(json.dumps(row, sort_keys=True) for row in records) + '\n')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({
        'queries': len(queries), 'records': len(records), 'splits': manifest['splits'],
        'manifest_sha256': hashlib.sha256((OUT / 'manifest.json').read_bytes()).hexdigest(),
        'queries_sha256': hashlib.sha256((OUT / 'queries.jsonl').read_bytes()).hexdigest(),
        'records_sha256': hashlib.sha256((OUT / 'records.jsonl').read_bytes()).hexdigest(),
    }))


if __name__ == '__main__':
    main()
