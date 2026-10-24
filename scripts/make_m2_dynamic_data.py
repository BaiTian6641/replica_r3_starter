#!/usr/bin/env python3
"""Generate fresh M2 chronological delayed-use data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/m2_dynamic'
SPECS = [
    ('m2_train_01', 'train', 'm2_train_a', 'Kestrel', 'deploy region', 'cobalt', 'saffron', 'obsidian'),
    ('m2_train_02', 'train', 'm2_train_b', 'Falcon', 'launch code', 'copper', 'amber', 'pearl'),
    ('m2_train_03', 'train', 'm2_train_c', 'Rook', 'backup window', 'teal', 'ochre', 'linen'),
    ('m2_train_04', 'train', 'm2_train_d', 'Heron', 'owner', 'silver', 'bronze', 'rose'),
    ('m2_train_05', 'train', 'm2_train_e', 'Ibis', 'review code', 'ochre', 'violet', 'moss'),
    ('m2_train_06', 'train', 'm2_train_f', 'Merlin', 'timeout seconds', 'bronze', 'indigo', 'slate'),
    ('m2_train_07', 'train', 'm2_train_g', 'Osprey', 'retry limit', 'violet', 'coral', 'onyx'),
    ('m2_train_08', 'train', 'm2_train_h', 'Egret', 'release label', 'indigo', 'quartz', 'sage'),
    ('m2_dev_01', 'dev', 'm2_dev_a', 'Sable', 'deploy region', 'amber', 'cobalt', 'pewter'),
    ('m2_dev_02', 'dev', 'm2_dev_b', 'Mica', 'launch code', 'crimson', 'copper', 'azure'),
    ('m2_dev_03', 'dev', 'm2_dev_c', 'Flint', 'backup window', 'emerald', 'teal', 'cinder'),
    ('m2_dev_04', 'dev', 'm2_dev_d', 'Onyx', 'owner', 'quartz', 'silver', 'umber'),
    ('m2_heldout_01', 'heldout', 'm2_heldout_a', 'Aster', 'review code', 'saffron', 'ochre', 'cedar'),
    ('m2_heldout_02', 'heldout', 'm2_heldout_b', 'Briar', 'timeout seconds', 'topaz', 'bronze', 'frost'),
    ('m2_heldout_03', 'heldout', 'm2_heldout_c', 'Cedar', 'retry limit', 'coral', 'violet', 'willow'),
    ('m2_heldout_04', 'heldout', 'm2_heldout_d', 'Dune', 'release label', 'ivory', 'indigo', 'harbor'),
]
REPLAY = [
    ('m2_replay_01', 'retention', 'm2_replay_a', 'capital of France', 'Paris'),
    ('m2_replay_02', 'retention', 'm2_replay_b', 'chemical symbol for water', 'H2O'),
]


def observation(name, attribute, value, index, other):
    return (f'The {attribute} for project {name} is {value}.\n'
            f'The owner of project {name} is person{index:02d}.\n'
            f'The launch code for project {name} is {other}.')


def record(case_id, suffix, subject, branch, text, observed, source, *, supersedes=None):
    return {
        'record_id': f'{case_id}_{suffix}', 'subject_id': subject, 'branch_id': branch,
        'text': text, 'observed_at': observed, 'event_at': observed,
        'purpose': 'm2_dynamic', 'kind': 'source', 'supersedes': supersedes,
        'source_id': source, 'authenticity': 'source_report' if supersedes is None else 'verified_source',
    }


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit('Use a fresh data directory')
    OUT.mkdir(parents=True)
    queries, records, cases = [], [], []
    for index, (cid, split, group, name, attr, value, other, corrected) in enumerate(SPECS, 1):
        target = observation(name, attr, value, index, other)
        distractor = (f'The review code for project {name} is {other}.\n'
                      f'The owner of project {name} is person{index + 20:02d}.\n'
                      f'The backup window for project {name} is {value}.')
        correction = f'Correction: the {attr} for project {name} is {corrected}.'
        query = f'What is the {attr} for project {name}?'
        queries.append({
            'as_of': 4, 'branch_id': cid, 'case_id': cid,
            'expected_answer': value, 'expected_record_id': f'{cid}_relevant',
            'observations': [target, distractor],
            'correction_observation': correction, 'corrected_answer': corrected,
            'purpose': 'm2_dynamic', 'query': query, 'source_group': group,
            'split': split, 'subject_id': 'm2_dynamic', 'world_id': group,
        })
        records.extend([
            record(cid, 'relevant', 'm2_dynamic', cid, target, 1, group),
            record(cid, 'distractor', 'm2_dynamic', cid, distractor, 2, group),
            record(cid, 'correction', 'm2_dynamic', cid, correction, 3, group,
                   supersedes=f'{cid}_relevant'),
        ])
        cases.append({'case_id': cid, 'split': split, 'source_group': group,
                      'attribute': attr, 'value': value, 'corrected_value': corrected})
    for cid, split, group, attr, value in REPLAY:
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
        'format': 'replica.m2_dynamic.dataset.v1',
        'created_at_utc': '2026-10-24',
        'source_revision': 'f97400052a43d642bbc6e9975e2397e3ae6a6b52',
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
