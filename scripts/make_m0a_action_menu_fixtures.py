#!/usr/bin/env python3
"""Generate the fresh M0a numeric-string action-menu family."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from make_e3_set_fixtures import GOAL, evaluator, file_text

OUT = ROOT / 'fixtures/e3_m0a_menus/cases'
CASES = [
    ('m0a_train_queue', 'train', 'm0a_train_a', 'QUEUE_CAPACITY', 3, 9,
     {'BURST_FACTOR': 0.25, 'IDLE_TIMEOUT_MS': 180}),
    ('m0a_dev_retry', 'dev', 'm0a_dev_a', 'RETRY_ATTEMPTS', 2, 6,
     {'BACKOFF_SECONDS': 0.4, 'MAX_BATCH_ITEMS': 5}),
    ('m0a_heldout_poll', 'heldout', 'm0a_heldout_a', 'POLL_INTERVAL_MS', 50, 350,
     {'BACKOFF_SECONDS': 0.4, 'RETRY_ATTEMPTS': 3}),
    ('m0a_heldout_heartbeat', 'heldout', 'm0a_heldout_b', 'HEARTBEAT_MS', 250, 1500,
     {'BACKOFF_SECONDS': 0.25, 'RETRY_ATTEMPTS': 2}),
]


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit('Use a fresh fixture directory')
    manifest = {
        'format': 'replica.e3_m0a_menus.dataset.v1',
        'created_at_utc': '2026-10-23',
        'goal': GOAL,
        'source_revision': 'f97400052a43d642bbc6e9975e2397e3ae6a6b52',
        'roles': ['train', 'dev', 'heldout'],
        'cases': [],
        'splits': {},
    }
    for cid, role, group, target, actual, expected, others in CASES:
        case_dir = OUT / cid
        (case_dir / 'workspace').mkdir(parents=True)
        (case_dir / 'evaluator').mkdir()
        originals = dict(others)
        originals[target] = actual
        (case_dir / 'workspace/app_settings.py').write_text(file_text(target, originals))
        (case_dir / 'evaluator/test_policy.py').write_text(evaluator(target, expected, others))
        repair = dict(others)
        repair[target] = expected
        payload = {
            'format': 'replica.e3.fixture.v1',
            'case_id': cid,
            'role': role,
            'goal': GOAL,
            'success_condition': {
                'condition_id': 'evaluator_test_passes',
                'description': 'python -m unittest evaluator/test_policy.py exits with status zero',
            },
            'constraints': [
                'Only workspace/app_settings.py may be modified.',
                'Change exactly one numeric constant with set_constant; leave every other setting untouched.',
                'Do not modify anything under the evaluator directory.',
                'Use only the allowed tools to inspect, edit, and test.',
            ],
            'allowed_tools': ['read_file', 'set_constant', 'run_test'],
            'workspace_files': ['workspace/app_settings.py'],
            'evaluator': 'evaluator/test_policy.py',
            'workspace_entry': 'workspace/app_settings.py',
            'reference_repair': {
                'path': 'workspace/app_settings.py',
                'content': file_text(target, repair),
            },
            'variant': {
                'constant': target,
                'expected_value': expected,
                'actual_value': actual,
                'description': 'one numeric policy bound below its required minimum',
            },
            'source_group': group,
        }
        (case_dir / 'case.json').write_text(json.dumps(payload, indent=2) + '\n')
        manifest['cases'].append({
            'case_id': cid, 'role': role, 'source_group': group,
            'constant': target, 'actual': actual, 'expected': expected,
            'other_constants': sorted(others),
        })
        manifest['splits'].setdefault(role, []).append(cid)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({
        'cases': len(CASES),
        'splits': {key: len(value) for key, value in manifest['splits'].items()},
        'manifest_sha256': hashlib.sha256((OUT / 'manifest.json').read_bytes()).hexdigest(),
    }))


if __name__ == '__main__':
    main()
