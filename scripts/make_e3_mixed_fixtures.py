#!/usr/bin/env python3
"""Generate a mixed-outcome set_constant family for action-policy probes.

Half the cases initially satisfy the evaluator and should be stopped after
verification; half require one set_constant repair. This makes the post-test
action differ at the same trajectory step. Source groups are disjoint.
"""
from __future__ import annotations

import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from make_e3_set_fixtures import evaluator, file_text
OUT = ROOT / 'fixtures/e3_mixed/cases'
GOAL = ('The policy test for workspace/app_settings.py must pass. Inspect the '
        'configuration, run the evaluator, and if a required bound fails change '
        'exactly that one numeric constant with set_constant; if it already '
        'passes, stop without editing.')
CASES = [
    ('mixed_train_repair_01', 'train', 'train_repair_a', 'RETRY_ATTEMPTS', 2, 5,
     {'BACKOFF_SECONDS': 0.4, 'REQUEST_TIMEOUT_MS': 1200}, True),
    ('mixed_train_repair_02', 'train', 'train_repair_b', 'REQUEST_TIMEOUT_MS', 1500, 9000,
     {'RETRY_ATTEMPTS': 3, 'BACKOFF_SECONDS': 0.25}, True),
    ('mixed_train_pass_03', 'train', 'train_pass_a', 'BACKOFF_SECONDS', 2.0, 1.5,
     {'MAX_RETRIES': 2, 'REQUEST_TIMEOUT_MS': 800}, False),
    ('mixed_train_pass_04', 'train', 'train_pass_b', 'POLL_ATTEMPTS', 9, 8,
     {'BACKOFF_SECONDS': 0.6, 'REQUEST_TIMEOUT_MS': 400}, False),
    ('mixed_dev_repair_05', 'dev', 'dev_repair_a', 'MAX_RETRIES', 1, 4,
     {'REQUEST_TIMEOUT_MS': 8000, 'BACKOFF_SECONDS': 0.3}, True),
    ('mixed_dev_pass_06', 'dev', 'dev_pass_a', 'CONNECT_TIMEOUT_MS', 2500, 2500,
     {'MAX_RETRIES': 3, 'BACKOFF_SECONDS': 0.5}, False),
    ('mixed_heldout_repair_07', 'heldout', 'heldout_repair_a', 'BACKOFF_SECONDS', 0.1, 1.5,
     {'RETRY_ATTEMPTS': 2, 'REQUEST_TIMEOUT_MS': 700}, True),
    ('mixed_heldout_repair_08', 'heldout', 'heldout_repair_b', 'POLL_ATTEMPTS', 2, 7,
     {'BACKOFF_SECONDS': 0.45, 'REQUEST_TIMEOUT_MS': 600}, True),
    ('mixed_heldout_pass_09', 'heldout', 'heldout_pass_a', 'MAX_RETRIES', 6, 6,
     {'REQUEST_TIMEOUT_MS': 900, 'BACKOFF_SECONDS': 0.35}, False),
    ('mixed_heldout_pass_10', 'heldout', 'heldout_pass_b', 'RETRY_ATTEMPTS', 4, 4,
     {'BACKOFF_SECONDS': 0.2, 'REQUEST_TIMEOUT_MS': 1000}, False),
]


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit('Use a fresh fixture directory')
    manifest = {'format': 'replica.e3_mixed.dataset.v1', 'created_at_utc': '2026-10-23',
                'goal': GOAL, 'cases': [], 'splits': {}}
    for case_id, role, group, target, actual, expected, others, repair in CASES:
        case_dir = OUT / case_id
        (case_dir / 'workspace').mkdir(parents=True)
        (case_dir / 'evaluator').mkdir()
        originals = dict(others); originals[target] = actual
        (case_dir / 'workspace/app_settings.py').write_text(file_text(target, originals))
        (case_dir / 'evaluator/test_policy.py').write_text(evaluator(target, expected, others))
        repaired = dict(others); repaired[target] = expected
        case = {
            'format': 'replica.e3.fixture.v1', 'case_id': case_id, 'role': role,
            'goal': GOAL,
            'success_condition': {'condition_id': 'evaluator_test_passes',
                                  'description': 'python -m unittest evaluator/test_policy.py exits with status zero'},
            'constraints': [
                'Only workspace/app_settings.py may be modified.',
                'If a repair is needed, change exactly one numeric constant with set_constant and leave every other setting untouched.',
                'Do not modify anything under the evaluator directory.',
                'Use only the allowed tools to inspect, edit and test.',
            ],
            'allowed_tools': ['read_file', 'set_constant', 'run_test'],
            'workspace_files': ['workspace/app_settings.py'],
            'evaluator': 'evaluator/test_policy.py',
            'workspace_entry': 'workspace/app_settings.py',
            'reference_repair': {'path': 'workspace/app_settings.py',
                                 'content': file_text(target, repaired)},
            'variant': {'constant': target, 'expected_value': expected,
                        'actual_value': actual,
                        'description': ('one numeric policy bound below its required minimum'
                                        if repair else 'all bounds already satisfy the policy')},
            'requires_repair': repair,
            'source_group': group,
        }
        (case_dir / 'case.json').write_text(json.dumps(case, indent=2) + '\n')
        manifest['cases'].append({'case_id': case_id, 'role': role,
                                  'source_group': group, 'requires_repair': repair})
        manifest['splits'].setdefault(role, []).append(case_id)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'cases': len(CASES), 'splits': manifest['splits'],
                      'manifest_sha256': hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
