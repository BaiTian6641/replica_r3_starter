#!/usr/bin/env python3
"""Generate the fresh E3 set_constant task family.

This is a separately declared narrower action space than the write_file arm.
Every case exposes only read_file, run_test and set_constant; the tool changes
one numeric literal and preserves all other settings. Train, dev, held-out and
interrupted source groups are disjoint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'fixtures/e3_set/cases'

CASES = [
    ('set_train_retry_01', 'train', 'train_a', 'RETRY_ATTEMPTS', 2, 5,
     {'BACKOFF_SECONDS': 0.4, 'REQUEST_TIMEOUT_MS': 1200}),
    ('set_train_timeout_02', 'train', 'train_a', 'REQUEST_TIMEOUT_MS', 1500, 9000,
     {'RETRY_ATTEMPTS': 3, 'BACKOFF_SECONDS': 0.25}),
    ('set_train_backoff_03', 'train', 'train_b', 'BACKOFF_SECONDS', 0.2, 1.5,
     {'MAX_RETRIES': 2, 'REQUEST_TIMEOUT_MS': 800}),
    ('set_train_poll_04', 'train', 'train_b', 'POLL_ATTEMPTS', 3, 8,
     {'BACKOFF_SECONDS': 0.6, 'REQUEST_TIMEOUT_MS': 400}),
    ('set_dev_retry_05', 'dev', 'dev_a', 'MAX_RETRIES', 1, 4,
     {'REQUEST_TIMEOUT_MS': 8000, 'BACKOFF_SECONDS': 0.3}),
    ('set_heldout_timeout_06', 'heldout', 'heldout_a', 'CONNECT_TIMEOUT_MS', 500, 2500,
     {'MAX_RETRIES': 3, 'BACKOFF_SECONDS': 0.5}),
    ('set_heldout_backoff_07', 'heldout', 'heldout_b', 'BACKOFF_SECONDS', 0.1, 1.5,
     {'RETRY_ATTEMPTS': 2, 'REQUEST_TIMEOUT_MS': 700}),
    ('set_heldout_poll_08', 'heldout', 'heldout_c', 'POLL_ATTEMPTS', 2, 7,
     {'BACKOFF_SECONDS': 0.45, 'REQUEST_TIMEOUT_MS': 600}),
    ('set_interrupted_retry_09', 'interrupted', 'interrupted_a', 'MAX_RETRIES', 2, 6,
     {'REQUEST_TIMEOUT_MS': 900, 'BACKOFF_SECONDS': 0.35}),
]
GOAL = ('The policy test for workspace/app_settings.py fails. Inspect the configuration, '
        'run the evaluator to learn the required bound, change exactly that one numeric '
        'constant with set_constant, and rerun the test to confirm it passes.')


def file_text(target, value):
    lines = ['"""Service client settings. Adjust values to satisfy the deployment policy."""']
    for name, number in value.items():
        lines.append(f'{name} = {number}')
    return '\n'.join(lines) + '\n'


def evaluator(target, expected, original):
    return '''"""Independent policy evaluator. Read-only; do not modify."""
import ast
import pathlib
import unittest

SETTINGS = pathlib.Path(__file__).resolve().parents[1] / "workspace" / "app_settings.py"
TARGET = "{target}"
MINIMUM = {expected}
ORIGINAL = {original!r}


def settings():
    tree = ast.parse(SETTINGS.read_text(encoding="utf-8"))
    values = {{}}
    for index, node in enumerate(tree.body):
        if index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \\
                and isinstance(node.value.value, str):
            continue
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            raise AssertionError("only simple name = number assignments are allowed")
        name, value = node.targets[0].id, node.value
        if not (isinstance(value, ast.Constant) and isinstance(value.value, (int, float))
                and not isinstance(value.value, bool)):
            raise AssertionError(f"{{name}} must be a numeric literal")
        if name in values:
            raise AssertionError(f"{{name}} must be assigned exactly once")
        values[name] = value.value
    return values


class PolicyTest(unittest.TestCase):
    def test_policy_bound(self):
        try:
            values = settings()
        except (OSError, UnicodeError, SyntaxError, ValueError, AssertionError) as error:
            self.fail(f"invalid workspace/app_settings.py: {{error}}")
        self.assertEqual(set(values), set(ORIGINAL) | {{TARGET}},
                         "original settings must be defined exactly")
        self.assertGreaterEqual(values[TARGET], MINIMUM,
                                f"{{TARGET}}={{values[TARGET]}} needs {{TARGET}} >= {{MINIMUM}}")
        for name, expected in ORIGINAL.items():
            self.assertEqual(values[name], expected, f"{{name}} must keep its original value")


if __name__ == "__main__":
    unittest.main()
'''.format(target=target, expected=expected, original=original)


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit('Use a fresh fixture directory')
    manifest = {'format': 'replica.e3_set.dataset.v1', 'created_at_utc': '2026-10-23',
                'goal': GOAL, 'cases': [], 'splits': {}}
    for case_id, role, group, target, actual, expected, others in CASES:
        case_dir = OUT / case_id
        (case_dir / 'workspace').mkdir(parents=True)
        (case_dir / 'evaluator').mkdir()
        originals = dict(others)
        originals[target] = actual
        workspace_text = file_text(target, originals)
        (case_dir / 'workspace/app_settings.py').write_text(workspace_text)
        (case_dir / 'evaluator/test_policy.py').write_text(
            evaluator(target, expected, others))
        repair = dict(others)
        repair[target] = expected
        case = {
            'format': 'replica.e3.fixture.v1', 'case_id': case_id, 'role': role,
            'goal': GOAL,
            'success_condition': {'condition_id': 'evaluator_test_passes',
                                  'description': 'python -m unittest evaluator/test_policy.py exits with status zero'},
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
            'reference_repair': {'path': 'workspace/app_settings.py',
                                 'content': file_text(target, repair)},
            'variant': {'constant': target, 'expected_value': expected,
                        'actual_value': actual,
                        'description': 'one numeric policy bound below its required minimum'},
            'source_group': group,
        }
        (case_dir / 'case.json').write_text(json.dumps(case, indent=2) + '\n')
        manifest['cases'].append({'case_id': case_id, 'role': role,
                                  'source_group': group, 'constant': target,
                                  'actual': actual, 'expected': expected})
        manifest['splits'].setdefault(role, []).append(case_id)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    digest = hashlib.sha256((OUT / 'manifest.json').read_bytes()).hexdigest()
    print(json.dumps({'cases': len(CASES), 'splits': manifest['splits'],
                      'manifest_sha256': digest}))


if __name__ == '__main__':
    main()
