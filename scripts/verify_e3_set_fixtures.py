#!/usr/bin/env python3
"""Validate the fresh E3 set_constant task family in isolated copies."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def run_test(case):
    return subprocess.run([sys.executable, 'evaluator/test_policy.py'],
                          cwd=case, capture_output=True, text=True, timeout=30).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'fixtures/e3_set/cases')
    args = parser.parse_args()
    cases = sorted(path for path in args.root.iterdir() if path.is_dir())
    if not cases:
        raise SystemExit('no cases')
    roles = Counter()
    checks = []
    for case in cases:
        meta_path = case / 'case.json'
        meta = json.loads(meta_path.read_text())
        required = {'format', 'case_id', 'role', 'goal', 'success_condition',
                    'constraints', 'allowed_tools', 'workspace_files', 'evaluator',
                    'workspace_entry', 'reference_repair', 'variant', 'source_group'}
        if set(meta) != required or meta['format'] != 'replica.e3.fixture.v1':
            raise SystemExit(f'{case.name}: invalid schema')
        if meta['allowed_tools'] != ['read_file', 'set_constant', 'run_test']:
            raise SystemExit(f'{case.name}: allowed_tools must be the bounded repair set')
        if not (case / 'workspace').is_dir() or not (case / meta['evaluator']).is_file():
            raise SystemExit(f'{case.name}: missing workspace/evaluator')
        if Path(meta['evaluator']).parts[0] == 'workspace':
            raise SystemExit(f'{case.name}: evaluator must be outside workspace')
        variant = meta['variant']
        if not meta['goal'] or str(variant['expected_value']) in meta['goal']:
            raise SystemExit(f'{case.name}: expected value leaked into goal')
        roles[meta['role']] += 1
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / case.name
            shutil.copytree(case, copy)
            if run_test(copy) == 0:
                raise SystemExit(f'{case.name}: fixture already passes')
            repair = meta['reference_repair']
            (copy / repair['path']).write_text(repair['content'], encoding='utf-8')
            if run_test(copy) != 0:
                raise SystemExit(f'{case.name}: reference repair does not pass')
            from replica_lab.runtime.tools import FixtureWorkspace
            tools = FixtureWorkspace(copy)
            result = tools.set_constant({'name': variant['constant'],
                                         'value': variant['expected_value']})
            if result.get('status') != 'ok':
                raise SystemExit(f'{case.name}: set_constant failed: {result}')
            if tools.run_test({}).get('receipt', {}).get('exit_code') != 0:
                raise SystemExit(f'{case.name}: set_constant repair does not pass')
            before = copy / 'workspace/app_settings.py'
            parsed = {}
            for line in before.read_text().splitlines():
                if ' = ' in line and not line.startswith(('"""', '#')):
                    name, value = line.split(' = ', 1)
                    parsed[name] = value
            if parsed.get(variant['constant']) != str(variant['expected_value']):
                raise SystemExit(f'{case.name}: tool wrote the wrong value')
            checks.append({'case_id': meta['case_id'], 'role': meta['role'],
                           'constant': variant['constant'],
                           'expected': variant['expected_value'],
                           'tool_passed': True})
    if roles != Counter({'train': 4, 'dev': 1, 'heldout': 3, 'interrupted': 1}):
        raise SystemExit(f'unexpected role counts: {roles}')
    print(json.dumps({'ok': True, 'cases': len(cases), 'roles': dict(roles),
                      'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
