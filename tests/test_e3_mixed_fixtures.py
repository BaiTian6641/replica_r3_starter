"""Mixed-outcome E3 fixtures: initial evaluator status and disjoint roles."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'fixtures/e3_mixed/cases'


def test_mixed_fixtures_initial_status_and_groups():
    manifest = json.loads((CASES / 'manifest.json').read_text())
    assert manifest['format'] == 'replica.e3_mixed.dataset.v1'
    roles = {}
    groups = set()
    for item in manifest['cases']:
        roles.setdefault(item['role'], []).append(item['case_id'])
        assert item['source_group'] not in groups
        groups.add(item['source_group'])
        case = json.loads((CASES / item['case_id'] / 'case.json').read_text())
        assert case['requires_repair'] is item['requires_repair']
        result = subprocess.run(
            [sys.executable, '-m', 'unittest', 'evaluator/test_policy.py'],
            cwd=CASES / item['case_id'], capture_output=True, text=True)
        assert (result.returncode == 0) is (not item['requires_repair'])
    assert len(roles['train']) == 4 and len(roles['dev']) == 2
    assert len(roles['heldout']) == 4
    assert sum(not item['requires_repair'] for item in manifest['cases']) == 5
