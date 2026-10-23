"""Fresh E3 mixed fixtures: initial statuses, novelty and disjoint groups."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'fixtures/e3_mixed_fresh/cases'


def test_fresh_mixed_initial_status_and_novel_constants():
    manifest = json.loads((CASES / 'manifest.json').read_text())
    old = json.loads((ROOT / 'fixtures/e3_set/cases/manifest.json').read_text())
    old_constants = {item['constant'] for item in old['cases']}
    groups = set()
    for item in manifest['cases']:
        assert item['source_group'] not in groups
        groups.add(item['source_group'])
        case = json.loads((CASES / item['case_id'] / 'case.json').read_text())
        assert case['variant']['constant'] not in old_constants
        assert case['requires_repair'] is item['requires_repair']
        result = subprocess.run(
            [sys.executable, '-m', 'unittest', 'evaluator/test_policy.py'],
            cwd=CASES / item['case_id'], capture_output=True, text=True)
        assert (result.returncode == 0) is (not item['requires_repair'])
    assert len(manifest['cases']) == 4
    assert sum(not item['requires_repair'] for item in manifest['cases']) == 2
