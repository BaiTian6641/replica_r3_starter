"""Fresh prompt-evidence fixtures: initial statuses and novelty."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; CASES=ROOT/'fixtures/e3_prompt/cases'
def test_prompt_fixture_initial_status():
    manifest=json.loads((CASES/'manifest.json').read_text())
    old=json.loads((ROOT/'fixtures/e3_mixed_fresh/cases/manifest.json').read_text())
    old_names=set()
    for item in old['cases']:
        old_names.add(json.loads((ROOT/'fixtures/e3_mixed_fresh/cases'/item['case_id']/'case.json').read_text())['variant']['constant'])
    groups=set()
    for item in manifest['cases']:
        assert item['source_group'] not in groups; groups.add(item['source_group'])
        case=json.loads((CASES/item['case_id']/'case.json').read_text())
        assert case['variant']['constant'] not in old_names
        r=subprocess.run([sys.executable,'-m','unittest','evaluator/test_policy.py'],cwd=CASES/item['case_id'],capture_output=True)
        assert (r.returncode==0) is (not item['requires_repair'])
