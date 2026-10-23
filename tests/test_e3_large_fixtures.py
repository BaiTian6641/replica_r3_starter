"""Large E3 family: disjoint splits and reference repair behavior."""
import json,shutil,tempfile
from pathlib import Path
from replica_lab.runtime.tools import FixtureWorkspace
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'fixtures/e3_set_large/cases'
def test_large_fixtures():
    m=json.loads((DATA/'manifest.json').read_text())
    assert len(m['splits']['train'])==12 and len(m['splits']['dev'])==3 and len(m['splits']['heldout'])==6
    groups={}
    for item in m['cases']:groups.setdefault(item['role'],set()).add(item['source_group'])
    for a in groups:
        for b in groups:
            if a!=b:assert not(groups[a]&groups[b])
    for item in m['cases']:
        with tempfile.TemporaryDirectory() as directory:
            copy=Path(directory)/item['case_id']; shutil.copytree(DATA/item['case_id'],copy)
            tools=FixtureWorkspace(copy)
            assert tools.verify_success()['passed'] is False
            repair=json.loads((copy/'case.json').read_text())['reference_repair']
            (copy/repair['path']).write_text(repair['content'])
            assert tools.verify_success()['passed'] is True

def test_d2_family_is_distinct_and_fresh():
    DATA=ROOT/'fixtures/e3_d2/cases'; m=json.loads((DATA/'manifest.json').read_text())
    assert len(m['splits']['train'])==8 and len(m['splits']['heldout'])==4
    groups={}
    for item in m['cases']:groups.setdefault(item['role'],set()).add(item['source_group'])
    for a in groups:
        for b in groups:
            if a!=b:assert not(groups[a]&groups[b])
