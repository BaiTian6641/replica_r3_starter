"""DAgger family: disjoint splits and reference repairs."""
import json,shutil,tempfile
from pathlib import Path
from replica_lab.runtime.tools import FixtureWorkspace
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'fixtures/e3_dagger/cases'
def test_dagger_fixtures():
    m=json.loads((DATA/'manifest.json').read_text())
    assert len(m['splits']['train'])==4 and len(m['splits']['heldout'])==4
    groups={}
    for x in m['cases']:groups.setdefault(x['role'],set()).add(x['source_group'])
    for a in groups:
        for b in groups:
            if a!=b:assert not(groups[a]&groups[b])
    for x in m['cases']:
        with tempfile.TemporaryDirectory() as d:
            c=Path(d)/x['case_id']; shutil.copytree(DATA/x['case_id'],c); t=FixtureWorkspace(c)
            assert t.verify_success()['passed'] is False
            r=json.loads((c/'case.json').read_text())['reference_repair']; (c/r['path']).write_text(r['content'])
            assert t.verify_success()['passed'] is True
