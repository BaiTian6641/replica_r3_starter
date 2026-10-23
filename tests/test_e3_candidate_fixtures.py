"""Candidate-reranking family: disjoint splits and reference repair."""
import json,shutil,tempfile
from pathlib import Path
from replica_lab.runtime.tools import FixtureWorkspace
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'fixtures/e3_candidate/cases'
def test_candidate_fixtures():
    m=json.loads((DATA/'manifest.json').read_text())
    assert len(m['splits']['dev'])==1 and len(m['splits']['heldout'])==4
    groups={x['role']:set() for x in m['cases']}
    for x in m['cases']:groups[x['role']].add(x['source_group'])
    assert not(groups['dev']&groups['heldout'])
    for x in m['cases']:
        with tempfile.TemporaryDirectory() as d:
            c=Path(d)/x['case_id'];shutil.copytree(DATA/x['case_id'],c);t=FixtureWorkspace(c)
            assert t.verify_success()['passed'] is False
            r=json.loads((c/'case.json').read_text())['reference_repair'];(c/r['path']).write_text(r['content'])
            assert t.verify_success()['passed'] is True
