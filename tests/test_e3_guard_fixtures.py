"""Fresh guard-family fixtures start failing and the disclosed repair passes."""
import json, shutil, tempfile
from pathlib import Path
from replica_lab.runtime.tools import FixtureWorkspace
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'fixtures/e3_guard/cases'
def test_guard_fixtures_are_valid_and_fresh():
    m=json.loads((DATA/'manifest.json').read_text())
    assert m['splits']['heldout']==['guard_heldout_clock_01','guard_heldout_queue_02','guard_heldout_grace_03']
    assert len(set(x['source_group'] for x in m['cases']))==len(m['cases'])
    for item in m['cases']:
        with tempfile.TemporaryDirectory() as directory:
            copy=Path(directory)/item['case_id']; shutil.copytree(DATA/item['case_id'],copy)
            tools=FixtureWorkspace(copy)
            assert tools.verify_success()['passed'] is False
            repair=json.loads((copy/'case.json').read_text())['reference_repair']
            (copy/repair['path']).write_text(repair['content'])
            assert tools.verify_success()['passed'] is True
