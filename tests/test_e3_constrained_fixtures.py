"""Fresh constrained-family fixtures and request-level mask metadata."""
import json, shutil, tempfile
from pathlib import Path
from replica_lab.runtime.tools import FixtureWorkspace
from tests.e3_runtime_helpers import ScriptedProposer, make_case, make_runtime, read_action
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'fixtures/e3_constrained/cases'
def test_constrained_fixtures_start_failing_and_repair_passes():
    m=json.loads((DATA/'manifest.json').read_text())
    assert len(m['splits']['heldout'])==3 and len(m['cases'])==4
    for item in m['cases']:
        with tempfile.TemporaryDirectory() as directory:
            copy=Path(directory)/item['case_id']; shutil.copytree(DATA/item['case_id'],copy)
            tools=FixtureWorkspace(copy)
            assert tools.verify_success()['passed'] is False
            repair=json.loads((copy/'case.json').read_text())['reference_repair']
            (copy/repair['path']).write_text(repair['content'])
            assert tools.verify_success()['passed'] is True

def test_mask_metadata_is_opt_in(tmp_path):
    tools=make_case(tmp_path)
    runtime=make_runtime(tools,tmp_path,mode='full',action_guard=True,constrain_actions=True)
    proposer=ScriptedProposer([read_action(), read_action()])
    runtime.run(proposer,max_decisions=2)
    assert proposer.requests[1].forbidden_action_strings == ('read',)
