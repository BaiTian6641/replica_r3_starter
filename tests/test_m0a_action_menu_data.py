"""M0a fixture and corrected-menu data checks (CPU only)."""
import json
from pathlib import Path

from replica_lab.runtime.candidate_actions import build_action_menu
from replica_lab.runtime.procedures import RepairProcedure
from replica_lab.runtime.tools import FixtureWorkspace
from tests.e3_runtime_helpers import make_state

ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / 'fixtures/e3_m0a_menus/cases'


def _manifest():
    return json.loads((CASE_ROOT / 'manifest.json').read_text())


def test_m0a_cases_have_disjoint_groups_and_distinct_targets():
    manifest = _manifest()
    assert manifest['format'] == 'replica.e3_m0a_menus.dataset.v1'
    cases = manifest['cases']
    assert len(cases) == 4
    assert {case['role'] for case in cases} == {'train', 'dev', 'heldout'}
    assert len({case['source_group'] for case in cases}) == 4
    assert len({case['constant'] for case in cases}) == 4
    assert len({case['case_id'] for case in cases}) == 4


def test_m0a_numeric_constants_reach_each_action_menu():
    for case in _manifest()['cases']:
        tools = FixtureWorkspace(CASE_ROOT / case['case_id'])
        menu = build_action_menu(tools, make_state(tools))
        repairs = {candidate['arguments']['name']: candidate['arguments']['value']
                   for candidate in menu if candidate.get('tool') == 'set_constant'}
        assert repairs[case['constant']] == float(case['actual'])
        assert len(repairs) == 3


def test_m0a_deterministic_procedure_repairs_every_fixture():
    for case in _manifest()['cases']:
        tools = FixtureWorkspace(CASE_ROOT / case['case_id'])
        initial = tools.run_test({})
        assert initial['receipt']['exit_code'] != 0
        run = RepairProcedure('m0a-reference', 'workspace/app_settings.py',
                              ('fixture:' + case['case_id'],)).execute(tools)
        assert run.status == 'succeeded', run.error
        assert run.final_exit_code == 0
        assert tools.verify_success()['passed'] is True
        assert len(run.receipts) == 4
