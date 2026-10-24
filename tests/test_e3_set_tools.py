"""Bounded set_constant tool and fresh fixture family; CPU only."""

from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def test_set_constant_repairs_only_the_target_and_preserves_others():
    from replica_lab.runtime.tools import FixtureWorkspace
    with tempfile.TemporaryDirectory() as directory:
        case = Path(directory) / 'case'
        shutil.copytree(ROOT / 'fixtures/e3_set/cases/set_train_retry_01', case)
        tools = FixtureWorkspace(case)
        before = (case / 'workspace/app_settings.py').read_text()
        result = tools.set_constant({'name': 'RETRY_ATTEMPTS', 'value': 5})
        assert result['status'] == 'ok'
        assert result['kind'] == 'set_constant'
        after = (case / 'workspace/app_settings.py').read_text()
        assert 'RETRY_ATTEMPTS = 5' in after
        assert 'BACKOFF_SECONDS = 0.4' in after
        assert 'REQUEST_TIMEOUT_MS = 1200' in after
        assert before != after
        assert tools.run_test({})['receipt']['exit_code'] == 0
        denied = tools.set_constant({'name': 'MISSING', 'value': 1})
        assert denied['status'] == 'denied'
        assert tools.set_constant({'name': 'RETRY_ATTEMPTS', 'value': True})['status'] == 'denied'


def test_set_constant_action_parses_and_unknown_tool_stays_unavailable():
    from replica_lab.runtime.actions import parse_action_proposal
    raw = json.dumps({'kind': 'invoke_tool', 'tool': 'set_constant',
                      'arguments': {'name': 'MAX_RETRIES', 'value': 4}})
    parsed = parse_action_proposal(raw, ('read_file', 'set_constant', 'run_test'))
    assert parsed.status == 'ok'
    assert parsed.proposal.args['tool'] == 'set_constant'
    denied = parse_action_proposal(
        json.dumps({'kind': 'invoke_tool', 'tool': 'set_constant',
                    'arguments': {'name': 'MAX_RETRIES', 'value': 4}}),
        ('read_file', 'run_test'))
    assert denied.status == 'denied' and denied.reason == 'tool_not_authorized'


def test_fresh_fixture_family_validator():
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/verify_e3_set_fixtures.py')],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['ok'] is True
    assert report['roles'] == {'train': 4, 'dev': 1, 'heldout': 3, 'interrupted': 1}


def test_long_path_failure_excerpt_retains_numeric_bound():
    from replica_lab.runtime.procedures import parse_failure_bound
    from replica_lab.runtime.tools import FixtureWorkspace
    with tempfile.TemporaryDirectory() as directory:
        case = Path(directory)
        for _ in range(6):
            case = case / ('long-path-segment-' + 'x' * 70)
        shutil.copytree(ROOT / 'fixtures/e3_m0a_menus/cases/m0a_heldout_poll', case)
        receipt = FixtureWorkspace(case).run_test({})['receipt']
        bound = parse_failure_bound(receipt['stderr_excerpt'])
        assert bound == {'name': 'POLL_INTERVAL_MS', 'actual': '50', 'expected': '350'}
