"""State-dependent action-space guard: no repair selection or evaluator access."""
import json

from tests.e3_runtime_helpers import (ScriptedProposer, make_case, make_runtime,
                                      read_action, write_action)


def run(tools, tmp_path, raws, *, enabled=True):
    runtime = make_runtime(tools, tmp_path, action_guard=enabled)
    proposer = ScriptedProposer(raws)
    summary = runtime.run(proposer, max_decisions=len(raws))
    return runtime, summary, proposer


def test_repeated_read_is_denied_but_explicit_disable_allows_it(tmp_path):
    tools = make_case(tmp_path)
    runtime, summary, proposer = run(tools, tmp_path, [read_action(), read_action()])
    assert [item['outcome']['status'] for item in summary['trace']] == ['completed', 'denied']
    assert 'action_guard' in summary['trace'][1]['outcome']['result']['reason']
    assert 'ACTION-SPACE GUARD' in proposer.prompts[0]
    assert 'read of: workspace/config.json' in proposer.prompts[1]

    tools2 = make_case(tmp_path / 'off')
    _, summary2, _ = run(tools2, tmp_path / 'off-run',
                         [read_action(), read_action()], enabled=False)
    assert [item['outcome']['status'] for item in summary2['trace']] == ['completed', 'completed']


def test_failed_test_cannot_repeat_until_a_successful_mutation(tmp_path):
    tools = make_case(tmp_path)
    test = json.dumps({'kind': 'invoke_tool', 'tool': 'run_test', 'arguments': {}})
    raws = [test, test, write_action(), test]
    runtime, summary, proposer = run(tools, tmp_path, raws)
    statuses = [item['outcome']['status'] for item in summary['trace']]
    assert statuses[0] == 'failed'
    assert statuses[1] == 'denied' and 'action_guard' in summary['trace'][1]['outcome']['result']['reason']
    assert statuses[2] == 'completed'
    assert statuses[3] in ('completed', 'failed')  # actual evaluator decides; no fabricated success
    assert 'run_test (a test already failed' in proposer.prompts[1]
