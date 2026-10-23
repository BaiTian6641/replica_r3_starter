"""Typed action parsing/validation, outcome frames and fixture tools."""

import json
from pathlib import Path
import shutil

import pytest

from replica_lab.runtime.actions import (MAX_PROPOSAL_BYTES, ActionProposal, OutcomeFrame,
                                         parse_action_proposal, validate_proposal)
from replica_lab.runtime.task_graph import Budget
from replica_lab.runtime.tools import (FixtureWorkspace, MAX_READ_BYTES, MAX_WRITE_BYTES,
                                       sha256_bytes)

from e3_runtime_helpers import make_case, make_state

ALLOWED = ('read_file', 'write_file', 'run_test')
REPO_ROOT = Path(__file__).resolve().parents[1]


class TestParsing:
    def test_exactly_one_object_with_metadata(self):
        raw = json.dumps({'kind': 'read', 'path': 'workspace/config.json',
                          'goal_ref': 'runtime-case',
                          'budget_estimate': {'tool_calls': 1},
                          'predicted_outcome': 'file contents'})
        parsed = parse_action_proposal(raw, ALLOWED, action_id='act-1',
                                       idempotency_key='ik-1')
        assert parsed.status == 'ok'
        assert parsed.proposal.action_id == 'act-1'
        assert parsed.proposal.goal_ref == 'runtime-case'
        assert parsed.proposal.predicted_outcome == 'file contents'
        assert dict(parsed.proposal.args) == {'path': 'workspace/config.json'}

    def test_code_fence_is_accepted(self):
        parsed = parse_action_proposal(
            '```json\n{"kind": "stop"}\n```', ALLOWED)
        assert parsed.status == 'ok' and parsed.kind == 'stop'

    @pytest.mark.parametrize('suffix', ['</s>', '<|im_end|>', '</s><|im_end|>',
                                        '<|endoftext|>', '<pad>'])
    def test_decoder_stop_tokens_are_transport_not_actions(self, suffix):
        raw = json.dumps({'kind': 'read', 'path': 'workspace/config.json'}) + suffix
        parsed = parse_action_proposal(raw, ALLOWED)
        assert parsed.status == 'ok' and parsed.kind == 'read'
        assert parsed.raw == raw.encode('utf-8')
        assert parse_action_proposal(raw + ' trailing prose', ALLOWED).status == 'denied'

    @pytest.mark.parametrize('raw,reason', [
        ('', 'empty_proposal'),
        ('not json at all', 'malformed_json'),
        ('{"kind": "stop"} {"kind": "read"}', 'multiple_json_objects'),
        ('{"kind": "stop"} trailing', 'multiple_json_objects'),
        ('[{"kind": "stop"}]', 'json_not_object'),
        ('{"kind": "delete"}', 'unknown_kind'),
        ('{"kind": "invoke_tool", "tool": "delete", "arguments": {}}', 'tool_not_authorized'),
        ('{"kind": "read", "path": "workspace/config.json", "foo": 1}',
         'unexpected_argument: [\'foo\']'),
        ('{"kind": "stop", "action_id": "forged"}', 'authority_field_supplied'),
        ('{"kind": "stop", "hypothetical": false}',
         'predicted_outcomes_are_hypothetical'),
        ('{"kind": "stop", "budget_estimate": []}', 'invalid_metadata'),
        ('{"kind": "write_file", "path": "workspace/config.json", "content": "x"}',
         'unknown_kind'),
    ])
    def test_denials_preserve_raw_bytes(self, raw, reason):
        parsed = parse_action_proposal(raw, ALLOWED)
        assert parsed.status == 'denied'
        assert parsed.reason == reason
        assert parsed.raw == raw.encode('utf-8')
        assert parsed.proposal is None

    def test_oversize_and_invalid_encoding(self):
        oversize = 'x' * (MAX_PROPOSAL_BYTES + 1)
        parsed = parse_action_proposal(oversize, ALLOWED)
        assert parsed.status == 'denied' and parsed.reason == 'oversize'
        invalid = parse_action_proposal(b'\xff\xfe', ALLOWED)
        assert invalid.status == 'denied' and invalid.reason == 'invalid_encoding'

    def test_read_requires_read_tool_authority(self):
        parsed = parse_action_proposal('{"kind": "read", "path": "workspace/config.json"}',
                                       ('write_file',))
        assert parsed.status == 'denied' and parsed.reason == 'tool_not_authorized'

    def test_invoke_tool_arguments_are_strict(self):
        missing = parse_action_proposal(
            '{"kind": "invoke_tool", "tool": "write_file", "arguments": {}}', ALLOWED)
        assert missing.status == 'denied' and 'Missing tool arguments' in missing.reason
        extra = parse_action_proposal(
            '{"kind": "invoke_tool", "tool": "read_file", "arguments": '
            '{"path": "workspace/config.json", "offset": 2}}', ALLOWED)
        assert extra.status == 'denied' and 'Unexpected tool arguments' in extra.reason


class TestValidation:
    def test_goal_subgoal_evidence_and_tool_scope(self, tmp_path):
        tools = make_case(tmp_path)
        state = make_state(tools)
        base = {'action_id': 'act-1', 'idempotency_key': 'ik-1', 'kind': 'read',
                'args': {'path': 'workspace/config.json'}, 'goal_ref': ''}
        proposal = ActionProposal(**base)
        registry = tools.registry()
        assert validate_proposal(proposal, state, registry, state.budget).ok
        wrong_goal = ActionProposal(**{**base, 'goal_ref': 'someone-else'})
        assert validate_proposal(wrong_goal, state, registry, state.budget).reason == \
            'goal_binding_mismatch'
        unknown_subgoal = ActionProposal(**{**base, 'subgoal_id': 'sg-x'})
        assert validate_proposal(unknown_subgoal, state, registry, state.budget).reason == \
            'unknown_subgoal'
        unknown_evidence = ActionProposal(**{**base, 'evidence_refs': ('ev-x',)})
        assert validate_proposal(unknown_evidence, state, registry, state.budget).reason == \
            'unknown_evidence_reference'
        invoke = ActionProposal(action_id='act-2', idempotency_key='ik-2',
                                kind='invoke_tool',
                                args={'tool': 'run_test', 'arguments': {}},
                                goal_ref='runtime-case')
        assert validate_proposal(invoke, state, None, state.budget).reason == \
            'tool_not_available'
        verify = ActionProposal(action_id='act-3', idempotency_key='ik-3', kind='verify',
                                args={'condition_id': 'missing'}, goal_ref='')
        assert validate_proposal(verify, state, registry, state.budget).reason == \
            'unknown_success_condition'

    def test_read_authority_and_budget(self, tmp_path):
        tools = make_case(tmp_path)
        state = make_state(tools, allowed_tools=('write_file',))
        read = ActionProposal(action_id='act-1', idempotency_key='ik-1', kind='read',
                              args={'path': 'workspace/config.json'}, goal_ref='')
        assert validate_proposal(read, state, tools.registry(), state.budget).reason == \
            'tool_not_authorized'
        broke = make_state(tools, budget=Budget(2, 0, 100, 100))
        write = ActionProposal(
            action_id='act-2', idempotency_key='ik-2', kind='invoke_tool',
            args={'tool': 'write_file',
                  'arguments': {'path': 'workspace/config.json', 'content': 'x'}},
            goal_ref='')
        validation = validate_proposal(write, broke, tools.registry(), broke.budget)
        assert validation.ok is False and validation.reason == 'budget_exhausted'


class TestOutcomeFrame:
    def test_actual_status_and_record(self, tmp_path):
        frame = OutcomeFrame(outcome_id='out-1', action_id='act-1', status='completed',
                             result={'ok': True}, verifier='host_tools',
                             spent={'tool_calls': 1}, observation_ids=('obs-1',),
                             receipt={'kind': 'run_test'})
        record = frame.to_record(3)
        assert record.actual is True and record.revision == 3
        assert record.receipt['kind'] == 'run_test'
        with pytest.raises(ValueError):
            OutcomeFrame(outcome_id='out-2', action_id='act-1', status='completed',
                         result={}, verifier='host', spent={}, actual=False)
        with pytest.raises(ValueError):
            OutcomeFrame(outcome_id='out-2', action_id='act-1', status='hypothetical',
                         result={}, verifier='host', spent={})


class TestFixtureWorkspace:
    def test_committed_fixture_loads_case_root_paths(self, tmp_path):
        source = REPO_ROOT / 'fixtures' / 'e3' / 'cases' / 'train_retry_01'
        target = tmp_path / 'train_retry_01'
        shutil.copytree(source, target)
        tools = FixtureWorkspace(target)
        assert tools.workspace_files == ('workspace/app_settings.py',)
        assert tools.workspace_entry == 'workspace/app_settings.py'
        assert tools.read_file({'path': 'workspace/app_settings.py'})['status'] == 'ok'
        assert tools.read_file({'path': 'app_settings.py'})['status'] == 'denied'
        assert 'RETRY_ATTEMPTS = 2' in tools.read_file(
            {'path': 'workspace/app_settings.py'})['content']

    def test_bare_workspace_entries_are_normalized(self, tmp_path):
        tools = make_case(tmp_path, prefix_paths=False)
        assert tools.workspace_files == ('workspace/config.json',)
        assert tools.workspace_entry == 'workspace/config.json'
        assert tools.read_file({'path': 'workspace/config.json'})['content'] == \
            '{"rate": 1.0}\n'

    def test_read_sandbox_and_bounds(self, tmp_path):
        tools = make_case(tmp_path)
        assert tools.read_file({'path': 'workspace/config.json'})['status'] == 'ok'
        for path in ('case.json', '../case.json', 'workspace/../case.json',
                     'workspace', '/workspace/config.json',
                     'evaluator/test_rate.py', 'workspace/evaluator/test_rate.py'):
            result = tools.read_file({'path': path})
            assert result['status'] in ('denied', 'error'), path
        (tools.workspace_dir / 'big.txt').write_text('x' * (MAX_READ_BYTES + 1))
        assert tools.read_file({'path': 'workspace/big.txt'})['error'] == 'file_too_large'
        outside = tmp_path / 'outside.txt'
        outside.write_text('secret')
        (tools.workspace_dir / 'link.txt').symlink_to(outside)
        assert tools.read_file({'path': 'workspace/link.txt'})['status'] == 'denied'

    def test_write_sandbox_and_bounds(self, tmp_path):
        tools = make_case(tmp_path)
        assert tools.write_file({'path': 'workspace/config.json',
                                 'content': '{"rate": 0.75}\n'})['status'] == 'ok'
        for path in ('case.json', '../case.json', 'workspace/../case.json',
                     'evaluator/test_rate.py', '/tmp/evil.txt'):
            result = tools.write_file({'path': path, 'content': 'x'})
            assert result['status'] == 'denied', path
        assert tools.write_file({'path': 'workspace/config.json',
                                 'content': 'é' * (MAX_WRITE_BYTES // 2 + 1)})['error'] == \
            'content_too_large'
        assert tools.write_file({'path': 'workspace/config.json',
                                 'content': 'x' * (MAX_WRITE_BYTES + 1)})['status'] == \
            'denied'
        (tools.workspace_dir / 'dir').mkdir()
        assert tools.write_file({'path': 'workspace/dir', 'content': 'x'})['error'] == \
            'target_not_regular'
        outside = tmp_path / 'outside.txt'
        outside.write_text('keep')
        (tools.workspace_dir / 'link.txt').symlink_to(outside)
        assert tools.write_file({'path': 'workspace/link.txt', 'content': 'x'})['status'] == \
            'denied'
        assert outside.read_text() == 'keep'

    def test_run_test_real_exit_code_and_independent_verify(self, tmp_path):
        tools = make_case(tmp_path)
        failing = tools.run_test({})
        assert failing['status'] == 'error'
        receipt = failing['receipt']
        assert receipt['kind'] == 'run_test'
        assert receipt['exit_code'] != 0
        assert receipt['command'][1] == 'evaluator/test_rate.py'
        assert receipt['cwd'] == str(tools.case_root)
        assert receipt['workspace_hashes_before'] == receipt['workspace_hashes_after']
        assert len(receipt['receipt_sha256']) == 64
        assert 'AssertionError' in (receipt['stderr_excerpt'] + receipt['stdout_excerpt'])
        tools.write_file({'path': 'workspace/config.json', 'content': '{"rate": 0.25}\n'})
        passing = tools.run_test({})
        assert passing['status'] == 'ok' and passing['receipt']['exit_code'] == 0
        verdict = tools.verify_success(['sc-1'])
        assert verdict['passed'] is True
        assert verdict['receipt']['kind'] == 'verify_success'
        assert verdict['receipt']['condition_ids'] == ['sc-1']
        assert verdict['receipt']['receipt_sha256'] != passing['receipt']['receipt_sha256']
        assert tools.run_test({'extra': 1})['status'] == 'denied'

    def test_reference_repair_never_leaks_through_tools(self, tmp_path):
        tools = make_case(tmp_path)
        assert tools.host_reference_repair()['content'] == '{"rate": 0.25}\n'
        observation = tools.read_file({'path': 'workspace/config.json'})
        assert '0.25' not in observation['content']
        assert '0.25' not in tools.run_test({})['receipt']['stdout_excerpt']


class TestWorkspaceStore:
    def test_roundtrip_and_binding_rejection(self, tmp_path):
        from replica_lab.runtime.workspace_store import WorkspaceStore
        from replica_lab.architecture.contracts import CoreState
        import torch
        store = WorkspaceStore(tmp_path / 'ws')
        core = CoreState(torch.ones(1, 2, 4), torch.ones(1, 2, dtype=torch.bool),
                         torch.zeros(1, 24), 'producer-1', 'a' * 64, 3, 2)
        store.save(core)
        loaded = store.load()
        assert loaded is not None and loaded.task_digest == 'a' * 64
        assert store.load(expected_task_digest='b' * 64) is None
        assert store.load(expected_revision=4) is None
        assert store.load(expected_revision=3, expected_producer='producer-1') is not None
        store.clear()
        assert store.load() is None
