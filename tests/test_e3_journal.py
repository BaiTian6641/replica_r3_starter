"""Journal: binding, hash chain, restart reconstruction, reconciliation, tamper."""

import json
import sqlite3

import pytest
import torch

from replica_lab.architecture.contracts import CoreState, fingerprint
from replica_lab.runtime.journal import (BindingMismatch, JournalFull, JournalTampered,
                                         PendingIntent, TaskJournal)
from replica_lab.runtime.loop import TaskRuntime
from replica_lab.runtime.task_graph import (ActionRecord, Budget, ExactTaskState,
                                            OutcomeRecord, SuccessCondition)
from replica_lab.runtime.tools import FixtureWorkspace, sha256_bytes
from replica_lab.runtime.workspace_store import WorkspaceStore

GOAL = 'Make the evaluator pass by setting the required value.'
REPAIR = '{"rate": 0.25}\n'


def make_case(root):
    case_root = root / 'cases' / 'journal-case'
    (case_root / 'workspace').mkdir(parents=True)
    (case_root / 'evaluator').mkdir(parents=True)
    (case_root / 'workspace' / 'config.json').write_text('{"rate": 1.0}\n')
    (case_root / 'evaluator' / 'test_rate.py').write_text(
        'import json, pathlib\n'
        'cfg = json.loads(pathlib.Path("workspace/config.json").read_text())\n'
        'assert cfg["rate"] == 0.25, "config fails the bound check"\n'
        'print("ok")\n')
    case = {
        'format': 'replica.e3.fixture.v1', 'case_id': 'journal-case', 'role': 'train',
        'goal': GOAL,
        'success_condition': {'condition_id': 'sc-1', 'description': 'evaluator exits 0'},
        'constraints': ['Only workspace/ is writable'],
        'allowed_tools': ['read_file', 'write_file', 'run_test'],
        'workspace_files': ['config.json'], 'evaluator': 'evaluator/test_rate.py',
        'workspace_entry': 'config.json',
        'reference_repair': {'path': 'workspace/config.json', 'content': REPAIR},
        'variant': {},
    }
    (case_root / 'case.json').write_text(json.dumps(case, indent=2))
    return FixtureWorkspace(case_root)


def make_binding(tools, **changes):
    values = {'task_id': tools.case_id, 'subject': 'subject-1', 'branch': 'e3',
              'purpose': 'test', 'source_epoch': 0, 'goal_digest': fingerprint(tools.goal),
              'component_versions': {'runtime': 'v1'}}
    values.update(changes)
    return values


def make_state(tools, **changes):
    values = dict(
        task_id=tools.case_id, subject='subject-1', branch='e3', purpose='test',
        source_epoch=0, goal=tools.goal,
        success_conditions=(SuccessCondition('sc-1', 'evaluator exits 0'),),
        constraints=(), subgoals=(), claims=(), evidence=(), obligations=(), actions=(),
        outcomes=(), notes=(), budget=Budget(8, 16, 4096, 512),
        binding={'runtime': 'v1'}, allowed_tools=('read_file', 'write_file', 'run_test'))
    values.update(changes)
    return ExactTaskState(**values)


def make_runtime(tools, tmp_path, *, workspace_store=None, **changes):
    journal = TaskJournal.open(tmp_path / 'task.sqlite', make_binding(tools))
    runtime = TaskRuntime(tools, journal, workspace_store=workspace_store)
    if not journal.initialize_done():
        runtime.initialize(make_state(tools, **changes))
    return runtime


def count_writes(tools):
    calls = {'write': 0, 'read': 0, 'test': 0}
    original_write, original_read, original_test = (tools.write_file, tools.read_file,
                                                    tools.run_test)

    def write(args):
        calls['write'] += 1
        return original_write(args)

    def read(args):
        calls['read'] += 1
        return original_read(args)

    def test(args):
        calls['test'] += 1
        return original_test(args)

    tools.write_file, tools.read_file, tools.run_test = write, read, test
    return calls


class TestBinding:
    def test_open_writes_binding_once_and_verifies(self, tmp_path):
        tools = make_case(tmp_path)
        path = tmp_path / 'task.sqlite'
        first = TaskJournal.open(path, make_binding(tools))
        second = TaskJournal.open(path, make_binding(tools))
        assert first.binding == second.binding
        assert second.epoch == 0
        binding_events = [event for event in second.events() if event.kind == 'binding']
        assert len(binding_events) == 1

    def test_binding_mismatch_rejected(self, tmp_path):
        tools = make_case(tmp_path)
        path = tmp_path / 'task.sqlite'
        TaskJournal.open(path, make_binding(tools))
        with pytest.raises(BindingMismatch):
            TaskJournal.open(path, make_binding(tools, subject='someone-else'))
        with pytest.raises(BindingMismatch):
            TaskJournal.open(path, make_binding(tools, source_epoch=3))
        with pytest.raises(BindingMismatch):
            TaskJournal.open(path, make_binding(tools, component_versions={'runtime': 'v2'}))
        with pytest.raises(BindingMismatch):
            TaskJournal.open(path, make_binding(
                tools, goal_digest=fingerprint('a different goal')))


class TestChain:
    def test_head_links_and_genesis(self, tmp_path):
        tools = make_case(tmp_path)
        journal = TaskJournal.open(tmp_path / 'task.sqlite', make_binding(tools))
        journal.initialize(make_state(tools))
        events = journal.events()
        assert events[0].prev_hash == '0' * 64
        assert journal.head == events[-1].hash
        assert all(events[i].hash == events[i + 1].prev_hash
                   for i in range(len(events) - 1))

    def test_idempotent_append_by_key(self, tmp_path):
        tools = make_case(tmp_path)
        journal = TaskJournal.open(tmp_path / 'task.sqlite', make_binding(tools))
        first = journal.append('budget', {'denied': 'decision'}, idempotency_key='k-1')
        second = journal.append('budget', {'denied': 'decision'}, idempotency_key='k-1')
        assert first.event_id == second.event_id
        assert len(journal.events()) == 2  # binding + one budget event
        with pytest.raises(ValueError):
            journal.append('revision', {'revision': 1}, idempotency_key='k-1')

    def test_event_count_bound(self, tmp_path):
        tools = make_case(tmp_path)
        journal = TaskJournal(tmp_path / 'task.sqlite', max_events=3)
        journal.append('binding', {'binding': make_binding(tools)})
        with pytest.raises(JournalFull):
            for index in range(5):
                journal.append('budget', {'n': index})

    def test_tamper_rejection(self, tmp_path):
        tools = make_case(tmp_path)
        path = tmp_path / 'task.sqlite'
        journal = TaskJournal.open(path, make_binding(tools))
        journal.initialize(make_state(tools))
        journal.close()
        db = sqlite3.connect(path)
        db.execute("UPDATE events SET payload = replace(payload, 'Make the evaluator', "
                   "'Break the evaluator') WHERE kind = 'init'")
        db.commit()
        db.close()
        with pytest.raises(JournalTampered):
            TaskJournal.open(path, make_binding(tools))
        with pytest.raises(JournalTampered):
            TaskJournal(path).load()


class TestReconstruction:
    def test_restart_rebuilds_exact_state(self, tmp_path):
        tools = make_case(tmp_path)
        runtime = make_runtime(tools, tmp_path)

        class Proposer:
            def __init__(self, raws):
                self.raws = list(raws)

            def __call__(self, request):
                raw = self.raws.pop(0)
                return _output(raw)

        def _output(raw):
            from replica_lab.runtime.loop import ProposalOutput
            return ProposalOutput(raw_text=raw, prompt_tokens=11,
                                  generated_token_ids=(7, 8), generated_tokens=2,
                                  frame_digest='frame', workspace=None,
                                  workspace_stamp=None, wall_seconds=0.0)

        first = Proposer([
            '{"kind": "read", "path": "workspace/config.json"}',
            '{"kind": "invoke_tool", "tool": "write_file", "arguments": '
            '{"path": "workspace/config.json", "content": "{\\"rate\\": 0.25}\\n"}}',
        ])
        summary = runtime.run(first, max_decisions=2)
        assert summary['decisions'] == 2
        live_digest = runtime.state.digest
        live_actions = [a.to_dict() for a in runtime.state.actions]

        reopened = TaskRuntime(tools, TaskJournal.open(tmp_path / 'task.sqlite',
                                                       make_binding(tools)))
        snapshot = reopened.journal.load()
        assert snapshot.state.digest == live_digest
        assert snapshot.pending == ()
        assert [a.to_dict() for a in snapshot.state.actions] == live_actions
        assert snapshot.state.outcomes[0].result['kind'] == 'read_file'
        assert snapshot.verifier.receipt_verified(
            snapshot.state.outcomes[0].result['sha256'])

    def test_pending_intents_survive_crash(self, tmp_path):
        tools = make_case(tmp_path)
        journal = TaskJournal.open(tmp_path / 'task.sqlite', make_binding(tools))
        journal.initialize(make_state(tools))
        journal.append('action_intent', {
            'action_id': 'act-crash', 'kind': 'invoke_tool', 'tool': 'write_file',
            'args': {'tool': 'write_file', 'arguments': {
                'path': 'workspace/config.json', 'content': REPAIR}},
            'authorized': True, 'idempotency_key': 'ik-1', 'subgoal_id': None,
            'content': REPAIR, 'content_sha256': sha256_bytes(REPAIR.encode()),
        }, idempotency_key='intent:ik-1')
        pending = journal.pending_intents()
        assert len(pending) == 1
        assert isinstance(pending[0], PendingIntent)
        assert pending[0].content_sha256 == sha256_bytes(REPAIR.encode())


class TestRecovery:
    def _journal_with_write_intent(self, tmp_path, *, pre_execute):
        tools = make_case(tmp_path)
        calls = count_writes(tools)
        journal = TaskJournal.open(tmp_path / 'task.sqlite', make_binding(tools))
        journal.initialize(make_state(tools))
        journal.append('action_intent', {
            'action_id': 'act-w', 'kind': 'invoke_tool', 'tool': 'write_file',
            'args': {'tool': 'write_file', 'arguments': {
                'path': 'workspace/config.json', 'content': REPAIR}},
            'authorized': True, 'idempotency_key': 'ik-w', 'subgoal_id': None,
            'content': REPAIR, 'content_sha256': sha256_bytes(REPAIR.encode()),
        }, idempotency_key='intent:ik-w')
        if pre_execute:
            tools.write_file({'path': 'workspace/config.json', 'content': REPAIR})
        return tools, journal, calls

    def test_landed_write_is_never_reexecuted(self, tmp_path):
        tools, journal, calls = self._journal_with_write_intent(tmp_path, pre_execute=True)
        calls['write'] = 0  # the setup write simulates the landed effect
        runtime = TaskRuntime(tools, journal)
        assert runtime.recover() == 1
        assert calls['write'] == 0  # content hash matched; no re-execution
        snapshot = journal.load()
        assert snapshot.pending == ()
        outcome = snapshot.state.outcomes[-1]
        assert outcome.status == 'completed'
        assert outcome.result['reconciled'] is True
        assert snapshot.state.actions[-1].state == 'completed'

    def test_unlanded_write_executes_exactly_once(self, tmp_path):
        tools, journal, calls = self._journal_with_write_intent(tmp_path, pre_execute=False)
        runtime = TaskRuntime(tools, journal)
        assert runtime.recover() == 1
        assert calls['write'] == 1
        assert tools.file_hash_if_exists('workspace/config.json') == sha256_bytes(
            REPAIR.encode())
        assert journal.recover() == tuple()  # second pass reconciles nothing
        assert calls['write'] == 1
        outcomes = [event for event in journal.events() if event.kind == 'action_outcome']
        assert len(outcomes) == 1

    def test_recorded_outcome_is_not_pending_and_never_reruns(self, tmp_path):
        tools = make_case(tmp_path)
        calls = count_writes(tools)
        journal = TaskJournal.open(tmp_path / 'task.sqlite', make_binding(tools))
        journal.initialize(make_state(tools))
        journal.append('action_intent', {
            'action_id': 'act-r', 'kind': 'read', 'tool': None,
            'args': {'path': 'workspace/config.json'}, 'authorized': True,
            'idempotency_key': 'ik-r', 'subgoal_id': None,
        }, idempotency_key='intent:ik-r')
        outcome = OutcomeRecord(
            outcome_id='out-r', action_id='act-r', status='completed', actual=True,
            result={'status': 'ok', 'kind': 'read', 'sha256': sha256_bytes(b'x')},
            revision=1, receipt={'kind': 'run_test', 'exit_code': 0,
                                 'receipt_sha256': sha256_bytes(b'receipt')})
        journal.append('action_outcome', {'outcome': outcome.to_dict(),
                                          'spent': {'tool_calls': 1}},
                       idempotency_key='outcome:act-r')
        assert journal.pending_intents() == ()
        runtime = TaskRuntime(tools, journal)
        assert runtime.recover() == 0
        assert calls['read'] == 0
        assert journal.load().state.outcomes[-1].receipt['kind'] == 'run_test'

    def test_read_only_rerun_only_without_receipt(self, tmp_path):
        tools = make_case(tmp_path)
        calls = count_writes(tools)
        journal = TaskJournal.open(tmp_path / 'task.sqlite', make_binding(tools))
        journal.initialize(make_state(tools))
        journal.append('action_intent', {
            'action_id': 'act-r', 'kind': 'read', 'tool': None,
            'args': {'path': 'workspace/config.json'}, 'authorized': True,
            'idempotency_key': 'ik-r',
            'subgoal_id': None,
        }, idempotency_key='intent:ik-r')
        runtime = TaskRuntime(tools, journal)
        assert runtime.recover() == 1
        assert calls['read'] == 1  # no receipt existed, so the read re-ran
        snapshot = journal.load()
        assert snapshot.pending == ()
        assert snapshot.state.outcomes[-1].status == 'completed'
        # A second recovery finds no pending intent and never re-runs the tool.
        assert TaskRuntime(tools, journal).recover() == 0
        assert calls['read'] == 1


class TestCorrection:
    def test_correction_bumps_revision_and_invalidates_workspace(self, tmp_path):
        tools = make_case(tmp_path)
        store = WorkspaceStore(tmp_path / 'ws')
        runtime = make_runtime(tools, tmp_path, workspace_store=store)
        core = CoreState(torch.randn(1, 2, 4, generator=torch.Generator().manual_seed(3)),
                         torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 24),
                         'producer-1', runtime.state.digest, runtime.state.revision, 2)
        store.save(core)
        previous, reason = runtime._load_previous_workspace(runtime.state)
        assert previous is not None and reason is None

        revision_before = runtime.state.revision
        corrected = runtime.correct(note='host: constraint added after review',
                                    constraint={'constraint_id': 'con-0',
                                                'text': 'never write outside workspace/'})
        assert corrected.revision == revision_before + 1
        assert any(c.constraint_id == 'con-0' for c in corrected.constraints)
        assert 'host: constraint added after review' in corrected.notes
        stale, reason = runtime._load_previous_workspace(runtime.state)
        assert stale is None and reason == 'stale_revision'

    def test_correction_requires_host_authority(self, tmp_path):
        tools = make_case(tmp_path)
        runtime = make_runtime(tools, tmp_path)
        with pytest.raises(ValueError):
            runtime.correct(author='model', note='nope')

    def test_repeated_correction_is_idempotent(self, tmp_path):
        tools = make_case(tmp_path)
        runtime = make_runtime(tools, tmp_path)
        payload = {'constraint_id': 'con-1', 'text': 'only workspace/ is writable'}
        first = runtime.journal.correct(note='reviewed', constraint=payload)
        revision = runtime.journal.load().state.revision
        second = runtime.journal.correct(note='reviewed', constraint=payload)
        assert second.event_id == first.event_id
        assert runtime.journal.load().state.revision == revision
        corrections = [event for event in runtime.journal.events()
                       if event.kind == 'correction']
        assert len(corrections) == 1

    def test_invalidation_marks_claim_unknown(self, tmp_path):
        tools = make_case(tmp_path)
        runtime = make_runtime(tools, tmp_path)
        from replica_lab.runtime.task_graph import TaskPatch, apply_patch
        patch = TaskPatch('add_claim', 'model', {
            'claim_id': 'cl-1', 'text': 'the config is fine', 'status': 'inferred'})
        result = apply_patch(runtime.state, patch, runtime.verifier)
        assert result.applied is True
        runtime.journal.append('patch', {'patch': patch.to_dict(),
                                         'result': {'applied': True, 'reason': 'applied'}})
        runtime.journal.invalidate(claim_id='cl-1', reason='contradicted by receipt')
        state = runtime.journal.load().state
        assert state.claims[0].status == 'unknown'
        assert state.claims[0].claim_id == 'cl-1'
