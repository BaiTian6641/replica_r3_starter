"""TaskRuntime: independent success, corrections, budgets, modes and recovery."""

from pathlib import Path

import pytest
import torch

from replica_lab.architecture.contracts import CoreState
from replica_lab.runtime.loop import MODES
from replica_lab.runtime.task_graph import Budget, TaskPatch, apply_patch
from replica_lab.runtime.workspace_store import WorkspaceStore

from e3_runtime_helpers import (GOAL, REPAIR, ScriptedProposer, make_case, make_runtime,
                                output, read_action, run_test_action, workspace_for,
                                write_action)


def statuses(summary):
    return [item['outcome']['status'] for item in summary['trace']]


class TestIndependentSuccess:
    def test_only_a_passing_host_check_satisfies_conditions(self, tmp_path):
        tools = make_case(tmp_path)
        runtime = make_runtime(tools, tmp_path)
        proposer = ScriptedProposer(['{"kind": "stop"}', write_action(),
                                     '{"kind": "stop"}'])
        summary = runtime.run(proposer, max_decisions=3)
        assert summary['decisions'] == 3
        assert statuses(summary) == ['failed', 'completed', 'completed']
        assert summary['trace'][0]['success_claimed'] is True
        assert summary['trace'][0]['success_condition_met'] is False
        assert summary['success'] is True
        assert summary['satisfied_conditions'] == ['sc-1']
        assert runtime.state.is_complete(runtime.verifier) is True
        last = runtime.state.outcomes[-1]
        assert last.receipt['kind'] == 'verify_success'
        assert last.receipt['passed'] is True
        assert runtime.verifier.receipt_verified(last.receipt['receipt_sha256'])

    def test_failing_test_keeps_the_loop_and_plan_change_allowed(self, tmp_path):
        tools = make_case(tmp_path)
        runtime = make_runtime(tools, tmp_path)
        proposer = ScriptedProposer([
            run_test_action(), read_action(), write_action(), run_test_action(),
            '{"kind": "stop"}'])
        summary = runtime.run(proposer, max_decisions=5)
        assert summary['decisions'] == 5
        assert statuses(summary) == ['failed', 'completed', 'completed', 'completed',
                                     'completed']
        assert summary['trace'][2]['outcome']['result']['status'] == 'ok'
        assert summary['success'] is True
        assert 'AssertionError' in (proposer.prompts[1])

    def test_malformed_action_is_denied_and_loop_continues(self, tmp_path):
        tools = make_case(tmp_path)
        runtime = make_runtime(tools, tmp_path)
        proposer = ScriptedProposer(['not a JSON action', write_action(),
                                     '{"kind": "stop"}'])
        summary = runtime.run(proposer, max_decisions=3)
        assert statuses(summary)[0] == 'denied'
        assert summary['trace'][0]['outcome']['result']['reason'].startswith(
            'parse_denied: malformed_json')
        assert summary['success'] is True

    def test_pending_input_pauses_and_survives_fresh_process(self, tmp_path):
        tools = make_case(tmp_path)
        runtime = make_runtime(tools, tmp_path)
        summary = runtime.run(ScriptedProposer(['{"kind": "ask", "question": "which file?"}']),
                              max_decisions=2)
        assert summary['decisions'] == 1
        assert summary['pending_input'] is True
        assert runtime.state.paused() is True
        fresh = make_runtime(tools, tmp_path)
        again = fresh.run(ScriptedProposer([write_action()]), max_decisions=2)
        assert again['decisions'] == 0 and again['pending_input'] is True
        assert fresh.state.paused() is True


class TestCorrectionAndWorkspace:
    def test_correction_rejects_stale_workspace_and_rebuilds(self, tmp_path):
        tools = make_case(tmp_path)
        store = WorkspaceStore(tmp_path / 'ws')
        runtime = make_runtime(tools, tmp_path, workspace_store=store)
        first = ScriptedProposer([read_action()], workspace_factory=workspace_for)
        runtime.run(first, max_decisions=1)
        stored = store.load()
        assert stored is not None
        assert stored.task_digest == runtime.state.digest
        assert stored.revision == runtime.state.revision

        revision = runtime.state.revision
        with pytest.raises(ValueError):
            runtime.correct(author='model', note='not allowed')
        corrected = runtime.correct(note='host: reviewed the workspace',
                                    constraint={'constraint_id': 'con-host',
                                                'text': 'keep changes minimal'})
        assert corrected.revision == revision + 1
        assert runtime.previous_workspace is None
        assert 'host: reviewed the workspace' in corrected.notes

        second = ScriptedProposer(['{"kind": "stop"}'],
                                  workspace_factory=workspace_for)
        summary = runtime.run(second, max_decisions=1)
        assert second.requests[0].previous_workspace is None
        assert 'workspace_discarded: stale_revision' in summary['trace'][0]['notes']
        assert store.load() is not None  # rebuilt and rebound by this decision
        assert store.load().revision == runtime.state.revision

    def test_host_can_revise_known_claim_with_evidence_but_model_cannot(self, tmp_path):
        tools = make_case(tmp_path)
        runtime = make_runtime(tools, tmp_path)
        proposer = ScriptedProposer([read_action()])
        runtime.run(proposer, max_decisions=1)
        read_outcome = runtime.state.outcomes[-1]
        digest = read_outcome.result['sha256']
        assert any(ref.digest == digest for ref in runtime.state.evidence)
        patch = TaskPatch('add_claim', 'model', {
            'claim_id': 'cl-1', 'text': 'the config sets a rate', 'status': 'inferred'})
        result = apply_patch(runtime.state, patch, runtime.verifier)
        assert result.applied is True
        runtime._journal_patch(result)
        runtime.state = result.state
        denied = apply_patch(runtime.state, TaskPatch('revise_claim', 'model', {
            'claim_id': 'cl-1', 'text': 'known fact', 'status': 'known',
            'evidence_ids': []}), runtime.verifier)
        assert denied.applied is False and denied.reason == 'model_may_not_add_known_claim'
        corrected = runtime.correct(
            note='host confirmed the reading',
            evidence={'evidence_id': 'ev-host', 'kind': 'observation',
                      'digest': digest, 'origin': 'host:read'},
            claim={'claim_id': 'cl-1', 'text': 'the config sets a rate',
                   'status': 'known', 'evidence_ids': ['ev-host']})
        claim = corrected.claim_map()['cl-1']
        assert claim.status == 'known' and claim.evidence_ids == ('ev-host',)

    def test_workspace_is_persisted_at_post_decision_binding(self, tmp_path):
        tools = make_case(tmp_path)
        store = WorkspaceStore(tmp_path / 'ws')
        runtime = make_runtime(tools, tmp_path, workspace_store=store)
        proposer = ScriptedProposer([read_action(), '{"kind": "ask", "question": "q"}'],
                                    workspace_factory=workspace_for)
        summary = runtime.run(proposer, max_decisions=2)
        assert summary['decisions'] == 2
        assert proposer.requests[0].previous_workspace is None
        second = proposer.requests[1]
        assert second.previous_workspace is not None
        # The second request's workspace came from the first post-decision binding.
        assert second.previous_workspace.task_digest == second.task_digest
        assert second.previous_workspace.revision == second.revision
        assert 'workspace_discarded' not in summary['trace'][1]['notes']
        final = store.load()
        assert final is not None and final.revision == runtime.state.revision
        assert final.task_digest == runtime.state.digest


class TestSessionProposerContract:
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return '{"kind": "stop"}'

    class Lease:
        def __init__(self, prompt_ids, prompt_mask, core):
            self.frame = type('Frame', (), {
                'sense': type('Sense', (), {'prompt_ids': prompt_ids,
                                            'prompt_mask': prompt_mask})(),
                'core': core, 'decision_id': 'decision-1'})()
            self.released = False

        def release(self):
            self.released = True

    class Adapter:
        def __init__(self, producer='prod'):
            self.core = object()
            self.manifest = type('Manifest', (), {'digest': 'manifest-digest'})()
            self.producer = producer
            self.sense_calls = 0

        def core_producer_key(self):
            return self.producer

        def sense(self, prompt_ids, prompt_mask):
            self.sense_calls += 1
            return object()

    class Session:
        def __init__(self, adapter):
            self.adapter = adapter
            self.prepared = []
            self.generated = 0

        def prepare(self, **kwargs):
            self.prepared.append(kwargs)
            core = None
            if kwargs['core_impl'] != 'none':
                core = CoreState(torch.zeros(1, 2, 4),
                                 torch.ones(1, 2, dtype=torch.bool),
                                 torch.zeros(1, 24), 'prod', kwargs['task_digest'],
                                 kwargs['revision'], 2,
                                 'feedforward' if kwargs['core_impl'] == 'feedforward'
                                 else 'recurrent')
            return TestSessionProposerContract.Lease(
                kwargs['prompt_ids'], kwargs['prompt_mask'], core)

        def generate(self, lease, **kwargs):
            self.generated += 1
            return torch.cat([lease.frame.sense.prompt_ids, torch.tensor([[9]])], dim=1)

        def verify(self, lease):
            return None

    def _request(self, mode='full', previous=None):
        from replica_lab.runtime.loop import ProposalRequest
        return ProposalRequest(
            prompt_text='hello', task_features=(0.5,) * 24,
            previous_workspace=previous, mode=mode, prompt_token_budget=100,
            generated_token_budget=8, task_digest='d' * 64, revision=3)

    def _proposer(self, monkeypatch, core_impl='recurrent'):
        from replica_lab.runtime.loop import SessionProposer
        import replica_lab.architecture.loading as loading
        monkeypatch.setattr(loading, 'encode_prompt',
                            lambda tokenizer, text, device: torch.tensor([[1, 2, 3]]))
        adapter = self.Adapter()
        session = self.Session(adapter)
        proposer = SessionProposer(session, self.Tokenizer(), task_id='case', goal='goal',
                                   scope=None, device='cpu', core_impl=core_impl)
        return proposer, session, adapter

    def _core(self, producer='prod', revision=3):
        return CoreState(torch.ones(1, 2, 4), torch.ones(1, 2, dtype=torch.bool),
                         torch.zeros(1, 24), producer, 'd' * 64, revision, 2)

    def test_core_disabled_passes_none_features_and_previous(self, monkeypatch):
        proposer, session, adapter = self._proposer(monkeypatch)
        result = proposer(self._request('core_disabled', previous=self._core()))
        kwargs = session.prepared[0]
        assert kwargs['core_impl'] == 'none'
        assert kwargs['task_features'] is None
        assert kwargs['previous_core'] is None
        assert kwargs['enabled_branches'] == ('memory',)
        assert kwargs['task_digest'] == 'd' * 64 and kwargs['revision'] == 3
        assert result.workspace is None
        assert adapter.sense_calls == 0

    def test_producer_mismatch_drops_workspace_with_note(self, monkeypatch):
        proposer, session, adapter = self._proposer(monkeypatch)
        result = proposer(self._request('full', previous=self._core(producer='old')))
        assert session.prepared[0]['previous_core'] is None
        assert 'workspace_discarded: producer_mismatch' in result.notes

    def test_matching_producer_and_zero_mode(self, monkeypatch):
        proposer, session, adapter = self._proposer(monkeypatch)
        proposer(self._request('full', previous=self._core()))
        assert session.prepared[0]['previous_core'] is not None
        proposer(self._request('workspace_zero', previous=self._core()))
        zeroed = session.prepared[1]['previous_core']
        assert zeroed is not None and float(zeroed.workspace.abs().sum()) == 0.0

    def test_ff_core_and_extra_inference_controls(self, monkeypatch):
        proposer, session, adapter = self._proposer(monkeypatch)
        proposer(self._request('ff_core', previous=self._core()))
        assert session.prepared[0]['core_impl'] == 'feedforward'
        assert session.prepared[0]['previous_core'] is None
        result = proposer(self._request('extra_inference'))
        assert session.prepared[1]['enabled_branches'] == ('memory',)
        assert adapter.sense_calls == 1
        assert 'extra_inference_passes=1' in result.notes


class TestCliContract:
    def _module(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / 'scripts' / 'run_e3_task.py'
        spec = importlib.util.spec_from_file_location('run_e3_task', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_budget_flags_are_cumulative_and_reported(self, tmp_path):
        module = self._module()
        tools = make_case(tmp_path)
        parser = module.build_parser()
        args = parser.parse_args([
            '--case', str(tools.case_root), '--journal', str(tmp_path / 'task.sqlite'),
            '--output', str(tmp_path / 'out'), '--max-decisions', '7',
            '--max-tool-calls', '5', '--max-prompt-tokens', '448',
            '--max-new-tokens', '128'])
        arguments = module.resolve_budget(args)
        assert arguments == {'decisions': 7, 'tool_calls': 5,
                             'prompt_tokens': 3136, 'generated_tokens': 896}
        state = module.build_initial_state(tools, args)
        assert state.budget.to_dict() == {
            'max_decisions': 7, 'max_tool_calls': 5, 'max_prompt_tokens': 3136,
            'max_generated_tokens': 896, 'used_decisions': 0, 'used_tool_calls': 0,
            'used_prompt_tokens': 0, 'used_generated_tokens': 0}
        binding = module.journal_binding(tools, args)
        assert 'budget' in binding['component_versions']
        assert module.build_core_spec(args).width == 256

    def test_cli_case_projection_excludes_host_only_fields(self, tmp_path):
        module = self._module()
        tools = make_case(tmp_path)
        _, case = module.read_case(tools.case_root)
        assert 'reference_repair' not in case
        assert 'variant' not in case
        assert 'workspace_entry' not in case  # not projected at all
        assert case['goal'] == tools.goal

    def test_seed_policy_is_deterministic_and_idempotent(self, tmp_path):
        from replica_lab.memory_system import EligibilityScope, RecordLedger
        module = self._module()
        scope = EligibilityScope('runtime-case', 'e3', 'test', 0,
                                 'assistant_for_source', None)
        record = module.seed_policy_record(scope)
        assert record.text == module.SEED_POLICY_TEXT
        assert record.observed_at == 0 and record.branch_id == 'e3'
        path = tmp_path / 'ledger.sqlite'
        with RecordLedger(path) as ledger:
            assert ledger.admit(record, authorized_subject=scope.subject_id,
                                authorized_branch=scope.branch_id) is True
        with RecordLedger(path) as ledger:
            assert ledger.admit(record, authorized_subject=scope.subject_id,
                                authorized_branch=scope.branch_id) is False
            snapshot = ledger.snapshot(scope)
            assert [item.record_id for item in snapshot.records] == [record.record_id]


class TestBudgets:
    def test_decision_budget_exhaustion_is_clean(self, tmp_path):
        tools = make_case(tmp_path)
        runtime = make_runtime(tools, tmp_path,
                               state_changes={'budget': Budget(1, 4, 4096, 512)})
        summary = runtime.run(ScriptedProposer(['{"kind": "stop"}']), max_decisions=5)
        assert summary['decisions'] == 1
        assert summary['budget_exhausted'] is True
        assert summary['success'] is False
        assert runtime.state.budget.used_decisions == 1

    @pytest.mark.parametrize('prompt_tokens,reason', [
        (5000, 'per_decision_token_budget_exceeded'),
        (1000, 'cumulative_token_budget_exceeded'),
    ])
    def test_token_budget_overrun_records_denial_and_stops(self, tmp_path,
                                                           prompt_tokens, reason):
        tools = make_case(tmp_path)
        budget = Budget(2, 4, 100, 100) if reason.startswith('cumulative') \
            else Budget(2, 4, 4096, 512)
        runtime = make_runtime(tools, tmp_path, state_changes={'budget': budget})
        proposal = output('{"kind": "stop"}', prompt_tokens=prompt_tokens)

        class OverBudgetProposer:
            def __init__(self):
                self.requests = []

            def __call__(self, request):
                self.requests.append(request)
                return proposal

        proposer = OverBudgetProposer()
        summary = runtime.run(proposer, max_decisions=2)
        assert summary['decisions'] == 1
        assert summary['budget_exhausted'] is True
        assert reason in summary['trace'][0]['outcome']['result']['reason']
        assert runtime.state.budget.used_decisions == 1
        assert runtime.state.budget.used_prompt_tokens == 0
        # The journal still reconstructs a single consistent state.
        assert runtime.journal.load().state.digest == runtime.state.digest


class TestModesAndPrompt:
    @pytest.mark.parametrize('mode', MODES)
    def test_every_mode_enforces_workspace_authority(self, tmp_path, mode):
        tools = make_case(tmp_path)
        shuffled = CoreState(torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.bool),
                             torch.zeros(1, 24), 'producer-1', 'other-digest', 0, 2)
        runtime = make_runtime(tools, tmp_path, mode=mode, shuffled_workspace=shuffled)
        case_before = (tools.case_root / 'case.json').read_text()
        proposer = ScriptedProposer([
            write_action(path='workspace/../case.json', content='owned')])
        summary = runtime.run(proposer, max_decisions=1)
        assert proposer.requests[0].mode == mode
        assert summary['trace'][0]['outcome']['status'] == 'denied'
        assert 'workspace' in summary['trace'][0]['outcome']['result']['error']
        assert (tools.case_root / 'case.json').read_text() == case_before
        assert summary['success'] is False

    def test_prompt_has_goal_history_schema_and_no_expected_values(self, tmp_path):
        tools = make_case(tmp_path)
        runtime = make_runtime(tools, tmp_path)
        proposer = ScriptedProposer([read_action(), '{"kind": "stop"}'])
        summary = runtime.run(proposer, max_decisions=2)
        first, second = proposer.prompts
        assert 'AUTHORITATIVE GOAL (immutable): ' + GOAL in first
        assert 'SUCCESS CONDITIONS' in first
        assert 'ALLOWED TOOLS: read_file, write_file, run_test' in second
        assert 'ACTION/OBSERVATION HISTORY' in second
        assert 'LAST OBSERVATION' in second
        assert '"rate": 1.0' in second
        assert '"kind": "stop"' in second
        assert '{"kind": "invoke_tool", "tool": "run_test", "arguments": {}}' in second
        assert len(second) <= runtime.max_prompt_tokens * 4
        assert REPAIR.strip() not in first + second
        assert '0.25' not in first + second
        assert runtime.state.goal == GOAL
        assert summary['trace'][1]['state_digest'] == runtime.state.digest

    def test_revision_order_is_intent_outcome_patch(self, tmp_path):
        tools = make_case(tmp_path)
        runtime = make_runtime(tools, tmp_path)
        proposer = ScriptedProposer([
            '{"kind": "revise", "note": "focus on workspace/config.json"}',
            '{"kind": "stop"}'])
        summary = runtime.run(proposer, max_decisions=2)
        assert summary['trace'][0]['outcome']['status'] == 'completed'
        assert runtime.state.notes[-1] == 'focus on workspace/config.json'
        events = runtime.journal.events()
        kinds = [event.kind for event in events]
        intent = kinds.index('action_intent')
        outcome = kinds.index('action_outcome', intent)
        patch = kinds.index('patch', outcome)
        assert intent < outcome < patch
        assert runtime.journal.load().state.notes[-1] == 'focus on workspace/config.json'

    def test_proposer_notes_are_merged_into_the_trace(self, tmp_path):
        tools = make_case(tmp_path)
        runtime = make_runtime(tools, tmp_path)

        class NoteProposer:
            def __call__(self, request):
                return output(read_action(), notes=(
                    'workspace_discarded: producer_mismatch',))

        summary = runtime.run(NoteProposer(), max_decisions=1)
        assert 'workspace_discarded: producer_mismatch' in summary['trace'][0]['notes']

    def test_summary_is_stable_and_complete(self, tmp_path):
        tools = make_case(tmp_path)
        runtime = make_runtime(tools, tmp_path)
        summary = runtime.run(ScriptedProposer(['{"kind": "stop"}']), max_decisions=1)
        stable = runtime.summary()
        assert stable['digest'] == summary['state_digest'] == runtime.state.digest
        assert stable['revision'] == summary['revision']
        assert stable['journal_head'] == runtime.journal.head
        assert stable['budget'] == runtime.state.budget.to_dict()
        assert stable['success_conditions'] == ['sc-1']
        assert stable['actions'][0]['kind'] == 'stop'
