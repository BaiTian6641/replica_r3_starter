"""E5 procedure candidate: verified execution, independent evaluation and publish."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from replica_lab.learning import ManifestRegistry
from replica_lab.runtime.learning_proposal import LearningProposal
from replica_lab.runtime.procedures import RepairProcedure, parse_failure_bound
from replica_lab.runtime.tools import FixtureWorkspace

ROOT = Path(__file__).resolve().parents[1]


def procedure():
    return RepairProcedure('repair-config', 'workspace/app_settings.py',
                           ('teacher:set_train_retry_01:step3',))


def run_case(case_id, procedure_value=procedure()):
    with tempfile.TemporaryDirectory() as directory:
        copy = Path(directory) / case_id
        shutil.copytree(ROOT / 'fixtures/e3_set/cases' / case_id, copy)
        tools = FixtureWorkspace(copy)
        result = procedure_value.execute(tools)
        verified = tools.verify_success().get('passed')
        return result, verified


def test_parse_failure_and_procedure_succeeds_on_unseen_case():
    assert parse_failure_bound('MAX_RETRIES=1 needs MAX_RETRIES >= 4') == {
        'name': 'MAX_RETRIES', 'actual': '1', 'expected': '4'}
    result, verified = run_case('set_heldout_timeout_06')
    assert result.status == 'succeeded' and result.final_exit_code == 0
    assert result.parsed['name'] == 'CONNECT_TIMEOUT_MS'
    assert verified is True
    assert len(result.receipts) == 4


def test_repair_procedure_does_not_claim_success_when_no_failure_is_disclosed():
    with tempfile.TemporaryDirectory() as directory:
        copy = Path(directory) / 'case'
        shutil.copytree(ROOT / 'fixtures/e3_set/cases/set_train_retry_01', copy)
        tools = FixtureWorkspace(copy)
        # Apply the reference repair first; the procedure must report not_required,
        # not a fabricated success receipt.
        (copy / 'workspace/app_settings.py').write_text(
            json.loads((copy / 'case.json').read_text())['reference_repair']['content'])
        result = procedure().execute(tools)
        assert result.status == 'not_required' and result.final_exit_code == 0
        assert result.parsed is None


def test_learning_proposal_binds_sources_and_plan():
    proposal = LearningProposal('proposal-1', None, ('event-1',), 'procedure',
                                'repair-config', {'train': 'set_train_*'},
                                {'cases': 4, 'tool_calls': 4}, 'host_tools',
                                'held-out fixture completion; no neural update')
    assert proposal.target_kind == 'procedure'
    assert len(proposal.digest) == 64
    with pytest.raises(ValueError):
        LearningProposal('bad', None, (), 'procedure', 'x', {'a': 1}, {'b': 2},
                         'host', 'plan')


def test_publish_accepted_procedure_and_reject_harmful_candidate(tmp_path):
    payload = {'base_revision': 'f97400052a43d642bbc6e9975e2397e3ae6a6b52',
               'identity_version': 'generic',
               'procedure': procedure().artifact().digest}
    from replica_lab.contracts import fingerprint
    registry = ManifestRegistry(tmp_path / 'registry')
    accepted = {'decision': 'accept', 'candidate_digest': fingerprint(payload),
                'parent_manifest_id': None, 'target': 'procedure'}
    manifest = registry.publish(payload, accepted, expected_parent=None)
    assert registry.active()['manifest_id'] == manifest['manifest_id']
    rejected = {'decision': 'reject', 'reason': 'harmful_repair',
                'candidate_digest': fingerprint(payload),
                'parent_manifest_id': manifest['manifest_id']}
    try:
        registry.publish(payload, rejected, expected_parent=manifest['manifest_id'])
    except PermissionError:
        pass
    else:
        raise AssertionError('rejected candidate must not publish')
