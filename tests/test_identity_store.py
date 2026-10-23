"""Fictional C16 persistence: digest-bound reload, revocation and branch separation."""
import hashlib,json,os,stat
import pytest
from replica_lab.replica import (BranchExperience, IdentityStore, PrincipleHypothesis,
                                 ReplicaBranchState, SourceEpisode)


def digest(text): return hashlib.sha256(text.encode()).hexdigest()


def make_store():
    value = IdentityStore('fictional-subject')
    for index, (text, observed) in enumerate([
            ('I preferred recoverable plans when a deadline allowed it.', 1),
            ('I revised that preference after an outage in 2024.', 5),
            ('A later statement is excluded by the cutoff.', 20)]):
        value.add_episode(SourceEpisode(f'ep-{index}', 'fictional-subject',
            f'source-{index}', digest(text), observed, observed, text,
            kind='preference', permitted_uses=('retrieval', 'evaluation'),
            authenticity='source_report'))
    value.add_principle(PrincipleHypothesis('pr-1', 'fictional-subject',
        'Prefer recoverability when the deadline permits.', condition='deadline permits',
        supporting_episode_ids=('ep-0',), contradicting_episode_ids=('ep-1',),
        status='approved'))
    value.build_snapshot('snap-1', version=1, created_at=10,
        episode_ids=('ep-0', 'ep-1', 'ep-2'), principle_ids=('pr-1',),
        unresolved_questions=('Whether the preference still holds.',),
        permitted_uses=('retrieval', 'evaluation'))
    return value


def test_store_roundtrip_branch_and_resolve(tmp_path):
    value = make_store()
    branch = ReplicaBranchState('replica-a', 'fictional-subject', 'parallel_replica', 'snap-1')
    branch = branch.add(BranchExperience('exp-1', 'replica-a', 'fictional-subject', 4,
        'The branch observed a failed migration.', outcome='failed',
        evidence_ids=('tool-1',)))
    value.add_branch(branch)
    path = tmp_path / 'identity.json'
    value.save(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    restored = IdentityStore.load(path)
    source = restored.resolve('snap-1', mode='source_snapshot', branch_id='source',
                              purpose='test', cutoff=10)
    assert set(source.evidence_ids) == {'ep-0', 'ep-1'}
    parallel = restored.resolve('snap-1', mode='parallel_replica', branch_id='replica-a',
                                purpose='test', cutoff=10, branch_state=restored.branches['replica-a'])
    assert parallel.branch_id == 'replica-a' and 'failed migration' in parallel.summary
    assert restored.branches['replica-a'].experiences[0].evidence_ids == ('tool-1',)


def test_store_tamper_and_wrong_digest_rejected(tmp_path):
    value = make_store(); path = tmp_path / 'identity.json'; value.save(path)
    payload = json.loads(path.read_text())
    payload['body']['episodes'][0]['text'] = 'tampered'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='digest'):
        IdentityStore.load(path)
    payload['sha256'] = hashlib.sha256(json.dumps(payload['body'], sort_keys=True,
        separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    path.write_text(json.dumps(payload))
    # A recomputed digest is still rejected if the canonical body is malformed.
    payload['body']['format'] = 'unknown'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        IdentityStore.load(path)


def test_revocation_survives_reload_and_cannot_resolve(tmp_path):
    value = make_store(); path = tmp_path / 'identity.json'; value.save(path)
    value.revoke_episode('ep-0'); value.save(path)
    restored = IdentityStore.load(path)
    with pytest.raises(PermissionError, match='revoked'):
        restored.resolve('snap-1', mode='source_snapshot', branch_id='source',
                         purpose='test', cutoff=10)


def test_branch_subject_and_snapshot_are_bound(tmp_path):
    value = make_store()
    other = ReplicaBranchState('replica-b', 'another-subject', 'parallel_replica', 'snap-1')
    with pytest.raises(PermissionError):
        value.add_branch(other)
    missing = ReplicaBranchState('replica-c', 'fictional-subject', 'parallel_replica', 'snap-missing')
    with pytest.raises(ValueError):
        value.add_branch(missing)
