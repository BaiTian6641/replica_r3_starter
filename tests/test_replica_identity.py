"""Fictional C16 contracts: source/branch isolation, null path and readability."""

import hashlib
import torch

from replica_lab.replica.identity import (
    BranchExperience, IdentityArchive, IdentityContext, IdentityManifest,
    PrincipleHypothesis, ReplicaBranchState, SourceEpisode, render_identity_context,
)
from replica_lab.replica.reader import IdentityReader


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def archive():
    value = IdentityArchive('fictional-subject')
    for index, (text, observed) in enumerate([
            ('I preferred recoverable plans when a deadline allowed it.', 1),
            ('I revised that preference after an outage in 2024.', 5),
            ('A later statement should be excluded by the cutoff.', 20)]):
        value.add_episode(SourceEpisode(
            f'ep-{index}', 'fictional-subject', f'source-{index}', digest(text),
            observed, observed, text, kind='preference',
            permitted_uses=('retrieval', 'evaluation'), authenticity='source_report'))
    value.add_principle(PrincipleHypothesis(
        'pr-1', 'fictional-subject', 'Prefer recoverability when the deadline permits.',
        condition='deadline permits', exceptions=('when the outage risk is acute',),
        supporting_episode_ids=('ep-0',), contradicting_episode_ids=('ep-1',),
        status='approved'))
    return value


def snapshot(value):
    return value.build_snapshot('snap-1', version=1, created_at=10,
                                episode_ids=('ep-0', 'ep-1', 'ep-2'),
                                principle_ids=('pr-1',),
                                unresolved_questions=('Whether the preference still holds.',),
                                permitted_uses=('retrieval', 'evaluation'))


def test_source_snapshot_cutoff_and_null_path():
    value = archive(); snap = snapshot(value)
    context = IdentityArchive.resolve(
        value, 'snap-1', mode='source_snapshot', branch_id='source', purpose='test',
        cutoff=10)
    assert context.mode == 'source_snapshot' and context.branch_id == 'source'
    assert set(context.evidence_ids) == {'ep-0', 'ep-1'}
    assert 'Prefer recoverability' in context.summary
    assert 'later statement' not in context.summary
    null = IdentityContext.null(subject_id='fictional-subject', branch_id='main',
                                purpose='test', cutoff=10)
    assert null.is_null and render_identity_context(null) == ''


def test_parallel_branch_is_separate_and_cannot_rewrite_source():
    value = archive(); snap = snapshot(value)
    branch = ReplicaBranchState('replica-a', 'fictional-subject', 'parallel_replica',
                                snap.snapshot_id)
    experience = BranchExperience('exp-1', 'replica-a', 'fictional-subject', 4,
                                  'The branch observed a failed migration.',
                                  outcome='failed', evidence_ids=('tool-1',))
    branch = branch.add(experience)
    context = value.resolve('snap-1', mode='parallel_replica', branch_id='replica-a',
                            purpose='test', cutoff=10, branch_state=branch)
    assert context.mode == 'parallel_replica' and context.branch_id == 'replica-a'
    assert context.evidence_ids == ('ep-0', 'ep-1')
    other = ReplicaBranchState('replica-a', 'fictional-subject', 'parallel_replica',
                               snap.snapshot_id)
    try:
        value.resolve('snap-1', mode='parallel_replica', branch_id='replica-b',
                      purpose='test', cutoff=10, branch_state=other)
    except PermissionError:
        pass
    else:
        raise AssertionError('mismatched branch state must be rejected')


def test_revocation_invalidates_snapshot_and_manifest_compatibility():
    value = archive(); snap = snapshot(value)
    context = value.resolve('snap-1', mode='source_snapshot', branch_id='source',
                            purpose='test', cutoff=10)
    manifest = context.manifest()
    assert manifest.compatible(IdentityManifest(
        'fictional-subject', snap.snapshot_id, snap.digest, manifest.schema_version,
        dict(manifest.producer_versions)))
    value.revoke_episode('ep-0')
    try:
        value.resolve('snap-1', mode='source_snapshot', branch_id='source',
                      purpose='test', cutoff=10)
    except PermissionError:
        pass
    else:
        raise AssertionError('revoked snapshot must not resolve')
    assert manifest.compatible(manifest)


def test_identity_reader_is_null_at_init_and_independent():
    torch.manual_seed(5)
    reader = IdentityReader(width=8, heads=2)
    hidden = torch.randn(1, 3, 8)
    slots = torch.randn(1, 2, 8)
    valid = torch.ones(1, 2, dtype=torch.bool)
    assert torch.equal(reader(hidden, slots, valid), torch.zeros_like(hidden))
    with torch.no_grad():
        reader.output_projection.weight.normal_(0.0, 0.1)
    changed = reader(hidden, slots, valid)
    assert not torch.equal(changed, torch.zeros_like(hidden))
    assert torch.equal(reader(hidden, slots, valid, enabled=False),
                       torch.zeros_like(hidden))
    assert torch.equal(reader(hidden, slots, torch.zeros(1, 2, dtype=torch.bool)),
                       torch.zeros_like(hidden))


def test_runtime_prompt_contains_identity_only_when_enabled(tmp_path):
    from tests.e3_runtime_helpers import make_case, make_runtime
    tools = make_case(tmp_path)
    value = archive(); snapshot(value)
    context = value.resolve('snap-1', mode='source_snapshot', branch_id='source',
                            purpose='test', cutoff=10)
    runtime = make_runtime(tools, tmp_path, identity=context)
    prompt = runtime._build_prompt(runtime.state)
    assert 'IDENTITY CONTEXT' in prompt and 'Prefer recoverability' in prompt
    assert 'SUCCESS CONDITION' in prompt
    null_runtime = make_runtime(tools, tmp_path / 'null', identity=None)
    assert 'IDENTITY CONTEXT' not in null_runtime._build_prompt(null_runtime.state)


def test_correction_is_temporal_and_does_not_leak_backwards():
    value = IdentityArchive('fictional-subject')
    old = SourceEpisode('old', 'fictional-subject', 'src', digest('old'), 1, 1,
                        'The old statement.', kind='preference',
                        permitted_uses=('retrieval',), authenticity='source_report')
    new = SourceEpisode('new', 'fictional-subject', 'src', digest('new'), 5, 5,
                        'The corrected statement.', kind='preference',
                        permitted_uses=('retrieval',), authenticity='source_report',
                        supersedes='old')
    for episode in (old, new):
        value.add_episode(episode)
    value.add_principle(PrincipleHypothesis(
        'pr', 'fictional-subject', 'A conditional preference.', status='approved',
        supporting_episode_ids=('old', 'new')))
    value.build_snapshot('snap', version=1, created_at=10,
                         episode_ids=('old', 'new'), principle_ids=('pr',),
                         permitted_uses=('retrieval',))
    early = value.resolve('snap', mode='source_snapshot', branch_id='source',
                          purpose='test', cutoff=3)
    late = value.resolve('snap', mode='source_snapshot', branch_id='source',
                         purpose='test', cutoff=10)
    assert early.evidence_ids == ('old',)
    assert late.evidence_ids == ('new',)
