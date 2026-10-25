"""Source rebinding must authorize consumed lineage, not just refresh metadata.

A state whose consumed content is no longer authorized (revoked, superseded
without consuming the correction, unadmitted, or from a future causal view)
must never regain publication rights through bind_source() alone. Only
consuming the authoritative change or an explicit verified rebuild can.
"""
from dataclasses import replace
import json

import pytest

from replica_lab.architecture.contracts import fingerprint
from replica_lab.runtime.exact_memory import episode_records
from replica_lab.runtime.memory_representation import FACT_REPRESENTATION, RAW_REPRESENTATION
from replica_lab.runtime.persistent_fast_memory import PersistentFastMemory
from test_exact_memory import Tokenizer, scope
from test_persistent_representation import sources, state_for

REPRESENTATIONS = [RAW_REPRESENTATION, FACT_REPRESENTATION]


def revoke_id(reference, episode, record_id):
    reference.revoke(record_id, authorized_subject='memory_reliability',
                     authorized_branch=episode['episode_id'])


def revoke_first(reference, episode, record):
    revoke_id(reference, episode, record.record_id)


@pytest.mark.parametrize('representation', REPRESENTATIONS)
def test_revocation_blocks_metadata_only_rebind_until_verified_rebuild(sources, tmp_path, representation):
    model, reference, episode, owner = sources
    memory, initial = state_for(sources, representation)
    memory.bind_source(reference, scope=scope(episode))
    revoke_first(reference, episode, initial[0])
    with pytest.raises(ValueError, match='stale'):
        memory.read()
    with pytest.raises(ValueError, match='absent from the authoritative view'):
        memory.bind_source(reference, scope=scope(episode))
    with pytest.raises(ValueError, match='stale'):
        memory.save(tmp_path / 'stale-contents', require_source_binding=True)
    assert not (tmp_path / 'stale-contents').exists()

    memory.rebuild(semantics='current_view', reference=reference, source_scope=scope(episode))
    lineage = [event_id for event_id, _ in memory.lineage]
    assert initial[0].record_id not in lineage and len(lineage) == len(initial) - 1
    memory.bind_source(reference, scope=scope(episode))
    metadata = memory.save(tmp_path / 'rebuilt', require_source_binding=True)
    loaded = PersistentFastMemory.load(
        tmp_path / 'rebuilt', model, Tokenizer(), owner, representation=representation,
        require_source_binding=True, source_reference=reference, source_scope=scope(episode))
    assert loaded.read().state_digest == metadata['state_digest']
    assert [event_id for event_id, _ in loaded.lineage] == lineage


@pytest.mark.parametrize('representation', REPRESENTATIONS)
def test_appending_after_revocation_cannot_dilute_stale_contents(sources, representation):
    model, reference, episode, owner = sources
    memory, initial = state_for(sources, representation)
    memory.bind_source(reference, scope=scope(episode))
    revoke_first(reference, episode, initial[1])
    survivor = next(record for record in initial if record.record_id != initial[1].record_id)
    extra = replace(survivor, record_id=f'{episode["episode_id"]}:extra',
                    text='The note for project Extra is 7.', event_at=99)
    assert reference.admit((extra,)) == 1
    memory.append_record(extra)
    assert memory.source_binding is None
    with pytest.raises(ValueError, match='absent from the authoritative view'):
        memory.bind_source(reference, scope=scope(episode))


@pytest.mark.parametrize('representation', REPRESENTATIONS)
def test_correction_requires_consuming_the_superseding_record_in_order(sources, representation):
    model, reference, episode, owner = sources
    memory, initial = state_for(sources, representation)
    _, correction = episode_records(episode)
    memory.bind_source(reference, scope=scope(episode))
    reference.correct_episode(episode)
    with pytest.raises(ValueError, match='absent from the authoritative view|never consumed'):
        memory.bind_source(reference, scope=scope(episode, 'corrected'))

    memory.append_record(correction)
    memory.bind_source(reference, scope=scope(episode, 'corrected'))
    assert correction.record_id in [event_id for event_id, _ in memory.lineage]

    reversed_state = PersistentFastMemory(model, Tokenizer(), owner, representation=representation)
    reversed_state.append_record(correction)
    reversed_state.append_record(next(record for record in initial
                                      if record.record_id == correction.supersedes))
    with pytest.raises(ValueError, match='absent from the authoritative view'):
        reversed_state.bind_source(reference, scope=scope(episode, 'corrected'))


@pytest.mark.parametrize('representation', REPRESENTATIONS)
def test_unaffected_records_stay_authorized_across_revocation_and_rebind(sources, tmp_path, representation):
    model, reference, episode, owner = sources
    memory, initial = state_for(sources, representation)
    memory.bind_source(reference, scope=scope(episode))
    revoke_first(reference, episode, initial[2])
    memory.rebuild(semantics='current_view', reference=reference, source_scope=scope(episode))
    unaffected = [record for record in initial if record.record_id != initial[2].record_id]
    assert sorted(event_id for event_id, _ in memory.lineage) == sorted(r.record_id for r in unaffected)
    memory.bind_source(reference, scope=scope(episode))
    memory.save(tmp_path / 'unaffected', require_source_binding=True)


@pytest.mark.parametrize('representation', REPRESENTATIONS)
def test_unadmitted_and_future_records_cannot_authorize_into_a_view(sources, representation):
    model, reference, episode, owner = sources
    memory, initial = state_for(sources, representation)
    _, correction = episode_records(episode)
    memory.append_record(correction)
    with pytest.raises(ValueError, match='absent from the authoritative view'):
        memory.bind_source(reference, scope=scope(episode))
    reference.correct_episode(episode)
    with pytest.raises(ValueError, match='absent from the authoritative view'):
        memory.bind_source(reference, scope=scope(episode))
    memory.bind_source(reference, scope=scope(episode, 'corrected'))


def test_unbacked_revoked_correction_text_cannot_rebind(sources):
    model, reference, episode, owner = sources
    _, correction = episode_records(episode)
    memory = PersistentFastMemory(model, Tokenizer(), owner, representation=RAW_REPRESENTATION)
    memory.append('page0', episode['observations'][0], records=())
    memory.bind_source(reference, scope=scope(episode))
    reference.correct_episode(episode)
    memory.append('page-correction', episode['correction_observation'], records=())
    memory.bind_source(reference, scope=scope(episode, 'corrected'))
    revoke_id(reference, episode, correction.supersedes)
    with pytest.raises(ValueError, match='stale'):
        memory.read()
    with pytest.raises(ValueError, match='revoked source text'):
        memory.bind_source(reference, scope=scope(episode, 'corrected'))


def test_offline_inspection_then_late_append_rebinds_only_fresh_content(sources, tmp_path):
    model, reference, episode, owner = sources
    memory, initial = state_for(sources, RAW_REPRESENTATION)
    memory.bind_source(reference, scope=scope(episode))
    memory.save(tmp_path / 'bound', require_source_binding=True)
    inspected = PersistentFastMemory.load(tmp_path / 'bound', model, Tokenizer(), owner)
    assert inspected.source_binding is None
    with pytest.raises(ValueError, match='Source-bound'):
        inspected.save(tmp_path / 'unverified', require_source_binding=True)

    inspected.append('late', 'The note for project Extra is 7.', records=())
    inspected.bind_source(reference, scope=scope(episode))
    inspected.save(tmp_path / 'late', require_source_binding=True)
    reloaded = PersistentFastMemory.load(
        tmp_path / 'late', model, Tokenizer(), owner, require_source_binding=True,
        source_reference=reference, source_scope=scope(episode))

    revoke_first(reference, episode, initial[0])
    with pytest.raises(ValueError, match='stale'):
        reloaded.read()
    with pytest.raises(ValueError, match='absent from the authoritative view'):
        reloaded.bind_source(reference, scope=scope(episode))
    with pytest.raises(ValueError, match='stale'):
        PersistentFastMemory.load(
            tmp_path / 'late', model, Tokenizer(), owner, require_source_binding=True,
            source_reference=reference, source_scope=scope(episode))


def test_failed_rebuild_preserves_the_last_valid_state_and_binding(sources, monkeypatch):
    memory, initial = state_for(sources, FACT_REPRESENTATION)
    model, reference, episode, owner = sources
    memory.bind_source(reference, scope=scope(episode))
    before = memory.read().state_digest
    lineage, events = list(memory.lineage), list(memory.events)
    binding = dict(memory.source_binding)

    from replica_lab.runtime import persistent_fast_memory as module

    def interrupted(self, records=None, *, semantics, **kwargs):
        raise RuntimeError('injected rebuild failure')

    monkeypatch.setattr(module.PersistentFastMemory, 'rebuild', interrupted)
    revoke_first(reference, episode, initial[0])
    with pytest.raises(RuntimeError, match='injected'):
        memory.rebuild(semantics='current_view', reference=reference, source_scope=scope(episode))
    monkeypatch.undo()
    assert memory.lineage == lineage and memory.events == events
    assert memory.source_binding is None or memory.source_binding == binding
    with pytest.raises(ValueError, match='absent from the authoritative view'):
        memory.bind_source(reference, scope=scope(episode))


def page_dependency_records(episode, records, page_index):
    lines = set(episode['observations'][page_index].splitlines())
    return tuple(record for record in records if record.text in lines)


def raw_page_state(model, owner, episode, initial):
    memory = PersistentFastMemory(model, Tokenizer(), owner, representation=RAW_REPRESENTATION)
    for index in (0, 1):
        memory.append(f'page{index}', episode['observations'][index],
                      records=page_dependency_records(episode, initial, index))
    return memory


def downgrade_page_dependency_manifest(path):
    """Simulate a versioned snapshot written before page dependency metadata."""
    manifest_path = path / 'persistent_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.pop('page_dependency_format')
    for event in manifest['events']:
        event.pop('dependencies')
    manifest['manifest_digest'] = fingerprint({key: value for key, value in manifest.items()
                                               if key != 'manifest_digest'})
    manifest_path.write_text(json.dumps(manifest))


def test_raw_append_without_records_rejects_rebind_even_with_unchanged_page_hash(sources, tmp_path):
    model, reference, episode, owner = sources
    initial, _ = episode_records(episode)
    page = episode['observations'][0]
    contained = next(record for record in initial if record.text in page.splitlines())
    memory = PersistentFastMemory(model, Tokenizer(), owner, representation=RAW_REPRESENTATION)
    memory.append('page0', page)
    page_hash = memory.lineage[0][1]
    assert memory.events[0]['dependencies'] is None
    with pytest.raises(ValueError, match='provenance is absent'):
        memory.bind_source(reference, scope=scope(episode))
    with pytest.raises(ValueError, match='provenance is unknown'):
        memory.save(tmp_path / 'unknown-provenance')
    assert not (tmp_path / 'unknown-provenance').exists()

    revoke_first(reference, episode, contained)
    assert memory.lineage[0][1] == page_hash
    with pytest.raises(ValueError, match='provenance is absent'):
        memory.bind_source(reference, scope=scope(episode))
    with pytest.raises(ValueError, match='provenance is unknown'):
        memory.save(tmp_path / 'unknown-provenance-after-revoke')
    assert not (tmp_path / 'unknown-provenance-after-revoke').exists()


def test_raw_page_contained_record_revocation_blocks_rebind_before_and_after_reload(sources, tmp_path):
    model, reference, episode, owner = sources
    _, initial = state_for(sources, RAW_REPRESENTATION)
    page0 = page_dependency_records(episode, initial, 0)
    assert initial[0] in page0 and len(page0) == 3
    memory = raw_page_state(model, owner, episode, initial)
    memory.bind_source(reference, scope=scope(episode))
    metadata = memory.save(tmp_path / 'bound-pages', require_source_binding=True)
    reloaded = PersistentFastMemory.load(
        tmp_path / 'bound-pages', model, Tokenizer(), owner, require_source_binding=True,
        source_reference=reference, source_scope=scope(episode))
    assert reloaded.read().state_digest == metadata['state_digest']
    assert reloaded.bind_source(reference, scope=scope(episode)) == memory.source_binding

    revoke_first(reference, episode, initial[0])
    for bound in (memory, reloaded):
        with pytest.raises(ValueError, match='stale'):
            bound.read()
        with pytest.raises(ValueError, match='page dependency was revoked'):
            bound.bind_source(reference, scope=scope(episode))
    with pytest.raises(ValueError, match='stale'):
        reloaded.save(tmp_path / 'stale-pages', require_source_binding=True)
    assert not (tmp_path / 'stale-pages').exists()
    with pytest.raises(ValueError, match='stale'):
        PersistentFastMemory.load(
            tmp_path / 'bound-pages', model, Tokenizer(), owner, require_source_binding=True,
            source_reference=reference, source_scope=scope(episode))

    memory.rebuild(semantics='current_view', reference=reference, source_scope=scope(episode))
    assert initial[0].record_id not in [event_id for event_id, _ in memory.lineage]
    assert len(memory.lineage) == len(initial) - 1
    memory.bind_source(reference, scope=scope(episode))
    rebuilt = memory.save(tmp_path / 'rebuilt-pages', require_source_binding=True)
    recovered = PersistentFastMemory.load(
        tmp_path / 'rebuilt-pages', model, Tokenizer(), owner, require_source_binding=True,
        source_reference=reference, source_scope=scope(episode))
    assert recovered.read().state_digest == rebuilt['state_digest']
    recovered.bind_source(reference, scope=scope(episode))


def test_raw_page_dependencies_reject_changed_missing_and_absent_provenance(sources):
    model, reference, episode, owner = sources
    _, initial = state_for(sources, RAW_REPRESENTATION)
    page0 = page_dependency_records(episode, initial, 0)

    changed = PersistentFastMemory(model, Tokenizer(), owner, representation=RAW_REPRESENTATION)
    changed.append('page0', episode['observations'][0], records=page0)
    changed.events[0]['dependencies'][0]['record_hash'] = '0' * 64
    with pytest.raises(ValueError, match='no longer matches'):
        changed.bind_source(reference, scope=scope(episode))

    missing = PersistentFastMemory(model, Tokenizer(), owner, representation=RAW_REPRESENTATION)
    absent_record = replace(page0[0], record_id=f'{episode["episode_id"]}:never-admitted')
    missing.append('page0', episode['observations'][0], records=(absent_record,))
    with pytest.raises(ValueError, match='missing from the ledger'):
        missing.bind_source(reference, scope=scope(episode))

    unknown = PersistentFastMemory(model, Tokenizer(), owner, representation=RAW_REPRESENTATION)
    unknown.append('page0', episode['observations'][0], records=page0)
    unknown.events[0]['dependencies'] = None
    with pytest.raises(ValueError, match='provenance is absent'):
        unknown.bind_source(reference, scope=scope(episode))


def test_raw_page_dependencies_must_be_explicit_exact_and_unique(sources):
    model, reference, episode, owner = sources
    _, initial = state_for(sources, RAW_REPRESENTATION)
    page0 = page_dependency_records(episode, initial, 0)
    memory = PersistentFastMemory(model, Tokenizer(), owner, representation=RAW_REPRESENTATION)
    absent = replace(page0[0], record_id=f'{episode["episode_id"]}:absent', text='absent text')
    with pytest.raises(ValueError, match='not present exactly'):
        memory.append('page0', episode['observations'][0], records=(absent,))
    with pytest.raises(TypeError, match='MemoryRecord'):
        memory.append('page0', episode['observations'][0], records=('not a record',))
    with pytest.raises(ValueError, match='unique'):
        memory.append('page0', episode['observations'][0], records=(page0[0], page0[0]))

    memory.append_page('page0', page0)
    assert [dependency['record_id'] for dependency in memory.events[0]['dependencies']] == [
        record.record_id for record in page0]
    memory.bind_source(reference, scope=scope(episode))
    memory.read()


def test_legacy_snapshots_without_page_dependencies_fail_closed_or_authorize_records(sources, tmp_path):
    model, reference, episode, owner = sources
    _, initial = state_for(sources, RAW_REPRESENTATION)

    raw = raw_page_state(model, owner, episode, initial)
    raw.bind_source(reference, scope=scope(episode))
    raw.save(tmp_path / 'raw-pages', require_source_binding=True)
    downgrade_page_dependency_manifest(tmp_path / 'raw-pages')
    inspected = PersistentFastMemory.load(tmp_path / 'raw-pages', model, Tokenizer(), owner)
    assert inspected.read().state_digest == raw.read().state_digest
    with pytest.raises(ValueError, match='provenance is absent'):
        inspected.bind_source(reference, scope=scope(episode))
    with pytest.raises(ValueError, match='provenance is absent'):
        PersistentFastMemory.load(
            tmp_path / 'raw-pages', model, Tokenizer(), owner, require_source_binding=True,
            source_reference=reference, source_scope=scope(episode))

    records, _ = state_for(sources, RAW_REPRESENTATION)
    records.bind_source(reference, scope=scope(episode))
    metadata = records.save(tmp_path / 'record-backed', require_source_binding=True)
    downgrade_page_dependency_manifest(tmp_path / 'record-backed')
    historical = PersistentFastMemory.load(
        tmp_path / 'record-backed', model, Tokenizer(), owner, require_source_binding=True,
        source_reference=reference, source_scope=scope(episode))
    assert historical.read().state_digest == metadata['state_digest']
    historical.bind_source(reference, scope=scope(episode))
    historical.save(tmp_path / 'record-backed-rebound', require_source_binding=True)


def test_raw_page_correction_dependency_requires_consuming_superseder(sources):
    model, reference, episode, owner = sources
    _, initial = state_for(sources, RAW_REPRESENTATION)
    _, correction = episode_records(episode)
    page0 = page_dependency_records(episode, initial, 0)
    memory = PersistentFastMemory(model, Tokenizer(), owner, representation=RAW_REPRESENTATION)
    memory.append('page0', episode['observations'][0], records=page0)
    memory.bind_source(reference, scope=scope(episode))
    reference.correct_episode(episode)
    with pytest.raises(ValueError, match='absent from the authoritative view|never consumed'):
        memory.bind_source(reference, scope=scope(episode, 'corrected'))

    memory.append_page('correction', (correction,))
    memory.bind_source(reference, scope=scope(episode, 'corrected'))
    assert correction.record_id in [dependency['record_id']
                                    for event in memory.events
                                    for dependency in event['dependencies']]
    revoke_first(reference, episode, initial[0])
    with pytest.raises(ValueError, match='page dependency was revoked'):
        memory.bind_source(reference, scope=scope(episode, 'corrected'))
    memory.rebuild(semantics='current_view', reference=reference, source_scope=scope(episode, 'corrected'))
    assert correction.record_id not in [event_id for event_id, _ in memory.lineage]
    memory.bind_source(reference, scope=scope(episode, 'corrected'))


def test_unaffected_raw_page_stays_authorized_when_another_page_dependency_is_revoked(sources):
    model, reference, episode, owner = sources
    _, initial = state_for(sources, RAW_REPRESENTATION)
    page1 = page_dependency_records(episode, initial, 1)
    assert len(page1) == 3
    memory = PersistentFastMemory(model, Tokenizer(), owner, representation=RAW_REPRESENTATION)
    memory.append('page1', episode['observations'][1], records=page1)
    memory.bind_source(reference, scope=scope(episode))
    revoke_first(reference, episode, initial[0])  # Contained in page0, not in this state.
    assert memory.bind_source(reference, scope=scope(episode))['view_digest'] == reference.view(
        scope(episode)).digest
    memory.read()
    revoke_first(reference, episode, page1[0])
    with pytest.raises(ValueError, match='page dependency was revoked'):
        memory.bind_source(reference, scope=scope(episode))


