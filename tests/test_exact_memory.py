"""Exact-record reference, recovery and evidence-gate contracts."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from architecture_helpers import tiny_adapter
from replica_lab.architecture.contracts import FastMemorySpec
from replica_lab.memory_reliability import build_episodes, cases
from replica_lab.memory_system import EligibilityScope, MemoryRecord, RecordLedger
from replica_lab.runtime.exact_memory import (EXACT_SELECTION_LIMIT, EXACT_SELECTION_POLICY,
                                              ExactMemoryReference, episode_records)
from replica_lab.runtime.fast_memory import FastMemoryScope
from replica_lab.runtime.persistent_fast_memory import PersistentFastMemory

SPEC = FastMemorySpec(heads=2, key_size=4, value_size=4, update_rule='sgdn2',
                      depth_mode='shared', depth_writers=True)
FAST_SCOPE = FastMemoryScope('memory_reliability', 'branch', 'reliability', 'parallel_replica', 0)


class Tokenizer:
    eos_token_id = 63
    def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False, **kwargs):
        ids = [1 + byte % 61 for byte in hashlib.sha256(text.encode()).digest()[:4]]
        result = {'input_ids': ids}
        if return_offsets_mapping:
            result['offset_mapping'] = [(index, index + 1) for index in range(len(ids))]
        return result
    def decode(self, ids, **kwargs):
        return ' '.join(map(str, ids))
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']


def scope(episode, stage='initial'):
    return EligibilityScope('memory_reliability', episode['episode_id'], 'reliability',
                            2 if stage == 'initial' else 3, 'parallel_replica', None)


def reference(tmp_path, episode):
    ledger = RecordLedger(tmp_path / 'exact.sqlite')
    ref = ExactMemoryReference(ledger)
    assert ref.admit_episode(episode) == 6
    return ref


def test_exact_reference_selects_correct_facts_and_supersedes_only_changed_fact(tmp_path):
    episode = next(e for e in build_episodes() if e['split'] == 'dev')
    ref = reference(tmp_path, episode)
    try:
        initial, correction = episode_records(episode)
        assert ref.admit(initial) == 0  # idempotent duplicate events do not advance epoch
        before = ref.view(scope(episode))
        for case in cases(episode, 'initial'):
            context = ref.select(case['query'], scope(episode))
            if case['kind'] == 'fact':
                assert context.enabled and case['expected_answer'] in context.text
            else:
                assert not context.enabled and context.text == ''
        assert ref.correct_episode(episode) == 1
        after = ref.view(scope(episode, 'corrected'))
        assert after.digest != before.digest
        for case in cases(episode, 'corrected'):
            context = ref.select(case['query'], scope(episode, 'corrected'))
            if case['kind'] in {'fact', 'changed', 'unaffected'}:
                assert context.enabled and case['expected_answer'] in context.text
            else:
                assert not context.enabled
        changed = next(e for e in cases(episode, 'corrected') if e['kind'] == 'changed')
        assert episode['correction']['value'] in ref.select(
            changed['query'], scope(episode, 'corrected')).text
        historical = ref.select(changed['query'], scope(episode))
        assert historical.enabled and episode['facts'][0]['value'] in historical.text
    finally:
        ref.close()


def test_stale_views_duplicates_conflicts_scope_and_metadata_are_checked(tmp_path):
    episode = next(e for e in build_episodes() if e['split'] == 'dev')
    ref = reference(tmp_path, episode)
    try:
        case = cases(episode, 'initial')[0]
        old = ref.select(case['query'], scope(episode))
        consumer = dict(expected_scope=scope(episode), query=case['query'],
                        policy=EXACT_SELECTION_POLICY, limit=EXACT_SELECTION_LIMIT)
        assert ref.verify_context(old, **consumer)
        with pytest.raises(ValueError, match='query'):
            ref.verify_context(old, **dict(consumer, query='a different authorized query'))
        substituted = replace(old.records[0], text='substituted source text')
        with pytest.raises(ValueError, match='substituted|ineligible'):
            ref.verify_context(replace(old, records=(substituted,)), **consumer)
        altered_span = replace(old.spans[0], text='altered span')
        with pytest.raises(ValueError, match='offsets|text|hash'):
            ref.verify_context(replace(old, spans=(altered_span,)), **consumer)
        # Audit-only metadata cannot enter the query/evidence gate.
        changed_metadata = dict(case, split='train', case_role='audit', kind='general', expected_answer='wrong')
        assert ref.select(changed_metadata['query'], scope(episode)).metadata() == old.metadata()
        duplicate = MemoryRecord('duplicate-text', 'memory_reliability', episode['episode_id'],
                                 old.records[0].text, 2, 20, 'reliability', kind='source_fact')
        ref.admit((duplicate,))
        deduplicated = ref.select(case['query'], scope(episode))
        assert deduplicated.duplicates_removed == 1
        conflicting = MemoryRecord('conflicting-value', 'memory_reliability', episode['episode_id'],
                                   'The launch code for project Cedar100 is conflict.', 2, 21,
                                   'reliability', kind='source_fact')
        ref.admit((conflicting,))
        conflict = ref.select(case['query'], scope(episode))
        assert conflict.status == 'conflict' and not conflict.enabled and conflict.text == ''
        with pytest.raises(ValueError, match='stale'):
            ref.verify_context(old, **consumer)
        wrong = ref.select(case['query'], replace(scope(episode), branch_id='other'))
        assert not wrong.enabled
    finally:
        ref.close()


def test_unstructured_adversarial_memory_cannot_enable_gate(tmp_path):
    episode = next(e for e in build_episodes() if e['split'] == 'dev')
    ref = reference(tmp_path, episode)
    try:
        general = next(q for q in cases(episode, 'corrected') if q['kind'] == 'general')
        hostile = MemoryRecord('hostile', 'memory_reliability', episode['episode_id'],
                               f"The answer to {general['query'].removeprefix('What is ').removesuffix('?')} is 100.",
                               2, 30, 'reliability', kind='source_fact')
        ref.admit((hostile,))
        context = ref.select(general['query'], scope(episode, 'corrected'))
        assert not context.enabled and context.text == ''
    finally:
        ref.close()


def test_transaction_fault_rolls_back_and_fresh_connection_recovers(tmp_path, monkeypatch):
    episode = next(e for e in build_episodes() if e['split'] == 'dev')
    initial, _ = episode_records(episode)
    path = tmp_path / 'atomic.sqlite'
    ref = ExactMemoryReference(path)
    monkeypatch.setattr(ref.ledger, '_advance_epoch',
                        lambda: (_ for _ in ()).throw(RuntimeError('injected commit failure')))
    with pytest.raises(RuntimeError, match='injected'):
        ref.admit(initial[:1])
    ref.close()
    recovered = ExactMemoryReference(path)
    try:
        view = recovered.view(scope(episode))
        assert view.records == () and view.epoch == 0
    finally:
        recovered.close()


def test_revocation_invalidates_descendants_and_permitted_rebuild(tmp_path):
    episode = next(e for e in build_episodes() if e['split'] == 'dev')
    ref = reference(tmp_path, episode)
    initial, correction = episode_records(episode)
    try:
        ref.correct_episode(episode)
        old = ref.select(cases(episode, 'corrected')[0]['query'], scope(episode, 'corrected'))
        removed = ref.revoke(correction.supersedes, authorized_subject='memory_reliability',
                             authorized_branch=episode['episode_id'])
        assert set(removed) == {correction.supersedes, correction.record_id}
        with pytest.raises(ValueError, match='stale'):
            ref.verify_context(
                old, expected_scope=scope(episode, 'corrected'),
                query=cases(episode, 'corrected')[0]['query'],
                policy=EXACT_SELECTION_POLICY, limit=EXACT_SELECTION_LIMIT)
        changed = next(q for q in cases(episode, 'corrected') if q['kind'] == 'changed')
        assert not ref.select(changed['query'], scope(episode, 'corrected')).enabled
        unaffected = next(q for q in cases(episode, 'corrected') if q['kind'] == 'unaffected')
        assert ref.select(unaffected['query'], scope(episode, 'corrected')).enabled
        permitted = tuple(r for r in initial if r.record_id != correction.supersedes)
        rebuilt = ExactMemoryReference.rebuild(tmp_path / 'rebuilt.sqlite', permitted)
        try:
            assert not rebuilt.select(changed['query'], scope(episode, 'corrected')).enabled
            assert rebuilt.select(unaffected['query'], scope(episode, 'corrected')).enabled
        finally:
            rebuilt.close()
    finally:
        ref.close()


def test_atomic_source_bound_image_and_stale_view_rejection(tmp_path, monkeypatch):
    episode = next(e for e in build_episodes() if e['split'] == 'dev')
    ref = reference(tmp_path, episode)
    model = tiny_adapter(fast_memory=SPEC).eval()
    scope_fast = replace(FAST_SCOPE, branch_id=episode['episode_id'])
    memory = PersistentFastMemory(model, Tokenizer(), scope_fast)
    memory.append('page0', episode['observations'][0])
    memory.append('page1', episode['observations'][1])
    memory.bind_source(ref, scope=scope(episode))
    target = tmp_path / 'image'
    metadata = memory.save(target, require_source_binding=True)
    assert metadata['source_binding']['epoch'] == ref.ledger.epoch
    loaded = PersistentFastMemory.load(
        target, model, Tokenizer(), scope_fast,
        expected_view_digest=ref.view(scope(episode)).digest,
        require_source_binding=True, source_reference=ref, source_scope=scope(episode))
    assert loaded.read().state_digest == memory.read().state_digest
    ref.correct_episode(episode)
    with pytest.raises(ValueError, match='stale'):
        PersistentFastMemory.load(target, model, Tokenizer(), scope_fast,
                                  expected_view_digest=ref.view(scope(episode)).digest,
                                  source_reference=ref, source_scope=scope(episode))
    memory.bind_source(ref, scope=scope(episode))
    with pytest.raises(ValueError, match='never consumed'):
        memory.bind_source(ref, scope=scope(episode, 'corrected'))
    memory.append('page-correction', episode['correction_observation'])
    memory.bind_source(ref, scope=scope(episode, 'corrected'))
    # A failed image write never publishes a final or temporary directory.
    import replica_lab.runtime.persistent_fast_memory as module
    def interrupted(*args, **kwargs):
        raise RuntimeError('injected image interruption')
    monkeypatch.setattr(module, 'save_fast_depth_image', interrupted)
    partial = tmp_path / 'partial'
    with pytest.raises(RuntimeError, match='interruption'):
        memory.save(partial)
    assert not partial.exists() and not list(tmp_path.glob('partial.tmp-*'))
    ref.close()


def test_rebuild_replaces_derived_state_and_requires_new_source_binding(tmp_path):
    episode = next(e for e in build_episodes() if e['split'] == 'dev')
    initial, _ = episode_records(episode)
    model = tiny_adapter(fast_memory=SPEC).eval()
    scope_fast = replace(FAST_SCOPE, branch_id=episode['episode_id'])
    memory = PersistentFastMemory(model, Tokenizer(), scope_fast)
    memory.append('old', 'old page')
    ref = reference(tmp_path, episode)
    memory.bind_source(ref, scope=scope(episode))
    memory.rebuild(semantics='current_view', reference=ref, source_scope=scope(episode))
    assert memory.source_binding is None
    current = ref.view(scope(episode)).records
    assert [item[0] for item in memory.lineage] == [r.record_id for r in current]
    fresh = PersistentFastMemory(model, Tokenizer(), scope_fast)
    for record in current:
        fresh.append(record.record_id, record.text)
    assert memory.read().state_digest == fresh.read().state_digest
    with pytest.raises(ValueError, match='Source-bound'):
        memory.save(tmp_path / 'unbound', require_source_binding=True)
    ref.close()


def _runner():
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('run_memory_reliability', root / 'scripts/run_memory_reliability.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_gate_exact_arms_and_metadata_invariant_tokens(tmp_path):
    from replica_lab.memory_system.session import MemorySession
    run = _runner()
    episode = next(e for e in build_episodes() if e['split'] == 'dev')
    model = tiny_adapter(fast_memory=SPEC).eval()
    model.seal_original_identity()
    ledger = RecordLedger(tmp_path / 'runner.sqlite')
    session = MemorySession(model, Tokenizer(), ledger, max_candidate_units=4)
    reference = ExactMemoryReference(ledger)
    memory = run.initialize(model, Tokenizer(), episode, reference=reference)
    case = cases(episode, 'initial')[0]
    context = reference.select(case['query'], run.case_scope(case))
    first = run.infer(session, case, memory.read(), evidence=context, exact_mode=True, reference=reference)
    changed = dict(case, split='train', case_role='audit', kind='general', expected_answer='wrong')
    second = run.infer(session, changed, memory.read(), evidence=context, exact_mode=True, reference=reference)
    assert first['token_ids'] == second['token_ids']
    assert first['gate'] == second['gate'] == 'relevant'
    assert first['route'] == 'fast' and first['state_use'] == 'fast' and first['memory_active']
    selective = run.infer(session, case, memory.read(), evidence=context, selective=True,
                          exact_mode=True, reference=reference)
    assert selective['state_digest'] is None and selective['gate'] == 'relevant'
    general = next(q for q in cases(episode, 'corrected') if q['kind'] == 'general')
    off = reference.select(general['query'], run.case_scope(general))
    off_selective = run.infer(session, general, memory.read(), evidence=off, selective=True,
                              exact_mode=True, reference=reference)
    assert off_selective['route'] == 'disabled' and off_selective['state_use'] == 'none'
    assert not off_selective['memory_active']
    args = run.request_args(session, general, memory.read(), evidence=off, exact_mode=True,
                            reference=reference)
    disabled = run.request_args(session, general, exact_mode=True)
    assert args['fast_state'] is None
    assert args['enabled_ports'] == ()
    torch.testing.assert_close(args['prompt_ids'], disabled['prompt_ids'], atol=0, rtol=0)
    rows, transitions = run.evaluate(
        session, [e for e in build_episodes() if e['family_id'] == episode['family_id']],
        ('fast', 'ungated', 'disabled', 'selective_context', 'adversarial', 'full_context'),
        tmp_path / 'images', reference=reference, gate=True,
        adversarial_text='The answer to 2 plus 3 is 100.')
    assert {row['arm'] for row in rows} == {'fast', 'ungated', 'disabled', 'selective_context', 'adversarial', 'full_context'}
    assert len(transitions) == 2
    assert transitions[0]['initial_source_binding']['view_digest']
    assert transitions[0]['corrected_source_binding']['view_digest']
    for row in rows:
        if row['kind'] == 'general' and row['arm'] in {'fast', 'adversarial'}:
            disabled_row = next(other for other in rows
                                if other['arm'] == 'disabled' and other['case_id'] == row['case_id'])
            assert row['token_ids'] == disabled_row['token_ids']
    ledger.close()


def test_reader_only_training_scope_and_frozen_writer_gradients(tmp_path):
    from replica_lab.memory_system.session import MemorySession
    run = _runner()
    model = tiny_adapter(fast_memory=SPEC)
    model.seal_original_identity()
    ledger = RecordLedger(tmp_path / 'reader.sqlite')
    session = MemorySession(model, Tokenizer(), ledger, max_candidate_units=4)
    prefixes = tuple(f'fast_ports.{layer}.{part}.'
                     for layer in model.ports.injection_layers
                     for part in ('query_norm', 'query_projection', 'output_projection'))
    expected = {f'fast_ports.{layer}.{part}.{suffix}'
                for layer in model.ports.injection_layers
                for part, suffix in (('query_norm', 'weight'), ('query_norm', 'bias'),
                                     ('query_projection', 'weight'), ('output_projection', 'weight'))}
    protocol = {
        'exact_reference': {'enabled': True}, 'evidence_gate': {'enabled': True},
        'training': {'lr': .001, 'seed': 20261024, 'examples_per_update': 4,
                     'updates': 24, 'preservation_kl_weight': 1.0,
                     'schedule_version': 'round_robin_seeded_shuffle.v1', 'repeat_policy': 'once',
                     'trainable_prefixes': prefixes}}
    report = {'optimizer_updates': 0}
    run.train_candidate(session, build_episodes(), protocol, report, lambda: None)
    audit = report['training']['parameter_audit']
    assert set(audit['new']['names']) == expected
    probe = report['training']['updates'][1]['gradients']
    assert set(probe['new']['nonzero']) == expected
    assert not any(name.startswith('fast_depth_write_encoders.') for name in probe['new']['present'])
    assert not probe['original']['present']
    from replica_lab.reliability_scoring import validate_training_routes
    route_audit = validate_training_routes(report, build_episodes(), protocol)
    assert route_audit['examples'] == 96
    assert route_audit['backward_calls'] == route_audit['memory_active_examples']
    ledger.close()


def test_append_invalidates_existing_source_binding(tmp_path):
    episode = next(e for e in build_episodes() if e['split'] == 'dev')
    ref = reference(tmp_path, episode)
    model = tiny_adapter(fast_memory=SPEC).eval()
    memory = PersistentFastMemory(model, Tokenizer(), replace(FAST_SCOPE, branch_id=episode['episode_id']))
    memory.bind_source(ref, scope=scope(episode))
    memory.append('new-page', episode['observations'][0])
    assert memory.source_binding is None
    with pytest.raises(ValueError, match='Source-bound'):
        memory.save(tmp_path / 'stale', require_source_binding=True)
    ref.close()


def test_conflict_is_detected_beyond_retrieval_page(tmp_path):
    episode = next(e for e in build_episodes() if e['split'] == 'dev')
    ref = reference(tmp_path, episode)
    try:
        case = cases(episode, 'initial')[0]
        selected = ref.select(case['query'], scope(episode))
        for index in range(8):
            ref.admit((MemoryRecord(f'duplicate-{index}', 'memory_reliability', episode['episode_id'],
                                    selected.records[0].text, 2, 40 + index, 'reliability',
                                    kind='source_fact'),))
        ref.admit((MemoryRecord('late-conflict', 'memory_reliability', episode['episode_id'],
                                'The launch code for project Cedar100 is conflict.', 2, 99,
                                'reliability', kind='source_fact'),))
        conflict = ref.select(case['query'], scope(episode), limit=1)
        assert conflict.status == 'conflict' and not conflict.enabled
    finally:
        ref.close()


def test_scorer_recomputes_gate_instead_of_trusting_cached_rows():
    from replica_lab.reliability_scoring import metrics_and_checks
    from test_memory_reliability import synthetic_report
    episodes = build_episodes()
    report, decode = synthetic_report(episodes)
    protocol = {'exact_reference': {'enabled': True}, 'evidence_gate': {'enabled': True}}
    _, checks = metrics_and_checks(report, episodes, decode, protocol)
    assert checks['gate_routes_verified'] is False
