"""Recovery gates: dataset truth, score tampering and incremental durable state."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from architecture_helpers import tiny_adapter
from replica_lab.answer_audit import audit_rows
from replica_lab.architecture.contracts import FastMemorySpec
from replica_lab.memory_reliability import build_episodes, audit_dataset, cases
from replica_lab.reliability_scoring import metrics_and_checks
from replica_lab.runtime.fast_memory import FastMemoryScope
from replica_lab.runtime.persistent_fast_memory import PersistentFastMemory

ROOT = Path(__file__).resolve().parents[1]
SPEC = FastMemorySpec(heads=2, key_size=4, value_size=4, update_rule='sgdn2', depth_mode='shared', depth_writers=True)
SCOPE = FastMemoryScope('fixture', 'branch', 'test', 'parallel_replica', 0)


def script(name):
    spec = importlib.util.spec_from_file_location(name.removesuffix('.py'), ROOT/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tokenizer:
    eos_token_id = 63
    def __call__(self, text, *, add_special_tokens):
        return {'input_ids': [1 + byte % 61 for byte in hashlib.sha256(text.encode()).digest()[:4]]}
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']
    def decode(self, ids, **kwargs):
        return ' '.join(map(str, ids))


def test_dataset_is_counterfactual_multiquery_and_not_last_value_solvable():
    episodes = build_episodes()
    audit = audit_dataset(episodes)
    assert audit['split_episodes'] == {'train': 8, 'dev': 2, 'heldout': 2}
    assert audit['counterfactual_pairs'] == 24
    assert audit['last_line_accuracy'] <= .25 and audit['first_line_accuracy'] <= .25
    assert audit['query_only_best_accuracy'] == .5
    assert audit['heldout_values_absent_from_training']
    for episode in episodes:
        initial, corrected = cases(episode, 'initial'), cases(episode, 'corrected')
        assert len(initial) == 5 and len(corrected) == 7
        changed = [q for q in corrected if q['kind'] == 'changed']
        assert len(changed) == 1
        for before, after in zip(initial, corrected):
            assert (before['expected_answer'] != after['expected_answer']) == (after['kind'] == 'changed')


@pytest.mark.parametrize('mutation', ['observation', 'correction', 'duplicate', 'query', 'family_split'])
def test_dataset_rejects_oracle_and_inventory_corruption(mutation):
    episodes = build_episodes()
    if mutation == 'observation':
        episodes[0]['observations'][0] += '\nThe launch code for project Intruder is amber.'
    elif mutation == 'correction':
        episodes[0]['correction_observation'] += ' It is obsolete.'
    elif mutation == 'duplicate':
        episodes.append(deepcopy(episodes[0]))
    elif mutation == 'query':
        episodes[0]['questions'][0]['query'] = 'Return the last value.'
    else:
        episodes[-1]['family_id'] = episodes[0]['family_id']
    with pytest.raises(ValueError):
        audit_dataset(episodes)


@pytest.mark.parametrize('mutation', ['gold', 'answer', 'correct', 'length', 'token', 'missing', 'duplicate', 'metadata'])
def test_answer_audit_rejects_report_tampering(mutation):
    oracle = {'q': {'expected_answer': 'blue', 'split': 'dev'}}
    rows = [dict(case_id='q', expected='blue', answer='blue', correct=True, token_ids=[4], generated_tokens=1, split='dev')]
    decode = lambda ids: {4: 'blue', 5: 'red'}[ids[0]]
    assert audit_rows(rows, oracle, decode)[0]['correct']
    if mutation in {'gold', 'answer', 'correct', 'length', 'token', 'metadata'}:
        key, value = {'gold': ('expected', 'red'), 'answer': ('answer', 'red'), 'correct': ('correct', False),
                      'length': ('generated_tokens', 2), 'token': ('token_ids', [5]), 'metadata': ('split', 'train')}[mutation]
        rows[0][key] = value
    elif mutation == 'missing':
        rows.clear()
    else:
        rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError):
        audit_rows(rows, oracle, decode)


def synthetic_report(episodes):
    oracle = [q for e in episodes if e['split'] != 'train' for stage in ('initial', 'corrected') for q in cases(e, stage)]
    values = sorted({q['expected_answer'] for q in oracle} | {'wrong'})
    report = {'transitions': [dict(episode_id=e['episode_id'], lineage_prefix_preserved=True, initial_cursor=2,
                                  corrected_cursor=3, reloaded_before_append=True, source_reencoded_on_reload=False,
                                  initial_digest='before', corrected_digest='after') for e in episodes if e['split'] != 'train']}
    for phase, arms in [('parent', ('fast', 'disabled')), ('candidate', ('fast', 'disabled', 'shuffled', 'last_chunk', 'full_context'))]:
        report[phase] = []
        for arm in arms:
            for q in oracle:
                answer = q['expected_answer'] if arm in {'fast', 'full_context'} or q['kind']=='general' else 'wrong'
                report[phase].append({k: q[k] for k in ('case_id', 'split', 'episode_id', 'stage', 'kind')} | {
                    'arm': arm, 'answer': answer, 'expected': q['expected_answer'], 'correct': answer == q['expected_answer'],
                    'token_ids': [values.index(answer)], 'generated_tokens': 1})
    return report, lambda ids: values[ids[0]]


def test_scorer_recomputes_scores_and_requires_complete_arms():
    episodes = build_episodes()
    report, decode = synthetic_report(episodes)
    metrics, checks = metrics_and_checks(report, episodes, decode)
    assert all(checks.values())
    report['metrics'] = metrics
    report['metrics']['candidate']['fast']['dev']['fact'] = 0.0
    with pytest.raises(ValueError, match='Cached metrics'):
        metrics_and_checks(report, episodes, decode)
    report.pop('metrics')
    report['candidate'] = [r for r in report['candidate'] if r['arm'] != 'disabled']
    with pytest.raises(ValueError, match='arm'):
        metrics_and_checks(report, episodes, decode)


def test_persistent_reload_then_append_preserves_prior_snapshot_and_lineage(tmp_path, monkeypatch):
    model = tiny_adapter(fast_memory=SPEC).eval()
    memory = PersistentFastMemory(model, Tokenizer(), SCOPE)
    memory.append('a', 'first page'); memory.append('b', 'second page')
    before = memory.read(); snapshot = before.weights.clone()
    memory.save(tmp_path/'before')
    encode = model.encode_fast_observation
    monkeypatch.setattr(model, 'encode_fast_observation', lambda *a, **k: pytest.fail('Reload must not encode old sources'))
    loaded = PersistentFastMemory.load(tmp_path/'before', model, Tokenizer(), SCOPE)
    assert loaded.read().state_digest == before.state_digest and len(loaded.lineage) == 2
    monkeypatch.setattr(model, 'encode_fast_observation', encode)
    memory.append('c', 'correction page'); loaded.append('c', 'correction page')
    torch.testing.assert_close(loaded.read().weights, memory.read().weights, atol=0, rtol=0)
    before.assert_immutable()
    torch.testing.assert_close(before.weights, snapshot, atol=0, rtol=0)
    assert len(loaded.lineage) == loaded.state.chunk_index == 3
    assert loaded.lineage[:2] == memory.lineage[:2]
    with pytest.raises(ValueError, match='fresh'):
        loaded.append('c', 'correction again')
    with pytest.raises(ValueError, match='metadata'):
        PersistentFastMemory.load(tmp_path/'before', model, Tokenizer(), replace(SCOPE, branch_id='wrong'))
    with torch.no_grad():
        next(model.fast_ports.parameters()).add_(1)
    with pytest.raises(ValueError, match='producer'):
        loaded.read()
    with pytest.raises(ValueError, match='metadata'):
        PersistentFastMemory.load(tmp_path/'before', model, Tokenizer(), SCOPE)


def test_persistent_training_graph_reaches_both_writers_and_rejects_persistence(tmp_path):
    model = tiny_adapter(fast_memory=SPEC)
    memory = PersistentFastMemory(model, Tokenizer(), SCOPE, training=True)
    memory.append('a', 'first page'); memory.append('b', 'second page')
    memory.read().weights.square().sum().backward()
    for writer in model.fast_depth_write_encoders.values():
        assert any(p.grad is not None and bool(p.grad.ne(0).any()) for p in writer.parameters())
    with pytest.raises(ValueError, match='Training graphs'):
        memory.save(tmp_path/'forbidden')


def test_persistent_rejects_truncation_and_chunk_exhaustion():
    class LongTokenizer:
        def __call__(self, *a, **kw):
            return {'input_ids': [1]*257}
    model = tiny_adapter(fast_memory=SPEC)
    memory = PersistentFastMemory(model, LongTokenizer(), SCOPE)
    with pytest.raises(ValueError, match='truncation'):
        memory.append('a', 'long')
    memory = PersistentFastMemory(model, Tokenizer(), SCOPE,
                                  config=replace(memory.config, max_chunks=1))
    memory.append('a', 'one')
    with pytest.raises(ValueError, match='quota'):
        memory.append('b', 'two')


def test_cpu_training_generation_and_checkpoint_producer_smoke(tmp_path):
    from replica_lab.memory_system import RecordLedger
    from replica_lab.memory_system.session import MemorySession
    run = script('run_memory_reliability.py')
    scorer = script('score_memory_reliability.py')
    model = tiny_adapter(fast_memory=SPEC).eval()
    model.seal_original_identity()
    tokenizer = Tokenizer()
    ledger = RecordLedger(tmp_path/'ledger.sqlite')
    session = MemorySession(model, tokenizer, ledger, max_candidate_units=4)
    episodes = build_episodes()
    try:
        protocol = {'training': {'lr': .001, 'seed': 20261024, 'examples_per_update': 4,
                                 'updates': 24, 'preservation_kl_weight': 1.0,
                                 'schedule_version': 'round_robin_seeded_shuffle.v1', 'repeat_policy': 'once'}}
        report = {'optimizer_updates': 0}
        run.train_candidate(session, episodes, protocol, report, lambda: None)
        assert report['optimizer_updates'] == 24
        assert len(set(report['training']['case_ids'])) == 96
        assert not report['training']['updates'][1]['gradients']['original']['present']
        receipt = model.save_extension(tmp_path/'extension')
        assert scorer.checkpoint_producer(tmp_path/'extension', receipt['weights_sha256']) == model.fast_version_key()
        episode = episodes[-1]
        memory = run.initialize(model, tokenizer, episode)
        case = cases(episode, 'initial')[0]
        args = run.request_args(session, case, memory.read())
        changed = run.request_args(session, dict(case, split='train', kind='general', case_role='audit'), memory.read())
        assert args['enabled_branches'] == changed['enabled_branches']
        torch.testing.assert_close(args['prompt_ids'], changed['prompt_ids'], atol=0, rtol=0)
        row = run.infer(session, case, memory.read())
        audit_rows([row], {case['case_id']: case}, tokenizer.decode)
    finally:
        ledger.close()
