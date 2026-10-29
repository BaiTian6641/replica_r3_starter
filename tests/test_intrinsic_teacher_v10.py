"""CPU engineering only; synthetic qualification never admits a native packet."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from replica_lab.architecture.intrinsic_memory_layout_v4 import spark_layout
from replica_lab.contextual_spark_fixture import build_contextual_adapter
from replica_lab.contextual_completion_objective_v1 import CompletionObjectiveSpec, stop_policy_identity
from replica_lab.contextual_teacher_objective_v1 import TeacherObjective
from replica_lab.contextual_memory_v5 import ContextualMemoryV5, ContextualMemoryError
from replica_lab.contextual_memory import canonical_spans
from replica_lab.contextual_release_v5 import ProducerReleaseCorpus
from replica_lab.intrinsic_memory_budget_v5 import WorkLedger, COUNTERS, sha_json
from replica_lab.intrinsic_memory_diagnostic_v5 import logits_digest
from replica_lab.intrinsic_memory_teacher_v10 import TeacherEngine
from replica_lab.intrinsic_memory_native_v5 import base_hash
from replica_lab.tiered_memory import MemoryScope
from test_tiered_memory import open_store

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True, scope='module')
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class Backend:
    def encode(self, text):
        return [3, 5, 8, 13, 17] if 'Evidence: ' in text else [3, 5]

    def answer_ids(self, text):
        return [{'red': 7, 'blue': 9, 'unknown': 11}.get(text, 3)]

    def decode(self, ids):
        return ' '.join({7: 'red', 9: 'blue', 11: 'unknown'}.get(i, 'token'+str(i)) for i in ids)


def rows():
    return [{'world_id': 'teacher_fit_'+str(i), 'question': 'Badge '+str(i//2)+'?',
             'text': 'The badge is '+('red' if i%2 == 0 else 'blue')+'.',
             'answer': 'red' if i%2 == 0 else 'blue'} for i in range(8)]


def scopes():
    return [MemoryScope(r['world_id'], 'main', 'engineering', 1000) for r in rows()]


def make(root, arm='two'):
    root.mkdir(exist_ok=True, parents=True)
    adapter, *_ = build_contextual_adapter(seed=17, layers=36, spine_sites=(3, 7),
                                           spine_slots=8, memory_size=256, vocab_size=128)
    spec = CompletionObjectiveSpec(128, 16, 1, 2, (1,), stop_policy_identity((1,)), 'a'*64, 'b'*64, 'none')
    objective = TeacherObjective()
    path = ContextualMemoryV5(adapter, spark_layout(arm, hidden_size=8),
        source_identity='CPU_teacher_fixture_v10', objective_identity=objective.identity(spec),
        execution_identity='CPU_teacher_fixture_v10', lr=0.0001)
    adapter.extension.spine = None
    plan = {'test': {**{k: 10000 for k in COUNTERS}, 'artifact_bytes': 128*1024**2, 'wall_seconds': 120}}
    ledger = WorkLedger.create(root/'ledger.json', identity='CPU_teacher_v10', plan=plan, deadline=time.monotonic()+120)
    ledger.enter('test')
    engine = TeacherEngine(path, Backend(), spec, ledger, max_new_tokens=3, output=root/'out',
        base_identity=base_hash(adapter.base), fit_identity=sha_json(rows()))
    engine.stage = 'test'
    return engine


def synthetic_qualification(engine, monkeypatch):
    def answer(row, batch, scope, **kwargs):
        assert kwargs['control'] == 'exact_text' and batch is None
        return {'raw_answer': row['answer'], 'token_ids': engine.backend.answer_ids(row['answer'])+[1],
                'terminated': True, 'truncated': False, 'correct': True, 'false_recall': False,
                'first_logits': None, 'synthetic_CPU_qualification': True}
    with monkeypatch.context() as scoped:
        scoped.setattr(engine, 'generate', answer)
        assert engine.prepare_teacher(rows(), scopes())


def source(engine, root, i=0):
    row, scope = rows()[i], scopes()[i]
    store = open_store(root, name='store'+str(i))
    span = canonical_spans(row['text'], record_id=row['world_id'], source_id=row['world_id'],
                           source_epoch=1, available_at=1)[0]
    engine.path.admit(store, span, scope=scope)
    return engine.path.read(store, 'badge', scope=scope), scope, store


@pytest.mark.parametrize('arm', ['two', 'full_attention_nine', 'all'])
def test_live_KL_reaches_producer_and_every_reader_without_teacher_or_base_gradient(tmp_path, monkeypatch, arm):
    e = make(tmp_path, arm)
    initial = base_hash(e.path.adapter.base)
    synthetic_qualification(e, monkeypatch)
    source_batch, scope, store = source(e, tmp_path)
    try:
        before = e.ledger.snapshot()['reserved'].copy()
        first = e.update(rows()[0], source_batch, scope)
        second = e.update(rows()[0], source_batch, scope)
        assert first['gradients']['producer']['norm'] == 0
        assert second['gradients']['producer']['norm'] > 0
        assert all(g['norm'] > 0 for g in second['gradients']['sites'].values())
        assert all(p.grad is None for p in e.path.adapter.base.parameters())
        assert all(not t.log_probs.requires_grad for t in e.targets.values())
        assert e.ledger.snapshot()['reserved']['forwards'] - before['forwards'] == 2
        assert first['teacher_forward_in_update'] is False
        assert base_hash(e.path.adapter.base) == initial
        assert len(e.path.assert_optimizer_ownership()) == 3
        e.freeze()
        assert not e.targets and not e.teacher_evidence and e.fitting_closed
        with pytest.raises(ValueError, match='closed'): e.update(rows()[0], source_batch, scope)
    finally:
        store.close()
        e.path.close()


def test_failed_or_incomplete_teacher_keeps_updates_closed(tmp_path, monkeypatch):
    e = make(tmp_path)
    with pytest.raises(ValueError, match='eight-world'): e.prepare_teacher(rows()[:7], scopes()[:7])
    def failed(row, *args, **kwargs):
        return {'raw_answer': 'unknown', 'token_ids': [11, 1], 'terminated': True, 'truncated': False,
                'correct': False, 'false_recall': False, 'first_logits': None}
    monkeypatch.setattr(e, 'generate', failed)
    assert not e.prepare_teacher(rows(), scopes())
    with pytest.raises(ValueError, match='teacher competence'): e.update(rows()[0], None, scopes()[0])
    assert e.ledger.snapshot()['reserved']['updates'] == 0
    assert not e.targets
    e.path.close()


def test_teacher_skips_readers_and_student_controls_use_live_values(tmp_path, monkeypatch):
    e = make(tmp_path)
    calls = []
    handles = [port.register_forward_hook(lambda *args: calls.append(True)) for port in e.path.reader.ports.values()]
    synthetic_qualification(e, monkeypatch)
    assert calls == []
    b, scope, store = source(e, tmp_path)
    wrong, wrong_scope, wrong_store = source(e, tmp_path, 1)
    try:
        e.update(rows()[0], b, scope);e.update(rows()[0], b, scope)
        assert calls
        enc = e.encoding(rows()[0])
        with torch.no_grad():
            live, other = e.live(b, scope), e.live(wrong, wrong_scope)
            values = {c: e.forward(enc.input_ids, enc.attention_mask, live, scope,
                                   control=c, wrong=other, wrong_scope=wrong_scope)
                      for c in ('contextual', 'wrong', 'shuffled', 'zero', 'absent')}
        assert all(not torch.equal(values['contextual'], v) for c, v in values.items() if c != 'contextual')
    finally:
        for h in handles: h.remove()
        store.close();wrong_store.close();e.path.close()


@pytest.mark.parametrize('damage', ['bytes', 'wrong_world', 'wrong_base', 'unqualified', 'substituted'])
def test_teacher_cache_refuses_tampering_and_stale_bindings(tmp_path, monkeypatch, damage):
    e = make(tmp_path/'parent')
    synthetic_qualification(e, monkeypatch)
    cache = tmp_path/'teacher.json'
    pin = e.save_teacher_cache(cache)
    other = make(tmp_path/'other')
    try:
        if damage == 'bytes': cache.write_text(cache.read_text()+' ')
        elif damage == 'wrong_world':
            modified = rows();modified[0]['text'] = 'Corrected source'
        elif damage == 'wrong_base': other.base_identity = 'f'*64
        elif damage == 'unqualified':
            from replica_lab.intrinsic_memory_execution_v5 import hash_file
            payload = json.loads(cache.read_text());payload['qualification'][0]['raw_answer'] = 'unknown'
            cache.write_text(json.dumps(payload));pin = {'bytes': cache.stat().st_size, 'sha256': hash_file(cache)}
        else:
            actual = tmp_path/'actual.json';cache.rename(actual);cache.symlink_to(actual)
        with pytest.raises(ValueError):
            other.load_teacher_cache(cache, pin, modified if damage == 'wrong_world' else rows())
        assert not other.teacher_qualified and not other.targets
    finally:
        e.path.close();other.path.close()


def test_correction_invalidates_live_source_even_with_valid_teacher_cache(tmp_path, monkeypatch):
    from dataclasses import replace
    e = make(tmp_path)
    synthetic_qualification(e, monkeypatch)
    b, scope, store = source(e, tmp_path)
    try:
        corrected = canonical_spans('The badge is blue.', record_id='corrected', source_id=rows()[0]['world_id'],
                                     source_epoch=2, available_at=2)[0]
        e.path.correct(store, b.spans[0].record_id, replace(corrected, supersedes=b.spans[0].record_id), scope=scope)
        before = {k: v.clone() for k, v in e.path._tensors().items()}
        with pytest.raises(ContextualMemoryError): e.update(rows()[0], b, scope)
        assert all(torch.equal(v, e.path._tensors()[k]) for k, v in before.items())
    finally:
        store.close();e.path.close()


def test_complete_extension_teacher_cache_and_optimizer_continue_in_fresh_process(tmp_path, monkeypatch):
    e = make(tmp_path/'parent')
    synthetic_qualification(e, monkeypatch)
    b, scope, store = source(e, tmp_path)
    e.update(rows()[0], b, scope);e.update(rows()[0], b, scope)
    pin = e.save_teacher_cache(tmp_path/'teacher.json')
    e.path.save(tmp_path/'checkpoint')
    with torch.no_grad():
        enc = e.encoding(rows()[0])
        expected = logits_digest(e.forward(enc.input_ids, enc.attention_mask, e.live(b, scope), scope))
    (tmp_path/'expected.json').write_text(json.dumps({'logits': expected, 'cache': pin, 'base': e.base_identity}))
    store.close();e.path.close()
    code = '''
from pathlib import Path
import json,sys,torch
sys.path[:0]=[sys.argv[2]+'/src',sys.argv[2]+'/tests']
from test_intrinsic_teacher_v10 import make,rows,scopes,open_store,ProducerReleaseCorpus,logits_digest,base_hash
torch.set_num_threads(1);root=Path(sys.argv[1]);e=make(root/'child')
expected=json.loads((root/'expected.json').read_text())
e.path.load(root/'checkpoint');e.path.assert_optimizer_ownership()
e.load_teacher_cache(root/'teacher.json',expected['cache'],rows())
store=open_store(root,name='store0');scope=scopes()[0]
corpus=ProducerReleaseCorpus(store,scope);corpus.rebuild(e.path);e.path.release_corpus=corpus
b=e.read(store,scope,'badge');enc=e.encoding(rows()[0])
with torch.no_grad():out=e.forward(enc.input_ids,enc.attention_mask,e.live(b,scope),scope)
assert logits_digest(out)==expected['logits']
event=e.update(rows()[0],b,scope)
assert event['changed_tensors'] and event['gradients']['producer']['norm']>0
assert base_hash(e.path.adapter.base)==expected['base']
assert all(g['lr']==0.0001 for g in e.path.optimizer.param_groups)
print(json.dumps({'fresh_logits_exact':True,'actual_continuation_updates':1,'teacher_forward_in_update':event['teacher_forward_in_update']}))
store.close();e.path.close()
'''
    child = subprocess.run([sys.executable, '-c', code, str(tmp_path), str(ROOT)], cwd=tmp_path,
                           env={k: v for k, v in os.environ.items() if k != 'PYTHONPATH'},
                           capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout)['actual_continuation_updates'] == 1
