import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from replica_lab.contextual_completion_objective_v1 import CompletionObjectiveSpec, stop_policy_identity
from replica_lab.intrinsic_memory_budget_v5 import WorkLedger, COUNTERS, BudgetError
from replica_lab.intrinsic_memory_diagnostic_v5 import DiagnosticEngine, right_padded_prefix, logits_digest
from test_contextual_memory_v5 import make, spans, scope, open_store

@pytest.fixture(autouse=True,scope='module')
def threads():
    n=torch.get_num_threads();torch.set_num_threads(1);yield;torch.set_num_threads(n)

class FixtureBackend:
    def encode(self, text):
        # Fixed question for counterfactual worlds; this is a tokenizer fixture,
        # never native tokenization or a capability claim.
        return [3,5]
    def answer_ids(self,answer):return [{'red':7,'blue':9,'unknown':11}.get(answer,3)]
    def decode(self,ids):return ' '.join({7:'red',9:'blue',11:'unknown'}.get(i,'token'+str(i)) for i in ids)

def spec():
    return CompletionObjectiveSpec(32,8,1,2,(1,),stop_policy_identity((1,)),'a'*64,'b'*64,'none')

def ledger(tmp_path, *, stages=('test',), cap=10000):
    plan={stage:{**{k:cap for k in COUNTERS},'wall_seconds':120} for stage in stages}
    l=WorkLedger.create(tmp_path/'ledger.json',identity='cpu_v5',plan=plan,deadline=time.monotonic()+120)
    l.enter(stages[0]);return l

def engine(tmp_path,arm='two'):
    p=make(arm,objective=spec().identity());l=ledger(tmp_path)
    e=DiagnosticEngine(p,FixtureBackend(),spec(),l,max_new_tokens=3,output=tmp_path/'out');e.stage='test'
    return e

def row(answer='red',i=0):return {'world_id':'f'+str(i),'question':'Badge?', 'text':'Badge means '+answer+'.','answer':answer}

def source(e,tmp_path,r,i=0):
    s=scope('f'+str(i));store=open_store(tmp_path,name='store'+str(i));p=e.path
    p.admit(store,spans(r['text'],name=r['world_id'])[0],scope=s)
    return p.read(store,'Badge',scope=s),s,store

def test_reservation_before_dispatch_failed_work_not_refunded_and_child_shared(tmp_path):
    l=ledger(tmp_path,cap=2)
    with pytest.raises(RuntimeError):
        with l.work('test','failure',updates=1):raise RuntimeError('failed after dispatch')
    code='''import sys
from replica_lab.intrinsic_memory_budget_v5 import WorkLedger
l=WorkLedger(sys.argv[1],identity='cpu_v5')
with l.work('test','child',updates=1):pass
'''
    r=subprocess.run([sys.executable,'-c',code,str(l.path)],env={**os.environ,'PYTHONPATH':'src','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=15)
    assert r.returncode==0,r.stderr
    with pytest.raises(BudgetError):l.reserve('test','forbidden',updates=1)
    d=l.snapshot();assert d['reserved']['updates']==2 and d['events'][0]['status']=='failed'

def test_stage_deadline_does_not_replenish_or_open_twice(tmp_path,monkeypatch):
    l=ledger(tmp_path);t=l.snapshot()['stages']['test']['deadline']
    monkeypatch.setattr('replica_lab.intrinsic_memory_budget_v5.time.monotonic',lambda:t+1)
    with pytest.raises(BudgetError):l.reserve('test','late',forwards=1)
    with pytest.raises(BudgetError):l.enter('test')

@pytest.mark.parametrize('arm',['two','full_attention_nine','all'])
def test_objective_gradient_prefix_controls_and_one_owner(tmp_path,arm):
    e=engine(tmp_path,arm);b,s,store=source(e,tmp_path,row());other,osco,ostore=source(e,tmp_path,row('blue',1),1)
    original=logits_digest(torch.cat([p.detach().flatten().float() for p in e.path.adapter.base.parameters()]))
    first=e.update(row(),b,s);second=e.update(row(),b,s)
    assert first['gradients']['producer']['norm']==0
    assert second['gradients']['producer']['norm']>0
    assert all(v['output_norm']>0 for v in second['gradients']['sites'].values())
    assert len(e.path.assert_optimizer_ownership())==3
    with torch.no_grad():live=e.live(b,s);wrong=e.live(other,osco)
    enc=e.encoding(row());teacher=e.forward(enc.input_ids,enc.attention_mask,live,s)
    gen=e.generate(row(),live,s);assert torch.equal(teacher[0,1],gen['first_logits'])
    with torch.no_grad():
        logits=[e.forward(enc.input_ids,enc.attention_mask,live,s,control=c,wrong=wrong,wrong_scope=osco)
                for c in ('contextual','wrong','shuffled','zero','absent')]
    assert all(not torch.equal(logits[0],v) for v in logits[1:])
    assert original==logits_digest(torch.cat([p.detach().flatten().float() for p in e.path.adapter.base.parameters()]))
    assert e.path.adapter.extension.spine is None
    e.freeze()
    with pytest.raises(ValueError,match='closed'):e.update(row(),b,s)
    store.close();ostore.close()

def test_fresh_process_complete_checkpoint_logits_and_actual_continuation(tmp_path):
    e=engine(tmp_path);b,s,store=source(e,tmp_path,row());e.update(row(),b,s);e.update(row(),b,s)
    e.path.save(tmp_path/'checkpoint')
    with torch.no_grad():
        live=e.live(b,s);enc=e.encoding(row());before=logits_digest(e.forward(enc.input_ids,enc.attention_mask,live,s))
    (tmp_path/'expected.json').write_text(json.dumps({'logits':before}))
    store.close()
    code='''import torch,sys,json
from pathlib import Path
from test_intrinsic_diagnostic_v5 import engine,row,scope,open_store,logits_digest
torch.set_num_threads(1);root=Path(sys.argv[1]);child=root/'child';child.mkdir()
e=engine(child);e.path.load(root/'checkpoint');s=scope('f0');store=open_store(root,name='store0')
# Old durable values are canonical sources for live re-encoding after learning.
from replica_lab.contextual_release_v5 import ProducerReleaseCorpus
c=ProducerReleaseCorpus(store,s);c.rebuild(e.path);e.path.release_corpus=c
b=e.read(store,s,'Badge');enc=e.encoding(row())
with torch.no_grad():out=e.forward(enc.input_ids,enc.attention_mask,e.live(b,s),s)
assert logits_digest(out)==json.loads((root/'expected.json').read_text())['logits']
r=e.update(row(),b,s);assert r['changed_tensors']
e.path.assert_optimizer_ownership();store.close()
'''
    r=subprocess.run([sys.executable,'-c',code,str(tmp_path)],env={**os.environ,'PYTHONPATH':'src:tests','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=30)
    assert r.returncode==0,r.stderr

def test_incomplete_context_refuses_instead_of_truncating(tmp_path):
    e=engine(tmp_path)
    with pytest.raises(ValueError):right_padded_prefix(list(range(3,12)),spec(),'cpu')
