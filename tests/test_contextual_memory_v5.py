from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest
import torch
import torch.nn.functional as F

from replica_lab.contextual_memory import canonical_spans, ContextualMemoryError
from replica_lab.contextual_memory_v5 import (ContextualMemoryV5, PendingContextualRead,
    ContextualReadFailure, atomic_json, digest)
from replica_lab.contextual_release_v5 import ProducerReleaseCorpus
from replica_lab.architecture.intrinsic_memory_layout_v4 import spark_layout
from replica_lab.contextual_spark_fixture import build_contextual_adapter, ids_and_mask, labels_from_ids
from replica_lab.tiered_memory import MemoryScope
from test_tiered_memory import open_store, scope

@pytest.fixture(autouse=True, scope='module')
def threads():
    n=torch.get_num_threads();torch.set_num_threads(1);yield;torch.set_num_threads(n)

def make(arm='two', objective=None):
    adapter,*_=build_contextual_adapter(seed=17,layers=36,spine_sites=(3,7),spine_slots=8,memory_size=256)
    p=ContextualMemoryV5(adapter,spark_layout(arm,hidden_size=8),source_identity='random_fixture17',
                         objective_identity=objective,lr=0.001)
    adapter.extension.spine=None
    return p

def spans(text='Badge means red.', name='r', available=5, valid_until=None):
    return canonical_spans(text,record_id=name,source_id='source',source_epoch=1,
                           available_at=available,valid_until=valid_until)

def snapshot_scope():return MemoryScope('user_a','main','assistant',20,mode='source_snapshot',snapshot_at=10)

def step(p, batch=None):
    ids,mask=ids_and_mask(4);p.optimizer.zero_grad(set_to_none=True)
    b=p.training_values(spans(),scope=scope()) if batch is None else p.training_from_stored(batch,scope=scope())
    logits=p.forward(ids,mask,b,scope=scope())
    F.cross_entropy(logits.flatten(0,1),labels_from_ids(ids).flatten()).backward();p.optimizer.step()
    return logits

@pytest.mark.parametrize('available,until,eligible',[(15,None,False),(5,15,True),(5,10,False),(10,None,True)])
def test_snapshot_cutoff_live_and_direct_validation(available,until,eligible):
    p=make();s=spans(available=available,valid_until=until)
    if eligible:
        b=p.training_values(s,scope=snapshot_scope());p.validate(b,snapshot_scope())
    else:
        with pytest.raises(ContextualMemoryError):p.training_values(s,scope=snapshot_scope())
        b=p.training_values(spans(),scope=snapshot_scope());b.spans=s
        with pytest.raises(ContextualMemoryError):p.validate(b,snapshot_scope())

@pytest.mark.parametrize('snapshot', [False,True])
@pytest.mark.parametrize('available,until',[(15,None),(5,15)])
def test_durable_time_scope_and_store_restart(tmp_path,snapshot,available,until):
    p=make();sc=snapshot_scope() if snapshot else scope(as_of=20)
    store=open_store(tmp_path);p.admit(store,spans(available=available,valid_until=until)[0],scope=sc)
    eligible=(available<=sc.cutoff and (until is None or until>sc.cutoff))
    for restart in (False,True):
        if restart:store.close();store=open_store(tmp_path)
        if eligible:
            b=p.read(store,'Badge',scope=sc);assert b.spans[0].available_at==available
        else:
            with pytest.raises(ContextualMemoryError):p.read(store,'Badge',scope=sc)
    store.close()

@pytest.mark.parametrize('arm', ['two','full_attention_nine','all'])
def test_masked_padding_invariance_and_graph(arm):
    p=make(arm);step(p);step(p);ids,mask=ids_and_mask(4)
    b=p.training_values(spans(),scope=scope());b.slots.retain_grad()
    def run(fill):
        c=p.snapshot(ids,b,scope=scope(),padding_value=fill)
        assert c.values.shape[2]==8
        return p.adapter.forward_text(ids,mask,use_projector=False,use_ports=False,use_spine=False,intrinsic_memory=c)
    a=run(0);assert torch.equal(a,run(1e6));a.sum().backward();assert b.slots.grad.abs().sum()>0
    assert p.adapter.base_unchanged()


def test_release_hash_cache_invalidates_after_optimizer_and_reload(tmp_path):
    p=make();v=p.encoder.release();calls=p.cost['producer_hash_calls']
    for _ in range(5):assert p.encoder.release()==v
    assert p.cost['producer_hash_calls']==calls
    step(p);step(p);assert p.encoder.release()!=v
    p.save(tmp_path/'ckpt');q=make();q.load(tmp_path/'ckpt');assert q.encoder.release()==p.encoder.release()

@pytest.mark.parametrize('damage',['objective','slot_policy','temporal_rule'])
def test_checkpoint_contract_refused_before_mutation(tmp_path,damage):
    p=make(objective={'eos':1,'template':'a'});step(p);p.save(tmp_path/'ckpt')
    q=make(objective={'eos':1,'template':'a'});before={k:v.clone() for k,v in q._tensors().items()}
    mf=tmp_path/'ckpt/manifest.json';m=json.loads(mf.read_text());m[damage]='different';mf.write_text(json.dumps(m))
    with pytest.raises(ContextualMemoryError):q.load(tmp_path/'ckpt')
    assert all(torch.equal(v,q._tensors()[k]) for k,v in before.items())


def admit_cold(p,store):
    for i in range(2):p.admit(store,spans('Badge shared '+str(i),name='r'+str(i))[0],scope=scope(),placement='cold')


def test_multiple_cold_tickets_no_partial_success_and_accumulated_cost(tmp_path):
    p=make();store=open_store(tmp_path);admit_cold(p,store)
    pending=p.read(store,'Badge shared',scope=scope())
    assert isinstance(pending,PendingContextualRead)
    assert not hasattr(pending,'slots')
    with pytest.raises(ContextualMemoryError):p.validate(pending,scope())
    b=p.resume(store,pending,scope=scope(),wait_seconds=1)
    assert len(b.spans)==2 and b.slots.shape==(8,256)
    assert b.read_costs['cold_completed_files']==10
    assert b.read_costs['read_file_calls']>=12
    assert b.read_costs['cold_completed_bytes']==b.read_costs['cold_reserved_bytes']
    p.close();store.close()

@pytest.mark.parametrize('mutation',['correction','revocation','corrupt','missing'])
def test_invalidated_or_unavailable_cold_payload_never_becomes_partial_success(tmp_path,mutation):
    p=make();store=open_store(tmp_path);admit_cold(p,store);pending=p.read(store,'Badge',scope=scope())
    if mutation=='correction':
        s=replace(spans('Badge changed',name='new',available=6)[0],source_epoch=2,supersedes='r0.span0')
        p.correct(store,'r0.span0',s,scope=scope())
    elif mutation=='revocation':store.revoke_record('r0.span0',reason='withdrawn',scope=scope())
    else:
        row=store.connection.execute('SELECT text_path FROM records WHERE record_id=?',('r0.span0',)).fetchone()
        f=store.root/row['text_path']
        if mutation=='corrupt':f.write_text('damaged')
        else:f.unlink()
    with pytest.raises(ContextualReadFailure) as e:p.resume(store,pending,scope=scope(),wait_seconds=1)
    assert e.value.costs['cold_reserved_bytes']>0
    p.close();store.close()


@pytest.mark.parametrize('change',['revocation','correction'])
def test_bounded_wait_and_correction_during_actual_io(tmp_path,monkeypatch,change):
    import replica_lab.contextual_memory_v5 as module
    original=module._cold_payloads;started=threading.Event();release=threading.Event()
    def delayed(*a):started.set();release.wait(3);return original(*a)
    monkeypatch.setattr(module,'_cold_payloads',delayed)
    p=make();store=open_store(tmp_path);admit_cold(p,store);pending=p.read(store,'Badge',scope=scope())
    t=time.monotonic();pending=p.resume(store,pending,scope=scope(),wait_seconds=0.01)
    assert isinstance(pending,PendingContextualRead) and time.monotonic()-t<0.5 and started.is_set()
    if change=='revocation':store.revoke_record('r1.span0',reason='withdrawn_during_io',scope=scope())
    else:p.correct(store,'r1.span0',replace(spans('Badge changed',name='new',available=6)[0],source_epoch=2,supersedes='r1.span0'),scope=scope())
    release.set()
    with pytest.raises(ContextualReadFailure):p.resume(store,pending,scope=scope(),wait_seconds=1)
    p.close();store.close()


def test_pending_ticket_continues_in_fresh_process(tmp_path):
    p=make();store=open_store(tmp_path);admit_cold(p,store);pending=p.read(store,'Badge',scope=scope());store.close()
    (tmp_path/'pending.json').write_text(json.dumps(asdict(pending)))
    code='''import json,torch,sys
from pathlib import Path
from test_contextual_memory_v5 import make,scope,open_store,PendingContextualRead
torch.set_num_threads(1);root=Path(sys.argv[1]);p=make();store=open_store(root)
t=PendingContextualRead(**json.loads((root/'pending.json').read_text()))
b=p.resume(store,t,scope=scope(),wait_seconds=1)
assert len(b.spans)==2 and b.read_costs['cold_completed_files']==10
p.close();store.close()
'''
    r=subprocess.run([sys.executable,'-c',code,str(tmp_path)],env={**os.environ,'PYTHONPATH':'src:tests','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=30)
    assert r.returncode==0,r.stderr


def test_existing_corpus_new_producer_release_and_interruption_recovery(tmp_path):
    p=make();store=open_store(tmp_path);p.admit(store,spans()[0],scope=scope())
    old=p.read(store,'Badge',scope=scope());corpus=ProducerReleaseCorpus(store,scope());m1=corpus.rebuild(p)
    p.release_corpus=corpus;before=p.read(store,'Badge',scope=scope(),wait_seconds=1);assert torch.equal(old.slots,before.slots)
    step(p,old);step(p,old)
    with pytest.raises(ContextualMemoryError):p.read(store,'Badge',scope=scope())
    def crash():raise RuntimeError('simulated_crash')
    with pytest.raises(RuntimeError):corpus.rebuild(p,before_publish=crash)
    assert corpus._current()[1]['generation']==m1['generation']
    m2=corpus.rebuild(p);assert m2['generation']>m1['generation']+1
    after=p.read(store,'Badge',scope=scope(),wait_seconds=1);assert not torch.equal(before.slots,after.slots)
    assert (corpus.root/'00000001/manifest.json').is_file()
    p.save(tmp_path/'ckpt');store.close();store=open_store(tmp_path);q=make();q.load(tmp_path/'ckpt')
    q.release_corpus=ProducerReleaseCorpus(store,scope());restored=q.read(store,'Badge',scope=scope(),wait_seconds=1)
    assert torch.equal(after.slots,restored.slots);store.close()

@pytest.mark.parametrize('change',['correction','revocation'])
def test_rebuild_excludes_source_invalidated_before_publish(tmp_path,change):
    p=make();store=open_store(tmp_path);p.admit(store,spans()[0],scope=scope());corpus=ProducerReleaseCorpus(store,scope());corpus.rebuild(p)
    def mutate():
        if change=='revocation':store.revoke_record('r.span0',reason='withdrawn',scope=scope())
        else:p.correct(store,'r.span0',replace(spans('Badge blue',name='new',available=6)[0],source_epoch=2,supersedes='r.span0'),scope=scope())
    with pytest.raises(ContextualMemoryError,match='invalidated'):corpus.rebuild(p,before_publish=mutate)
    assert corpus._current()[1]['generation']==1
    if change=='correction':
        m=corpus.rebuild(p);assert set(m['records'])=={'new.span0'}
    store.close()


def test_release_mixed_versions_and_missing_latest_authority_refused(tmp_path):
    p=make();store=open_store(tmp_path);p.admit(store,spans()[0],scope=scope());corpus=ProducerReleaseCorpus(store,scope());corpus.rebuild(p)
    mf=corpus.root/'00000001/manifest.json';m=json.loads(mf.read_text());m['records']['r.span0']['producer_release']='wrong';m['record_digest']=digest(m['records']);mf.write_text(json.dumps(m))
    ptr=corpus.root/'current.json';v=json.loads(ptr.read_text());v['manifest_sha256']=hashlib.sha256(mf.read_bytes()).hexdigest();ptr.write_text(json.dumps(v))
    with pytest.raises(ContextualMemoryError,match='mixed'):corpus._current()
    mf.unlink()
    with pytest.raises(ContextualMemoryError):corpus._current()
    store.close()


def test_first_release_interruption_and_lost_pointer_fail_closed(tmp_path):
    p=make();store=open_store(tmp_path);p.admit(store,spans()[0],scope=scope())
    corpus=ProducerReleaseCorpus(store,scope())
    def crash():raise RuntimeError('interrupted before publication')
    with pytest.raises(RuntimeError):corpus.rebuild(p,before_publish=crash)
    corpus=ProducerReleaseCorpus(store,scope());assert corpus._current() is None
    assert corpus.rebuild(p)['generation']==2
    (corpus.root/'current.json').unlink()
    with pytest.raises(ContextualMemoryError):corpus._current()
    with pytest.raises(ContextualMemoryError):ProducerReleaseCorpus(store,scope())
    store.close()


def test_release_pending_invalidated_by_publication_or_revocation(tmp_path):
    p=make();store=open_store(tmp_path);p.admit(store,spans()[0],scope=scope())
    corpus=ProducerReleaseCorpus(store,scope());corpus.rebuild(p);p.release_corpus=corpus
    pending=p.read(store,'Badge',scope=scope());assert isinstance(pending,PendingContextualRead)
    corpus.rebuild(p)
    with pytest.raises(ContextualReadFailure,match='publication'):p.resume(store,pending,scope=scope(),wait_seconds=1)
    pending=p.read(store,'Badge',scope=scope());store.revoke_record('r.span0',reason='withdrawn',scope=scope())
    with pytest.raises(ContextualReadFailure):p.resume(store,pending,scope=scope(),wait_seconds=1)
    p.close();store.close()


def test_recover_uncertain_dispatch_keeps_reservation_and_retry_bound(tmp_path):
    p=make();store=open_store(tmp_path);admit_cold(p,store);pending=p.read(store,'Badge',scope=scope())
    path=p._ticket_path(store,pending.ticket_id);doc=json.loads(path.read_text())
    required=sum(t['required_bytes'] for t in doc['tickets'])
    # Durable state at a crash just after dispatch reservation, before completion.
    doc['attempts']=1;doc['costs']['cold_reserved_bytes']=required;doc['costs']['cold_reserved_files']=10
    p._persist(store,pending.ticket_id,doc);store.close();store=open_store(tmp_path);q=make()
    recovered=q.recover_pending(store,pending.ticket_id,scope=scope())
    b=q.resume(store,recovered,scope=scope(),wait_seconds=1)
    assert b.read_costs['cold_reserved_bytes']==2*required
    assert b.read_costs['cold_completed_bytes']==required
    assert b.read_costs['ticket_write_bytes_reserved']>0
    with pytest.raises(ContextualReadFailure):q.recover_pending(store,pending.ticket_id,scope=scope())
    q.close();store.close()


def test_release_reopens_at_later_time_but_expired_records_never_reappear(tmp_path):
    p=make();store=open_store(tmp_path);early=scope(as_of=10)
    p.admit(store,spans(valid_until=15)[0],scope=early)
    corpus=ProducerReleaseCorpus(store,early);corpus.rebuild(p)
    later=scope(as_of=12);p.release_corpus=ProducerReleaseCorpus(store,later)
    assert p.read(store,'Badge',scope=later,wait_seconds=1).spans[0].valid_until==15
    expired=scope(as_of=20);p.release_corpus=ProducerReleaseCorpus(store,expired)
    with pytest.raises(ContextualMemoryError):p.read(store,'Badge',scope=expired,wait_seconds=1)
    p.close();store.close()


def test_durable_overlay_requires_release_authority_even_after_loading(tmp_path):
    p=make();store=open_store(tmp_path);p.admit(store,spans()[0],scope=scope())
    p.release_corpus=ProducerReleaseCorpus(store,scope());p.release_corpus.rebuild(p)
    b=p.read(store,'Badge',scope=scope(),wait_seconds=1);p.release_corpus=None
    with pytest.raises(ContextualMemoryError,match='authority'):p.validate(b,scope())
    p.close();store.close()


def test_producer_change_at_publish_boundary_keeps_previous_release(tmp_path):
    p=make();store=open_store(tmp_path);p.admit(store,spans()[0],scope=scope())
    c=ProducerReleaseCorpus(store,scope());c.rebuild(p)
    def mutate():step(p);step(p)  # the first zero-output update opens the producer path
    with pytest.raises(ContextualMemoryError,match='changed before publication'):c.rebuild(p,before_publish=mutate)
    assert c._current()[1]['generation']==1
    store.close()
