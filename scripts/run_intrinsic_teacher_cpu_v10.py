#!/usr/bin/env python3
"""Integrated teacher-alignment 2/9/36 CPU workflow; synthetic qualification, never native evidence.

Uses the real producer/store/readers and complete objective with a random tiny
BF16 backbone and explicit tokenizer fixture. No calibration bypass exists in
the native launcher. This diagnostic does not claim semantic fitting success.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from replica_lab.contextual_spark_fixture import build_contextual_adapter
from replica_lab.architecture.intrinsic_memory_layout_v4 import spark_layout
from replica_lab.contextual_memory import canonical_spans
from replica_lab.contextual_memory_v5 import ContextualMemoryV5
from replica_lab.contextual_release_v5 import ProducerReleaseCorpus
from replica_lab.contextual_completion_objective_v1 import CompletionObjectiveSpec,stop_policy_identity
from replica_lab.intrinsic_memory_diagnostic_v5 import logits_digest,CONTROLS
from replica_lab.intrinsic_memory_teacher_v10 import TeacherEngine
from replica_lab.contextual_teacher_objective_v1 import TeacherObjective
from replica_lab.intrinsic_memory_native_v5 import base_hash
from replica_lab.intrinsic_memory_budget_v5 import sha_json
from replica_lab.intrinsic_memory_budget_v5 import WorkLedger,COUNTERS,atomic
from replica_lab.tiered_memory import TieredMemoryStore,TierCeilings,MemoryScope

class Backend:
    def encode(self,text):return [3,5,8,13,17] if 'Evidence: ' in text else [3,5]
    def answer_ids(self,text):return [{'red':7,'blue':9,'unknown':11}.get(text,3)]
    def decode(self,ids):return ' '.join({7:'red',9:'blue',11:'unknown'}.get(i,'token'+str(i)) for i in ids)

def spec():return CompletionObjectiveSpec(128,16,1,2,(1,),stop_policy_identity((1,)),'a'*64,'b'*64,'none')

def make(arm):
    a,*_=build_contextual_adapter(seed=17,layers=36,spine_sites=(3,7),spine_slots=8,memory_size=256,vocab_size=128)
    p=ContextualMemoryV5(a,spark_layout(arm,hidden_size=8),source_identity='cpu_random_teacher_v10',objective_identity=TeacherObjective().identity(spec()),lr=0.0001,execution_identity='CPU_teacher_workflow_v10')
    a.extension.spine=None;return p

def store_at(root,i):return TieredMemoryStore(root/('store'+str(i)),profile_id='P1',enforce_volume_free=False,
    ceilings=TierCeilings(hot_bytes=32*1024**2,warm_bytes=32*1024**2,cold_bytes=32*1024**2,
        identity_reserved_bytes=65536,replay_reserved_bytes=262144,workspace_quota_bytes=1048576,
        artifact_budget_bytes=4*1048576,minimum_volume_free_bytes=0))

def row(i):return {'world_id':'cpu'+str(i),'question':'What color is the badge?',
                   'text':'The badge is '+('red' if i%2==0 else 'blue')+'.','answer':'red' if i%2==0 else 'blue'}
def scope(i):return MemoryScope('cpu'+str(i),'main','engineering',1000)

def run_arm(arm,root):
    root.mkdir();p=make(arm);baseline=logits_digest(torch.cat([v.detach().float().flatten() for v in p.adapter.base.parameters()]))
    plan={stage:{**{k:10000 for k in COUNTERS},'artifact_bytes':256*1024**2,'wall_seconds':120} for stage in ('mechanics','restart')}
    ledger=WorkLedger.create(root/'ledger.json',identity='cpu_'+arm,plan=plan,deadline=time.monotonic()+240)
    ledger.enter('mechanics');e=TeacherEngine(p,Backend(),spec(),ledger,max_new_tokens=3,output=root/'diagnostic',base_identity=base_hash(p.adapter.base),fit_identity=sha_json([row(i) for i in range(8)]));e.stage='mechanics'
    sources=[];stores=[]
    for i in range(2):
        store=store_at(root,i);stores.append(store);r=row(i)
        span=canonical_spans(r['text'],record_id=r['world_id'],source_id=r['world_id'],source_epoch=1,available_at=1)[0]
        with e.measured('admit',source_encodes=1,writes=1):p.admit(store,span,scope=scope(i))
        sources.append(e.read(store,scope(i),'badge'))
    if e.device.type!='cpu' or e.spec.vocab_size!=128:
        raise ValueError('synthetic teacher qualification is CPU tiny-fixture only')
    # The random tiny backbone cannot demonstrate competence. Supply an explicit
    # oracle fixture ONLY for control-flow qualification; target forwards and
    # all producer/reader updates below are real live tensor computations.
    actual_generate=e.generate
    def fixture_answer(r,*args,**kwargs):
        return {'raw_answer':r['answer'],'token_ids':e.backend.answer_ids(r['answer'])+[1],
            'terminated':True,'truncated':False,'correct':True,'false_recall':False,
            'first_logits':None,'synthetic_CPU_qualification':True}
    e.generate=fixture_answer
    assert e.prepare_teacher([row(i) for i in range(8)],[scope(i) for i in range(8)])
    e.generate=actual_generate
    teacher_cache=e.save_teacher_cache(root/'teacher_targets.json')
    updates=[e.update(row(i%2),sources[i%2],scope(i%2)) for i in range(2)]
    with torch.no_grad():live=e.live(sources[0],scope(0));wrong=e.live(sources[1],scope(1))
    c=ProducerReleaseCorpus(stores[0],scope(0))
    with e.measured('release',source_encodes=1,writes=1):c.rebuild(p)
    p.release_corpus=c;durable=e.read(stores[0],scope(0),'badge')
    enc=e.encoding(row(0));p.release_corpus=None
    # Rebuilt durable pins require the publication validator during parity.
    p.release_corpus=c
    with torch.no_grad():
        a=e.forward(enc.input_ids,enc.attention_mask,live,scope(0))
        before_durable=dict(p.cost)
        b=e.forward(enc.input_ids,enc.attention_mask,durable,scope(0))
        durable_cost={k:p.cost[k]-before_durable[k] for k in p.cost}
    assert torch.equal(a,b);p.release_corpus=None
    before=dict(p.cost)
    with torch.no_grad(),e.layer_probe() as layers:
        profiled=e.forward(enc.input_ids,enc.attention_mask,live,scope(0))
    delta={k:p.cost[k]-before[k] for k in p.cost}
    assert delta['snapshot_copies']==1 and delta['retrieval_calls']==0 and delta['admission_calls']==0
    assert set(layers)==set(str(s) for s in p.spec.sites)
    controls={}
    for control in CONTROLS:
        result=e.generate(row(0),live,scope(0),control=control,wrong=wrong,wrong_scope=scope(1));result.pop('first_logits')
        controls[control]=result
    e.freeze()
    assert not e.targets and not e.teacher_evidence
    with e.measured('checkpoint',artifact_bytes=128*1024**2,writes=1):p.save(root/'checkpoint')
    expected={'logits_sha256':logits_digest(a),'base_sha256':baseline,'teacher_cache':teacher_cache}
    atomic(root/'expected.json',expected)
    ledger.leave('mechanics',status='completed')
    for store in stores:store.close()
    p.close();ledger.enter('restart')
    child=subprocess.run([sys.executable,__file__,'--child',arm,'--output',str(root)],cwd=ROOT,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=120)
    (root/'child.stdout').write_text(child.stdout);(root/'child.stderr').write_text(child.stderr)
    if child.returncode:raise RuntimeError(child.stderr)
    return {'arm':arm,'reader_count':len(p.spec.sites),'early_reader_gradients':updates[1]['gradients']['sites'],
        'producer_gradient':updates[1]['gradients']['producer'],'single_forward_cost':delta,'layer_profiles':layers,
        'durable_forward_cost':durable_cost,'corpus_cost':c.cost,
        'round_trip_logits_exact':True,'controls':controls,'restart':json.loads((root/'restart.json').read_text()),
        'synthetic_teacher_qualification':True,'teacher_answers_verified':False,
        'teacher_target_cache_bytes':teacher_cache['bytes'],
        'physical_artifact_bytes':sum(v.stat().st_size for v in root.rglob('*') if v.is_file()),
        'costs':p.cost,'original_dtype':'bfloat16','native_admissible':False}

def child(arm,root):
    p=make(arm);p.load(root/'checkpoint');ledger=WorkLedger(root/'ledger.json',identity='cpu_'+arm)
    e=TeacherEngine(p,Backend(),spec(),ledger,max_new_tokens=3,output=root/'diagnostic',base_identity=base_hash(p.adapter.base),fit_identity=sha_json([row(i) for i in range(8)]));e.stage='restart'
    expected=json.loads((root/'expected.json').read_text())
    e.load_teacher_cache(root/'teacher_targets.json',expected['teacher_cache'],[row(i) for i in range(8)])
    store=store_at(root,0);p.release_corpus=ProducerReleaseCorpus(store,scope(0));b=e.read(store,scope(0),'badge')
    enc=e.encoding(row(0))
    with torch.no_grad():logits=e.forward(enc.input_ids,enc.attention_mask,e.live(b,scope(0)),scope(0))
    expected=json.loads((root/'expected.json').read_text());assert logits_digest(logits)==expected['logits_sha256']
    p.resume_training();update=e.update(row(0),b,scope(0))
    assert expected['base_sha256']==logits_digest(torch.cat([v.detach().float().flatten() for v in p.adapter.base.parameters()]))
    atomic(root/'restart.json',{'fresh_process_logits_exact':True,'actual_continuation_updates':1,
        'changed_tensors':update['changed_tensors'],'original_weights_unchanged':True,
        'optimizer_owner_count':1,'teacher_cache_restored':True,'teacher_forward_in_continuation':update['teacher_forward_in_update'],
        'uninterrupted_update_equivalence_measured':False})
    ledger.leave('restart',status='completed');p.close();store.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--child',choices=('two','full_attention_nine','all'));args=parser.parse_args()
    torch.set_num_threads(1)
    if args.child:child(args.child,args.output);return
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    results=[run_arm(a,args.output/a) for a in ('two','full_attention_nine','all')]
    atomic(args.output/'result.json',{'format':'replica.intrinsic_memory.teacher_cpu_workflow.v10','arms':results,
        'wall_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'synthetic_teacher_qualification':True,'teacher_answers_verified':False,
        'native_model_loads':0,'native_competence':False,'semantic_fitting_claim':False,'useful_recall':False,
        'persistent_learning':False,'RSI':False})

if __name__=='__main__':main()
