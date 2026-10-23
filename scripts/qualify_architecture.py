#!/usr/bin/env python3
"""Bounded, preregistered E1/E2 native qualification; original model reused once."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ[key] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def execute(args, profiler, report):
    import torch
    from replica_lab.architecture.loading import load_local_mps, encode_prompt
    from replica_lab.architecture.minicpm5 import MiniCPM5Adapter
    from replica_lab.architecture.contracts import TaskGraph, PendingWrite
    from replica_lab.memory_system import MemoryRecord, RecordLedger, EligibilityScope
    from replica_lab.memory_system.session import MemorySession
    from replica_lab.optimizers import build_optimizer

    protocol = json.loads(args.protocol.read_text())
    if protocol['format'] != 'replica.e1_e2.qualification.v1':
        raise ValueError('Unknown qualification protocol')
    expected = {'device':'mps','dtype':'float32','native_cache':False,'native_cpu_fallback':False,
                'source_tokens_per_unit':128,'memory_units':16,'slots_per_unit':4,
                'memory_width':256,'ports':[13,27],'observer':41}
    if any(protocol.get(k)!=v for k,v in expected.items()):
        raise ValueError('Protocol hardware/dimensions differ from the implemented native port')
    alignment=protocol['alignment']
    if (alignment['optimizer']!='muon8bit' or alignment['regime']!='modules_only'
            or alignment['grad_accum']!=len(alignment['order'])
            or sorted(alignment['order'])!=list(range(len(protocol['cases'])))
            or not 1<=alignment['warmup_updates']<=alignment['updates']):
        raise ValueError('Protocol training scope/order/warmup mismatch')
    if protocol['ranking']['grad_accum']!=2 or protocol['ranking']['trainable']!=['reranker','sense_ports.13']:
        raise ValueError('Protocol ranking scope mismatch')
    torch.manual_seed(protocol['seed'])
    model, tokenizer, source_lock = load_local_mps(args.model_path, args.model_lock, profiler=profiler)
    if source_lock['revision']!=protocol['source_revision']:
        raise ValueError('Protocol source differs from loaded original')
    report.update(manifest=model.manifest.to_dict(), manifest_digest=model.manifest.digest,
                  source_lock=source_lock, parameters=model.parameter_audit(), checks={}, generations={})
    initial = {n:p.detach().clone() for n,p in model.new_named_parameters()}
    original_stamp = model.parameter_stamp(original_only=True)
    ledger_path = args.output / 'records.sqlite'
    ledger = RecordLedger(ledger_path)
    session = MemorySession(model, tokenizer, ledger, cache_dir=args.output/'record_latents')
    cases = protocol['cases']
    subject, purpose = 'synthetic_e12', 'architecture_diagnostic'
    for case in cases:
        ledger.admit(MemoryRecord(case['id'], subject, case['id'], case['source'], 1, 1,
                                  purpose, source_id='authored_e1_e2_fixture', authenticity='unverified'),
                     authorized_subject=subject, authorized_branch=case['id'], source_authorized=True)
    task = TaskGraph('e1_e2', 'Read the scoped synthetic project record')
    def scope(case):return EligibilityScope(subject, case['id'], purpose, 10)
    def prompt(case):return protocol['prompt'].format(entity=case['entity'])
    def kwargs(case, **overrides):
        ids = encode_prompt(tokenizer, prompt(case))
        values = dict(prompt_ids=ids, prompt_mask=torch.ones_like(ids,dtype=torch.bool),
                      query=case['entity'], scope=scope(case), task=task)
        values.update(overrides)
        return values
    def action(case):
        ids = encode_prompt(tokenizer, prompt(case))
        response = tokenizer(case['answer'],add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
        full = torch.cat((ids,torch.tensor([response],device='mps',dtype=torch.long)),dim=1)
        if full.shape[1] > protocol['action_tokens']:raise ValueError('Action length exceeds preregistered bound')
        labels=full.clone();labels[:,:ids.shape[1]]=-100
        return full,labels
    def mark(name, **fields):
        profiler.mark(name, **fields)
        dump(args.output/'progress.json',report)
        print(json.dumps({'phase':name,**fields}),flush=True)
    def require(name, value, **details):
        report['checks'][name]={'passed':bool(value),**details}
        if not value:raise AssertionError(name)
    def logits(case, enabled_ports=None):
        request=session.prepare(**kwargs(case,enabled_ports=enabled_ports))
        try:
            with torch.no_grad():
                return session.forward(request,request.frame.sense.prompt_ids).logits[:,-1].detach().clone()
        finally:request.release()
    def generate(case, *, enabled_ports=None, exact=False, name=None):
        request=session.prepare(**kwargs(case,enabled_ports=enabled_ports))
        citations=[]
        try:
            if exact:
                citations=[asdict(session.exact_text(request,s)) for s in request.frame.memory.spans]
                selection=request.selection
                request.release()
                ids=encode_prompt(tokenizer,'Observed source records:\n'+'\n'.join(c['text'] for c in citations)
                                  +'\n\n'+prompt(case))
                request=session.prepare(**kwargs(case,prompt_ids=ids,
                    prompt_mask=torch.ones_like(ids,dtype=torch.bool),selection=selection,enabled_ports=()))
            selected=[s.record_id for s in request.frame.memory.spans]
            tokens=session.generate(request,max_new_tokens=protocol['generation_max_new_tokens'],
                                    eos_token_id=tokenizer.eos_token_id)
            ids=list(request.generated_ids)
            raw=tokenizer.decode(ids,skip_special_tokens=False)
            answer=tokenizer.decode(ids,skip_special_tokens=True)
            return {'case_id':case['id'],'arm':name or ('exact_text' if exact else 'latent_only'),
                    'expected':case['answer'],'completion_ids':ids,'raw_text':raw,'text':answer,
                    'correct':answer.strip()==case['answer'],'selected_records':selected,
                    'citations':citations,'prompt_tokens':request.frame.sense.prompt_ids.shape[1],
                    'query_contains_source':exact}
        finally:request.release()

    # Exact zero branches must reproduce the same native unpadded FP32 calculation.
    request=session.prepare(**kwargs(cases[0]))
    ids=request.frame.sense.prompt_ids
    with torch.no_grad():
        native=model.base(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits
        augmented=session.forward(request,ids).logits
        difference=float((native-augmented).abs().max())
    require('initial_null_native_parity',difference<=1e-5,max_absolute_difference=difference)
    request.release();del native,augmented,request
    # Compare the source prefix to an observation-only diagnostic of the original native layer.
    source_ids=torch.tensor([tokenizer(cases[0]['source'],add_special_tokens=False)['input_ids']],device='mps')
    captured=[]
    handle=model.base.model.layers[13].register_forward_hook(lambda m,a,o:captured.append(o.detach().clone()))
    try:
        with torch.no_grad():model.base(input_ids=source_ids,use_cache=False)
    finally:handle.remove()
    with torch.no_grad():features=model.encode_observations(source_ids,torch.ones_like(source_ids,dtype=torch.bool))
    difference=float((features-captured[0]).abs().max())
    require('source_prefix_native_parity',difference<=1e-5,max_absolute_difference=difference)
    del features,captured
    report['generations']['initial']=[generate(c) for c in cases]
    mark('initial_parity_and_generation')

    # Perturb only new projections for wiring probes, then restore their original bytes.
    with torch.no_grad():
        for port in model.memory_ports.values():port.output_projection.weight.normal_(0,0.002)
    disabled=logits(cases[0],())
    for layer in model.ports.injection_layers:
        active=logits(cases[0],(layer,))
        delta=float((active-disabled).abs().max())
        require(f'port_{layer}_nonzero',bool(torch.isfinite(active).all()) and delta>1e-6,
                max_absolute_logit_difference=delta)
    a=logits(cases[0]);b=logits(cases[1]);again=logits(cases[0])
    require('request_A_B_A_isolation',torch.equal(a,again) and not torch.equal(a,b),
            same_request_max_difference=float((a-again).abs().max()),
            different_source_max_difference=float((a-b).abs().max()))
    del active,disabled,a,b,again
    for regime in ('modules_only','full_augmented'):
        model.set_regime(regime);model.train();model.zero_grad(set_to_none=True)
        full,labels=action(cases[0])
        with session.training_request(**kwargs(cases[0])) as request:
            request.frame.memory.slots.retain_grad()
            output=session.forward(request,full,labels=labels)
            session.backward(request,output.loss)
            slot_gradient=float(request.frame.memory.slots.grad.abs().sum())
            audit=model.gradient_audit()
            expected_base=381 if regime=='full_augmented' else 0
            active_names=[n for n,p in model.new_named_parameters() if n.startswith(('record_writer.','memory_ports.'))]
            require(f'{regime}_checkpoint_backward',
                    len(audit['original']['present'])==expected_base and not audit['original']['nonfinite']
                    and set(active_names)<=set(audit['new']['present']) and not audit['new']['nonfinite']
                    and slot_gradient>0,loss=float(output.loss),slot_gradient_l1=slot_gradient,gradients=audit)
        del output,full,labels,request
        model.zero_grad(set_to_none=True);gc.collect();torch.mps.empty_cache()
        mark(f'{regime}_backward')
    # Isolate source encoding gradients from answer-pass gradients.
    features=model.encode_observations(source_ids,torch.ones_like(source_ids,dtype=torch.bool))
    slots,_=model.record_writer(features[:,None],torch.ones((1,1,source_ids.shape[1]),device='mps',dtype=torch.bool))
    weights=torch.linspace(-1,1,slots.numel(),device='mps').reshape_as(slots)
    (slots*weights).sum().backward()
    audit=model.gradient_audit()
    require('D2_source_gradient_isolated', 'model.embed_tokens.weight' in audit['original']['nonzero']
            and any(n.startswith('model.layers.13.') for n in audit['original']['nonzero'])
            and not any(n.startswith('model.layers.14.') for n in audit['original']['present'])
            and 'lm_head.weight' not in audit['original']['present']
            and not audit['original']['nonfinite'],gradients=audit)
    del features,slots,weights
    model.zero_grad(set_to_none=True);model.set_regime('modules_only');gc.collect();torch.mps.empty_cache()
    # Explicit sense API wiring, separate from the E2 answer objective.
    sense=model.sense(encode_prompt(tokenizer,prompt(cases[0])),torch.ones_like(encode_prompt(tokenizer,prompt(cases[0])),dtype=torch.bool))
    sum((value*torch.linspace(-1,1,value.shape[-1],device='mps')).sum() for _,value in sense.summaries).backward()
    sensor_names=[n for n,p in model.new_named_parameters() if n.startswith('sense_ports.')]
    audit=model.gradient_audit()
    require('sense_ports_gradient_wiring',set(sensor_names)<=set(audit['new']['nonzero']) and not audit['new']['nonfinite'],
            trained_by_answer_loss=False,gradients=audit['new'])
    del sense;model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for name,p in model.new_named_parameters():p.copy_(initial[name])
    del initial;model.eval();gc.collect();torch.mps.empty_cache()
    mark('wiring_complete')

    optimizer=build_optimizer(model,'muon8bit',lr_base=0,lr_new=protocol['alignment']['lr_new'],
        lr_muon_base=0,lr_muon_new=protocol['alignment']['lr_muon_new'],weight_decay=0)
    report['alignment']={'optimizer':optimizer.audit(),'updates':[]}
    for update in range(1,protocol['alignment']['updates']+1):
        model.train();optimizer.zero_grad(set_to_none=True);total=0
        for index in protocol['alignment']['order']:
            case=cases[index];full,labels=action(case)
            with session.training_request(**kwargs(case)) as request:
                output=session.forward(request,full,labels=labels)
                loss=output.loss/protocol['alignment']['grad_accum']
                total+=float(loss.detach());session.backward(request,loss)
            del output,loss,request,full,labels
        norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                 protocol['alignment']['gradient_clip'],error_if_nonfinite=True))
        ratio=min(1,update/protocol['alignment']['warmup_updates'])
        for group in optimizer.param_groups:
            group['lr']=protocol['alignment']['lr_muon_new' if group['algorithm']=='muon8bit' else 'lr_new']*ratio
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        report['alignment']['updates'].append({'update':update,'mean_response_loss':total,'gradient_norm':norm})
        mark('alignment_update',update=update,mean_response_loss=total)
    report['alignment']['optimizer_after']=optimizer.audit()
    del optimizer;model.eval();gc.collect();torch.mps.empty_cache()
    report['generations']['final_latent']=[generate(c) for c in cases]
    report['generations']['disabled']=[generate(c,enabled_ports=(),name='ports_disabled') for c in cases]
    report['generations']['exact_text']=[generate(c,exact=True) for c in cases]
    mark('alignment_generations_complete')
    # Ranking has its own objective and trainable scope. It cannot train via hard argsort.
    rank_scope=EligibilityScope(subject,'ranking',purpose,10)
    for entity,label in (('Aster','amber'),('Birch','blue')):
        ledger.admit(MemoryRecord('rank_'+entity,subject,'ranking',
                     f'The approved release label for project {entity} is {label}.',1,1,purpose),
                     authorized_subject=subject,authorized_branch='ranking',source_authorized=True)
    previous_flags={n:p.requires_grad for n,p in model.new_named_parameters()}
    for name,p in model.new_named_parameters():
        p.requires_grad_(name.startswith(('reranker.','sense_ports.13.')))
    rank_optimizer=build_optimizer(model,'muon8bit',lr_base=0,lr_new=protocol['ranking']['lr'],
                                   lr_muon_new=protocol['ranking']['lr'],weight_decay=0)
    report['ranking']={'updates':[],'optimizer':rank_optimizer.audit(),
                       'note':'Exposed two-record diagnostic, not held-out retrieval quality'}
    def rank_args(entity):
        ids=encode_prompt(tokenizer,protocol['prompt'].format(entity=entity))
        return dict(prompt_ids=ids,prompt_mask=torch.ones_like(ids,dtype=torch.bool),
                    query='project release label',scope=rank_scope,task=task,train_sense=True)
    def rank_scores():
        result=[]
        for entity in ('Aster','Birch'):
            request=session.prepare(**rank_args(entity),rerank=True)
            result.append({'query_entity':entity,'ordered_records':[s.record_id for s in request.frame.memory.spans]})
            request.release()
        return result
    report['ranking']['before']=rank_scores()
    for update in range(1,protocol['ranking']['updates']+1):
        model.train();rank_optimizer.zero_grad(set_to_none=True);total=0
        for entity in ('Aster','Birch'):
            with session.training_request(**rank_args(entity)) as request:
                loss=session.ranking_loss(request,'rank_'+entity)/2
                total+=float(loss.detach());session.backward(request,loss)
            del request,loss
        if update==1:report['ranking']['gradient_audit']=model.gradient_audit()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True)
        rank_optimizer.step();rank_optimizer.zero_grad(set_to_none=True)
        report['ranking']['updates'].append({'update':update,'loss':total})
        mark('ranking_update',update=update,loss=total)
    report['ranking']['after']=rank_scores()
    del rank_optimizer
    for name,p in model.new_named_parameters():p.requires_grad_(previous_flags[name])
    model.eval();gc.collect();torch.mps.empty_cache()
    # Native dense candidate generation is source-only and eligibility-first.
    dense=session.dense_candidates('Aster',scope(cases[0]))
    require('dense_candidate_provenance',tuple(r.record_id for r in dense.records)==(cases[0]['id'],),
            record_ids=[r.record_id for r in dense.records],producer=dense.dense_producer_id)

    # A record the lexical candidate path excludes cannot reach the model.
    invariance=[]
    for rid,text,owner,observed in (
        ('irrelevant','Zebras graze peacefully.',subject,2),
        ('foreign','The approved release label for project Aster is gold.','other_fixture',2),
        ('future','The approved release label for project Aster is green.',subject,20)):
        ledger.admit(MemoryRecord(rid,owner,cases[0]['id'],text,observed,observed,purpose),
                     authorized_subject=owner,authorized_branch=cases[0]['id'],source_authorized=True)
        result=generate(cases[0],name=rid+'_added');invariance.append(result)
        require(rid+'_excluded_invariance',result['selected_records']==[cases[0]['id']]
                and result['completion_ids']==report['generations']['final_latent'][0]['completion_ids'])
    report['generations']['excluded_records']=invariance
    # Save one nonzero extension, retaining one original source. No base or optimizer copy.
    extension_dir=args.output/'selected_extension'
    receipt=model.save_extension(extension_dir)
    report['extension']=receipt
    before_reload=logits(cases[0])
    replacement=MiniCPM5Adapter(model.base,model.manifest).to('mps').eval()
    replacement.seal_original_identity();replacement.load_extension(extension_dir)
    model=replacement
    session=MemorySession(model,tokenizer,ledger,cache_dir=args.output/'record_latents')
    after_reload=logits(cases[0])
    require('nonzero_extension_reload',torch.equal(before_reload,after_reload),
            max_absolute_difference=float((before_reload-after_reload).abs().max()),
            base_saved=receipt['base_saved'])
    require('persistent_latent_reload',session.cache_stats['latent_hits']>0,cache_stats=dict(session.cache_stats))
    del before_reload,after_reload
    report['generations']['reloaded']=[generate(cases[0])]
    # Source correction invalidates every old frame and persistent slot payload.
    stale=session.prepare(**kwargs(cases[0]))
    old_epoch=ledger.epoch
    correction=MemoryRecord('aster_corrected',subject,cases[0]['id'],cases[1]['source'],
                            5,5,purpose,supersedes=cases[0]['id'])
    session.commit_observation(PendingWrite(correction.record_id,correction,old_epoch),
                              authorized_subject=subject,authorized_branch=cases[0]['id'],source_authorized=True)
    try:session.verify(stale)
    except ValueError:stale_rejected=stale.state=='invalid' and stale.frame is None
    else:stale_rejected=False
    require('correction_rejects_stale_frame',stale_rejected,old_epoch=old_epoch,new_epoch=ledger.epoch)
    ledger.close();ledger=RecordLedger(ledger_path)
    session=MemorySession(model,tokenizer,ledger,cache_dir=args.output/'record_latents')
    corrected_case={**cases[0],'answer':'blue'}
    corrected=generate(corrected_case,name='corrected_after_restart')
    corrected_exact=generate(corrected_case,exact=True,name='corrected_exact_text')
    report['generations']['correction_restart']=[corrected,corrected_exact]
    require('correction_restart_source',corrected['selected_records']==['aster_corrected'])
    # Export sealed source identity is checked again, and no originals were ever updated.
    require('original_parameters_unchanged',original_stamp==model.parameter_stamp(original_only=True))
    payloads=list(args.output.rglob('*.safetensors'))
    require('no_base_checkpoint_copy',all(p.name=='extensions.safetensors' or p.parent.name=='record_latents' for p in payloads),
            payloads=[{'path':str(p.relative_to(args.output)),'bytes':p.stat().st_size} for p in payloads])
    report['cache_stats']=session.cache_stats
    latent=report['generations']['final_latent'];exact=report['generations']['exact_text']
    report['acceptance']={
        'E1_native_interfaces':all(c['passed'] for c in report['checks'].values()),
        'E2_latent_answers_correct':sum(r['correct'] for r in latent),'E2_latent_total':len(latent),
        'E2_exact_answers_correct':sum(r['correct'] for r in exact),'E2_exact_total':len(exact),
        'E2_counterfactual_pairs':sum(latent[i]['correct'] and latent[i+1]['correct']
                                      and latent[i]['completion_ids']!=latent[i+1]['completion_ids'] for i in (0,2)),
        'E2_correction_generated_correct':corrected['correct'],
        'E2_correction_exact_correct':corrected_exact['correct'],
        'claims':'Exposed synthetic integration only; no generalization, cognition or self-improvement evidence'}
    report['acceptance']['E2_diagnostic_passed']=(all(r['correct'] for r in latent+exact)
        and report['acceptance']['E2_counterfactual_pairs']==2 and corrected['correct'] and corrected_exact['correct'])
    ledger.close()
    mark('qualification_complete',acceptance=report['acceptance'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['mps'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-path',type=Path,default=ROOT/'checkpoints/minicpm5_source')
    parser.add_argument('--model-lock',type=Path,default=ROOT/'configs/minicpm5_sealed_lock.local.json')
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/e1_e2_qualification_v1.json')
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):parser.error('Use a fresh output directory')
    if shutil.disk_usage(ROOT).free<5*1024**3:parser.error('At least5GiB disk headroom required')
    from replica_lab.profiling import RunProfiler
    from replica_lab.architecture.loading import exclusive_mps_worker
    args.output.mkdir(parents=True,exist_ok=True)
    report={'status':'started','started_at_utc':datetime.now(timezone.utc).isoformat(),
            'arguments':{k:str(v) for k,v in vars(args).items()},
            'protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
            'protocol':json.loads(args.protocol.read_text()),
            'packages':{n:importlib.metadata.version(n) for n in ('torch','transformers','safetensors')},
            'implementation_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__),*(ROOT/'src/replica_lab/architecture').glob('*.py'),
                          *(ROOT/'src/replica_lab/memory_system').glob('*.py'),
                          ROOT/'src/replica_lab/optimizers.py',ROOT/'src/replica_lab/neural.py']},
            'free_disk_before_bytes':shutil.disk_usage(ROOT).free,'downloads':0,'original_model_loads':1,
            'environment':{k:os.environ[k] for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE',
                                                   'HF_DATASETS_OFFLINE','PYTORCH_ENABLE_MPS_FALLBACK')}}
    profiler=RunProfiler('mps',args.output)
    try:
        with exclusive_mps_worker(ROOT):
            profiler.start()
            execute(args,profiler,report)
            report['status']='completed'
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        report['profile']=profiler.stop()
        report['free_disk_after_bytes']=shutil.disk_usage(ROOT).free
        report['artifact_bytes']=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
        report['ended_at_utc']=datetime.now(timezone.utc).isoformat()
        dump(args.output/('run_report.json' if report['status']=='completed' else 'failure_report.json'),report)
    print(json.dumps({'report':str(args.output/'run_report.json'),'acceptance':report['acceptance']}),flush=True)


if __name__=='__main__':main()
