"""Official BF16 v5 worker; called only after external admission by the launcher."""
from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import time

import torch

from .architecture.spark2_5 import load_spark2_5_config, official_bf16_precision, load_tokenizer_identity, resolve_thinking_mode
from .architecture.spark_adapter import Spark2_5Adapter
from .architecture.intrinsic_memory_layout_v4 import spark_layout
from .contextual_completion_objective_v1 import CompletionObjectiveSpec, stop_policy_identity
from .contextual_memory import canonical_spans, ContextualMemoryError
from .contextual_memory_v5 import ContextualMemoryV5
from .contextual_release_v5 import ProducerReleaseCorpus
from .intrinsic_memory_budget_v5 import atomic, sha_json
from .intrinsic_memory_diagnostic_v5 import DiagnosticEngine, logits_digest, right_padded_prefix, PROMPT
from .intrinsic_memory_execution_v5 import bound_json, hash_file
from .spark_device_placement_v2 import place_frozen_spark_base, _inventory, _content_digest
from .tiered_memory import TieredMemoryStore, MemoryScope, TierCeilings


class NativeBackend:
    def __init__(self, tokenizer, template, mode):
        self.tokenizer,self.template,self.mode=tokenizer,template,mode
    def encode(self,content):
        text=self.tokenizer.apply_chat_template([{'role':'user','content':content}],
             chat_template=self.template,tokenize=False,add_generation_prompt=True,**self.mode)
        return self.tokenizer(text,add_special_tokens=False)['input_ids']
    def answer_ids(self,text):return self.tokenizer(text,add_special_tokens=False)['input_ids']
    def decode(self,ids):return self.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)


def base_hash(base):
    named,rows,_=_inventory(base)
    return _content_digest([v for v in named if v[0].startswith('parameter/')],
                           [v for v in rows if v[0].startswith('parameter/')])[0]


def verified_source_classes(directory, raw, source_execution):
    from transformers.dynamic_module_utils import get_cached_module_file, get_class_in_module, HF_MODULES_CACHE
    # Import the verified source classes without instantiating weights. Check
    # HF's actual cached executable copies as well as the original source files.
    classes={};cached={}
    for key in ('AutoConfig','AutoModelForCausalLM'):
        file=raw['auto_map'][key].split('.')[0]+'.py'
        cached[key]=get_cached_module_file(str(directory),file,local_files_only=True)
    # Cache population does not execute these source classes. Verify every local
    # executable dependency before importing either class, then check its origin.
    for cache_root in { (Path(HF_MODULES_CACHE)/v).parent for v in cached.values() }:
        for relative,expected in source_execution['files'].items():
            if hash_file(cache_root/relative)!=expected:raise ValueError('substituted cached source dependency')
    for key in ('AutoConfig','AutoModelForCausalLM'):
        cls=get_class_in_module(raw['auto_map'][key].split('.')[-1],cached[key],force_reload=True)
        if hash_file(Path(inspect.getfile(cls)))!=hash_file(directory/(raw['auto_map'][key].split('.')[0]+'.py')):
            raise ValueError('cached class origin differs')
        classes[key]=cls
    return classes


def load_native(root, protocol, arm, ledger, stage):
    from transformers import AutoTokenizer
    if not torch.backends.mps.is_available():raise ValueError('native MPS unavailable; no fallback')
    source=bound_json(root,protocol['source']);directory=root/source['source_root']
    raw=json.loads((directory/'config.json').read_text());config=load_spark2_5_config(raw)
    template=(directory/'chat_template.jinja').read_text()
    tokenizer_identity=load_tokenizer_identity(config,json.loads((directory/'tokenizer_config.json').read_text()),
        template,tokenizer_json=(directory/'tokenizer.json').read_text())
    classes=verified_source_classes(directory,raw,bound_json(root,protocol['source_execution']))
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=True)
    if not tokenizer.is_fast or (tokenizer.eos_token_id,tokenizer.pad_token_id)!=(1,2):
        raise ValueError('bound fast tokenizer/EOS/PAD differs')
    gen=protocol['generation']
    spec=CompletionObjectiveSpec(config.geometry.vocab_size,gen['context_tokens'],1,2,(1,),
        stop_policy_identity((1,)),hash_file(directory/'tokenizer.json'),
        sha_json({'chat_template_sha256':hash_file(directory/'chat_template.jinja'),'user_prompt':PROMPT,
                  'thinking_mode':resolve_thinking_mode(tokenizer_identity,'none'),'add_generation_prompt':True}),'none')
    with ledger.work(stage,'official_bf16_model_load',model_loads=1) as event:
        base=classes['AutoModelForCausalLM'].from_pretrained(directory,
            config=classes['AutoConfig'].from_pretrained(directory,local_files_only=True),
            local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager',use_safetensors=True)
        if any(p.dtype!=torch.bfloat16 for p in base.parameters()):raise ValueError('original BF16 parameter dtype differs')
        if sum(p.numel() for p in base.parameters())!=4112079360:raise ValueError('official parameter inventory differs')
        placement=place_frozen_spark_base(base,device='mps');base.eval()
        precision=official_bf16_precision(config,artifact_sha256=sha_json(source['files']))
        adapter=Spark2_5Adapter(base,config,precision,device='mps',require_official_geometry=True,
            projector_in_dim=config.geometry.hidden_size,zero_init_output=True,memory_size=256,heads=4,
            spine_sites=(3,7),spine_slots=8)
        adapter.eval();adapter.seal_original_identity()
        path=ContextualMemoryV5(adapter,spark_layout(arm),source_identity=sha_json(source['files']),
            objective_identity=spec.identity(),execution_identity=sha_json({'protocol':protocol,'arm':arm}),
            lr=protocol['training']['lr'],initialization_seed=17)
        adapter.extension.spine=None
        event.update(placement=placement,precision=adapter.base_snapshot(),
                     frozen_parameter_sha256=base_hash(base),original_dtype='bfloat16',extension_dtype='float32')
    backend=NativeBackend(tokenizer,template,resolve_thinking_mode(tokenizer_identity,'none'))
    return path,backend,spec,event['frozen_parameter_sha256']


def store_at(output, world):
    # Bounded real E storage; P stays at the existing P1 profile and is never
    # allocated/updated by this reader experiment. Logical tiers are not VRAM.
    return TieredMemoryStore(output/'stores'/world,profile_id='P1',ceilings=TierCeilings(
        hot_bytes=32*1024**2,warm_bytes=32*1024**2,cold_bytes=32*1024**2,
        identity_reserved_bytes=64*1024,replay_reserved_bytes=256*1024,
        workspace_quota_bytes=1024**2,artifact_budget_bytes=4*1024**2,
        minimum_volume_free_bytes=25*1024**3),enforce_volume_free=True)


def world_scope(row):return MemoryScope(row['world_id'],'main','intrinsic_readout',1000)


def admit_rows(engine, rows, output):
    batches,scopes,stores=[],[],[]
    for row in rows:
        store=store_at(output,row['world_id']);scope=world_scope(row)
        span=canonical_spans(row['text'],record_id=row['world_id'],source_id=row['world_id'],source_epoch=1,available_at=1)
        if len(span)!=1:raise ValueError('one bounded canonical span required by this fixed experiment')
        with engine.measured('canonical_admission',source_encodes=1,writes=1):
            result=engine.path.admit(store,span[0],scope=scope)
            if not result.outcome.ok:raise ValueError('canonical admission failed')
        batch=engine.read(store,scope,row['question']);batches.append(batch);scopes.append(scope);stores.append(store)
    return batches,scopes,stores


def write_receipt(output,name,payload):
    atomic(output/(name+'.json'),payload)


def close_stage(engine, name, outcome):
    engine.ledger.leave(name,status=outcome)


def begin_stage(engine,name):
    engine.ledger.enter(name);engine.stage=name


def run_main(root,protocol,arm,ledger,output,path,backend,spec,initial_hash):
    engine=DiagnosticEngine(path,backend,spec,ledger,max_new_tokens=protocol['generation']['max_new_tokens'],output=output)
    result={'format':'replica.intrinsic_memory.native_result.v5','source_kind':'official_bf16','arm':arm,
            'competence_passed':False,'fit_memory_dependence_passed':False,'useful_recall_passed':False,
            'persistent_P_updates':0,'promotion':False,'restart_requested':False}
    stores=[]
    try:
        begin_stage(engine,'baseline')
        calibration=bound_json(root,protocol['data']['calibration'])['rows'];answers=[]
        if len(calibration)!=8:raise ValueError('incomplete calibration')
        for row in calibration:
            answer=engine.generate(row,None,world_scope(row),control='exact_text');answer.pop('first_logits')
            answers.append({'world_id':row['world_id'],**answer})
            write_receipt(output,'baseline',{'rows':answers,'complete':len(answers)==8})
        passed=len(answers)==8 and all(r['correct'] and not r['false_recall'] and not r['truncated'] for r in answers)
        result['competence_passed']=passed
        close_stage(engine,'baseline','passed' if passed else 'failed_competence')
        if not passed:
            result['status']='baseline_failed_dependent_stages_closed';return result
        # Fitting labels are parsed only after the frozen pre-fit baseline gate.
        rows=bound_json(root,protocol['data']['fit'])['rows']
        if len(rows)!=8 or any(rows[i]['question']!=rows[i^1]['question'] or rows[i]['answer']==rows[i^1]['answer'] for i in range(8)):
            raise ValueError('counterfactual fitting corpus differs')
        begin_stage(engine,'numerical')
        ids,mask=right_padded_prefix(engine.tokens(rows[0]),spec,engine.device)
        spans=canonical_spans(rows[0]['text'],record_id='numerical',source_id='calibration',source_epoch=1,available_at=1)
        with engine.measured('initial_source_encoding',source_encodes=1):b=path.training_values(spans,scope=world_scope(rows[0]))
        with torch.no_grad():
            disabled=engine.forward(ids,mask,None,world_scope(rows[0]))
            enabled=engine.forward(ids,mask,b,world_scope(rows[0]))
        if not torch.equal(disabled,enabled):raise ValueError('zero-init disabled-extension parity failed')
        del disabled,enabled,b
        write_receipt(output,'numerical',{'disabled_zero_init_exact':True,'single_optimizer_owner':path.assert_optimizer_ownership()})
        close_stage(engine,'numerical','passed')
        begin_stage(engine,'prepare')
        sources,scopes,stores=admit_rows(engine,rows,output)
        close_stage(engine,'prepare','completed')
        begin_stage(engine,'fit');updates=[]
        for index in range(32):
            i=index%8;updates.append(engine.update(rows[i],sources[i],scopes[i]))
            write_receipt(output,'fit',{'updates':updates,'planned_updates':32})
            if index==1 and (updates[-1]['gradients']['producer']['norm']<=0
                    or any(v['output_norm']<=0 for v in updates[-1]['gradients']['sites'].values())):
                raise ValueError('producer/early reader gradient path failed after zero-init update')
        close_stage(engine,'fit','completed');result['fit_updates']=32
        begin_stage(engine,'fit_probes')
        probes=engine.fitting_probes(rows,sources,scopes,stores);write_receipt(output,'fit_probes',probes)
        result['fit_memory_dependence_passed']=probes['fit_memory_dependence_passed']
        close_stage(engine,'fit_probes','passed' if result['fit_memory_dependence_passed'] else 'failed_memory_dependence')
        engine.freeze()
        # Freeze is recorded before any development parsing. No adaptive retry.
        write_receipt(output,'fitting_closed',{'closed':True,'updates':32,'objective':spec.identity(),
                       'parameter_sha256':{k:logits_digest(v) for k,v in path._tensors().items()}})
        begin_stage(engine,'checkpoint')
        with torch.no_grad():
            live=engine.live(sources[0],scopes[0]);enc=engine.encoding(rows[0])
            canary=engine.forward(enc.input_ids,enc.attention_mask,live,scopes[0])
        expected=logits_digest(canary)
        cap=protocol['arms'][arm]['budget']['checkpoint']['artifact_bytes']
        with engine.measured('extension_optimizer_checkpoint',artifact_bytes=cap,writes=1) as event:
            manifest=path.save(output/'checkpoint')
            actual=sum(p.stat().st_size for p in (output/'checkpoint').rglob('*') if p.is_file())
            event['actual']={'artifact_bytes':actual};event['contains_original_weights']=False
        write_receipt(output,'restart_request',{'canary_logits_sha256':expected,'world_id':rows[0]['world_id'],
            'parent_model_pid':os.getpid(),
            'objective':spec.identity(),'checkpoint_manifest_sha256':hash_file(output/'checkpoint/manifest.json'),
            'original_parameters_sha256':initial_hash,'updates_allowed':1,'uninterrupted_equivalence_claim':False})
        result['restart_requested']=True;close_stage(engine,'checkpoint','completed')
        if result['fit_memory_dependence_passed']:
            begin_stage(engine,'development')
            development=bound_json(root,protocol['data']['development'])['rows']
            batches,dscopes,dstores=admit_rows(engine,development,output);stores.extend(dstores)
            evaluation=engine.evaluate(development,batches,dscopes);write_receipt(output,'development',evaluation)
            result['useful_recall_passed']=evaluation['useful_recall_passed']
            close_stage(engine,'development','passed' if result['useful_recall_passed'] else 'failed_utility')
            begin_stage(engine,'correction')
            row=development[0];store=dstores[0];scope=dscopes[0];old=batches[0]
            replacement=development[1]['answer'];newtext=row['text'].replace(row['answer'],replacement)
            span=canonical_spans(newtext,record_id=row['world_id']+'.corrected',source_id=row['world_id'],source_epoch=2,available_at=2)[0]
            span=replace(span,supersedes=old.spans[0].record_id)
            with engine.measured('chronological_correction',source_encodes=1,writes=1):path.correct(store,old.spans[0].record_id,span,scope=scope)
            try:path.validate(old,scope)
            except ContextualMemoryError:stale_rejected=True
            else:raise ValueError('correction retained stale snapshot')
            new=engine.read(store,scope,row['question']);changed={**row,'text':newtext,'answer':replacement}
            corrected=engine.generate(changed,new,scope);corrected.pop('first_logits')
            exact=engine.generate(changed,None,scope,control='exact_text');exact.pop('first_logits')
            with engine.measured('revocation',writes=1):store.revoke_record(span.record_id,reason='predeclared_probe',scope=scope)
            try:engine.read(store,scope,row['question'])
            except ContextualMemoryError:revoked_read_rejected=True
            else:raise ValueError('revoked source remained readable')
            revoked=engine.generate({**row,'text':'','answer':'unknown'},None,scope,control='absent');revoked.pop('first_logits')
            write_receipt(output,'correction',{'stale_snapshot_rejected':stale_rejected,'revoked_retrieval_rejected':revoked_read_rejected,
                 'corrected_contextual':corrected,'corrected_exact':exact,'after_revocation':revoked,'learned_unlearning_claim':False})
            close_stage(engine,'correction','completed')
        result['status']='completed_frozen_fit_diagnostic'
        return result
    finally:
        result['path_costs']=dict(path.cost)
        result['final_original_parameters_sha256']=base_hash(path.adapter.base)
        result['original_weights_unchanged']=result['final_original_parameters_sha256']==initial_hash
        write_receipt(output,'main_result',result)
        for store in stores:store.close()
        path.close()
        if not result['original_weights_unchanged']:raise ValueError('original parameter bytes changed')


def run_restart(root,protocol,arm,ledger,output,path,backend,spec,initial_hash):
    request=json.loads((output/'restart_request.json').read_text())
    if request.get('parent_model_pid')==os.getpid() or type(request.get('parent_model_pid')) is not int:
        raise ValueError('restart must use a different model process')
    if request['objective']!=spec.identity() or hash_file(output/'checkpoint/manifest.json')!=request['checkpoint_manifest_sha256']:
        raise ValueError('restart objective/checkpoint differs')
    if initial_hash!=request['original_parameters_sha256']:raise ValueError('restart original base differs')
    with ledger.work('restart','extension_optimizer_restore') as event:
        event['artifact_file_bytes']=sum(p.stat().st_size for p in (output/'checkpoint').rglob('*') if p.is_file())
        path.load(output/'checkpoint');path.assert_optimizer_ownership()
        event['checksum_and_deserialization_io_included_in_wall']=True
    engine=DiagnosticEngine(path,backend,spec,ledger,max_new_tokens=8,output=output);engine.stage='restart'
    row=bound_json(root,protocol['data']['fit'])['rows'][0];scope=world_scope(row);store=store_at(output,row['world_id'])
    try:
        path.release_corpus=ProducerReleaseCorpus(store,scope)
        source=engine.read(store,scope,row['question']);enc=engine.encoding(row)
        with torch.no_grad():canary=engine.forward(enc.input_ids,enc.attention_mask,engine.live(source,scope),scope)
        equal=logits_digest(canary)==request['canary_logits_sha256']
        if not equal:raise ValueError('fresh-process logits differ')
        path.resume_training();update=engine.update(row,source,scope)
        unchanged=base_hash(path.adapter.base)==initial_hash
        if not unchanged:raise ValueError('restart modified original weights')
        result={'fresh_process_logits_exact':equal,'actual_continuation_updates':1,'update':update,
            'parent_model_pid':request['parent_model_pid'],'continuation_model_pid':os.getpid(),
            'original_weights_unchanged':unchanged,'optimizer_ownership':path.assert_optimizer_ownership(),
            'uninterrupted_update_equivalence_measured':False,'path_costs':path.cost,
            'corpus_costs':path.release_corpus.cost,'persistent_P_updates':0}
        write_receipt(output,'restart_result',result);ledger.leave('restart',status='completed')
        return result
    finally:store.close();path.close()
