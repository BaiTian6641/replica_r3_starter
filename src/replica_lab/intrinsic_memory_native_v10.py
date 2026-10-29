"""Native v10: frozen full-context teacher targets and 128 planned memory updates."""
from dataclasses import replace
import json
import os

import torch

from .contextual_memory import canonical_spans, ContextualMemoryError
from .intrinsic_memory_diagnostic_v5 import logits_digest, right_padded_prefix
from .intrinsic_memory_teacher_v10 import TeacherEngine
from .contextual_teacher_objective_v1 import TeacherObjective
from .contextual_release_v5 import ProducerReleaseCorpus
from .intrinsic_memory_execution_v5 import bound_json, hash_file
from .intrinsic_memory_native_v5 import (load_native as load_previous, base_hash, world_scope,
    admit_rows, write_receipt, close_stage, begin_stage, store_at)


def load_native(root, protocol, arm, ledger, stage):
    path, backend, spec, initial_hash = load_previous(root, protocol, arm, ledger, stage)
    path.objective_identity = TeacherObjective().identity(spec)
    return path, backend, spec, initial_hash


def run_main(root,protocol,arm,ledger,output,path,backend,spec,initial_hash):
    engine=TeacherEngine(path,backend,spec,ledger,max_new_tokens=protocol['generation']['max_new_tokens'],output=output,
                         base_identity=initial_hash,fit_identity=protocol['data']['fit']['sha256'])
    result={'format':'replica.intrinsic_memory.native_result.v10','source_kind':'official_bf16','arm':arm,
            'competence_passed':False,'teacher_competence_passed':False,'fit_memory_dependence_passed':False,'useful_recall_passed':False,
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
        begin_stage(engine,'teacher')
        result['teacher_competence_passed']=engine.prepare_teacher(rows,scopes)
        if not result['teacher_competence_passed']:
            close_stage(engine,'teacher','failed_teacher_competence')
            result['status']='teacher_failed_dependent_stages_closed'
            return result
        teacher_cache=engine.save_teacher_cache(output/'teacher_targets.json')
        result['teacher_cache']=teacher_cache
        result['teacher_parameter_updates']=0
        result['direct_CE_reference']=protocol['diagnostic_predecessor']
        close_stage(engine,'teacher','passed')
        updates=[];curve=[]
        for end in protocol['fit_probe_prefixes']:
            fit_stage=f'fit_{end:03d}';probe_stage=f'fit_probes_{end:03d}'
            begin_stage(engine,fit_stage)
            for index in range(end-32,end):
                i=index%8;updates.append(engine.update(rows[i],sources[i],scopes[i]))
                write_receipt(output,'fit',{'updates':updates,'planned_updates':128})
                if index==1 and (updates[-1]['gradients']['producer']['norm']<=0
                        or any(v['output_norm']<=0 for v in updates[-1]['gradients']['sites'].values())):
                    raise ValueError('producer/early reader gradient path failed after zero-init update')
            close_stage(engine,fit_stage,'completed');result['fit_updates']=end
            begin_stage(engine,probe_stage)
            probes=engine.fitting_probes(rows,sources,scopes,stores)
            write_receipt(output,probe_stage,probes)
            curve.append({'updates':end,'correct':probes['correct'],
                          'memory_dependence_passed':probes['fit_memory_dependence_passed'],
                          'counterfactual_pairs_both_correct':probes['counterfactual_pairs_both_correct']})
            write_receipt(output,'fit_curve',{'prefixes':curve,'selection':'fixed_final_128_only',
                          'early_stopping':False,'development_opened':False})
            result['fit_memory_dependence_passed']=probes['fit_memory_dependence_passed']
            close_stage(engine,probe_stage,'passed' if result['fit_memory_dependence_passed'] else 'failed_memory_dependence')
        result['fit_curve']=curve
        engine.freeze()
        # Freeze is recorded before any development parsing. No adaptive retry.
        write_receipt(output,'fitting_closed',{'closed':True,'updates':128,'objective':path.objective_identity,
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
            'objective':path.objective_identity,'checkpoint_manifest_sha256':hash_file(output/'checkpoint/manifest.json'),
            'original_parameters_sha256':initial_hash,'teacher_cache':teacher_cache,'updates_allowed':1,'uninterrupted_equivalence_claim':False})
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
    if request['objective']!=path.objective_identity or hash_file(output/'checkpoint/manifest.json')!=request['checkpoint_manifest_sha256']:
        raise ValueError('restart objective/checkpoint differs')
    if initial_hash!=request['original_parameters_sha256']:raise ValueError('restart original base differs')
    with ledger.work('restart','extension_optimizer_restore') as event:
        event['artifact_file_bytes']=sum(p.stat().st_size for p in (output/'checkpoint').rglob('*') if p.is_file())
        path.load(output/'checkpoint');path.assert_optimizer_ownership()
        event['checksum_and_deserialization_io_included_in_wall']=True
    engine=TeacherEngine(path,backend,spec,ledger,max_new_tokens=8,output=output,
                         base_identity=initial_hash,fit_identity=protocol['data']['fit']['sha256']);engine.stage='restart'
    fit_rows=bound_json(root,protocol['data']['fit'])['rows']
    engine.load_teacher_cache(output/'teacher_targets.json',request['teacher_cache'],fit_rows)
    row=fit_rows[0];scope=world_scope(row);store=store_at(output,row['world_id'])
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
            'corpus_costs':path.release_corpus.cost,'persistent_P_updates':0,
            'teacher_cache_restored':True,'teacher_parameter_updates':0,'objective':path.objective_identity}
        write_receipt(output,'restart_result',result);ledger.leave('restart',status='completed')
        return result
    finally:store.close();path.close()
