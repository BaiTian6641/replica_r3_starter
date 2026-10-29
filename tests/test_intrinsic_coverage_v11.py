import copy
import hashlib
import json
from pathlib import Path

import pytest

from replica_lab.intrinsic_memory_coverage_v11 import (
    FORMAT, PACKET, AUTHORIZATION, TRAINING, TEACHER, COVERAGE, BUDGET_DERIVATION,
    PREFIXES, plan, totals, validate_protocol,
)
from replica_lab.intrinsic_memory_worlds_v11 import (
    worlds, validate_corpus, public_split_metadata, oracle, FAMILIES,
)
from replica_lab.intrinsic_memory_execution_v11 import (
    validate_bindings, validate_authorization, validate_predecessor, validate_fresh_data,
)

ROOT = Path(__file__).resolve().parents[1]


def candidate(previous):
    p = copy.deepcopy(previous)
    p.update(format=FORMAT, training=copy.deepcopy(TRAINING), teacher=copy.deepcopy(TEACHER),
             coverage=copy.deepcopy(COVERAGE), budget_derivation=copy.deepcopy(BUDGET_DERIVATION),
             fit_probe_prefixes=list(PREFIXES),
             fit_curve_selection='all_512_updates_fixed_final_prefix_all_groups_only_opens_development')
    p['density_controls']['initial_study'] = 'feasibility_two_predeclared_rules_one_seed'
    for arm, spec in p['arms'].items():
        spec.update(output_root='output_v11/'+arm, budget=plan(arm), total_reservations=totals(plan(arm)))
    return p


def test_complete_balanced_source_worlds_and_split_group_separation():
    corpus = [validate_corpus(worlds(split), split) for split in ('calibration','fit','development')]
    assert [len(doc['rows']) for doc in corpus] == [8,32,16]
    for key in ('world_id','family'):
        identities = [{r[key] for r in doc['rows']} for doc in corpus]
        assert all(not identities[i] & identities[j] for i in range(3) for j in range(i))
    assert all(len(r['text'].encode()) <= 256 for d in corpus for r in d['rows'])
    for split in ('calibration','fit','development'):
        metadata = public_split_metadata(split)
        assert 'answer' not in json.dumps(metadata) and 'text' not in metadata
    facts = [{'entity':'A','color':'red','revision':1},{'entity':'A','color':'blue','revision':2}]
    assert oracle(facts,'A') == 'blue' and oracle(facts,'B') == 'unknown'
    with pytest.raises(ValueError):oracle(facts+[facts[1]],'A')


@pytest.mark.parametrize('damage',['answer','source','order','group','split','pair','incomplete'])
def test_corpus_rejects_inconsistent_or_reassigned_evidence(damage):
    doc = worlds('fit')
    if damage == 'answer':doc['rows'][0]['answer']='gold'
    elif damage == 'source':doc['rows'][0]['text']+=' Wrong.'
    elif damage == 'order':doc['rows'][16]['facts'].reverse()
    elif damage == 'group':doc['rows'][0]['criterion_group']=3
    elif damage == 'split':doc['split']='development'
    elif damage == 'pair':doc['rows'][1]['question']='A different query?'
    else:doc['rows'].pop()
    with pytest.raises(ValueError):validate_corpus(doc,'fit')


def test_stage_budget_accounts_all_larger_teacher_fit_controls_and_restart():
    previous = json.loads((ROOT/'configs/spark_intrinsic_alignment_v10.json').read_text())
    p = candidate(previous);validate_protocol(p)
    assert p['fit_gate']==previous['fit_gate'] and p['utility']==previous['utility']
    for arm,spec in p['arms'].items():
        t=spec['total_reservations']
        assert t['forwards']==8829 and t['generated_tokens']==7256 and t['input_tokens']==2260224
        assert t['updates']==t['backwards']==513 and t['model_loads']==2
        assert t['source_encodes']==949 and t['retrievals']==179 and t['writes']==392
        assert t['wall_seconds']==7080 and spec['wall_seconds']==7200
        assert t['artifact_bytes']=={'two':256,'full_attention_nine':512,'all':1152}[arm]*1024**2
        assert p['training']['updates']//p['coverage']['fit_cases']==16


@pytest.mark.parametrize('damage',['group_escape','teacher','criterion','schedule','forward','wall','artifact','derivation'])
def test_no_implicit_budget_or_group_criterion_changes(damage):
    p=candidate(json.loads((ROOT/'configs/spark_intrinsic_alignment_v10.json').read_text()))
    if damage=='group_escape':p['coverage']['development_groups']=1
    elif damage=='teacher':p['teacher']['qualification']['correct_min']=31
    elif damage=='criterion':p['utility']['better_than'].remove('shuffled')
    elif damage=='schedule':p['training']['updates']=1024
    elif damage=='forward':p['arms']['two']['budget']['teacher']['forwards']=32
    elif damage=='wall':p['arms']['two']['wall_seconds']=7300
    elif damage=='artifact':p['arms']['two']['budget']['admission']['artifact_bytes']=0
    else:p['budget_derivation']['total_forwards']='old v10 counts'
    with pytest.raises(ValueError):validate_protocol(p)


def synthetic(tmp_path):
    from test_intrinsic_alignment_v10 import synthetic as prior_fixture
    from test_intrinsic_execution_v5 import write
    packet,previous,_=prior_fixture(tmp_path)
    previous_pin=write(tmp_path,'teacher_reference.json',previous)
    output=previous['arms']['two']['output_root']
    result_pin=write(tmp_path,output+'/result.json',{
        'protocol_sha256':previous_pin['sha256'],'source_kind':'official_bf16','arm':'two',
        'native_competence':True,'useful_recall':False,
        'main':{'teacher_competence_passed':True,'fit_memory_dependence_passed':True,'fit_updates':128,'original_weights_unchanged':True},
        'restart':{'fresh_process_logits_exact':True,'actual_continuation_updates':1}})
    exposure={'fresh_for_successor':False,'opened_split':previous['data']['development'],'source_attempt':result_pin,
              'first_measured_opening_in_this_lineage':write(tmp_path,output+'/development.json',{'closed':False}),
              'fitting_frozen_before_open':write(tmp_path,output+'/fitting_closed.json',{'closed':True}),
              'fixed_final_fit_gate':write(tmp_path,output+'/fit_probes_128.json',{'passed':True})}
    p=candidate(previous)
    p.update(diagnostic_predecessor=result_pin,experiment_variant={'predecessor':previous_pin},
             prior_development_exposure=write(tmp_path,'exposure.json',exposure))
    p['data']={s:write(tmp_path,'new_data/'+s+'.json',worlds(s)) for s in ('calibration','fit','development')}
    m={'format':'replica.intrinsic_memory.coverage_split_manifest.v11',
       'generator_policy':'predeclared_source_order_revision_oracle_no_generated_explanations',
       'splits':{s:{'file':pin,'public_identity':public_split_metadata(s)} for s,pin in p['data'].items()}}
    p['data_manifest']=write(tmp_path,'split_manifest.json',m)
    pp=write(tmp_path,'protocol.json',p)
    write(tmp_path,'packet.json',{'format':PACKET,'experiment_version':11,'native_engine_version':11,
                                'protocol':pp,'arm':'two','attempts':1,'retries':0,'output_root':'output_v11/two'})
    return packet,p,previous


def test_admission_checks_fresh_identity_without_parsing_new_labels(tmp_path,monkeypatch):
    import replica_lab.intrinsic_memory_execution_v11 as execution
    packet,p,_=synthetic(tmp_path)
    original=execution.bound_json; opened=[]
    forbidden={x['path'] for x in p['data'].values()}
    def checked(root,pin):
        assert pin['path'] not in forbidden
        opened.append(pin['path']);return original(root,pin)
    monkeypatch.setattr(execution,'bound_json',checked)
    _,_,receipt=validate_bindings(tmp_path,packet)
    assert receipt['fresh_data']['development_labels_parsed'] is False
    assert p['data_manifest']['path'] in opened


@pytest.mark.parametrize('damage',['old_split_renamed','old_split_same_path','roots','manifest_missing','teacher_failure','restart_missing','placement','missing_exposure'])
def test_fresh_split_and_completed_reference_are_required(tmp_path,damage):
    from test_intrinsic_execution_v5 import write
    packet,p,previous=synthetic(tmp_path);validate_bindings(tmp_path,packet)
    if damage in ('old_split_renamed','old_split_same_path'):
        old=previous['data']['development']
        if damage=='old_split_renamed':
            (tmp_path/'renamed.json').write_bytes((tmp_path/old['path']).read_bytes())
            from replica_lab.intrinsic_memory_execution_v5 import binding
            old=binding(tmp_path,'renamed.json')
        p['data']['development']=old
        m=json.loads((tmp_path/p['data_manifest']['path']).read_text());m['splits']['development']['file']=old
        p['data_manifest']=write(tmp_path,p['data_manifest']['path'],m)
    elif damage=='roots':
        m=json.loads((tmp_path/p['data_manifest']['path']).read_text());m['splits']['development']['public_identity']['decision_roots']=[]
        p['data_manifest']=write(tmp_path,p['data_manifest']['path'],m)
    elif damage=='manifest_missing':(tmp_path/p['data_manifest']['path']).unlink()
    elif damage in ('teacher_failure','restart_missing'):
        r=json.loads((tmp_path/p['diagnostic_predecessor']['path']).read_text())
        if damage=='teacher_failure':r['main']['teacher_competence_passed']=False
        else:r['restart']['actual_continuation_updates']=0
        p['diagnostic_predecessor']=write(tmp_path,p['diagnostic_predecessor']['path'],r)
    elif damage=='placement':p['placement']='pre_MLP'
    else:(tmp_path/p['prior_development_exposure']['path']).unlink()
    with pytest.raises((ValueError,FileNotFoundError)):
        validate_predecessor(tmp_path,p);validate_fresh_data(tmp_path,p)


@pytest.mark.parametrize('file',['configuration_spark.py','modeling_spark.py'])
@pytest.mark.parametrize('damage',['changed','missing','substituted'])
def test_executable_source_is_bound_before_model_loading(tmp_path,file,damage):
    packet,p,_=synthetic(tmp_path);validate_bindings(tmp_path,packet)
    f=tmp_path/'source'/file
    if damage=='changed':f.write_text(f.read_text()+'\nchanged=True\n')
    elif damage=='missing':f.unlink()
    else:
        substitute=tmp_path/'substitute.py';substitute.write_bytes(f.read_bytes());f.unlink();f.symlink_to(substitute)
    loaded=[]
    with pytest.raises(ValueError):validate_bindings(tmp_path,packet);loaded.append(True)
    assert loaded==[]


def test_single_attempt_and_density_scientific_gates_remain_closed(tmp_path):
    from test_intrinsic_execution_v5 import write
    from test_intrinsic_alignment_v10 import authorization as old_authorization
    packet,p,_=synthetic(tmp_path);auth=old_authorization(tmp_path,packet,p)
    receipt=json.loads(auth.read_text());receipt['format']=AUTHORIZATION;write(tmp_path,'auth.json',receipt)
    validate_authorization(tmp_path,packet,auth)
    (tmp_path/'output_v11/two').mkdir(parents=True)
    with pytest.raises(ValueError,match='consumed'):validate_authorization(tmp_path,packet,auth)
    obj=json.loads(packet.read_text());obj.update(arm='all',output_root='output_v11/all');write(tmp_path,'packet.json',obj)
    auth=old_authorization(tmp_path,packet,p);receipt=json.loads(auth.read_text());receipt['format']=AUTHORIZATION;write(tmp_path,'auth.json',receipt)
    with pytest.raises(ValueError,match='useful two-site'):validate_authorization(tmp_path,packet,auth)
