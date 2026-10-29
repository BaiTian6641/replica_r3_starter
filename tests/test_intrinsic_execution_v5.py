import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from replica_lab.intrinsic_memory_execution_v5 import (binding,local_closure,validate_bindings,
    validate_protocol,validate_authorization,hash_file,FORMAT,PACKET,AUTHORIZATION)
from replica_lab.intrinsic_memory_budget_v5 import WorkLedger,sha_json,COUNTERS
from replica_lab.execution_manifest import build_execution_manifest
from test_intrinsic_diagnostic_v5 import FixtureBackend,spec,make,row

ROOT=Path(__file__).resolve().parents[1]

def write(root,name,data):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,sort_keys=True,indent=2)+'\n');return binding(root,name)

def synthetic(tmp_path):
    protocol=json.loads((ROOT/'configs/spark_intrinsic_diagnostic_v5.json').read_text())
    source=tmp_path/'source';source.mkdir()
    (source/'configuration_spark.py').write_text('class Config: pass\n')
    (source/'modeling_spark.py').write_text('from .configuration_spark import Config\nclass Model: pass\n')
    write(tmp_path,'source/config.json',{'auto_map':{'AutoConfig':'configuration_spark.Config','AutoModelForCausalLM':'modeling_spark.Model'}})
    files=[{'path':p.relative_to(tmp_path).as_posix(),'sha256':hash_file(p),'bytes':p.stat().st_size} for p in sorted(source.iterdir())]
    protocol['source']=write(tmp_path,'source_receipt.json',{'source_identity':'official_bf16',
        'repo_id':protocol['source_repo'],'revision':protocol['source_revision'],'derivative':False,
        'source_root':'source','files':files})
    protocol['source_execution']=write(tmp_path,'source_execution.json',build_execution_manifest(source,artifact_sha256=sha_json(files)))
    (tmp_path/'entry.py').write_text('pass\n')
    protocol['implementation']=write(tmp_path,'implementation.json',{'entries':['entry.py'],'files':local_closure(tmp_path,['entry.py'])})
    protocol['environment']=write(tmp_path,'environment.json',{'fixture_only':True})
    protocol['data']={key:write(tmp_path,key+'.json',{'rows':[]}) for key in ('fit','development','calibration')}
    pin=write(tmp_path,'protocol.json',protocol)
    packet={'format':PACKET,'protocol':pin,'arm':'two','attempts':1,'retries':0,
            'output_root':protocol['arms']['two']['output_root']}
    write(tmp_path,'packet.json',packet)
    return tmp_path/'packet.json'

@pytest.mark.parametrize('file',['modeling_spark.py','configuration_spark.py'])
@pytest.mark.parametrize('failure',['changed','missing','substituted'])
def test_actual_model_and_config_code_rejected_before_loading(tmp_path,file,failure):
    packet=synthetic(tmp_path);validate_bindings(tmp_path,packet)
    p=tmp_path/'source'/file
    if failure=='changed':p.write_text(p.read_text()+'\nchanged=True\n')
    elif failure=='missing':p.unlink()
    else:
        other=tmp_path/'substitute.py';other.write_bytes(p.read_bytes());p.unlink();p.symlink_to(other)
    loaded=[]
    with pytest.raises(ValueError):
        validate_bindings(tmp_path,packet)
        loaded.append('model loader would execute here')
    assert not loaded

def test_added_local_source_dependency_and_stale_local_implementation_refused(tmp_path):
    packet=synthetic(tmp_path)
    (tmp_path/'source/extra.py').write_text('pass\n')
    with pytest.raises(ValueError):validate_bindings(tmp_path,packet)
    (tmp_path/'source/extra.py').unlink();(tmp_path/'entry.py').write_text('altered=True\n')
    with pytest.raises(ValueError):validate_bindings(tmp_path,packet)

def test_exact_authorization_cannot_be_reused_for_source_arm_or_output(tmp_path):
    packet=synthetic(tmp_path);p,protocol,_=validate_bindings(tmp_path,packet)
    receipt={'format':AUTHORIZATION,'decision':'authorized','packet_sha256':hash_file(packet),
        'protocol_sha256':p['protocol']['sha256'],'arm':'two','source_kind':'official_bf16',
        'output_root':p['output_root'],'attempts':1,'retries':0,
        'budget':protocol['arms']['two']['total_reservations'],
        'implementation_sha256':protocol['implementation']['sha256'],
        'environment_sha256':protocol['environment']['sha256'],
        'human_approval_reference':'synthetic_unit_fixture','human_approval_text':'not a real launch approval'}
    write(tmp_path,'auth.json',receipt);validate_authorization(tmp_path,packet,tmp_path/'auth.json')
    for k,v in [('source_kind','dequantized_fp8_derivative'),('arm','all'),('retries',1),('human_approval_reference','')]:
        altered={**receipt,k:v};write(tmp_path,'bad.json',altered)
        with pytest.raises(ValueError):validate_authorization(tmp_path,packet,tmp_path/'bad.json')
    (tmp_path/p['output_root']).mkdir(parents=True)
    with pytest.raises(ValueError,match='consumed'):validate_authorization(tmp_path,packet,tmp_path/'auth.json')

@pytest.mark.parametrize('field,value',[('slots',4),('base_dtype','float32'),('persistent_P_updates',1),('optimizer','AdamW')])
def test_protocol_keeps_fixed_numerical_and_owner_boundaries(field,value):
    p=json.loads((ROOT/'configs/spark_intrinsic_diagnostic_v5.json').read_text());p[field]=value
    with pytest.raises(ValueError):validate_protocol(p)

def test_each_arm_has_complete_independent_budget():
    p=json.loads((ROOT/'configs/spark_intrinsic_diagnostic_v5.json').read_text());validate_protocol(p)
    assert len({a['output_root'] for a in p['arms'].values()})==3
    for arm in p['arms'].values():
        assert arm['budget']['fit_probes']['forwards']==448
        assert arm['total_reservations']['forwards']==957
        assert arm['total_reservations']['updates']==33
        assert arm['total_reservations']['model_loads']==2
    p['arms']['all']['budget']['fit']['updates']+=1
    with pytest.raises(ValueError):validate_protocol(p)

def test_failed_or_incomplete_baseline_never_parses_fit_or_development(tmp_path,monkeypatch):
    from replica_lab import intrinsic_memory_native_v5 as native
    torch.set_num_threads(1)
    p=make(objective=spec().identity());calls=[]
    def data(root,pin):
        calls.append(pin['stage'])
        assert pin['stage']=='calibration','dependent labels opened'
        return {'rows':[{**row(i=i),'answer':'never_emitted'} for i in range(8)]}
    monkeypatch.setattr(native,'bound_json',data)
    plan={'baseline':{**{k:0 for k in COUNTERS},'forwards':24,'input_tokens':24*8,'generated_tokens':24,'wall_seconds':30}}
    l=WorkLedger.create(tmp_path/'ledger.json',identity='CPU_gate',plan=plan,deadline=time.monotonic()+30)
    protocol={'generation':{'max_new_tokens':3},'data':{k:{'stage':k} for k in ('calibration','fit','development')}}
    out=tmp_path/'output';out.mkdir()
    result=native.run_main(tmp_path,protocol,'two',l,out,p,FixtureBackend(),spec(),native.base_hash(p.adapter.base))
    assert result['status']=='baseline_failed_dependent_stages_closed'
    assert not result['competence_passed'] and calls==['calibration']
    assert l.snapshot()['reserved']['updates']==0 and not (out/'checkpoint').exists()

def test_dedicated_launcher_refuses_missing_approval_without_importing_weights():
    r=subprocess.run([sys.executable,str(ROOT/'scripts/run_spark_intrinsic_diagnostic_v5.py'),
        '--packet',str(ROOT/'reports/intrinsic_memory_successor_v5/native_packet_two_v5.json'),'--launch'],
        cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=10)
    assert r.returncode!=0 and 'exact external authorization' in r.stderr


def test_complete_cpu_worker_path_freezes_before_closed_development(tmp_path,monkeypatch):
    from replica_lab import intrinsic_memory_native_v5 as native
    from replica_lab.intrinsic_memory_diagnostic_v5 import DiagnosticEngine
    from test_tiered_memory import open_store
    torch.set_num_threads(1)
    protocol=json.loads((ROOT/'configs/spark_intrinsic_diagnostic_v5.json').read_text())
    protocol['generation']['max_new_tokens']=3
    protocol['data']={k:{'stage':k} for k in ('calibration','fit','development')}
    fitting=[row('red' if i%2==0 else 'blue',i) for i in range(8)];calls=[]
    def data(root,pin):
        calls.append(pin['stage'])
        assert pin['stage']!='development','development parsed despite failed fitting gate'
        return {'rows':fitting}
    monkeypatch.setattr(native,'bound_json',data)
    monkeypatch.setattr(native,'store_at',lambda output,world:open_store(output/'stores',name=world))
    original_generate=DiagnosticEngine.generate
    def generate(self,r,*args,**kwargs):
        if self.stage=='baseline':
            # Gate fixture only: no native competence claim is made by this test.
            return {'raw_answer':r['answer'],'correct':True,'false_recall':False,
                    'truncated':False,'first_logits':None}
        return original_generate(self,r,*args,**kwargs)
    monkeypatch.setattr(DiagnosticEngine,'generate',generate)
    original_probes=DiagnosticEngine.fitting_probes
    def probes(self,*args):
        observed=original_probes(self,*args)
        observed['fit_memory_dependence_passed']=False
        return observed
    monkeypatch.setattr(DiagnosticEngine,'fitting_probes',probes)
    plan=protocol['arms']['two']['budget']
    l=WorkLedger.create(tmp_path/'ledger.json',identity='CPU_worker_path',plan=plan,deadline=time.monotonic()+120)
    p=make(objective=spec().identity());out=tmp_path/'output';out.mkdir()
    result=native.run_main(tmp_path,protocol,'two',l,out,p,FixtureBackend(),spec(),native.base_hash(p.adapter.base))
    assert result['fit_updates']==32 and result['restart_requested']
    assert result['original_weights_unchanged'] and not result['useful_recall_passed']
    assert calls==['calibration','fit'] and (out/'fitting_closed.json').exists()
    assert (out/'checkpoint/manifest.json').exists() and not (out/'development.json').exists()


def test_outer_launcher_owns_whole_lifetime_and_failure_consumes_attempt(tmp_path,monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    sys.path.insert(0,str(ROOT/'scripts'))
    module_spec=importlib.util.spec_from_file_location('v5_launcher_test',ROOT/'scripts/run_spark_intrinsic_diagnostic_v5.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    packet=synthetic(tmp_path);p,protocol,_=validate_bindings(tmp_path,packet)
    # This test injects authorization validation, never issues a real receipt.
    def authorized(root,packet,receipt):
        if (root/p['output_root']).exists():raise ValueError('consumed')
        return {'test_fixture':True}
    monkeypatch.setattr(module,'validate_authorization',authorized)
    auth=tmp_path/'fixture_receipt';auth.write_text('synthetic only')
    seen=[]
    def supervisor(command,**kwargs):
        seen.append((command,kwargs));return SimpleNamespace(exit_code=124)
    monkeypatch.setattr(module,'run_wall_bounded',supervisor)
    args=SimpleNamespace(packet=packet,authorization=auth)
    assert module.launch(args)==124
    command,kw=seen[0]
    assert '--supervised' in command and kw['policy'].wall_seconds==7200
    assert kw['env']['PYTORCH_ENABLE_MPS_FALLBACK']=='0'
    assert kw['env']['HF_HUB_OFFLINE']=='1' and kw['authorization_identity']==hash_file(auth)
    with pytest.raises(ValueError,match='consumed'):module.launch(args)


def test_substituted_HF_cache_is_refused_before_source_class_import(tmp_path,monkeypatch):
    import types
    from replica_lab.intrinsic_memory_native_v5 import verified_source_classes
    packet=synthetic(tmp_path)
    source=tmp_path/'source';raw=json.loads((source/'config.json').read_text())
    manifest=json.loads((tmp_path/'source_execution.json').read_text());cache=tmp_path/'cache';cache.mkdir()
    for name in manifest['files']:(cache/name).write_bytes((source/name).read_bytes())
    (cache/'modeling_spark.py').write_text('raise RuntimeError("substituted cache executed")\n')
    imported=[];module=types.ModuleType('transformers.dynamic_module_utils')
    module.HF_MODULES_CACHE=str(cache)
    module.get_cached_module_file=lambda directory,name,**kwargs:name
    module.get_class_in_module=lambda *args,**kwargs:imported.append(args)
    monkeypatch.setitem(sys.modules,'transformers.dynamic_module_utils',module)
    with pytest.raises(ValueError,match='cached source dependency'):verified_source_classes(source,raw,manifest)
    assert not imported
