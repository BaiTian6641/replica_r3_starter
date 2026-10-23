#!/usr/bin/env python3
"""Fresh E3 family for a bounded full-backbone action diagnostic."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from make_e3_set_fixtures import evaluator,file_text
OUT=ROOT/'fixtures/e3_d2/cases'
GOAL=('The policy test for workspace/app_settings.py fails. Inspect the configuration, '
      'run the evaluator to learn the required bound, change exactly that one numeric '
      'constant with set_constant, and rerun the test to confirm it passes.')
def cases():
    out=[]
    specs=[
      ('d2_train_01','train','d2_train_a','MAX_WORKERS',2,9,{'AUX_LIMIT':4}),
      ('d2_train_02','train','d2_train_b','BUFFER_MS',150,1100,{'AUX_DEPTH':6}),
      ('d2_train_03','train','d2_train_c','RETRY_DELAY_S',0.2,1.3,{'AUX_LIMIT':7}),
      ('d2_train_04','train','d2_train_d','PAGE_SIZE_KB',32,128,{'AUX_DEPTH':3}),
      ('d2_train_05','train','d2_train_a','HEARTBEAT_MS',250,1600,{'AUX_LIMIT':5}),
      ('d2_train_06','train','d2_train_b','MAX_INFLIGHT',3,11,{'AUX_DEPTH':8}),
      ('d2_train_07','train','d2_train_c','SOCKET_TIMEOUT_S',0.4,2.6,{'AUX_LIMIT':9}),
      ('d2_train_08','train','d2_train_d','COMPACT_MS',800,4800,{'AUX_DEPTH':2}),
      ('d2_dev_01','dev','d2_dev_a','LEASE_MS',300,2200,{'AUX_LIMIT':11}),
      ('d2_dev_02','dev','d2_dev_b','THROTTLE_QPS',5,22,{'AUX_DEPTH':7}),
      ('d2_heldout_01','heldout','d2_heldout_a','WINDOW_MS',200,1800,{'AUX_LIMIT':6}),
      ('d2_heldout_02','heldout','d2_heldout_b','MAX_RETRY',1,7,{'AUX_DEPTH':4}),
      ('d2_heldout_03','heldout','d2_heldout_c','DRAIN_MS',400,2600,{'AUX_LIMIT':8}),
      ('d2_heldout_04','heldout','d2_heldout_d','CONN_BUDGET',6,24,{'AUX_DEPTH':9}),
    ]
    return specs
def main():
    if OUT.exists() and any(OUT.iterdir()):raise SystemExit('Use a fresh fixture directory')
    specs=cases(); manifest={'format':'replica.e3_d2.dataset.v1','created_at_utc':'2026-10-23','goal':GOAL,'cases':[],'splits':{}}
    for cid,role,group,target,actual,expected,others in specs:
        case=OUT/cid; (case/'workspace').mkdir(parents=True); (case/'evaluator').mkdir()
        originals=dict(others); originals[target]=actual
        (case/'workspace/app_settings.py').write_text(file_text(target,originals))
        (case/'evaluator/test_policy.py').write_text(evaluator(target,expected,others))
        repair=dict(others); repair[target]=expected
        payload={'format':'replica.e3.fixture.v1','case_id':cid,'role':role,'goal':GOAL,
          'success_condition':{'condition_id':'evaluator_test_passes','description':'python -m unittest evaluator/test_policy.py exits with status zero'},
          'constraints':['Only workspace/app_settings.py may be modified.','Change exactly one numeric constant with set_constant; leave every other setting untouched.','Do not modify anything under the evaluator directory.','Use only the allowed tools to inspect, edit, and test.'],
          'allowed_tools':['read_file','set_constant','run_test'],'workspace_files':['workspace/app_settings.py'],
          'evaluator':'evaluator/test_policy.py','workspace_entry':'workspace/app_settings.py',
          'reference_repair':{'path':'workspace/app_settings.py','content':file_text(target,repair)},
          'variant':{'constant':target,'expected_value':expected,'actual_value':actual,'description':'one numeric policy bound below its required minimum'},
          'source_group':group}
        (case/'case.json').write_text(json.dumps(payload,indent=2)+'\n')
        manifest['cases'].append({'case_id':cid,'role':role,'source_group':group,'constant':target,'actual':actual,'expected':expected})
        manifest['splits'].setdefault(role,[]).append(cid)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'cases':len(specs),'splits':{k:len(v) for k,v in manifest['splits'].items()},
      'manifest_sha256':hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()}))
if __name__=='__main__':main()
