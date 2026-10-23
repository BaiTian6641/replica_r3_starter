#!/usr/bin/env python3
"""Fresh E3 set_constant family for DAgger-style action aggregation."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from make_e3_set_fixtures import evaluator,file_text
OUT=ROOT/'fixtures/e3_dagger/cases'
GOAL=('The policy test for workspace/app_settings.py fails. Inspect the configuration, '
      'run the evaluator to learn the required bound, change exactly that one numeric '
      'constant with set_constant, and rerun the test to confirm it passes.')
SPECS=[
 ('dagger_train_01','train','dagger_train_a','MAX_ATTEMPTS',2,10,{'AUX_LIMIT':4}),
 ('dagger_train_02','train','dagger_train_b','WAIT_MS',200,1400,{'AUX_DEPTH':5}),
 ('dagger_train_03','train','dagger_train_c','BACKOFF_S',0.25,1.6,{'AUX_LIMIT':6}),
 ('dagger_train_04','train','dagger_train_d','CONN_LIMIT',4,16,{'AUX_DEPTH':3}),
 ('dagger_dev_01','dev','dagger_dev_a','LEASE_MS',300,2300,{'AUX_LIMIT':8}),
 ('dagger_heldout_01','heldout','dagger_heldout_a','WINDOW_MS',250,1900,{'AUX_DEPTH':4}),
 ('dagger_heldout_02','heldout','dagger_heldout_b','RETRY_MAX',1,7,{'AUX_LIMIT':7}),
 ('dagger_heldout_03','heldout','dagger_heldout_c','DRAIN_MS',450,2700,{'AUX_DEPTH':6}),
 ('dagger_heldout_04','heldout','dagger_heldout_d','BUFFER_KB',24,96,{'AUX_LIMIT':5}),
]
def main():
    if OUT.exists() and any(OUT.iterdir()):raise SystemExit('Use a fresh fixture directory')
    manifest={'format':'replica.e3_dagger.dataset.v1','created_at_utc':'2026-10-23','goal':GOAL,'cases':[],'splits':{}}
    for cid,role,group,target,actual,expected,others in SPECS:
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
    print(json.dumps({'cases':len(SPECS),'splits':{k:len(v) for k,v in manifest['splits'].items()},
      'manifest_sha256':hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()}))
if __name__=='__main__':main()
