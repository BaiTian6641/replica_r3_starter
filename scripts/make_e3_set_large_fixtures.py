#!/usr/bin/env python3
"""Large fresh E3 set_constant family for a supervision-scale action experiment."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from make_e3_set_fixtures import evaluator,file_text
OUT=ROOT/'fixtures/e3_set_large/cases'
GOAL=('The policy test for workspace/app_settings.py fails. Inspect the configuration, '
      'run the evaluator to learn the required bound, change exactly that one numeric '
      'constant with set_constant, and rerun the test to confirm it passes.')
CASES=[]
def add(cid,role,group,target,actual,expected,others):CASES.append((cid,role,group,target,actual,expected,others))
for i in range(12):
    name=['RETRY_ATTEMPTS','REQUEST_TIMEOUT_MS','BACKOFF_SECONDS','POLL_ATTEMPTS','MAX_RETRIES','CONNECT_TIMEOUT_MS'][i%6]
    actual=[1,700,0.15,2,1,350][i%6]; expected=[6,4200,1.4,9,6,2400][i%6]
    others={'AUX_DELAY_S':0.5,'AUX_LIMIT':10} if i%2 else {'AUX_TIMEOUT_MS':900,'AUX_DEPTH':4}
    add(f'large_train_{i+1:02d}','train',f'large_train_{chr(97+i%4)}',name,actual,expected,others)
for i in range(3):
    add(f'large_dev_{i+1:02d}','dev',f'large_dev_{chr(97+i)}','LEASE_TTL_S',0.4+i*0.1,2.0+i,
        {'AUX_DEPTH':5,'AUX_LIMIT':12})
for i in range(6):
    name=['CLOCK_SKEW_MS','QUEUE_DEPTH','SHUTDOWN_GRACE_S','CACHE_TTL_S','DRAIN_TIMEOUT_S','SHARD_COUNT'][i]
    actual=[80,3,0.15,0.3,0.25,2][i]; expected=[1200,14,2.1,4.5,2.8,11][i]
    add(f'large_heldout_{i+1:02d}','heldout',f'large_heldout_{chr(97+i)}',name,actual,expected,
        {'AUX_DEPTH':3,'AUX_LIMIT':8})
add('large_interrupted_01','interrupted','large_interrupted_a','RETRY_ATTEMPTS',2,7,
    {'AUX_DEPTH':2,'AUX_LIMIT':6})
def main():
    if OUT.exists() and any(OUT.iterdir()):raise SystemExit('Use a fresh fixture directory')
    manifest={'format':'replica.e3_set.large.v1','created_at_utc':'2026-10-23','goal':GOAL,'cases':[],'splits':{}}
    for cid,role,group,target,actual,expected,others in CASES:
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
    print(json.dumps({'cases':len(CASES),'splits':{k:len(v) for k,v in manifest['splits'].items()},
      'manifest_sha256':hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()}))
if __name__=='__main__':main()
