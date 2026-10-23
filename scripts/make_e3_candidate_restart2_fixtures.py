#!/usr/bin/env python3
"""Fresh E3 family for model-scored action candidate reranking."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from make_e3_set_fixtures import evaluator,file_text
OUT=ROOT/'fixtures/e3_candidate_restart2/cases'
GOAL=('The policy test for workspace/app_settings.py fails. Inspect the configuration, '
      'run the evaluator to learn the required bound, change exactly that one numeric '
      'constant with set_constant, and rerun the test to confirm it passes.')
SPECS=[
 ('restart2_heldout_01','heldout','restart2_a','MAX_WORKERS',2,11,{'AUX_DEPTH':4}),
 ('restart2_heldout_02','heldout','restart2_b','HEARTBEAT_MS',300,1900,{'AUX_LIMIT':8}),
 ('restart2_heldout_03','heldout','restart2_c','BUFFER_MS',180,1500,{'AUX_DEPTH':6}),
]
def main():
    if OUT.exists() and any(OUT.iterdir()):raise SystemExit('Use a fresh fixture directory')
    m={'format':'replica.e3_candidate_restart2.dataset.v1','created_at_utc':'2026-10-23','goal':GOAL,'cases':[],'splits':{}}
    for cid,role,group,target,actual,expected,others in SPECS:
        c=OUT/cid;(c/'workspace').mkdir(parents=True);(c/'evaluator').mkdir()
        originals=dict(others);originals[target]=actual
        (c/'workspace/app_settings.py').write_text(file_text(target,originals))
        (c/'evaluator/test_policy.py').write_text(evaluator(target,expected,others))
        repair=dict(others);repair[target]=expected
        payload={'format':'replica.e3.fixture.v1','case_id':cid,'role':role,'goal':GOAL,
          'success_condition':{'condition_id':'evaluator_test_passes','description':'python -m unittest evaluator/test_policy.py exits with status zero'},
          'constraints':['Only workspace/app_settings.py may be modified.','Change exactly one numeric constant with set_constant; leave every other setting untouched.','Do not modify anything under the evaluator directory.','Use only the allowed tools to inspect, edit, and test.'],
          'allowed_tools':['read_file','set_constant','run_test'],'workspace_files':['workspace/app_settings.py'],
          'evaluator':'evaluator/test_policy.py','workspace_entry':'workspace/app_settings.py',
          'reference_repair':{'path':'workspace/app_settings.py','content':file_text(target,repair)},
          'variant':{'constant':target,'expected_value':expected,'actual_value':actual,'description':'one numeric policy bound below its required minimum'},'source_group':group}
        (c/'case.json').write_text(json.dumps(payload,indent=2)+'\n')
        m['cases'].append({'case_id':cid,'role':role,'source_group':group,'constant':target,'actual':actual,'expected':expected})
        m['splits'].setdefault(role,[]).append(cid)
    (OUT/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps({'cases':len(SPECS),'splits':{k:len(v) for k,v in m['splits'].items()},'manifest_sha256':hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()}))
if __name__=='__main__':main()
