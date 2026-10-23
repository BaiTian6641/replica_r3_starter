#!/usr/bin/env python3
"""Fresh prompt-evidence family with names/values absent from prior arms."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from make_e3_set_fixtures import evaluator,file_text
OUT=ROOT/'fixtures/e3_prompt/cases'
GOAL=('The policy test for workspace/app_settings.py must pass. Inspect the '
      'configuration, run the evaluator, and if a required bound fails change '
      'exactly that one numeric constant with set_constant; if it already '
      'passes, stop without editing.')
CASES=[
 ('prompt_heldout_repair_01','heldout','prompt_repair_a','MAX_CONNECTIONS',10,40,
  {'BATCH_SIZE':32,'IDLE_TIMEOUT_SECONDS':45},True),
 ('prompt_heldout_repair_02','heldout','prompt_repair_b','BATCH_SIZE',8,64,
  {'MAX_CONNECTIONS':24,'IDLE_TIMEOUT_SECONDS':90},True),
 ('prompt_heldout_pass_03','heldout','prompt_pass_a','MAX_CONNECTIONS',40,40,
  {'BATCH_SIZE':32,'IDLE_TIMEOUT_SECONDS':45},False),
 ('prompt_heldout_pass_04','heldout','prompt_pass_b','IDLE_TIMEOUT_SECONDS',90,60,
  {'MAX_CONNECTIONS':24,'BATCH_SIZE':64},False),
]
def main():
    if OUT.exists() and any(OUT.iterdir()): raise SystemExit('Use a fresh directory')
    manifest={'format':'replica.e3_mixed.dataset.v1','created_at_utc':'2026-10-23',
              'goal':GOAL,'cases':[],'splits':{}}
    for cid,role,group,target,actual,expected,others,repair in CASES:
        d=OUT/cid; (d/'workspace').mkdir(parents=True); (d/'evaluator').mkdir()
        originals=dict(others); originals[target]=actual
        (d/'workspace/app_settings.py').write_text(file_text(target,originals))
        (d/'evaluator/test_policy.py').write_text(evaluator(target,expected,others))
        repaired=dict(others); repaired[target]=expected
        case={'format':'replica.e3.fixture.v1','case_id':cid,'role':role,'goal':GOAL,
          'success_condition':{'condition_id':'evaluator_test_passes',
            'description':'python -m unittest evaluator/test_policy.py exits with status zero'},
          'constraints':['Only workspace/app_settings.py may be modified.',
            'If a repair is needed, change exactly one numeric constant with set_constant and leave every other setting untouched.',
            'Do not modify anything under the evaluator directory.',
            'Use only the allowed tools to inspect, edit and test.'],
          'allowed_tools':['read_file','set_constant','run_test'],
          'workspace_files':['workspace/app_settings.py'],
          'evaluator':'evaluator/test_policy.py','workspace_entry':'workspace/app_settings.py',
          'reference_repair':{'path':'workspace/app_settings.py','content':file_text(target,repaired)},
          'variant':{'constant':target,'expected_value':expected,'actual_value':actual,
            'description':'one numeric policy bound below its required minimum' if repair else 'all bounds already satisfy the policy'},
          'requires_repair':repair,'source_group':group}
        (d/'case.json').write_text(json.dumps(case,indent=2)+'\n')
        manifest['cases'].append({'case_id':cid,'role':role,'source_group':group,'requires_repair':repair})
        manifest['splits'].setdefault(role,[]).append(cid)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'cases':len(CASES),'manifest_sha256':hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()}))
if __name__=='__main__': main()
