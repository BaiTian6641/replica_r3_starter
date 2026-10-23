#!/usr/bin/env python3
"""CPU acceptance checks for the two-process mixed restart run."""
from __future__ import annotations
import argparse, json, sqlite3, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))

def load(path): return json.loads(Path(path).read_text())

def tool_name(item):
    action=item.get('action') or {}; return action.get('tool') if action.get('kind')=='invoke_tool' else None

def receipt_exit(item):
    outcome=item.get('outcome') or {}; result=outcome.get('result') or {}
    receipt=result.get('receipt') if isinstance(result,dict) else None
    return (receipt or {}).get('exit_code') if isinstance(receipt,dict) else None

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--process1',type=Path,required=True); p.add_argument('--process2',type=Path,required=True)
    p.add_argument('--case',type=Path,required=True); p.add_argument('--source-case',type=Path,required=True)
    p.add_argument('--journal',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    r1,r2=load(a.process1/'session_report.json'),load(a.process2/'session_report.json')
    source=load(a.source_case/'case.json'); current=load(a.case/'case.json')
    t1=r1.get('summary',{}).get('trace',[]); t2=r2.get('summary',{}).get('trace',[])
    ids1={x.get('action',{}).get('action_id') for x in t1 if x.get('action')}
    ids2={x.get('action',{}).get('action_id') for x in t2 if x.get('action')}
    failed1=[x for x in t1 if tool_name(x)=='run_test' and receipt_exit(x) not in (None,0)]
    passed2=[x for x in t2 if tool_name(x)=='run_test' and receipt_exit(x)==0]
    repairs=[x for x in t2 if tool_name(x)=='set_constant' and x.get('outcome',{}).get('status')=='completed']
    workspace=a.case/'workspace/app_settings.py'
    expected=source.get('reference_repair',{}).get('content')
    checks={
      'process1_nonzero_test':bool(failed1),
      'process1_two_decisions':len(t1)==2,
      'process1_incomplete':r1.get('summary',{}).get('success') is False,
      'process1_no_repair':not any(tool_name(x)=='set_constant' for x in t1),
      'process2_completed':r2.get('summary',{}).get('success') is True,
      'process2_independent_pass':bool(passed2),
      'process2_single_repair':len(repairs)==1,
      'process2_no_completed_action_replay':not bool(ids1 & ids2),
      'workspace_matches_repair':bool(expected) and workspace.read_text()==expected,
      'evaluator_unmodified': (a.case/source['evaluator']).read_bytes()==(a.source_case/source['evaluator']).read_bytes(),
      'no_malformed_or_false_claims':sum(x.get('malformed_actions',0) for x in (r1,r2))==0 and sum(x.get('false_success_claims',0) for x in (r1,r2))==0,
      'journal_state_complete': bool(r2.get('final_state',{}).get('satisfied_conditions')),
    }
    report={'status':'passed' if all(checks.values()) else 'failed','checks':checks,
            'process1':{'success':r1.get('summary',{}).get('success'),'decisions':len(t1),
                        'tool_calls':r1.get('summary',{}).get('budget',{}).get('used_tool_calls')},
            'process2':{'success':r2.get('summary',{}).get('success'),'decisions':len(t2),
                        'tool_calls':r2.get('summary',{}).get('budget',{}).get('used_tool_calls')},
            'repairs':[x.get('action',{}).get('args') for x in repairs],
            'claims':'One copied synthetic case; fresh-process continuation only.'}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'checks':checks}))
if __name__=='__main__': main()
