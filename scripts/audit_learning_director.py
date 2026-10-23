#!/usr/bin/env python3
"""Read-only C13 director audit over actual declared outcome summaries."""
from __future__ import annotations
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from replica_lab.runtime.director import DirectorBudget,LearningDirector,OutcomeSummary
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/c13_director_audit_v1.json');p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();c=json.loads(a.config.read_text());b=c['budget'];director=LearningDirector(budget=DirectorBudget(**b));rows=[]
 for item in c['observations']:
  summary=OutcomeSummary(task_family=item['task_family'],failure_class=item['failure_class'],attempts=item['attempts'],successes=item['successes'],source_event_ids=tuple(item['source_event_ids']),role='development',verified_receipts=item['verified_receipts'],cost_units=item['cost_units'])
  d=director.review([summary]);rows.append({'id':item['id'],'status':d.status,'reason':d.reason,'target':d.proposal.target_kind if d.proposal else None,'source_event_ids':list(item['source_event_ids'])})
 report={'format':'replica.c13.director_audit.report.v1','created_at_utc':datetime.now(timezone.utc).isoformat(),'config_sha256':__import__('hashlib').sha256(a.config.read_bytes()).hexdigest(),'results':rows,'trained':False,'claims':'Deterministic director audit only; no learned-selector accuracy or capability claim.'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'completed','results':rows}))
if __name__=='__main__':main()
