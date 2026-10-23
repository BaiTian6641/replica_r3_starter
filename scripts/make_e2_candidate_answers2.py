#!/usr/bin/env python3
"""Fresh source-grounded answer-selection data for candidate reranking."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/e2_candidate_answers2';SUBJECT='e2_candidate_answers';PURPOSE='candidate_answer_eval';AS_OF=20
SPECS=[
 ('answer2_01','Ember','deploy region','indigo','teal'),
 ('answer2_02','Fjord','launch code','carmine','peach'),
 ('answer2_03','Glade','backup window','sienna','plum'),
 ('answer2_04','Harbor','review code','olive','rose'),
]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    if OUT.exists() and any(OUT.iterdir()):raise SystemExit('Use a fresh data directory')
    records=[];queries=[]
    for i,(cid,entity,attr,value,other_value) in enumerate(SPECS):
        branch=cid; other_entity='Spruce' if entity!='Spruce' else 'Willow'; other_attr='owner' if attr!='owner' else 'timeout seconds'
        texts=[
          (f'{cid}_relevant',f'The {attr} for project {entity} is {value}.'),
          (f'{cid}_same_entity',f'The {other_attr} for project {entity} is {other_value}.'),
          (f'{cid}_same_attr',f'The {attr} for project {other_entity} is {other_value}.'),
          (f'{cid}_owner',f'The owner of project {entity} is operator{i+1}.'),
        ]
        for rid,text in texts:
            records.append({'record_id':rid,'subject_id':SUBJECT,'branch_id':branch,'text':text,
              'observed_at':1,'event_at':0,'purpose':PURPOSE,'kind':'source',
              'supersedes':None,'source_id':f'candidate_answer_{i}','authenticity':'source_report'})
        queries.append({'case_id':cid,'subject_id':SUBJECT,'branch_id':branch,'purpose':PURPOSE,'as_of':AS_OF,
          'query':f'What is the {attr} for project {entity}?','expected_answer':value,
          'expected_record_id':f'{cid}_relevant','candidates':[value,other_value,f'owner operator{i+1}','unknown'],
          'split':'heldout','source_group':f'answer_group_{i}'})
    OUT.mkdir(parents=True,exist_ok=True)
    for name,rows in (('records.jsonl',records),('queries.jsonl',queries)):
        with (OUT/name).open('w') as f:
            for row in rows:f.write(json.dumps(row,sort_keys=True)+'\n')
    m={'format':'replica.e2.candidate_answers2.v1','created_at_utc':'2026-10-23','cases':len(queries),
       'records':len(records),'files':{'records_sha256':sha(OUT/'records.jsonl'),'queries_sha256':sha(OUT/'queries.jsonl')},
       'claims':'Synthetic source-grounded answer-selection data; no training.'}
    (OUT/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps({'cases':len(queries),'records':len(records),'manifest_sha256':sha(OUT/'manifest.json')}))
if __name__=='__main__':main()
