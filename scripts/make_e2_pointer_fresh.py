#!/usr/bin/env python3
"""Fresh, untouched-value consumer pages for the published pointer candidate."""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'data/e2_pointer_fresh'
SUBJECT='e2_pointer_fresh'; PURPOSE='pointer_consumer_eval'; AS_OF=14
ENTITIES=['Briar','Cinder','Dune','Elm','Fable','Glacier','Hollow','Iris','Jade','Kestrel','Lark','Moss']
ATTRS=['deploy region','launch code','backup window','owner','review code','timeout seconds','retry limit','release label']
VALUES=['saffron','cobalt','ochre','vermilion','cerulean','chartreuse','magenta','tangerine','periwinkle','turquoise','burgundy','lavender']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'checkpoints/minicpm5_source',local_files_only=True,trust_remote_code=False)
    records=[]; queries=[]; lengths={}
    for i,e in enumerate(ENTITIES):
        cid=f'fresh_pointer_{i+1:02d}'; group=f'fresh_{i%4}'; attr=ATTRS[i%len(ATTRS)]; value=VALUES[i]
        rel=f'{cid}_relevant'; other_e=ENTITIES[(i+3)%len(ENTITIES)]; other_a=ATTRS[(i+1)%len(ATTRS)]
        texts=[(rel,f'The {attr} for project {e} is {value}.'),
               (f'{cid}_same_entity',f'The {other_a} for project {e} is {VALUES[(i+2)%len(VALUES)]}.'),
               (f'{cid}_same_attr',f'The {attr} for project {other_e} is {VALUES[(i+5)%len(VALUES)]}.'),
               (f'{cid}_owner',f'The owner of project {e} is person{i:02d}.')]
        for rid,text in texts:
            records.append({'record_id':rid,'subject_id':SUBJECT,'branch_id':cid,'text':text,
                'observed_at':1,'event_at':0,'purpose':PURPOSE,'kind':'source','supersedes':None,
                'source_id':group,'authenticity':'source_report'})
            n=len(tok(text,add_special_tokens=False)['input_ids']); lengths[rid]=n
            if not 0<n<=128: raise ValueError((rid,n))
        queries.append({'case_id':cid,'source_group':group,'split':'heldout','subject_id':SUBJECT,
            'branch_id':cid,'purpose':PURPOSE,'as_of':AS_OF,
            'query':f'What is the {attr} for project {e}?','expected_record_id':rel,
            'expected_answer':value,'forbidden_record_ids':[],'kind':'fresh_pointer','records':[x[0] for x in texts]})
    OUT.mkdir(parents=True,exist_ok=True)
    for name,rows in (('records.jsonl',records),('queries.jsonl',queries)):
        with (OUT/name).open('w') as f:
            for row in rows: f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n')
    manifest={'format':'replica.e2.pointer_fresh.v1','created_at_utc':'2026-10-23',
      'subject_id':SUBJECT,'purpose':PURPOSE,'as_of':AS_OF,
      'counts':{'cases':len(queries),'records':len(records),'by_split':{'heldout':len(queries)}},
      'groups':{f'fresh_{i}':{'split':'heldout','cases':[f'fresh_pointer_{j+1:02d}' for j in range(i,12,4)]} for i in range(4)},
      'token_counts':lengths,
      'files':{'records_sha256':sha(OUT/'records.jsonl'),'queries_sha256':sha(OUT/'queries.jsonl')},
      'claims':'Fresh synthetic pointer consumer pages; no training.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['counts']))
if __name__=='__main__': main()
