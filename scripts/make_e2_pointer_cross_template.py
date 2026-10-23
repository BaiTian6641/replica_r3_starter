#!/usr/bin/env python3
"""Fresh cross-template family for context-head generalization (no training)."""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'data/e2_pointer_cross_template'
SUBJECT='e2_pointer_cross_template'; PURPOSE='pointer_cross_template'; AS_OF=16
ENTITIES=['Aspen','Birch','Cedar','Dune','Ember','Fjord','Glade','Harbor','Iris','Juniper','Kelp','Lumen']
ATTRS=['deploy region','launch code','backup window','owner','review code','timeout seconds','retry limit','release label']
VALUES=['cobalt','saffron','ochre','vermilion','cerulean','chartreuse','magenta','tangerine','periwinkle','turquoise','burgundy','lavender']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def relevant_text(template,attr,entity,value):
    if template==0: return f'Configuration for project {entity}: the {attr} is {value}.'
    if template==1: return f'Project {entity} uses {value} as its {attr}.'
    if template==2: return f'{attr} for project {entity} = {value}.'
    return f'For project {entity}, {attr}: {value}.'
def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'checkpoints/minicpm5_source',local_files_only=True,trust_remote_code=False)
    records=[]; queries=[]; lengths={}
    for i,e in enumerate(ENTITIES):
        cid=f'cross_{i+1:02d}'; group=f'cross_{i%4}'; attr=ATTRS[i%len(ATTRS)]; value=VALUES[i]; tmpl=i%4
        rel=f'{cid}_relevant'; other_e=ENTITIES[(i+5)%len(ENTITIES)]; other_a=ATTRS[(i+1)%len(ATTRS)]
        texts=[(rel,relevant_text(tmpl,attr,e,value)),
               (f'{cid}_same_entity',relevant_text((tmpl+1)%4,other_a,e,VALUES[(i+2)%len(VALUES)])),
               (f'{cid}_same_attr',relevant_text((tmpl+2)%4,attr,other_e,VALUES[(i+4)%len(VALUES)])),
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
            'expected_answer':value,'forbidden_record_ids':[],'kind':'cross_template',
            'records':[x[0] for x in texts]})
    OUT.mkdir(parents=True,exist_ok=True)
    for name,rows in (('records.jsonl',records),('queries.jsonl',queries)):
        with (OUT/name).open('w') as f:
            for row in rows:f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n')
    manifest={'format':'replica.e2.pointer_fresh.v1','created_at_utc':'2026-10-23',
      'subject_id':SUBJECT,'purpose':PURPOSE,'as_of':AS_OF,
      'counts':{'cases':12,'records':48,'by_split':{'heldout':12}},
      'groups':{f'cross_{i}':{'split':'heldout','cases':[f'cross_{j+1:02d}' for j in range(i,12,4)]} for i in range(4)},
      'token_counts':lengths,'files':{'records_sha256':sha(OUT/'records.jsonl'),'queries_sha256':sha(OUT/'queries.jsonl')},
      'claims':'Fresh cross-template context-head evaluation; no training.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['counts']))
if __name__=='__main__': main()
