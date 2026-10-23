#!/usr/bin/env python3
"""Fresh varied-template retrieval data for a reranker candidate."""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'data/e2_cross_rerank'
SUBJECT='e2_cross_rerank'; PURPOSE='cross_template_rerank'; AS_OF=17
ENTITIES=['Aspen','Birch','Cedar','Dune','Ember','Fjord','Glade','Harbor','Iris','Juniper','Kelp','Lumen','Maple','Nectar','Orchid','Pike','Quill','Reed','Sage','Topaz']
ATTRS=['deploy region','launch code','backup window','owner','review code','timeout seconds','retry limit','release label']
VALUES=['cobalt','saffron','ochre','vermilion','cerulean','chartreuse','magenta','tangerine','periwinkle','turquoise','burgundy','lavender','carmine','sienna','azure','olive','peach','plum','rose','rust']
def txt(template,attr,entity,value):
    if template==0: return f'Configuration for project {entity}: the {attr} is {value}.'
    if template==1: return f'Project {entity} uses {value} as its {attr}.'
    if template==2: return f'{attr} for project {entity} = {value}.'
    return f'For project {entity}, {attr}: {value}.'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'checkpoints/minicpm5_source',local_files_only=True,trust_remote_code=False)
    specs=[]
    for i in range(8): specs.append((f'cr_train_{i+1:02d}',f'train_{i%4}','train',ENTITIES[i],ATTRS[i%len(ATTRS)],VALUES[i],i))
    for i in range(4): specs.append((f'cr_dev_{i+1:02d}',f'dev_{i%2}','dev',ENTITIES[8+i],ATTRS[(i+2)%len(ATTRS)],VALUES[8+i],8+i))
    for i in range(8): specs.append((f'cr_heldout_{i+1:02d}',f'heldout_{i%4}','heldout',ENTITIES[12+i],ATTRS[(i+4)%len(ATTRS)],VALUES[12+i],12+i))
    records=[]; queries=[]; counts={}
    for cid,g,split,e,a,v,idx in specs:
        rel=f'{cid}_relevant'; oe=ENTITIES[(idx+5)%len(ENTITIES)]; oa=ATTRS[(ATTRS.index(a)+1)%len(ATTRS)]
        texts=[(rel,txt(idx%4,a,e,v)),
               (f'{cid}_same_entity',txt((idx+1)%4,oa,e,VALUES[(idx+2)%len(VALUES)])),
               (f'{cid}_same_attr',txt((idx+2)%4,a,oe,VALUES[(idx+4)%len(VALUES)])),
               (f'{cid}_owner',f'The owner of project {e} is person{idx:02d}.')]
        for rid,text in texts:
            records.append({'record_id':rid,'subject_id':SUBJECT,'branch_id':cid,'text':text,
              'observed_at':1,'event_at':0,'purpose':PURPOSE,'kind':'source','supersedes':None,
              'source_id':g,'authenticity':'source_report'})
            n=len(tok(text,add_special_tokens=False)['input_ids']); counts[rid]=n
            if not 0<n<=128: raise ValueError((rid,n))
        queries.append({'case_id':cid,'source_group':g,'split':split,'subject_id':SUBJECT,
          'branch_id':cid,'purpose':PURPOSE,'as_of':AS_OF,'query':f'What is the {a} for project {e}?',
          'expected_record_id':rel,'expected_answer':v,'forbidden_record_ids':[],
          'kind':'cross_template_rerank','records':[x[0] for x in texts]})
    OUT.mkdir(parents=True,exist_ok=True)
    for name,rows in (('records.jsonl',records),('queries.jsonl',queries)):
        with (OUT/name).open('w') as f:
            for row in rows:f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n')
    groups={}
    for q in queries:groups.setdefault(q['source_group'],{'split':q['split'],'cases':[]})['cases'].append(q['case_id'])
    manifest={'format':'replica.e2.cross_rerank.dataset.v1','created_at_utc':'2026-10-23',
      'subject_id':SUBJECT,'purpose':PURPOSE,'as_of':AS_OF,
      'counts':{'cases':len(queries),'records':len(records),
        'by_split':{s:sum(q['split']==s for q in queries) for s in ('train','dev','heldout')}},
      'groups':groups,'token_counts':counts,
      'files':{'records_sha256':sha(OUT/'records.jsonl'),'queries_sha256':sha(OUT/'queries.jsonl')},
      'claims':'Synthetic cross-template retrieval data; no answer-generation candidate.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['counts']))
if __name__=='__main__': main()
