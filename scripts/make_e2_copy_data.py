#!/usr/bin/env python3
"""Fresh source-separated novel-value data for the record-token copy head."""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'data/e2_copy'
SUBJECT='e2_copy'; PURPOSE='copy_eval'; AS_OF=13
ENTITIES=['Alder','Basin','Cobalt','Dell','Ember','Fern','Grove','Hearth','Inlet','Juniper',
 'Kite','Lark','Meadow','Nectar','Orchid','Pike','Quill','Reed','Sage','Thistle',
 'Umber','Vale','Wren','Yarrow','Zephyr','Aurora','Bramble','Cinder','Dune','Elm',
 'Flint','Glade','Harbor','Iris','Jasper','Kelp','Lumen','Maple','Nimbus','Onyx']
VALUES=['amber','blue','silver','crimson','teal','violet','ochre','indigo','copper','emerald',
 'bronze','saffron','slate','ivory','quartz','topaz','coral','rose','pearl','olive',
 'ruby','jade','opal','azure','mauve','umber','lilac','flint','hazel','maroon',
 'mint','navy','peach','plum','rust','sage','vermilion','cerulean','chartreuse','magenta']
ATTRS=['deploy region','launch code','backup window','owner','review code','timeout seconds','retry limit','release label']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'checkpoints/minicpm5_source',local_files_only=True,trust_remote_code=False)
    specs=[]
    for i in range(24): specs.append((f'copy_train_{i+1:02d}',f'train_{i//4}','train',ENTITIES[i],ATTRS[i%len(ATTRS)],VALUES[i],i))
    for i in range(6): specs.append((f'copy_dev_{i+1:02d}',f'dev_{i//3}','dev',ENTITIES[24+i],ATTRS[(i+2)%len(ATTRS)],VALUES[24+i],24+i))
    for i in range(10): specs.append((f'copy_heldout_{i+1:02d}',f'heldout_{i//2}','heldout',ENTITIES[30+i],ATTRS[(i+4)%len(ATTRS)],VALUES[30+i],30+i))
    records=[]; queries=[]; lengths={}
    for cid,g,split,e,a,v,idx in specs:
        rel=f'{cid}_relevant'; other_e=ENTITIES[(idx+5)%len(ENTITIES)]; other_a=ATTRS[(ATTRS.index(a)+1)%len(ATTRS)]
        texts=[(rel,f'The {a} for project {e} is {v}.'),
               (f'{cid}_same_entity',f'The {other_a} for project {e} is {VALUES[(idx+3)%len(VALUES)]}.'),
               (f'{cid}_same_attr',f'The {a} for project {other_e} is {VALUES[(idx+7)%len(VALUES)]}.'),
               (f'{cid}_owner',f'The owner of project {e} is person{idx:02d}.')]
        for rid,text in texts:
            records.append({'record_id':rid,'subject_id':SUBJECT,'branch_id':cid,'text':text,
              'observed_at':1,'event_at':0,'purpose':PURPOSE,'kind':'source','supersedes':None,
              'source_id':g,'authenticity':'source_report'})
            n=len(tok(text,add_special_tokens=False)['input_ids'])
            if not 0<n<=128: raise ValueError((cid,n))
            lengths[rid]=n
        queries.append({'case_id':cid,'source_group':g,'split':split,'subject_id':SUBJECT,
          'branch_id':cid,'purpose':PURPOSE,'as_of':AS_OF,
          'query':f'What is the {a} for project {e}?','expected_record_id':rel,
          'expected_answer':v,'forbidden_record_ids':[],'kind':'novel_value_copy','records':[x[0] for x in texts]})
    OUT.mkdir(parents=True,exist_ok=True)
    for name,rows in (('records.jsonl',records),('queries.jsonl',queries)):
        with (OUT/name).open('w') as f:
            for row in rows:f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n')
    groups={}
    for q in queries:groups.setdefault(q['source_group'],{'split':q['split'],'cases':[]})['cases'].append(q['case_id'])
    manifest={'format':'replica.e2_copy.dataset.v1','created_at_utc':'2026-10-23',
      'subject_id':SUBJECT,'purpose':PURPOSE,'as_of':AS_OF,
      'counts':{'cases':len(queries),'records':len(records),
        'by_split':{s:sum(q['split']==s for q in queries) for s in ('train','dev','heldout')}},
      'groups':groups,'token_counts':lengths,
      'files':{'records_sha256':sha(OUT/'records.jsonl'),'queries_sha256':sha(OUT/'queries.jsonl')},
      'claims':'Fresh synthetic novel-value copy data; answer tokens are present in the selected relevant source record.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['counts']))
if __name__=='__main__': main()
