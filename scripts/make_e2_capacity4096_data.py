#!/usr/bin/env python3
"""Generate a 4096-unit capacity bank: 4092 short records + one four-chunk record.

The bank is synthetic and only measures the existing source-encode/writer/read
path. It contains no labels, task semantics or personal data.
"""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'data/e2_capacity4096'
SUBJECT='e2_capacity4096'; PURPOSE='capacity_probe'; BRANCH='capacity'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def short(tokenizer, seed):
    sentences=[
      f'Capacity source record {seed} describes a routine configuration note.',
      'It records scheduling, maintenance, review, documentation and audit details.',
      'The note is synthetic and has no external provenance or personal data.',
      'Each statement is deliberately ordinary so the workload is reproducible.',
      'No answer or task label is attached to this capacity-only fixture.',
      'The record remains within one declared 128-token source chunk.',
      'Operators use the note only to exercise the source encoder and writer.',
      'The text contains no instruction to the model and no hidden answer.',
      'Retention is limited to the declared capacity experiment directory.',
      'The same structure is repeated for each short capacity record.',
      'The measurement records actual token counts rather than configured maxima.',
      'A final sentence keeps the record inside the intended token band.']
    text=' '.join(sentences); ids=tokenizer(text,add_special_tokens=False)['input_ids']
    if len(ids)>128:
        text=tokenizer.decode(ids[:128],skip_special_tokens=True)
        ids=tokenizer(text,add_special_tokens=False)['input_ids']
    while len(ids)<120:
        text+=' Additional synthetic capacity filler text.'
        ids=tokenizer(text,add_special_tokens=False)['input_ids']
    if not 120<=len(ids)<=128: raise ValueError((seed,len(ids)))
    return text,len(ids)
def long_record(tokenizer):
    sentence=('The capacity handbook contains routine notes about scheduling, reviews, '
              'maintenance, documentation, audits, training, and recovery procedures. ')
    text=sentence*22; ids=tokenizer(text,add_special_tokens=False)['input_ids']
    if len(ids)>4*128:
        text=tokenizer.decode(ids[:4*128],skip_special_tokens=True)
        ids=tokenizer(text,add_special_tokens=False)['input_ids']
    if not 3*128+1<=len(ids)<=4*128: raise ValueError(len(ids))
    return text,len(ids)
def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'checkpoints/minicpm5_source',local_files_only=True,trust_remote_code=False)
    records=[]; counts={}
    for i in range(1,4093):
        text,n=short(tok,i); rid=f'cap4096_short_{i:04d}'
        records.append({'record_id':rid,'subject_id':SUBJECT,'branch_id':BRANCH,
            'text':text,'observed_at':1,'event_at':0,'purpose':PURPOSE,'kind':'source',
            'supersedes':None,'source_id':'capacity_fixture','authenticity':'unverified'})
        counts[rid]=n
    text,n=long_record(tok); records.append({'record_id':'cap4096_long','subject_id':SUBJECT,
        'branch_id':BRANCH,'text':text,'observed_at':1,'event_at':0,'purpose':PURPOSE,
        'kind':'source','supersedes':None,'source_id':'capacity_fixture',
        'authenticity':'unverified'}); counts['cap4096_long']=n
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'records.jsonl').open('w',encoding='utf-8') as f:
        for row in records: f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    manifest={'format':'replica.e2_capacity.dataset.v9','created_at_utc':'2026-10-23',
      'subject_id':SUBJECT,'branch_id':BRANCH,'purpose':PURPOSE,'as_of':10,
      'records':len(records),'units':4096,'short_records':4092,
      'long_record_spans':(n+127)//128,'token_counts':counts,
      'files':{'records_sha256':sha(OUT/'records.jsonl')},
      'claims':'Synthetic 4096-unit capacity bank; no labels or quality claim.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'records':len(records),'units':4096,'long_tokens':n,
      'short_tokens':[counts[f'cap4096_short_{i:04d}'] for i in range(1,29)]}))
if __name__=='__main__':main()
